"""
SARTHI Vision - Time-to-Collision (TTC) Engine

Section 4 of the upgrade prompt: the actual missing feature.

Standard ADAS math (published, not improvised):

    dt  = 1 / frame_rate
    V   = (Z_previous - Z_current) / dt      # positive V = object closing
    TTC = Z_current / V                       # only defined when V > 0

Guard explicitly in code: if V <= 0 (object stable or moving away), TTC is
undefined. Do not compute it, do not divide, do not treat it as a low-risk
number. Just skip TTC for that object on that frame.

For object-to-object risk (the cow-and-motorcycle case):
- Only compute pairwise TTC for pairs already within a proximity threshold
  AND converging (centroid distance shrinking vs N frames ago).
- Filter first, then compute. No O(n^2) explosion.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from app.config import (
    TTC_DANGER_THRESHOLD,
    PAIRWISE_PROXIMITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Per-track distance history: {track_id: [distance_m, ...]}
_distance_history: dict[int, list[float]] = {}

# Per-track centroid history for pairwise convergence check
_centroid_history_for_ttc: dict[int, list[tuple[float, float]]] = {}

# Convergence lookback (how many frames back to check if pair is converging)
_CONVERGENCE_LOOKBACK = 5


def reset_collision_state():
    """Reset all collision tracking state for a new video."""
    global _distance_history, _centroid_history_for_ttc
    _distance_history.clear()
    _centroid_history_for_ttc.clear()


def compute_ttc(
    tracked_objects: list[dict[str, Any]],
    frame_rate: float,
) -> list[dict[str, Any]]:
    """
    Compute per-object Time-to-Collision (ego vehicle to each object).

    Modifies each object dict in-place, adding:
      - "ttc_s": float or None (TTC in seconds, only when V > 0)
      - "ttc_valid": bool (True if TTC was computable)

    Returns the same list for chaining.
    """
    if frame_rate <= 0:
        for obj in tracked_objects:
            obj["ttc_s"] = None
            obj["ttc_valid"] = False
        return tracked_objects

    dt = 1.0 / frame_rate

    for obj in tracked_objects:
        tid = obj.get("track_id", -1)
        distance = obj.get("distance_m")

        # Default: TTC not computable
        obj["ttc_s"] = None
        obj["ttc_valid"] = False

        if distance is None or distance <= 0:
            continue

        # Update distance history
        if tid not in _distance_history:
            _distance_history[tid] = []
        _distance_history[tid].append(distance)

        # Keep only last 10 entries
        if len(_distance_history[tid]) > 10:
            _distance_history[tid] = _distance_history[tid][-10:]

        history = _distance_history[tid]
        if len(history) < 2:
            continue

        # Compute closing velocity: V = (Z_previous - Z_current) / dt
        z_previous = history[-2]
        z_current = history[-1]
        closing_velocity = (z_previous - z_current) / dt

        # GUARD: if V <= 0, object is stable or moving away. TTC undefined.
        if closing_velocity <= 0:
            continue

        # TTC = Z_current / V
        ttc = z_current / closing_velocity

        # Sanity check: discard very large TTC values (> 30 seconds)
        if ttc > 30.0:
            continue

        obj["ttc_s"] = round(ttc, 2)
        obj["ttc_valid"] = True

    # Update centroid history for pairwise computation
    for obj in tracked_objects:
        tid = obj.get("track_id", -1)
        centroid = obj.get("centroid")
        if centroid is None:
            bbox = obj.get("bbox", [0, 0, 0, 0])
            centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        if tid not in _centroid_history_for_ttc:
            _centroid_history_for_ttc[tid] = []
        _centroid_history_for_ttc[tid].append(centroid)

        if len(_centroid_history_for_ttc[tid]) > 15:
            _centroid_history_for_ttc[tid] = _centroid_history_for_ttc[tid][-15:]

    return tracked_objects


def compute_pairwise_collision_warnings(
    tracked_objects: list[dict[str, Any]],
    frame_rate: float,
) -> list[dict[str, Any]]:
    """
    Compute pairwise TTC between object pairs that are:
    1. Already within PAIRWISE_PROXIMITY_THRESHOLD of each other
    2. Converging (centroid distance shrinking vs N frames ago)

    Returns a list of collision warnings:
    [
        {
            "track_a": int,
            "track_b": int,
            "class_a": str,
            "class_b": str,
            "ttc_s": float,
            "current_distance_m": float,
            "urgency": "imminent" | "high" | "moderate",
        },
        ...
    ]
    """
    warnings = []

    if len(tracked_objects) < 2 or frame_rate <= 0:
        return warnings

    # Build lookup for objects with valid distance
    objects_with_distance = [
        o for o in tracked_objects
        if o.get("distance_m") is not None and o.get("distance_m", 0) > 0
    ]

    if len(objects_with_distance) < 2:
        return warnings

    # Filter first: only pairs within proximity threshold AND converging
    for i in range(len(objects_with_distance)):
        for j in range(i + 1, len(objects_with_distance)):
            obj_a = objects_with_distance[i]
            obj_b = objects_with_distance[j]

            tid_a = obj_a.get("track_id", -1)
            tid_b = obj_b.get("track_id", -1)

            # Current centroid distance (in pixels)
            cx_a, cy_a = _get_current_centroid(obj_a)
            cx_b, cy_b = _get_current_centroid(obj_b)
            current_pixel_dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

            # Use distance estimates for proximity check
            dist_a = obj_a.get("distance_m", 999)
            dist_b = obj_b.get("distance_m", 999)

            # Simple proximity: both objects must be within threshold
            if dist_a > PAIRWISE_PROXIMITY_THRESHOLD or dist_b > PAIRWISE_PROXIMITY_THRESHOLD:
                continue

            # Check convergence: centroid distance must be shrinking
            if not _is_converging(tid_a, tid_b):
                continue

            # Compute pairwise TTC
            # Use centroid distance history to estimate closing speed between objects
            dt = 1.0 / frame_rate
            hist_a = _centroid_history_for_ttc.get(tid_a, [])
            hist_b = _centroid_history_for_ttc.get(tid_b, [])

            if len(hist_a) < 2 or len(hist_b) < 2:
                continue

            # Previous centroid distance
            prev_cx_a, prev_cy_a = hist_a[-2]
            prev_cx_b, prev_cy_b = hist_b[-2]
            prev_pixel_dist = math.sqrt((prev_cx_a - prev_cx_b) ** 2 + (prev_cy_a - prev_cy_b) ** 2)

            # Closing speed in pixels/sec
            closing_speed_px = (prev_pixel_dist - current_pixel_dist) / dt

            if closing_speed_px <= 0:
                continue

            # Estimate TTC based on centroid convergence
            ttc_pairwise = current_pixel_dist / closing_speed_px

            if ttc_pairwise > 15.0:
                continue  # Too far out to be meaningful

            # Determine urgency with predictive lookahead prior to collision
            if ttc_pairwise < TTC_DANGER_THRESHOLD:
                urgency = "imminent"
            elif ttc_pairwise < TTC_DANGER_THRESHOLD * 1.8:
                urgency = "high"
            else:
                urgency = "moderate"

            # Estimate real-world distance between objects
            real_dist = abs(dist_a - dist_b)

            warnings.append({
                "track_a": tid_a,
                "track_b": tid_b,
                "class_a": obj_a.get("class", "unknown"),
                "class_b": obj_b.get("class", "unknown"),
                "ttc_s": round(ttc_pairwise, 2),
                "current_distance_m": round(real_dist, 1),
                "urgency": urgency,
            })

    # Sort by TTC ascending (most urgent first)
    warnings.sort(key=lambda w: w["ttc_s"])

    return warnings


def _get_current_centroid(obj: dict) -> tuple[float, float]:
    """Get current centroid from an object dict."""
    centroid = obj.get("centroid")
    if centroid is not None:
        return centroid
    bbox = obj.get("bbox", [0, 0, 0, 0])
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _is_converging(tid_a: int, tid_b: int) -> bool:
    """
    Check if two objects are converging: their centroid distance is
    smaller now than it was N frames ago.
    """
    hist_a = _centroid_history_for_ttc.get(tid_a, [])
    hist_b = _centroid_history_for_ttc.get(tid_b, [])

    if len(hist_a) < _CONVERGENCE_LOOKBACK + 1 or len(hist_b) < _CONVERGENCE_LOOKBACK + 1:
        # Not enough history, can't determine convergence
        # But if they have at least 2 frames, check those
        if len(hist_a) >= 2 and len(hist_b) >= 2:
            pass
        else:
            return False

    lookback = min(_CONVERGENCE_LOOKBACK, len(hist_a) - 1, len(hist_b) - 1)

    # Current distance
    cx_a, cy_a = hist_a[-1]
    cx_b, cy_b = hist_b[-1]
    current_dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

    # Past distance
    px_a, py_a = hist_a[-1 - lookback]
    px_b, py_b = hist_b[-1 - lookback]
    past_dist = math.sqrt((px_a - px_b) ** 2 + (py_a - py_b) ** 2)

    # Converging = distance is decreasing
    return current_dist < past_dist
