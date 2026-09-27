"""
SARTHI Vision - Risk Scoring and Scene Chaos Score

Per-object risk scoring (upgraded with TTC):
  risk_score = w1*class_risk + w2*proximity + w3*motion + w4*closing + w5*ttc

Scene Chaos Score (0-100):
  Composite of object count, Shannon entropy, mean proximity,
  high-risk share, and TTC urgency. A scene with a sub-2-second TTC
  anywhere should read as more chaotic than one where every object is stable.

motion_factor definition (since there is no trajectory planner):
  - High: object is both closing AND trending toward horizontal center
  - Moderate: only one of those conditions is true
  - Low: stationary, moving away, or moving laterally out of frame
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from app.config import (
    RISK_W1_CLASS,
    RISK_W2_PROXIMITY,
    RISK_W3_MOTION,
    RISK_W4_CLOSING,
    RISK_W5_TTC,
    CLASS_RISK_WEIGHTS,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MODERATE,
    CHAOS_W_COUNT,
    CHAOS_W_ENTROPY,
    CHAOS_W_PROXIMITY,
    CHAOS_W_HIGH_RISK,
    CHAOS_W_TTC,
    TTC_DANGER_THRESHOLD,
    ANIMAL_CLASSES,
    SIREN_CHAOS_BOOST,
)

logger = logging.getLogger(__name__)


def compute_risk_scores(
    tracked_objects: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """
    Compute per-object risk score and level, now including TTC.

    Modifies each object dict in-place, adding:
      - "risk_score": float 0-1
      - "risk_level": "Low", "Moderate", or "High"
      - "risk_explanation": plain-language explanation

    Returns the same list for chaining.
    """
    frame_center_x = frame_width / 2.0

    for obj in tracked_objects:
        cls = obj.get("class", "unknown")
        distance = obj.get("distance_m")
        status = obj.get("status", "Stationary")
        rel_dx = obj.get("rel_dx", 0.0)
        rel_dy = obj.get("rel_dy", 0.0)
        closing_speed = obj.get("closing_speed", 0.0)
        ttc = obj.get("ttc_s")
        bbox = obj.get("bbox", [0, 0, 0, 0])
        centroid_x = (bbox[0] + bbox[2]) / 2.0

        # --- Component 1: Class risk weight ---
        class_weight = CLASS_RISK_WEIGHTS.get(cls, 0.5)

        # --- Component 2: Proximity factor (calibrated for 0-40m automotive envelope) ---
        if distance is not None and distance > 0:
            proximity_factor = max(0.0, min(1.0, (40.0 - distance) / 38.0))
        else:
            proximity_factor = 0.35

        # --- Component 3: Motion factor ---
        is_closing = closing_speed < -0.1
        is_toward_center = False

        if status == "Moving":
            obj_offset_from_center = centroid_x - frame_center_x
            if obj_offset_from_center > 0 and rel_dx < 0:
                is_toward_center = True
            elif obj_offset_from_center < 0 and rel_dx > 0:
                is_toward_center = True
            elif abs(obj_offset_from_center) < frame_width * 0.15:
                is_toward_center = True

        if is_closing and is_toward_center:
            motion_factor = 0.9
        elif is_closing or is_toward_center:
            motion_factor = 0.5
        elif status == "Moving":
            motion_factor = 0.3
        else:
            motion_factor = 0.1

        # --- Component 4: Closing speed factor ---
        if closing_speed < -0.5:
            closing_factor = 0.9
        elif closing_speed < -0.1:
            closing_factor = 0.5
        elif closing_speed > 0.1:
            closing_factor = 0.1
        else:
            closing_factor = 0.2

        # --- Component 5: TTC factor (new) ---
        if ttc is not None and ttc > 0:
            if ttc < TTC_DANGER_THRESHOLD:
                ttc_factor = 1.0  # Imminent collision
            elif ttc < TTC_DANGER_THRESHOLD * 2:
                ttc_factor = 0.7  # Approaching danger
            elif ttc < TTC_DANGER_THRESHOLD * 5:
                ttc_factor = 0.4  # Worth monitoring
            else:
                ttc_factor = 0.1  # Plenty of time
        else:
            ttc_factor = 0.0  # TTC not applicable (not closing)

        # --- Combined risk score ---
        risk_score = (
            RISK_W1_CLASS * class_weight
            + RISK_W2_PROXIMITY * proximity_factor
            + RISK_W3_MOTION * motion_factor
            + RISK_W4_CLOSING * closing_factor
            + RISK_W5_TTC * ttc_factor
        )

        # Clamp to [0, 1]
        risk_score = max(0.0, min(1.0, risk_score))

        # --- Risk level classification ---
        # Prior Predictive Alert: sub-threshold TTC always escalates to High before impact
        if ttc is not None and ttc < TTC_DANGER_THRESHOLD:
            risk_level = "High"
        elif ttc is not None and ttc < TTC_DANGER_THRESHOLD * 1.8:
            # Advance predictive warning (issued up to ~7.2s prior to collision)
            risk_level = "High"
        elif cls in ANIMAL_CLASSES and distance is not None and distance <= 22.0:
            # Unpredictable live animal on roadway within vehicle braking distance
            risk_level = "High"
        elif risk_score >= RISK_THRESHOLD_HIGH:
            risk_level = "High"
        elif risk_score >= RISK_THRESHOLD_MODERATE or (cls in ANIMAL_CLASSES and distance is not None and distance <= 32.0):
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # --- Generate plain-language explanation ---
        explanation = _generate_risk_explanation(
            cls=cls,
            distance=distance,
            status=status,
            risk_level=risk_level,
            is_closing=is_closing,
            is_toward_center=is_toward_center,
            class_weight=class_weight,
            proximity_factor=proximity_factor,
            ttc=ttc,
        )

        obj["risk_score"] = round(risk_score, 3)
        obj["risk_level"] = risk_level
        obj["risk_explanation"] = explanation

    return tracked_objects


def compute_chaos_score(
    tracked_objects: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    siren_active: bool = False,
) -> float:
    """
    Compute Scene Chaos Score (0-100) for a single frame.

    Components:
    1. Object count relative to frame area
    2. Shannon entropy of the class mix present
    3. Mean proximity of all objects
    4. Share of objects scored as High risk
    5. TTC urgency: sub-threshold TTC elevates chaos
    6. Siren detection: audible siren boosts chaos (v3 Section 2)
    """
    if not tracked_objects:
        return 0.0

    n_objects = len(tracked_objects)

    # --- Component 1: Object density ---
    density_score = min(1.0, n_objects / 20.0) * 100

    # --- Component 2: Shannon entropy of class mix ---
    class_counts: dict[str, int] = {}
    for obj in tracked_objects:
        cls = obj.get("class", "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    entropy = 0.0
    for count in class_counts.values():
        p = count / n_objects
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(max(len(class_counts), 1)) if class_counts else 0
    entropy_score = (entropy / max_entropy * 100) if max_entropy > 0 else 0

    # --- Component 3: Mean proximity ---
    distances = [
        obj.get("distance_m") for obj in tracked_objects
        if obj.get("distance_m") is not None
    ]
    if distances:
        mean_dist = sum(distances) / len(distances)
        proximity_score = min(100.0, (10.0 / max(mean_dist, 1.0)) * 100)
    else:
        proximity_score = 50.0

    # --- Component 4: High-risk share ---
    high_risk_count = sum(
        1 for obj in tracked_objects if obj.get("risk_level") == "High"
    )
    high_risk_share = (high_risk_count / n_objects) * 100

    # --- Component 5: TTC urgency (new) ---
    # A scene with any sub-threshold TTC reads as more chaotic
    ttc_values = [
        obj.get("ttc_s") for obj in tracked_objects
        if obj.get("ttc_s") is not None and obj.get("ttc_s", 999) > 0
    ]
    if ttc_values:
        min_ttc = min(ttc_values)
        if min_ttc < TTC_DANGER_THRESHOLD:
            ttc_score = 100.0  # Imminent collision in scene
        elif min_ttc < TTC_DANGER_THRESHOLD * 3:
            ttc_score = 60.0
        elif min_ttc < TTC_DANGER_THRESHOLD * 5:
            ttc_score = 30.0
        else:
            ttc_score = 10.0
    else:
        ttc_score = 0.0  # No closing objects

    # --- Weighted composite ---
    chaos_score = (
        CHAOS_W_COUNT * density_score
        + CHAOS_W_ENTROPY * entropy_score
        + CHAOS_W_PROXIMITY * proximity_score
        + CHAOS_W_HIGH_RISK * high_risk_share
        + CHAOS_W_TTC * ttc_score
    )

    # --- Component 6: Siren boost (v3 Section 2) ---
    # A scene with an audible siren is more chaotic regardless of visual state
    if siren_active:
        chaos_score += SIREN_CHAOS_BOOST

    return round(max(0.0, min(100.0, chaos_score)), 1)


def _generate_risk_explanation(
    cls: str,
    distance: float | None,
    status: str,
    risk_level: str,
    is_closing: bool,
    is_toward_center: bool,
    class_weight: float,
    proximity_factor: float,
    ttc: float | None = None,
) -> str:
    """
    Generate a plain-language explanation of why an object received its risk score.
    """
    parts = [f"{risk_level} risk"]

    # Class reasoning
    if class_weight >= 0.8:
        parts.append(f"{cls} class (unpredictable motion)")
    elif class_weight >= 0.6:
        parts.append(f"{cls} class (moderate predictability)")
    else:
        parts.append(f"{cls} class")

    # Distance
    if distance is not None:
        parts.append(f"{distance}m distance (estimated)")
    else:
        parts.append("distance unavailable")

    # Motion status
    parts.append(status.lower())

    # TTC
    if ttc is not None and ttc > 0:
        parts.append(f"TTC {ttc:.1f}s")

    # Closing behavior
    if is_closing and is_toward_center:
        parts.append("closing toward center of lane")
    elif is_closing:
        parts.append("approaching")
    elif is_toward_center:
        parts.append("trending toward lane center")

    return ": ".join(parts[:2]) + ", " + ", ".join(parts[2:])
