"""
SARTHI Vision - Ego-Motion-Compensated Motion Classification

This is the single most technically important piece of the prototype.
Per build prompt Section 6.2:

Algorithm:
1. Collect centroid displacement vectors for all tracked objects vs N frames ago.
2. Compute MEDIAN displacement vector across all tracked boxes (approximates
   camera ego-motion, since most of the scene is static).
3. Subtract the median from each object's displacement.
4. If relative displacement > threshold, classify as Moving; else Stationary.

Sparse-scene fallback (< 3 objects):
When fewer than 3 objects are tracked, the median trick fails (it just becomes
the single object's own displacement, forcing relative = 0). In this case,
fall back to optical flow (cv2.calcOpticalFlowFarneback) on the full frame
for direct ego-motion estimation.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from app.config import (
    MOTION_LOOKBACK_FRAMES,
    MOTION_PIXEL_THRESHOLD,
    MIN_OBJECTS_FOR_MEDIAN,
)

logger = logging.getLogger(__name__)

# Cache for the previous frame (for optical flow fallback)
_prev_gray_frame: np.ndarray | None = None


def reset_motion_state():
    """Reset the optical flow cache for a new video."""
    global _prev_gray_frame
    _prev_gray_frame = None


def classify_motion(
    tracked_objects: list[dict[str, Any]],
    centroid_histories: dict[int, list[tuple[float, float]]],
    current_frame: np.ndarray,
    lookback: int = MOTION_LOOKBACK_FRAMES,
    threshold: float = MOTION_PIXEL_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Classify each tracked object as 'Moving' or 'Stationary' with
    ego-motion compensation.

    Modifies each object dict in-place, adding:
      - "status": "Moving" or "Stationary"
      - "rel_dx": relative horizontal displacement (after ego-motion subtraction)
      - "rel_dy": relative vertical displacement
      - "raw_dx": raw displacement (before ego-motion subtraction)
      - "raw_dy": raw displacement

    Returns the same list for chaining.
    """
    global _prev_gray_frame

    # Compute raw displacement vectors for all objects that have enough history
    displacement_vectors = {}
    for obj in tracked_objects:
        tid = obj["track_id"]
        history = centroid_histories.get(tid, [])

        if len(history) < 2:
            # Not enough history, default to Stationary
            obj["status"] = "Stationary"
            obj["rel_dx"] = 0.0
            obj["rel_dy"] = 0.0
            obj["raw_dx"] = 0.0
            obj["raw_dy"] = 0.0
            continue

        # Look back N frames (or as far as history goes)
        lookback_idx = max(0, len(history) - 1 - lookback)
        old_cx, old_cy = history[lookback_idx]
        new_cx, new_cy = history[-1]

        raw_dx = new_cx - old_cx
        raw_dy = new_cy - old_cy

        obj["raw_dx"] = raw_dx
        obj["raw_dy"] = raw_dy
        displacement_vectors[tid] = (raw_dx, raw_dy)

    # Only process objects that have displacement vectors
    objects_with_displacement = [
        o for o in tracked_objects if o["track_id"] in displacement_vectors
    ]

    if not objects_with_displacement:
        return tracked_objects

    # --- Determine ego-motion estimate ---
    num_tracked = len(displacement_vectors)

    if num_tracked >= MIN_OBJECTS_FOR_MEDIAN:
        # MEDIAN METHOD: robust ego-motion approximation
        # Most objects are static background/roadside, so median displacement
        # approximates the camera's own movement.
        all_dx = [v[0] for v in displacement_vectors.values()]
        all_dy = [v[1] for v in displacement_vectors.values()]
        ego_dx = float(np.median(all_dx))
        ego_dy = float(np.median(all_dy))
    else:
        # SPARSE SCENE FALLBACK: optical flow on full frame
        # The median method needs >= 3 objects. With 1-2 objects,
        # it just equals the object's own displacement, which zeros out
        # the relative displacement and falsely reads everything as Stationary.
        ego_dx, ego_dy = _estimate_ego_motion_optical_flow(current_frame)

    # --- Subtract ego-motion and classify ---
    for obj in objects_with_displacement:
        tid = obj["track_id"]
        raw_dx, raw_dy = displacement_vectors[tid]

        rel_dx = raw_dx - ego_dx
        rel_dy = raw_dy - ego_dy

        obj["rel_dx"] = rel_dx
        obj["rel_dy"] = rel_dy

        # Magnitude of relative displacement
        rel_magnitude = np.sqrt(rel_dx ** 2 + rel_dy ** 2)

        if rel_magnitude > threshold:
            obj["status"] = "Moving"
        else:
            obj["status"] = "Stationary"

    # Update previous frame for next optical flow call
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    _prev_gray_frame = gray

    return tracked_objects


def _estimate_ego_motion_optical_flow(current_frame: np.ndarray) -> tuple[float, float]:
    """
    Estimate camera ego-motion using dense optical flow (Farneback method)
    on the full frame. Returns (ego_dx, ego_dy) in pixels.

    This is the fallback for sparse scenes where the median-of-tracked-boxes
    method cannot reliably estimate ego-motion.
    """
    global _prev_gray_frame

    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    if _prev_gray_frame is None:
        _prev_gray_frame = gray
        return (0.0, 0.0)

    # Ensure same dimensions
    if _prev_gray_frame.shape != gray.shape:
        _prev_gray_frame = gray
        return (0.0, 0.0)

    try:
        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            _prev_gray_frame, gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # The median flow vector across the entire frame approximates
        # the camera's global motion (ego-motion)
        ego_dx = float(np.median(flow[..., 0]))
        ego_dy = float(np.median(flow[..., 1]))

        # Scale to match the lookback window
        # (optical flow gives per-frame displacement, we need N-frame displacement)
        ego_dx *= MOTION_LOOKBACK_FRAMES
        ego_dy *= MOTION_LOOKBACK_FRAMES

        return (ego_dx, ego_dy)

    except Exception as e:
        logger.warning("Optical flow estimation failed: %s", e)
        return (0.0, 0.0)
