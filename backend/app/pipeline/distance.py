"""
SARTHI Vision - Distance Estimation

Two-part monocular distance estimation per build prompt Section 6.3:

1. PRIMARY (known-size classes): Pinhole camera model
   distance_m = (real_world_width_m * focal_length_px) / bbox_width_px

2. SECONDARY (unknown-size classes like pothole): Depth-Anything-V2-Small
   - Run depth estimation to get relative depth map
   - Calibrate using 2-3 confident primary-method anchors from the same frame
   - Linear fit between relative depth and metric distance
   - Carry forward calibration across frames when anchors unavailable

Edge case: frames with zero primary anchors (e.g. only a pothole and empty road)
carry forward the most recent successful calibration from a nearby frame.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from app.config import (
    DEFAULT_FOCAL_LENGTH_PX,
    REAL_WORLD_WIDTHS_M,
)

logger = logging.getLogger(__name__)

# Depth-Anything model singleton
_depth_model = None
_depth_processor = None

# Persistent calibration state (carried forward across frames)
_last_calibration: dict[str, float] | None = None  # {"scale": ..., "offset": ...}


def reset_distance_state():
    """Reset calibration state and distance history for a new video."""
    global _last_calibration, _distance_history
    _last_calibration = None
    _distance_history = {}


def _load_depth_model():
    """Lazy-load Depth-Anything-V2-Small model."""
    global _depth_model, _depth_processor
    if _depth_model is not None:
        return

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        logger.info("Loading Depth-Anything-V2-Small model...")
        _depth_processor = AutoImageProcessor.from_pretrained(model_name)
        _depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
        _depth_model.eval()
        logger.info("Depth-Anything-V2-Small loaded successfully")
    except Exception as e:
        logger.warning("Failed to load Depth-Anything model: %s", e)
        _depth_model = None
        _depth_processor = None


def estimate_distances(
    tracked_objects: list[dict[str, Any]],
    frame: np.ndarray,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
    bbox_histories: dict[int, list[list[float]]] | None = None,
) -> list[dict[str, Any]]:
    """
    Estimate distance for each tracked object.

    Modifies each object dict in-place, adding:
      - "distance_m": estimated distance in meters (or None)
      - "distance_available": True if distance was computed
      - "distance_method": "pinhole" or "depth_model" or "unavailable"
      - "closing_speed": rate of distance change (m/frame, negative = approaching)

    Returns the same list for chaining.
    """
    global _last_calibration

    if not tracked_objects:
        return tracked_objects

    # --- Phase 1: Primary (pinhole) distance for known-size classes ---
    primary_anchors = []  # [(depth_value, metric_distance)] for calibration

    for obj in tracked_objects:
        cls = obj.get("class", "")
        bbox = obj.get("bbox", [0, 0, 0, 0])
        bbox_width_px = bbox[2] - bbox[0]

        if bbox_width_px <= 0:
            obj["distance_m"] = None
            obj["distance_available"] = False
            obj["distance_method"] = "unavailable"
            obj["closing_speed"] = 0.0
            continue

        known_width = REAL_WORLD_WIDTHS_M.get(cls)

        if known_width is not None and bbox_width_px > 10:
            # Pinhole formula
            distance_m = (known_width * focal_length_px) / bbox_width_px
            # Clamp to reasonable range (0.5m to 200m)
            distance_m = max(0.5, min(200.0, distance_m))
            obj["distance_m"] = round(distance_m, 1)
            obj["distance_available"] = True
            obj["distance_method"] = "pinhole"

            # Store as calibration anchor
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            primary_anchors.append({
                "pixel": (cx, cy),
                "distance_m": distance_m,
                "confidence": obj.get("confidence", 0.5),
            })
        else:
            # Will be filled by depth model (Phase 2)
            obj["distance_m"] = None
            obj["distance_available"] = False
            obj["distance_method"] = "pending_depth"

    # --- Phase 2: Secondary (Depth-Anything-V2) for remaining objects ---
    pending_objects = [o for o in tracked_objects if o.get("distance_method") == "pending_depth"]

    if pending_objects:
        depth_map = _get_depth_map(frame)

        if depth_map is not None:
            # Calibrate depth map using primary anchors
            calibration = _calibrate_depth(depth_map, primary_anchors)

            if calibration is not None:
                _last_calibration = calibration
            elif _last_calibration is not None:
                # Carry forward previous calibration
                calibration = _last_calibration
                logger.debug("Using carried-forward depth calibration")

            for obj in pending_objects:
                bbox = obj.get("bbox", [0, 0, 0, 0])
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                # Clamp to depth map bounds
                cx = max(0, min(cx, depth_map.shape[1] - 1))
                cy = max(0, min(cy, depth_map.shape[0] - 1))

                # Sample depth at object center (use small patch for robustness)
                patch_size = 5
                y1_p = max(0, cy - patch_size)
                y2_p = min(depth_map.shape[0], cy + patch_size)
                x1_p = max(0, cx - patch_size)
                x2_p = min(depth_map.shape[1], cx + patch_size)
                depth_value = float(np.median(depth_map[y1_p:y2_p, x1_p:x2_p]))

                if calibration is not None:
                    # Apply calibration: distance = scale * depth + offset
                    distance_m = calibration["scale"] * depth_value + calibration["offset"]
                    distance_m = max(0.5, min(200.0, distance_m))
                    obj["distance_m"] = round(distance_m, 1)
                    obj["distance_available"] = True
                    obj["distance_method"] = "depth_model"
                else:
                    # No calibration available at all
                    obj["distance_m"] = None
                    obj["distance_available"] = False
                    obj["distance_method"] = "unavailable"
        else:
            # Depth model not available
            for obj in pending_objects:
                obj["distance_m"] = None
                obj["distance_available"] = False
                obj["distance_method"] = "unavailable"

    # --- Phase 3: Compute closing speed from distance history ---
    for obj in tracked_objects:
        obj["closing_speed"] = _compute_closing_speed(
            obj["track_id"], obj.get("distance_m"), bbox_histories
        )

    return tracked_objects


# Distance history for closing speed computation
_distance_history: dict[int, list[float]] = {}


def _compute_closing_speed(
    track_id: int,
    current_distance: float | None,
    bbox_histories: dict[int, list[list[float]]] | None,
) -> float:
    """
    Compute rate of distance change (m/frame).
    Negative = object is approaching (closing).
    Positive = object is moving away.
    """
    global _distance_history

    if current_distance is None:
        return 0.0

    if track_id not in _distance_history:
        _distance_history[track_id] = []

    _distance_history[track_id].append(current_distance)

    # Keep only last 10 entries
    if len(_distance_history[track_id]) > 10:
        _distance_history[track_id] = _distance_history[track_id][-10:]

    history = _distance_history[track_id]
    if len(history) < 2:
        return 0.0

    # Simple: rate of change between last two measurements
    closing_speed = history[-1] - history[-2]
    return round(closing_speed, 3)


def _get_depth_map(frame: np.ndarray) -> np.ndarray | None:
    """
    Run Depth-Anything-V2-Small on a frame to get a relative depth map.
    Returns a 2D numpy array of relative depth values (higher = farther).
    """
    _load_depth_model()

    if _depth_model is None or _depth_processor is None:
        return None

    try:
        import torch
        from PIL import Image

        # Convert BGR to RGB PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Preprocess
        inputs = _depth_processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = _depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original frame size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False,
        )

        depth_map = prediction.squeeze().cpu().numpy()
        return depth_map

    except Exception as e:
        logger.warning("Depth estimation failed: %s", e)
        return None


def _calibrate_depth(
    depth_map: np.ndarray,
    anchors: list[dict],
) -> dict[str, float] | None:
    """
    Calibrate the relative depth map using primary-method distance anchors.
    Performs a simple linear regression: distance_m = scale * depth_value + offset

    Requires at least 2 confident anchors for a meaningful fit.
    """
    if len(anchors) < 2:
        return None

    # Sort by confidence, take top 3
    sorted_anchors = sorted(anchors, key=lambda a: a["confidence"], reverse=True)[:3]

    depth_values = []
    metric_distances = []

    for anchor in sorted_anchors:
        px, py = anchor["pixel"]
        # Clamp to depth map bounds
        px = max(0, min(px, depth_map.shape[1] - 1))
        py = max(0, min(py, depth_map.shape[0] - 1))

        # Sample depth (use median of small patch)
        ps = 5
        y1 = max(0, py - ps)
        y2 = min(depth_map.shape[0], py + ps)
        x1 = max(0, px - ps)
        x2 = min(depth_map.shape[1], px + ps)

        depth_val = float(np.median(depth_map[y1:y2, x1:x2]))
        depth_values.append(depth_val)
        metric_distances.append(anchor["distance_m"])

    depth_arr = np.array(depth_values)
    dist_arr = np.array(metric_distances)

    # Check for degenerate case (all same depth value)
    if np.std(depth_arr) < 1e-6:
        return None

    # Simple linear fit: dist = scale * depth + offset
    try:
        coeffs = np.polyfit(depth_arr, dist_arr, deg=1)
        scale = float(coeffs[0])
        offset = float(coeffs[1])
        return {"scale": scale, "offset": offset}
    except Exception:
        return None
