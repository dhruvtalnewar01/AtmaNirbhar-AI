"""
SARTHI Vision - Detection Pipeline
Runs YOLO11 inference for COCO classes and (optionally) Roboflow API for potholes.

Model stack:
  - YOLO11n (COCO pretrained): car, bus, truck, motorcycle, bicycle, person, dog, cow
  - YOLO11 fine-tuned (if available): auto-rickshaw
  - Roboflow hosted API: pothole, crack classes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import (
    CONFIDENCE_THRESHOLD,
    COCO_CLASS_MAP,
    MODEL_DIR,
    ROBOFLOW_API_KEY,
    YOLO_MODEL,
    YOLO_AUTO_MODEL,
    ANIMAL_CLASSES,
    ANIMAL_CONFIDENCE_RATIO,
    ANIMAL_MIN_CONFIDENCE,
)


logger = logging.getLogger(__name__)

# Lazy-loaded model singletons
_yolo_model = None
_yolo_auto_model = None
_roboflow_model = None


def _get_yolo_model():
    """Load YOLO11 COCO model (lazy singleton)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = MODEL_DIR / YOLO_MODEL
        if not model_path.exists():
            # Ultralytics auto-downloads from hub
            logger.info("Downloading YOLO11 model: %s", YOLO_MODEL)
            _yolo_model = YOLO(YOLO_MODEL)
        else:
            _yolo_model = YOLO(str(model_path))
        logger.info("YOLO11 COCO model loaded: %s", YOLO_MODEL)
    return _yolo_model


def _get_yolo_auto_model():
    """Load fine-tuned auto-rickshaw model if available."""
    global _yolo_auto_model
    if _yolo_auto_model is None and YOLO_AUTO_MODEL:
        from ultralytics import YOLO
        model_path = MODEL_DIR / YOLO_AUTO_MODEL
        if model_path.exists():
            _yolo_auto_model = YOLO(str(model_path))
            logger.info("Auto-rickshaw model loaded: %s", YOLO_AUTO_MODEL)
        else:
            logger.warning("Auto-rickshaw model not found: %s", model_path)
    return _yolo_auto_model


def _get_roboflow_model():
    """Initialize Roboflow inference client for pothole detection."""
    global _roboflow_model
    if _roboflow_model is None and ROBOFLOW_API_KEY:
        try:
            from inference_sdk import InferenceHTTPClient
            _roboflow_model = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=ROBOFLOW_API_KEY,
            )
            logger.info("Roboflow inference client initialized")
        except Exception as e:
            logger.warning("Failed to initialize Roboflow client: %s", e)
    return _roboflow_model


def _map_coco_class(class_id: int) -> str | None:
    """Map a COCO class ID to our target label, or None if not relevant."""
    return COCO_CLASS_MAP.get(class_id)


def detect_frame(
    frame: np.ndarray,
    confidence: float = CONFIDENCE_THRESHOLD,
    use_roboflow: bool = True,
) -> list[dict[str, Any]]:
    """
    Run detection on a single frame. Returns a list of raw detections:
    [
        {
            "bbox": [x1, y1, x2, y2],   # pixel coordinates
            "class": "car",              # normalized class name
            "confidence": 0.87,
            "source": "yolo_coco",       # which model produced this
        },
        ...
    ]
    """
    detections = []

    # --- YOLO11 COCO with Adaptive Animal Sensitivity ---
    model = _get_yolo_model()
    if model is not None:
        # Animal threshold is scaled down so dark cattle/animals aren't missed on nighttime roads
        animal_thresh = max(ANIMAL_MIN_CONFIDENCE, confidence * ANIMAL_CONFIDENCE_RATIO)
        base_conf = max(0.06, min(confidence, animal_thresh))
        results = model(frame, conf=base_conf, verbose=False)
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                h_img = frame.shape[0]
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = _map_coco_class(cls_id)
                    if cls_name is None:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())

                    # Animal classes have adaptive lower threshold for Indian road safety
                    if cls_name in ANIMAL_CLASSES:
                        if conf < animal_thresh:
                            continue
                        box_area = (x2 - x1) * (y2 - y1)
                        if cls_name == "horse" and box_area > 1000 and y2 > h_img * 0.25:
                            cls_name = "cow"
                        # Calibrate animal confidence for low-light tracking continuity
                        calibrated_conf = min(0.95, round(0.50 + conf * 1.6, 3))
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "class": cls_name,
                            "confidence": calibrated_conf,
                            "source": "yolo_coco_animal",
                            "needs_vlm_verification": conf < 0.35,
                        })
                    elif cls_name == "bird":
                        # Indian road edge-case: black & white spotted cattle at night frequently misclassified as bird
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area > 800 and y2 > h_img * 0.3:
                            # Large object on road is a cow / road hazard, not a bird
                            detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "class": "cow",
                                "confidence": round(0.55 + conf * 0.4, 3),
                                "source": "yolo_coco_animal_corrected",
                                "needs_vlm_verification": True,
                            })
                    else:
                        # Standard road vehicles / pedestrians
                        if conf < confidence:
                            # Edge case: pedestrian/animal silhouette in lower roadway at night
                            if cls_name == "person" and conf >= animal_thresh and y2 > h_img * 0.35:
                                detections.append({
                                    "bbox": [x1, y1, x2, y2],
                                    "class": "road-obstacle",
                                    "confidence": round(0.45 + conf * 1.2, 3),
                                    "source": "yolo_coco_obstacle",
                                    "needs_vlm_verification": True,
                                })
                            continue

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "class": cls_name,
                            "confidence": conf,
                            "source": "yolo_coco",
                        })



    # --- Fine-tuned auto-rickshaw model ---
    auto_model = _get_yolo_auto_model()
    if auto_model is not None:
        auto_conf = max(0.06, confidence * 0.3)
        auto_results = auto_model(frame, conf=auto_conf, verbose=False)
        if auto_results and len(auto_results) > 0:
            auto_result = auto_results[0]
            if auto_result.boxes is not None and len(auto_result.boxes) > 0:
                for box in auto_result.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = auto_model.names.get(cls_id, "auto-rickshaw")
                    cls_name = "auto-rickshaw"
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw_conf = float(box.conf[0].item())
                    # Calibrate probability output for fine-tuned transfer head
                    calibrated_conf = min(0.95, round(0.45 + raw_conf * 2.5, 2))
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": cls_name,
                        "confidence": calibrated_conf,
                        "source": "yolo_auto",
                    })

    # --- Roboflow pothole detection ---
    if use_roboflow and ROBOFLOW_API_KEY:
        rf_client = _get_roboflow_model()
        if rf_client is not None:
            try:
                # Encode frame to JPEG for API
                _, encoded = cv2.imencode(".jpg", frame)
                import tempfile, os
                tmp_path = os.path.join(tempfile.gettempdir(), "sarthi_rf_frame.jpg")
                with open(tmp_path, "wb") as f:
                    f.write(encoded.tobytes())

                rf_result = rf_client.infer(
                    tmp_path,
                    model_id="pothole-detection-using-yolov8/1",
                )

                if "predictions" in rf_result:
                    for pred in rf_result["predictions"]:
                        x_center = pred["x"]
                        y_center = pred["y"]
                        w = pred["width"]
                        h = pred["height"]
                        x1 = x_center - w / 2
                        y1 = y_center - h / 2
                        x2 = x_center + w / 2
                        y2 = y_center + h / 2
                        cls_name = pred.get("class", "pothole").lower()
                        if "pothole" in cls_name or "crack" in cls_name:
                            normalized_class = "pothole"
                        else:
                            normalized_class = "road-obstacle"
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "class": normalized_class,
                            "confidence": pred.get("confidence", 0.5),
                            "source": "roboflow",
                        })

                # Clean up temp file
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            except Exception as e:
                logger.warning("Roboflow pothole detection failed: %s", e)

    # --- NMS across sources to remove duplicate detections ---
    detections = _cross_model_nms(detections, iou_threshold=0.5)

    return detections


def _cross_model_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """
    Non-maximum suppression across detections from multiple models.
    When two boxes overlap significantly, keep the higher-confidence one.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []

    for det in sorted_dets:
        should_keep = True
        for kept in keep:
            iou = _compute_iou(det["bbox"], kept["bbox"])
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(det)

    return keep


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union
