"""
SARTHI Vision - Object Tracking with Majority Voting
Uses Ultralytics' built-in ByteTrack tracker with persist=True for stable
track IDs across a frame sequence. Maintains centroid history buffer and
track-level majority voting for class stabilization.

Section 3.1 of upgrade prompt: for every tracked object, keep a rolling
history of per-frame class labels. Display and act on the majority class
across that window, not the current frame's raw label alone.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from app.config import (
    CONFIDENCE_THRESHOLD,
    MODEL_DIR,
    YOLO_MODEL,
    MOTION_LOOKBACK_FRAMES,
    CLASS_VOTE_WINDOW,
    VLM_CONFIDENCE_THRESHOLD,
    ANIMAL_CLASSES,
    ANIMAL_CONFIDENCE_RATIO,
    ANIMAL_MIN_CONFIDENCE,
    COCO_CLASS_MAP,
)


logger = logging.getLogger(__name__)

_tracker_model = None


def _get_tracker_model():
    """Load YOLO model configured for tracking (lazy singleton)."""
    global _tracker_model
    if _tracker_model is None:
        from ultralytics import YOLO
        model_path = MODEL_DIR / YOLO_MODEL
        if not model_path.exists():
            _tracker_model = YOLO(YOLO_MODEL)
        else:
            _tracker_model = YOLO(str(model_path))
        logger.info("Tracker model loaded: %s", YOLO_MODEL)
    return _tracker_model


class ObjectTracker:
    """
    Wraps Ultralytics .track() to provide:
    - Persistent track IDs across frames
    - Centroid history buffer (last N frames per track)
    - Track-to-class mapping with majority voting
    - VLM verification flagging for ambiguous detections
    """

    def __init__(self):
        self.centroid_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self.class_history: dict[int, list[str]] = defaultdict(list)
        self.bbox_history: dict[int, list[list[float]]] = defaultdict(list)
        self.frame_count = 0
        self._model = None

    def reset(self):
        """Reset tracker state for a new video/image sequence."""
        self.centroid_history.clear()
        self.class_history.clear()
        self.bbox_history.clear()
        self.frame_count = 0
        # Force a new model instance to reset internal tracker state
        self._model = None

    def track_frame(
        self,
        frame: np.ndarray,
        raw_detections: list[dict[str, Any]] | None = None,
        confidence: float = CONFIDENCE_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """
        Run tracking on a single frame.

        If raw_detections is None, runs YOLO tracking directly.
        Otherwise, maps raw detections to tracked objects using the
        built-in tracker.

        Returns list of tracked detections with:
        - track_id
        - class (majority-voted)
        - class_raw (this frame's raw label)
        - needs_vlm_verification (True if ambiguous)
        """
        model = self._get_model()
        self.frame_count += 1

        # Compute adaptive animal threshold for Indian road safety
        animal_thresh = max(ANIMAL_MIN_CONFIDENCE, confidence * ANIMAL_CONFIDENCE_RATIO)
        track_conf = max(0.06, min(confidence, animal_thresh))

        # Use ultralytics built-in tracker
        # persist=True keeps track IDs stable across the frame sequence
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=track_conf,
            verbose=False,
        )

        tracked_objects = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                h_img = frame.shape[0]
                for box in result.boxes:
                    # Skip if no track ID assigned yet
                    if box.id is None:
                        continue

                    track_id = int(box.id[0].item())
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Map COCO class ID to our label
                    cls_name = COCO_CLASS_MAP.get(cls_id)
                    if cls_name is None:
                        continue

                    box_area = (x2 - x1) * (y2 - y1)

                    # Indian road animal hazard handling
                    if cls_name in ANIMAL_CLASSES:
                        if conf < animal_thresh:
                            continue
                        if cls_name == "horse" and box_area > 1000 and y2 > h_img * 0.25:
                            cls_name = "cow"
                        conf = min(0.95, round(0.50 + conf * 1.6, 3))
                    elif cls_id == 14 or cls_name == "bird":
                        if box_area > 800 and y2 > h_img * 0.3:
                            cls_name = "cow"
                            conf = round(0.55 + conf * 0.4, 3)
                        else:
                            continue
                    else:
                        if conf < confidence:
                            if cls_name == "person" and conf >= animal_thresh and y2 > h_img * 0.35:
                                cls_name = "road-obstacle"
                            else:
                                continue

                    # Compute centroid
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # Update history buffers
                    self.centroid_history[track_id].append((cx, cy))
                    self.class_history[track_id].append(cls_name)
                    self.bbox_history[track_id].append([x1, y1, x2, y2])

                    # Trim to lookback window
                    max_centroid_history = MOTION_LOOKBACK_FRAMES + 2
                    if len(self.centroid_history[track_id]) > max_centroid_history:
                        self.centroid_history[track_id] = (
                            self.centroid_history[track_id][-max_centroid_history:]
                        )
                    if len(self.bbox_history[track_id]) > max_centroid_history:
                        self.bbox_history[track_id] = (
                            self.bbox_history[track_id][-max_centroid_history:]
                        )

                    # Class history: keep last CLASS_VOTE_WINDOW entries
                    if len(self.class_history[track_id]) > CLASS_VOTE_WINDOW:
                        self.class_history[track_id] = (
                            self.class_history[track_id][-CLASS_VOTE_WINDOW:]
                        )

                    # --- Majority Voting (Section 3.1) ---
                    majority_class = self.get_majority_class(track_id)

                    # --- VLM Verification Flagging (Section 3.2) ---
                    # Flag if raw class disagrees with majority AND confidence is low
                    needs_vlm = (
                        cls_name != majority_class
                        and conf < VLM_CONFIDENCE_THRESHOLD
                        and len(self.class_history[track_id]) >= 3
                    )

                    tracked_objects.append({
                        "track_id": track_id,
                        "bbox": [x1, y1, x2, y2],
                        "class": majority_class,  # Use majority-voted class
                        "class_raw": cls_name,     # Raw per-frame class
                        "confidence": conf,
                        "centroid": (cx, cy),
                        "source": "yolo_tracked",
                        "needs_vlm_verification": needs_vlm,
                    })

        # Merge all enriched detections (animals, obstacles, auto, roboflow) from raw_detections
        if raw_detections:
            enriched = [
                d for d in raw_detections
                if d.get("source") in (
                    "yolo_auto", "roboflow", "yolo_coco_animal",
                    "yolo_coco_animal_corrected", "yolo_coco_obstacle"
                )
            ]
            for det in enriched:
                # Find if any tracked object overlaps significantly
                best_match = self._find_best_overlap(det["bbox"], tracked_objects)
                if best_match is not None:
                    # If raw detection has specific animal / obstacle class that improves tracking
                    if det["class"] in ("cow", "auto-rickshaw", "road-obstacle", "dog") and best_match["class"] in ("person", "bird", "car", "unknown"):
                        best_match["class"] = det["class"]
                        best_match["class_raw"] = det["class"]
                        best_match["confidence"] = max(best_match["confidence"], det["confidence"])
                        self.class_history[best_match["track_id"]].append(det["class"])
                    continue

                # Assign a synthetic track ID (high range to avoid collision)
                synthetic_id = 10000 + hash(
                    f"{det['class']}_{det['bbox'][0]:.0f}_{det['bbox'][1]:.0f}"
                ) % 9000

                cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                cy = (det["bbox"][1] + det["bbox"][3]) / 2.0

                self.centroid_history[synthetic_id].append((cx, cy))
                self.class_history[synthetic_id].append(det["class"])
                self.bbox_history[synthetic_id].append(det["bbox"])

                tracked_objects.append({
                    "track_id": synthetic_id,
                    "bbox": det["bbox"],
                    "class": det["class"],
                    "class_raw": det["class"],
                    "confidence": det["confidence"],
                    "centroid": (cx, cy),
                    "source": det["source"],
                    "needs_vlm_verification": det.get("needs_vlm_verification", False),
                })

        return tracked_objects


    def inject_vlm_result(self, track_id: int, vlm_class: str):
        """
        Inject a VLM verification result into the class history.
        This contributes to the majority vote but never overwrites
        the track's confident history outright.
        """
        if track_id in self.class_history:
            self.class_history[track_id].append(vlm_class)
            # Trim to window
            if len(self.class_history[track_id]) > CLASS_VOTE_WINDOW:
                self.class_history[track_id] = (
                    self.class_history[track_id][-CLASS_VOTE_WINDOW:]
                )

    def get_centroid_history(self, track_id: int) -> list[tuple[float, float]]:
        """Get the centroid history for a given track ID."""
        return self.centroid_history.get(track_id, [])

    def get_bbox_history(self, track_id: int) -> list[list[float]]:
        """Get bounding box history for a given track ID."""
        return self.bbox_history.get(track_id, [])

    def get_majority_class(self, track_id: int) -> str:
        """
        Get the most frequently assigned class for a track (majority voting).
        Section 3.1: a track that reads "cow" nine times and "motorcycle"
        once should display as "cow", full stop.
        """
        history = self.class_history.get(track_id, [])
        if not history:
            return "unknown"
        return Counter(history).most_common(1)[0][0]

    def get_active_track_ids(self) -> set[int]:
        """Get all currently tracked IDs that have recent centroid entries."""
        return set(self.centroid_history.keys())

    def _get_model(self):
        """Lazy-load the tracking model."""
        if self._model is None:
            from ultralytics import YOLO
            model_path = MODEL_DIR / YOLO_MODEL
            if not model_path.exists():
                self._model = YOLO(YOLO_MODEL)
            else:
                self._model = YOLO(str(model_path))
        return self._model

    def _find_best_overlap(
        self, bbox: list[float], tracked: list[dict]
    ) -> dict | None:
        """Find a tracked object with significant IoU overlap."""
        best_iou = 0.0
        best_match = None
        for t in tracked:
            iou = self._iou(bbox, t["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = t
        return best_match if best_iou > 0.4 else None

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
