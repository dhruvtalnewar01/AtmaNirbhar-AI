"""
SARTHI Vision - Audio-Based Emergency Vehicle Detection (YAMNet)

Section 2 of upgrade prompt v3: detect sirens in video audio tracks.

Architecture:
  - YAMNet: pretrained MobileNetV1-based classifier for 521 AudioSet classes
  - Runs on audio track ONLY, completely independent of video frame sampling
  - Siren events feed into Scene Chaos Score as an additive contributor
  - Audio events do NOT get forced into the per-object TTC pipeline
  - No-audio case (image upload): "no_audio_input", never a false negative

Audio-visual correlation (Section 2.5 of v3):
  When a siren is detected AND a visually tracked object matches an emergency
  vehicle class (ambulance, fire truck, police car), the two signals are
  correlated. The visual object's own tracked distance and TTC take over at
  that point. Before correlation, siren detection is directionally unknown.
"""

from __future__ import annotations

import logging
import tempfile
import os
from typing import Any

import numpy as np

from app.config import (
    SIREN_CONFIDENCE_THRESHOLD,
    SIREN_CHAOS_BOOST,
)

logger = logging.getLogger(__name__)

# Lazy-loaded YAMNet model and class map
_yamnet_model = None
_yamnet_class_names: list[str] = []

# Emergency-vehicle-relevant class indices (populated at load time from YAMNet's own class map)
_siren_class_indices: list[int] = []

# Siren-relevant class name substrings to filter from YAMNet's 521 classes
_SIREN_KEYWORDS = [
    "siren",
    "emergency vehicle",
    "ambulance",
    "police car",
    "fire engine",
    "fire truck",
]


def _load_yamnet():
    """Lazy-load YAMNet model from TensorFlow Hub."""
    global _yamnet_model, _yamnet_class_names, _siren_class_indices

    if _yamnet_model is not None:
        return

    try:
        import tensorflow_hub as hub
        import csv
        import io

        logger.info("Loading YAMNet model from TensorFlow Hub...")
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

        # Load the class map from the model itself (not hardcoded)
        class_map_path = _yamnet_model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, "r") as f:
            reader = csv.DictReader(f)
            _yamnet_class_names = [row["display_name"] for row in reader]

        # Enumerate siren/emergency-relevant classes at load time
        _siren_class_indices = []
        for idx, name in enumerate(_yamnet_class_names):
            name_lower = name.lower()
            if any(kw in name_lower for kw in _SIREN_KEYWORDS):
                _siren_class_indices.append(idx)
                logger.info("YAMNet siren class [%d]: %s", idx, name)

        if not _siren_class_indices:
            logger.warning(
                "No siren-relevant classes found in YAMNet class map. "
                "Audio detection will not produce siren events."
            )

        logger.info(
            "YAMNet loaded: %d total classes, %d siren-relevant",
            len(_yamnet_class_names),
            len(_siren_class_indices),
        )

    except Exception as e:
        logger.warning("Failed to load YAMNet: %s", e)
        _yamnet_model = None


def extract_audio_from_video(video_path: str) -> tuple[np.ndarray | None, bool]:
    """
    Extract audio track from a video file, resample to 16kHz mono float32.

    Returns:
        (waveform, has_audio)
        - waveform: numpy array of float32 samples at 16kHz, or None
        - has_audio: True if the video had an audio track (even if silent)
    """
    try:
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip
        import scipy.signal

        clip = VideoFileClip(video_path)

        if clip.audio is None:
            clip.close()
            return None, False

        # Extract audio as numpy array
        # fps=16000 resamples to YAMNet's native 16kHz
        audio_array = clip.audio.to_soundarray(fps=16000)
        clip.close()

        if audio_array is None or len(audio_array) == 0:
            return None, False

        # Convert to mono if stereo
        if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
            waveform = np.mean(audio_array, axis=1)
        else:
            waveform = audio_array.flatten()

        # Convert to float32 in [-1, 1] range
        waveform = waveform.astype(np.float32)
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        return waveform, True

    except Exception as e:
        logger.warning("Audio extraction failed: %s", e)
        return None, False


def detect_sirens(
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> list[dict[str, Any]]:
    """
    Run YAMNet siren detection on an audio waveform.

    Args:
        waveform: mono float32 audio at 16kHz
        sample_rate: sample rate (should be 16000)

    Returns:
        List of siren detection events:
        [
            {
                "start_s": 1.5,
                "end_s": 2.5,
                "confidence": 0.82,
                "class_name": "Siren",
                "window_index": 3,
            },
            ...
        ]
    """
    _load_yamnet()

    if _yamnet_model is None:
        logger.warning("YAMNet not loaded; skipping siren detection")
        return []

    if not _siren_class_indices:
        return []

    try:
        import tensorflow as tf

        # Ensure waveform is float32
        waveform_tf = tf.cast(waveform, tf.float32)

        # Run YAMNet inference
        # Returns: scores (time_windows x 521), embeddings, spectrogram
        scores, embeddings, spectrogram = _yamnet_model(waveform_tf)
        scores_np = scores.numpy()  # shape: (num_windows, 521)

        siren_events = []
        num_windows = scores_np.shape[0]

        # YAMNet uses ~0.96s windows with ~0.48s hop
        window_duration = 0.96
        hop_duration = 0.48

        for window_idx in range(num_windows):
            window_scores = scores_np[window_idx]

            # Check all siren-relevant classes for this window
            for class_idx in _siren_class_indices:
                confidence = float(window_scores[class_idx])

                if confidence >= SIREN_CONFIDENCE_THRESHOLD:
                    start_s = window_idx * hop_duration
                    end_s = start_s + window_duration
                    class_name = _yamnet_class_names[class_idx]

                    siren_events.append({
                        "start_s": round(start_s, 2),
                        "end_s": round(end_s, 2),
                        "confidence": round(confidence, 3),
                        "class_name": class_name,
                        "window_index": window_idx,
                    })

        # Merge overlapping/adjacent siren events
        merged = _merge_siren_events(siren_events)

        if merged:
            logger.info(
                "Siren detection: %d events found (from %d raw windows)",
                len(merged),
                len(siren_events),
            )

        return merged

    except Exception as e:
        logger.warning("YAMNet inference failed: %s", e)
        return []


def _merge_siren_events(events: list[dict]) -> list[dict]:
    """
    Merge overlapping or adjacent siren events into contiguous spans.
    Keeps the highest confidence across merged windows.
    """
    if not events:
        return []

    # Sort by start time
    sorted_events = sorted(events, key=lambda e: e["start_s"])
    merged = [sorted_events[0].copy()]

    for event in sorted_events[1:]:
        prev = merged[-1]
        # Merge if overlapping or adjacent (within 1s gap)
        if event["start_s"] <= prev["end_s"] + 1.0:
            prev["end_s"] = max(prev["end_s"], event["end_s"])
            prev["confidence"] = max(prev["confidence"], event["confidence"])
            # Keep the most specific class name
            if len(event["class_name"]) > len(prev["class_name"]):
                prev["class_name"] = event["class_name"]
        else:
            merged.append(event.copy())

    return merged


def is_siren_active_at(
    siren_events: list[dict],
    timestamp_s: float,
) -> tuple[bool, float, str]:
    """
    Check if any siren event is active at a given timestamp.

    Returns:
        (is_active, confidence, class_name)
    """
    for event in siren_events:
        if event["start_s"] <= timestamp_s <= event["end_s"]:
            return True, event["confidence"], event["class_name"]
    return False, 0.0, ""


def get_audio_status(
    has_audio_track: bool,
    siren_events: list[dict],
) -> dict[str, Any]:
    """
    Determine the audio signal status for the System Status strip.

    Returns one of three distinct states (Section 1, constraint 4):
    - "no_audio_input": static image or video without audio track
    - "siren_detected": siren events found above threshold
    - "no_siren_detected": audio was analyzed but no sirens found (confident negative)
    """
    if not has_audio_track:
        return {
            "status": "absent",
            "label": "No audio input",
            "detail": "Upload contains no audio track",
        }

    if siren_events:
        max_conf = max(e["confidence"] for e in siren_events)
        return {
            "status": "active",
            "label": "Siren detected",
            "detail": f"{len(siren_events)} event(s), peak confidence {max_conf:.0%}",
        }

    return {
        "status": "clear",
        "label": "No siren detected",
        "detail": "Audio analyzed, no emergency sirens found",
    }


# Emergency vehicle classes for audio-visual correlation
EMERGENCY_VEHICLE_VISUAL_CLASSES = {
    "bus", "truck",  # ambulance/fire truck often detected as bus/truck by COCO
}
