"""
SARTHI Vision - VLM Verification (GPT-4o-mini)

Section 3.2 of the upgrade prompt: escalate genuine ambiguity to the VLM, sparingly.

Rules (non-negotiable):
1. The VLM may VERIFY, it may NEVER ORIGINATE.
   It is only called on a crop that YOLO already produced and already flagged
   as low-confidence or as an inconsistent track.
2. It picks among a fixed class list you provide.
3. It never runs on every frame, never proposes a detection the base pipeline
   didn't already suggest, and never overrides a high-confidence result.
4. Rate-limited: at most 1 VLM call per track per 5 frames.
5. Uses gpt-4o-mini (not full gpt-4o) to stay credit-conscious.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import cv2
import numpy as np

from app.config import (
    OPENAI_API_KEY,
    OPENAI_VLM_MODEL,
    VLM_CLASS_LIST,
)

logger = logging.getLogger(__name__)

# Rate-limit tracking: {track_id: last_frame_verified}
_vlm_last_verified: dict[int, int] = {}
_VLM_COOLDOWN_FRAMES = 15  # Min frames between VLM calls for the same track
_MAX_VLM_CALLS_PER_VIDEO = 4
_vlm_call_count = 0

# Lazy-loaded OpenAI client
_openai_client = None


def _get_openai_client():
    """Lazy-load OpenAI client. Returns None if API key not configured."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if not OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set, VLM verification disabled")
        return None

    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized for VLM verification (model: %s)", OPENAI_VLM_MODEL)
        return _openai_client
    except Exception as e:
        logger.warning("Failed to initialize OpenAI client: %s", e)
        return None


def reset_vlm_state():
    """Reset rate-limit tracking for a new video."""
    global _vlm_last_verified, _vlm_call_count
    _vlm_last_verified.clear()
    _vlm_call_count = 0


def should_verify(
    track_id: int,
    raw_class: str,
    majority_class: str,
    confidence: float,
    confidence_threshold: float,
    current_frame_idx: int,
    explicit_flag: bool = False,
) -> bool:
    """
    Determine if a specific detection warrants VLM verification.

    Conditions (ALL must be true):
    1. OpenAI API key is configured
    2. Global VLM call budget per video has not been exceeded
    3. Rate-limit cooldown has passed for this track
    4. Either explicit_flag is set OR raw class disagrees with majority class at low confidence
    """
    global _vlm_call_count
    if not OPENAI_API_KEY or _vlm_call_count >= _MAX_VLM_CALLS_PER_VIDEO:
        return False

    # Rate-limit: check cooldown
    last_verified = _vlm_last_verified.get(track_id, -999)
    if current_frame_idx - last_verified < _VLM_COOLDOWN_FRAMES:
        return False

    if explicit_flag or raw_class in ("road-obstacle", "bird"):
        return True

    # Only verify when raw class disagrees with majority
    if raw_class == majority_class:
        return False

    # Only verify low-confidence detections
    if confidence >= confidence_threshold:
        return False

    return True



def verify_crop_with_vlm(
    frame: np.ndarray,
    bbox: list[float],
    track_id: int,
    current_frame_idx: int,
) -> str | None:
    """
    Send a cropped detection region to GPT-4o-mini for class verification.

    Args:
        frame: Full BGR frame
        bbox: [x1, y1, x2, y2] bounding box in pixel coordinates
        track_id: Track ID (for rate-limiting)
        current_frame_idx: Current frame index (for rate-limiting)

    Returns:
        Verified class name from VLM_CLASS_LIST, or None if verification failed.
        This result is used ONLY to contribute to the majority vote, never to
        override the track's confident history.
    """
    client = _get_openai_client()
    if client is None:
        return None

    # Update rate-limit tracker
    global _vlm_call_count
    _vlm_call_count += 1
    _vlm_last_verified[track_id] = current_frame_idx

    try:
        # Crop the bounding box region
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        # Encode crop as base64 JPEG
        _, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # Constrained prompt: pick exactly one class from the fixed list
        class_list_str = ", ".join(VLM_CLASS_LIST)
        prompt = (
            f"Which of these classes best matches the object in this cropped image: "
            f"[{class_list_str}]? "
            f"Respond with exactly one class name from the list and nothing else."
        )

        response = client.chat.completions.create(
            model=OPENAI_VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "low",  # low detail to minimize tokens/cost
                            },
                        },
                    ],
                }
            ],
            max_tokens=20,
            temperature=0,
        )

        vlm_answer = response.choices[0].message.content.strip().lower()

        # Validate: answer must be in our fixed class list
        if vlm_answer in [c.lower() for c in VLM_CLASS_LIST]:
            logger.info(
                "VLM verified track %d as '%s' (frame %d)",
                track_id,
                vlm_answer,
                current_frame_idx,
            )
            return vlm_answer
        else:
            # VLM returned something outside our list, discard
            logger.debug(
                "VLM returned '%s' which is not in class list, ignoring",
                vlm_answer,
            )
            return None

    except Exception as e:
        logger.warning("VLM verification failed for track %d: %s", track_id, e)
        return None
