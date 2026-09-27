"""
SARTHI Vision - Adaptive Low-Light Enhancement

Section 2 of the upgrade prompt: fix night-time misclassification.

Before detection, on every sampled frame:
1. Compute mean frame brightness (grayscale mean pixel value).
2. If brightness < LOW_LIGHT_THRESHOLD: apply CLAHE on the L-channel of LAB.
3. Bright frames skip entirely (zero added latency).

CLAHE (Contrast Limited Adaptive Histogram Equalization) redistributes
local contrast in the luminance channel without distorting color, making
dark regions readable for the detector while keeping bright regions stable.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.config import LOW_LIGHT_THRESHOLD

logger = logging.getLogger(__name__)

# Reusable CLAHE instance (thread-safe for single-process uvicorn)
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def enhance_frame(
    frame: np.ndarray,
    brightness_threshold: int = LOW_LIGHT_THRESHOLD,
) -> tuple[np.ndarray, bool, float]:
    """
    Conditionally enhance a frame if it falls below the brightness threshold.

    Args:
        frame: BGR image (numpy array from cv2)
        brightness_threshold: mean brightness below which CLAHE is applied (0-255)

    Returns:
        (enhanced_frame, was_enhanced, mean_brightness)
        - enhanced_frame: the (possibly enhanced) BGR frame
        - was_enhanced: True if CLAHE was applied
        - mean_brightness: the measured mean brightness (0-255)
    """
    # Step 1: Compute mean brightness quickly via grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    # Step 2: Skip if frame is well-lit
    if mean_brightness >= brightness_threshold:
        return frame, False, mean_brightness

    # Step 3: Apply CLAHE on the L-channel of LAB color space
    # LAB separates luminance from color, so we can boost contrast
    # without shifting hues (which would confuse the detector).
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to the luminance channel only
        l_enhanced = _clahe.apply(l_channel)

        # Merge back and convert to BGR
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        logger.debug(
            "Low-light enhancement applied: brightness=%.1f (threshold=%d)",
            mean_brightness,
            brightness_threshold,
        )
        return enhanced, True, mean_brightness

    except Exception as e:
        logger.warning("CLAHE enhancement failed, returning original frame: %s", e)
        return frame, False, mean_brightness
