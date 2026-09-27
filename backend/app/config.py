"""
SARTHI Vision - Configuration
Centralized configuration constants. All tunable values are exposed here,
not buried in logic files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent.resolve()
MODEL_DIR = (BASE_DIR / "app" / "models").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Model Configuration ---
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11s.pt")
YOLO_AUTO_MODEL = os.getenv("YOLO_AUTO_MODEL", "")  # Fine-tuned auto-rickshaw model
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

# --- Animal & Vulnerable Road Hazard Sensitivity ---
# Indian road hazards (cows, stray cattle, dogs, etc.) frequently blend with dark asphalt at night
ANIMAL_CLASSES = {"cow", "dog", "horse", "sheep", "goat", "cat", "elephant", "bear"}
ANIMAL_CONFIDENCE_RATIO = float(os.getenv("ANIMAL_CONFIDENCE_RATIO", "0.45"))
ANIMAL_MIN_CONFIDENCE = float(os.getenv("ANIMAL_MIN_CONFIDENCE", "0.08"))


# --- Roboflow ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_POTHOLE_PROJECT = "pothole-detection-using-yolov8"
ROBOFLOW_POTHOLE_VERSION = 1

# --- OpenAI (VLM verification + voice alerts) ---
# Key lives in .env only, gitignored, loaded via python-dotenv.
# Never hardcoded, never logged, never displayed in the UI.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_VLM_MODEL = os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")

# --- Processing Limits ---
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_VIDEO_DURATION_S = int(os.getenv("MAX_VIDEO_DURATION_S", "45"))
FRAME_SAMPLE_RATE = int(os.getenv("FRAME_SAMPLE_RATE", "6"))  # fps

# --- Camera / Distance Estimation ---
# Default focal length in pixels for a typical dashcam (wide-angle ~120 deg FOV)
# This is a configurable constant, not a hardcoded magic number.
DEFAULT_FOCAL_LENGTH_PX = float(os.getenv("DEFAULT_FOCAL_LENGTH_PX", "800"))

# Real-world widths in meters for known object classes.
# Used in the pinhole distance estimation formula:
#   distance_m = (real_world_width_m * focal_length_px) / bbox_width_px
REAL_WORLD_WIDTHS_M = {
    "car": 1.8,
    "bus": 2.5,
    "truck": 2.4,
    "motorcycle": 0.8,
    "bicycle": 0.6,
    "person": 0.5,
    "auto-rickshaw": 1.4,
    "cow": 0.6,
    "dog": 0.35,
    "goat": 0.35,
    "horse": 0.7,
    "sheep": 0.4,
    "elephant": 2.5,
    "bear": 0.8,
    "cat": 0.3,
}

# --- Motion Classification ---
# Number of frames to look back for displacement calculation
MOTION_LOOKBACK_FRAMES = int(os.getenv("MOTION_LOOKBACK_FRAMES", "8"))

# Pixel displacement threshold for Moving vs Stationary classification
# (after ego-motion subtraction)
MOTION_PIXEL_THRESHOLD = float(os.getenv("MOTION_PIXEL_THRESHOLD", "5.0"))

# Minimum number of tracked objects to use median-based ego-motion estimation.
# Below this, fall back to optical flow.
MIN_OBJECTS_FOR_MEDIAN = 3

# --- Low-Light Enhancement ---
# Mean frame brightness below this threshold triggers CLAHE enhancement.
# Tuned for 0-255 scale. Start around 60-80, tune against dark clips.
LOW_LIGHT_THRESHOLD = int(os.getenv("LOW_LIGHT_THRESHOLD", "70"))

# --- VLM Verification ---
# Only fire VLM verification when detection confidence is below this threshold
# AND the raw class disagrees with the track's majority class.
VLM_CONFIDENCE_THRESHOLD = float(os.getenv("VLM_CONFIDENCE_THRESHOLD", "0.5"))
# Rolling window size for track-level majority voting (frames)
CLASS_VOTE_WINDOW = int(os.getenv("CLASS_VOTE_WINDOW", "15"))

# --- Time-to-Collision (Predictive Prior Horizon) ---
# TTC below this threshold (seconds) triggers imminent collision warning prior to impact.
TTC_DANGER_THRESHOLD = float(os.getenv("TTC_DANGER_THRESHOLD", "4.0"))
# Only compute pairwise TTC for objects within this distance of each other.
PAIRWISE_PROXIMITY_THRESHOLD = float(os.getenv("PAIRWISE_PROXIMITY_THRESHOLD", "25.0"))

# --- Audio Detection (YAMNet Siren) ---
# Confidence threshold for YAMNet siren class to trigger an event.
# Tuned conservatively to avoid false positives in noisy urban audio.
SIREN_CONFIDENCE_THRESHOLD = float(os.getenv("SIREN_CONFIDENCE_THRESHOLD", "0.25"))
# How much a siren detection boosts the Scene Chaos Score (additive, 0-30 range)
SIREN_CHAOS_BOOST = float(os.getenv("SIREN_CHAOS_BOOST", "20.0"))

# --- Risk Scoring ---
# Weights for the risk score formula:
#   risk = w1*class_weight + w2*proximity + w3*motion + w4*closing_speed + w5*ttc
RISK_W1_CLASS = 0.20
RISK_W2_PROXIMITY = 0.25
RISK_W3_MOTION = 0.20
RISK_W4_CLOSING = 0.15
RISK_W5_TTC = 0.20

# Class risk weights: higher = more inherently risky (unpredictable motion)
CLASS_RISK_WEIGHTS = {
    "person": 0.9,
    "cow": 0.95,
    "dog": 0.85,
    "goat": 0.85,
    "horse": 0.8,
    "sheep": 0.75,
    "elephant": 0.95,
    "bear": 0.95,
    "cat": 0.7,
    "auto-rickshaw": 0.6,
    "motorcycle": 0.65,
    "bicycle": 0.7,
    "car": 0.4,
    "bus": 0.35,
    "truck": 0.35,
    "pothole": 0.5,
    "road-obstacle": 0.55,
}

# Risk level thresholds (score 0-1 mapped to Low/Moderate/High)
RISK_THRESHOLD_HIGH = 0.65
RISK_THRESHOLD_MODERATE = 0.35

# --- Chaos Score ---
# Weights for Scene Chaos Score components (0-100 composite)
CHAOS_W_COUNT = 0.20        # Object count relative to frame area
CHAOS_W_ENTROPY = 0.20      # Shannon entropy of class mix
CHAOS_W_PROXIMITY = 0.25    # Average proximity (closer = more chaos)
CHAOS_W_HIGH_RISK = 0.15    # Share of high-risk objects
CHAOS_W_TTC = 0.20          # Any sub-threshold TTC elevates chaos

# --- COCO Class ID to Label Mapping (subset we care about) ---
# Correct COCO/YOLO class IDs verified against ultralytics model.names
COCO_CLASS_MAP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",       # Rare but possible on roads
    15: "cat",
    16: "dog",
    17: "horse",      # Common on Indian rural roads
    18: "sheep",      # Common on Indian rural roads
    19: "cow",        # High-priority animal hazard
    20: "elephant",   # Rare but critical (North/South India)
    21: "bear",       # Rare but critical (hill areas)
}

# Classes that COCO YOLO can detect out of the box
COCO_SUPPORTED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "cow", "cat", "horse", "sheep", "elephant", "bear",
}

# All classes we aim to detect (COCO + fine-tuned + API)
ALL_TARGET_CLASSES = COCO_SUPPORTED_CLASSES | {"auto-rickshaw", "pothole", "goat", "road-obstacle"}

# Fixed class list for VLM verification prompts
VLM_CLASS_LIST = sorted(ALL_TARGET_CLASSES)

# --- Supported file types ---
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_FILE_TYPES = SUPPORTED_IMAGE_TYPES | SUPPORTED_VIDEO_TYPES
