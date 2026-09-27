"""
SARTHI Vision - Pydantic Response Schemas
Typed models for the API response payload. Every field that appears in the
frontend comes from here, enforcing the no-fabrication rule at the type level.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Pixel-space bounding box [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedObject(BaseModel):
    """A single detected obstacle in a single frame."""
    track_id: int = Field(..., description="Persistent tracking ID across frames")
    class_name: str = Field(..., description="e.g. 'Auto-Rickshaw', 'Bovine (Cow)'")
    class_raw: str = Field(..., description="Raw class label from model, e.g. 'cow'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    status: str = Field(..., description="'Moving' or 'Stationary'")
    distance_m: Optional[float] = Field(None, description="Estimated distance in meters")
    distance_available: bool = Field(True, description="False if distance could not be computed")
    distance_method: str = Field(
        "unavailable",
        description="'pinhole', 'depth_model', 'pinhole_fallback', or 'unavailable'"
    )
    risk_level: str = Field(..., description="'Low', 'Moderate', or 'High'")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Raw risk score 0-1")
    risk_explanation: str = Field(
        ...,
        description="Plain-language explanation of why this risk score was assigned"
    )
    # Motion vector (relative, after ego-motion subtraction)
    rel_dx: float = Field(0.0, description="Relative horizontal displacement")
    rel_dy: float = Field(0.0, description="Relative vertical displacement")
    closing_speed: float = Field(0.0, description="Rate of distance decrease (m/frame)")
    # Time-to-Collision (Section 4)
    ttc_s: Optional[float] = Field(None, description="Time to collision in seconds (None if not closing)")


class CollisionWarning(BaseModel):
    """A pairwise collision warning between two tracked objects."""
    track_a: int
    track_b: int
    class_a: str
    class_b: str
    ttc_s: float = Field(..., description="Estimated time to collision between the pair")
    current_distance_m: float = Field(..., description="Current estimated distance between objects")
    urgency: str = Field(..., description="'imminent', 'high', or 'moderate'")


class VoiceAlert(BaseModel):
    """A voice alert generated from a deterministic fact."""
    urgency: str = Field(..., description="'imminent' or 'high'")
    fact: dict = Field(..., description="Structured fact that generated this alert")
    phrase: str = Field(..., description="Natural language phrase for TTS")
    timestamp_s: float = Field(0.0, description="Timestamp in seconds when alert was generated")
    audio_available: bool = Field(False, description="True if audio was synthesized")


class NearMissEvent(BaseModel):
    """A near-miss event logged when TTC dropped below danger threshold."""
    timestamp_s: float
    frame_index: int
    track_id: Optional[int] = None
    track_a: Optional[int] = None
    track_b: Optional[int] = None
    ttc_s: float
    risk_level: str
    class_name: str = ""
    event_type: str = Field("ego_object", description="'ego_object' or 'object_object'")


class AudioEvent(BaseModel):
    """A siren/emergency vehicle audio detection event from YAMNet."""
    start_s: float = Field(..., description="Start time of the siren event in seconds")
    end_s: float = Field(..., description="End time of the siren event in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Peak detection confidence")
    class_name: str = Field(..., description="YAMNet class name (e.g. 'Siren', 'Ambulance (siren)')")


class SignalStatus(BaseModel):
    """Status of a single sensor/signal for the System Status strip."""
    status: str = Field(
        ...,
        description="'active', 'degraded', 'absent', or 'clear'"
    )
    label: str = Field(..., description="Human-readable status label")
    detail: str = Field("", description="Additional detail about the status")


class SystemStatus(BaseModel):
    """
    System Status strip: shows which signals are currently active.
    Section 3 of v3 upgrade: the demonstrable version of sensor redundancy.
    """
    camera: SignalStatus = Field(
        default_factory=lambda: SignalStatus(
            status="active", label="Camera active", detail=""
        )
    )
    depth: SignalStatus = Field(
        default_factory=lambda: SignalStatus(
            status="active", label="Depth active", detail=""
        )
    )
    audio: SignalStatus = Field(
        default_factory=lambda: SignalStatus(
            status="absent", label="No audio input", detail=""
        )
    )


class PipelineTimings(BaseModel):
    """
    Per-stage wall-clock timing measurements.
    Section 4 of v3 upgrade: honest, measured latency numbers.
    """
    per_stage_avg_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Average time per stage in milliseconds"
    )
    per_stage_total_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Total time per stage across all frames in milliseconds"
    )
    per_frame_avg_ms: float = Field(0.0, description="Average per-frame processing time in ms")
    per_frame_min_ms: float = Field(0.0, description="Minimum per-frame time in ms")
    per_frame_max_ms: float = Field(0.0, description="Maximum per-frame time in ms")
    total_frames_timed: int = Field(0, description="Number of frames timed")
    total_pipeline_ms: float = Field(0.0, description="Total pipeline wall-clock time in ms")


class FrameResult(BaseModel):
    """Detection results for a single frame."""
    frame_index: int
    timestamp_s: float = Field(..., description="Timestamp in seconds within the video")
    objects: list[DetectedObject] = []
    chaos_score: float = Field(0.0, ge=0.0, le=100.0, description="Scene Chaos Score 0-100")
    object_count: int = 0
    frame_width: int = 0
    frame_height: int = 0
    collision_warnings: list[CollisionWarning] = Field(
        default_factory=list,
        description="Pairwise collision warnings for this frame"
    )
    voice_alerts: list[VoiceAlert] = Field(
        default_factory=list,
        description="Voice alerts generated for this frame"
    )
    low_light_enhanced: bool = Field(False, description="True if CLAHE was applied to this frame")
    mean_brightness: float = Field(0.0, description="Mean frame brightness (0-255)")
    # Audio detection state for this frame's timestamp
    siren_active: bool = Field(False, description="True if a siren is detected at this frame's timestamp")
    siren_confidence: float = Field(0.0, description="Siren detection confidence at this timestamp")
    siren_class: str = Field("", description="Detected siren class name (if active)")
    # Per-frame processing time
    frame_processing_ms: float = Field(0.0, description="Wall-clock processing time for this frame in ms")


class AnalysisResponse(BaseModel):
    """Complete response from POST /api/analyze."""
    success: bool = True
    media_type: str = Field(..., description="'image' or 'video'")
    total_frames: int
    frame_width: int
    frame_height: int
    fps: float = Field(..., description="Original video FPS (or 0 for images)")
    sample_fps: float = Field(..., description="FPS at which frames were sampled")
    duration_s: float = Field(0.0, description="Video duration in seconds")
    duration_capped: bool = Field(
        False, description="True if the video was longer than the cap and was truncated"
    )
    frames: list[FrameResult]
    aggregate_chaos_score: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Average Chaos Score across all processed frames"
    )
    unique_classes: list[str] = Field(
        default_factory=list,
        description="List of distinct obstacle classes detected"
    )
    total_unique_tracks: int = Field(0, description="Total distinct objects tracked")
    processing_time_s: float = Field(0.0, description="Wall-clock processing time")
    near_miss_events: list[NearMissEvent] = Field(
        default_factory=list,
        description="Timeline of near-miss events across the clip"
    )
    vlm_verifications: int = Field(0, description="Number of VLM verification calls made")
    frames_enhanced: int = Field(0, description="Number of frames that received CLAHE enhancement")
    # v3 additions
    audio_events: list[AudioEvent] = Field(
        default_factory=list,
        description="Siren/emergency vehicle audio detection events"
    )
    has_audio_track: bool = Field(
        False,
        description="True if the upload contained an audio track"
    )
    system_status: SystemStatus = Field(
        default_factory=SystemStatus,
        description="Per-signal sensor status for the System Status strip"
    )
    pipeline_timings: PipelineTimings = Field(
        default_factory=PipelineTimings,
        description="Per-stage wall-clock timing measurements"
    )


class ErrorResponse(BaseModel):
    """Error response for failed analysis."""
    success: bool = False
    error: str
    error_type: str = Field(
        ..., description="'unsupported_type', 'file_too_large', 'corrupt_file', "
                         "'no_detections', 'processing_error'"
    )


# --- Display name mapping for UI-friendly labels ---
DISPLAY_NAMES = {
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "dog": "Canine (Dog)",
    "cow": "Bovine (Cow)",
    "goat": "Caprine (Goat)",
    "cat": "Feline (Cat)",
    "horse": "Equine (Horse)",
    "sheep": "Ovine (Sheep)",
    "elephant": "Elephant",
    "bear": "Bear",
    "bird": "Bird",
    "auto-rickshaw": "Auto-Rickshaw",
    "pothole": "Pothole",
    "road-obstacle": "Road Obstacle",
}


def get_display_name(raw_class: str) -> str:
    """Get a UI-friendly display name for a raw class label."""
    return DISPLAY_NAMES.get(raw_class, raw_class.replace("-", " ").title())
