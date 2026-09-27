/**
 * SARTHI Vision - TypeScript Type Definitions
 * Mirrors the backend Pydantic schemas exactly.
 */

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectedObject {
  track_id: number;
  class_name: string;
  class_raw: string;
  confidence: number;
  bbox: BoundingBox;
  status: "Moving" | "Stationary";
  distance_m: number | null;
  distance_available: boolean;
  distance_method: "pinhole" | "depth_model" | "pinhole_fallback" | "unavailable";
  risk_level: "Low" | "Moderate" | "High";
  risk_score: number;
  risk_explanation: string;
  rel_dx: number;
  rel_dy: number;
  closing_speed: number;
  ttc_s: number | null;
}

export interface CollisionWarning {
  track_a: number;
  track_b: number;
  class_a: string;
  class_b: string;
  ttc_s: number;
  current_distance_m: number;
  urgency: "imminent" | "high" | "moderate";
}

export interface VoiceAlert {
  urgency: "imminent" | "high";
  fact: Record<string, unknown>;
  phrase: string;
  timestamp_s: number;
  audio_available: boolean;
}

export interface NearMissEvent {
  timestamp_s: number;
  frame_index: number;
  track_id?: number;
  track_a?: number;
  track_b?: number;
  ttc_s: number;
  risk_level: string;
  class_name: string;
  event_type: "ego_object" | "object_object";
}

// v3: Audio detection event from YAMNet
export interface AudioEvent {
  start_s: number;
  end_s: number;
  confidence: number;
  class_name: string;
}

// v3: Per-signal status for System Status strip
export interface SignalStatus {
  status: "active" | "degraded" | "absent" | "clear";
  label: string;
  detail: string;
}

// v3: System Status showing which signals are active
export interface SystemStatus {
  camera: SignalStatus;
  depth: SignalStatus;
  audio: SignalStatus;
}

// v3: Pipeline timing measurements
export interface PipelineTimings {
  per_stage_avg_ms: Record<string, number>;
  per_stage_total_ms: Record<string, number>;
  per_frame_avg_ms: number;
  per_frame_min_ms: number;
  per_frame_max_ms: number;
  total_frames_timed: number;
  total_pipeline_ms: number;
}

export interface FrameResult {
  frame_index: number;
  timestamp_s: number;
  objects: DetectedObject[];
  chaos_score: number;
  object_count: number;
  frame_width: number;
  frame_height: number;
  collision_warnings: CollisionWarning[];
  voice_alerts: VoiceAlert[];
  low_light_enhanced: boolean;
  mean_brightness: number;
  // v3: Audio state at this frame
  siren_active: boolean;
  siren_confidence: number;
  siren_class: string;
  // v3: Per-frame processing time
  frame_processing_ms: number;
}

export interface AnalysisResponse {
  success: boolean;
  media_type: "image" | "video";
  total_frames: number;
  frame_width: number;
  frame_height: number;
  fps: number;
  sample_fps: number;
  duration_s: number;
  duration_capped: boolean;
  frames: FrameResult[];
  aggregate_chaos_score: number;
  unique_classes: string[];
  total_unique_tracks: number;
  processing_time_s: number;
  near_miss_events: NearMissEvent[];
  vlm_verifications: number;
  frames_enhanced: number;
  // v3 additions
  audio_events: AudioEvent[];
  has_audio_track: boolean;
  system_status: SystemStatus;
  pipeline_timings: PipelineTimings;
}

export interface ErrorResponse {
  success: false;
  error: string;
  error_type:
    | "unsupported_type"
    | "file_too_large"
    | "corrupt_file"
    | "no_detections"
    | "processing_error";
}

export type ApiResponse = AnalysisResponse | ErrorResponse;

// Risk level color mapping
export const RISK_COLORS: Record<string, string> = {
  High: "#EF4444",
  Moderate: "#F59E0B",
  Low: "#10B981",
};

// Risk level border colors (slightly transparent for overlays)
export const RISK_BORDER_COLORS: Record<string, string> = {
  High: "rgba(239, 68, 68, 0.95)",
  Moderate: "rgba(245, 158, 11, 0.95)",
  Low: "rgba(16, 185, 129, 0.95)",
};

// Risk level background colors (for label cards)
export const RISK_BG_COLORS: Record<string, string> = {
  High: "rgba(239, 68, 68, 0.16)",
  Moderate: "rgba(245, 158, 11, 0.16)",
  Low: "rgba(16, 185, 129, 0.16)",
};

// System status signal colors
export const SIGNAL_COLORS: Record<string, string> = {
  active: "#10B981",   // green
  degraded: "#F59E0B", // amber
  absent: "#6B7280",   // gray
  clear: "#10B981",    // green (confirmed clear)
};

// Theme tokens for Warm Ivory & Cherry
export const THEME = {
  cherry: "#670626",
  cherryGlow: "#A31243",
  ivory: "#F7F1EA",
  ivoryCream: "#FFFDF9",
  background: "#0C0306",
};
