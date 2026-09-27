"""
SARTHI Vision - FastAPI Backend
Main entrypoint. Handles file upload, orchestrates the detection pipeline,
and returns structured JSON for the frontend canvas renderer.

Data flow:
  Upload file -> save -> sample frames (video) or use single image ->
  enhance (CLAHE if dark) -> detect -> track (majority vote) ->
  VLM verify (if ambiguous) -> motion -> distance -> TTC ->
  collision warnings -> risk -> voice alerts -> audio detection ->
  system status -> pipeline timings -> JSON payload ->
  frontend canvas rendering
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.config import (
    CONFIDENCE_THRESHOLD,
    FRAME_SAMPLE_RATE,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_VIDEO_DURATION_S,
    SUPPORTED_FILE_TYPES,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    UPLOAD_DIR,
    TTC_DANGER_THRESHOLD,
    VLM_CONFIDENCE_THRESHOLD,
)
from app.schemas import (
    AnalysisResponse,
    AudioEvent,
    BoundingBox,
    CollisionWarning,
    DetectedObject,
    ErrorResponse,
    FrameResult,
    NearMissEvent,
    PipelineTimings,
    SignalStatus,
    SystemStatus,
    VoiceAlert,
    get_display_name,
)
from app.pipeline.enhance import enhance_frame
from app.pipeline.detect import detect_frame
from app.pipeline.track import ObjectTracker
from app.pipeline.motion import classify_motion, reset_motion_state
from app.pipeline.distance import estimate_distances, reset_distance_state
from app.pipeline.collision import (
    compute_ttc,
    compute_pairwise_collision_warnings,
    reset_collision_state,
)
from app.pipeline.risk import compute_risk_scores, compute_chaos_score
from app.pipeline.verify import (
    should_verify,
    verify_crop_with_vlm,
    reset_vlm_state,
)
from app.pipeline.voice import generate_voice_alerts, synthesize_speech
from app.pipeline.audio import (
    extract_audio_from_video,
    detect_sirens,
    is_siren_active_at,
    get_audio_status,
)
from app.pipeline.timing import PipelineTimer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AtmaNirbhar AI API",
    description=(
        "India-first autonomous driving Multi-Agents AI Intelligence for unstructured roads. "
        "Detects, tracks, and risk-scores obstacles in dashcam footage with multi-sensor fusion."
    ),
    version="3.0.0",
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend is running."""
    return {"status": "ok", "service": "atmanirbhar-ai-api", "version": "3.0.0"}


@app.post("/api/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.1, le=1.0),
):
    """
    Analyze an uploaded image or video for road obstacles.

    Returns a structured JSON payload with per-frame detections,
    risk scores, TTC, collision warnings, voice alerts, audio siren
    detection, system status, pipeline timings, and Scene Chaos Score.
    """
    start_time = time.time()

    # --- Validate file type ---
    if file.filename is None:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="No filename provided.",
                error_type="unsupported_type",
            ).model_dump(),
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_FILE_TYPES:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=f"Unsupported file type: '{ext}'. Supported types: "
                      f"{', '.join(sorted(SUPPORTED_FILE_TYPES))}",
                error_type="unsupported_type",
            ).model_dump(),
        )

    # --- Save uploaded file ---
    upload_id = str(uuid.uuid4())[:8]
    temp_path = UPLOAD_DIR / f"{upload_id}{ext}"

    try:
        # Check file size while saving (streaming)
        total_size = 0
        with open(temp_path, "wb") as f:
            while chunk := await file.read(8192):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE_BYTES:
                    f.close()
                    temp_path.unlink(missing_ok=True)
                    return JSONResponse(
                        status_code=400,
                        content=ErrorResponse(
                            error=f"File exceeds the {MAX_FILE_SIZE_MB} MB size limit.",
                            error_type="file_too_large",
                        ).model_dump(),
                    )
                f.write(chunk)
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=f"Failed to read uploaded file: {str(e)}",
                error_type="corrupt_file",
            ).model_dump(),
        )

    try:
        if ext in SUPPORTED_IMAGE_TYPES:
            result = _process_image(str(temp_path), confidence)
        else:
            result = _process_video(str(temp_path), confidence)

        result["processing_time_s"] = round(time.time() - start_time, 2)
        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Processing failed")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=f"Processing failed: {str(e)}",
                error_type="processing_error",
            ).model_dump(),
        )
    finally:
        # Clean up uploaded file
        temp_path.unlink(missing_ok=True)


@app.api_route("/api/voice-alert", methods=["GET", "POST"])
def voice_alert_endpoint(text: str = Query(..., min_length=1, max_length=200)):
    """
    Synthesize a voice alert from text using OpenAI TTS-1.
    Accepts both GET and POST requests.
    Runs synchronously in FastAPI worker threadpool so it never blocks the event loop.
    Returns audio/mpeg bytes.
    """
    audio_bytes = synthesize_speech(text)
    if audio_bytes is None:
        return JSONResponse(
            status_code=503,
            content={"error": "TTS synthesis unavailable. Check OPENAI_API_KEY."},
        )
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Content-Disposition": "inline; filename=warning.mp3",
        },
    )


def _process_image(path: str, confidence: float) -> dict:
    """Process a single image through the full pipeline."""
    timer = PipelineTimer()
    timer.start_frame()

    frame = cv2.imread(path)
    if frame is None:
        raise ValueError("Could not read image file. It may be corrupted or in an unsupported format.")

    h, w = frame.shape[:2]

    # Reset state
    reset_motion_state()
    reset_distance_state()
    reset_collision_state()
    reset_vlm_state()
    tracker = ObjectTracker()

    # Enhance (CLAHE if dark)
    with timer.stage("enhance"):
        enhanced_frame, was_enhanced, brightness = enhance_frame(frame)

    # Detect (on enhanced frame)
    with timer.stage("detect"):
        raw_detections = detect_frame(enhanced_frame, confidence=confidence)

    # Track (single frame, so tracking is just assignment)
    with timer.stage("track"):
        tracked = tracker.track_frame(enhanced_frame, raw_detections, confidence=confidence)

    # Motion (single frame: all stationary)
    for obj in tracked:
        obj["status"] = "Stationary"
        obj["rel_dx"] = 0.0
        obj["rel_dy"] = 0.0
        obj["raw_dx"] = 0.0
        obj["raw_dy"] = 0.0

    # Distance
    with timer.stage("distance"):
        estimate_distances(tracked, enhanced_frame, bbox_histories=None)

    # TTC (single frame: no history, all None)
    compute_ttc(tracked, frame_rate=0)

    # Collision warnings (none for single frame)
    collision_warnings = []

    # Risk
    with timer.stage("risk"):
        compute_risk_scores(tracked, w, h)
        chaos = compute_chaos_score(tracked, w, h, siren_active=False)

    # Voice alerts
    voice_alerts_raw = generate_voice_alerts(tracked, collision_warnings, timestamp_s=0)

    timer.end_frame()

    # --- System Status (v3 Section 3) ---
    # Images have no audio track: show "No audio input" explicitly
    camera_status = SignalStatus(
        status="degraded" if was_enhanced else "active",
        label="Low-light enhanced" if was_enhanced else "Camera active",
        detail=f"Brightness: {brightness:.0f}/255" if was_enhanced else "",
    )

    # Determine depth signal status from tracked objects
    depth_status = _compute_depth_status(tracked)

    audio_status_dict = get_audio_status(has_audio_track=False, siren_events=[])
    audio_signal = SignalStatus(**audio_status_dict)

    system_status = SystemStatus(
        camera=camera_status,
        depth=depth_status,
        audio=audio_signal,
    )

    # Build response
    frame_result = _build_frame_result(
        tracked, 0, 0.0, w, h, chaos,
        collision_warnings=collision_warnings,
        voice_alerts=voice_alerts_raw,
        low_light_enhanced=was_enhanced,
        mean_brightness=brightness,
        siren_active=False,
        siren_confidence=0.0,
        siren_class="",
        frame_processing_ms=timer.get_per_frame_ms()[0] if timer.get_per_frame_ms() else 0,
    )

    all_classes = list(set(obj.get("class", "") for obj in tracked))
    all_track_ids = set(obj.get("track_id", 0) for obj in tracked)

    return AnalysisResponse(
        media_type="image",
        total_frames=1,
        frame_width=w,
        frame_height=h,
        fps=0,
        sample_fps=0,
        duration_s=0,
        duration_capped=False,
        frames=[frame_result],
        aggregate_chaos_score=chaos,
        unique_classes=all_classes,
        total_unique_tracks=len(all_track_ids),
        near_miss_events=[],
        vlm_verifications=0,
        frames_enhanced=1 if was_enhanced else 0,
        audio_events=[],
        has_audio_track=False,
        system_status=system_status,
        pipeline_timings=PipelineTimings(**timer.get_timings()),
    ).model_dump()


def _process_video(path: str, confidence: float) -> dict:
    """Process a video through the full pipeline, frame by frame."""
    timer = PipelineTimer()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file. It may be corrupted or in an unsupported format.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames_raw / fps if fps > 0 else 0

    # Duration cap
    duration_capped = False
    if duration > MAX_VIDEO_DURATION_S:
        duration_capped = True
        duration = MAX_VIDEO_DURATION_S

    # Compute which frames to sample
    sample_interval = max(1, int(fps / FRAME_SAMPLE_RATE))
    max_frame = int(duration * fps)

    # Effective sample rate for TTC computation
    effective_sample_fps = fps / sample_interval if sample_interval > 0 else fps

    # --- AUDIO DETECTION (v3 Section 2) ---
    # Run independently of video frame sampling
    with timer.stage("audio"):
        waveform, has_audio_track = extract_audio_from_video(path)
        siren_events: list[dict] = []
        if waveform is not None:
            siren_events = detect_sirens(waveform)

    # Reset pipeline state
    reset_motion_state()
    reset_distance_state()
    reset_collision_state()
    reset_vlm_state()
    tracker = ObjectTracker()

    frame_results = []
    all_track_ids = set()
    all_classes = set()
    chaos_scores = []
    near_miss_events = []
    vlm_count = 0
    frames_enhanced_count = 0
    frame_idx = 0
    sample_idx = 0
    any_depth_degraded = False
    any_depth_unavailable = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= max_frame:
            break

        # Only process sampled frames
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps if fps > 0 else 0
        timer.start_frame()

        # --- ENHANCE (Section 2: CLAHE for low-light) ---
        with timer.stage("enhance"):
            enhanced_frame, was_enhanced, brightness = enhance_frame(frame)
        if was_enhanced:
            frames_enhanced_count += 1

        # --- DETECT (on enhanced frame) ---
        with timer.stage("detect"):
            raw_detections = detect_frame(enhanced_frame, confidence=confidence)

        # --- TRACK (with majority voting, Section 3.1) ---
        with timer.stage("track"):
            tracked = tracker.track_frame(enhanced_frame, raw_detections, confidence=confidence)

        # --- VLM VERIFY (Section 3.2: sparingly, only for ambiguous detections) ---
        with timer.stage("verify"):
            for obj in tracked:
                if obj.get("needs_vlm_verification", False):
                    majority = tracker.get_majority_class(obj["track_id"])
                    if should_verify(
                        track_id=obj["track_id"],
                        raw_class=obj.get("class_raw", obj.get("class", "")),
                        majority_class=majority,
                        confidence=obj["confidence"],
                        confidence_threshold=VLM_CONFIDENCE_THRESHOLD,
                        current_frame_idx=sample_idx,
                        explicit_flag=obj.get("needs_vlm_verification", False),
                    ):

                        vlm_result = verify_crop_with_vlm(
                            enhanced_frame,
                            obj["bbox"],
                            obj["track_id"],
                            sample_idx,
                        )
                        if vlm_result:
                            tracker.inject_vlm_result(obj["track_id"], vlm_result)
                            # Re-fetch majority class after injection
                            obj["class"] = tracker.get_majority_class(obj["track_id"])
                            vlm_count += 1

        # --- MOTION (ego-motion compensated, Section 6.2) ---
        with timer.stage("motion"):
            classify_motion(
                tracked,
                tracker.centroid_history,
                enhanced_frame,
            )

        # --- DISTANCE (pinhole + depth model) ---
        with timer.stage("distance"):
            bbox_histories = {
                tid: tracker.get_bbox_history(tid)
                for tid in tracker.get_active_track_ids()
            }
            estimate_distances(tracked, enhanced_frame, bbox_histories=bbox_histories)

        # Track depth signal health
        for obj in tracked:
            method = obj.get("distance_method", "unavailable")
            if method == "pinhole_fallback":
                any_depth_degraded = True
            elif method == "unavailable":
                any_depth_unavailable = True

        # --- TTC (Section 4: time-to-collision) ---
        with timer.stage("ttc"):
            compute_ttc(tracked, frame_rate=effective_sample_fps)

        # --- PAIRWISE COLLISION WARNINGS (Section 4: cow-motorcycle case) ---
        collision_warnings_raw = compute_pairwise_collision_warnings(
            tracked, frame_rate=effective_sample_fps
        )
        collision_warnings = [
            CollisionWarning(**cw) for cw in collision_warnings_raw
        ]

        # --- AUDIO-VISUAL STATE (v3 Section 2) ---
        siren_active, siren_conf, siren_cls = is_siren_active_at(siren_events, timestamp)

        # --- RISK SCORING (with TTC + siren integration) ---
        with timer.stage("risk"):
            compute_risk_scores(tracked, w, h)
            chaos = compute_chaos_score(tracked, w, h, siren_active=siren_active)
        chaos_scores.append(chaos)

        # --- VOICE ALERTS (Section 5 + v3 siren) ---
        with timer.stage("voice"):
            voice_alerts_raw = generate_voice_alerts(
                tracked, collision_warnings_raw, timestamp_s=timestamp,
                siren_active=siren_active,
                siren_confidence=siren_conf,
                siren_class=siren_cls,
            )
        voice_alerts = [VoiceAlert(**va, audio_available=False) for va in voice_alerts_raw]

        # --- NEAR-MISS EVENTS (Section 6) ---
        for obj in tracked:
            ttc = obj.get("ttc_s")
            if ttc is not None and ttc < TTC_DANGER_THRESHOLD:
                near_miss_events.append(NearMissEvent(
                    timestamp_s=timestamp,
                    frame_index=sample_idx,
                    track_id=obj.get("track_id"),
                    ttc_s=ttc,
                    risk_level=obj.get("risk_level", "High"),
                    class_name=get_display_name(obj.get("class", "")),
                    event_type="ego_object",
                ))

        for cw in collision_warnings_raw:
            if cw.get("urgency") == "imminent":
                near_miss_events.append(NearMissEvent(
                    timestamp_s=timestamp,
                    frame_index=sample_idx,
                    track_a=cw["track_a"],
                    track_b=cw["track_b"],
                    ttc_s=cw["ttc_s"],
                    risk_level="High",
                    class_name=f"{cw['class_a']} + {cw['class_b']}",
                    event_type="object_object",
                ))

        # Collect metadata
        for obj in tracked:
            all_track_ids.add(obj.get("track_id", 0))
            all_classes.add(obj.get("class", ""))

        timer.end_frame()

        # Build frame result
        fr = _build_frame_result(
            tracked, sample_idx, timestamp, w, h, chaos,
            collision_warnings=collision_warnings_raw,
            voice_alerts=voice_alerts_raw,
            low_light_enhanced=was_enhanced,
            mean_brightness=brightness,
            siren_active=siren_active,
            siren_confidence=siren_conf,
            siren_class=siren_cls,
            frame_processing_ms=timer.get_per_frame_ms()[-1] if timer.get_per_frame_ms() else 0,
        )
        frame_results.append(fr)

        frame_idx += 1
        sample_idx += 1

    cap.release()

    # Handle zero-detection case
    avg_chaos = (
        sum(chaos_scores) / len(chaos_scores) if chaos_scores else 0.0
    )

    # --- System Status (v3 Section 3) ---
    camera_status = SignalStatus(
        status="degraded" if frames_enhanced_count > 0 else "active",
        label="Low-light enhanced" if frames_enhanced_count > 0 else "Camera active",
        detail=f"{frames_enhanced_count} frames enhanced" if frames_enhanced_count > 0 else "",
    )

    if any_depth_unavailable:
        depth_status = SignalStatus(
            status="degraded",
            label="Depth degraded",
            detail="Some objects had no distance estimate",
        )
    elif any_depth_degraded:
        depth_status = SignalStatus(
            status="degraded",
            label="Depth (pinhole only)",
            detail="Depth model unavailable, using size heuristic",
        )
    else:
        depth_status = SignalStatus(
            status="active",
            label="Depth active",
            detail="Pinhole + depth model",
        )

    audio_status_dict = get_audio_status(has_audio_track, siren_events)
    audio_signal = SignalStatus(**audio_status_dict)

    system_status = SystemStatus(
        camera=camera_status,
        depth=depth_status,
        audio=audio_signal,
    )

    # Build audio event models
    audio_event_models = [AudioEvent(**e) for e in siren_events]

    return AnalysisResponse(
        media_type="video",
        total_frames=len(frame_results),
        frame_width=w,
        frame_height=h,
        fps=fps,
        sample_fps=FRAME_SAMPLE_RATE,
        duration_s=round(duration, 2),
        duration_capped=duration_capped,
        frames=frame_results,
        aggregate_chaos_score=round(avg_chaos, 1),
        unique_classes=list(all_classes),
        total_unique_tracks=len(all_track_ids),
        near_miss_events=near_miss_events,
        vlm_verifications=vlm_count,
        frames_enhanced=frames_enhanced_count,
        audio_events=audio_event_models,
        has_audio_track=has_audio_track,
        system_status=system_status,
        pipeline_timings=PipelineTimings(**timer.get_timings()),
    ).model_dump()


def _compute_depth_status(tracked_objects: list[dict]) -> SignalStatus:
    """Determine depth signal status from tracked object distance methods."""
    if not tracked_objects:
        return SignalStatus(status="active", label="Depth active", detail="No objects to measure")

    methods = [obj.get("distance_method", "unavailable") for obj in tracked_objects]
    has_depth_model = any(m == "depth_model" for m in methods)
    has_pinhole = any(m == "pinhole" for m in methods)
    has_unavailable = any(m == "unavailable" for m in methods)

    if has_unavailable and not has_pinhole and not has_depth_model:
        return SignalStatus(
            status="absent",
            label="Distance unavailable",
            detail="No known-size objects for calibration",
        )
    elif has_unavailable:
        return SignalStatus(
            status="degraded",
            label="Depth partial",
            detail="Some objects lack distance estimates",
        )
    elif has_depth_model:
        return SignalStatus(
            status="active",
            label="Depth active",
            detail="Pinhole + depth model",
        )
    elif has_pinhole:
        return SignalStatus(
            status="active",
            label="Depth active",
            detail="Pinhole estimation",
        )
    return SignalStatus(status="active", label="Depth active", detail="")


def _build_frame_result(
    tracked_objects: list[dict],
    frame_index: int,
    timestamp: float,
    frame_width: int,
    frame_height: int,
    chaos_score: float,
    collision_warnings: list | None = None,
    voice_alerts: list | None = None,
    low_light_enhanced: bool = False,
    mean_brightness: float = 0.0,
    siren_active: bool = False,
    siren_confidence: float = 0.0,
    siren_class: str = "",
    frame_processing_ms: float = 0.0,
) -> FrameResult:
    """Convert tracked object dicts into a typed FrameResult."""
    detected = []
    for obj in tracked_objects:
        bbox = obj.get("bbox", [0, 0, 0, 0])
        raw_class = obj.get("class", "unknown")

        # Graceful degradation: "Unclassified obstacle" for unresolved objects
        display_name = get_display_name(raw_class)
        if raw_class == "unknown" or raw_class == "":
            display_name = "Unclassified obstacle"

        # Determine distance display method
        dist_method = obj.get("distance_method", "unavailable")

        detected.append(DetectedObject(
            track_id=obj.get("track_id", 0),
            class_name=display_name,
            class_raw=raw_class,
            confidence=round(obj.get("confidence", 0.0), 3),
            bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
            status=obj.get("status", "Stationary"),
            distance_m=obj.get("distance_m"),
            distance_available=obj.get("distance_available", False),
            distance_method=dist_method,
            risk_level=obj.get("risk_level", "Low"),
            risk_score=obj.get("risk_score", 0.0),
            risk_explanation=obj.get("risk_explanation", ""),
            rel_dx=obj.get("rel_dx", 0.0),
            rel_dy=obj.get("rel_dy", 0.0),
            closing_speed=obj.get("closing_speed", 0.0),
            ttc_s=obj.get("ttc_s"),
        ))

    # Build collision warning models
    cw_models = []
    if collision_warnings:
        for cw in collision_warnings:
            if isinstance(cw, dict):
                cw_models.append(CollisionWarning(**cw))
            else:
                cw_models.append(cw)

    # Build voice alert models
    va_models = []
    if voice_alerts:
        for va in voice_alerts:
            if isinstance(va, dict):
                va_models.append(VoiceAlert(**va, audio_available=False))
            else:
                va_models.append(va)

    return FrameResult(
        frame_index=frame_index,
        timestamp_s=round(timestamp, 3),
        objects=detected,
        chaos_score=chaos_score,
        object_count=len(detected),
        frame_width=frame_width,
        frame_height=frame_height,
        collision_warnings=cw_models,
        voice_alerts=va_models,
        low_light_enhanced=low_light_enhanced,
        mean_brightness=round(mean_brightness, 1),
        siren_active=siren_active,
        siren_confidence=round(siren_confidence, 3),
        siren_class=siren_class,
        frame_processing_ms=round(frame_processing_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
