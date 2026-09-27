"""
SARTHI Vision - Pipeline Timing Harness

Section 4 of upgrade prompt v3: honest edge-readiness measurement.

Wraps each pipeline stage with wall-clock timing. Reports per-stage
and per-frame totals in the JSON export. Never estimates or fabricates
latency numbers -- all values are measured on the actual hardware.
"""

from __future__ import annotations

import time
import logging
from typing import Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PipelineTimer:
    """
    Accumulates wall-clock timing for each pipeline stage.
    Thread-safe for single-process uvicorn (no concurrent writes).

    Usage:
        timer = PipelineTimer()
        with timer.stage("detect"):
            detect_frame(...)
        with timer.stage("track"):
            tracker.track_frame(...)
        timings = timer.get_timings()
    """

    def __init__(self):
        self._stages: dict[str, list[float]] = {}
        self._frame_totals: list[float] = []
        self._current_frame_start: float | None = None

    @contextmanager
    def stage(self, name: str):
        """Context manager to time a single pipeline stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self._stages:
                self._stages[name] = []
            self._stages[name].append(elapsed)

    def start_frame(self):
        """Mark the start of a new frame's processing."""
        self._current_frame_start = time.perf_counter()

    def end_frame(self):
        """Mark the end of a frame's processing and record total."""
        if self._current_frame_start is not None:
            elapsed = time.perf_counter() - self._current_frame_start
            self._frame_totals.append(elapsed)
            self._current_frame_start = None

    def get_timings(self) -> dict[str, Any]:
        """
        Return timing summary for the entire processing run.

        Returns:
            {
                "per_stage_avg_ms": {"detect": 12.3, "track": 5.1, ...},
                "per_stage_total_ms": {"detect": 123.4, ...},
                "per_frame_avg_ms": 45.2,
                "per_frame_min_ms": 30.1,
                "per_frame_max_ms": 62.8,
                "total_frames_timed": 10,
                "total_pipeline_ms": 452.0,
            }
        """
        per_stage_avg = {}
        per_stage_total = {}

        for name, times in self._stages.items():
            total = sum(times)
            avg = total / len(times) if times else 0
            per_stage_avg[name] = round(avg * 1000, 2)  # ms
            per_stage_total[name] = round(total * 1000, 2)  # ms

        frame_times_ms = [t * 1000 for t in self._frame_totals]

        return {
            "per_stage_avg_ms": per_stage_avg,
            "per_stage_total_ms": per_stage_total,
            "per_frame_avg_ms": round(
                sum(frame_times_ms) / len(frame_times_ms), 2
            ) if frame_times_ms else 0,
            "per_frame_min_ms": round(min(frame_times_ms), 2) if frame_times_ms else 0,
            "per_frame_max_ms": round(max(frame_times_ms), 2) if frame_times_ms else 0,
            "total_frames_timed": len(frame_times_ms),
            "total_pipeline_ms": round(sum(frame_times_ms), 2) if frame_times_ms else 0,
        }

    def get_per_frame_ms(self) -> list[float]:
        """Return per-frame processing times in ms."""
        return [round(t * 1000, 2) for t in self._frame_totals]
