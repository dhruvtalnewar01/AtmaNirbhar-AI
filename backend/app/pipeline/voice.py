"""
SARTHI Vision - Voice Alert System

Section 5 of the upgrade prompt: voice alerts.

Rules (non-negotiable):
1. The LLM NARRATES, it NEVER DECIDES risk.
   Every voice alert's underlying fact (which object, what distance, what TTC,
   what risk tier) is computed by deterministic code before the LLM sees it.
   The LLM's only job is phrasing that fact as a short spoken sentence.
2. Urgency tiers (computed entirely by deterministic logic):
   - Moderate risk: visual indicator only, no sound.
   - High risk: LLM-phrased spoken sentence via tts-1.
   - Imminent (TTC < 2s): pre-written urgent cue immediately, then narration.
3. Queue: most urgent pending alert plays first. No overlapping speech.
4. Every spoken sentence is traceable to the structured fact that generated it.
5. Uses tts-1 (not Realtime API) to stay credit-conscious.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.config import (
    OPENAI_API_KEY,
    OPENAI_VLM_MODEL,
    OPENAI_TTS_MODEL,
    TTC_DANGER_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Lazy-loaded OpenAI client
_openai_client = None


def _get_openai_client():
    """Lazy-load OpenAI client."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized for voice alerts (TTS: %s)", OPENAI_TTS_MODEL)
        return _openai_client
    except Exception as e:
        logger.warning("Failed to initialize OpenAI client for voice: %s", e)
        return None


def generate_voice_alerts(
    tracked_objects: list[dict[str, Any]],
    collision_warnings: list[dict[str, Any]],
    timestamp_s: float = 0.0,
    siren_active: bool = False,
    siren_confidence: float = 0.0,
    siren_class: str = "",
) -> list[dict[str, Any]]:
    """
    Generate voice alert metadata for high-risk and imminent events.

    Returns a list of VoiceAlert dicts:
    [
        {
            "urgency": "imminent" | "high",
            "fact": {
                "class": "cow",
                "track_id": 3,
                "distance_m": 4.2,
                "ttc_s": 1.5,
                "risk_level": "High",
                "type": "ego_object" | "object_object" | "siren",
            },
            "phrase": "Cow detected 4 meters ahead, collision in 1.5 seconds.",
            "timestamp_s": 12.5,
        },
        ...
    ]

    Audio synthesis is handled separately via the /api/voice-alert endpoint.
    This function only generates the structured alert metadata.
    """
    alerts = []

    # --- 1. Imminent pairwise collision warnings (highest priority) ---
    for warning in collision_warnings:
        if warning.get("urgency") == "imminent":
            fact = {
                "class_a": warning["class_a"],
                "class_b": warning["class_b"],
                "track_a": warning["track_a"],
                "track_b": warning["track_b"],
                "ttc_s": warning["ttc_s"],
                "distance_m": warning["current_distance_m"],
                "type": "object_object",
            }
            phrase = _build_imminent_phrase_pairwise(warning)
            alerts.append({
                "urgency": "imminent",
                "fact": fact,
                "phrase": phrase,
                "timestamp_s": timestamp_s,
            })

    # --- 2. Per-object imminent TTC warnings ---
    for obj in tracked_objects:
        ttc = obj.get("ttc_s")
        risk_level = obj.get("risk_level", "Low")

        if ttc is not None and ttc < TTC_DANGER_THRESHOLD:
            fact = {
                "class": obj.get("class", "unknown"),
                "track_id": obj.get("track_id", 0),
                "distance_m": obj.get("distance_m"),
                "ttc_s": ttc,
                "risk_level": risk_level,
                "type": "ego_object",
            }
            phrase = _build_imminent_phrase(obj)
            alerts.append({
                "urgency": "imminent",
                "fact": fact,
                "phrase": phrase,
                "timestamp_s": timestamp_s,
            })

    # --- 3. Siren detection alert (v3 Section 2) ---
    # Phrased honestly: tells what's known (siren nearby),
    # NOT a false claim about direction or distance.
    if siren_active and siren_confidence > 0:
        fact = {
            "class": siren_class or "Siren",
            "confidence": siren_confidence,
            "type": "siren",
        }
        # Honest phrasing: no direction/distance claims
        phrase = "Siren detected nearby, exact location unknown. Proceed with caution."
        alerts.append({
            "urgency": "high",
            "fact": fact,
            "phrase": phrase,
            "timestamp_s": timestamp_s,
        })

    # --- 4. High risk objects (not imminent) ---
    for obj in tracked_objects:
        risk_level = obj.get("risk_level", "Low")
        ttc = obj.get("ttc_s")

        # Skip if already handled as imminent
        if ttc is not None and ttc < TTC_DANGER_THRESHOLD:
            continue

        if risk_level == "High":
            fact = {
                "class": obj.get("class", "unknown"),
                "track_id": obj.get("track_id", 0),
                "distance_m": obj.get("distance_m"),
                "ttc_s": ttc,
                "risk_level": risk_level,
                "type": "ego_object",
            }
            phrase = _build_high_risk_phrase(obj)
            alerts.append({
                "urgency": "high",
                "fact": fact,
                "phrase": phrase,
                "timestamp_s": timestamp_s,
            })

    # Sort by urgency (imminent first), then by TTC (lowest first)
    urgency_order = {"imminent": 0, "high": 1}
    alerts.sort(key=lambda a: (
        urgency_order.get(a["urgency"], 99),
        a["fact"].get("ttc_s", 999) or 999,
    ))

    return alerts


# In-memory TTS audio cache: text -> mp3 bytes
_tts_cache: dict[str, bytes] = {}

def synthesize_speech(text: str) -> bytes | None:
    """
    Synthesize speech audio from text using OpenAI TTS-1 with memory caching.
    Returns raw audio bytes (mp3 format) or None if synthesis failed.
    """
    normalized = text.strip()
    if not normalized:
        return None

    if normalized in _tts_cache:
        return _tts_cache[normalized]

    client = _get_openai_client()
    if client is None:
        return None

    try:
        response = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice="onyx",  # Deep, authoritative voice for warnings
            input=normalized,
            speed=1.1,  # Slightly faster for urgency
        )
        audio_data = response.content
        if audio_data:
            # Cache up to 100 phrases
            if len(_tts_cache) > 100:
                _tts_cache.clear()
            _tts_cache[normalized] = audio_data
        return audio_data

    except Exception as e:
        logger.warning("TTS synthesis failed: %s", e)
        return None



def phrase_with_llm(fact: dict[str, Any]) -> str | None:
    """
    Use GPT-4o-mini to phrase a structured fact as a natural spoken sentence.
    The LLM only phrases, it never decides risk or modifies the fact.
    """
    client = _get_openai_client()
    if client is None:
        return None

    try:
        fact_description = _fact_to_text(fact)
        prompt = (
            "You are a vehicle safety alert system. "
            "Given this detection fact, write ONE short spoken alert sentence (under 15 words). "
            "Be direct and clear. Do not add information not in the fact. "
            "Do not use em dashes. Use periods or commas only.\n\n"
            f"Fact: {fact_description}"
        )

        response = client.chat.completions.create(
            model=OPENAI_VLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,
        )

        phrase = response.choices[0].message.content.strip()
        # Remove any em dashes the LLM might insert
        phrase = phrase.replace("\u2014", ",").replace("\u2013", ",")
        return phrase

    except Exception as e:
        logger.warning("LLM phrasing failed: %s", e)
        return None


def _build_imminent_phrase(obj: dict) -> str:
    """Build early predictive collision alert phrase before impact."""
    cls = obj.get("class", "obstacle").replace("-", " ").title()
    distance = obj.get("distance_m")
    ttc = obj.get("ttc_s")

    dist_str = f"{distance:.0f} meters" if distance else "ahead"
    ttc_str = f"{ttc:.1f} seconds" if ttc else "3 seconds"

    return f"Collision predicted in {ttc_str}. {cls} {dist_str} ahead, apply brakes."


def _build_imminent_phrase_pairwise(warning: dict) -> str:
    """Build early predictive alert for object-to-object convergence before impact."""
    cls_a = warning.get("class_a", "object").replace("-", " ").title()
    cls_b = warning.get("class_b", "object").replace("-", " ").title()
    ttc = warning.get("ttc_s")

    ttc_str = f"{ttc:.1f} seconds" if ttc else "3 seconds"
    return f"Collision predicted in {ttc_str}. {cls_a} and {cls_b} converging."


def _build_high_risk_phrase(obj: dict) -> str:
    """Build a prior predictive high-risk cautionary phrase."""
    cls = obj.get("class", "obstacle").replace("-", " ").title()
    distance = obj.get("distance_m")
    status = obj.get("status", "detected")
    ttc = obj.get("ttc_s")

    dist_str = f"{distance:.0f} meters" if distance else "ahead"
    if ttc is not None and ttc > 0:
        return f"Warning. {cls} detected {dist_str}, collision predicted in {ttc:.1f} seconds."

    closing = "closing rapidly" if obj.get("closing_speed", 0) < -0.1 else status.lower()
    return f"Caution. {cls} detected {dist_str}, {closing}."


def _fact_to_text(fact: dict) -> str:
    """Convert a structured fact dict to a human-readable description for the LLM."""
    parts = []

    if fact.get("type") == "object_object":
        parts.append(f"{fact.get('class_a', 'object')} and {fact.get('class_b', 'object')} converging")
        if fact.get("ttc_s"):
            parts.append(f"collision in {fact['ttc_s']} seconds")
        if fact.get("distance_m"):
            parts.append(f"{fact['distance_m']}m apart")
    else:
        parts.append(f"{fact.get('class', 'obstacle')} detected")
        if fact.get("track_id"):
            parts.append(f"track {fact['track_id']}")
        if fact.get("distance_m"):
            parts.append(f"{fact['distance_m']}m away")
        if fact.get("ttc_s"):
            parts.append(f"collision in {fact['ttc_s']} seconds")
        parts.append(f"risk: {fact.get('risk_level', 'unknown')}")

    return ", ".join(parts)
