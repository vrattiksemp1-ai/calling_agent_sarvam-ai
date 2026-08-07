"""Turn metrics and cost estimation.

Sarvam is a paid cloud API, so per-turn provider costs are estimated from the
published rate cards (configurable in Settings / .env):
  - STT: per audio hour (saaras:v3)
  - TTS: per 10k characters (bulbul:v3)
  - LLM: per million input / output tokens (sarvam-105b)

These are estimates for display only; the invoice from Sarvam is authoritative.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.config import Settings
from backend.schemas import TurnMetrics


def estimate_stt_cost(duration_ms: int, rate_per_hour: float) -> float:
    return round(duration_ms / 3_600_000.0 * rate_per_hour, 6)


def estimate_tts_cost(char_count: int, rate_per_10k: float) -> float:
    return round(char_count / 10_000.0 * rate_per_10k, 6)


def estimate_llm_cost(
    usage: dict, input_rate_per_million: float, output_rate_per_million: float
) -> float:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return round(
        prompt_tokens / 1_000_000.0 * input_rate_per_million
        + completion_tokens / 1_000_000.0 * output_rate_per_million,
        6,
    )


@dataclass
class TurnTimings:
    started: float = field(default_factory=time.monotonic)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turn_id: str = ""
    audio_duration_ms: int = 0
    stt_latency_ms: int = 0
    llm_latency_ms: int = 0
    tts_latency_ms: int = 0
    transcript_char_count: int = 0
    response_char_count: int = 0
    settings: Settings | None = None
    llm_usage: dict | None = None
    tts_attempted: bool = False
    transport: str = "api"
    stt_provider: str | None = None
    llm_provider: str | None = None
    tts_provider: str | None = None
    language_expected: str | None = None
    language_detected: str | None = None
    stt_language: str | None = None
    reply_script: str | None = None
    language_mismatch: bool = False
    language_repair: str | None = None
    reasoning_mode: str | None = None
    repair_count: int = 0
    retry_count: int = 0
    fallback_count: int = 0
    transport_retry_count: int = 0
    llm_attempt_count: int = 0
    tts_mode: str | None = None
    session_id: str | None = None
    assistant_message_id: int | None = None
    phase_elapsed_ms: dict[str, int] = field(default_factory=dict)
    phase_durations_ms: dict[str, int] = field(default_factory=dict)
    phase_timestamps: dict[str, str] = field(default_factory=dict)
    _persisted_phases: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if not self.turn_id:
            # Correlates structured logs and ProviderEvent rows without changing
            # the existing SQLite schema.
            import uuid

            self.turn_id = uuid.uuid4().hex
        if self.settings is not None:
            self.llm_provider = self.settings.llm_provider
            configured_reasoning = (
                self.settings.llm_reasoning_effort or ""
            ).strip().lower()
            self.reasoning_mode = configured_reasoning or "provider_default"

    def mark(self, phase: str, *, at: float | None = None) -> int:
        """Record the first occurrence of a phase relative to turn receipt."""
        now = time.monotonic() if at is None else at
        elapsed = max(0, int((now - self.started) * 1000))
        if phase not in self.phase_elapsed_ms:
            self.phase_elapsed_ms[phase] = elapsed
            wall = self.started_at.timestamp() + (elapsed / 1000.0)
            self.phase_timestamps[phase] = datetime.fromtimestamp(
                wall, timezone.utc
            ).isoformat()
        return self.phase_elapsed_ms[phase]

    def add_duration(self, phase: str, duration_ms: int) -> int:
        """Accumulate repeated delays such as deferred-turn polls."""
        value = max(0, int(duration_ms))
        self.phase_durations_ms[phase] = (
            self.phase_durations_ms.get(phase, 0) + value
        )
        self.mark(phase)
        return self.phase_durations_ms[phase]

    def duration_between(self, start_phase: str, end_phase: str) -> int | None:
        start = self.phase_elapsed_ms.get(start_phase)
        end = self.phase_elapsed_ms.get(end_phase)
        if start is None or end is None:
            return None
        return max(0, end - start)

    def caller_perceived(self) -> int | None:
        """Caller speech end (or webhook receipt) to outbound response audio."""
        start_phase = (
            "utterance_end"
            if "utterance_end" in self.phase_elapsed_ms
            else "transcript_received"
        )
        return self.duration_between(start_phase, "first_outbound_audio")

    def dimensions(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "stt_provider": self.stt_provider,
            "llm_provider": self.llm_provider,
            "tts_provider": self.tts_provider,
            "language_expected": self.language_expected,
            "language_detected": self.language_detected,
            "stt_language": self.stt_language,
            "reply_script": self.reply_script,
            "language_mismatch": self.language_mismatch,
            "language_repair": self.language_repair,
            "reasoning_mode": self.reasoning_mode,
            "repair_count": self.repair_count,
            "retry_count": self.retry_count,
            "fallback_count": self.fallback_count,
            "transport_retry_count": self.transport_retry_count,
            "llm_attempt_count": self.llm_attempt_count,
            "tts_mode": self.tts_mode,
        }

    def log(self, logger, event: str = "turn_telemetry", **extra: Any) -> None:
        """Emit machine-readable telemetry without requiring a DB migration."""
        payload = {
            "event": event,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "caller_perceived_latency_ms": self.caller_perceived(),
            "total_elapsed_ms": self.total(),
            "phase_elapsed_ms": self.phase_elapsed_ms,
            "phase_durations_ms": self.phase_durations_ms,
            "phase_timestamps": self.phase_timestamps,
            "dimensions": self.dimensions(),
            **extra,
        }
        logger.info("telemetry=%s", json.dumps(payload, sort_keys=True))

    def total(self) -> int:
        return max(0, int((time.monotonic() - self.started) * 1000))

    def estimated_stt_cost(self) -> float:
        if not self.settings:
            return 0.0
        return estimate_stt_cost(self.audio_duration_ms, self.settings.stt_rate_per_hour_inr)

    def estimated_tts_cost(self) -> float:
        if not self.settings or not self.tts_attempted:
            return 0.0
        return estimate_tts_cost(self.response_char_count, self.settings.tts_rate_per_10k_chars_inr)

    def llm_cost(self) -> float:
        if not self.settings or not self.llm_usage:
            return 0.0
        return estimate_llm_cost(
            self.llm_usage,
            self.settings.llm_input_rate_per_million_inr,
            self.settings.llm_output_rate_per_million_inr,
        )

    def to_schema(self, llm_usage: dict | None = None) -> TurnMetrics:
        llm_cost = self.llm_cost() if llm_usage is None else (
            estimate_llm_cost(
                llm_usage,
                self.settings.llm_input_rate_per_million_inr,
                self.settings.llm_output_rate_per_million_inr,
            )
            if self.settings and llm_usage
            else 0.0
        )
        total_cost = round(
            self.estimated_stt_cost() + self.estimated_tts_cost() + llm_cost, 6
        )
        return TurnMetrics(
            audio_duration_ms=self.audio_duration_ms,
            stt_latency_ms=self.stt_latency_ms,
            llm_latency_ms=self.llm_latency_ms,
            tts_latency_ms=self.tts_latency_ms,
            total_turn_latency_ms=self.total(),
            caller_perceived_latency_ms=self.caller_perceived(),
            phase_durations_ms=dict(self.phase_durations_ms),
            phase_timestamps=dict(self.phase_timestamps),
            telemetry_dimensions=self.dimensions(),
            transcript_character_count=self.transcript_char_count,
            response_character_count=self.response_char_count,
            estimated_provider_cost=total_cost,
        )


_PHASE_PROVIDERS = {
    "vad_speech_start": "sarvam",
    "vad_speech_end": "sarvam",
    "stt_first_partial": "sarvam",
    "stt_first_final": "sarvam",
    "semantic_endpoint_adjustment": "sarvam",
    "utterance_end": "twilio",
    "transcript_received": "twilio",
    "llm_first_token": "llm",
    "llm_completed": "llm",
    "tts_first_audio": "sarvam",
    "tts_reconnect": "sarvam",
    "first_outbound_audio": "twilio",
    "redirect_issued": "twilio",
    "redirect_poll_received": "twilio",
    "redirect_transit_delay": "twilio",
    "redirect_poll_delay": "twilio",
    "playback_mark": "twilio",
    "playback_duration": "twilio",
    "interruption_clear": "twilio",
    "interruption_clear_delay": "twilio",
}


def persist_turn_telemetry(db, session_id: str | None, timings: TurnTimings) -> None:
    """Append phase rows using the existing ProviderEvent table only."""
    if not session_id:
        return
    from backend.models import ProviderEvent

    timings.session_id = session_id
    phases = set(timings.phase_elapsed_ms) | set(timings.phase_durations_ms)
    for phase in sorted(phases - timings._persisted_phases):
        latency = timings.phase_durations_ms.get(
            phase, timings.phase_elapsed_ms.get(phase)
        )
        provider = _PHASE_PROVIDERS.get(phase, "backend")
        if phase == "transcript_received" and timings.stt_provider:
            provider = timings.stt_provider
        elif phase == "llm_completed" and timings.llm_provider:
            provider = timings.llm_provider
        elif phase == "tts_first_audio" and timings.tts_provider:
            provider = timings.tts_provider
        elif phase in {
            "first_outbound_audio",
            "playback_mark",
            "playback_duration",
            "interruption_clear",
            "interruption_clear_delay",
        }:
            provider = (
                "exotel"
                if timings.transport == "exotel_agent_stream"
                else "twilio"
            )
        db.add(
            ProviderEvent(
                session_id=session_id,
                provider=provider[:32],
                event_type=phase[:32],
                status="ok",
                latency_ms=latency,
                request_id=timings.turn_id,
            )
        )
        timings._persisted_phases.add(phase)
