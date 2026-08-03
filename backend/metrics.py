"""Turn metrics and cost estimation.

Sarvam is a paid cloud API, so per-turn provider costs are estimated from the
published rate cards (configurable in Settings / .env):
  - STT: per audio hour (saaras:v3)
  - TTS: per 10k characters (bulbul:v3)
  - LLM: per million input / output tokens (sarvam-105b)

These are estimates for display only; the invoice from Sarvam is authoritative.
"""

import time
from dataclasses import dataclass, field

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
    audio_duration_ms: int = 0
    stt_latency_ms: int = 0
    llm_latency_ms: int = 0
    tts_latency_ms: int = 0
    transcript_char_count: int = 0
    response_char_count: int = 0
    settings: Settings | None = None
    llm_usage: dict | None = None
    tts_attempted: bool = False

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
            transcript_character_count=self.transcript_char_count,
            response_character_count=self.response_char_count,
            estimated_provider_cost=total_cost,
        )
