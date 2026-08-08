"""Sarvam Cloud Lead Agent - Pydantic application settings.

All values can be overridden through environment variables or a local .env file.
This project is fully independent from the other MVP project and talks to the
Sarvam cloud API (STT + TTS + optional chat) rather than to local services.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sarvam TTS accepts target_language_code in this closed set.
SARVAM_TTS_LANGUAGES = {
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN",
    "od-IN", "pa-IN", "ta-IN", "te-IN",
}

# Maps detected_language (en, hi, gu, en-hi, ...) to a valid TTS target code.
DETECTED_TO_TTS_LANGUAGE = {
    "en": "en-IN",
    "en-in": "en-IN",
    "en-gb": "en-IN",
    "en-us": "en-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "en-hi": "hi-IN",
    "hi-en": "hi-IN",
    "hinglish": "hi-IN",
    "gu": "gu-IN",
    "gu-in": "gu-IN",
    "gujarati": "gu-IN",
    "gujlish": "gu-IN",
}


def tts_language_code_for(detected_language: str | None, default: str) -> str:
    key = (detected_language or "").strip().lower()
    code = DETECTED_TO_TTS_LANGUAGE.get(key, default)
    return code if code in SARVAM_TTS_LANGUAGES else default


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Sarvam Cloud Lead Agent"
    app_host: str = "0.0.0.0"
    app_port: int = 8021
    debug: bool = False

    database_url: str = "sqlite:///./data/sarvam_leads.db"

    sarvam_base_url: str = "https://api.sarvam.ai"
    sarvam_api_key: str = ""
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_mode: str = "transcribe"
    sarvam_language_code: str = "unknown"
    # Realtime phone STT is opt-in so browser/file uploads keep the established
    # REST path. The call session falls back to local VAD + REST STT if the
    # realtime socket cannot be established.
    sarvam_realtime_stt_enabled: bool = False
    sarvam_realtime_stt_url: str = (
        "wss://api.sarvam.ai/speech-to-text-realtime/ws"
    )
    sarvam_realtime_stt_model: str = "saaras:v3-realtime"
    sarvam_realtime_stt_vad_threshold: float = 0.5
    sarvam_realtime_stt_silence_ms: int = 400
    sarvam_realtime_stt_min_speech_ms: int = 100
    sarvam_semantic_endpointing_enabled: bool = True
    sarvam_semantic_fast_silence_ms: int = 320
    sarvam_semantic_slow_silence_ms: int = 550
    sarvam_realtime_stt_fallback_enabled: bool = True
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "ritu"
    sarvam_tts_language_code: str = "gu-IN"
    # Bulbul v3 expressiveness/speed. Sarvam recommends ~0.7-0.8 for warm
    # conversational agents; 1.0 is natural pace.
    sarvam_tts_temperature: float = 0.75
    sarvam_tts_pace: float = 1.0
    # Output sample rate for Sarvam TTS. 8000 Hz is best for Twilio telephony
    # (avoids Twilio's lower-quality resampling of full-band audio); higher
    # rates (24000/48000) only matter for non-phone playback.
    sarvam_tts_speech_sample_rate: int = 8000
    # When true, use Sarvam's /text-to-speech/stream endpoint so Twilio <Play>
    # starts before the whole reply is synthesized. Set to false to fall back to
    # the buffered /text-to-speech endpoint (Sarvam 30-free-units model or API
    # keys without streaming access).
    sarvam_tts_streaming: bool = True
    # Persistent WebSocket TTS is used only by bidirectional phone transports.
    # HTTP streaming and buffered synthesis remain available as fallbacks.
    sarvam_realtime_tts_enabled: bool = False
    sarvam_realtime_tts_url: str = (
        "wss://api.sarvam.ai/v1/text-to-speech/stream"
    )
    sarvam_realtime_tts_codec: str = "linear16"
    sarvam_request_timeout: float = 120.0

    llm_provider: str = "sarvam"
    llm_base_url: str = "https://api.sarvam.ai"
    llm_api_key: str = ""
    llm_model: str = "sarvam-105b"
    llm_timeout: float = 120.0
    llm_use_json_mode: bool = True
    # Higher than 0.2 so wording feels human; still low enough for JSON structure.
    llm_temperature: float = 0.55
    # Sarvam reasoning/thinking mode. Empty = leave the provider default (Sarvam
    # defaults to thinking ON, which adds several seconds of hidden reasoning and
    # completion cost per turn). "none"/"off"/"false" sends reasoning_effort=null
    # to disable it; "low"/"medium"/"high" requests that level.
    llm_reasoning_effort: str = ""
    llm_max_tokens: int = 0
    llm_streaming_enabled: bool = False
    # Phone calls prioritize time-to-first-token. This override is deliberately
    # independent of the browser/API reasoning setting.
    phone_llm_reasoning_effort: str = "none"
    # Optional faster/smaller model for phone/Gather turns only. Empty = llm_model.
    # Example with openai-compatible: PHONE_LLM_MODEL=gpt-4o-mini
    phone_llm_model: str = ""
    # Shorter phone completions (Gather path). 0 falls back to llm_max_tokens or 140.
    phone_llm_max_tokens: int = 200
    phone_llm_temperature: float = 0.4
    # Compact system prompt on phone transports (less TTFT).
    phone_prompt_compact: bool = True

    # Twilio Gather (trial) latency knobs — do not expect Media-Streams 2–3s here.
    # Wait longer inline before Pause/Redirect so more turns return Play+Gather.
    gather_inline_budget_seconds: float = 5.0
    # Silent hold before Redirect. 0 = Redirect immediately (saves ~1s).
    gather_poll_pause_seconds: int = 0
    # Twilio speechTimeout: seconds of silence after speech, or "auto".
    gather_speech_timeout: str = "auto"
    # Whole Gather listen window before empty SpeechResult.
    gather_timeout_seconds: int = 7
    # After this many consecutive empty SpeechResults, speak a "still there?"
    # prompt instead of silent re-listen. 1 = prompt on first empty.
    gather_empty_prompt_after: int = 3

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    twilio_from_number: str = ""
    twilio_test_phone_number: str = ""
    twilio_verified_numbers: str = ""
    twilio_trial_mode: bool = True
    public_base_url: str = ""
    twilio_call_public_base_url: str = ""
    twilio_status_callback_url: str = ""
    twilio_turn_webhook_secret: str = ""
    twilio_call_timeout: float = 60.0

    # Outbound carrier selection. Existing clients omit the provider and keep
    # using Twilio by default; callers may opt into Exotel per request.
    telephony_provider: str = "twilio"
    exotel_base_url: str = "https://api.in.exotel.com"
    exotel_account_sid: str = ""
    exotel_api_key: str = ""
    exotel_api_token: str = ""
    exotel_caller_id: str = ""
    exotel_flow_id: str = ""
    exotel_status_callback_url: str = ""
    exotel_call_timeout: float = 60.0

    default_language: str = "en"
    max_audio_mb: int = 15

    # Business identity injected into the agent's system prompt so the opening
    # and the whole call represent the brand instead of a generic assistant.
    business_name: str = "Vrattiks"
    business_description: str = "a technology and software company focused on building AI-powered solutions for businesses and individuals."

    # BDE outbound persona + call facts (see config/call_profile.json).
    agent_name: str = "Shivangi"
    call_profile_path: str = "config/call_profile.json"
    # Optional JSON array/string override for products; empty keeps file values.
    business_products: str = ""
    contact_source_channel: str = ""
    contact_source_detail: str = ""
    default_lead_full_name: str = ""
    default_lead_company_name: str = ""
    default_lead_job_title: str = ""
    default_lead_city: str = ""
    default_lead_business_type: str = ""
    default_lead_field_note: str = ""
    default_lead_additional_notes: str = ""
    # Soft AI disclosure in opening (keeps compliance; wording still BDE-like).
    disclose_ai_assistant: bool = True

    # Verbose pipeline tracing: listen/transcript/LLM/TTS/play with I/O + timings.
    pipeline_trace_enabled: bool = True
    pipeline_trace_max_chars: int = 2000

    retain_audio: bool = False
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    call_rate_limit_per_minute: int = 5
    allowed_origins: str = ""
    cors_enabled: bool = False

    temp_dir: str = str(PROJECT_ROOT / "storage" / "tmp")
    ffmpeg_path: str = "ffmpeg"

    stt_rate_per_hour_inr: float = 30.0
    tts_rate_per_10k_chars_inr: float = 30.0
    llm_input_rate_per_million_inr: float = 4.0
    llm_output_rate_per_million_inr: float = 16.0

    @field_validator("sarvam_tts_speaker", mode="before")
    @classmethod
    def _clean_tts_speaker(cls, value: object) -> object:
        """Strip inline comments / whitespace from .env speaker values."""
        if not isinstance(value, str):
            return value
        cleaned = value.split("#", 1)[0].strip().strip("\"'")
        return cleaned or "ritu"

    @field_validator("telephony_provider")
    @classmethod
    def _validate_telephony_provider(cls, value: str) -> str:
        provider = (value or "twilio").strip().lower()
        if provider not in {"twilio", "exotel"}:
            raise ValueError("TELEPHONY_PROVIDER must be 'twilio' or 'exotel'")
        return provider

    @property
    def resolved_temp_dir(self) -> Path:
        path = Path(self.temp_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def twilio_from(self) -> str:
        """The From number to use for outbound calls (new name wins, old kept as alias)."""
        return self.twilio_phone_number or self.twilio_from_number

    @property
    def public_base(self) -> str:
        """Public base URL (new name wins, old kept as alias)."""
        return self.public_base_url or self.twilio_call_public_base_url

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
