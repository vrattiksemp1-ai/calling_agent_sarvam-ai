"""Sarvam Cloud Lead Agent - Pydantic application settings.

All values can be overridden through environment variables or a local .env file.
This project is fully independent from the other MVP project and talks to the
Sarvam cloud API (STT + TTS + optional chat) rather than to local services.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sarvam TTS accepts target_language_code in this closed set.
SARVAM_TTS_LANGUAGES = {
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN",
    "od-IN", "pa-IN", "ta-IN", "te-IN",
}

# Maps detected_language (en, hi, en-hi, ...) to a valid TTS target code.
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
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "shubh"
    sarvam_tts_language_code: str = "hi-IN"
    sarvam_request_timeout: float = 120.0

    llm_provider: str = "sarvam"
    llm_base_url: str = "https://api.sarvam.ai"
    llm_api_key: str = ""
    llm_model: str = "sarvam-105b"
    llm_timeout: float = 120.0
    llm_use_json_mode: bool = True

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_call_public_base_url: str = ""
    twilio_call_timeout: float = 60.0

    default_language: str = "en"
    max_audio_mb: int = 15

    retain_audio: bool = False
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    allowed_origins: str = ""
    cors_enabled: bool = False

    temp_dir: str = str(PROJECT_ROOT / "storage" / "tmp")
    ffmpeg_path: str = "ffmpeg"

    stt_rate_per_hour_inr: float = 30.0
    tts_rate_per_10k_chars_inr: float = 30.0
    llm_input_rate_per_million_inr: float = 4.0
    llm_output_rate_per_million_inr: float = 16.0

    @property
    def resolved_temp_dir(self) -> Path:
        path = Path(self.temp_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
