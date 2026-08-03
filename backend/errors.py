"""Application-specific exceptions and the canonical JSON error shape.

All API errors are rendered as:

    {
      "error": {
        "code": "PROVIDER_UNAVAILABLE",
        "message": "Human-readable message",
        "retryable": true,
        "details": null
      }
    }
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AppError(Exception):
    code: str
    message: str
    retryable: bool = False
    details: Any = None
    status_code: int = 500


class ProviderUnavailableError(AppError):
    def __init__(self, message: str, details: Any = None):
        super().__init__(
            code="PROVIDER_UNAVAILABLE",
            message=message,
            retryable=True,
            details=details,
            status_code=503,
        )


class SttError(AppError):
    def __init__(self, message: str, details: Any = None, retryable: bool = True):
        super().__init__(
            code="STT_TRANSCRIPTION_FAILED",
            message=message,
            retryable=retryable,
            details=details,
            status_code=502,
        )


class TtsError(AppError):
    def __init__(self, message: str, details: Any = None, retryable: bool = True):
        super().__init__(
            code="TTS_GENERATION_FAILED",
            message=message,
            retryable=retryable,
            details=details,
            status_code=502,
        )


class LlmError(AppError):
    def __init__(self, message: str, details: Any = None, retryable: bool = True):
        super().__init__(
            code="LLM_UNAVAILABLE",
            message=message,
            retryable=retryable,
            details=details,
            status_code=502,
        )


class LlmStructuredOutputError(AppError):
    def __init__(self, message: str, details: Any = None):
        super().__init__(
            code="INVALID_LLM_STRUCTURED_RESPONSE",
            message=message,
            retryable=False,
            details=details,
            status_code=502,
        )


class FfmpegMissingError(AppError):
    def __init__(self, details: Any = None):
        super().__init__(
            code="FFMPEG_MISSING",
            message="FFmpeg is required to prepare audio but was not found. "
            "Install FFmpeg and ensure it is on PATH (see SETUP.md).",
            retryable=False,
            details=details,
            status_code=500,
        )


class InvalidAudioError(AppError):
    def __init__(self, message: str, details: Any = None):
        super().__init__(
            code="INVALID_AUDIO",
            message=message,
            retryable=False,
            details=details,
            status_code=400,
        )


class SessionNotFoundError(AppError):
    def __init__(self, session_id: str):
        super().__init__(
            code="SESSION_NOT_FOUND",
            message=f"Session {session_id} does not exist or has been deleted.",
            retryable=False,
            status_code=404,
        )


class LeadNotFoundError(AppError):
    def __init__(self, lead_id: str):
        super().__init__(
            code="LEAD_NOT_FOUND",
            message=f"Lead {lead_id} does not exist or has been deleted.",
            retryable=False,
            status_code=404,
        )
