"""Pydantic request/response schemas shared across the API."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.models import LEAD_FIELDS


class AudioUpload(BaseModel):
    filename: str = Field(..., max_length=255)
    content_type: str = ""
    size: int = 0


class TextMessage(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class ConfirmRequest(BaseModel):
    confirmed: bool
    corrections: str | None = Field(default=None, max_length=1000)


class TurnMetrics(BaseModel):
    audio_duration_ms: int = 0
    stt_latency_ms: int = 0
    llm_latency_ms: int = 0
    tts_latency_ms: int = 0
    total_turn_latency_ms: int = 0
    caller_perceived_latency_ms: int | None = None
    phase_durations_ms: dict[str, int] = Field(default_factory=dict)
    phase_timestamps: dict[str, str] = Field(default_factory=dict)
    telemetry_dimensions: dict[str, Any] = Field(default_factory=dict)
    transcript_character_count: int = 0
    response_character_count: int = 0
    estimated_provider_cost: float = 0.0


class LeadOut(BaseModel):
    id: int | None = None
    session_id: str
    fields: dict[str, Any] = Field(default_factory=dict)
    skipped_fields: list[str] = Field(default_factory=list)
    qualification_score: int = 0
    qualification_level: str = "cold"
    missing_important_fields: list[str] = Field(default_factory=list)
    recommended_next_action: str = ""
    conversation_status: str = "in_progress"
    consent_confirmed: bool = False
    summary_confirmed: bool = False


class TurnResponse(BaseModel):
    session_id: str
    transcript: str
    assistant_message: str
    audio_base64: str | None = None
    audio_mime: str | None = None
    lead: LeadOut
    current_state: str
    conversation_status: str
    metrics: TurnMetrics
    warning: str | None = None
    debug: dict[str, Any] | None = None


class ProviderStatus(BaseModel):
    provider: str
    status: Literal["ok", "degraded", "error"]
    message: str
    latency_ms: int | None = None
    details: dict[str, Any] | None = None


class HealthOut(BaseModel):
    status: str = "ok"
    app: str
    database: str = "ok"
    provider: str = "configured"
    version: str = "1.0.0"


class ConfigOut(BaseModel):
    provider: str
    debug: bool
    default_language: str
    max_audio_mb: int
    stt_model: str
    tts_model: str
    voice: str
    llm_model: str


class SessionOut(BaseModel):
    id: str
    status: str
    current_state: str
    language: str
    created_at: str
    updated_at: str
    message_count: int = 0
    turn_count: int = 0
    average_latency_ms: int = 0
    lead: LeadOut | None = None


class LeadListEntry(BaseModel):
    id: int
    session_id: str
    full_name: str | None
    phone_number: str | None
    email: str | None
    qualification_score: int
    qualification_level: str
    conversation_status: str
    created_at: str


LEAD_FIELD_NAMES = LEAD_FIELDS


class SessionSummary(BaseModel):
    session_id: str
    turn_count: int
    average_latency_ms: int
    completion_status: str
    qualification_score: int
    qualification_level: str
    collected_field_count: int
    missing_important_fields: list[str]
    estimated_provider_cost: float
