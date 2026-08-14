"""SQLAlchemy ORM models for the Sarvam Cloud Lead Agent.

Every table is specific to this project. Production stores them in PostgreSQL.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

LEAD_FIELDS = [
    "full_name",
    "phone_number",
    "email",
    "company_name",
    "job_title",
    "city",
    "country",
    "preferred_language",
    "business_type",
    "product_or_service_interest",
    "business_requirement",
    "main_problem",
    "current_solution",
    "estimated_budget",
    "purchase_timeline",
    "decision_maker_status",
    "team_size",
    "preferred_contact_method",
    "preferred_contact_time",
    "additional_notes",
    "consent_to_contact",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    status: Mapped[str] = mapped_column(String(16), default="active")
    current_state: Mapped[str] = mapped_column(String(40), default="greeting")
    language: Mapped[str] = mapped_column(String(16), default="")
    skipped_fields: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    lead: Mapped["Lead | None"] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )
    provider_events: Mapped[list["ProviderEvent"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    audio_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stt_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tts_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_turn_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_provider_cost: Mapped[float] = mapped_column(Float, default=0.0)
    error_category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[Session] = relationship(back_populates="messages")


class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), unique=True, index=True
    )
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    phone_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    email: Mapped[str | None] = mapped_column(String(200), nullable=True)
    company_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    job_title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    city: Mapped[str | None] = mapped_column(String(200), nullable=True)
    country: Mapped[str | None] = mapped_column(String(120), nullable=True)
    preferred_language: Mapped[str | None] = mapped_column(String(64), nullable=True)
    business_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    product_or_service_interest: Mapped[str | None] = mapped_column(Text, nullable=True)
    business_requirement: Mapped[str | None] = mapped_column(Text, nullable=True)
    main_problem: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_solution: Mapped[str | None] = mapped_column(Text, nullable=True)
    estimated_budget: Mapped[str | None] = mapped_column(String(200), nullable=True)
    purchase_timeline: Mapped[str | None] = mapped_column(String(200), nullable=True)
    decision_maker_status: Mapped[str | None] = mapped_column(String(200), nullable=True)
    team_size: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preferred_contact_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preferred_contact_time: Mapped[str | None] = mapped_column(String(200), nullable=True)
    additional_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    consent_to_contact: Mapped[str | None] = mapped_column(String(16), nullable=True)

    qualification_score: Mapped[int] = mapped_column(Integer, default=0)
    qualification_level: Mapped[str] = mapped_column(String(16), default="cold")
    missing_important_fields: Mapped[list] = mapped_column(JSON, default=list)
    recommended_next_action: Mapped[str] = mapped_column(String(300), default="")
    conversation_status: Mapped[str] = mapped_column(String(20), default="in_progress")
    consent_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    summary_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    session: Mapped[Session] = relationship(back_populates="lead")
    field_history: Mapped[list["LeadFieldHistory"]] = relationship(
        back_populates="lead", cascade="all, delete-orphan"
    )


class LeadFieldHistory(Base):
    __tablename__ = "lead_field_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lead_id: Mapped[int] = mapped_column(
        ForeignKey("leads.id", ondelete="CASCADE"), index=True
    )
    field_name: Mapped[str] = mapped_column(String(64))
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    lead: Mapped[Lead] = relationship(back_populates="field_history")


class ProviderEvent(Base):
    __tablename__ = "provider_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    event_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="ok")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[Session | None] = relationship(back_populates="provider_events")
