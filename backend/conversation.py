"""Conversation engine.

The engine owns session lifecycle, lead field merging, corrections, consent
handling, deterministic scoring and completion criteria. The LLM only supplies
wording and field extraction; the backend controls everything structural.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session as OrmSession

from backend import prompts, scoring, state_machine, validation
from backend.errors import LlmStructuredOutputError
from backend.llm_parsing import SAFE_FALLBACK_MESSAGE, parse_with_repair
from backend.models import (
    LEAD_FIELDS,
    Lead,
    LeadFieldHistory,
    Message,
    ProviderEvent,
    Session,
)
from backend.prompts import REFUSAL_TOKEN
from backend.providers.llm_client import LlmClient
from backend.schemas import LeadOut
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationEngine:
    def __init__(
        self,
        llm_client: LlmClient,
        business_name: str | None = None,
        business_description: str | None = None,
    ):
        self._llm = llm_client
        self._business_name = business_name
        self._business_description = business_description

    def _lead_fields(self, lead: Lead | None) -> dict[str, str]:
        if lead is None:
            return {}
        return {name: (getattr(lead, name) or "") for name in LEAD_FIELDS}

    def _apply_extraction(
        self, db: OrmSession, lead: Lead, parsed, session: Session
    ) -> None:
        skipped = list(session.skipped_fields or [])
        for name in parsed.fields_to_clear:
            if name in LEAD_FIELDS:
                setattr(lead, name, None)
        for name, raw_value in (parsed.extracted_fields or {}).items():
            if name not in LEAD_FIELDS:
                continue
            value = str(raw_value).strip()
            if value.lower() in {REFUSAL_TOKEN, "__refused__", "refused", "__skip__"}:
                setattr(lead, name, None)
                if name not in skipped:
                    skipped.append(name)
                continue
            if value in {"", "null", "None", "none", "pending"}:
                continue
            old_value = getattr(lead, name)
            if old_value and old_value != value:
                db.add(
                    LeadFieldHistory(
                        lead_id=lead.id,
                        field_name=name,
                        old_value=str(old_value),
                        new_value=value,
                    )
                )
            setattr(lead, name, value)
            if name in skipped:
                skipped = [s for s in skipped if s != name]
        session.skipped_fields = list(dict.fromkeys(skipped))
        lead.raw_json = self._lead_fields(lead)
        db.add(session)
        db.add(lead)

    def _record_provider_event(
        self, db: OrmSession, session_id: str | None, event_type: str, status: str, latency_ms: int | None, error_code: str | None = None, request_id: str | None = None
    ) -> None:
        db.add(
            ProviderEvent(
                session_id=session_id,
                provider="llm",
                event_type=event_type,
                status=status,
                latency_ms=latency_ms,
                error_code=error_code,
                request_id=request_id,
            )
        )

    async def _llm_turn(
        self,
        db: OrmSession,
        session: Session,
        lead: Lead | None,
        user_text: str,
        history: list[dict],
        timings,
    ) -> tuple[str, int, dict, object | None]:
        """Run the LLM and return (raw, latency, usage, parsed-or-None)."""
        async def producer(repair: bool) -> str:
            messages = prompts.build_messages(
                history,
                self._lead_fields(lead),
                list(session.skipped_fields or []),
                session.current_state,
                repair=repair,
                business_name=self._business_name,
                business_description=self._business_description,
            )
            raw, latency, usage = await self._llm.generate(messages)
            timings.llm_latency_ms = latency
            timings.llm_usage = usage
            return raw

        parsed, error = await parse_with_repair(producer)
        if parsed is None:
            raise LlmStructuredOutputError(
                "The assistant could not produce a structured reply.",
                details=error,
            )
        return parsed

    async def generate_greeting(
        self, timings, language: str | None = None
    ) -> tuple[str, int, dict]:
        """Produce the opening line via the LLM before any caller input.

        Previously the call opened with the same hard-coded GREETING verbatim on
        every call. This generates a natural, varied opening using the same
        system prompt (state = "greeting", no history). Returns
        (greeting_text, latency_ms, usage).
        """
        async def producer(repair: bool) -> str:
            messages = prompts.build_messages(
                [], {}, [], "greeting", repair=repair, language=language,
                business_name=self._business_name,
                business_description=self._business_description,
            )
            raw, latency, usage = await self._llm.generate(messages)
            timings.llm_latency_ms = latency
            timings.llm_usage = usage
            return raw

        parsed, error = await parse_with_repair(producer)
        if parsed is None:
            raise LlmStructuredOutputError(
                "The assistant could not produce a structured greeting.",
                details=error,
            )
        return (
            parsed.assistant_message or "",
            timings.llm_latency_ms or 0,
            timings.llm_usage or {},
        )

    async def process_turn(
        self,
        db: OrmSession,
        session: Session,
        user_text: str,
        timings,
    ) -> tuple[Lead, object]:
        if session.status == "abandoned":
            lead = session.lead
            return lead, self._abandoned_response(session, lead)

        lead = session.lead
        if lead is None:
            lead = Lead(session_id=session.id)
            db.add(lead)
            db.flush()

        history = [
            {"role": m.role, "content": m.content}
            for m in db.query(Message)
            .filter(Message.session_id == session.id)
            .order_by(Message.id)
        ]
        history.append({"role": "user", "content": user_text})

        parsed = await self._llm_turn(db, session, lead, user_text, history, timings)

        if getattr(parsed, "detected_language", None):
            session.language = parsed.detected_language

        if session.current_state == "greeting":
            session.current_state = "collecting_identity"

        self._apply_extraction(db, lead, parsed, session)
        db.flush()

        fields = self._lead_fields(lead)
        result = scoring.score_lead(fields)

        suggested = parsed.next_state or session.current_state
        transition = state_machine.transition_allowed(
            session.current_state, suggested, fields
        )
        if transition.allowed:
            session.current_state = transition.state
        else:
            if suggested in state_machine.TERMINAL:
                session.current_state = state_machine.default_next_state(
                    session.current_state, fields
                )
            else:
                session.current_state = transition.state

        consent_text = fields.get("consent_to_contact") or ""
        if validation.consent_bool(consent_text):
            if not lead.consent_confirmed:
                lead.consent_confirmed = True
                db.add(lead)
        elif validation.is_consent_value(consent_text):
            if lead.consent_confirmed:
                lead.consent_confirmed = False
                db.add(lead)

        if session.current_state == "requesting_consent":
            if validation.consent_bool(consent_text) or validation.is_consent_value(consent_text):
                transition2 = state_machine.transition_allowed(
                    "requesting_consent", "reviewing_summary", fields
                )
                if transition2.allowed:
                    session.current_state = "reviewing_summary"

        if parsed.conversation_complete:
            if session.current_state == "completed":
                session.status = "completed"
                lead.conversation_status = "completed"
                lead.summary_confirmed = True
            else:
                transition = state_machine.transition_allowed(
                    session.current_state, "completed", fields
                )
                if transition.allowed:
                    session.current_state = "completed"
                    session.status = "completed"
                    lead.conversation_status = "completed"
                    lead.summary_confirmed = True

        if session.current_state == "abandoned":
            session.status = "abandoned"
            lead.conversation_status = "abandoned"
            db.add(lead)

        lead.qualification_score = result.score
        lead.qualification_level = result.level
        lead.missing_important_fields = result.missing_important_fields
        lead.recommended_next_action = result.recommended_next_action
        db.add(lead)

        self._record_provider_event(
            db, session.id, "llm", "ok", timings.llm_latency_ms or 0
        )

        db.add(
            Message(
                session_id=session.id,
                role="user",
                content=user_text,
                audio_duration_ms=timings.audio_duration_ms,
                stt_latency_ms=timings.stt_latency_ms,
                error_category="stt" if timings.stt_latency_ms == 0 and timings.audio_duration_ms else None,
            )
        )
        db.add(
            Message(
                session_id=session.id,
                role="assistant",
                content=parsed.assistant_message,
                llm_latency_ms=timings.llm_latency_ms,
                total_turn_latency_ms=timings.total(),
                estimated_provider_cost=timings.llm_cost(),
            )
        )
        db.add(session)
        return lead, parsed

    async def handle_confirmation(
        self, db: OrmSession, session: Session, confirmed: bool, corrections: str | None
    ) -> tuple[Lead, str, dict]:
        lead = session.lead
        if lead is None or session.current_state not in {
            "reviewing_summary",
            "requesting_consent",
            "completed",
        }:
            return lead, SAFE_FALLBACK_MESSAGE, {}

        if confirmed:
            if validation.consent_bool(lead.consent_to_contact) and not lead.consent_confirmed:
                lead.consent_confirmed = True
            lead.conversation_status = "completed"
            lead.summary_confirmed = True
            session.current_state = "completed"
            session.status = "completed"
            fields = self._lead_fields(lead)
            result = scoring.score_lead(fields)
            lead.qualification_score = result.score
            lead.qualification_level = result.level
            lead.missing_important_fields = result.missing_important_fields
            lead.recommended_next_action = result.recommended_next_action
            db.add(lead)
            db.add(session)
            return (
                lead,
                "Thanks for confirming. Your details are saved. Have a good day!",
                {"conversation_status": "completed"},
            )

        db.add(
            Message(
                session_id=session.id,
                role="user",
                content=corrections or "I need to correct something.",
                error_category="confirmation",
            )
        )
        return lead, "Sure, tell me what to correct.", {"needs_correction": True}

    def summarize_session(self, db: OrmSession, session: Session) -> dict:
        lead = session.lead
        fields = self._lead_fields(lead)
        messages = (
            db.query(Message)
            .filter(Message.session_id == session.id)
            .order_by(Message.id)
            .all()
        )
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        turn_count = len(assistant_msgs)
        avg_latency = 0
        total_cost = 0.0
        latencies = [m.total_turn_latency_ms for m in assistant_msgs if m.total_turn_latency_ms]
        if latencies:
            avg_latency = sum(latencies) // len(latencies)
        total_cost = sum(m.estimated_provider_cost or 0 for m in messages)
        result = scoring.score_lead(fields)
        collected = sum(1 for v in fields.values() if v and v.strip())
        return {
            "session_id": session.id,
            "turn_count": turn_count,
            "average_latency_ms": avg_latency,
            "completion_status": session.status,
            "qualification_score": lead.qualification_score if lead else result.score,
            "qualification_level": lead.qualification_level if lead else result.level,
            "collected_field_count": collected,
            "missing_important_fields": result.missing_important_fields,
            "estimated_provider_cost": round(total_cost, 4),
        }

    def _abandoned_response(self, session: Session, lead: Lead | None) -> object:
        class _Simple:
            assistant_message = "This conversation was closed."
            detected_language = session.language or "en"
            extracted_fields = {}
            fields_to_clear = []
            next_state = "abandoned"
            conversation_complete = False
            needs_confirmation = False

        return _Simple()

    def to_lead_out(self, lead: Lead | None, session: Session) -> LeadOut:
        if lead is None:
            return LeadOut(session_id=session.id)
        return LeadOut(
            id=lead.id,
            session_id=session.id,
            fields=self._lead_fields(lead),
            skipped_fields=list(session.skipped_fields or []),
            qualification_score=lead.qualification_score,
            qualification_level=lead.qualification_level,
            missing_important_fields=list(lead.missing_important_fields or []),
            recommended_next_action=lead.recommended_next_action,
            conversation_status=lead.conversation_status,
            consent_confirmed=lead.consent_confirmed,
            summary_confirmed=lead.summary_confirmed,
        )
