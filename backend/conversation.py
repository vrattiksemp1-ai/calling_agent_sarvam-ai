"""Conversation engine.

The engine owns session lifecycle, lead field merging, corrections, consent
handling, deterministic scoring and completion criteria. The LLM only supplies
wording and field extraction; the backend controls everything structural.
"""

from datetime import datetime, timezone
from collections.abc import Awaitable, Callable

from sqlalchemy.orm import Session as OrmSession

from backend import prompts, scoring, state_machine, validation
from backend.call_profile import CallProfile, build_call_profile
from backend.config import Settings
from backend.errors import LlmStructuredOutputError
from backend.llm_parsing import (
    SAFE_FALLBACK_MESSAGE,
    parse_structured_response,
    parse_with_repair,
)
from backend.models import (
    LEAD_FIELDS,
    Lead,
    LeadFieldHistory,
    Message,
    ProviderEvent,
    Session,
)
from backend.language_utils import (
    infer_script_language,
    resolve_turn_language,
)
from backend.metrics import persist_turn_telemetry
from backend.prompts import REFUSAL_TOKEN
from backend.providers.llm_client import LlmClient
from backend.schemas import LeadOut
from backend.sentiment import rolling_transcript_style
from backend.streaming_json import AssistantMessageStreamParser
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
        *,
        call_profile: CallProfile | None = None,
        agent_name: str | None = None,
        disclose_ai_assistant: bool = True,
        settings: Settings | None = None,
    ):
        self._llm = llm_client
        self._settings = settings
        self._disclose_ai_assistant = disclose_ai_assistant
        if call_profile is not None:
            self._call_profile = call_profile
        elif settings is not None:
            self._call_profile = build_call_profile(settings)
        else:
            self._call_profile = CallProfile(
                agent_name=agent_name or "Shivangi",
                business_name=business_name or "Vrattiks",
                business_description=business_description
                or (
                    "a technology and software company focused on building "
                    "AI-powered solutions for businesses and individuals."
                ),
            )
        self._business_name = (
            business_name or self._call_profile.business_name or "Vrattiks"
        )
        self._business_description = (
            business_description or self._call_profile.business_description
        )
        self._agent_name = agent_name or self._call_profile.agent_name or "Shivangi"

    @property
    def call_profile(self) -> CallProfile:
        return self._call_profile

    def set_call_profile(self, profile: CallProfile) -> None:
        """Swap active profile (e.g. per-call lead overrides)."""
        self._call_profile = profile
        self._business_name = profile.business_name
        self._business_description = profile.business_description
        self._agent_name = profile.agent_name

    def _prompt_kwargs(self) -> dict:
        return {
            "business_name": self._business_name,
            "business_description": self._business_description,
            "agent_name": self._agent_name,
            "disclose_ai_assistant": self._disclose_ai_assistant,
            "call_profile": self._call_profile,
        }

    def _lead_fields(self, lead: Lead | None) -> dict[str, str]:
        if lead is None:
            return {}
        return {name: (getattr(lead, name) or "") for name in LEAD_FIELDS}

    def apply_known_lead_fields(self, lead: Lead, extra: dict[str, str] | None = None) -> None:
        """Prefill Lead columns from call profile / overrides without overwriting."""
        known = self._call_profile.lead.as_lead_fields()
        if extra:
            known.update({k: v for k, v in extra.items() if (v or "").strip()})
        for name, value in known.items():
            if name not in LEAD_FIELDS:
                continue
            current = getattr(lead, name, None)
            if not current and value:
                setattr(lead, name, value)
        if (self._call_profile.lead.field_note or "").strip():
            note = self._call_profile.lead.field_note.strip()
            existing = (lead.additional_notes or "").strip()
            marker = f"field_note: {note}"
            if marker not in existing:
                lead.additional_notes = f"{existing}; {marker}".strip("; ")

    def _normalize_greeting(self, text: str, language: str | None = None) -> str:
        """Return LLM greeting as-spoken. No hardcoded script prefixes.

        Persona / AI disclosure / time-check are enforced by the prompt +
        CALL PROFILE facts, not by stitching fixed sentences here.
        """
        del language  # kept for call-site compatibility
        return (text or "").strip()

    @staticmethod
    def _reply_language_mismatch(reply: str, expected: str) -> bool:
        """True when a pinned gu/hi turn got a Latin-only English-looking reply."""
        if not reply or not reply.strip():
            return False
        script = infer_script_language(reply)
        if script == expected:
            return False
        if script in {"gu", "hi"} and script != expected:
            return True
        latin = sum(1 for ch in reply if ("a" <= ch.lower() <= "z"))
        return script is None and latin >= 12

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
        language: str | None = None,
        on_assistant_chunk: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, int, dict, object | None]:
        """Run the LLM and return (raw, latency, usage, parsed-or-None)."""
        style_signal = rolling_transcript_style(history)
        if on_assistant_chunk is not None:
            timings.llm_attempt_count += 1
            messages = prompts.build_messages(
                history,
                self._lead_fields(lead),
                list(session.skipped_fields or []),
                session.current_state,
                repair=False,
                language=language,
                style_signal=style_signal,
                **self._prompt_kwargs(),
            )
            parser = AssistantMessageStreamParser()
            raw_parts: list[str] = []
            completion_ms = 0
            usage: dict = {}
            try:
                async for event in self._llm.stream_generate(
                    messages,
                    max_tokens=self._llm._settings.llm_max_tokens or 220,
                    max_retries=0,
                    reasoning_effort=self._llm._settings.phone_llm_reasoning_effort,
                ):
                    if event.type == "delta":
                        raw_parts.append(event.delta)
                        if "llm_first_token" not in timings.phase_elapsed_ms:
                            timings.mark("llm_first_token")
                        audible = parser.feed(event.delta)
                        if audible:
                            await on_assistant_chunk(audible)
                    elif event.type == "done":
                        completion_ms = event.completion_latency_ms or 0
                        usage = event.usage
                parser.finish()
            except (ValueError, TypeError) as exc:
                raise LlmStructuredOutputError(
                    "The streamed assistant response was incomplete.",
                    details=str(exc)[:300],
                ) from exc
            raw = "".join(raw_parts)
            parsed = parse_structured_response(raw)
            timings.llm_latency_ms = completion_ms
            timings.llm_usage = usage
            timings.mark("llm_completed")
            if parsed is None:
                # Speech may already be audible. Retrying here would speak a
                # second answer, so fail the turn without applying any state.
                raise LlmStructuredOutputError(
                    "The streamed assistant response was not valid structured JSON."
                )
            return parsed

        async def producer(repair: bool) -> str:
            if repair:
                timings.repair_count += 1
            timings.llm_attempt_count += 1
            messages = prompts.build_messages(
                history,
                self._lead_fields(lead),
                list(session.skipped_fields or []),
                session.current_state,
                repair=repair,
                language=language,
                style_signal=style_signal,
                **self._prompt_kwargs(),
            )
            # Phone turns need one fast shot: no provider retries, short replies.
            raw, latency, usage = await self._llm.generate(
                messages,
                max_tokens=self._llm._settings.llm_max_tokens or 220,
                max_retries=0,
                reasoning_effort=(
                    self._llm._settings.phone_llm_reasoning_effort
                    if getattr(timings, "transport", "api") != "api"
                    else None
                ),
            )
            timings.retry_count += max(
                0, getattr(self._llm, "last_attempt_count", 1) - 1
            )
            # A structured-output repair is a second provider call. Summing both
            # calls avoids reporting only the faster final attempt.
            timings.llm_latency_ms += latency
            timings.llm_usage = usage
            return raw

        parsed, error = await parse_with_repair(producer)
        timings.mark("llm_completed")
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
            if repair:
                timings.repair_count += 1
            timings.llm_attempt_count += 1
            messages = prompts.build_messages(
                [],
                self._call_profile.lead.as_lead_fields(),
                [],
                "greeting",
                repair=repair,
                language=language,
                **self._prompt_kwargs(),
            )
            raw, latency, usage = await self._llm.generate(
                messages,
                reasoning_effort=self._llm._settings.phone_llm_reasoning_effort,
            )
            timings.retry_count += max(
                0, getattr(self._llm, "last_attempt_count", 1) - 1
            )
            timings.llm_latency_ms += latency
            timings.llm_usage = usage
            return raw

        parsed, error = await parse_with_repair(producer)
        timings.mark("llm_completed")
        if parsed is None:
            raise LlmStructuredOutputError(
                "The assistant could not produce a structured greeting.",
                details=error,
            )
        return (
            self._normalize_greeting(parsed.assistant_message or "", language),
            timings.llm_latency_ms or 0,
            timings.llm_usage or {},
        )

    async def process_turn(
        self,
        db: OrmSession,
        session: Session,
        user_text: str,
        timings,
        stt_language: str | None = None,
        on_assistant_chunk: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[Lead, object]:
        timings.stt_language = stt_language
        if session.status == "abandoned":
            lead = session.lead
            return lead, self._abandoned_response(session, lead)

        lead = session.lead
        if lead is None:
            lead = Lead(session_id=session.id)
            self.apply_known_lead_fields(lead)
            db.add(lead)
            db.flush()

        history = [
            {"role": m.role, "content": m.content}
            for m in db.query(Message)
            .filter(Message.session_id == session.id)
            .order_by(Message.id)
        ]
        history.append({"role": "user", "content": user_text})

        # Resolve a hard language pin when we have a clear signal; otherwise let
        # the LLM follow the latest utterance (important for mid-call switches).
        prior_language = (session.language or "").strip().lower() or None
        turn_language = resolve_turn_language(
            user_text,
            prior_language=prior_language,
            stt_language=stt_language,
        )
        timings.language_expected = turn_language or prior_language
        llm_kwargs = {"language": turn_language}
        # Avoid changing the invocation shape for existing subclasses and test
        # doubles that override the established buffered _llm_turn signature.
        if on_assistant_chunk is not None:
            llm_kwargs["on_assistant_chunk"] = on_assistant_chunk
        parsed = await self._llm_turn(
            db, session, lead, user_text, history, timings, **llm_kwargs
        )

        # Prefer the hard pin when present. Otherwise trust the model, then the
        # script of its reply, then keep the prior language.
        model_detected = (
            getattr(parsed, "detected_language", None) or ""
        ).strip().lower() or None
        reply_script = infer_script_language(parsed.assistant_message or "")
        timings.language_detected = model_detected
        timings.reply_script = reply_script
        if turn_language in {"gu", "hi", "en"}:
            session.language = turn_language
            parsed.detected_language = turn_language
        elif model_detected in {"gu", "hi", "en", "en-hi"}:
            session.language = "hi" if model_detected == "en-hi" else model_detected
        elif reply_script:
            session.language = reply_script
            parsed.detected_language = reply_script
        elif prior_language:
            session.language = prior_language
            parsed.detected_language = prior_language

        # If we hard-pinned gu/hi from Indic script but the model answered in
        # English, that often means phone ASR wrote English phonetics in
        # Gujarati/Hindi letters. Trust an explicit English detection from the
        # model (real switch). Only use a fast fallback when the model also
        # claimed gu/hi but wrote Latin English.
        if turn_language in {"gu", "hi"} and self._reply_language_mismatch(
            parsed.assistant_message or "", turn_language
        ):
            timings.language_mismatch = True
            if model_detected in {"en", "en-hi"}:
                logger.info(
                    "Indic-script ASR with English reply; switching session to en"
                )
                session.language = "en"
                parsed.detected_language = "en"
                timings.language_repair = "model_language_switch"
            else:
                logger.warning(
                    "LLM reply language mismatched pin=%s; using fast fallback",
                    turn_language,
                )
                from backend.telephony.call_manager import fallback_text

                parsed.assistant_message = fallback_text("repeat", turn_language)
                parsed.detected_language = turn_language
                session.language = turn_language
                timings.language_repair = "localized_repeat_fallback"
                timings.fallback_count += 1

        # Soft anti-repeat: if the model pasted the previous assistant line,
        # force a short clarification instead of playing the same audio again.
        if history:
            last_assistant = next(
                (
                    m["content"]
                    for m in reversed(history)
                    if m.get("role") == "assistant" and m.get("content")
                ),
                None,
            )
            current = (parsed.assistant_message or "").strip()
            if (
                last_assistant
                and current
                and current == last_assistant.strip()
            ):
                from backend.telephony.call_manager import fallback_text

                lang = session.language or turn_language or "en"
                parsed.assistant_message = fallback_text("repeat", lang)

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
        timings.language_detected = (
            getattr(parsed, "detected_language", None) or session.language or None
        )
        persist_turn_telemetry(db, session.id, timings)

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
