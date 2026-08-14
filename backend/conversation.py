"""Conversation engine.

The engine owns session lifecycle, lead field merging, corrections, consent
handling, deterministic scoring and completion criteria. The LLM only supplies
wording and field extraction; the backend controls everything structural.
"""

import re
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
    detect_explicit_language_switch,
    infer_script_language,
    resolve_turn_language,
)
from backend.metrics import persist_turn_telemetry
from backend.pipeline_trace import trace
from backend.prompts import REFUSAL_TOKEN
from backend.providers.llm_client import LlmClient
from backend.schemas import LeadOut
from backend.sentiment import rolling_transcript_style
from backend.streaming_json import AssistantMessageStreamParser
from backend.tts_persona import resolve_tts_gender
from backend.utils.logging import get_logger

logger = get_logger(__name__)

_NAME_PATTERNS = (
    re.compile(
        r"(?:my\s+name\s+is|i\s+am|i'm|this\s+is)\s+([A-Za-z][A-Za-z.'\-]{1,40})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:માય\s*નેમ\s*ઇસ|મારું\s*નામ|મારુ\s*નામ)\s*([^\s,?.!]{2,40})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:मेरा\s*नाम|मेरा\s*नेम)\s*(?:है\s*)?([^\s,?.!]{2,40})",
        re.IGNORECASE,
    ),
)
_AVAIL_YES = re.compile(
    r"\b(yes|yeah|yep|sure|ok|okay|haan|haa|હા|हां|બોલો|bolo)\b",
    re.IGNORECASE,
)
_SOURCE_Q = re.compile(
    r"(where\s+are\s+you\s+calling|where\s+you\s+calling|where\s+do\s+you\s+work|where\s+from|"
    r"કોલિંગ\s*સોંગ|કોલિંગ\s*ફ્રોમ|ક્યાંથી|ક્યાં\s*કામ|कहां\s*से\s*कॉल|कहां\s*काम)",
    re.IGNORECASE,
)
_SERVICE_Q = re.compile(
    r"(service|services|product|products|provide|સર્વિસ|પ્રોવાઇડ|प्रोडक्ट|सेवा)",
    re.IGNORECASE,
)
_INTRO_RE = re.compile(
    r"(my name is|"
    r"this is\s+\w+\s+from|"
    r"i'?m\s+(?!calling\b)(?:an?\s+)?(?:ai\s+assistant\s+)?\w+\s+from|"
    r"ai assistant from|"
    r"હું\s+.+\s+છું|બોલું\s*છું|માંથી\s*.{0,24}\s*બોલ|"
    r"do you have (a )?(moment|time|couple)|થોડો\s*સમય|થોડીવાર|થોડી\s*મિનિટ|વ્યસ્ત\s*છો|"
    r"मेरा\s*नाम|बोल\s*रही|समय\s*है|व्यस्त)",
    re.IGNORECASE,
)
_CONTINUATION_RE = re.compile(
    r"\bi'?m calling (from|because)\b|"
    r"\bnice to meet you\b|"
    r"\bwe (help|offer|build|provide)\b|"
    r"\bdo you (currently )?use\b|"
    r"lead qualification|software for|ai solutions|ai-powered tools|"
    r"biggest challenge|current setup",
    re.IGNORECASE,
)
# Common bad ASR/LLM Gujarati spellings for persona names.
_NAME_SPELLING_FIXES = (
    (re.compile(r"શિવંગી"), "શિવાંગી"),
    (re.compile(r"શિવાગી"), "શિવાંગી"),
    (re.compile(r"વ્રટિક્સ"), "વ્રત્તિક્સ"),
    (re.compile(r"વૃત્તાંતિક્સ"), "વ્રત્તિક્સ"),
    (re.compile(r"વૃત્તિક્સ"), "વ્રત્તિક્સ"),
)
_TIME_ASK_RE = re.compile(
    r"(do you have|couple of minutes|few minutes|a moment|થોડી?\s*મિનિટ|થોડો\s*સમય|થોડીવાર|વ્યસ્ત\s*છો|समय\s*(है|हो)|व्यस्त)",
    re.IGNORECASE,
)
_EMAIL_MENTION = re.compile(r"\b(email|e-mail|mail\s*id|ઇમેઇલ|ईमेल)\b", re.I)
_REFUSE_MENTION = re.compile(
    r"\b(no|not|don'?t|refuse|skip|nahi|ના|नहीं)\b", re.I
)
_ASKING_RE = re.compile(
    r"(say\s+that\s+again|say\s+again|repeat\s+that|didn't\s+catch|"
    r"સમજાયું\s*નહીં|ફરી\s*(એક\s*)?વાર|"
    r"समझ\s*नहीं|दोबारा|"
    r"કરી\s*શકું|कर\s*सकूं|shall\s+i|can\s+i\s+call)",
    re.I,
)
_AGENT_NAME_Q = re.compile(
    r"(your\s+name|what\s+is\s+your\s+name|who\s+are\s+you|may\s+i\s+know\s+your\s+name|"
    r"आप(?:का|की)?\s*नाम|आप\s*कौन|तुम(?:्हार)?ा\s*नाम|"
    r"તમારું\s*નામ|તમે\s*કોણ)",
    re.IGNORECASE,
)
_MIDCALL_INDIC_REINTRO = re.compile(
    r"હું\s+.{0,80}(માંથી|થી).{0,50}છું|"
    r"मैं\s+.{0,80}से\s+.{0,50}(हूँ|हूं|हू)",
    re.IGNORECASE,
)
_EXPLAIN_REQ_RE = re.compile(
    r"(आप\s*बताइए|आप\s*बताओ|बताइए|बताओ|explain|tell\s+me\s+more|"
    r"how\s+does\s+(it|this)\s+work|kaise\s+kaam|કામ\s*કરે)",
    re.IGNORECASE,
)
_VAGUE_CONTINUE_RE = re.compile(
    r"(मैं\s*आगे\s*बताऊँ|मैं\s*आगे\s*बताऊं|"
    r"ટૂંકમાં\s*આગળ\s*કહું|આગળ\s*કહું|"
    r"shall\s+i\s+continue|continue\s+with\s+a\s+quick)",
    re.IGNORECASE,
)
_ONLY_CALLING_Q = re.compile(
    r"(only|sirf|સિર્ફ|सिर्फ|just).{0,24}(calling|કોલ|कॉल)",
    re.IGNORECASE,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def infer_full_name_from_text(text: str | None) -> str | None:
    """Best-effort name salvage when the LLM forgets extracted_fields.full_name."""
    if not text or not text.strip():
        return None
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text.strip())
        if not match:
            continue
        name = (match.group(1) or "").strip(" .,!?:;\"'")
        if len(name) < 2:
            continue
        # Drop trailing filler words from Gu/En ASR.
        for stop in (" છે", " है", " chhe", " hai"):
            if name.lower().endswith(stop.strip().lower()):
                name = name[: -len(stop)].strip()
        if name.lower() in {"yes", "no", "ok", "okay", "haan", "હા"}:
            continue
        return name
    return None


def looks_like_source_question(text: str | None) -> bool:
    return bool(text and _SOURCE_Q.search(text))


def looks_like_service_question(text: str | None) -> bool:
    return bool(text and _SERVICE_Q.search(text))


def looks_like_only_calling_question(text: str | None) -> bool:
    return bool(text and _ONLY_CALLING_Q.search(text))


def looks_like_explain_request(text: str | None) -> bool:
    return bool(text and _EXPLAIN_REQ_RE.search(text))


def looks_like_user_confirmation(text: str | None) -> bool:
    return bool(text and _AVAIL_YES.search(text))


def looks_like_vague_continue_reply(text: str | None) -> bool:
    return bool(text and _VAGUE_CONTINUE_RE.search(text))


def looks_like_agent_name_question(text: str | None) -> bool:
    return bool(text and _AGENT_NAME_Q.search(text))


def looks_like_reintroduction(
    text: str | None,
    *,
    agent_name: str = "",
    business_name: str = "",
    user_text: str | None = None,
) -> bool:
    """True when the reply looks like a fresh opening / time-check restart.

    Mentions of the company while continuing a pitch ("I'm calling from…",
    "nice to meet you…", product questions) are NOT treated as re-intros.
    """
    if not text or not text.strip():
        return False
    t = text.strip()
    tl = t.lower()
    agent = (agent_name or "").strip().lower()
    biz = (business_name or "").strip().lower()

    # Brief first-person source answer is allowed when caller asked where from.
    if looks_like_source_question(user_text):
        if re.search(r"ફોન|કોલ|calling|कॉल|कॉल कर", t, re.IGNORECASE):
            if not _TIME_ASK_RE.search(t):
                if not re.search(
                    r"બોલું\s*છું|मेरा\s*नाम|my name is|this is",
                    t,
                    re.IGNORECASE,
                ):
                    return False

    self_intro = bool(_INTRO_RE.search(t))
    if agent and agent in tl and re.search(
        r"\b(my name is|this is)\b|"
        r"\bi'?m\s+(?!calling\b)",
        tl,
    ):
        self_intro = True
    if re.search(
        r"બોલું\s*છું|માંથી\s*.{0,40}\s*બોલ|ફોન\s*કરી\s*રહી|"
        r"बोल\s*रही\s*हूँ|मेरा\s*नाम",
        t,
        re.IGNORECASE,
    ):
        self_intro = True
    if "ai assistant from" in tl or "ai assistant," in tl:
        self_intro = True

    if not self_intro:
        return False

    # Continuing the sales flow while naming the company is allowed.
    if _CONTINUATION_RE.search(t) and not _TIME_ASK_RE.search(t):
        return False

    if _TIME_ASK_RE.search(t):
        return True
    # Dense opening restart without an explicit time-check.
    if self_intro and (biz and biz in tl) and re.search(
        r"\b(my name is|this is|ai assistant)\b|"
        r"બોલું\s*છું|मेरा\s*नाम",
        tl,
    ):
        return True
    if _MIDCALL_INDIC_REINTRO.search(t):
        if _CONTINUATION_RE.search(t) and not _TIME_ASK_RE.search(t):
            return False
        return True
    return False


def normalize_spoken_names(text: str | None, language: str | None = None) -> str:
    """Rewrite common bad Gu/Hi spellings of agent/company names for TTS."""
    if not text:
        return ""
    out = text
    lang = (language or "").strip().lower()
    has_gu = any("઀" <= ch <= "૿" for ch in out)
    if lang.startswith("gu") or lang == "gujlish" or has_gu:
        for pattern, repl in _NAME_SPELLING_FIXES:
            out = pattern.sub(repl, out)
    return out


def _suggest_next_step_hint(current_state: str, fields: dict) -> str | None:
    """State-driven fallback focus after the caller's latest intent is handled."""
    state = (current_state or "").strip().lower()
    if state == "greeting":
        return "Opening: intro, availability check, then name if needed."
    if state == "collecting_identity":
        if not (fields.get("full_name") or "").strip():
            return "Collect or confirm caller name if still unknown."
        return "Name known — move to business context or requirements."
    if state in {"collecting_contact", "collecting_business_context"}:
        return "Light context: business type, company, or what they need."
    if state == "collecting_requirement":
        return "Discovery: their need, questions, or custom requirements from CALL PROFILE."
    if state in {
        "collecting_budget",
        "collecting_timeline",
        "collecting_authority",
        "collecting_preferences",
    }:
        return "Only if natural: budget, timeline, decision role, or contact preferences."
    if state == "requesting_consent":
        return "Ask explicit consent to follow up."
    if state == "reviewing_summary":
        return "Summarize collected facts and confirm."
    return None


def _history_confirms_availability(history: list[dict]) -> bool:
    """True only when a yes-like reply directly followed an availability check."""
    for index, message in enumerate(history[:-1]):
        if message.get("role") != "assistant":
            continue
        if not _TIME_ASK_RE.search(message.get("content") or ""):
            continue
        next_message = history[index + 1]
        if next_message.get("role") == "user" and _AVAIL_YES.search(
            next_message.get("content") or ""
        ):
            return True
    return False


def build_progress_hints(
    history: list[dict],
    fields: dict,
    user_text: str | None,
    *,
    agent_name: str,
    business_name: str,
    current_state: str = "",
    call_profile: CallProfile | None = None,
) -> str:
    """State- and history-driven reminders so the model does not restart the call."""
    hints: list[str] = []
    assistant_turns = [
        m.get("content") or ""
        for m in history
        if m.get("role") == "assistant" and (m.get("content") or "").strip()
    ]
    last_assistant = assistant_turns[-1] if assistant_turns else ""
    latest_is_confirmation = looks_like_user_confirmation(user_text)

    hints.append(
        "Decision order for THIS turn: first answer the caller's latest intent; "
        "then acknowledge any supplied information; only then use the current "
        "state and missing fields to choose one natural next question. The state "
        "is context, not a rigid script or mandatory sequence."
    )
    step = _suggest_next_step_hint(current_state, fields)
    if step:
        hints.append(
            f"Fallback focus after resolving the latest message "
            f"({current_state or 'in progress'}): {step}"
        )

    if assistant_turns and current_state != "greeting":
        hints.append(
            f"The call is already in progress with {agent_name} from {business_name}. "
            "Do not produce another opening, availability check, or company introduction. "
            "Mention identity/company only when directly asked or needed to answer."
        )
    if _history_confirms_availability(history):
        hints.append(
            "Availability was already confirmed in an earlier adjacent question/answer. "
            "Do not check availability again."
        )
    if latest_is_confirmation and last_assistant:
        hints.append(
            "The latest caller message confirms the immediately preceding assistant "
            "turn. Treat the pending action or question as accepted, carry it out now, "
            "and do not repeat or rephrase the same permission question."
        )
    if last_assistant and looks_like_vague_continue_reply(last_assistant):
        hints.append(
            "The preceding assistant turn asked only for permission to continue. "
            "Do not ask permission again; provide the pending information or choose "
            "one concrete next question from the latest intent, state, and missing fields."
        )
    if (fields.get("full_name") or "").strip():
        fn = fields["full_name"].strip()
        if fn.lower() not in {"__pending__", "pending", "__refused__", "refused"}:
            hints.append(
                f"Caller's name is already known: {fn}. Use it; do not re-ask. "
                "Do not repeat a full greeting + company + product pitch."
            )
    elif infer_full_name_from_text(user_text):
        hints.append(
            "Caller just gave their name in this utterance — set extracted_fields.full_name."
        )
    if looks_like_source_question(user_text):
        hints.append(
            f"Latest intent is a source/identity question. Answer it directly in first "
            f"person using {business_name} and CALL PROFILE facts. Do not replay the opening."
        )
    if looks_like_explain_request(user_text):
        hints.append(
            "Latest intent requests an explanation. Explain now using only CALL PROFILE "
            "products_and_services and their configured summaries. Do not ask whether "
            "the caller wants the explanation they just requested."
        )
    if looks_like_service_question(user_text) or looks_like_only_calling_question(
        user_text
    ):
        hints.append(
            "Latest intent asks about product/service scope. Answer from the complete "
            "CALL PROFILE products_and_services list; do not narrow the answer to one "
            "offering unless the caller did."
        )
    if detect_explicit_language_switch(user_text) == "en":
        hints.append(
            "Caller asked to speak English. Switch to English THIS turn and continue "
            "the same call — do not re-introduce yourself."
        )
    switch = detect_explicit_language_switch(user_text)
    if switch == "hi":
        hints.append(
            "The latest intent changes the language to Hindi. Continue the same turn "
            "and same topic in Hindi; do not restart any earlier call stage."
        )
    elif switch == "gu":
        hints.append(
            "The latest intent changes the language to Gujarati. Continue the same turn "
            "and same topic in Gujarati; do not restart any earlier call stage."
        )
    if looks_like_agent_name_question(user_text):
        hints.append(
            f"Caller asked your name. Answer briefly: you are {agent_name} from "
            f"{business_name}. Do not restart the opening pitch."
        )
    if call_profile and call_profile.products_and_services:
        names = ", ".join(p.name for p in call_profile.products_and_services[:4])
        hints.append(
            f"Configured product/service names available for factual answers: {names}."
        )
    return "\n".join(f"- {h}" for h in hints)


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

    def _prompt_kwargs(self, *, compact: bool = False) -> dict:
        speaker = "ritu"
        gender_override = ""
        if self._settings is not None:
            speaker = (self._settings.sarvam_tts_speaker or "ritu").strip() or "ritu"
            gender_override = self._settings.sarvam_tts_speaker_gender or ""

        return {
            "business_name": self._business_name,
            "business_description": self._business_description,
            "agent_name": self._agent_name,
            "disclose_ai_assistant": self._disclose_ai_assistant,
            "call_profile": self._call_profile,
            "compact": compact,
            "tts_speaker": speaker,
            "tts_gender": resolve_tts_gender(speaker, gender_override),
        }

    @staticmethod
    def _is_phone_transport(timings) -> bool:
        transport = (getattr(timings, "transport", None) or "api").strip().lower()
        return transport not in {"", "api", "browser"}

    def _phone_llm_kwargs(self, timings) -> dict:
        """Model/token/temp overrides for Gather / Media Streams turns."""
        settings = self._llm._settings
        if not self._is_phone_transport(timings):
            return {
                "max_tokens": settings.llm_max_tokens or 220,
                "reasoning_effort": None,
            }
        max_tokens = settings.phone_llm_max_tokens or settings.llm_max_tokens or 140
        kwargs: dict = {
            "max_tokens": max_tokens,
            "reasoning_effort": settings.phone_llm_reasoning_effort,
            "temperature": settings.phone_llm_temperature,
        }
        phone_model = (settings.phone_llm_model or "").strip()
        if phone_model:
            kwargs["model"] = phone_model
        return kwargs

    def _use_compact_prompt(self, timings) -> bool:
        settings = self._llm._settings
        return bool(
            self._is_phone_transport(timings)
            and getattr(settings, "phone_prompt_compact", True)
        )

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
        self,
        db: OrmSession,
        lead: Lead,
        parsed,
        session: Session,
        *,
        user_text: str | None = None,
    ) -> None:
        skipped = list(session.skipped_fields or [])
        extracted = dict(parsed.extracted_fields or {})

        # Salvage name when the model spoke it but forgot extracted_fields.
        if not (extracted.get("full_name") or getattr(lead, "full_name", None)):
            inferred = infer_full_name_from_text(user_text)
            if inferred:
                extracted["full_name"] = inferred
                parsed.extracted_fields = extracted

        # Drop bogus email refusals (e.g. "call me later" mis-tagged as refuse email).
        email_val = str(extracted.get("email") or "").strip().lower()
        if email_val in {REFUSAL_TOKEN, "__refused__", "refused", "__skip__"}:
            ut = user_text or ""
            if not (_EMAIL_MENTION.search(ut) and _REFUSE_MENTION.search(ut)):
                extracted.pop("email", None)
                parsed.extracted_fields = extracted
                logger.info("Dropped bogus email refusal from extraction")

        for name in parsed.fields_to_clear:
            if name in LEAD_FIELDS:
                setattr(lead, name, None)
        for name, raw_value in extracted.items():
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
            if value.lower() in {"__pending__", "__skip__"}:
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
        compact = self._use_compact_prompt(timings)
        phone_kwargs = self._phone_llm_kwargs(timings)
        progress_hints = build_progress_hints(
            history,
            self._lead_fields(lead),
            user_text,
            agent_name=self._call_profile.agent_name,
            business_name=self._call_profile.business_name,
            current_state=session.current_state,
            call_profile=self._call_profile,
        )
        if on_assistant_chunk is not None:
            timings.llm_attempt_count += 1
            messages = prompts.build_messages(
                history,
                self._lead_fields(lead),
                list(session.skipped_fields or []),
                session.current_state,
                language=language,
                style_signal=None if compact else style_signal,
                progress_hints=progress_hints,
                **self._prompt_kwargs(compact=compact),
            )
            parser = AssistantMessageStreamParser()
            raw_parts: list[str] = []
            completion_ms = 0
            usage: dict = {}
            try:
                async for event in self._llm.stream_generate(
                    messages,
                    max_retries=0,
                    **phone_kwargs,
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

        timings.llm_attempt_count += 1
        messages = prompts.build_messages(
            history,
            self._lead_fields(lead),
            list(session.skipped_fields or []),
            session.current_state,
            language=language,
            style_signal=None if compact else style_signal,
            progress_hints=progress_hints,
            **self._prompt_kwargs(compact=compact),
        )
        gen_kwargs = dict(phone_kwargs)
        if not self._is_phone_transport(timings):
            gen_kwargs = {
                "max_tokens": self._llm._settings.llm_max_tokens or 220,
                "reasoning_effort": None,
            }
        raw, latency, usage = await self._llm.generate(
            messages,
            max_retries=0,
            **gen_kwargs,
        )
        timings.llm_latency_ms += latency
        timings.llm_usage = usage
        parsed = parse_structured_response(raw)
        timings.mark("llm_completed")
        if parsed is None:
            raise LlmStructuredOutputError(
                "The assistant could not produce a structured reply.",
                details="Single-attempt LLM response was not valid structured JSON.",
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
        timings.llm_attempt_count += 1
        compact = bool(
            getattr(self._llm._settings, "phone_prompt_compact", True)
        )
        messages = prompts.build_messages(
            [],
            self._call_profile.lead.as_lead_fields(),
            [],
            "greeting",
            language=language,
            **self._prompt_kwargs(compact=compact),
        )
        phone_kwargs = {
            "max_tokens": (
                self._llm._settings.phone_llm_max_tokens
                or self._llm._settings.llm_max_tokens
                or 140
            ),
            "reasoning_effort": self._llm._settings.phone_llm_reasoning_effort,
            "temperature": self._llm._settings.phone_llm_temperature,
        }
        phone_model = (self._llm._settings.phone_llm_model or "").strip()
        if phone_model:
            phone_kwargs["model"] = phone_model
        raw, latency, usage = await self._llm.generate(
            messages,
            max_retries=0,
            **phone_kwargs,
        )
        timings.llm_latency_ms = latency
        timings.llm_usage = usage
        parsed = parse_structured_response(raw)
        timings.mark("llm_completed")
        if parsed is None:
            raise LlmStructuredOutputError(
                "The assistant could not produce a structured greeting.",
                details="Single-attempt greeting response was not valid structured JSON.",
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
        timings.session_id = session.id
        trace(
            "conversation.turn.start",
            turn_id=timings.turn_id,
            session_id=session.id,
            transport=getattr(timings, "transport", None),
            state=session.current_state,
            language_prior=prior_language,
            language_expected=turn_language or prior_language,
            stt_language=stt_language,
            user_text=user_text,
            history_turns=len(history),
            streaming=on_assistant_chunk is not None,
        )
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
        # model (real switch). Otherwise use a local fallback; never make a
        # second LLM call.
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

        # Continuity is handled by the first-pass prompt. Never make a second
        # LLM call merely because the generated wording restarts or stalls.
        lang = session.language or turn_language or "en"
        biz = self._call_profile.business_name
        agent = self._call_profile.agent_name
        abandoning = (parsed.next_state or "").strip().lower() == "abandoned"

        if abandoning:
            msg = (parsed.assistant_message or "").strip()
            if (
                not msg
                or _ASKING_RE.search(msg)
                or looks_like_reintroduction(
                    msg,
                    agent_name=agent,
                    business_name=biz,
                    user_text=user_text,
                )
            ):
                from backend.telephony.call_manager import fallback_text

                parsed.assistant_message = fallback_text("goodbye", lang)
                logger.info("Sanitized abandon message to goodbye")

        parsed.assistant_message = normalize_spoken_names(
            parsed.assistant_message, lang
        )

        if session.current_state == "greeting":
            session.current_state = "collecting_identity"

        self._apply_extraction(db, lead, parsed, session, user_text=user_text)
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
        extracted = getattr(parsed, "extracted_fields", None) or {}
        trace(
            "conversation.turn.end",
            turn_id=timings.turn_id,
            session_id=session.id,
            state=session.current_state,
            next_state_suggested=getattr(parsed, "next_state", None),
            language=session.language,
            assistant_message=parsed.assistant_message,
            extracted_fields=extracted,
            llm_latency_ms=timings.llm_latency_ms,
            stt_latency_ms=timings.stt_latency_ms,
            total_elapsed_ms=timings.total(),
            conversation_complete=bool(
                getattr(parsed, "conversation_complete", False)
            ),
        )
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
