"""HTTP API routes for the Sarvam Cloud Lead Agent."""

import base64

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from backend.audio import prepare_audio, validate_upload
from backend.config import Settings, get_settings
from backend.conversation import ConversationEngine
from backend.database import session_scope
from backend.errors import LeadNotFoundError, SessionNotFoundError, TtsError
from backend.exports import lead_to_csv_bytes, lead_to_json_bytes
from backend.metrics import TurnTimings
from backend.models import Lead, Message, Session
from backend.providers.sarvam_client import SarvamClient
from backend.schemas import (
    ConfirmRequest,
    ConfigOut,
    HealthOut,
    LeadListEntry,
    ProviderStatus,
    SessionOut,
    SessionSummary,
    TextMessage,
    TurnResponse,
)
from backend.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _iso(dt) -> str:
    return dt.isoformat() if dt else ""


def _error_response(exc: BaseException, debug: bool) -> dict:
    code = getattr(exc, "code", "INTERNAL_ERROR")
    message = getattr(exc, "message", "Unexpected server error.")
    retryable = getattr(exc, "retryable", False)
    details = getattr(exc, "details", None)
    if debug and details is None:
        details = str(exc)
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": bool(retryable),
            "details": details,
        }
    }


def _turn_response(
    session: Session,
    transcript: str,
    assistant_message: str,
    timings: TurnTimings,
    engine: ConversationEngine,
    *,
    audio_base64: str | None = None,
    audio_mime: str | None = None,
    warning: str | None = None,
    debug: dict | None = None,
) -> TurnResponse:
    return TurnResponse(
        session_id=session.id,
        transcript=transcript,
        assistant_message=assistant_message,
        audio_base64=audio_base64,
        audio_mime=audio_mime,
        lead=engine.to_lead_out(session.lead, session),
        current_state=session.current_state,
        conversation_status=session.status,
        metrics=timings.to_schema(timings.llm_usage),
        warning=warning,
        debug=debug,
    )


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    settings: Settings = request.app.state.settings
    db_state: OrmSession = request.app.state.session_factory
    db_ok = "ok"
    try:
        with session_scope(db_state) as db:
            db.execute(select(1))
    except Exception:
        db_ok = "error"
    return HealthOut(status="ok" if db_ok == "ok" else "degraded", app=settings.app_name, database=db_ok)


@router.get("/api/config", response_model=ConfigOut)
def config(request: Request) -> ConfigOut:
    settings: Settings = request.app.state.settings
    return ConfigOut(
        provider="sarvam-cloud",
        debug=settings.debug,
        default_language=settings.default_language,
        max_audio_mb=settings.max_audio_mb,
        stt_model=settings.sarvam_stt_model,
        tts_model=settings.sarvam_tts_model,
        voice=settings.sarvam_tts_speaker,
        llm_model=settings.llm_model,
    )


@router.get("/api/provider/status", response_model=ProviderStatus)
async def provider_status(request: Request) -> ProviderStatus:
    sarvam: SarvamClient = request.app.state.sarvam_client
    llm = request.app.state.llm_client
    speech = await sarvam.health_check()
    llm_status = await llm.health_check()
    if speech.status == "error" or llm_status.status == "error":
        status = "error"
    elif speech.status == "degraded" or llm_status.status == "degraded":
        status = "degraded"
    else:
        status = "ok"
    return ProviderStatus(
        provider="sarvam-cloud",
        status=status,
        message="Speech service: {speech}. LLM: {llm}.".format(
            speech=speech.message, llm=llm_status.message
        ),
        latency_ms=(speech.latency_ms or 0) + (llm_status.latency_ms or 0),
        details={"speech": speech.model_dump(), "llm": llm_status.model_dump()},
    )


@router.post("/api/sessions")
async def create_session(request: Request):
    settings: Settings = request.app.state.settings
    sarvam: SarvamClient = request.app.state.sarvam_client
    greeting = (
        "Hi there! I'm the lead qualification assistant. To get you the "
        "right information quickly, may I ask your name first?"
    )
    audio_base64: str | None = None
    audio_mime: str | None = None
    warning: str | None = None
    try:
        audio_bytes, audio_mime, _ = await sarvam.synthesize(greeting, None)
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    except Exception:
        warning = "Speech synthesis failed; the greeting is shown as text."
    with session_scope(request.app.state.session_factory) as db:
        session = Session(language=settings.default_language)
        db.add(session)
        db.flush()
        engine: ConversationEngine = request.app.state.conversation_engine
        return {
            "session_id": session.id,
            "status": session.status,
            "current_state": session.current_state,
            "language": session.language,
            "created_at": _iso(session.created_at),
            "greeting": greeting,
            "audio_base64": audio_base64,
            "audio_mime": audio_mime,
            "warning": warning,
            "lead": engine.to_lead_out(None, session).model_dump(),
        }


def _get_session(request: Request, session_id: str) -> tuple[OrmSession, Session]:
    with session_scope(request.app.state.session_factory) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return db, session


@router.get("/api/sessions/{session_id}", response_model=SessionOut)
def get_session(request: Request, session_id: str) -> SessionOut:
    with session_scope(request.app.state.session_factory) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        engine: ConversationEngine = request.app.state.conversation_engine
        messages = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.id)
            .all()
        )
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        latencies = [m.total_turn_latency_ms for m in assistant_msgs if m.total_turn_latency_ms]
        avg = sum(latencies) // len(latencies) if latencies else 0
        return SessionOut(
            id=session.id,
            status=session.status,
            current_state=session.current_state,
            language=session.language,
            created_at=_iso(session.created_at),
            updated_at=_iso(session.updated_at),
            message_count=len(messages),
            turn_count=len(assistant_msgs),
            average_latency_ms=avg,
            lead=engine.to_lead_out(session.lead, session),
        )


@router.get("/api/sessions/{session_id}/summary", response_model=SessionSummary)
def session_summary(request: Request, session_id: str) -> SessionSummary:
    with session_scope(request.app.state.session_factory) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        engine: ConversationEngine = request.app.state.conversation_engine
        return SessionSummary(**engine.summarize_session(db, session))


@router.post("/api/sessions/{session_id}/audio")
async def process_audio(
    request: Request,
    session_id: str,
    file: UploadFile = File(...),
    retain_audio: bool = False,
) -> dict:
    settings: Settings = request.app.state.settings
    with session_scope(request.app.state.session_factory) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        engine: ConversationEngine = request.app.state.conversation_engine
        sarvam: SarvamClient = request.app.state.sarvam_client

        timings = TurnTimings(settings=settings)
        prepared = None
        try:
            data = await file.read()
            validate_upload(
                file.filename or "recording.webm",
                file.content_type or "",
                len(data),
                settings.max_audio_mb,
            )
            prepared = await prepare_audio(
                data, file.filename or "recording.webm", file.content_type or "", settings
            )
            timings.audio_duration_ms = prepared.duration_ms

            transcript, stt_latency, stt_language = await sarvam.transcribe(
                prepared.wav_path, prepared.duration_ms
            )
            timings.stt_latency_ms = stt_latency
            timings.transcript_char_count = len(transcript)

            lead, parsed = await engine.process_turn(
                db, session, transcript, timings, stt_language=stt_language
            )
            timings.response_char_count = len(parsed.assistant_message)

            audio_base64: str | None = None
            audio_mime: str | None = None
            warning: str | None = None
            tts_latency = 0
            if session.status != "abandoned":
                try:
                    audio_bytes, audio_mime, tts_latency = await sarvam.synthesize(
                        parsed.assistant_message, parsed.detected_language
                    )
                    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
                    timings.tts_attempted = True
                except TtsError as exc:
                    logger.warning("TTS failed; continuing in text mode: %s", exc.code)
                    warning = "Speech synthesis failed; the reply is shown as text."
                except Exception:
                    warning = "Speech synthesis failed; the reply is shown as text."
            timings.tts_latency_ms = tts_latency

            last_assistant = (
                db.query(Message)
                .filter(
                    Message.session_id == session.id,
                    Message.role == "assistant",
                )
                .order_by(Message.id.desc())
                .first()
            )
            if last_assistant is not None:
                last_assistant.tts_latency_ms = tts_latency or None
                last_assistant.total_turn_latency_ms = timings.total()
                last_assistant.estimated_provider_cost = round(
                    (last_assistant.estimated_provider_cost or 0.0)
                    + timings.estimated_stt_cost()
                    + timings.estimated_tts_cost(),
                    6,
                )
                db.add(last_assistant)
            return _turn_response(
                session,
                transcript,
                parsed.assistant_message,
                timings,
                engine,
                audio_base64=audio_base64,
                audio_mime=audio_mime,
                warning=warning,
            ).model_dump()
        finally:
            if prepared is not None:
                prepared.cleanup()


@router.post("/api/sessions/{session_id}/message")
async def process_message(
    request: Request, session_id: str, body: TextMessage
) -> dict:
    settings: Settings = request.app.state.settings
    sarvam: SarvamClient = request.app.state.sarvam_client
    with session_scope(request.app.state.session_factory) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        engine: ConversationEngine = request.app.state.conversation_engine
        timings = TurnTimings(settings=settings)
        transcript = body.text.strip()
        timings.transcript_char_count = len(transcript)
        lead, parsed = await engine.process_turn(db, session, transcript, timings)
        timings.response_char_count = len(parsed.assistant_message)

        audio_base64: str | None = None
        audio_mime: str | None = None
        warning: str | None = None
        tts_latency = 0
        if session.status != "abandoned":
            try:
                audio_bytes, audio_mime, tts_latency = await sarvam.synthesize(
                    parsed.assistant_message, parsed.detected_language
                )
                audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
                timings.tts_attempted = True
            except TtsError as exc:
                logger.warning("TTS failed; continuing in text mode: %s", exc.code)
                warning = "Speech synthesis failed; the reply is shown as text."
            except Exception:
                warning = "Speech synthesis failed; the reply is shown as text."
        timings.tts_latency_ms = tts_latency

        return _turn_response(
            session,
            transcript,
            parsed.assistant_message,
            timings,
            engine,
            audio_base64=audio_base64,
            audio_mime=audio_mime,
            warning=warning,
        ).model_dump()


@router.post("/api/sessions/{session_id}/confirm")
async def confirm_session(request: Request, session_id: str, body: ConfirmRequest) -> dict:
    with session_scope(request.app.state.session_factory) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        engine: ConversationEngine = request.app.state.conversation_engine
        lead, message, extra = await engine.handle_confirmation(
            db, session, body.confirmed, body.corrections
        )
        return {
            "session_id": session.id,
            "assistant_message": message,
            "lead": engine.to_lead_out(lead, session).model_dump(),
            "conversation_status": session.status,
            "current_state": session.current_state,
            **extra,
        }


@router.post("/api/sessions/{session_id}/reset")
def reset_session(request: Request, session_id: str) -> dict:
    settings: Settings = request.app.state.settings
    with session_scope(request.app.state.session_factory) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        engine: ConversationEngine = request.app.state.conversation_engine
        session.status = "active"
        session.current_state = "greeting"
        session.language = settings.default_language
        session.skipped_fields = []
        lead = session.lead
        if lead is not None:
            for col in lead.__table__.columns.keys():
                if col not in {"id", "session_id"}:
                    setattr(lead, col, None)
        db.query(Message).filter(Message.session_id == session_id).delete()
        db.add(session)
        return {
            "session_id": session.id,
            "status": session.status,
            "current_state": session.current_state,
            "lead": engine.to_lead_out(lead, session).model_dump(),
        }


@router.get("/api/sessions/{session_id}/lead")
def get_session_lead(request: Request, session_id: str) -> dict:
    with session_scope(request.app.state.session_factory) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        engine: ConversationEngine = request.app.state.conversation_engine
        return engine.to_lead_out(session.lead, session).model_dump()


@router.get("/api/leads")
def list_leads(request: Request) -> list[dict]:
    with session_scope(request.app.state.session_factory) as db:
        leads = db.execute(select(Lead).order_by(Lead.id.desc()).limit(500)).scalars().all()
        return [
            LeadListEntry(
                id=lead.id,
                session_id=lead.session_id,
                full_name=lead.full_name,
                phone_number=lead.phone_number,
                email=lead.email,
                qualification_score=lead.qualification_score,
                qualification_level=lead.qualification_level,
                conversation_status=lead.conversation_status,
                created_at=_iso(lead.created_at),
            ).model_dump()
            for lead in leads
        ]


@router.get("/api/leads/{lead_id}")
def get_lead(request: Request, lead_id: int) -> dict:
    with session_scope(request.app.state.session_factory) as db:
        lead = db.get(Lead, lead_id)
        if lead is None:
            raise LeadNotFoundError(str(lead_id))
        return {
            "lead_id": lead.id,
            "session_id": lead.session_id,
            "fields": {
                name: getattr(lead, name)
                for name in (
                    "full_name",
                    "phone_number",
                    "email",
                    "company_name",
                    "job_title",
                    "qualification_score",
                    "qualification_level",
                    "conversation_status",
                    "created_at",
                )
            },
        }


@router.get("/api/leads/{lead_id}/export.json")
def export_lead_json(request: Request, lead_id: int) -> Response:
    with session_scope(request.app.state.session_factory) as db:
        lead = db.get(Lead, lead_id)
        if lead is None:
            raise LeadNotFoundError(str(lead_id))
        return Response(
            content=lead_to_json_bytes(lead),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="lead-{lead_id}.json"'},
        )


@router.get("/api/leads/{lead_id}/export.csv")
def export_lead_csv(request: Request, lead_id: int) -> Response:
    with session_scope(request.app.state.session_factory) as db:
        lead = db.get(Lead, lead_id)
        if lead is None:
            raise LeadNotFoundError(str(lead_id))
        return Response(
            content=lead_to_csv_bytes(lead),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="lead-{lead_id}.csv"'},
        )


@router.delete("/api/sessions/{session_id}")
def delete_session(request: Request, session_id: str) -> dict:
    with session_scope(request.app.state.session_factory) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        db.delete(session)
        return {"deleted": True, "session_id": session_id}


@router.delete("/api/leads/{lead_id}")
def delete_lead(request: Request, lead_id: int) -> dict:
    with session_scope(request.app.state.session_factory) as db:
        lead = db.get(Lead, lead_id)
        if lead is None:
            raise LeadNotFoundError(str(lead_id))
        db.delete(lead)
        return {"deleted": True, "lead_id": lead_id}
