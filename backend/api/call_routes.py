"""REST + WebSocket routes for real outbound phone calls.

  POST   /api/calls              Place an outbound call (E.164 + trial check)
  GET    /api/calls/numbers      Verified destination numbers (dropdown source)
  GET    /api/calls/{sid}        Call status (used by the web UI)
  DELETE /api/calls/{sid}        Hang up an in-progress call
  GET/POST /api/calls/twiml      TwiML served to Twilio (stream or turn loop)
  GET/POST /api/calls/turn       <Gather input="speech"> transcript webhook
  GET/POST /api/calls/turn-result Deferred turn result (slow LLM Redirect)
  GET    /api/calls/audio/{id}   Hosted TTS audio for <Play> (file or live stream)
  POST   /api/calls/stream-status Twilio Media Streams lifecycle callback
  POST   /api/calls/status       Twilio status callback (signature-validated)
  WS     /api/calls/stream       Twilio Media Streams audio endpoint
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.telephony.call_manager import CallRecord, CallRegistry, CallSession
from backend.telephony.phone import InvalidPhoneNumberError, normalize_e164
from backend.telephony.turn_flow import TurnFlow, TtsStream
from backend.telephony.twilio_service import TwilioService
from backend.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/calls", tags=["calls"])


class PlaceCallRequest(BaseModel):
    to: str = Field(..., description="Destination phone number in E.164 format, e.g. +919876543210")


class CallStatusOut(BaseModel):
    call_sid: str
    status: str
    to: str
    from_: str = Field(alias="from")
    session_id: str | None = None
    error: str | None = None

    model_config = {"populate_by_name": True}


def _status_out(record: CallRecord) -> CallStatusOut:
    return CallStatusOut(
        call_sid=record.call_sid,
        status=record.status,
        to=record.to_number,
        from_=record.from_number,
        session_id=record.session_id,
        error=record.error,
    )


def _twilio(request: Request) -> TwilioService:
    return request.app.state.twilio_client


@router.get("/numbers")
async def list_verified_numbers(request: Request) -> dict:
    """Destination numbers the caller may choose from (dropdown source).

    Trial mode: numbers verified in the Twilio Console plus any configured
    fallback numbers. Non-trial mode: returns all (no restriction), the
    frontend simply allows free-form entry.
    """
    twilio = _twilio(request)
    if not twilio.trial_mode:
        return {"verified_numbers": [], "trial_mode": False}
    numbers = await twilio.verified_numbers()
    return {"verified_numbers": numbers, "trial_mode": True}


@router.post("", response_model=CallStatusOut)
async def place_call(request: Request, body: PlaceCallRequest) -> CallStatusOut:
    twilio = _twilio(request)
    registry: CallRegistry = request.app.state.call_registry
    turn_flow: TurnFlow = request.app.state.turn_flow

    try:
        to = normalize_e164(body.to)
    except InvalidPhoneNumberError as exc:
        raise exc

    # Single-active-call: hang up anything still ringing/in-progress so stacked
    # dashboard clicks cannot fight each other on a trial account.
    for prior in registry.active_calls():
        logger.info(
            "Ending prior active call %s before placing new outbound to %s",
            prior.call_sid,
            to,
        )
        try:
            turn_flow._drop_pending(prior.call_sid)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.warning("Failed clearing pending turn for %s", prior.call_sid)
        await twilio.complete_call(prior.call_sid)
        registry.update(prior.call_sid, status="completed", error="superseded")

    result = await twilio.start_call(to)
    record = CallRecord(
        call_sid=result.call_sid,
        to_number=result.to,
        from_number=result.from_,
        status=result.status,
    )
    registry.add(record)
    logger.info("Call registered: %s", result.call_sid)
    return _status_out(record)


@router.get("/twiml")
@router.post("/twiml")
async def call_twiml(request: Request) -> Response:
    """TwiML served to Twilio when a call is answered.

    Trial accounts strip the <Stream> verb from TwiML, so they get the
    <Gather input="speech"> turn loop instead; upgraded accounts get the
    real-time Media Streams TwiML.
    """
    twilio = _twilio(request)
    turn_flow: TurnFlow = request.app.state.turn_flow
    call_sid = ""
    if request.method == "POST":
        form = await request.form()
        call_sid = str(form.get("CallSid") or "")
    if not call_sid:
        call_sid = request.query_params.get("CallSid", "")
    logger.info(
        "Serving TwiML to %s call_sid=%s",
        request.client.host if request.client else "?",
        call_sid or "?",
    )
    if twilio.trial_mode:
        content = await turn_flow.greeting_twiml(call_sid=call_sid or None)
    else:
        content = twilio.stream_twiml()
    return Response(content=content, media_type="text/xml")


@router.get("/turn")
@router.post("/turn")
async def turn_callback(request: Request) -> Response:
    """Webhook for each <Gather input="speech"> transcript (trial turn loop).

    Twilio transcribes the caller's speech and POSTs it here (SpeechResult);
    we run the conversation engine, synthesize the reply with Sarvam TTS and
    return the next TwiML (<Play> the reply, then <Gather> the next utterance,
    or <Hangup/> once the conversation completes).
    """
    twilio = _twilio(request)
    form = await request.form()
    params = {key: value for key, value in form.items()}

    # Reconstruct the full URL exactly as Twilio signed it (same as /status).
    base = twilio.public_base().rstrip("/")
    url = base + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    signature = request.headers.get("X-Twilio-Signature", "")
    token = request.query_params.get("turn_token", "")
    if not twilio.validate_turn_callback(url, params, signature, token):
        from backend.telephony.twilio_service import InvalidTwilioSignatureError

        logger.warning(
            "Rejected turn callback for %s (signature=%s)",
            url,
            "present" if signature else "missing",
        )
        raise InvalidTwilioSignatureError()

    call_sid = params.get("CallSid") or ""
    speech_result = params.get("SpeechResult") or ""
    turn_flow: TurnFlow = request.app.state.turn_flow
    content = await turn_flow.process_webhook(call_sid, speech_result)
    return Response(content=content, media_type="text/xml")


@router.get("/turn-result")
@router.post("/turn-result")
async def turn_result(request: Request) -> Response:
    """Poll endpoint for a turn that needed more than Twilio's ~15s budget.

    /turn starts LLM+TTS in the background and Redirects here when the work
    is still running; we either return the finished TwiML or Pause+Redirect
    again until it is ready (or a poll limit is hit).
    """
    twilio = _twilio(request)
    form = {}
    if request.method == "POST":
        form = await request.form()
    params = {key: value for key, value in form.items()}
    # Merge query params Twilio includes on the Redirect URL.
    for key, value in request.query_params.multi_items():
        params.setdefault(key, value)

    base = twilio.public_base().rstrip("/")
    url = base + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    signature = request.headers.get("X-Twilio-Signature", "")
    token = request.query_params.get("turn_token", "") or params.get("turn_token", "")
    if not twilio.validate_turn_callback(url, params, signature, token):
        from backend.telephony.twilio_service import InvalidTwilioSignatureError

        logger.warning(
            "Rejected turn-result callback for %s (signature=%s)",
            url,
            "present" if signature else "missing",
        )
        raise InvalidTwilioSignatureError()

    call_sid = (
        request.query_params.get("call_sid")
        or params.get("call_sid")
        or params.get("CallSid")
        or ""
    )
    pending_token = (
        request.query_params.get("pending") or params.get("pending") or ""
    )
    turn_flow: TurnFlow = request.app.state.turn_flow
    content = await turn_flow.poll_pending_turn(call_sid, pending_token)
    return Response(content=content, media_type="text/xml")


async def _tts_stream_iterator(
    stream: TtsStream, turn_flow: TurnFlow, token: str
) -> AsyncIterator[bytes]:
    """Drain a live TTS stream into the HTTP response as chunks arrive."""
    empty_reads = 0
    try:
        while True:
            if stream.done.is_set() and stream.queue.empty():
                break
            try:
                chunk = await asyncio.wait_for(stream.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if stream.error is not None:
                    break
                empty_reads += 1
                if empty_reads > 15:
                    break
                continue
            empty_reads = 0
            yield chunk
    finally:
        turn_flow.drop_tts_stream(token)


@router.get("/audio/{file_id}")
async def get_tts_audio(request: Request, file_id: str) -> Response:
    """Serve hosted TTS WAV so Twilio <Play> can fetch the agent's voice.

    Streaming TTS replies are served live from the in-memory chunk queue so
    playback can start before synthesis completes; the buffered file path is
    kept for the non-streaming fallback.
    """
    turn_flow: TurnFlow = request.app.state.turn_flow
    stream = turn_flow.get_tts_stream(file_id)
    if stream is not None:
        if stream.error is not None:
            raise stream.error
        return StreamingResponse(
            _tts_stream_iterator(stream, turn_flow, file_id),
            media_type=stream.audio_type,
        )
    settings: Settings = request.app.state.settings
    path = settings.resolved_temp_dir / "tts" / f"{file_id}.wav"
    if not path.is_file():
        from backend.errors import AppError

        raise AppError(
            code="AUDIO_NOT_FOUND",
            message="The audio file was not found or has expired.",
            retryable=False,
            status_code=404,
        )
    return FileResponse(path, media_type="audio/wav")


@router.get("/{call_sid}", response_model=CallStatusOut)
def get_call_status(request: Request, call_sid: str) -> CallStatusOut:
    registry: CallRegistry = request.app.state.call_registry
    record = registry.get(call_sid)
    if record is None:
        from backend.errors import AppError

        raise AppError(
            code="CALL_NOT_FOUND",
            message=f"No call with SID {call_sid}.",
            retryable=False,
            status_code=404,
        )
    return _status_out(record)


@router.delete("/{call_sid}", response_model=CallStatusOut)
async def hang_up_call(request: Request, call_sid: str) -> CallStatusOut:
    twilio = _twilio(request)
    registry: CallRegistry = request.app.state.call_registry
    turn_flow: TurnFlow = request.app.state.turn_flow
    record = registry.get(call_sid)
    if record is None:
        from backend.errors import AppError

        raise AppError(
            code="CALL_NOT_FOUND",
            message=f"No call with SID {call_sid}.",
            retryable=False,
            status_code=404,
        )
    try:
        turn_flow._drop_pending(call_sid)
    except Exception:  # noqa: BLE001
        logger.warning("Failed clearing pending turn for hangup %s", call_sid)
    await twilio.complete_call(call_sid)
    registry.update(call_sid, status="completed")
    return _status_out(record)


@router.post("/stream-status")
async def stream_status_callback(request: Request) -> dict:
    """Twilio Media Streams status callback: stream-started / stream-stopped /
    stream-error. Logs the details (incl. StreamError) for diagnostics."""
    form = await request.form()
    params = {key: value for key, value in form.items()}
    logger.info("Stream status callback: %s", params)
    return {"ok": True}


@router.post("/status")
async def call_status_callback(request: Request) -> dict:
    """Twilio status callback webhook (validated by X-Twilio-Signature)."""
    twilio = _twilio(request)
    form = await request.form()
    params = {key: value for key, value in form.items()}

    # Reconstruct the full URL exactly as Twilio signed it. Behind a TLS
    # terminator like ngrok, request.url.scheme is "http", so rebuild from the
    # configured public base URL (which is what we registered with Twilio).
    base = twilio.public_base().rstrip("/")
    url = base + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    signature = request.headers.get("X-Twilio-Signature", "")
    if not twilio.validate_signature(url, params, signature):
        from backend.telephony.twilio_service import InvalidTwilioSignatureError

        logger.warning("Rejected status callback with invalid signature for %s", url)
        raise InvalidTwilioSignatureError()

    registry: CallRegistry = request.app.state.call_registry
    call_sid = params.get("CallSid") or ""
    status = (params.get("CallStatus") or "").lower()
    if status in {"initiated", "ringing", "answered", "in-progress", "completed", "busy", "failed", "no-answer", "canceled"}:
        if status in {"busy", "failed", "no-answer", "canceled"}:
            registry.update(call_sid, status="completed", error=status)
        else:
            registry.update(call_sid, status=status)
    return {"ok": True}


@router.websocket("/stream")
async def call_stream(ws: WebSocket) -> None:
    settings: Settings = ws.app.state.settings
    session = CallSession(
        settings=settings,
        session_factory=ws.app.state.session_factory,
        engine=ws.app.state.conversation_engine,
        sarvam=ws.app.state.sarvam_client,
        twilio=ws.app.state.twilio_client,
        registry=ws.app.state.call_registry,
        ws=ws,
    )
    try:
        await session.run()
    except WebSocketDisconnect:
        pass
