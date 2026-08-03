"""REST + WebSocket routes for real outbound phone calls.

  POST   /api/calls              Place an outbound call to a phone number
  GET    /api/calls/{sid}        Call status (used by the web UI)
  DELETE /api/calls/{sid}        Hang up an in-progress call
  WS     /api/calls/stream       Twilio Media Streams audio endpoint
  POST   /api/calls/{sid}/status Twilio status callback (webhook)
"""

from __future__ import annotations

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.telephony.call_manager import CallRecord, CallRegistry, CallSession
from backend.telephony.twilio_client import TwilioClient
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


@router.post("", response_model=CallStatusOut)
async def place_call(request: Request, body: PlaceCallRequest) -> CallStatusOut:
    settings: Settings = request.app.state.settings
    twilio: TwilioClient = request.app.state.twilio_client
    registry: CallRegistry = request.app.state.call_registry

    sid = await twilio.start_outbound_call(body.to.strip())
    record = CallRecord(
        call_sid=sid,
        to_number=body.to.strip(),
        from_number=settings.twilio_from_number,
    )
    registry.add(record)
    logger.info("Call registered: %s", sid)
    return _status_out(record)


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
    twilio: TwilioClient = request.app.state.twilio_client
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
    await twilio.complete_call(call_sid)
    registry.update(call_sid, status="completed")
    return _status_out(record)


@router.post("/{call_sid}/status")
async def call_status_callback(request: Request, call_sid: str) -> dict:
    registry: CallRegistry = request.app.state.call_registry
    form = await request.form()
    status = (form.get("CallStatus") or "").lower()
    if status in {"initiated", "ringing", "in-progress", "completed", "busy", "failed", "no-answer", "canceled"}:
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
