"""Focused tests for caller-perceived turn telemetry."""

import re
from datetime import datetime, timezone

import httpx
import pytest

from backend.conversation import ConversationEngine
from backend.database import create_engine_and_session, session_scope
from backend.metrics import TurnTimings, persist_turn_telemetry
from backend.models import Message, ProviderEvent, Session
from backend.telephony.call_manager import CallSession, CallRegistry
from tests.conftest import make_mock_llm_client, make_settings, structured_json


def test_turn_timings_reports_caller_perceived_phases():
    timings = TurnTimings(
        started=100.0,
        started_at=datetime(2026, 8, 7, tzinfo=timezone.utc),
        transport="twilio_media_stream",
        stt_provider="sarvam",
    )

    timings.mark("utterance_end", at=100.0)
    timings.mark("transcript_received", at=100.2)
    timings.mark("llm_completed", at=100.5)
    timings.mark("tts_first_audio", at=100.7)
    timings.mark("first_outbound_audio", at=101.0)
    timings.add_duration("redirect_poll_delay", 125)

    metrics = timings.to_schema()
    assert metrics.caller_perceived_latency_ms == 1000
    assert set(metrics.phase_timestamps) >= {
        "transcript_received",
        "llm_completed",
        "tts_first_audio",
        "first_outbound_audio",
    }
    assert metrics.phase_durations_ms["redirect_poll_delay"] == 125
    assert metrics.telemetry_dimensions["transport"] == "twilio_media_stream"
    assert metrics.telemetry_dimensions["stt_provider"] == "sarvam"


def test_provider_events_use_existing_schema_and_turn_correlation(tmp_path):
    url = f"sqlite:///{(tmp_path / 'telemetry.db').as_posix()}"
    _, factory = create_engine_and_session(url)
    timings = TurnTimings(
        started=10.0,
        transport="twilio_gather",
        stt_provider="twilio",
        llm_provider="sarvam",
        tts_provider="sarvam",
    )
    timings.mark("transcript_received", at=10.0)
    timings.mark("llm_completed", at=10.3)

    with session_scope(factory) as db:
        session = Session(language="en")
        db.add(session)
        db.flush()
        session_id = session.id
        persist_turn_telemetry(db, session_id, timings)

    timings.mark("tts_first_audio", at=10.5)
    timings.mark("first_outbound_audio", at=10.6)
    with session_scope(factory) as db:
        persist_turn_telemetry(db, session_id, timings)
        # Re-persisting the same object must not duplicate phase rows.
        persist_turn_telemetry(db, session_id, timings)

    with session_scope(factory) as db:
        events = (
            db.query(ProviderEvent)
            .filter(ProviderEvent.session_id == session_id)
            .order_by(ProviderEvent.id)
            .all()
        )

    assert {event.event_type for event in events} == {
        "transcript_received",
        "llm_completed",
        "tts_first_audio",
        "first_outbound_audio",
    }
    assert {event.request_id for event in events} == {timings.turn_id}
    providers = {event.event_type: event.provider for event in events}
    assert providers["transcript_received"] == "twilio"
    assert providers["llm_completed"] == "sarvam"
    assert providers["tts_first_audio"] == "sarvam"
    assert providers["first_outbound_audio"] == "twilio"


def test_gather_turn_persists_first_outbound_audio(client):
    from twilio.request_validator import RequestValidator

    url = "https://example.ngrok-free.app/api/calls/turn"
    params = {
        "CallSid": "CA-telemetry-1",
        "SpeechResult": "my name is rahul sharma",
    }
    signature = RequestValidator("token-test").compute_signature(url, params)
    turn = client.post(
        "/api/calls/turn",
        data=params,
        headers={"X-Twilio-Signature": signature},
    )
    assert turn.status_code == 200
    match = re.search(r"/api/calls/audio/([0-9a-f]+)", turn.text)
    assert match

    audio = client.get(f"/api/calls/audio/{match.group(1)}")
    assert audio.status_code == 200

    record = client.app.state.call_registry.get("CA-telemetry-1")
    with session_scope(client.app.state.session_factory) as db:
        events = (
            db.query(ProviderEvent)
            .filter(ProviderEvent.session_id == record.session_id)
            .all()
        )
        assistant = (
            db.query(Message)
            .filter(
                Message.session_id == record.session_id,
                Message.role == "assistant",
            )
            .order_by(Message.id.desc())
            .first()
        )

    by_type = {event.event_type: event for event in events}
    assert set(by_type) >= {
        "transcript_received",
        "llm_completed",
        "tts_first_audio",
        "first_outbound_audio",
    }
    turn_ids = {
        by_type[name].request_id
        for name in (
            "transcript_received",
            "llm_completed",
            "tts_first_audio",
            "first_outbound_audio",
        )
    }
    assert len(turn_ids) == 1
    assert assistant.total_turn_latency_ms == by_type["first_outbound_audio"].latency_ms


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_playback_mark_and_interruption_clear_are_observable(tmp_path):
    ws = _FakeWebSocket()
    session = CallSession(
        settings=make_settings(tmp_path),
        session_factory=None,
        engine=None,
        sarvam=None,
        twilio=None,
        registry=CallRegistry(),
        ws=ws,
    )
    timings = TurnTimings(started=10.0, transport="twilio_media_stream")
    timings.mark("first_outbound_audio", at=10.1)

    session._pending_marks["reply-1"] = timings
    session._handle_playback_mark("reply-1")
    assert "playback_mark" in timings.phase_timestamps
    assert "playback_duration" in timings.phase_durations_ms

    session._playing = True
    session._playback_timings = timings
    await session._clear_playback()
    assert ws.sent[-1]["event"] == "clear"
    assert "interruption_clear" in timings.phase_timestamps
    assert "interruption_clear_delay" in timings.phase_durations_ms


@pytest.mark.asyncio
async def test_repair_and_language_mismatch_have_separate_dimensions(tmp_path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "not valid json"}}],
                    "usage": {},
                },
            )
        return structured_json(
            "This is an English answer that does not match Gujarati.",
            detected_language="gu",
            extracted_fields={},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path, llm_reasoning_effort="none")
    engine = ConversationEngine(make_mock_llm_client(settings, handler))
    _, factory = create_engine_and_session(
        f"sqlite:///{(tmp_path / 'language-telemetry.db').as_posix()}"
    )
    with session_scope(factory) as db:
        session = Session(language="gu")
        db.add(session)
        db.flush()
        timings = TurnTimings(settings=settings)
        _, parsed = await engine.process_turn(
            db,
            session,
            "મારું નામ રાહુલ છે",
            timings,
            stt_language="gu",
        )

    dimensions = timings.dimensions()
    assert parsed.detected_language == "gu"
    assert timings.repair_count == 1
    assert timings.llm_attempt_count == 2
    assert dimensions["reasoning_mode"] == "none"
    assert dimensions["language_mismatch"] is True
    assert dimensions["language_repair"] == "localized_repeat_fallback"
