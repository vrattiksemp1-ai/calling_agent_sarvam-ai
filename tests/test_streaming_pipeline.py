"""Deterministic protocol tests for the fully streaming provider path."""

import base64
import asyncio
import json

import httpx
import pytest

from backend.conversation import ConversationEngine
from backend.database import create_engine_and_session
from backend.errors import LlmStructuredOutputError
from backend.metrics import TurnTimings
from backend.models import Lead, Session
from backend.providers.llm_client import LlmClient
from backend.providers.sarvam_client import SarvamClient, SarvamSttEvent
from backend.streaming_json import (
    AssistantMessageStreamParser,
    FirstSpeechChunkBuffer,
)
from backend.telephony.call_manager import CallRegistry, CallSession
from backend.telephony.endpointing import SemanticEndpointing
from tests.conftest import make_settings


class MockWebSocket:
    def __init__(self, incoming=()):
        self.incoming = list(incoming)
        self.sent = []
        self.closed = False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        if not self.incoming:
            raise StopAsyncIteration
        return self.incoming.pop(0)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_realtime_stt_protocol_partial_final_and_controls(tmp_path):
    socket = MockWebSocket(
        [
            json.dumps({"event": "session.begin", "request_id": "s1"}),
            json.dumps({"event": "vad.speech_start", "utterance_idx": 0}),
            json.dumps(
                {
                    "event": "transcript.partial",
                    "utterance_idx": 0,
                    "text": "કેમ",
                }
            ),
            json.dumps(
                {
                    "event": "transcript.final",
                    "utterance_idx": 0,
                    "text": "કેમ છો",
                    "language": "gu-IN",
                }
            ),
        ]
    )
    captured = {}

    async def connect(url, headers):
        captured.update(url=url, headers=headers)
        return socket

    settings = make_settings(tmp_path, sarvam_api_key="secret")
    client = SarvamClient(settings, stt_connect_factory=connect)
    session = await client.open_realtime_stt()
    await session.send_audio(b"\x01\x02")
    await session.update_config(threshold=0.7, silence_ms=350, min_speech_ms=80)
    await session.ping()
    events = [await session.receive() for _ in range(4)]

    assert "model=saaras%3Av3-realtime" in captured["url"]
    assert "language_code=auto" in captured["url"]
    assert "endpointing=vad" in captured["url"]
    assert "silence_duration_ms=400" in captured["url"]
    assert captured["headers"] == {"api-subscription-key": "secret"}
    assert socket.sent[0]["event"] == "config.update"
    assert base64.b64decode(socket.sent[1]["audio"]) == b"\x01\x02"
    assert socket.sent[1]["event"] == "audio_input"
    assert socket.sent[2]["silence_duration_ms"] == 350
    assert socket.sent[3] == {"event": "ping"}
    assert [event.type for event in events] == [
        "session.begin",
        "vad.speech_start",
        "transcript.partial",
        "transcript.final",
    ]
    assert events[-1].transcript == "કેમ છો"
    await session.close()
    assert socket.sent[-1] == {"event": "end"}


@pytest.mark.asyncio
async def test_realtime_tts_config_chunks_flush_and_completion(tmp_path):
    audio = b"\x01\x00" * 160
    socket = MockWebSocket(
        [
            json.dumps(
                {
                    "type": "audio",
                    "data": {"audio": base64.b64encode(audio).decode()},
                }
            ),
            json.dumps(
                {
                    "type": "event",
                    "data": {"event_type": "final", "message": "done"},
                }
            ),
        ]
    )

    async def connect(url, headers):
        assert "/v1/text-to-speech/stream?" in url
        assert "model=bulbul%3Av3" in url
        assert "send_completion_event=true" in url
        assert headers["api-subscription-key"] == "secret"
        return socket

    client = SarvamClient(
        make_settings(tmp_path, sarvam_api_key="secret"),
        tts_connect_factory=connect,
    )
    session = await client.open_realtime_tts("gu", codec="linear16")
    await session.send_text("કેમ છો")
    await session.flush()
    await session.ping()

    assert socket.sent[0]["type"] == "config"
    assert socket.sent[0]["data"]["language_code"] == "gu-IN"
    assert socket.sent[0]["data"]["speech_sample_rate"] == 8000
    assert socket.sent[1] == {"type": "text", "data": {"text": "કેમ છો"}}
    assert socket.sent[2] == {"type": "flush"}
    assert (await session.receive()).audio == audio
    assert (await session.receive()).type == "complete"
    await session.close()
    assert socket.closed is True


def _sse_response(parts, usage=None):
    records = []
    for part in parts:
        records.append(
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": part}, "finish_reason": None}]},
                ensure_ascii=False,
            )
            + "\n\n"
        )
    records.append(
        "data: "
        + json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": usage or {},
            }
        )
        + "\n\n"
    )
    records.append("data: [DONE]\n\n")
    return httpx.Response(
        200,
        content="".join(records).encode(),
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.asyncio
async def test_llm_sse_stream_captures_usage_and_preserves_generate(tmp_path):
    captured = []
    complete = '{"assistant_message":"hello","detected_language":"en"}'

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        if body.get("stream"):
            return _sse_response(
                [complete[:20], complete[20:]],
                {"prompt_tokens": 3, "completion_tokens": 4},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": complete}}],
                "usage": {"total_tokens": 7},
            },
        )

    settings = make_settings(tmp_path, llm_reasoning_effort="low")
    client = LlmClient(
        settings,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    events = [event async for event in client.stream_generate([{"role": "user", "content": "hi"}])]
    buffered, _, usage = await client.generate([{"role": "user", "content": "hi"}])

    assert "".join(event.delta for event in events if event.type == "delta") == complete
    assert events[-1].type == "done"
    assert events[-1].usage == {"prompt_tokens": 3, "completion_tokens": 4}
    assert events[-1].first_token_latency_ms is not None
    assert captured[0]["stream"] is True
    assert "stream_options" not in captured[0]
    assert "stream" not in captured[1]
    assert buffered == complete
    assert usage == {"total_tokens": 7}


def test_streaming_json_parser_handles_gujarati_unicode_and_escapes():
    raw = (
        '{"next_state":"collecting_contact","assistant_message":'
        '"કેમ છો? \\u0aa4\\u0aae\\u0abe\\u0ab0\\u0ac1\\u0a82 નામ \\"શું\\"?\\n",'
        '"detected_language":"gu"}'
    )
    parser = AssistantMessageStreamParser()
    output = "".join(parser.feed(raw[index : index + 3]) for index in range(0, len(raw), 3))
    parser.finish()
    assert output == 'કેમ છો? તમારું નામ "શું"?\n'


def test_first_speech_chunk_waits_for_sentence_and_has_bounded_fallback():
    sentence = FirstSpeechChunkBuffer()
    assert sentence.feed("Thanks for your ") == ""
    assert sentence.feed("time. What is your name?") == "Thanks for your time."
    assert sentence.feed(" This is ignored.") == ""

    bounded = FirstSpeechChunkBuffer(minimum_chars=8, maximum_chars=16)
    assert bounded.feed("This reply has no punctuation yet") == "This reply has"


def test_semantic_endpointing_handles_complete_and_incomplete_clauses():
    endpointing = SemanticEndpointing(base_ms=400, fast_ms=320, slow_ms=550)

    assert endpointing.recommend("હા બરાબર") == 320
    assert endpointing.recommend("My number is 9876.") == 320
    assert endpointing.recommend("મારે માહિતી જોઈએ છે અને") == 550
    assert endpointing.recommend("I need pricing because") == 550
    assert endpointing.recommend("My name is Rahul") == 400


@pytest.mark.asyncio
async def test_streamed_state_applies_only_after_validated_completion(tmp_path):
    reply = {
        "assistant_message": "Thanks Rahul. What is your phone number?",
        "detected_language": "en",
        "extracted_fields": {"full_name": "Rahul"},
        "fields_to_clear": [],
        "next_state": "collecting_contact",
        "conversation_complete": False,
        "needs_confirmation": False,
    }
    raw = json.dumps(reply)

    def handler(request):
        return _sse_response([raw[:35], raw[35:70], raw[70:]])

    settings = make_settings(tmp_path, llm_streaming_enabled=True)
    llm = LlmClient(
        settings,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    engine = ConversationEngine(llm)
    _, factory = create_engine_and_session(settings.database_url)
    db = factory()
    session = Session(current_state="collecting_identity", language="en")
    db.add(session)
    db.flush()
    observed = []

    async def on_chunk(chunk):
        lead = db.query(Lead).filter(Lead.session_id == session.id).first()
        observed.append((chunk, lead.full_name if lead else None, session.current_state))

    lead, _ = await engine.process_turn(
        db,
        session,
        "My name is Rahul",
        TurnTimings(settings=settings),
        on_assistant_chunk=on_chunk,
    )
    assert observed
    assert all(name is None for _, name, _ in observed)
    assert all(state == "collecting_identity" for _, _, state in observed)
    assert lead.full_name == "Rahul"
    assert session.current_state == "collecting_contact"
    db.close()


@pytest.mark.asyncio
async def test_invalid_stream_never_applies_extraction(tmp_path):
    invalid = '{"assistant_message":"Audible text","extracted_fields":{"full_name":"Bad"}'

    def handler(request):
        return _sse_response([invalid])

    settings = make_settings(tmp_path)
    llm = LlmClient(
        settings,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    engine = ConversationEngine(llm)
    _, factory = create_engine_and_session(settings.database_url)
    db = factory()
    session = Session(current_state="collecting_identity", language="en")
    db.add(session)
    db.flush()
    spoken = []

    with pytest.raises(LlmStructuredOutputError):
        await engine.process_turn(
            db,
            session,
            "My name is Bad",
            TurnTimings(settings=settings),
            on_assistant_chunk=lambda chunk: _append(spoken, chunk),
        )
    lead = db.query(Lead).filter(Lead.session_id == session.id).first()
    assert spoken == ["Audible text"]
    assert lead.full_name is None
    assert session.current_state == "collecting_identity"
    db.close()


async def _append(target, value):
    target.append(value)


class _NoopTransport:
    name = "test"

    def __init__(self):
        self.cleared = asyncio.Event()

    async def clear(self):
        self.cleared.set()


class _BlockingStt:
    def __init__(self):
        self.sent = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.sent:
            self.sent = True
            return SarvamSttEvent(type="vad.speech_start")
        await asyncio.Event().wait()

    async def send_audio(self, audio):
        return None

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_server_speech_start_immediately_cancels_response(tmp_path):
    transport = _NoopTransport()
    stt = _BlockingStt()

    class Sarvam:
        async def open_realtime_stt(self):
            return stt

    session = CallSession(
        make_settings(tmp_path, sarvam_realtime_stt_enabled=True),
        None,
        None,
        Sarvam(),
        None,
        CallRegistry(),
        transport=transport,
    )
    session._session_id = "session-1"
    session._call_sid = "call-1"
    session._playing = True
    session._response_task = asyncio.create_task(asyncio.Event().wait())
    task = asyncio.create_task(session._run_realtime_stt())

    await asyncio.wait_for(transport.cleared.wait(), timeout=1)
    assert session._response_task is None
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert stt.closed is True


@pytest.mark.asyncio
async def test_realtime_stt_connect_failure_uses_buffered_fallback(tmp_path, monkeypatch):
    class Sarvam:
        async def open_realtime_stt(self):
            raise OSError("offline")

    session = CallSession(
        make_settings(
            tmp_path,
            sarvam_realtime_stt_enabled=True,
            sarvam_realtime_stt_fallback_enabled=True,
        ),
        None,
        None,
        Sarvam(),
        None,
        CallRegistry(),
        transport=_NoopTransport(),
    )
    session._ready_utterance = b"\x01\x00" * 160
    labels = []

    async def replace(coroutine, *, label):
        labels.append(label)
        coroutine.close()

    monkeypatch.setattr(session, "_replace_response", replace)
    await session._run_realtime_stt()
    assert labels == ["stt-fallback"]
