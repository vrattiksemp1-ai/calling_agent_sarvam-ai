"""Bhashini speech provider + per-conversation provider selection."""

from __future__ import annotations

import base64
import io
import json
import wave

import httpx
import pytest

from backend.errors import ProviderUnavailableError, TtsError
from backend.main import build_app
from backend.providers.bhashini_client import (
    BhashiniClient,
    bhashini_language,
    normalize_tts_audio,
)
from backend.providers.speech_router import SpeechRouter, transcribe_with_hint
from tests.conftest import (
    TestClientContext,
    make_mock_llm_client,
    make_mock_sarvam_client,
    make_mock_twilio_client,
    make_settings,
    lead_llm_handler,
    sarvam_handler,
)

CONFIG_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
INFER_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"


def _wav(rate: int, seconds: float = 0.1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x10\x00" * int(rate * seconds))
    return buf.getvalue()


def _rate_of(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getframerate()


class BhashiniMock:
    """Mock of the documented config + compute endpoints; records requests."""

    def __init__(self, tts_rate: int = 22050, ald=None):
        self.requests: list[httpx.Request] = []
        self.tts_rate = tts_rate
        # langPrediction list returned by audio-lang-detection (None = 500).
        self.ald = ald

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content or b"{}")
        task = body["pipelineTasks"][0]
        if str(request.url) == CONFIG_URL:
            lang = task["config"]["language"]["sourceLanguage"]
            return httpx.Response(
                200,
                json={
                    "pipelineResponseConfig": [
                        {
                            "taskType": task["taskType"],
                            "config": [
                                {"serviceId": "other/svc", "language": {"sourceLanguage": "xx"}},
                                {
                                    "serviceId": f"svc-{task['taskType']}-{lang}",
                                    "language": {"sourceLanguage": lang},
                                },
                            ],
                        }
                    ],
                    "pipelineInferenceAPIEndPoint": {
                        "callbackUrl": INFER_URL,
                        "inferenceApiKey": {"name": "Authorization", "value": "infer-key"},
                    },
                },
            )
        if str(request.url) == INFER_URL:
            if task["taskType"] == "audio-lang-detection":
                if self.ald is None:
                    return httpx.Response(500, json={"detail": {"kind": "DHRUVA-101"}})
                return httpx.Response(
                    200,
                    json={"pipelineResponse": [{"output": [{"langPrediction": self.ald}]}]},
                )
            if task["taskType"] == "asr":
                lang = task["config"]["language"]["sourceLanguage"]
                text = {"gu": "નમસ્તે", "hi": "नमस्ते"}.get(lang, " namaste ")
                return httpx.Response(
                    200,
                    json={"pipelineResponse": [{"taskType": "asr", "output": [{"source": text}]}]},
                )
            audio = base64.b64encode(_wav(self.tts_rate)).decode("ascii")
            return httpx.Response(
                200,
                json={
                    "pipelineResponse": [
                        {
                            "taskType": "tts",
                            "config": {"samplingRate": self.tts_rate},
                            "audio": [{"audioContent": audio, "audioUri": None}],
                        }
                    ]
                },
            )
        return httpx.Response(404, json={})


def _bhashini(settings, mock) -> BhashiniClient:
    return BhashiniClient(settings, http_client=httpx.AsyncClient(transport=httpx.MockTransport(mock)))


def _keyed_settings(tmp_path, **extra):
    return make_settings(
        tmp_path,
        sarvam_api_key="test-key",
        llm_api_key="test-key",
        bhashini_user_id="user-1",
        bhashini_ulca_api_key="ulca-1",
        **extra,
    )


def _asr_requests(mock):
    return [
        r for r in mock.requests
        if str(r.url) == INFER_URL and json.loads(r.content)["pipelineTasks"][0]["taskType"] == "asr"
    ]


# ---------- client ----------


@pytest.mark.asyncio
async def test_synthesize_uses_config_then_compute_and_resamples(tmp_path):
    mock = BhashiniMock(tts_rate=22050)
    client = _bhashini(_keyed_settings(tmp_path), mock)

    audio, mime, _ = await client.synthesize("નમસ્તે", "gu")
    await client.synthesize("ફરી", "gu-IN")

    assert mime == "audio/wav"
    assert _rate_of(audio) == 8000  # phone-safe rate regardless of what Bhashini sent
    config_calls = [r for r in mock.requests if str(r.url) == CONFIG_URL]
    assert len(config_calls) == 1, "config call must be cached per (task, language)"
    assert config_calls[0].headers["userID"] == "user-1"
    assert config_calls[0].headers["ulcaApiKey"] == "ulca-1"
    compute = [r for r in mock.requests if str(r.url) == INFER_URL][0]
    assert compute.headers["Authorization"] == "infer-key"
    task = json.loads(compute.content)["pipelineTasks"][0]
    assert task["config"]["serviceId"] == "svc-tts-gu"
    assert task["config"]["language"]["sourceLanguage"] == "gu"
    assert task["config"]["gender"] == "female"
    assert task["config"]["samplingRate"] == 8000
    await client.aclose()


@pytest.mark.asyncio
async def test_transcribe_sends_language_hint_and_wav_rate(tmp_path):
    mock = BhashiniMock()
    client = _bhashini(_keyed_settings(tmp_path), mock)
    path = tmp_path / "turn.wav"
    path.write_bytes(_wav(16000))

    text, _, language = await client.transcribe(path, 100, language_hint="hinglish")

    assert text == "नमस्ते"
    assert language == "hi"
    compute = [r for r in _asr_requests(mock)][0]
    payload = json.loads(compute.content)
    config = payload["pipelineTasks"][0]["config"]
    assert config["serviceId"] == "svc-asr-hi"
    assert config["samplingRate"] == 16000
    assert payload["inputData"]["audio"][0]["audioContent"]
    await client.aclose()


@pytest.mark.asyncio
async def test_direct_inference_key_skips_config_call(tmp_path):
    mock = BhashiniMock()
    settings = make_settings(
        tmp_path,
        bhashini_inference_api_key="direct-key",
        bhashini_asr_service_id="bhashini/bodhan/asr-transcribe-flex",
        bhashini_tts_service_id="Bhashini/IITM/TTS",
    )
    client = _bhashini(settings, mock)
    await client.synthesize("hello", "en")
    assert all(str(r.url) != CONFIG_URL for r in mock.requests)
    task = json.loads(mock.requests[0].content)["pipelineTasks"][0]
    assert task["config"]["serviceId"] == "Bhashini/IITM/TTS"
    assert mock.requests[0].headers["Authorization"] == "direct-key"
    await client.aclose()


@pytest.mark.asyncio
async def test_missing_keys_degrade_cleanly(tmp_path):
    client = _bhashini(make_settings(tmp_path), BhashiniMock())
    status = await client.health_check()
    assert status.status == "degraded"
    assert "BHASHINI_USER_ID" in status.message
    with pytest.raises(TtsError):
        await client.synthesize("hello", "en")
    with pytest.raises(ProviderUnavailableError):
        await client.open_realtime_stt()
    await client.aclose()


@pytest.mark.asyncio
async def test_language_switch_redoes_asr_in_detected_language(tmp_path):
    mock = BhashiniMock(ald=[{"langCode": "gu", "scriptCode": "gujr", "langScore": 0.99}])
    client = _bhashini(_keyed_settings(tmp_path), mock)
    path = tmp_path / "turn.wav"
    path.write_bytes(_wav(8000))
    text, _, language = await client.transcribe(path, 100, language_hint="en")
    assert (text, language) == ("નમસ્તે", "gu")
    assert [json.loads(r.content)["pipelineTasks"][0]["config"]["language"]["sourceLanguage"]
            for r in _asr_requests(mock)][-1] == "gu"
    await client.aclose()


@pytest.mark.asyncio
async def test_detection_failure_keeps_expected_language(tmp_path):
    mock = BhashiniMock(ald=None)  # detection service errors
    client = _bhashini(_keyed_settings(tmp_path), mock)
    path = tmp_path / "turn.wav"
    path.write_bytes(_wav(8000))
    text, _, language = await client.transcribe(path, 100, language_hint="gu")
    assert (text, language) == ("નમસ્તે", "gu")
    assert len(_asr_requests(mock)) == 1
    await client.aclose()


def test_pick_language_rule():
    pick = BhashiniClient._pick_language
    assert pick([{"langCode": "gu", "langScore": 1.0}], 0.9) == "gu"
    assert pick([{"langCode": "hi", "langScore": 0.99}], 0.9) == "hi"
    # Observed: English scores ~0.6 and is sometimes mislabelled (e.g. "mr").
    assert pick([{"langCode": "mr", "langScore": 0.6}], 0.9) == "en"
    assert pick([{"langCode": "en", "langScore": 0.65}], 0.9) == "en"
    # A confident language we do not support: keep the expected language.
    assert pick([{"langCode": "ta", "langScore": 0.99}], 0.9) is None
    assert pick([], 0.9) is None


def test_float_wav_from_bhashini_is_decoded():
    import array
    import struct

    from backend.providers.bhashini_client import _riff_to_pcm16

    floats = array.array("f", [0.0, 0.5, -0.5, 1.2, -1.2] * 100)
    data = floats.tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, 8000, 8000 * 4, 4, 32)
    wav = (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE"
           + b"fmt " + struct.pack("<I", len(fmt)) + fmt
           + b"data" + struct.pack("<I", len(data)) + data)
    pcm, rate = _riff_to_pcm16(wav)
    samples = array.array("h"); samples.frombytes(pcm)
    assert rate == 8000 and len(samples) == 500
    assert list(samples[:5]) == [0, 16383, -16383, 32767, -32768]  # clipped, not wrapped
    out = normalize_tts_audio(wav, None, 8000)
    assert _rate_of(out) == 8000


def test_language_mapping_and_audio_normalization():
    assert bhashini_language("gu-IN") == "gu"
    assert bhashini_language("gujlish") == "gu"
    assert bhashini_language("en-hi") == "hi"
    assert bhashini_language("ta", "en") == "en"  # outside en/hi/gu -> default
    assert bhashini_language(None, "gu") == "gu"
    # Raw PCM (no RIFF header) is wrapped using the reported rate, then resampled.
    raw = b"\x01\x00" * 2205
    out = normalize_tts_audio(raw, 22050, 8000)
    assert out[:4] == b"RIFF" and _rate_of(out) == 8000


def test_bhashini_speech_cost_is_zero_sarvam_is_not(tmp_path):
    from backend.metrics import TurnTimings

    settings = make_settings(tmp_path)

    def timings(provider):
        t = TurnTimings(settings=settings, stt_provider=provider, tts_provider=provider)
        t.audio_duration_ms = 60_000
        t.response_char_count = 1_000
        t.tts_attempted = True
        return t

    assert timings("bhashini").estimated_stt_cost() == 0.0
    assert timings("bhashini").estimated_tts_cost() == 0.0
    assert timings("sarvam").estimated_stt_cost() > 0.0
    assert timings("sarvam").estimated_tts_cost() > 0.0


# ---------- router ----------


class _Hinted:
    accepts_language_hint = True

    async def transcribe(self, path, duration_ms, language_hint=None):
        return language_hint


class _Plain:
    async def transcribe(self, path, duration_ms):
        return "no-hint"


@pytest.mark.asyncio
async def test_router_selection_is_scoped():
    router = SpeechRouter({"sarvam": _Plain(), "bhashini": _Hinted()}, default="sarvam")
    assert router.current_name == "sarvam"
    with router.use("bhashini"):
        assert await router.transcribe("x", 1, language_hint="gu") == "gu"
    assert router.current_name == "sarvam"
    assert await router.transcribe("x", 1, language_hint="gu") == "no-hint"
    assert router.get("unknown") is router.get("sarvam")
    assert await transcribe_with_hint(_Plain(), "x", 1, "gu") == "no-hint"


# ---------- app wiring ----------


@pytest.fixture
def bhashini_app(tmp_path, monkeypatch):
    import backend.api.routes as routes
    from backend.audio import PreparedAudio

    settings = _keyed_settings(
        tmp_path,
        twilio_account_sid="AC-test",
        twilio_auth_token="token-test",
        twilio_phone_number="+15005550006",
        public_base_url="https://example.ngrok-free.app",
        twilio_test_phone_number="+919876543210",
    )

    async def fake_prepare_audio(data, filename, content_type, settings, ffmpeg_bin=None):
        path = settings.resolved_temp_dir / "test.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_wav(16000, 1.0))
        return PreparedAudio(wav_path=path, duration_ms=1000, original_name=filename)

    monkeypatch.setattr(routes, "prepare_audio", fake_prepare_audio)
    mock = BhashiniMock()
    app = build_app(
        settings,
        sarvam_client=make_mock_sarvam_client(settings, sarvam_handler),
        bhashini_client=_bhashini(settings, mock),
        llm_client=make_mock_llm_client(settings, lead_llm_handler),
        twilio_client=make_mock_twilio_client(settings),
    )
    return app, mock


def test_config_lists_both_providers(bhashini_app):
    app, _ = bhashini_app
    with TestClientContext(app) as c:
        data = c.get("/api/config").json()
    ids = {p["id"]: p for p in data["speech_providers"]}
    assert set(ids) == {"sarvam", "bhashini"}
    assert ids["bhashini"]["configured"] is True
    assert data["speech_provider"] == "sarvam"


def test_bhashini_session_uses_bhashini_for_stt_and_tts(bhashini_app):
    app, mock = bhashini_app
    with TestClientContext(app) as c:
        created = c.post("/api/sessions?speech_provider=bhashini").json()
        assert created["speech_provider"] == "bhashini"
        assert created["audio_base64"]
        sid = created["session_id"]
        turn = c.post(
            f"/api/sessions/{sid}/audio?speech_provider=bhashini",
            files={"file": ("rec.webm", b"fake", "audio/webm")},
        ).json()
    assert turn["transcript"] == "namaste"  # session language en; detection unavailable
    assert turn["audio_base64"]
    dims = turn["metrics"]["telemetry_dimensions"]
    assert dims["stt_provider"] == "bhashini"
    assert dims["tts_provider"] == "bhashini"
    tasks = [json.loads(r.content)["pipelineTasks"][0]["taskType"] for r in mock.requests if str(r.url) == INFER_URL]
    assert "asr" in tasks and tasks.count("tts") >= 2


def test_sarvam_remains_default_and_bhashini_untouched(bhashini_app):
    app, mock = bhashini_app
    with TestClientContext(app) as c:
        created = c.post("/api/sessions").json()
    assert created["speech_provider"] == "sarvam"
    assert mock.requests == []


def test_invalid_provider_is_rejected(bhashini_app):
    app, _ = bhashini_app
    with TestClientContext(app) as c:
        resp = c.post("/api/sessions?speech_provider=google")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_SPEECH_PROVIDER"


def test_place_call_records_speech_provider(bhashini_app):
    app, mock = bhashini_app
    with TestClientContext(app) as c:
        resp = c.post(
            "/api/calls",
            json={"to": "+919876543210", "speech_provider": "bhashini"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
    assert data["speech_provider"] == "bhashini"
    record = app.state.call_registry.get(data["call_sid"])
    assert record.speech_provider == "bhashini"
    assert app.state.call_registry.last_speech_provider == "bhashini"
    # Trial Twilio: the opening was synthesized in advance with Bhashini.
    assert record.greeting_twiml and any(str(r.url) == INFER_URL for r in mock.requests)


@pytest.mark.asyncio
async def test_call_session_binds_concrete_provider(tmp_path):
    from backend.telephony.call_manager import CallRecord, CallRegistry, CallSession

    settings = _keyed_settings(tmp_path, sarvam_realtime_stt_enabled=True)
    sarvam = make_mock_sarvam_client(settings, sarvam_handler)
    bhashini = _bhashini(settings, BhashiniMock())
    router = SpeechRouter({"sarvam": sarvam, "bhashini": bhashini})
    registry = CallRegistry()
    registry.add(CallRecord(call_sid="CA1", to_number="+91", from_number="+1", speech_provider="bhashini"))

    class _Transport:
        name = "twilio_media_stream"

    session = CallSession(
        settings=settings,
        session_factory=None,
        engine=None,
        sarvam=router,
        twilio=None,
        registry=registry,
        transport=_Transport(),
    )
    session._call_sid = "CA1"
    session._bind_speech_provider()
    assert session._sarvam is bhashini
    assert session._speech_name == "bhashini"
    # Realtime Sarvam STT must not start for Bhashini calls.
    assert getattr(session._sarvam, "supports_realtime_stt") is False
    await sarvam.aclose()
    await bhashini.aclose()
