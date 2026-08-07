"""Shared pytest fixtures.

All provider clients are mocked with httpx.MockTransport - no real Sarvam API
key or internet access is required.
"""

import base64
import json
import os
import tempfile

import httpx
import pytest

os.environ["DATABASE_URL"] = (
    "sqlite:///" + tempfile.mkdtemp().replace("\\", "/") + "/conftest.db"
)
os.environ["SARVAM_API_KEY"] = ""
os.environ["LLM_API_KEY"] = ""
os.environ["LLM_REASONING_EFFORT"] = ""
# Blank any Twilio values that a developer's local .env may contain, so tests
# never hit the real Twilio API or leak credentials.
os.environ["TWILIO_ACCOUNT_SID"] = ""
os.environ["TWILIO_AUTH_TOKEN"] = ""
os.environ["TWILIO_PHONE_NUMBER"] = ""
os.environ["TWILIO_FROM_NUMBER"] = ""
os.environ["TWILIO_TEST_PHONE_NUMBER"] = ""
os.environ["TWILIO_VERIFIED_NUMBERS"] = ""
os.environ["PUBLIC_BASE_URL"] = ""
os.environ["TWILIO_CALL_PUBLIC_BASE_URL"] = ""
os.environ["TELEPHONY_PROVIDER"] = "twilio"
os.environ["EXOTEL_ACCOUNT_SID"] = ""
os.environ["EXOTEL_API_KEY"] = ""
os.environ["EXOTEL_API_TOKEN"] = ""
os.environ["EXOTEL_CALLER_ID"] = ""

from backend.config import Settings
from backend.main import build_app
from backend.providers.llm_client import LlmClient
from backend.providers.sarvam_client import SarvamClient
from backend.telephony.twilio_service import TwilioService


def make_settings(tmp_path, **overrides) -> Settings:
    base = {
        "database_url": f"sqlite:///{str(tmp_path / 'test.db').replace(chr(92), '/')}",
        "temp_dir": str(tmp_path / "tmp"),
        "rate_limit_enabled": False,
        "debug": False,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def make_mock_llm_client(settings, handler):
    transport = httpx.MockTransport(handler)
    return LlmClient(settings, http_client=httpx.AsyncClient(transport=transport))


def make_mock_sarvam_client(settings, handler):
    transport = httpx.MockTransport(handler)
    return SarvamClient(settings, http_client=httpx.AsyncClient(transport=transport))


class FakeCallInstance:
    def __init__(self, sid="CA-test-call-sid", status="queued", to=None, from_=None):
        self.sid = sid
        self.status = status
        self.to = to
        self._from = from_


class FakeCalls:
    def __init__(self):
        self.created: list[dict] = []
        self.updated: list[str] = []

    def create(self, to, from_, url=None, status_callback=None, status_callback_event=None):
        sid = (
            "CA-test-call-sid"
            if not self.created
            else f"CA-test-call-sid-{len(self.created) + 1}"
        )
        self.created.append(
            {"to": to, "from_": from_, "url": url, "status_callback": status_callback, "sid": sid}
        )
        return FakeCallInstance(sid=sid, status="queued", to=to, from_=from_)

    def __call__(self, sid):
        self.updated.append(sid)
        return FakeCallInstance(sid=sid, status="completed")


class FakeOutgoingCallerId:
    def __init__(self, phone_number, friendly_name=""):
        self.phone_number = phone_number
        self.friendly_name = friendly_name


class FakeOutgoingCallerIds:
    def __init__(self, numbers=("+917048211715",)):
        self.numbers = list(numbers)

    def list(self, limit=None):
        return [FakeOutgoingCallerId(n) for n in self.numbers]


class FakeTwilioClient:
    def __init__(self):
        self.calls = FakeCalls()
        self.outgoing_caller_ids = FakeOutgoingCallerIds()


def make_mock_twilio_client(settings):
    """Build a TwilioService backed by a fake SDK client (no network)."""
    service = TwilioService(settings, client=FakeTwilioClient())
    return service


def _wav_bytes() -> bytes:
    return bytes.fromhex(
        "524946460400000057415645666d74201000000001000100401f0000803e0000020010006461746100000000"
    )


def sarvam_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/speech-to-text"):
        return httpx.Response(
            200,
            json={
                "transcript": "hello this is a test utterance",
                "language_code": "gu-IN",
            },
        )
    if request.url.path.endswith("/text-to-speech/stream"):
        return httpx.Response(
            200, content=_wav_bytes(), headers={"content-type": "audio/wav"}
        )
    if request.url.path.endswith("/text-to-speech"):
        return httpx.Response(
            200, json={"audios": [base64.b64encode(_wav_bytes()).decode("ascii")]}
        )
    return httpx.Response(200, json={})


def structured_json(content: str, **extra) -> httpx.Response:
    reply = {
        "assistant_message": content,
        "detected_language": "en",
        "extracted_fields": {},
        "fields_to_clear": [],
        "next_state": "",
        "conversation_complete": False,
        "needs_confirmation": False,
    }
    reply.update(extra)
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-mock",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(reply),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        },
    )


def user_utterance(messages) -> str:
    """Return the real last user utterance, skipping the injected state block."""
    for m in reversed(messages):
        if m["role"] != "user":
            continue
        content = m["content"] or ""
        if content.startswith("\n\nCurrent state:") or content.startswith("\n\nYour previous reply"):
            continue
        return content
    return ""


def lead_llm_handler(request: httpx.Request) -> httpx.Response:
    """Conversational mock that drives a complete lead-qualification flow."""
    payload = json.loads(request.content or b"{}")
    messages = payload.get("messages", [])
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    lower = user_utterance(messages).lower()

    extracted, next_state, msg = {}, "", ""
    if "name is rahul" in lower or "i am rahul" in lower:
        extracted = {"full_name": "Rahul Sharma"}
        next_state = "collecting_contact"
        msg = "Thanks Rahul! What is the best phone number or email to reach you?"
    elif "phone" in lower or "email" in lower or "98765" in lower:
        extracted = {"phone_number": "+919876543210", "email": "rahul@acme.in"}
        next_state = "collecting_requirement"
        msg = "Got it. What is your main requirement or what are you looking for?"
    elif "crm" in lower or "requirement" in lower:
        extracted = {
            "business_requirement": "I need a CRM with voice automation",
            "company_name": "Acme Retail",
            "product_or_service_interest": "CRM software",
            "estimated_budget": "Rs 50000 per year",
            "purchase_timeline": "within 3 months",
            "decision_maker_status": "yes",
        }
        next_state = "requesting_consent"
        msg = "May I contact you later about this? (yes/no)"
    elif "contact" in lower and ("yes" in lower or "sure" in lower or "haan" in lower):
        extracted = {"consent_to_contact": "yes"}
        next_state = "reviewing_summary"
        msg = "Here is the summary of your details. Can you confirm this is correct? (yes/no)"
    elif "confirm" in lower and "yes" in lower:
        next_state = "completed"
        msg = "Confirmed! Your lead has been saved. Thank you!"
    else:
        next_state = "collecting_identity"
        msg = "Could you tell me your name?"

    if "repair" in system.lower() and next_state == "":
        msg = "I still need your name - what should I call you?"

    return structured_json(msg, extracted_fields=extracted, next_state=next_state,
                           conversation_complete=(next_state == "completed"))


@pytest.fixture
def settings(tmp_path):
    return make_settings(
        tmp_path,
        sarvam_api_key="test-key",
        llm_api_key="test-key",
        twilio_account_sid="AC-test",
        twilio_auth_token="token-test",
        twilio_phone_number="+15005550006",
        public_base_url="https://example.ngrok-free.app",
        twilio_test_phone_number="+919876543210",
    )


@pytest.fixture
def llm_client(settings):
    return make_mock_llm_client(settings, lead_llm_handler)


@pytest.fixture
def sarvam_client(settings):
    return make_mock_sarvam_client(settings, sarvam_handler)


@pytest.fixture
def twilio_client(settings):
    return make_mock_twilio_client(settings)


@pytest.fixture
def app(settings, llm_client, sarvam_client, twilio_client, monkeypatch):
    import backend.api.routes as routes
    from backend.audio import PreparedAudio

    async def fake_prepare_audio(data, filename, content_type, settings, ffmpeg_bin=None):
        path = settings.resolved_temp_dir / "test.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        # 44-byte header + 1 second of 16-bit mono 16 kHz
        path.write_bytes(b"\x00" * 44 + b"\x00" * 32000)
        return PreparedAudio(wav_path=path, duration_ms=1000, original_name=filename)

    monkeypatch.setattr(routes, "prepare_audio", fake_prepare_audio)
    return build_app(
        settings,
        sarvam_client=sarvam_client,
        llm_client=llm_client,
        twilio_client=twilio_client,
    )


@pytest.fixture
def client(app):
    with TestClientContext(app) as c:
        yield c


class TestClientContext:
    def __init__(self, app):
        from fastapi.testclient import TestClient

        self._client = TestClient(app)

    def __enter__(self):
        return self._client.__enter__()

    def __exit__(self, *exc):
        return self._client.__exit__(*exc)
