"""Gather-optimized Phase 2 latency knobs."""

import json

import httpx
import pytest

from backend.call_profile import CallProfile, ProductOffering
from backend.conversation import ConversationEngine
from backend.metrics import TurnTimings
from backend.prompts import build_messages, build_system_prompt
from backend.telephony.turn_flow import TURN_INLINE_BUDGET_SECONDS, TURN_POLL_PAUSE_SECONDS
from tests.conftest import make_mock_llm_client, make_settings, structured_json


def test_gather_defaults_favor_lower_hold():
    assert TURN_INLINE_BUDGET_SECONDS >= 5.0
    assert TURN_POLL_PAUSE_SECONDS == 0


def test_compact_phone_prompt_is_shorter_than_full():
    profile = CallProfile(
        agent_name="Shivangi",
        business_name="Vrattiks",
        business_description="AI company",
        products_and_services=[
            ProductOffering(name="Voice agent", summary="phone AI")
        ],
    )
    full = build_system_prompt(
        "Vrattiks", "AI company", call_profile=profile, compact=False
    )
    compact = build_system_prompt(
        "Vrattiks", "AI company", call_profile=profile, compact=True
    )
    assert len(compact) < len(full)
    assert "Shivangi" in compact
    assert "Field meanings:" not in compact
    assert "CALL PROFILE facts" in compact


def test_compact_messages_skip_glossary_and_trim_history():
    history = [{"role": "user", "content": f"msg-{i}"} for i in range(12)]
    messages = build_messages(
        history,
        {},
        [],
        "collecting_identity",
        compact=True,
        call_profile=CallProfile(),
    )
    system = messages[0]["content"]
    assert "Field meanings:" not in system
    # system + last 6 history + state user = 8
    assert len(messages) == 8


def test_turn_flow_gather_attrs_use_settings(tmp_path):
    from backend.telephony.turn_flow import TurnFlow
    from backend.telephony.call_manager import CallRegistry
    from unittest.mock import MagicMock

    settings = make_settings(
        tmp_path,
        gather_speech_timeout="1",
        gather_timeout_seconds=5,
        gather_poll_pause_seconds=0,
        gather_inline_budget_seconds=5.0,
    )
    twilio = MagicMock()
    twilio.turn_url.return_value = "https://example.test/turn"
    twilio.turn_result_url.return_value = "https://example.test/turn-result"
    flow = TurnFlow(
        settings=settings,
        session_factory=MagicMock(),
        engine=MagicMock(),
        sarvam=MagicMock(),
        twilio=twilio,
        registry=CallRegistry(),
    )
    attrs = flow._gather_attrs("https://example.test/turn", "en")
    assert 'speechTimeout="1"' in attrs
    assert 'timeout="5"' in attrs

    from backend.telephony.turn_flow import PendingTurn

    pending = PendingTurn(call_sid="CA1", token="tok", language="en")
    twiml = flow._pending_redirect_twiml(pending)
    assert "<Pause" not in twiml
    assert "<Redirect" in twiml


@pytest.mark.asyncio
async def test_phone_transport_uses_phone_model_and_compact(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return structured_json(
            "Got it — what should I call you?",
            next_state="collecting_identity",
        )

    settings = make_settings(
        tmp_path,
        phone_llm_model="fast-phone-model",
        phone_llm_max_tokens=140,
        phone_prompt_compact=True,
    )
    engine = ConversationEngine(
        make_mock_llm_client(settings, handler),
        settings=settings,
    )
    from backend.database import create_engine_and_session
    from backend.models import Session

    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'phone.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="en", current_state="collecting_identity")
        db.add(session)
        db.commit()
        session_id = session.id

    with factory() as db:
        session = db.get(Session, session_id)
        await engine.process_turn(
            db,
            session,
            "hello",
            TurnTimings(settings=settings, transport="twilio_gather"),
        )
        db.commit()

    payload = captured["payload"]
    assert payload["model"] == "fast-phone-model"
    assert payload["max_tokens"] == 140
    system = next(m["content"] for m in payload["messages"] if m["role"] == "system")
    assert "Field meanings:" not in system
