"""Session language persistence tests (ConversationEngine.process_turn)."""

import json

import httpx
import pytest

from backend.conversation import ConversationEngine
from backend.metrics import TurnTimings
from backend.models import Session
from backend.telephony.call_manager import fallback_text
from tests.conftest import make_mock_llm_client, make_settings, structured_json


def _handler(detected_language: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            "Kem cho? Tamaru naam shu chhe?",
            detected_language=detected_language,
            extracted_fields={},
            next_state="collecting_identity",
        )

    return handler


def _make_engine_and_session(tmp_path, name, detected_language):
    from backend.database import create_engine_and_session

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, _handler(detected_language))
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / name).replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="en")
        db.add(session)
        db.commit()
    return ConversationEngine(llm), factory, session.id


@pytest.mark.asyncio
async def test_process_turn_persists_detected_language(tmp_path):
    engine, factory, session_id = _make_engine_and_session(
        tmp_path, "lang.db", detected_language="gu"
    )
    with factory() as db:
        session = db.get(Session, session_id)
        lead, parsed = await engine.process_turn(
            db, session, "kem cho maru naam rahul chhe", TurnTimings()
        )
        assert parsed.detected_language == "gu"
        db.commit()
    with factory() as db:
        assert db.get(Session, session_id).language == "gu"


@pytest.mark.asyncio
async def test_process_turn_keeps_default_language(tmp_path):
    engine, factory, session_id = _make_engine_and_session(
        tmp_path, "lang_en.db", detected_language="en"
    )
    with factory() as db:
        session = db.get(Session, session_id)
        await engine.process_turn(db, session, "hello there", TurnTimings())
        db.commit()
    with factory() as db:
        assert db.get(Session, session_id).language == "en"


def test_fallback_text_is_language_aware():
    assert "નમસ્તે" in fallback_text("greeting", "gu")
    assert "નમસ્તે" in fallback_text("greeting", "gu-in")
    assert "माफ़" in fallback_text("repeat", "hi")
    assert "thank" in fallback_text("goodbye", "en").lower()
    assert "I'm sorry" in fallback_text("repeat", None)


@pytest.mark.asyncio
async def test_process_turn_abandoned_sets_session_status(tmp_path):
    from backend.database import create_engine_and_session

    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            "Theek che, aavjo! Bye.",
            detected_language="gu",
            extracted_fields={},
            next_state="abandoned",
        )

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, handler)
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'abandon.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="gu")
        db.add(session)
        db.commit()
    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session.id)
        await engine.process_turn(db, session, "bye, hang up the call", TurnTimings())
        db.commit()
    with factory() as db:
        s = db.get(Session, session.id)
        assert s.status == "abandoned"
        assert s.current_state == "abandoned"
        assert s.lead.conversation_status == "abandoned"


def test_system_prompt_has_human_language_and_hangup_rules():
    from backend.prompts import build_system_prompt

    prompt = build_system_prompt()
    assert "TONE AND EMOTION" in prompt
    assert "CLARIFICATION" in prompt
    assert "ENDING THE CALL" in prompt
    assert "switches language mid-conversation" in prompt
    assert "Never ask again for a field" in prompt
    assert "abandoned" in prompt
    assert "warm, professional" in prompt
    assert "natural spoken fillers" in prompt
    assert "overly chummy" in prompt
    assert "everyday spoken Gujarati" in prompt
    assert "decide by INTENT, not by matching words" in prompt
    assert "preferred_contact_time" in prompt
