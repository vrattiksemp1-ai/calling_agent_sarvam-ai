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
    # Greeting emergency shells are fact-based (may be romanized Indic).
    gu = fallback_text("greeting", "gu")
    assert "AI assistant" in gu
    assert "Shivangi" in gu or "hu " in gu.lower() or "Hello" in gu
    assert "AI assistant" in fallback_text("greeting", "gujlish")
    assert fallback_text("repeat", "hi")
    bye = " ".join(fallback_text("goodbye", "en") for _ in range(8)).lower()
    assert "thank" in bye or "appreciate" in bye or "bye" in bye
    assert fallback_text("repeat", None)


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
    assert "Keep the caller ENGAGED" in prompt
    assert "Stay in ONE language for the whole turn" in prompt
    assert "Follow the caller's LATEST message language" in prompt
    assert "Explicit switch phrases must be obeyed immediately" in prompt
    assert "Never ask again for a field" in prompt
    assert "abandoned" in prompt
    assert "warm, professional" in prompt
    assert "natural spoken fillers" in prompt
    assert "overly chummy" in prompt
    assert "EVERYDAY SPOKEN phone Gujarati" in prompt
    assert "decide by INTENT, not by matching words" in prompt
    assert "preferred_contact_time" in prompt
    assert "Name capture is mandatory" in prompt
    assert "acknowledge" in prompt
    assert "NEVER ask a question and then disconnect in the same turn" in prompt
    assert "Write for the VOICE engine" in prompt
    assert "opening/greeting" in prompt
    assert "do NOT introduce" in prompt


def test_script_language_inference_no_word_lists():
    import backend.language_utils as lu

    assert hasattr(lu, "map_stt_language_code")
    assert hasattr(lu, "infer_script_language")
    assert hasattr(lu, "resolve_turn_language")
    assert hasattr(lu, "detect_explicit_language_switch")
    assert not hasattr(lu, "detect_utterance_language")
    assert not hasattr(lu, "GUJARATI_ROMAN_MARKERS")
    assert not hasattr(lu, "HINDI_ROMAN_MARKERS")
    assert not hasattr(lu, "ENGLISH_WORDS")
    assert lu.infer_script_language("મારે એક ટૂલ જોઈએ છે") == "gu"
    assert lu.infer_script_language("मेरा नाम राहुल है") == "hi"
    assert lu.infer_script_language("hello there") is None
    assert lu.detect_explicit_language_switch("english mein baat karo") == "en"
    assert lu.detect_explicit_language_switch("gujarati ma bolo") == "gu"
    assert (
        lu.resolve_turn_language(
            "Currently we are using spreadsheets.",
            prior_language="gu",
        )
        == "en"
    )
    assert (
        lu.resolve_turn_language(
            "english please speak english",
            prior_language="gu",
        )
        == "en"
    )
    # Gujarati-script ASR of "speaking English" must switch immediately.
    assert lu.detect_explicit_language_switch("સ્પીકિંગ ઇંગલિશ") == "en"
    assert (
        lu.resolve_turn_language(
            "સ્પીકિંગ ઇંગલિશ",
            prior_language="gu",
        )
        == "en"
    )
    assert lu.detect_explicit_language_switch("can we speak in english") == "en"


@pytest.mark.asyncio
async def test_process_turn_switches_when_model_detects_english_on_indic_asr(tmp_path):
    """English spoken through Gujarati-script ASR should be allowed to switch."""
    from backend.database import create_engine_and_session

    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            "Hello! What kind of tool do you need?",
            detected_language="en",
            extracted_fields={},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, handler)
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'gu_script.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="gu")
        db.add(session)
        db.commit()
        session_id = session.id
    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session_id)
        _, parsed = await engine.process_turn(
            db,
            session,
            'મારે એક ઓટો મશીન ટૂલ બનાવવું છે',
            TurnTimings(),
        )
        db.commit()
        assert parsed.detected_language == "en"
        assert session.language == "en"


@pytest.mark.asyncio
async def test_process_turn_fallback_when_model_claims_gu_but_writes_english(tmp_path):
    from backend.database import create_engine_and_session

    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            "Hello! What kind of tool do you need?",
            detected_language="gu",
            extracted_fields={},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, handler)
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'gu_fallback.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="gu")
        db.add(session)
        db.commit()
        session_id = session.id
    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session_id)
        _, parsed = await engine.process_turn(
            db,
            session,
            'મારે એક ઓટો મશીન ટૂલ બનાવવું છે',
            TurnTimings(),
        )
        db.commit()
        assert parsed.detected_language == "gu"
        assert session.language == "gu"
        assert "સોરી" in parsed.assistant_message or "માફ" in parsed.assistant_message or "ફરી" in parsed.assistant_message



@pytest.mark.asyncio
async def test_process_turn_agent_decides_when_no_stt_language(tmp_path):
    from backend.database import create_engine_and_session

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return structured_json(
            "Got it, what exactly are you looking for?",
            detected_language="en",
            extracted_fields={},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, handler)
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'agent.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="gu")
        db.add(session)
        db.commit()
    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session.id)
        await engine.process_turn(
            db, session, "Currently we are using spreadsheets.", TurnTimings()
        )
        db.commit()
    system = next(
        m["content"]
        for m in captured["payload"]["messages"]
        if m["role"] == "system"
    )
    assert "The caller speaks English. Reply entirely in English." in system
    with factory() as db:
        assert db.get(Session, session.id).language == "en"


@pytest.mark.asyncio
async def test_process_turn_pins_stt_language(tmp_path):
    from backend.database import create_engine_and_session

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return structured_json(
            "Hello!",
            detected_language="en",
            extracted_fields={},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, handler)
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'stt.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="gu")
        db.add(session)
        db.commit()
    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session.id)
        await engine.process_turn(
            db, session, "Hello!", TurnTimings(), stt_language="hi"
        )
        db.commit()
    system = next(
        m["content"]
        for m in captured["payload"]["messages"]
        if m["role"] == "system"
    )
    assert "The caller speaks Hindi. Reply entirely in Hindi." in system
    with factory() as db:
        assert db.get(Session, session.id).language == "hi"
