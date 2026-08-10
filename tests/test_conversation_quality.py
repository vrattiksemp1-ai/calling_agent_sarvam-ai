"""Guards for name salvage, bogus email refusal, and truncated LLM JSON."""

import json

import httpx
import pytest

from backend.conversation import (
    ConversationEngine,
    continuity_reply,
    infer_full_name_from_text,
    looks_like_reintroduction,
    looks_like_source_question,
)
from backend.llm_parsing import parse_structured_response
from backend.metrics import TurnTimings
from backend.models import Message, Session
from tests.conftest import make_mock_llm_client, make_settings, structured_json


def test_infer_full_name_latin_and_gujarati_asr():
    assert infer_full_name_from_text("my name is Jay") == "Jay"
    assert infer_full_name_from_text("માય નેમ ઇસ જય") == "જય"
    assert looks_like_source_question("વેર આર યુ કોલિંગ સોંગ")
    assert looks_like_source_question("આ ક્યાં કામ હૈ")


def test_gujarati_reintro_and_source_continuity():
    gu_reintro = "હું શિવાંગી બોલું છું, વૃત્તાંતિક્સ માંથી ફોન કરી રહી છું. થોડો સમય છે?"
    assert looks_like_reintroduction(
        gu_reintro, agent_name="Shivangi", business_name="Vrattiks"
    )
    reply = continuity_reply(
        language="gu",
        business_name="Vrattiks",
        user_text="આ ક્યાં કામ હૈ",
    )
    assert "Vrattiks" in reply
    assert "ફોન" in reply
    bye = continuity_reply(language="gu", business_name="Vrattiks", abandoning=True)
    assert "આવજો" in bye or "આભાર" in bye


def test_salvage_truncated_structured_json():
    raw = (
        '{\n  "assistant_message": "Perfect, I will call at 4 PM.",\n'
        '  "detected_language": "en",\n'
        '  "extracted_fields": {"preferred_contact_time": "today 4:00 PM"},\n'
        '  "fields_to_clear": [],\n'
        '  "next_state": "completed",\n'
        '  "conversation_complete": true,'
    )
    parsed = parse_structured_response(raw)
    assert parsed is not None
    assert "4 PM" in parsed.assistant_message
    assert parsed.extracted_fields["preferred_contact_time"] == "today 4:00 PM"
    assert parsed.conversation_complete is True


def test_salvage_truncated_inside_extracted_fields_with_int_value():
    """Live failure: cut mid-extracted_fields with team_size as a number."""
    raw = (
        '{\n  "assistant_message": "Got it - about 20 people on the team. '
        'Do you already use a CRM?",\n'
        '  "detected_language": "en",\n'
        '  "extracted_fields": {\n'
        '    "team_size": 20,\n'
        '    "preferred_contact_method'
    )
    parsed = parse_structured_response(raw)
    assert parsed is not None
    assert "20" in parsed.assistant_message
    assert parsed.extracted_fields.get("team_size") == "20"
    assert parsed.conversation_complete is False
    assert "preferred_contact_method" not in parsed.extracted_fields
    assert "conversation_complete" not in parsed.extracted_fields


def test_coerce_nested_bools_and_junk_fields():
    raw = json.dumps(
        {
            "assistant_message": "Thanks — noted.",
            "detected_language": "en",
            "extracted_fields": {
                "team_size": 20,
                "conversation_complete": False,
                "needs_confirmation": False,
                "made_up_field": "nope",
                "email": "__refused__",
            },
            "fields_to_clear": [],
            "next_state": "discovery",
            "conversation_complete": False,
            "needs_confirmation": False,
        }
    )
    parsed = parse_structured_response(raw)
    assert parsed is not None
    assert parsed.extracted_fields["team_size"] == "20"
    assert "made_up_field" not in parsed.extracted_fields
    assert parsed.conversation_complete is False


@pytest.mark.asyncio
async def test_process_turn_salvages_name_and_drops_bogus_email_refusal(tmp_path):
    from backend.database import create_engine_and_session

    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            "Jay, thanks. Shall I continue?",
            detected_language="en",
            extracted_fields={"email": "__refused__"},
            next_state="collecting_business_context",
        )

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, handler)
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'name.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="en", current_state="collecting_identity")
        db.add(session)
        db.commit()
        session_id = session.id
    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session_id)
        lead, _ = await engine.process_turn(
            db, session, "my name is Jay", TurnTimings()
        )
        db.commit()
        assert lead.full_name == "Jay"
        assert lead.email is None
        assert "email" not in (session.skipped_fields or [])


@pytest.mark.asyncio
async def test_process_turn_blocks_pipeline_rewind(tmp_path):
    from backend.database import create_engine_and_session
    from backend.models import Lead

    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            "What is your name again?",
            detected_language="en",
            extracted_fields={"preferred_contact_time": "today 4:00 PM"},
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, handler)
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'rewind.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="en", current_state="collecting_contact")
        db.add(session)
        db.flush()
        lead = Lead(session_id=session.id, phone_number="+919999999999")
        db.add(lead)
        db.commit()
        session_id = session.id
    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session_id)
        await engine.process_turn(
            db, session, "today 4:00 PM", TurnTimings()
        )
        db.commit()
        assert session.current_state == "collecting_contact"


@pytest.mark.asyncio
async def test_process_turn_forces_source_answer_over_gu_reintro(tmp_path):
    from backend.database import create_engine_and_session

    reintro = "હું શિવાંગી બોલું છું, વૃત્તાંતિક્સ માંથી ફોન કરી રહી છું. થોડો સમય છે?"
    prior = "હાય, હું શિવાંગી છું."

    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            reintro,
            detected_language="gu",
            extracted_fields={},
            next_state="greeting",
        )

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, handler)
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'source.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="gu", current_state="collecting_identity")
        db.add(session)
        db.flush()
        db.add(Message(session_id=session.id, role="assistant", content=prior))
        db.commit()
        session_id = session.id
    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session_id)
        _, parsed = await engine.process_turn(
            db, session, "આ ક્યાં કામ હૈ", TurnTimings()
        )
        db.commit()
        assert (
            "Vrattiks" in parsed.assistant_message
            or "વૃત્ત" in parsed.assistant_message
        )
        assert "થોડો સમય" not in parsed.assistant_message
        assert session.current_state != "greeting"


@pytest.mark.asyncio
async def test_process_turn_duplicate_abandon_says_goodbye(tmp_path):
    from backend.database import create_engine_and_session

    prior = "સોરી, સમજાયું નહીં. ફરી એક વાર કહો તો?"

    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            prior,
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
        session = Session(language="gu", current_state="collecting_identity")
        db.add(session)
        db.flush()
        db.add(Message(session_id=session.id, role="assistant", content=prior))
        db.commit()
        session_id = session.id
    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session_id)
        _, parsed = await engine.process_turn(
            db, session, "ના", TurnTimings()
        )
        db.commit()
        assert "ફરી" not in parsed.assistant_message
        assert (
            "આવજો" in parsed.assistant_message
            or "આભાર" in parsed.assistant_message
        )
        assert session.current_state == "abandoned"
