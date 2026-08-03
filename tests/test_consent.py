"""Consent handling tests: flow-level, driven through the API with mocked LLM."""

import json

import httpx
import pytest

from backend.conversation import ConversationEngine
from tests.conftest import make_mock_llm_client, make_settings, structured_json, user_utterance
from backend.metrics import TurnTimings
from backend.models import Session, Lead


def _consent_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        lower = user_utterance(payload["messages"]).lower()
        if "contact" in lower and ("may" in lower or "?" in lower):
            return structured_json(
                "May I contact you later about this? Please answer yes or no.",
                extracted_fields={},
                next_state="requesting_consent",
            )
        if "confirm" in lower:
            return structured_json(
                "Saved. Goodbye!",
                extracted_fields={},
                next_state="completed",
                conversation_complete=True,
            )
        if "yes" in lower or "no" in lower:
            consent = "yes" if ("yes" in lower and "no" not in lower) else "no"
            return structured_json(
                "Thanks. Here is your summary - confirm yes or correct me.",
                extracted_fields={"consent_to_contact": consent},
                next_state="reviewing_summary",
            )
        return structured_json("What is your name?", next_state="collecting_identity")

    return handler


def _basics(consent=None):
    fields = {
        "full_name": "Rahul",
        "phone_number": "9876543210",
        "business_requirement": "need crm",
    }
    if consent:
        fields["consent_to_contact"] = consent
    return fields


def _make_engine_and_session(tmp_path, name):
    from backend.database import create_engine_and_session

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, _consent_handler())
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / name).replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="en")
        db.add(session)
        db.commit()
    return ConversationEngine(llm), factory, session.id


@pytest.mark.asyncio
async def test_consent_required_before_completion(tmp_path):
    from backend.llm_parsing import LLMStructuredResponse

    engine, factory, sid = _make_engine_and_session(tmp_path, "consent.db")
    with factory() as db:
        session = db.get(Session, sid)
        lead = Lead(session_id=session.id)
        db.add(lead)
        db.flush()
        parsed = LLMStructuredResponse(
            assistant_message="ok",
            extracted_fields=_basics(),
            next_state="collecting_requirement",
        )
        engine._apply_extraction(db, lead, parsed, session)
        session.current_state = "requesting_consent"
        db.commit()

        lead, parsed = await engine.process_turn(db, session, "yes you can contact me", TurnTimings())
        db.commit()
        assert lead.consent_to_contact == "yes"
        assert lead.consent_confirmed is True
        assert session.current_state == "reviewing_summary"
        assert session.status != "completed"

        lead, parsed = await engine.process_turn(db, session, "yes I confirm", TurnTimings())
        db.commit()
        assert session.current_state == "completed"
        assert session.status == "completed"
        assert lead.conversation_status == "completed"
        assert lead.summary_confirmed is True


@pytest.mark.asyncio
async def test_consent_no_blocks_completion(tmp_path):
    from backend.conversation import ConversationEngine as BaseEngine
    from backend.llm_parsing import LLMStructuredResponse

    engine, factory, sid = _make_engine_and_session(tmp_path, "consent_no.db")

    class BadLlmTurnEngine(BaseEngine):
        async def _llm_turn(self, db, session, lead, user_text, history, timings):
            class _Bad:
                assistant_message = "done"
                detected_language = "en"
                extracted_fields = {}
                fields_to_clear = []
                next_state = "completed"
                conversation_complete = True
                needs_confirmation = False

            return _Bad()

    bad_engine = BadLlmTurnEngine(engine._llm)

    with factory() as db:
        session = db.get(Session, sid)
        lead = Lead(session_id=session.id)
        db.add(lead)
        db.flush()
        parsed = LLMStructuredResponse(
            assistant_message="ok",
            extracted_fields=_basics("no"),
            next_state="requesting_consent",
        )
        engine._apply_extraction(db, lead, parsed, session)
        session.current_state = "requesting_consent"
        db.commit()

        # LLM claims the conversation is complete, but consent is "no":
        # the backend must refuse to complete.
        lead, parsed = await bad_engine.process_turn(db, session, "no thanks", TurnTimings())
        db.commit()
        assert session.current_state != "completed"
        assert session.status != "completed"
