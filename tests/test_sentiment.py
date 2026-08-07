import json

import httpx
import pytest

from backend.conversation import ConversationEngine
from backend.database import create_engine_and_session
from backend.metrics import TurnTimings
from backend.models import Session
from backend.sentiment import rolling_transcript_style, style_prompt_block
from tests.conftest import make_mock_llm_client, make_settings, structured_json


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I am frustrated, please stop asking the same thing.", "frustrated"),
        ("मैं परेशान हूँ, बार बार मत पूछिए।", "frustrated"),
        ("હું હેરાન છું, વારંવાર એક જ વાત ના પૂછો.", "frustrated"),
        ("I have no time, make it quick.", "rushed"),
        ("बहुत अच्छा, धन्यवाद।", "positive"),
        ("બહુ સારું, આભાર.", "positive"),
    ],
)
def test_transcript_style_signal_across_languages(text, expected):
    signal = rolling_transcript_style([{"role": "user", "content": text}])
    assert signal.label == expected
    assert signal.source == "rolling_transcript"
    assert signal.consequential is False


def test_style_prompt_has_hard_nonconsequential_constraints():
    block = style_prompt_block(
        rolling_transcript_style(["I am frustrated and in a hurry"])
    )
    assert "not acoustic emotion" in block
    assert "ONLY for empathy, wording, and cadence" in block
    assert "Never use it to infer or alter consent" in block
    assert "qualification" in block
    assert "consequential action" in block


@pytest.mark.asyncio
async def test_style_signal_does_not_change_lead_extraction(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured["system"] = next(
            item["content"]
            for item in payload["messages"]
            if item["role"] == "system"
        )
        return structured_json(
            "I understand. What is the best contact number?",
            extracted_fields={
                "full_name": "Anita Rao",
                "business_requirement": "inventory automation",
            },
            next_state="collecting_contact",
        )

    settings = make_settings(tmp_path)
    llm = make_mock_llm_client(settings, handler)
    _, factory = create_engine_and_session(
        f"sqlite:///{str(tmp_path / 'sentiment.db').replace(chr(92), '/')}"
    )
    with factory() as db:
        session = Session(language="en")
        db.add(session)
        db.commit()
        session_id = session.id

    engine = ConversationEngine(llm)
    with factory() as db:
        session = db.get(Session, session_id)
        lead, _ = await engine.process_turn(
            db,
            session,
            "I am frustrated. My name is Anita Rao and I need inventory automation.",
            TurnTimings(settings=settings),
        )
        db.commit()
        assert lead.full_name == "Anita Rao"
        assert lead.business_requirement == "inventory automation"
        assert lead.consent_to_contact is None
        assert lead.consent_confirmed is False
    assert "TRANSCRIPT STYLE SIGNAL" in captured["system"]
    assert "frustrated" in captured["system"]
    assert "Never use it to infer or alter consent" in captured["system"]


@pytest.mark.asyncio
async def test_generated_greeting_always_discloses_ai(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return structured_json(
            "Hello, may I ask your name?",
            next_state="collecting_identity",
        )

    settings = make_settings(tmp_path)
    engine = ConversationEngine(
        make_mock_llm_client(settings, handler),
        business_name="Vrattiks",
        business_description="technology company",
    )
    for language in ("en", "hi", "gu"):
        greeting, _, _ = await engine.generate_greeting(
            TurnTimings(settings=settings), language=language
        )
        assert "AI assistant" in greeting
        assert "Vrattiks" in greeting
