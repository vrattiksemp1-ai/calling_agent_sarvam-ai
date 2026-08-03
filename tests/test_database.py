"""Database tests: schema creation, persistence, cascades."""

import pytest

from backend.database import create_engine_and_session
from backend.models import (
    LEAD_FIELDS,
    Lead,
    LeadFieldHistory,
    Message,
    ProviderEvent,
    Session,
)


@pytest.fixture
def factory(tmp_path):
    url = f"sqlite:///{str(tmp_path / 'db.db').replace(chr(92), '/')}"
    _, factory = create_engine_and_session(url)
    return factory


def test_lead_fields_list_is_complete():
    assert len(LEAD_FIELDS) == 21
    assert "consent_to_contact" in LEAD_FIELDS
    assert "full_name" in LEAD_FIELDS


def test_create_session_and_lead(factory):
    with factory() as db:
        session = Session(language="en")
        db.add(session)
        db.flush()
        lead = Lead(session_id=session.id, full_name="Rahul Sharma")
        db.add(lead)
        db.commit()

        sid = session.id
    with factory() as db:
        session = db.get(Session, sid)
        assert session.lead.full_name == "Rahul Sharma"


def test_lead_field_history(factory):
    with factory() as db:
        session = Session(language="en")
        db.add(session)
        db.flush()
        lead = Lead(session_id=session.id, full_name="Rahul")
        db.add(lead)
        db.flush()
        db.add(LeadFieldHistory(lead_id=lead.id, field_name="full_name", old_value="Rahul", new_value="Rahul Sharma"))
        db.commit()
        lid = lead.id
    with factory() as db:
        lead = db.get(Lead, lid)
        assert len(lead.field_history) == 1
        assert lead.field_history[0].new_value == "Rahul Sharma"


def test_messages_and_provider_events(factory):
    with factory() as db:
        session = Session(language="en")
        db.add(session)
        db.flush()
        db.add(Message(session_id=session.id, role="user", content="hi"))
        db.add(Message(session_id=session.id, role="assistant", content="hello"))
        db.add(ProviderEvent(session_id=session.id, provider="sarvam", event_type="stt", status="ok"))
        db.commit()
        sid = session.id
    with factory() as db:
        session = db.get(Session, sid)
        assert len(session.messages) == 2
        assert len(session.provider_events) == 1


def test_delete_session_cascades(factory):
    with factory() as db:
        session = Session(language="en")
        db.add(session)
        db.flush()
        db.add(Lead(session_id=session.id, full_name="X"))
        db.add(Message(session_id=session.id, role="user", content="hi"))
        db.add(ProviderEvent(session_id=session.id, provider="llm", event_type="llm", status="ok"))
        db.commit()
        sid = session.id
    with factory() as db:
        session = db.get(Session, sid)
        db.delete(session)
        db.commit()
    with factory() as db:
        assert db.get(Session, sid) is None
        assert db.query(Message).filter(Message.session_id == sid).count() == 0
        assert db.query(Lead).filter(Lead.session_id == sid).count() == 0


def test_lead_defaults(factory):
    with factory() as db:
        session = Session(language="en")
        db.add(session)
        db.flush()
        lead = Lead(session_id=session.id)
        db.add(lead)
        db.commit()
        assert lead.qualification_score == 0
        assert lead.qualification_level == "cold"
        assert lead.conversation_status == "in_progress"
