"""Export tests: JSON and CSV output for leads."""

import csv
import io
import json

import pytest

from backend.database import create_engine_and_session
from backend.exports import (
    lead_to_csv_bytes,
    lead_to_json_bytes,
    leads_to_csv_bytes,
    leads_to_json_bytes,
)
from backend.models import Lead, Session


@pytest.fixture
def lead():
    with create_engine_and_session("sqlite:///:memory:")[1]() as db:
        session = Session(language="en")
        db.add(session)
        db.flush()
        l = Lead(
            session_id=session.id,
            full_name="Rahul Sharma",
            phone_number="+919876543210",
            email="rahul@acme.in",
            company_name="Acme Retail",
            qualification_score=100,
            qualification_level="hot",
            conversation_status="completed",
            consent_to_contact="yes",
        )
        db.add(l)
        db.commit()
        return l


def test_json_export_contains_fields(lead):
    data = json.loads(lead_to_json_bytes(lead).decode("utf-8"))
    assert data["full_name"] == "Rahul Sharma"
    assert data["phone_number"] == "+919876543210"
    assert data["qualification_level"] == "hot"
    assert data["lead_id"] == lead.id


def test_csv_export_has_header_and_row(lead):
    content = lead_to_csv_bytes(lead).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["full_name"] == "Rahul Sharma"
    assert rows[0]["qualification_score"] == "100"


def test_multiple_leads_export(lead):
    with create_engine_and_session("sqlite:///:memory:")[1]() as db:
        s2 = Session(language="en")
        db.add(s2)
        db.flush()
        l2 = Lead(session_id=s2.id, full_name="Priya Patel", email="priya@acme.in")
        db.add(l2)
        db.commit()
        # Build a list of leads by reconstructing objects
        leads = [lead, l2]
    data = json.loads(leads_to_json_bytes(leads).decode("utf-8"))
    assert len(data) == 2
    csv_bytes = leads_to_csv_bytes(leads)
    assert csv_bytes.count(b"full_name") >= 1


def test_empty_list_export():
    assert leads_to_json_bytes([]) == b"[]"
    assert leads_to_csv_bytes([]) == b""


def test_indian_chars_preserved():
    with create_engine_and_session("sqlite:///:memory:")[1]() as db:
        session = Session(language="en")
        db.add(session)
        db.flush()
        l = Lead(session_id=session.id, full_name="राहुल", additional_notes="हिंदी टिप्पणी")
        db.add(l)
        db.commit()
        data = json.loads(lead_to_json_bytes(l).decode("utf-8"))
        assert data["full_name"] == "राहुल"
