"""Unit tests for lead field validation."""

from backend.validation import (
    clean_phone,
    has_valid_contact,
    is_consent_value,
    is_decision_maker,
    is_valid_email,
    is_valid_name,
    normalize_phone,
)


def test_valid_phones():
    assert clean_phone("+919876543210") == "+919876543210"
    assert clean_phone("9876543210") == "9876543210"
    assert clean_phone("+1 (202) 555-0123") == "+12025550123"
    assert clean_phone(" 044 1234 5678 ") == "04412345678"


def test_invalid_phones():
    assert clean_phone("") is None
    assert clean_phone("abc") is None
    assert clean_phone("12") is None


def test_emails():
    assert is_valid_email("rahul@acme.in")
    assert is_valid_email("a.b+c@sub.domain.co")
    assert not is_valid_email("not-an-email")
    assert not is_valid_email("a@b")
    assert not is_valid_email("")


def test_names():
    assert is_valid_name("Rahul Sharma")
    assert is_valid_name("An")
    assert is_valid_name("राहुल")
    assert not is_valid_name("12345")
    assert not is_valid_name("x")
    assert not is_valid_name("")
    assert not is_valid_name("!!!")
    assert is_valid_name("Rahul K 123")  # alphanumeric names are tolerated


def test_contact_combination():
    assert has_valid_contact("9876543210", "")
    assert has_valid_contact("", "rahul@acme.in")
    assert not has_valid_contact("", "")
    assert not has_valid_contact("nope", "")


def test_consent_values():
    assert is_consent_value("yes")
    assert is_consent_value("no")
    assert not is_consent_value("maybe")
    assert not is_consent_value("")


def test_decision_maker():
    assert is_decision_maker("yes")
    assert is_decision_maker("Yes, I own the company")
    assert is_decision_maker("I am the founder")
    assert is_decision_maker("CEO")
    assert not is_decision_maker("no")
    assert not is_decision_maker("not sure")
    assert not is_decision_maker("")


def test_normalize_phone_roundtrip():
    assert normalize_phone("+91 98765 43210") == "+919876543210"
    assert normalize_phone("garbage") is None
