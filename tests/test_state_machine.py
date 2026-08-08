"""Unit tests for the conversation state machine."""

from backend import state_machine
from backend.state_machine import default_next_state, is_valid_state, transition_allowed


def test_all_states_valid():
    for state in state_machine.STATES:
        assert is_valid_state(state)


def test_pipeline_forward_not_backward():
    fields = {"full_name": "Rahul", "phone_number": "9876543210",
              "business_requirement": "need crm", "consent_to_contact": "yes"}
    r = transition_allowed("collecting_identity", "collecting_contact", fields)
    assert r.allowed
    r = transition_allowed("collecting_contact", "collecting_identity", fields)
    assert not r.allowed
    assert r.state == "collecting_contact"


def test_greeting_only_advances_to_identity():
    r = transition_allowed("greeting", "collecting_requirement", {})
    assert not r.allowed
    assert r.state == "greeting"


def test_consent_blocked_until_basics_met():
    fields_empty = {}
    r = transition_allowed("collecting_requirement", "requesting_consent", fields_empty)
    assert not r.allowed


def test_consent_allowed_when_basics_met():
    fields = {"full_name": "Rahul", "phone_number": "9876543210",
              "business_requirement": "need crm"}
    r = transition_allowed("collecting_requirement", "requesting_consent", fields)
    assert r.allowed


def test_reviewing_summary_requires_consent():
    fields = {"full_name": "Rahul", "phone_number": "9876543210",
              "business_requirement": "need crm"}
    r = transition_allowed("requesting_consent", "reviewing_summary", fields)
    assert not r.allowed
    fields["consent_to_contact"] = "yes"
    r = transition_allowed("requesting_consent", "reviewing_summary", fields)
    assert r.allowed


def test_completion_only_from_review_with_all_criteria():
    fields = {"full_name": "Rahul", "phone_number": "9876543210",
              "business_requirement": "need crm", "consent_to_contact": "yes"}
    r = transition_allowed("collecting_requirement", "completed", fields)
    assert not r.allowed
    r = transition_allowed("reviewing_summary", "completed", fields)
    assert r.allowed


def test_abandon_any_active_state():
    assert transition_allowed("greeting", "abandoned", {}).allowed


def test_terminal_is_immutable():
    assert not transition_allowed("completed", "collecting_identity", {}).allowed
    assert not transition_allowed("abandoned", "reviewing_summary", {}).allowed


def test_unknown_state_rejected():
    r = transition_allowed("greeting", "not_a_state", {})
    assert not r.allowed
    assert r.state == "greeting"


def test_default_next_state():
    assert default_next_state("greeting", {}) == "collecting_identity"
    assert default_next_state("collecting_identity", {}) == "collecting_contact"
    order = state_machine.PIPELINE
    assert default_next_state(order[2], {}) == order[3]
