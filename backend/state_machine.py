"""Conversation state machine.

Backend code controls allowed state transitions and completion criteria.
The LLM may *suggest* next_state, but the backend validates it and overrides
invalid or premature suggestions.
"""

from dataclasses import dataclass

from backend import validation
from backend.scoring import completion_basics_met

STATES = [
    "greeting",
    "collecting_identity",
    "collecting_contact",
    "collecting_business_context",
    "collecting_requirement",
    "collecting_budget",
    "collecting_timeline",
    "collecting_authority",
    "collecting_preferences",
    "requesting_consent",
    "reviewing_summary",
    "completed",
    "abandoned",
]

PIPELINE = [
    "collecting_identity",
    "collecting_contact",
    "collecting_business_context",
    "collecting_requirement",
    "collecting_budget",
    "collecting_timeline",
    "collecting_authority",
    "collecting_preferences",
]

TERMINAL = {"completed", "abandoned"}


@dataclass(frozen=True)
class TransitionResult:
    state: str
    allowed: bool
    reason: str = ""


def is_valid_state(state: str) -> bool:
    return state in STATES


def transition_allowed(current: str, next_state: str, fields: dict[str, str | None]) -> TransitionResult:
    """Decide whether the LLM's suggested next_state is acceptable."""
    if not is_valid_state(next_state):
        return TransitionResult(current, False, f"Unknown state: {next_state}")
    if current in TERMINAL:
        return TransitionResult(current, False, "Session is already finished.")
    if next_state == current:
        return TransitionResult(current, True, "")

    if next_state in PIPELINE and current in PIPELINE:
        # Forward (or same-stage) only — never rewind the sales pipeline.
        # Rewinds caused loops like collecting_contact → collecting_identity
        # after a JSON repair, which made the agent re-ask for name/time.
        cur_idx = PIPELINE.index(current)
        nxt_idx = PIPELINE.index(next_state)
        if nxt_idx >= cur_idx:
            return TransitionResult(next_state, True, "")
        return TransitionResult(
            current,
            False,
            f"Cannot move backward from {current} to {next_state}.",
        )

    if next_state == "requesting_consent":
        if not completion_basics_met(fields):
            return TransitionResult(
                current,
                False,
                "Consent must not be requested until identity, contact and a requirement are known.",
            )
        return TransitionResult(next_state, True, "")

    if next_state == "reviewing_summary":
        if not completion_basics_met(fields):
            return TransitionResult(
                current,
                False,
                "Summary requires identity, contact and a requirement to be known.",
            )
        if not validation.consent_bool(fields.get("consent_to_contact")):
            return TransitionResult(
                current,
                False,
                "Consent must be answered 'yes' before the summary is reviewed.",
            )
        return TransitionResult(next_state, True, "")

    if next_state == "completed":
        all_criteria = completion_basics_met(fields) and validation.consent_bool(
            fields.get("consent_to_contact")
        )
        if current != "reviewing_summary" or not all_criteria:
            return TransitionResult(
                current,
                False,
                "Session can only be completed from reviewing_summary with all criteria met.",
            )
        return TransitionResult(next_state, True, "")

    if next_state == "abandoned":
        return TransitionResult(next_state, True, "")

    return TransitionResult(current, False, f"No legal transition from {current} to {next_state}.")


def default_next_state(current: str, fields: dict[str, str | None]) -> str:
    """Backend fallback when the LLM suggestion is rejected."""
    if current == "greeting":
        return "collecting_identity"
    if current in PIPELINE:
        order = PIPELINE
        idx = order.index(current)
        # Skip optional-ish stages only when there is nothing to collect there.
        if current == "collecting_preferences" or idx == len(order) - 1:
            if completion_basics_met(fields) and validation.consent_bool(
                fields.get("consent_to_contact")
            ):
                return "reviewing_summary"
            return "requesting_consent"
        return order[idx + 1]
    if current == "requesting_consent":
        if validation.consent_bool(fields.get("consent_to_contact")):
            return "reviewing_summary"
        return current
    if current == "reviewing_summary":
        return current
    return current
