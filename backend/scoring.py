"""Deterministic lead qualification scoring.

Scoring lives in backend code, not in the LLM prompt, so two identical leads
always produce the same result.
"""

from dataclasses import dataclass, field

from backend import validation
from backend.models import LEAD_FIELDS

SCORE_NAME = 5
SCORE_CONTACT = 15
SCORE_REQUIREMENT = 15
SCORE_INTEREST = 10
SCORE_BUDGET = 15
SCORE_TIMELINE = 15
SCORE_DECISION_MAKER = 10
SCORE_COMPANY = 5
SCORE_CONSENT = 10
MAX_SCORE = 100

COLD_MAX = 39
WARM_MAX = 69

IMPORTANT_FIELDS = [
    "full_name",
    "phone_number",
    "email",
    "business_requirement",
    "product_or_service_interest",
    "consent_to_contact",
]

REQUIRED_BASICS = ("full_name", "phone_number", "email", "business_requirement", "product_or_service_interest")


@dataclass
class ScoreResult:
    score: int
    level: str
    missing_important_fields: list[str] = field(default_factory=list)
    recommended_next_action: str = ""

    def as_dict(self) -> dict:
        return {
            "qualification_score": self.score,
            "qualification_level": self.level,
            "missing_important_fields": self.missing_important_fields,
            "recommended_next_action": self.recommended_next_action,
        }


def level_for_score(score: int) -> str:
    if score <= COLD_MAX:
        return "cold"
    if score <= WARM_MAX:
        return "warm"
    return "hot"


def action_for_level(level: str) -> str:
    if level == "cold":
        return "Add to nurture list; re-qualify later when timing and budget improve."
    if level == "warm":
        return "Follow up within a few days with a tailored solution and pricing."
    return "Route to sales immediately for a personal follow-up."


def score_lead(fields: dict[str, str | None]) -> ScoreResult:
    score = 0

    name = (fields.get("full_name") or "").strip()
    phone = (fields.get("phone_number") or "").strip()
    email = (fields.get("email") or "").strip()
    requirement = (fields.get("business_requirement") or "").strip()
    interest = (fields.get("product_or_service_interest") or "").strip()
    budget = (fields.get("estimated_budget") or "").strip()
    timeline = (fields.get("purchase_timeline") or "").strip()
    decision = fields.get("decision_maker_status")
    company = (fields.get("company_name") or "").strip()
    consent = (fields.get("consent_to_contact") or "").strip()

    if validation.is_valid_name(name):
        score += SCORE_NAME

    if validation.has_valid_contact(phone, email):
        score += SCORE_CONTACT

    if len(requirement) >= 3:
        score += SCORE_REQUIREMENT
    if len(interest) >= 3:
        score += SCORE_INTEREST
    if len(budget) >= 2:
        score += SCORE_BUDGET
    if len(timeline) >= 2:
        score += SCORE_TIMELINE
    if validation.is_decision_maker(decision):
        score += SCORE_DECISION_MAKER
    if len(company) >= 2:
        score += SCORE_COMPANY
    if validation.consent_bool(consent):
        score += SCORE_CONSENT

    score = max(0, min(MAX_SCORE, score))
    level = level_for_score(score)

    missing = []
    if not validation.is_valid_name(name):
        missing.append("full_name")
    if not validation.has_valid_contact(phone, email):
        missing.append("phone_number" if not phone else "email")
    if len(requirement) < 3 and len(interest) < 3:
        missing.append("business_requirement")
    if not consent:
        missing.append("consent_to_contact")

    return ScoreResult(
        score=score,
        level=level,
        missing_important_fields=sorted(set(missing)),
        recommended_next_action=action_for_level(level),
    )


def completion_basics_met(fields: dict[str, str | None]) -> bool:
    """Name (or identifying label) + a contact method + requirement/interest."""
    name_ok = validation.is_valid_name((fields.get("full_name") or ""))
    contact_ok = validation.has_valid_contact(
        fields.get("phone_number"), fields.get("email")
    )
    req_ok = len((fields.get("business_requirement") or "").strip()) >= 3 or len(
        (fields.get("product_or_service_interest") or "").strip()
    ) >= 3
    return name_ok and contact_ok and req_ok
