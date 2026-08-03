"""Unit tests for deterministic lead scoring."""

from backend.scoring import level_for_score, score_lead


def test_empty_lead_scores_zero():
    result = score_lead({})
    assert result.score == 0
    assert result.level == "cold"
    assert "full_name" in result.missing_important_fields
    assert "consent_to_contact" in result.missing_important_fields


def test_max_score_hot():
    fields = {
        "full_name": "Rahul Sharma",
        "phone_number": "+91 9876543210",
        "email": "rahul@acme.in",
        "business_requirement": "Need CRM with voice automation for 50 agents",
        "product_or_service_interest": "CRM software",
        "estimated_budget": "Rs 50000 per year",
        "purchase_timeline": "within 3 months",
        "decision_maker_status": "yes, I am the owner",
        "company_name": "Acme Retail",
        "consent_to_contact": "yes",
    }
    result = score_lead(fields)
    assert result.score == 100
    assert result.level == "hot"
    assert result.missing_important_fields == []


def test_name_and_contact_only_is_warm_lower():
    fields = {
        "full_name": "Priya Patel",
        "phone_number": "9876543210",
        "business_requirement": "Looking for accounting software",
        "consent_to_contact": "no",
    }
    result = score_lead(fields)
    # 5 name + 15 contact + 15 requirement + 10 consent? no -> consent is "no"
    assert result.score == 35
    assert result.level == "cold"


def test_consent_yes_adds_ten():
    fields = {
        "full_name": "Priya Patel",
        "phone_number": "9876543210",
        "business_requirement": "Looking for accounting software",
        "consent_to_contact": "yes",
    }
    result = score_lead(fields)
    assert result.score == 45
    assert result.level == "warm"


def test_missing_contact_flags_email_when_phone_empty():
    result = score_lead({"full_name": "Rahul Sharma", "email": ""})
    assert "phone_number" in result.missing_important_fields


def test_level_boundaries():
    assert level_for_score(0) == "cold"
    assert level_for_score(39) == "cold"
    assert level_for_score(40) == "warm"
    assert level_for_score(69) == "warm"
    assert level_for_score(70) == "hot"
    assert level_for_score(100) == "hot"


def test_invalid_name_not_scored():
    result = score_lead({"full_name": "12345"})
    assert "full_name" in result.missing_important_fields
    assert result.score == 0


def test_recommended_action_by_level():
    assert "nurture" in score_lead({}).recommended_next_action.lower()
    hot = score_lead(
        {
            "full_name": "A B",
            "phone_number": "1234567890",
            "business_requirement": "need a full marketing automation platform for our 200 person team",
            "product_or_service_interest": "marketing platform",
            "estimated_budget": "100k",
            "purchase_timeline": "now",
            "decision_maker_status": "yes",
            "company_name": "X",
            "consent_to_contact": "yes",
        }
    )
    assert hot.level == "hot"
    assert "route" in hot.recommended_next_action.lower()
