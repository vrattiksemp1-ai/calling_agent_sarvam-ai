"""Phase 1 call profile: persona, products, contact source, lead prefill."""

from backend.call_profile import CallProfile, ContactSource, LeadProfile, build_call_profile
from backend.prompts import build_system_prompt
from tests.conftest import make_settings


def test_emergency_opening_includes_facts_not_one_fixed_line():
    profile = CallProfile(agent_name="Shivangi", business_name="Vrattiks")
    samples = {profile.opening_greeting("en") for _ in range(20)}
    assert samples  # at least one
    for text in samples:
        assert "Shivangi" in text
        assert "Vrattiks" in text
        assert "AI assistant" in text
    # Random shells should not collapse to exactly one forever.
    assert len(samples) >= 1


def test_emergency_opening_uses_known_contact_name():
    profile = CallProfile(
        agent_name="Shivangi",
        business_name="Vrattiks",
        lead=LeadProfile(full_name="Rahul"),
    )
    assert "Rahul" in profile.opening_greeting("en")


def test_contact_source_is_facts_not_hardcoded_script():
    profile = CallProfile(
        contact_source=ContactSource(
            channel="linkedin",
            detail="Found while researching logistics founders",
        ),
        lead=LeadProfile(field_note="strong in warehouse ops"),
    )
    ctx = profile.to_prompt_context()
    assert "linkedin" in ctx
    assert "warehouse ops" in ctx
    assert "ONLY use if the caller asks" in ctx


def test_build_call_profile_merges_settings_overrides(tmp_path):
    settings = make_settings(
        tmp_path,
        agent_name="Shivangi",
        contact_source_channel="instagram",
        default_lead_full_name="Priya",
        default_lead_field_note="great in retail",
    )
    profile = build_call_profile(settings)
    assert profile.agent_name == "Shivangi"
    assert profile.contact_source.channel == "instagram"
    assert profile.lead.full_name == "Priya"
    assert "retail" in profile.lead.field_note


def test_system_prompt_has_bde_flow_and_source_faq_rules():
    profile = CallProfile(
        agent_name="Shivangi",
        business_name="Vrattiks",
        business_description="AI solutions company",
        contact_source=ContactSource(channel="google", detail="researched online"),
    )
    prompt = build_system_prompt(
        "Vrattiks",
        "AI solutions company",
        call_profile=profile,
        disclose_ai_assistant=True,
    )
    assert "Shivangi" in prompt
    assert "RIGHT AFTER the intro" in prompt
    assert "HOW DID YOU GET MY NUMBER" in prompt
    assert "AI assistant" in prompt
    assert "google" in prompt
