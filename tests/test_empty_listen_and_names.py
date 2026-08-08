"""Empty-listen prompts, name spelling, and soft re-intro detection."""

from backend.call_profile import CallProfile
from backend.conversation import looks_like_reintroduction, normalize_spoken_names


def test_pitch_followup_is_not_reintroduction():
    pitch = (
        "Nice to meet you, Dhyani. I'm calling from Vrattiks, a tech company "
        "that builds AI solutions. Do you currently use any software for "
        "lead qualification?"
    )
    assert not looks_like_reintroduction(
        pitch, agent_name="Shivangi", business_name="Vrattiks"
    )


def test_gujarati_opening_is_reintroduction():
    gu = (
        "હું શિવાંગી બોલું છું, વ્રત્તિક્સ માંથી ફોન કરી રહી છું. "
        "થોડો સમય છે?"
    )
    assert looks_like_reintroduction(
        gu, agent_name="Shivangi", business_name="Vrattiks"
    )


def test_normalize_spoken_names_fixes_gujarati():
    bad = "હું શિવંગી છું, વ્રટિક્સથી બોલું છું."
    fixed = normalize_spoken_names(bad, "gu")
    assert "શિવાંગી" in fixed
    assert "વ્રત્તિક્સ" in fixed
    assert "શિવંગી" not in fixed
    assert "વ્રટિક્સ" not in fixed


def test_call_profile_spoken_names():
    profile = CallProfile()
    assert profile.spoken_agent_name("gu") == "શિવાંગી"
    assert profile.spoken_business_name("gu") == "વ્રત્તિક્સ"
    assert "શિવાંગી" in profile.opening_greeting("gu")
