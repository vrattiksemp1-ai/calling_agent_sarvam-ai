"""TTS speaker gender is injected into the LLM prompt."""

from backend.call_profile import CallProfile
from backend.conversation import ConversationEngine
from backend.prompts import build_system_prompt
from backend.tts_persona import resolve_tts_gender, voice_persona_prompt_block
from tests.conftest import make_mock_llm_client, make_settings


def test_resolve_tts_gender_from_speaker_catalog():
    assert resolve_tts_gender("ritu") == "female"
    assert resolve_tts_gender("priya") == "female"
    assert resolve_tts_gender("shubh") == "male"
    assert resolve_tts_gender("ratan") == "male"


def test_resolve_tts_gender_override_wins():
    assert resolve_tts_gender("ritu", "male") == "male"
    assert resolve_tts_gender("shubh", "female") == "female"


def test_full_prompt_uses_female_agreement_for_ritu():
    prompt = build_system_prompt(tts_speaker="ritu")
    assert "VOICE PERSONA" in prompt
    assert 'TTS speaker "ritu"' in prompt
    assert "a woman" in prompt
    assert "कर रही हूँ" in prompt
    assert "Shivangi only" in prompt
    assert "You are a woman named" not in prompt


def test_compact_prompt_uses_male_agreement_for_shubh():
    prompt = build_system_prompt(
        compact=True,
        tts_speaker="shubh",
        agent_name="Amit",
        call_profile=CallProfile(agent_name="Amit", business_name="Vrattiks"),
    )
    assert "VOICE PERSONA" in prompt
    assert "a man" in prompt
    assert "shubh" in prompt
    assert "कर रहा हूँ" in prompt
    assert "Amit" in prompt


def test_persona_block_does_not_use_speaker_as_spoken_name():
    block = voice_persona_prompt_block("Shivangi", "ritu", "female")
    assert "Shivangi" in block
    assert "Introduce yourself as Shivangi" in block


def test_engine_passes_configured_speaker_into_prompt(tmp_path):
    settings = make_settings(tmp_path, sarvam_tts_speaker="shubh")
    client = make_mock_llm_client(settings, lambda request: None)
    engine = ConversationEngine(client, settings=settings)
    kwargs = engine._prompt_kwargs(compact=True)
    assert kwargs["tts_speaker"] == "shubh"
    assert kwargs["tts_gender"] == "male"
