"""TTS speaker gender used to keep LLM wording aligned with the spoken voice.

Indic first-person verbs are gendered. If the LLM writes masculine forms and a
female speaker (e.g. ritu) reads them, the call sounds like a man in a woman's
voice. Gender is inferred from the configured Sarvam speaker, with an optional
explicit override.
"""

from __future__ import annotations

# Bulbul v3 catalog (plus a few v2 names still seen in older .env files).
FEMALE_TTS_SPEAKERS = frozenset(
    {
        "ritu",
        "priya",
        "neha",
        "pooja",
        "simran",
        "kavya",
        "ishita",
        "shreya",
        "roopa",
        "tanya",
        "shruti",
        "suhani",
        "kavitha",
        "rupali",
        "amelia",
        "sophia",
        "meera",
        "pavithra",
        "maitreyi",
        "maitri",
        "anushka",
        "manisha",
        "vidya",
        "arya",
    }
)

MALE_TTS_SPEAKERS = frozenset(
    {
        "shubh",
        "aditya",
        "rahul",
        "rohan",
        "amit",
        "dev",
        "ratan",
        "varun",
        "manan",
        "sumit",
        "kabir",
        "aayan",
        "ashutosh",
        "advait",
        "anand",
        "tarun",
        "sunny",
        "mani",
        "gokul",
        "vijay",
        "mohit",
        "rehan",
        "soham",
        "amol",
        "abhilash",
        "karun",
        "hitesh",
    }
)


def normalize_tts_gender(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in {"male", "man", "m", "masculine"}:
        return "male"
    if raw in {"female", "woman", "f", "w", "feminine"}:
        return "female"
    return ""


def resolve_tts_gender(speaker: str | None, override: str | None = None) -> str:
    """Return 'female' or 'male' for the active TTS speaker."""
    forced = normalize_tts_gender(override)
    if forced:
        return forced
    key = (speaker or "").strip().lower()
    if key in MALE_TTS_SPEAKERS:
        return "male"
    if key in FEMALE_TTS_SPEAKERS:
        return "female"
    # Product default is a female BDE voice (ritu / Shivangi).
    return "female"


def voice_persona_prompt_block(
    agent_name: str,
    speaker: str,
    gender: str,
    *,
    compact: bool = False,
) -> str:
    """Instruction block so LLM self-talk matches the TTS voice gender."""
    name = (agent_name or "the agent").strip() or "the agent"
    voice = (speaker or "ritu").strip() or "ritu"
    is_female = resolve_tts_gender(voice, gender) != "male"
    role = "woman" if is_female else "man"
    gender_word = "female" if is_female else "male"

    if compact:
        if is_female:
            agreement = (
                "In Gujarati/Hindi use feminine self-agreement only "
                "(રહી છું / कर रही हूँ). Never masculine self-forms "
                "(રહ્યો / रहा हूँ)."
            )
        else:
            agreement = (
                "In Gujarati/Hindi use masculine self-agreement only "
                "(રહ્યો છું / कर रहा हूँ). Never feminine self-forms "
                "(રહી છું / रही हूँ)."
            )
        return (
            f"- VOICE PERSONA: You are {name}, a {role}. TTS speaker "
            f'"{voice}" is {gender_word}. {agreement} Introduce yourself as '
            f"{name} only — never say the TTS speaker name. Gender the "
            "caller's verbs independently of your own."
        )

    if is_female:
        examples = (
            "Gujarati: રહી છું, કરી રહી છું, બોલું છું, આવી, ગઈ. "
            "Hindi: कर रही हूँ, बोल रही हूँ, आई हूँ. "
            "Never masculine self-forms (રહ્યો, કરી રહ્યો, रहा हूँ, कर रहा हूँ)."
        )
    else:
        examples = (
            "Gujarati: રહ્યો છું, કરી રહ્યો છું, બોલું છું, આવ્યો, ગયો. "
            "Hindi: कर रहा हूँ, बोल रहा हूँ, आया हूँ. "
            "Never feminine self-forms (રહી, કરી રહી, रही हूँ, कर रही हूँ)."
        )

    return (
        f"  - VOICE PERSONA: You are {name}, a {role}. Your reply will be "
        f'spoken by TTS speaker "{voice}" ({gender_word}). Introduce yourself '
        f"as {name} only — never say the TTS speaker name.\n"
        f"    In Gujarati/Hindi always use {gender_word} first-person agreement "
        f"for YOURSELF only. {examples} Do not change the caller's gender."
    )
