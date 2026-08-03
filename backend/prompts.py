"""Prompt construction for the lead-qualification LLM.

Internal prompts are never exposed to the user. The model is asked to reply
with a single JSON object matching the documented structured output shape.
"""

from backend.models import LEAD_FIELDS

FIELD_GLOSSARY = {
    "full_name": "User's name",
    "phone_number": "Contact phone (with country code if given)",
    "email": "Contact email",
    "company_name": "Company or business name",
    "job_title": "User's role/title",
    "city": "City",
    "country": "Country",
    "preferred_language": "Language the user wants to continue in",
    "business_type": "Type of business (e.g. retail, services)",
    "product_or_service_interest": "What product/service the user is interested in",
    "business_requirement": "Clear description of the requirement or need",
    "main_problem": "The core problem the user wants to solve",
    "current_solution": "How the user solves it today",
    "estimated_budget": "Approximate budget range",
    "purchase_timeline": "When they plan to buy (e.g. this month, 3 months)",
    "decision_maker_status": "Whether the user decides on purchases (yes/no/role)",
    "team_size": "Number of employees",
    "preferred_contact_method": "phone/email/whatsapp etc.",
    "preferred_contact_time": "Best time to contact",
    "additional_notes": "Anything else useful",
    "consent_to_contact": "yes/no - explicit consent to be contacted",
}

REFUSAL_TOKEN = "__refused__"

SYSTEM_PROMPT_TEMPLATE = """You are a friendly, natural lead-qualification voice agent.
You collect details for a sales follow-up by talking to the user in a short, warm,
conversational way - NOT by interrogating them.

Rules:
- Ask ONE main question per turn, and keep responses to 1-3 short sentences.
- Acknowledge the user's answer naturally before asking the next question.
- Detect the user's language and reply in the same language. Support English,
  Hindi, and Hinglish. Never randomly mix languages. Hinglish is only used when
  the user speaks Hinglish.
- Do NOT ask for information already collected. Skip optional fields if the user
  refuses (then record that field as "{refusal}").
- If the user corrects earlier information, put the field in extracted_fields with
  the corrected value, and add the old field name to fields_to_clear if it should
  be replaced (the new value in extracted_fields overwrites it anyway).
- NEVER fabricate or guess a field the user did not provide.
- The current state tells you what stage you are at. Suggest a sensible next_state,
  but the backend validates it.
- When you have identity + contact + a requirement, ask for consent explicitly
  ("may I contact you...?"). Set consent_to_contact to yes/no when answered.
- When consent is answered, summarize the collected details and ask the user to
  confirm or correct. Set needs_confirmation=true while reviewing the summary.
- Only set conversation_complete=true after the user has confirmed the summary.
- Keep the tone casual, short and human. Avoid formal/robotic phrasing.

Reply with ONE JSON object only - no prose, no markdown fences - matching EXACTLY:
{{
  "assistant_message": "your reply to the user",
  "detected_language": "language code (en, hi, en-hi for Hinglish, ...)",
  "extracted_fields": {{ "field_name": "value the user provided" }},
  "fields_to_clear": ["field_name", ...],
  "next_state": "one of the allowed states below",
  "conversation_complete": false,
  "needs_confirmation": false
}}

Allowed field names (use only these): {fields}
Allowed states: {states}

To mark a refused optional field: "extracted_fields": {{ "field_name": "{refusal}" }}.
"""

REPAIR_INSTRUCTION = (
    "\n\nYour previous reply was not valid JSON. Reply with ONLY a single JSON "
    "object matching the required schema. No prose, no markdown fences."
)


def build_system_prompt() -> str:
    fields_csv = ", ".join(LEAD_FIELDS)
    states_csv = ", ".join(
        [
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
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        fields=fields_csv, states=states_csv, refusal=REFUSAL_TOKEN
    )


def current_field_state_text(fields: dict, skipped: list[str]) -> str:
    lines = []
    for name in LEAD_FIELDS:
        value = fields.get(name)
        if value:
            lines.append(f"{name}: {value}")
        elif name in skipped:
            lines.append(f"{name}: (user refused)")
    return "\n".join(lines) if lines else "(none collected yet)"


def build_messages(
    history: list[dict],
    fields: dict,
    skipped: list[str],
    current_state: str,
    *,
    repair: bool = False,
    include_field_glossary: bool = True,
) -> list[dict]:
    state_block = (
        "\n\nCurrent state: "
        + current_state
        + "\nCollected fields so far:\n"
        + current_field_state_text(fields, skipped)
        + "\n\nUse the history to continue the conversation naturally. "
        + "Ask the next single question based on what is still missing."
    )
    system = build_system_prompt()
    if include_field_glossary:
        glossary = "\nField meanings:\n" + "\n".join(
            f"- {k}: {v}" for k, v in FIELD_GLOSSARY.items()
        )
        system += glossary
    messages: list[dict] = [{"role": "system", "content": system}]
    # Include recent history only to stay within local model context windows.
    for msg in history[-12:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": state_block})
    if repair:
        messages.append({"role": "user", "content": REPAIR_INSTRUCTION})
    return messages
