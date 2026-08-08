"""Outbound BDE call profile: agent persona, products, contact source, lead.

Loaded from config/call_profile.json (or CALL_PROFILE_PATH), with Settings /
per-call overrides. Facts are injected into the LLM; spoken wording is never
hard-coded one-liners.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.config import PROJECT_ROOT, Settings

DEFAULT_PROFILE_PATH = PROJECT_ROOT / "config" / "call_profile.json"


@dataclass
class ProductOffering:
    name: str
    summary: str = ""


@dataclass
class ContactSource:
    channel: str = ""
    detail: str = ""


@dataclass
class LeadProfile:
    full_name: str = ""
    company_name: str = ""
    job_title: str = ""
    city: str = ""
    business_type: str = ""
    field_note: str = ""
    additional_notes: str = ""
    phone_number: str = ""

    def as_lead_fields(self) -> dict[str, str]:
        """Map known lead facts onto Lead columns (skip empties)."""
        mapping = {
            "full_name": self.full_name,
            "company_name": self.company_name,
            "job_title": self.job_title,
            "city": self.city,
            "business_type": self.business_type,
            "phone_number": self.phone_number,
            "additional_notes": self.additional_notes,
        }
        return {k: v.strip() for k, v in mapping.items() if (v or "").strip()}


@dataclass
class CallProfile:
    agent_name: str = "Shivangi"
    business_name: str = "Vrattiks"
    # Exact Indic spellings for TTS (Gujarati: Shivangi / Vrattiks).
    agent_name_gu: str = "શિવાંગી"
    business_name_gu: str = "વ્રત્તિક્સ"
    agent_name_hi: str = "शिवांगी"
    business_name_hi: str = "व्रत्तिक्स"
    business_description: str = (
        "a technology and software company focused on building AI-powered "
        "solutions for businesses and individuals."
    )
    products_and_services: list[ProductOffering] = field(default_factory=list)
    contact_source: ContactSource = field(default_factory=ContactSource)
    lead: LeadProfile = field(default_factory=LeadProfile)

    def spoken_agent_name(self, language: str | None = None) -> str:
        lang = (language or "en").strip().lower()
        if lang.startswith("gu") or lang == "gujlish":
            return (self.agent_name_gu or self.agent_name).strip()
        if lang.startswith("hi") or lang == "hinglish":
            return (self.agent_name_hi or self.agent_name).strip()
        return (self.agent_name or "Shivangi").strip()

    def spoken_business_name(self, language: str | None = None) -> str:
        lang = (language or "en").strip().lower()
        if lang.startswith("gu") or lang == "gujlish":
            return (self.business_name_gu or self.business_name).strip()
        if lang.startswith("hi") or lang == "hinglish":
            return (self.business_name_hi or self.business_name).strip()
        return (self.business_name or "Vrattiks").strip()

    def to_prompt_context(self, *, compact: bool = False) -> str:
        """Human-readable facts block for the system/user prompt."""
        products = self.products_and_services or []
        if products:
            if compact:
                product_lines = "; ".join(
                    f"{p.name}" + (f" ({p.summary})" if p.summary else "")
                    for p in products[:4]
                )
            else:
                product_lines = "\n".join(
                    f"  - {p.name}: {p.summary}".rstrip(": ") for p in products
                )
        else:
            product_lines = (
                "(none configured)"
                if compact
                else "  - (not configured; speak only at a high level about the company)"
            )

        source_channel = (self.contact_source.channel or "").strip()
        source_detail = (self.contact_source.detail or "").strip()
        if source_channel or source_detail:
            source_block = (
                f"{source_channel or 'unspecified'}: {source_detail or 'n/a'}"
                if compact
                else (
                    f"  channel: {source_channel or '(unspecified)'}\n"
                    f"  detail: {source_detail or '(unspecified)'}"
                )
            )
        else:
            source_block = (
                "(unset — answer generally if asked, do not invent)"
                if compact
                else (
                    "  (not configured — if asked how you got the number, answer "
                    "gracefully like a real BDE without inventing a fake story)"
                )
            )

        lead = self.lead
        lead_bits = []
        for label, value in (
            ("full_name", lead.full_name),
            ("company_name", lead.company_name),
            ("job_title", lead.job_title),
            ("city", lead.city),
            ("business_type", lead.business_type),
            ("field_note", lead.field_note),
            ("additional_notes", lead.additional_notes),
            ("phone_number", lead.phone_number),
        ):
            if (value or "").strip():
                lead_bits.append(f"{label}={value.strip()}")
        spelling = (
            f"Gujarati spellings: agent={self.agent_name_gu or self.agent_name}, "
            f"company={self.business_name_gu or self.business_name}; "
            f"Hindi spellings: agent={self.agent_name_hi or self.agent_name}, "
            f"company={self.business_name_hi or self.business_name}"
        )
        if compact:
            lead_block = ", ".join(lead_bits) if lead_bits else "(none)"
            desc = (self.business_description or "").strip()
            company_bit = (
                f"{self.business_name} ({desc})" if desc else self.business_name
            )
            return (
                "CALL PROFILE facts (paraphrase; not a script): "
                f"agent={self.agent_name}; company={company_bit}; "
                f"products={product_lines}; source={source_block}; "
                f"contact={lead_block}. "
                f"When speaking Gujarati/Hindi use exact spellings — {spelling}."
            )

        lead_block = (
            "\n".join(f"  {bit.replace('=', ': ', 1)}" for bit in lead_bits)
            if lead_bits
            else "  (no prefilled lead details — ask for name and handle missing info gracefully)"
        )
        return (
            "CALL PROFILE (internal facts — paraphrase naturally; do not read as a script):\n"
            f"- Agent name: {self.agent_name}\n"
            f"- Company: {self.business_name} — {self.business_description}\n"
            f"- Spoken name spellings (use EXACTLY in that language for TTS):\n"
            f"  - Gujarati: {self.agent_name_gu or self.agent_name} / "
            f"{self.business_name_gu or self.business_name}\n"
            f"  - Hindi: {self.agent_name_hi or self.agent_name} / "
            f"{self.business_name_hi or self.business_name}\n"
            f"- Products and services:\n{product_lines}\n"
            f"- Contact source facts (ONLY use if the caller asks how you got their number):\n"
            f"{source_block}\n"
            f"- Known about this contact:\n{lead_block}\n"
        )

    def opening_greeting(self, language: str | None = None) -> str:
        """LAST-RESORT emergency opening only when the LLM greeting fails.

        Normal calls must use ConversationEngine.generate_greeting so wording
        varies each time. This only packs required facts into a randomly
        chosen shell — never the primary spoken path.
        """
        import random

        lang = (language or "en").strip().lower()
        name = self.spoken_agent_name(lang)
        company = self.spoken_business_name(lang)
        contact = (self.lead.full_name or "").strip()

        # Keep emergency shells short and fact-based. Wording is randomly
        # chosen so even the fallback is not one frozen sentence forever.
        if lang.startswith("hi") or lang == "hinglish":
            if contact:
                variants = [
                    f"Namaste, kya {contact}? Main {name}, {company} se AI assistant. Do minute milenge?",
                    f"Hello {contact} ji, {name} bol rahi hoon {company} se AI assistant. Abhi time hai?",
                ]
            else:
                variants = [
                    f"Namaste, main {name} bol rahi hoon, {company} se AI assistant. Abhi do minute hain?",
                    f"Hello, {name} yahan se, {company} ki AI assistant. Thodi baat ho sakti hai?",
                    f"Namaskar, mera naam {name} hai, {company} se AI assistant. Time hai kya abhi?",
                ]
            return random.choice(variants)

        if lang.startswith("gu") or lang == "gujlish":
            if contact:
                variants = [
                    f"Hello, {contact}? Hu {name}, {company} ni AI assistant. Be minute male?",
                    f"Namaste {contact}, {name} bolu chu {company} thi AI assistant. Hamanah time che?",
                ]
            else:
                variants = [
                    f"Hello, hu {name} bolu chu, {company} ni AI assistant. Hamanah be minute che?",
                    f"Namaste, {name} ahinthi, {company} ni AI assistant. Hamanah vaat thai shake?",
                    f"Hi, maru naam {name} che, {company} taraphthi AI assistant. Time che hamanah?",
                ]
            return random.choice(variants)

        if contact:
            variants = [
                f"Hi, am I speaking with {contact}? This is {name}, an AI assistant from {company}. Do you have a minute?",
                f"Hello {contact}, {name} here — AI assistant from {company}. Is now a good time?",
            ]
        else:
            variants = [
                f"Hi, this is {name}, an AI assistant from {company}. Do you have a couple of minutes right now?",
                f"Hello, {name} here from {company} — I'm an AI assistant. Got two minutes to talk?",
                f"Hi there, I'm {name}, an AI assistant calling from {company}. Is now a good time for a quick chat?",
            ]
        return random.choice(variants)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _parse_products(raw: Any) -> list[ProductOffering]:
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [ProductOffering(name="Offerings", summary=raw.strip())]
    if not isinstance(raw, list):
        return []
    products: list[ProductOffering] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            products.append(ProductOffering(name=item.strip()))
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            summary = str(item.get("summary") or item.get("pitch") or "").strip()
            if name:
                products.append(ProductOffering(name=name, summary=summary))
    return products


def load_call_profile_file(path: Path | None = None) -> dict[str, Any]:
    profile_path = path or DEFAULT_PROFILE_PATH
    if not profile_path.is_file():
        return {}
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_call_profile(
    settings: Settings,
    *,
    lead_overrides: dict[str, Any] | None = None,
    phone_number: str | None = None,
) -> CallProfile:
    """Merge JSON profile + Settings + optional per-call lead overrides."""
    path = Path(settings.call_profile_path) if settings.call_profile_path else DEFAULT_PROFILE_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    file_data = load_call_profile_file(path)

    agent_name = (
        (settings.agent_name or "").strip()
        or str(file_data.get("agent_name") or "").strip()
        or "Shivangi"
    )
    business_name = (
        (settings.business_name or "").strip()
        or str(file_data.get("business_name") or "").strip()
        or "Vrattiks"
    )
    business_description = (
        (settings.business_description or "").strip()
        or str(file_data.get("business_description") or "").strip()
    )

    products = _parse_products(file_data.get("products_and_services"))
    if settings.business_products.strip():
        products = _parse_products(settings.business_products) or products

    src_file = _as_dict(file_data.get("contact_source"))
    contact_source = ContactSource(
        channel=(
            (settings.contact_source_channel or "").strip()
            or str(src_file.get("channel") or "").strip()
        ),
        detail=(
            (settings.contact_source_detail or "").strip()
            or str(src_file.get("detail") or "").strip()
        ),
    )

    lead_file = _as_dict(file_data.get("lead"))
    overrides = lead_overrides or {}
    lead = LeadProfile(
        full_name=str(
            overrides.get("full_name")
            or settings.default_lead_full_name
            or lead_file.get("full_name")
            or ""
        ).strip(),
        company_name=str(
            overrides.get("company_name")
            or settings.default_lead_company_name
            or lead_file.get("company_name")
            or ""
        ).strip(),
        job_title=str(
            overrides.get("job_title")
            or settings.default_lead_job_title
            or lead_file.get("job_title")
            or ""
        ).strip(),
        city=str(
            overrides.get("city")
            or settings.default_lead_city
            or lead_file.get("city")
            or ""
        ).strip(),
        business_type=str(
            overrides.get("business_type")
            or settings.default_lead_business_type
            or lead_file.get("business_type")
            or ""
        ).strip(),
        field_note=str(
            overrides.get("field_note")
            or settings.default_lead_field_note
            or lead_file.get("field_note")
            or ""
        ).strip(),
        additional_notes=str(
            overrides.get("additional_notes")
            or settings.default_lead_additional_notes
            or lead_file.get("additional_notes")
            or ""
        ).strip(),
        phone_number=str(
            phone_number
            or overrides.get("phone_number")
            or lead_file.get("phone_number")
            or ""
        ).strip(),
    )

    return CallProfile(
        agent_name=agent_name,
        business_name=business_name,
        agent_name_gu=str(
            file_data.get("agent_name_gu") or "શિવાંગી"
        ).strip(),
        business_name_gu=str(
            file_data.get("business_name_gu") or "વ્રત્તિક્સ"
        ).strip(),
        agent_name_hi=str(
            file_data.get("agent_name_hi") or "शिवांगी"
        ).strip(),
        business_name_hi=str(
            file_data.get("business_name_hi") or "व्रत्तिक्स"
        ).strip(),
        business_description=business_description
        or (
            "a technology and software company focused on building AI-powered "
            "solutions for businesses and individuals."
        ),
        products_and_services=products,
        contact_source=contact_source,
        lead=lead,
    )


def profile_as_dict(profile: CallProfile) -> dict[str, Any]:
    return asdict(profile)
