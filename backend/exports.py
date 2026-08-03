"""Lead export helpers (JSON and CSV)."""

import csv
import io
from typing import Any

from backend.models import LEAD_FIELDS, Lead


def lead_to_dict(lead: Lead) -> dict[str, Any]:
    data: dict[str, Any] = {
        "lead_id": lead.id,
        "session_id": lead.session_id,
        "qualification_score": lead.qualification_score,
        "qualification_level": lead.qualification_level,
        "missing_important_fields": list(lead.missing_important_fields or []),
        "recommended_next_action": lead.recommended_next_action,
        "conversation_status": lead.conversation_status,
        "consent_confirmed": lead.consent_confirmed,
        "summary_confirmed": lead.summary_confirmed,
        "created_at": lead.created_at.isoformat() if lead.created_at else None,
        "updated_at": lead.updated_at.isoformat() if lead.updated_at else None,
    }
    for name in LEAD_FIELDS:
        data[name] = getattr(lead, name)
    return data


def lead_to_json_bytes(lead: Lead) -> bytes:
    import json

    return json.dumps(lead_to_dict(lead), indent=2, ensure_ascii=False).encode("utf-8")


def leads_to_json_bytes(leads: list[Lead]) -> bytes:
    import json

    return json.dumps(
        [lead_to_dict(lead) for lead in leads], indent=2, ensure_ascii=False
    ).encode("utf-8")


def lead_to_csv_bytes(lead: Lead) -> bytes:
    data = lead_to_dict(lead)
    return _rows_to_csv([data])


def leads_to_csv_bytes(leads: list[Lead]) -> bytes:
    return _rows_to_csv([lead_to_dict(lead) for lead in leads])


def _rows_to_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    columns = list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8-sig")
