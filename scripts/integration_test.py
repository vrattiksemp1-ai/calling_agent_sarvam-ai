#!/usr/bin/env python3
"""Optional manual integration test for the Sarvam Cloud Lead Agent.

Starts the FastAPI app with fully mocked providers (no real Sarvam API key or
internet needed) and exercises the full text-mode conversation flow plus
JSON/CSV export.

Usage:
    python scripts/integration_test.py
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.config import Settings  # noqa: E402
from backend.main import build_app  # noqa: E402


def _user_text(messages) -> str:
    """Return the real last user utterance, skipping the injected state block."""
    for m in reversed(messages):
        if m["role"] != "user":
            continue
        content = m.get("content", "") or ""
        if content.startswith("\n\nCurrent state:") or content.startswith("\n\nYour previous reply"):
            continue
        return content
    return ""


def _mock_llm_handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content or b"{}")
    messages = payload.get("messages", [])
    content = _user_text(messages)

    extracted = {}
    next_state = "collecting_identity"
    if "rahul" in content.lower() and ("call me" in content.lower() or "name is" in content.lower() or "i am" in content.lower()):
        extracted = {"full_name": "Rahul Sharma"}
        next_state = "collecting_contact"
    elif "98" in content or "email" in content.lower():
        extracted = {"phone_number": "+919876543210"}
        next_state = "collecting_requirement"
    elif "crm" in content.lower():
        extracted = {
            "business_requirement": "I need a CRM with voice automation",
            "company_name": "Acme Retail",
            "product_or_service_interest": "CRM software",
            "estimated_budget": "Rs 50000 per year",
            "purchase_timeline": "within 3 months",
            "decision_maker_status": "yes",
        }
        next_state = "requesting_consent"
    elif "contact" in content.lower() and "yes" in content.lower():
        extracted = {"consent_to_contact": "yes"}
        next_state = "reviewing_summary"
    elif "confirm" in content.lower() and "yes" in content.lower():
        next_state = "completed"

    reply = {
        "assistant_message": "Thanks! What is your phone number or email?",
        "detected_language": "en",
        "extracted_fields": extracted,
        "fields_to_clear": [],
        "next_state": next_state,
        "conversation_complete": next_state == "completed",
        "needs_confirmation": next_state == "reviewing_summary",
    }
    if next_state == "collecting_contact":
        reply["assistant_message"] = "Great, thanks Rahul! What is the best phone number or email to reach you?"
    elif next_state == "collecting_requirement":
        reply["assistant_message"] = "Got it. What is your main requirement or what are you looking for?"
    elif next_state == "requesting_consent":
        reply["assistant_message"] = "May I contact you later about this? (yes/no)"
    elif next_state == "reviewing_summary":
        reply["assistant_message"] = "Here is the summary... Can you confirm this is correct? (yes/no)"
    elif next_state == "completed":
        reply["assistant_message"] = "Confirmed! Your lead is saved. Thank you!"

    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-mock",
            "choices": [{"message": {"content": json.dumps(reply), "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        },
    )


def _mock_sarvam_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/speech-to-text"):
        return httpx.Response(200, json={"transcript": "my name is rahul and I need a CRM system"})
    if request.url.path.endswith("/text-to-speech"):
        # Tiny valid WAV header so playback tests do not crash.
        wav = bytes.fromhex("524946460400000057415645666d74201000000001000100401f0000803e0000020010006461746100000000")
        return httpx.Response(200, json={"audios": [base64.b64encode(wav).decode("ascii")]})
    return httpx.Response(404, json={"error": "not found"})


async def main() -> int:
    settings = Settings(
        database_url="sqlite:///./storage/tmp/integration_test.db",
        rate_limit_enabled=False,
        debug=False,
    )
    transport = httpx.MockTransport(_mock_llm_handler)
    llm = __import__("backend.providers.llm_client", fromlist=["LlmClient"]).LlmClient(
        settings, http_client=httpx.AsyncClient(transport=transport)
    )
    transport2 = httpx.MockTransport(_mock_sarvam_handler)
    sarvam = __import__(
        "backend.providers.sarvam_client", fromlist=["SarvamClient"]
    ).SarvamClient(settings, http_client=httpx.AsyncClient(transport=transport2))

    app = build_app(settings, sarvam_client=sarvam, llm_client=llm)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        print("[OK] /health:", health.json())

        sess = client.post("/api/sessions").json()
        sid = sess["session_id"]
        print("[OK] session created:", sid)

        turns = [
            "my name is rahul",
            "my number is 9876543210",
            "I need a CRM with voice automation",
            "yes you can contact me",
            "yes I confirm",
        ]
        for turn in turns:
            r = client.post(f"/api/sessions/{sid}/message", json={"text": turn})
            assert r.status_code == 200, r.text
            body = r.json()
            print(f"  user> {turn}")
            print(f"  agent> {body['assistant_message']}")
            metrics = body.get("metrics") or {}
            if metrics.get("estimated_provider_cost"):
                print(f"  cost> Rs {metrics['estimated_provider_cost']:.4f} this turn")

        summary = client.get(f"/api/sessions/{sid}/summary").json()
        assert summary["completion_status"] == "completed", summary
        print("[OK] summary:", summary)

        lead = client.get(f"/api/sessions/{sid}/lead").json()
        assert lead["fields"]["full_name"] == "Rahul Sharma", lead
        assert lead["qualification_score"] >= 70, lead
        print("[OK] lead:", lead["qualification_score"], lead["qualification_level"])

        lid = lead["id"]
        j = client.get(f"/api/leads/{lid}/export.json")
        assert j.status_code == 200 and "full_name" in j.text
        c = client.get(f"/api/leads/{lid}/export.csv")
        assert c.status_code == 200 and "full_name" in c.text
        print("[OK] JSON + CSV export")

        dl = client.delete(f"/api/sessions/{sid}")
        assert dl.status_code == 200
        print("[OK] session deletion")

    print("\nAll integration checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
