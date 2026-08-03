#!/usr/bin/env python3
"""Check the Sarvam cloud speech API (STT + TTS) and your subscription key.

Usage:
    python scripts/check_sarvam.py            # reads .env
    python scripts/check_sarvam.py --base-url https://api.sarvam.ai --model saaras:v3
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from backend.config import get_settings  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="Check Sarvam speech API connectivity.")
    parser.add_argument("--base-url", default=None, help="e.g. https://api.sarvam.ai")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    settings = get_settings()
    base_url = (args.base_url or settings.sarvam_base_url).rstrip("/")

    print(f"Sarvam base URL : {base_url}")
    print(f"STT model       : {settings.sarvam_stt_model}")
    print(f"TTS model       : {settings.sarvam_tts_model} / speaker {settings.sarvam_tts_speaker}")
    print(f"API key set     : {'yes' if settings.sarvam_api_key else 'no'}")
    print()

    if not settings.sarvam_api_key:
        print("[WARN] SARVAM_API_KEY is empty. Set it in .env before running the agent.")
        print("       Connectivity checks below will likely fail with 401.")

    headers = {"api-subscription-key": settings.sarvam_api_key}
    try:
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            resp = await client.get(
                f"{base_url}/text-to-speech/voices",
                headers=headers,
                params={"model": settings.sarvam_tts_model, "target_language_code": settings.sarvam_tts_language_code},
            )
    except httpx.HTTPError as exc:
        print(f"[FAIL] Sarvam API is not reachable: {exc}")
        print("  Check your network and SARVAM_BASE_URL in .env.")
        return 1

    if resp.status_code >= 500:
        print(f"[FAIL] Sarvam returned HTTP {resp.status_code}: {resp.text[:300]}")
        return 1
    if resp.status_code == 401 or resp.status_code == 403:
        print(f"[FAIL] Sarvam rejected the subscription key (HTTP {resp.status_code}).")
        return 1
    if resp.status_code >= 400:
        print(f"[WARN] Sarvam returned HTTP {resp.status_code}: {resp.text[:300]}")
        print("       The /text-to-speech/voices endpoint may differ for your plan;")
        print("       this is informational - the agent still starts.")
        return 0

    print("[OK] Sarvam speech API is reachable and the key was accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
