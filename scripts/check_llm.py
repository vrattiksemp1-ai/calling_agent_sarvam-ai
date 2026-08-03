#!/usr/bin/env python3
"""Check the configured LLM endpoint (Sarvam chat API or any OpenAI-compatible server).

Usage:
    python scripts/check_llm.py            # reads .env
    python scripts/check_llm.py --base-url https://api.sarvam.ai --model sarvam-105b
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from backend.config import get_settings  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="Check the configured LLM endpoint.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    settings = get_settings()
    base_url = (args.base_url or settings.llm_base_url).rstrip("/")
    model = args.model or settings.llm_model
    key = settings.llm_api_key

    print(f"LLM base URL : {base_url}")
    print(f"LLM model    : {model}")
    print(f"LLM provider : {settings.llm_provider}")
    print()

    headers = {
        "Authorization": f"Bearer {key}",
        "api-subscription-key": key,
    }
    try:
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            resp = await client.get(f"{base_url}/v1/models", headers=headers)
    except httpx.HTTPError as exc:
        print(f"[FAIL] LLM endpoint is not reachable: {exc}")
        print("  Check LLM_BASE_URL / LLM_API_KEY in .env.")
        return 1

    if resp.status_code >= 500:
        print(f"[FAIL] LLM returned HTTP {resp.status_code}: {resp.text[:300]}")
        return 1
    if resp.status_code == 401 or resp.status_code == 403:
        print(f"[FAIL] LLM endpoint rejected the key (HTTP {resp.status_code}).")
        return 1
    if resp.status_code >= 400:
        print(f"[WARN] LLM /v1/models returned HTTP {resp.status_code}; the chat call may still work.")
        print("       Continuing with the chat probe...")
    else:
        print("[OK] LLM endpoint is reachable.")

    try:
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            body = await client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                    "max_tokens": 8,
                },
            )
        if body.status_code < 400:
            print(f"[OK] Model '{model}' generated a reply.")
        else:
            print(f"[WARN] Model '{model}' responded with HTTP {body.status_code}: {body.text[:300]}")
            print("       Check the model name in .env.")
            return 1
    except httpx.HTTPError as exc:
        print(f"[WARN] Could not verify model '{model}': {exc}")
        print("       Check the model name in .env.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
