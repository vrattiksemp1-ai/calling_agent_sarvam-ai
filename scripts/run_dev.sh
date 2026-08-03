#!/usr/bin/env bash
# Sarvam Cloud Lead Agent - development launcher (Linux/macOS)
# Usage:  ./scripts/run_dev.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Virtual environment
if [ ! -d ".venv" ]; then
  echo "[1/5] Creating virtual environment..."
  python3 -m venv .venv
fi
".venv/bin/python" -m pip install --disable-pip-version-check -q -r requirements.txt

# 2. Configuration
if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[2/5] Created .env from .env.example - please add your SARVAM_API_KEY."
  else
    echo "ERROR: missing .env.example" >&2
    exit 1
  fi
else
  echo "[2/5] Using existing .env"
fi

# 3. Health checks (informational)
echo "[3/5] Checking dependencies..."
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "  [WARN] ffmpeg not found. Install FFmpeg first (see SETUP.md)."
else
  echo "  [OK] ffmpeg found"
fi
".venv/bin/python" scripts/check_sarvam.py || echo "  [WARN] Sarvam API not reachable yet - add your key and check connectivity."
".venv/bin/python" scripts/check_llm.py || echo "  [WARN] LLM endpoint not reachable yet - check LLM_PROVIDER/keys in .env."

# 4. Database init
echo "[4/5] Initializing database..."
".venv/bin/python" -c "from backend.database import create_engine_and_session, make_database_url; from backend.config import get_settings; create_engine_and_session(make_database_url(get_settings())); print('  [OK] Database ready')"

# 5. Run server
echo "[5/5] Starting app at http://localhost:8021  (Ctrl+C to stop)"
exec ".venv/bin/python" -m uvicorn backend.main:app --host 0.0.0.0 --port 8021
