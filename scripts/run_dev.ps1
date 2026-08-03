# Sarvam Cloud Lead Agent - development launcher (Windows PowerShell)
# Usage:  .\scripts\run_dev.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# 1. Virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    if (-not $?) { throw "Failed to create virtual environment. Is Python 3.11+ installed?" }
}
& ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if (-not $?) { throw "Failed to install requirements." }

# 2. Configuration
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "[2/5] Created .env from .env.example - please add your SARVAM_API_KEY." -ForegroundColor Yellow
    } else {
        throw "Missing .env.example"
    }
} else {
    Write-Host "[2/5] Using existing .env"
}

# 3. Health checks (informational; failures are warnings, not blockers)
Write-Host "[3/5] Checking dependencies..." -ForegroundColor Cyan
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "  [WARN] ffmpeg not found on PATH. Install FFmpeg first (see SETUP.md)." -ForegroundColor Yellow
} else {
    Write-Host "  [OK] ffmpeg found"
}
& ".venv\Scripts\python.exe" scripts\check_sarvam.py
if (-not $?) { Write-Host "  [WARN] Sarvam API not reachable yet - add your key and check connectivity." -ForegroundColor Yellow }
& ".venv\Scripts\python.exe" scripts\check_llm.py
if (-not $?) { Write-Host "  [WARN] LLM endpoint not reachable yet - check LLM_PROVIDER/keys in .env." -ForegroundColor Yellow }

# 4. Database init (creates data/sarvam_leads.db via SQLAlchemy metadata)
Write-Host "[4/5] Initializing database..." -ForegroundColor Cyan
& ".venv\Scripts\python.exe" -c "from backend.database import create_engine_and_session, make_database_url; from backend.config import get_settings; create_engine_and_session(make_database_url(get_settings())); print('  [OK] Database ready')"

# 5. Run server
Write-Host "[5/5] Starting app at http://localhost:8021  (Ctrl+C to stop)" -ForegroundColor Green
& ".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8021
