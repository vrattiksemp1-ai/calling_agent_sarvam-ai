# SETUP — Sarvam Cloud Lead Agent

A beginner-friendly, step-by-step guide to run this project on **Windows
(PowerShell)**. If you are on Linux or macOS, the equivalent commands are noted
at the end. Take your time; every step is verifiable.

---

## What you need before starting

1. **Python 3.11 or newer** installed and on your PATH.
2. **FFmpeg** (for converting microphone recordings to WAV).
3. **A Sarvam API subscription key** from https://www.sarvam.ai (used for both
   STT and TTS; voice mode will not work without it).
4. **An LLM**: either a Sarvam chat API key (same console) or any
   OpenAI-compatible endpoint. Defaults assume the Sarvam chat API
   (`https://api.sarvam.ai/v1/chat/completions`, model `sarvam-105b`).

If you only want to test the web UI in **text mode**, you still need an LLM,
but not FFmpeg or the Sarvam speech key.

---

## Step-by-step (Windows PowerShell)

### 1. Open a terminal in the project folder

```powershell
cd D:\MVP_calling_agent\sarvam-api-mvp
```

Verify Python is available:

```powershell
python --version
```

You should see something like `Python 3.12.x`. If you get "not recognized",
install Python from https://www.python.org/downloads/ and tick **"Add python.exe
to PATH"** during installation, then reopen the terminal.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

This creates a `.venv` folder. It keeps this project's dependencies separate
from the rest of your computer.

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the start of your prompt.

> **Trouble activating?** If you get an "execution policy" error, see
> [TROUBLESHOOTING.md](TROUBLESHOOTING.md). If PowerShell is not your thing,
> you can also use Command Prompt and run `.venv\Scripts\activate.bat`.

### 4. Install the dependencies

```powershell
pip install -r requirements.txt
```

Wait for it to finish. This installs FastAPI, SQLAlchemy, httpx, pydantic, and
the test tooling.

### 5. Configure the application

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` in any text editor and add your keys:

- `SARVAM_API_KEY=your-key-here` - required for voice (STT + TTS).
- `LLM_API_KEY=your-key-here` - required for the LLM when
  `LLM_PROVIDER=sarvam` (uses the same Sarvam console key). If you use a
  different OpenAI-compatible LLM, set `LLM_PROVIDER=openai-compatible`,
  `LLM_BASE_URL`, and `LLM_MODEL` instead.

Other values you may want to change:

- `SARVAM_TTS_SPEAKER` - which Sarvam voice to use (default `shubh`).
- `SARVAM_TTS_LANGUAGE_CODE` - fallback TTS language (default `hi-IN`).
- `APP_PORT` - the port the web app uses (default `8021`).
- The cost rates at the bottom (display estimates only).

Save the file.

### 6. Verify your environment (recommended)

Run the mock test suite (no keys or internet needed):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

You should see a green `95 passed` (or a similar count). If tests fail, see
TROUBLESHOOTING.md.

### 7. Start the app

```powershell
.\scripts\run_dev.ps1
```

Or, if you prefer the raw command:

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8021
```

You should see logs like:

```
Uvicorn running on http://0.0.0.0:8021
```

### 8. Open the web UI

In a browser go to:

```
http://localhost:8021
```

You should see the lead-agent chat interface with a microphone button.

### 9. Verify your cloud connectivity

Run the check scripts (these DO call the real Sarvam API):

```powershell
.\.venv\Scripts\python.exe scripts\check_sarvam.py
.\.venv\Scripts\python.exe scripts\check_llm.py
```

Or just look at the provider status in the app's debug panel. Green/ok means
your key works and the endpoints are reachable.

### 10. Have a conversation

- **Text mode**: type "my name is rahul", then a phone number, then your
  requirement. The agent will ask for consent, summarise, and save.
- **Voice mode**: press the mic button, talk, release. The agent replies with
  its voice. Watch the **Estimated cost** line to see per-turn cloud spend.

---

## Optional: real phone calls (Twilio trial + ngrok)

The "Call my phone" button places an outbound call through Twilio, streams the
call audio to this app, and the agent talks to the caller live. This works on a
**free Twilio trial account** with a few restrictions:

- You can only call **numbers you have verified** in the Twilio console.
- Outgoing call minutes are capped by your trial credit.

### 11. Create the Twilio trial account and number

1. Sign up at https://www.twilio.com/try-twilio (a valid phone number and email
   are required for verification).
2. From the Twilio console **Home** page copy your **Account SID** and
   **Auth Token**.
3. Under **Phone Numbers > Manage > Buy a Number** get a number. Trial accounts
   can pick a number with a monthly credit; no payment card is needed.
4. Under **Phone Numbers > Verified Caller IDs** add the phone number you will
   call for testing (the agent must be able to dial it).

### 12. Expose this app to the internet with ngrok

Twilio needs a public URL to reach the audio WebSocket. If you do not have
ngrok, install it from https://ngrok.com/download (the free plan is fine):

```powershell
ngrok http 8021
```

Copy the `Forwarding` HTTPS address, e.g. `https://abc123.ngrok-free.app`.
Leave ngrok running in that terminal.

### 13. Configure the call settings in .env

Open `.env` and fill in:

```ini
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+15005550006   # your Twilio number, E.164 format
TWILIO_CALL_PUBLIC_BASE_URL=https://abc123.ngrok-free.app
```

Restart the app after editing `.env`. The app turns the base URL into
`wss://abc123.ngrok-free.app/api/calls/stream` for Twilio automatically.

### 14. Test a call

1. Open `http://localhost:8021`.
2. Enter a **verified** phone number (E.164 format, e.g. `+919876543210`).
3. Click **Call my phone** and answer when it rings.

The agent greets you, asks for your name, qualifies the lead, and hangs up
automatically when the conversation completes. Track the call from the status
text under the call bar; the resulting lead appears in the lead panel as usual.

> **Trial gotchas**: if the number is not verified, Twilio fails with a
> 40002/21211-style error and no call is placed. If the call connects but you
> hear silence, check that ngrok is still running and that
> `TWILIO_CALL_PUBLIC_BASE_URL` matches the forwarding address exactly.

---

## Where your data lives

| Thing | Location |
| --- | --- |
| Leads database | `data/sarvam_leads.db` |
| Temporary audio | `storage/tmp/` (auto-cleaned) |
| Retained audio (only if `RETAIN_AUDIO=true`) | `storage/tmp/retained/` |
| Logs (console) | Your terminal |

---

## Linux / macOS equivalents

```bash
# from the project folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env and add your keys
./scripts/run_dev.sh        # or: python -m uvicorn backend.main:app --port 8021
```

Everything else (browser at `http://localhost:8021`, tests) is identical.

---

## Next steps

- Read [README.md](README.md) for architecture, API reference, scoring model,
  cost estimation, and sample conversations.
- If something is not working, check [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
- To run in Docker instead: `docker compose up --build`.
