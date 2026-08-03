# TROUBLESHOOTING — Sarvam Cloud Lead Agent

Common problems and their fixes. Work through the section that matches your
symptom.

---

## Installation / PowerShell issues

### "pip install" says "not recognized" or "command not found"

Your environment is not active or Python is not on PATH.

- Activate the venv first: `.\.venv\Scripts\Activate.ps1`
- If `python` itself is not found, reinstall Python and tick
  **"Add python.exe to PATH"**, then reopen the terminal.
- Check you are in the project folder: `cd D:\MVP_calling_agent\sarvam-api-mvp`

### "Activate.ps1 cannot be loaded because running scripts is disabled"

Windows blocks PowerShell scripts by default. Run this once, then retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

(`-Scope Process` only affects the current terminal, which is safer than
changing the system policy.)

Alternative: use Command Prompt and run `.venv\Scripts\activate.bat`.

### ".env: The term '.env' is not recognized"

PowerShell syntax, not a file problem. To copy the example file use:

```powershell
Copy-Item .env.example .env
```

### "python -m pytest" starts downloading / can't find pytest

Install the requirements first, and stay inside the activated venv:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

---

## The app won't start

### "ModuleNotFoundError: No module named 'backend'"

Run the server from the project root (`D:\MVP_calling_agent\sarvam-api-mvp`),
not from a subfolder. Or use the run script which sets the working directory.

### "address already in use" / port 8021 is taken

Something else is using port 8021.

- Change the port in `.env` (`APP_PORT=8022`) and restart.
- Or find and stop the other process:
  `netstat -ano | findstr :8021`, then `taskkill /PID <pid> /F`.

### "Uvicorn running on http://0.0.0.0:8021" but browser can't connect

- Try `http://127.0.0.1:8021` instead of `localhost`.
- Check Windows Firewall is not blocking Python (only relevant if you connect
  from another device).

---

## Voice mode doesn't work

### "Sarvam is not reachable" / status shows degraded or error

The backend cannot talk to the Sarvam cloud API.

1. Confirm `SARVAM_API_KEY` is set in `.env` (the provider status shows
   "degraded" when the key is missing).
2. Verify the base URL in `.env`: `SARVAM_BASE_URL=https://api.sarvam.ai`.
3. Run `scripts\check_sarvam.py` and read its message - it distinguishes
   "not reachable" (network/proxy) from "key rejected" (401/403) from
   "endpoint differs for your plan" (harmless).

### "ffmpeg not found" / FfmpegMissingError

The uploaded audio cannot be converted without FFmpeg.

- Install FFmpeg and add it to PATH (on Windows the simplest way is
  `winget install ffmpeg` or https://ffmpeg.org/download.html).
- Reopen the terminal after installing so PATH updates.
- Verify: `ffmpeg -version`
- If ffmpeg lives somewhere unusual, set its path in `.env`:
  `FFMPEG_PATH=C:\tools\ffmpeg\bin\ffmpeg.exe`

### "Speech synthesis failed; the reply is shown as text"

Sarvam TTS answered with an error. Common causes:

- The selected `SARVAM_TTS_SPEAKER` is not available for `bulbul:v3` - change
  it to a documented speaker (e.g. `shubh`, `meera`, `pavithra`).
- The fallback `SARVAM_TTS_LANGUAGE_CODE` is not in the closed set
  (`bn-IN, en-IN, gu-IN, hi-IN, kn-IN, ml-IN, mr-IN, od-IN, pa-IN, ta-IN,
  te-IN`). The detected call language is mapped automatically; if detection
  returns something unusual, the fallback is used.
- The conversation continues in text mode, so the agent still works.

### "Speech-to-text failed"

Sarvam STT rejected the audio. Common causes:

- Quota exhausted or key out of credits - check the Sarvam console.
- The audio was unintelligible or empty.
- `SARVAM_STT_MODEL` is not `saaras:v3` (default).
- Text mode still works if STT is down.

### The agent replies in the wrong voice

Set a different `SARVAM_TTS_SPEAKER` in `.env` and restart.

### Microphone button does nothing

- Grant the browser permission for the microphone (lock icon in the address
  bar).
- Use Chrome or Edge; the MediaRecorder API is used for recording.
- Check the debug panel for the exact error.

---

## The LLM isn't replying

### "LLM not reachable" / timeout in the debug panel

The backend cannot reach your LLM endpoint.

- Sarvam chat API: set `LLM_PROVIDER=sarvam`, `LLM_MODEL=sarvam-105b`, and
  `LLM_API_KEY` (or reuse `SARVAM_API_KEY` when `LLM_API_KEY` is empty).
- OpenAI-compatible server: set `LLM_PROVIDER=openai-compatible`,
  `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
- Run `scripts\check_llm.py`.

### The agent answers but repeats the same question

- The model may be weak. Try a stronger model or set `LLM_MODEL` accordingly.
- Check `LLM_USE_JSON_MODE`. If the model mangles JSON, the backend repairs it
  once and otherwise uses a safe fallback.
- `scripts\check_llm.py` shows whether the model can produce structured output.

### Every reply says "I still need your name"

The LLM is not returning the structured JSON, so nothing gets stored. See
above; switching to a stronger model or raising `LLM_TIMEOUT` usually fixes it.

---

## Cost questions

### Costs look higher/lower than my Sarvam invoice

The displayed costs are **estimates** computed from the rates at the bottom of
`.env`. Update them to match your actual plan; the Sarvam invoice is
authoritative. Text-only turns do not charge TTS (only the LLM).

### Can I see the total cost of a conversation?

Yes - the session summary endpoint (`GET /api/sessions/{id}/summary`) returns
`estimated_provider_cost`, and the UI shows a running "Estimated cost" total.

---

## Data / export issues

### "lead not found" when exporting

Exports live under `/api/leads/{id}/export.json` and `/api/leads/{id}/export.csv`.
Use the lead **id** (from `/api/leads`), not the session id.

### CSV opens with wrong characters in Excel

The CSV is UTF-8 with BOM. Import it in Excel via **Data > From Text/CSV**
rather than double-clicking.

### I want to keep recordings for review

Set `RETAIN_AUDIO=true` in `.env`. Normalized WAVs are then kept under
`storage/tmp/retained/`.

---

## Tests

### Tests fail or hang

- Make sure no `.venv` from a *different* project is active and that you run
  `pytest` from the project root.
- Tests are fully mocked and need no keys, internet, or services. If a test
  touches the network, something is wrong with the environment (proxy settings).

### "ModuleNotFoundError" inside tests

Run `python -m pytest` (the `-m` makes Python add the current folder to the
path), or ensure you installed requirements into THIS project's `.venv`.

---

## Still stuck?

- Re-read the relevant part of [SETUP.md](SETUP.md).
- Enable more logging by setting `DEBUG=true` in `.env` and restarting; the
  debug panel in the UI shows provider health and last errors.
- Confirm your `SARVAM_API_KEY`, `SARVAM_BASE_URL`, `LLM_BASE_URL` and ports by
  running the two check scripts; they print human-readable results.
