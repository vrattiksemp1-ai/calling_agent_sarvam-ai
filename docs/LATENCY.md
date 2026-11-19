# Voice Agent Latency — What Makes Replies Slow

This document explains **why the agent takes several seconds to speak** after the caller finishes talking, based on the current `sarvam-api-mvp` implementation.

**Short answer:** On the **Twilio trial / Gather path** (what you are using today), latency is **architectural**, not a single bug. The pipeline is mostly **sequential**: user stops speaking → Twilio STT → webhook → LLM (full JSON) → TTS → Twilio fetches audio → caller hears reply. A realistic target is **~4–6 seconds** per turn, not 2–3 seconds.

---

## 1. How we measure latency

Every phone turn logs structured telemetry (`turn_telemetry`, `first_outbound_audio`).

| Metric | Meaning |
|--------|---------|
| **`caller_perceived_latency_ms`** | Time from **transcript received** (Gather) or **utterance end** (Media Streams) until **first outbound audio** reaches Twilio. This is what the caller feels. |
| **`llm_ms`** | Sarvam LLM time to return full structured JSON. |
| **`tts_ms`** / **`tts.first_audio`** | TTS time until first audio chunk is ready. |
| **`stt_ms`** | Sarvam STT time — **0 on Gather path** (Twilio does STT). |
| **`transcript_to_llm_done_ms`** | Webhook receipt → LLM finished. |
| **`llm_done_to_tts_first_ms`** | LLM finished → TTS first chunk. |
| **`tts_first_to_outbound_ms`** | TTS ready → Twilio actually plays audio (network + `<Play>` fetch). |

Example from a healthy Gather turn:

```
transcript_received → llm_completed: ~1173ms
llm_completed → tts_first_audio:     ~1178ms
tts_first_audio → first_outbound:    ~352ms
caller_perceived_latency_ms:         ~2703ms
```

Heavy product replies or long JSON can push this toward **4s+**.

---

## 2. Two call paths (only one is “fast-capable”)

| | **Gather (trial default)** | **Media Streams (paid Twilio)** |
|--|---------------------------|----------------------------------|
| **Enabled when** | `TWILIO_TRIAL_MODE=true` | Trial off + upgraded account |
| **STT** | Twilio Speech Recognition | Sarvam STT (REST or realtime WS) |
| **Audio to caller** | `<Play>` public URL | WebSocket PCM frames (20 ms chunks) |
| **Streaming overlap** | **No** LLM↔TTS overlap | **Optional** with feature flags |
| **Realistic latency** | **~4–6 s** | **~2–3 s** (with tuning + flags) |

**You are on the Gather path** unless trial mode is disabled and Twilio Media Streams is active.

Relevant code: `backend/telephony/turn_flow.py` (Gather), `backend/telephony/call_manager.py` (Media Streams).

---

## 3. End-to-end pipeline (Gather / trial)

What happens on each user turn:

```
Caller speaks
    ↓
Twilio <Gather> listens (speechTimeout, timeout window)
    ↓
Twilio STT → POST /api/calls/turn with SpeechResult
    ↓
ConversationEngine.process_turn()
    → LLM (sarvam-105b, ~4k char prompt, JSON output, max 320 tokens)
    ↓
Sarvam TTS (HTTP stream or buffered WAV)
    ↓
Audio hosted at PUBLIC_BASE_URL/api/calls/audio/{id}
    ↓
TwiML returned: <Play url> + <Gather> (listen again)
    ↓
Twilio HTTP-fetches audio from your server → plays to caller
```

Everything in the middle runs **one after another**. There is no “speak while thinking” on this path.

---

## 4. Latency contributors (ranked by impact)

### A. Twilio Gather architecture (largest fixed cost)

| Cause | Why it adds delay |
|-------|-------------------|
| **No Media Streams on trial** | Trial Twilio strips `<Stream>`; only `<Gather speech>` works. Real-time bidirectional audio is unavailable. |
| **Twilio-side STT** | Transcript arrives only **after** the user stops speaking and Twilio finishes recognition. We cannot start LLM while the user is still talking. |
| **Webhook round trip** | Each turn is an HTTP POST to your server. Network + Twilio processing adds hundreds of ms. |
| **`<Play>` via public URL** | Twilio must **fetch** TTS audio from `PUBLIC_BASE_URL` (e.g. ngrok / Render). Extra RTT after TTS is ready (`tts_first_to_outbound_ms`). |
| **Gather silence detection** | `GATHER_SPEECH_TIMEOUT=auto` and `GATHER_TIMEOUT_SECONDS=7` control how long Twilio waits after speech ends before posting. Tuning affects responsiveness vs cut-offs. |
| **15 s webhook cap** | Twilio webhooks hard-cap at ~15 s. Slow turns may **Redirect** to `/turn-result` and poll — adds perceived delay on heavy turns. |

**Cannot be fully removed without upgrading Twilio and switching to Media Streams.**

---

### B. Sequential LLM → TTS (no streaming overlap on Gather)

| Step | Typical range (from logs) | Notes |
|------|---------------------------|-------|
| LLM full completion | **~900–1400 ms** | Must finish entire JSON before TTS starts |
| TTS first audio | **~850–1200 ms** | Starts only after LLM text exists |
| Twilio Play fetch | **~300–900 ms** | Depends on host latency (Render/ngrok) |

**Total core processing often ~2–2.7 s** before Twilio even starts playing.

On Gather, `LLM_STREAMING_ENABLED=false` by default. Even when TTS uses HTTP streaming (`SARVAM_TTS_STREAMING=true`), streaming starts **after** the full assistant line is known — not while the LLM is still generating.

---

### C. Sarvam LLM (model + prompt + JSON)

| Cause | Impact |
|-------|--------|
| **Model `sarvam-105b`** | Large model; phone turns target quality over raw speed. |
| **~4k character system prompt** | Every turn sends full BDE prompt + history + state block. More input tokens → higher time-to-first-token and completion time. |
| **Structured JSON output** | Model must emit valid JSON (`assistant_message`, `extracted_fields`, `next_state`, …). Parsing/repair adds failure modes; phone uses `PHONE_LLM_MAX_TOKENS=320`. |
| **Hidden reasoning** | If `PHONE_LLM_REASONING_EFFORT` is not `none`, Sarvam may spend **several extra seconds** in thinking mode. **Keep it `none` on phone.** |
| **Repair retry** | Invalid JSON triggers one repair LLM call → doubles LLM time on bad turns. |

Mitigations already in config:

- `PHONE_LLM_REASONING_EFFORT=none`
- `PHONE_PROMPT_COMPACT=true`
- `PHONE_LLM_MAX_TOKENS=320`

---

### D. Sarvam TTS

| Mode | Behavior | Latency |
|------|----------|---------|
| **HTTP stream** (`SARVAM_TTS_STREAMING=true`) | First chunks while synthesizing; used for Gather `<Play>` | `first_audio_ms` ~850–1200 ms typical |
| **Buffered** (fallback) | Full WAV before any audio | Often **slower**; used if stream fails or first chunk timeout (2 s) exceeded |
| **WebSocket realtime** | Overlap with LLM on Media Streams only | Requires `SARVAM_REALTIME_TTS_ENABLED=true` + non-trial path |

Greeting path uses **buffered** TTS once at call start (~3–4 s in logs) — acceptable because it runs before the user speaks.

---

### E. Deployment / network

| Cause | Impact |
|-------|--------|
| **`PUBLIC_BASE_URL` distance** | Twilio (often US/SG edge) fetches audio from your host (e.g. Render). High `tts_first_to_outbound_ms` → check hosting region and tunnel stability. |
| **Cold starts** | Serverless/containers waking up can spike first-turn latency. |
| **Sarvam API availability** | When LLM/TTS is unreachable, turns fail fast (~250 ms) with fallback “say again” — different failure mode from slow success. |

---

### F. Application logic (usually smaller than A–E)

| Cause | Impact |
|-------|--------|
| **Language mismatch guard** | User switches language (e.g. says “Hindi”); LLM replies in wrong script → reply discarded → Hindi “repeat” fallback. Wastes a full turn; feels like agent is ignoring the user. |
| **Empty Gather** | Twilio returns empty `SpeechResult` → silent re-listen or “still there?” — not LLM latency but feels like delay. |
| **Dynamic greeting failure** | Falls back to static greeting text; TTS may still run but content/quality differs. |

---

## 5. What “streaming” does and does not do today

| Feature | Gather (trial) | Media Streams |
|---------|----------------|---------------|
| LLM token streaming to TTS | **No** | Yes, if `LLM_STREAMING_ENABLED=true` |
| Sarvam HTTP TTS stream | **Yes** (after full reply text) | Yes |
| Sarvam realtime TTS WebSocket | **No** | Yes, if `SARVAM_REALTIME_TTS_ENABLED=true` |
| Sarvam realtime STT | **No** | Yes, if `SARVAM_REALTIME_STT_ENABLED=true` |
| Speak while user is talking | **No** | Partially (with realtime STT + streaming) |

**Important:** “TTS streaming” on Gather only streams **synthesis** of an already-complete sentence. It does **not** stream **LLM generation** into speech.

---

## 6. Slow-turn Redirect flow (when inline budget is exceeded)

If LLM + TTS cannot finish inside the inline webhook window:

1. `/api/calls/turn` waits up to **`GATHER_INLINE_BUDGET_SECONDS`** (default **5.0 s**).
2. If not done → TwiML **`<Redirect>`** to `/api/calls/turn-result`.
3. Poll up to **`TURN_POLL_WAIT_SECONDS` × `MAX_TURN_POLLS`** (12 s × 6).

This prevents Twilio “could not reach TwiML” errors but **adds silence/hold** on slow turns.

Env knobs: `GATHER_INLINE_BUDGET_SECONDS`, `GATHER_POLL_PAUSE_SECONDS` (0 saves ~1 s).

---

## 7. Observed numbers from live calls

From recent production-style logs (Gather, Sarvam healthy):

| Turn type | `caller_perceived_latency_ms` |
|-----------|-------------------------------|
| Short reply (name question) | ~2.0–2.7 s |
| Medium discovery reply | ~2.1–2.6 s |
| Greeting (buffered TTS at answer) | ~4.4 s (LLM ~1.2 s + TTS ~3.2 s) |

When LLM/TTS fails (API key, outage): ~250 ms to error fallback — caller hears “say again” in English, not slow Gujarati.

**Target documented in `.env.example`:** Gather **~4–6 s** is realistic; **2–3 s** expects Media Streams + realtime flags.

---

## 8. Configuration reference

| Variable | Default | Latency effect |
|----------|---------|----------------|
| `TWILIO_TRIAL_MODE` | `true` | Forces Gather path |
| `PHONE_LLM_REASONING_EFFORT` | `none` | Must stay `none` on phone |
| `PHONE_LLM_MAX_TOKENS` | `320` | Lower = faster, less truncation risk |
| `PHONE_PROMPT_COMPACT` | `true` | Shorter prompt → faster LLM |
| `GATHER_INLINE_BUDGET_SECONDS` | `5.0` | Inline vs Redirect tradeoff |
| `GATHER_POLL_PAUSE_SECONDS` | `0` | Non-zero adds ~1 s on redirect |
| `GATHER_SPEECH_TIMEOUT` | `auto` | End-of-speech detection |
| `GATHER_TIMEOUT_SECONDS` | `7` | Max listen window |
| `SARVAM_TTS_STREAMING` | `true` | HTTP stream vs full buffer |
| `SARVAM_TTS_SPEECH_SAMPLE_RATE` | `8000` | Telephony rate (avoid resample) |
| `LLM_STREAMING_ENABLED` | `false` | Media Streams only |
| `SARVAM_REALTIME_TTS_ENABLED` | `false` | Media Streams only |
| `SARVAM_REALTIME_STT_ENABLED` | `false` | Media Streams only |
| `PUBLIC_BASE_URL` | required | Twilio must reach this for `<Play>` |

---

## 9. How to read logs for a slow turn

Search for these events in order:

1. `gather.transcript.received` — user text arrived
2. `llm.request` → `llm.response` — check `latency_ms`
3. `gather.tts.start` → `tts.first_audio` — check `first_audio_ms`
4. `first_outbound_audio` — check `caller_perceived_latency_ms`
5. `phase_breakdown_ms` — see which segment dominates

**If `llm_ms` is high:** prompt size, reasoning mode, model choice, or JSON repair.  
**If `llm_done_to_tts_first_ms` is high:** TTS stream/buffer issue.  
**If `tts_first_to_outbound_ms` is high:** Twilio ↔ your server network / hosting.  
**If `gather.turn.redirect` appears:** turn exceeded inline budget.

---

## 10. What would actually reduce latency

### On current Gather / trial (incremental)

- Keep `PHONE_LLM_REASONING_EFFORT=none`
- Keep replies short (prompt already limits to 1–2 sentences)
- Ensure `SARVAM_TTS_STREAMING=true` and stable `PUBLIC_BASE_URL`
- Host close to Twilio edge; avoid slow tunnels for production
- Fix language-switch handling so good LLM replies are not thrown away
- Avoid unnecessary Redirects (tune inline budget only if server is fast enough)

**Expected ceiling:** still roughly **3–5 s** on many turns because of Gather + sequential LLM→TTS.

### For materially faster (~2–3 s)

Requires **architecture change**, not a single env tweak:

1. **Upgrade Twilio** — disable trial mode, enable **Media Streams**
2. Enable **`SARVAM_REALTIME_STT_ENABLED=true`**
3. Enable **`SARVAM_REALTIME_TTS_ENABLED=true`**
4. Enable **`LLM_STREAMING_ENABLED=true`**
5. Optional: faster phone LLM via `PHONE_LLM_MODEL` (OpenAI-compatible mini model)

See `README.md` — “For either paid bidirectional carrier…” section.

---

## 11. Summary table

| Layer | Causes delay? | Fixable without Twilio upgrade? |
|-------|---------------|----------------------------------|
| Twilio Gather STT + webhook | **Yes — major** | Partially (timeouts only) |
| Sequential LLM then TTS | **Yes — major** | No on Gather |
| Sarvam LLM (105b, JSON, prompt) | **Yes — moderate** | Partially (compact prompt, reasoning off) |
| Sarvam TTS | **Yes — moderate** | Partially (streaming on) |
| Public URL `<Play>` fetch | **Yes — moderate** | Partially (better hosting) |
| Language mismatch fallbacks | **Yes — UX** | Yes (code fix) |
| API outages / bad key | Fail fast, bad UX | Yes (credentials) |
| Media Streams + realtime flags | **Removes most fixed cost** | **Requires paid Twilio** |

---

## 12. Related files

| File | Role |
|------|------|
| `backend/telephony/turn_flow.py` | Gather turn loop, TTS host, Redirect |
| `backend/telephony/call_manager.py` | Media Streams, VAD, optional realtime |
| `backend/conversation.py` | LLM turn, language guard, streaming hook |
| `backend/metrics.py` | `TurnTimings`, `caller_perceived_latency_ms` |
| `backend/config.py` | All latency-related settings |
| `.env.example` | Documented defaults and targets |

---

*Last updated: 2026-08-10 — reflects Gather trial path as primary deployment.*
