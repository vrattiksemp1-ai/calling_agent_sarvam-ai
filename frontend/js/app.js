"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  providerBadge: $("provider-badge"),
  micBadge: $("mic-badge"),
  startBtn: $("start-btn"),
  talkBtn: $("talk-btn"),
  toggleModeBtn: $("toggle-mode-btn"),
  stopAudioBtn: $("stop-audio-btn"),
  resetBtn: $("reset-btn"),
  recIndicator: $("rec-indicator"),
  processingIndicator: $("processing-indicator"),
  speakingIndicator: $("speaking-indicator"),
  levelBar: $("level-bar"),
  errorBox: $("error-box"),
  transcript: $("transcript"),
  textForm: $("text-form"),
  textInput: $("text-input"),
  sendBtn: $("send-btn"),
  score: $("score"),
  level: $("level"),
  missingFields: $("missing-fields"),
  nextAction: $("next-action"),
  costEstimate: $("cost-estimate"),
  leadFields: $("lead-fields"),
  exportJsonBtn: $("export-json-btn"),
  exportCsvBtn: $("export-csv-btn"),
  debugPanel: $("debug-panel"),
  debugOutput: $("debug-output"),
  callNumber: $("call-number"),
  callBtn: $("call-btn"),
  callStatus: $("call-status"),
};

const state = {
  sessionId: null,
  leadId: null,
  debug: false,
  recording: false,
  tapMode: false,
  micStream: null,
  recorder: null,
  chunks: [],
  audioCtx: null,
  analyser: null,
  audioEl: null,
  audio: null,
  totalCost: 0,
};

/* ---------- helpers ---------- */

function setBadge(el, text, kind) {
  el.textContent = text;
  el.className = "badge badge-" + (kind || "neutral");
}

function setChip(el, text, kind) {
  el.textContent = text;
  el.className = "chip chip-" + (kind || "neutral");
}

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.classList.remove("hidden");
}

function clearError() {
  els.errorBox.classList.add("hidden");
}

function setProcessing(on) {
  els.processingIndicator.classList.toggle("hidden", !on);
  els.talkBtn.disabled = on || !state.sessionId || state.recording;
  els.sendBtn.disabled = on || !state.sessionId || !els.textInput.value.trim();
}

function setSpeaking(on) {
  els.speakingIndicator.classList.toggle("hidden", !on);
  els.stopAudioBtn.disabled = !on;
}

function addMessage(role, text, note) {
  const holder = document.createElement("div");
  holder.className = "msg " + (role === "user" ? "msg-user" : "msg-agent");
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "You" : "Agent";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  holder.appendChild(who);
  holder.appendChild(bubble);
  if (note) {
    const noteEl = document.createElement("div");
    noteEl.className = "audio-note";
    noteEl.textContent = note;
    holder.appendChild(noteEl);
  }
  els.transcript.appendChild(holder);
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function resetTranscript() {
  els.transcript.innerHTML = "";
  const hint = document.createElement("p");
  hint.className = "empty-hint";
  hint.textContent = "Start a conversation to begin. Speak or type to qualify a lead.";
  els.transcript.appendChild(hint);
}

function renderLead(lead) {
  if (!lead) return;
  els.leadId = lead.id || els.leadId;
  els.score.textContent = lead.qualification_score;
  setChip(els.level, lead.qualification_level, {
    cold: "neutral", warm: "warn", hot: "ok",
  }[lead.qualification_level] || "neutral");
  els.missingFields.textContent = lead.missing_important_fields.length
    ? lead.missing_important_fields.join(", ")
    : "none";
  els.nextAction.textContent = lead.recommended_next_action || "–";
  els.leadFields.innerHTML = "";
  const skipped = lead.skipped_fields || [];
  const interesting = [
    "full_name", "phone_number", "email", "company_name", "job_title",
    "city", "country", "business_type", "product_or_service_interest",
    "business_requirement", "main_problem", "estimated_budget",
    "purchase_timeline", "decision_maker_status", "team_size",
    "preferred_contact_method", "preferred_contact_time", "consent_to_contact",
  ];
  for (const key of interesting) {
    const value = (lead.fields || {})[key];
    if (!value) continue;
    const dl = document.createElement("dl");
    dl.className = "lead-field";
    const dt = document.createElement("dt");
    dt.textContent = key.replace(/_/g, " ");
    const dd = document.createElement("dd");
    dd.textContent = skipped.includes(key) ? "(refused)" : value;
    dl.appendChild(dt);
    dl.appendChild(dd);
    els.leadFields.appendChild(dl);
  }
  els.exportJsonBtn.disabled = !els.leadId;
  els.exportCsvBtn.disabled = !els.leadId;
}

function renderCost(turnCost) {
  state.totalCost = state.totalCost + (turnCost || 0);
  els.costEstimate.textContent =
    "₹" + (turnCost || 0).toFixed(4) + " this turn / ₹" + state.totalCost.toFixed(4) + " total";
}

function debugLog(obj) {
  if (!state.debug) return;
  els.debugPanel.classList.remove("hidden");
  const line = JSON.stringify(obj, null, 2);
  els.debugOutput.textContent = line;
}

async function api(url, options) {
  const resp = await fetch(url, options);
  let payload = null;
  try { payload = await resp.json(); } catch (_) { /* non-JSON */ }
  if (!resp.ok) {
    const err = payload && payload.error ? payload.error : { message: "Request failed (" + resp.status + ")." };
    const e = new Error(err.message);
    e.code = err.code;
    e.retryable = err.retryable;
    throw e;
  }
  return payload;
}

/* ---------- microphone ---------- */

async function requestMic() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setBadge(els.micBadge, "Microphone: unsupported", "err");
    showError("This browser does not support microphone access. Use a recent version of Chrome, Edge, Firefox or Safari.");
    return false;
  }
  try {
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    setBadge(els.micBadge, "Microphone: granted", "ok");
    try {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      state.analyser = state.audioCtx.createAnalyser();
      state.analyser.fftSize = 256;
      state.audioCtx.createMediaStreamSource(state.micStream).connect(state.analyser);
      drawLevel();
    } catch (_) { /* level meter is optional */ }
    return true;
  } catch (err) {
    let message = "Microphone access was denied. Allow microphone permission and try again.";
    if (err && (err.name === "NotFoundError" || err.name === "DevicesNotFoundError")) {
      message = "No microphone was found. Connect a microphone and try again.";
    } else if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
      message = "Microphone permission was denied. Grant permission in the browser and retry.";
    }
    setBadge(els.micBadge, "Microphone: blocked", "err");
    showError(message);
    return false;
  }
}

function drawLevel() {
  if (!state.analyser) return;
  const data = new Uint8Array(state.analyser.frequencyBinCount);
  state.analyser.getByteFrequencyData(data);
  let peak = 0;
  for (let i = 0; i < data.length; i++) if (data[i] > peak) peak = data[i];
  els.levelBar.style.width = Math.min(100, Math.round(peak / 2.55)) + "%";
  requestAnimationFrame(drawLevel);
}

/* ---------- recording (push-to-talk) ---------- */

function startRecording() {
  if (!state.sessionId || state.recording || !state.micStream) return;
  state.chunks = [];
  const mime = pickMimeType();
  state.recorder = new MediaRecorder(state.micStream, mime ? { mimeType: mime } : undefined);
  state.recorder.ondataavailable = (e) => { if (e.data.size > 0) state.chunks.push(e.data); };
  state.recorder.onstop = sendRecording;
  state.recorder.start();
  state.recording = true;
  els.recIndicator.classList.remove("hidden");
  els.recIndicator.classList.add("rec-active");
  els.talkBtn.textContent = "Release to send";
  els.talkBtn.classList.add("recording");
}

function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  try { state.recorder.stop(); } catch (_) { /* already stopped */ }
  els.recIndicator.classList.add("hidden");
  els.recIndicator.classList.remove("rec-active");
  els.talkBtn.textContent = state.tapMode ? "Stop (tap again)" : "Hold to talk";
  els.talkBtn.classList.remove("recording");
}

function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  return candidates.find((c) => window.MediaRecorder && MediaRecorder.isTypeSupported(c)) || "";
}

async function sendRecording() {
  if (state.chunks.length === 0) {
    showError("The recording was empty. Please record again.");
    return;
  }
  const blob = new Blob(state.chunks, { type: state.recorder.mimeType || "audio/webm" });
  const mime = state.recorder.mimeType || "audio/webm";
  const ext = mime.includes("mp4") ? "mp4" : "webm";
  const form = new FormData();
  form.append("file", blob, "recording." + ext);
  clearError();
  setProcessing(true);
  try {
    const data = await api(`/api/sessions/${state.sessionId}/audio`, { method: "POST", body: form });
    handleTurn(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setProcessing(false);
  }
}

/* ---------- conversation ---------- */

async function startConversation() {
  clearError();
  const ok = await requestMic();
  if (!ok) return;
  try {
    const data = await api("/api/sessions", { method: "POST" });
    state.sessionId = data.session_id;
    state.totalCost = 0;
    resetTranscript();
    renderLead(data.lead);
    renderCost(0);
    addMessage("agent", data.greeting);
    if (data.warning) showError(data.warning);
    if (data.audio_base64) {
      playAudio(data.audio_base64, data.audio_mime || "audio/wav");
    }
    enableControls(true);
    setBadge(els.providerBadge, "Provider: configured", "ok");
    if (state.tapMode) {
      els.talkBtn.textContent = "Tap to talk";
    }
  } catch (err) {
    showError("Could not start a session: " + err.message);
  }
}

async function sendText() {
  const text = els.textInput.value.trim();
  if (!text || !state.sessionId) return;
  els.textInput.value = "";
  addMessage("user", text);
  clearError();
  setProcessing(true);
  try {
    const data = await api(`/api/sessions/${state.sessionId}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    handleTurn(data, { textOnly: true });
  } catch (err) {
    showError(err.message);
  } finally {
    setProcessing(false);
  }
}

function handleTurn(data) {
  if (data.transcript) addMessage("user", data.transcript, data.warning || null);
  addMessage("agent", data.assistant_message);
  renderLead(data.lead);
  debugLog(data);
  const turnCost = (data.metrics && data.metrics.estimated_provider_cost) || 0;
  renderCost(turnCost);
  if (data.warning) showError(data.warning);
  if (data.audio_base64) {
    playAudio(data.audio_base64, data.audio_mime || "audio/wav");
  }
  if (data.conversation_status === "completed") {
    showError("Conversation completed and lead saved. You can export it or reset.");
  }
}

function playAudio(base64Data, mime) {
  stopAudio();
  const bytes = atob(base64Data);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  state.audio = new Blob([arr], { type: mime });
  const url = URL.createObjectURL(state.audio);
  state.audioEl = new Audio(url);
  state.audioEl.onplay = () => setSpeaking(true);
  state.audioEl.onended = () => { setSpeaking(false); URL.revokeObjectURL(url); };
  state.audioEl.onerror = () => { setSpeaking(false); URL.revokeObjectURL(url); };
  state.audioEl.play().catch(() => {
    setSpeaking(false);
    showError("The agent's audio could not be played, but the text reply is shown above.");
  });
}

function stopAudio() {
  if (state.audioEl) {
    try { state.audioEl.pause(); state.audioEl.src = ""; } catch (_) { /* noop */ }
    state.audioEl = null;
  }
  setSpeaking(false);
}

function enableControls(on) {
  els.talkBtn.disabled = !on;
  els.toggleModeBtn.disabled = !on;
  els.resetBtn.disabled = !on;
  els.textInput.disabled = !on;
  els.sendBtn.disabled = !on;
  if (!on) {
    els.talkBtn.textContent = "Hold to talk";
    els.talkBtn.classList.remove("recording");
  }
}

async function resetConversation() {
  if (!state.sessionId) return;
  stopAudio();
  clearError();
  try {
    const data = await api(`/api/sessions/${state.sessionId}/reset`, { method: "POST" });
    state.totalCost = 0;
    resetTranscript();
    renderLead(data.lead);
    renderCost(0);
    addMessage("agent", "Conversation reset. " + (data.greeting || "Ready when you are."));
  } catch (err) {
    showError(err.message);
  }
}

async function refreshStatus() {
  try {
    const data = await api("/api/provider/status");
    setBadge(els.providerBadge, "Provider: " + data.status, data.status);
    debugLog(data);
  } catch (_) {
    setBadge(els.providerBadge, "Provider: unreachable", "err");
  }
}

async function loadConfig() {
  try {
    const data = await api("/api/config");
    state.debug = data.debug;
    if (data.debug) els.debugPanel.classList.remove("hidden");
  } catch (_) { /* config is optional at load */ }
}

function exportLead(format) {
  if (!state.leadId) return;
  window.location.href = `/api/leads/${state.leadId}/export.${format}`;
}

/* ---------- phone call ---------- */

async function placeCall() {
  const number = els.callNumber.value.trim();
  if (!number) {
    showError("Enter your phone number first (E.164 format, e.g. +919876543210).");
    return;
  }
  clearError();
  els.callBtn.disabled = true;
  els.callStatus.textContent = "Placing call…";
  try {
    const data = await api("/api/calls", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ to: number }),
    });
    els.callStatus.textContent = "Calling " + number + " — answer your phone and talk to the agent.";
    pollCallStatus(data.call_sid, number);
  } catch (err) {
    els.callStatus.textContent = "";
    showError(err.message);
  } finally {
    els.callBtn.disabled = false;
  }
}

async function pollCallStatus(callSid, number) {
  try {
    const data = await api(`/api/calls/${callSid}`);
    const status = data.status;
    if (status === "completed" || status === "failed") {
      els.callStatus.textContent = "Call to " + number + ": " + status + ".";
      return;
    }
    els.callStatus.textContent = "Call " + status + " — the agent will speak when you answer.";
    setTimeout(() => pollCallStatus(callSid, number), 2000);
  } catch (_) {
    els.callStatus.textContent = "Call ended.";
  }
}

/* ---------- event wiring ---------- */

els.startBtn.addEventListener("click", startConversation);
els.sendBtn.addEventListener("click", (e) => { e.preventDefault(); sendText(); });
els.textForm.addEventListener("submit", (e) => { e.preventDefault(); sendText(); });
els.resetBtn.addEventListener("click", resetConversation);
els.exportJsonBtn.addEventListener("click", () => exportLead("json"));
els.exportCsvBtn.addEventListener("click", () => exportLead("csv"));
els.stopAudioBtn.addEventListener("click", stopAudio);
els.callBtn.addEventListener("click", placeCall);
els.callNumber.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); placeCall(); } });

els.toggleModeBtn.addEventListener("click", () => {
  state.tapMode = !state.tapMode;
  els.toggleModeBtn.textContent = state.tapMode ? "Use hold-to-talk" : "Use tap-to-talk";
  els.talkBtn.textContent = state.tapMode ? "Tap to talk" : "Hold to talk";
  if (!state.tapMode && state.recording) stopRecording();
});

const TALK_HOLD_MS = 300;
let pressTimer = null;

els.talkBtn.addEventListener("mousedown", (e) => { e.preventDefault(); if (state.tapMode) return; pressTimer = setTimeout(startRecording, TALK_HOLD_MS); });
els.talkBtn.addEventListener("mouseup", () => { if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; } if (!state.tapMode) stopRecording(); });
els.talkBtn.addEventListener("mouseleave", () => { if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; } if (!state.tapMode && state.recording) stopRecording(); });
els.talkBtn.addEventListener("touchstart", (e) => { e.preventDefault(); if (state.tapMode) { state.recording ? stopRecording() : startRecording(); return; } pressTimer = setTimeout(startRecording, TALK_HOLD_MS); }, { passive: false });
els.talkBtn.addEventListener("touchend", (e) => { e.preventDefault(); if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; } if (!state.tapMode) stopRecording(); }, { passive: false });

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && document.activeElement === els.talkBtn && !state.tapMode && !e.repeat) {
    e.preventDefault();
    startRecording();
  }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && document.activeElement === els.talkBtn && !state.tapMode) {
    e.preventDefault();
    stopRecording();
  }
});

loadConfig();
refreshStatus();
