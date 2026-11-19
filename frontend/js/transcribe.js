"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  provider: $("speech-provider"),
  lockHint: $("provider-lock-hint"),
  badge: $("provider-badge"),
  form: $("upload-form"),
  file: $("audio-file"),
  dropZone: $("drop-zone"),
  dropText: $("drop-text"),
  numSpeakers: $("num-speakers"),
  notice: $("provider-notice"),
  btn: $("transcribe-btn"),
  error: $("error-box"),
  progressBox: $("progress-box"),
  progressMessage: $("progress-message"),
  progressElapsed: $("progress-elapsed"),
  progressBar: $("progress-bar"),
  resultBox: $("result-box"),
  summary: $("result-summary"),
  download: $("download-btn"),
  copy: $("copy-btn"),
  text: $("transcript-text"),
  recent: $("recent-list"),
};

const PROVIDER_STORAGE_KEY = "speechProvider"; // shared with the calling page
const LABELS = { sarvam: "Sarvam", bhashini: "Bhashini" };
const NOTICES = {
  sarvam: "Sarvam transcribes the whole file in one batch job (up to 2 hours) and labels speakers itself. A 30-minute recording usually takes a few minutes.",
  bhashini: "Bhashini transcribes the text accurately, but it can only identify speakers in 2-minute parts: on recordings longer than 2 minutes the speaker numbers are unreliable (one person may appear as several speakers). Use Sarvam when speaker labels matter. Bhashini allows proof-of-concept use only.",
};
const STATUS_BADGE = { ok: "ok", degraded: "warn", error: "err" };

const state = { busy: false, pollTimer: null, providerInfo: {} };

function show(el, on) { el.classList.toggle("hidden", !on); }
function showError(msg) { els.error.textContent = msg; show(els.error, true); }
function clearError() { show(els.error, false); }

async function api(url, options) {
  const opts = options ? { ...options } : {};
  const headers = new Headers(opts.headers || {});
  headers.set("ngrok-skip-browser-warning", "true");
  opts.headers = headers;
  const resp = await fetch(url, opts);
  let payload = null;
  try { payload = await resp.json(); } catch (_) { /* non-JSON */ }
  if (!resp.ok) {
    const err = payload && payload.error ? payload.error : { message: "Request failed (" + resp.status + ")." };
    throw new Error(err.message);
  }
  return payload;
}

function provider() { return els.provider.value || "sarvam"; }

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return "–";
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
}

function formatSize(bytes) {
  return bytes >= 1048576 ? (bytes / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(bytes / 1024)) + " KB";
}

/* ---------- provider ---------- */

function applyProvider() {
  els.notice.textContent = NOTICES[provider()] || "";
  els.numSpeakers.disabled = provider() === "bhashini";
  const info = state.providerInfo[provider()];
  if (info && info.configured === false) {
    showError(LABELS[provider()] + " keys are not set yet. Add them to .env and restart the server.");
  } else {
    clearError();
  }
  refreshStatus();
}

async function refreshStatus() {
  const p = provider();
  els.badge.textContent = LABELS[p] + ": checking…";
  els.badge.className = "badge badge-neutral";
  try {
    const data = await api("/api/provider/status?speech_provider=" + encodeURIComponent(p));
    if (p !== provider()) return;
    els.badge.textContent = LABELS[p] + ": " + data.status;
    els.badge.className = "badge badge-" + (STATUS_BADGE[data.status] || "neutral");
    els.badge.title = data.message || "";
  } catch (_) {
    els.badge.textContent = LABELS[p] + ": unreachable";
    els.badge.className = "badge badge-err";
  }
}

async function loadConfig() {
  let saved = null;
  try { saved = localStorage.getItem(PROVIDER_STORAGE_KEY); } catch (_) { /* optional */ }
  try {
    const data = await api("/api/config");
    const providers = data.speech_providers || [];
    if (providers.length) {
      els.provider.innerHTML = "";
      providers.forEach((p) => {
        state.providerInfo[p.id] = p;
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.label + (p.configured ? "" : " (keys not set)");
        els.provider.appendChild(opt);
      });
      const ids = providers.map((p) => p.id);
      els.provider.value = ids.includes(saved) ? saved : (ids.includes(data.speech_provider) ? data.speech_provider : ids[0]);
    }
  } catch (_) {
    if (saved && LABELS[saved]) els.provider.value = saved;
  }
  applyProvider();
}

/* ---------- upload + polling ---------- */

function setBusy(on) {
  state.busy = on;
  els.btn.disabled = on || !els.file.files.length;
  els.btn.textContent = on ? "Transcribing…" : "Transcribe";
  els.provider.disabled = on;
  els.file.disabled = on;
  els.numSpeakers.disabled = on || provider() === "bhashini";
  show(els.lockHint, on);
}

function setProgress(message, fraction, elapsed) {
  show(els.progressBox, true);
  els.progressMessage.textContent = message;
  els.progressBar.style.width = Math.round((fraction || 0) * 100) + "%";
  els.progressElapsed.textContent = elapsed != null ? formatDuration(elapsed) + " elapsed" : "";
}

function onFileChosen() {
  const f = els.file.files[0];
  els.dropText.textContent = f ? `${f.name} (${formatSize(f.size)})` : "Choose a file or drop it here";
  els.btn.disabled = state.busy || !f;
}

function uploadWithProgress(form) {
  // XMLHttpRequest reports upload progress; large meeting files take a while.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/transcriptions");
    xhr.setRequestHeader("ngrok-skip-browser-warning", "true");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress(`Uploading ${Math.round((e.loaded / e.total) * 100)}%`, 0.02 * (e.loaded / e.total), null);
    };
    xhr.onload = () => {
      let payload = null;
      try { payload = JSON.parse(xhr.responseText); } catch (_) { /* non-JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
      else reject(new Error((payload && payload.error && payload.error.message) || `Upload failed (${xhr.status}).`));
    };
    xhr.onerror = () => reject(new Error("Upload failed. Check your connection and try again."));
    xhr.send(form);
  });
}

async function startTranscription(e) {
  e.preventDefault();
  const f = els.file.files[0];
  if (!f || state.busy) return;
  clearError();
  show(els.resultBox, false);
  setBusy(true);
  setProgress("Uploading 0%", 0, null);
  const form = new FormData();
  form.append("file", f, f.name);
  form.append("speech_provider", provider());
  if (els.numSpeakers.value && provider() === "sarvam") form.append("num_speakers", els.numSpeakers.value);
  try {
    const job = await uploadWithProgress(form);
    pollJob(job.job_id);
  } catch (err) {
    setBusy(false);
    show(els.progressBox, false);
    showError(err.message);
  }
}

async function pollJob(jobId) {
  try {
    const job = await api("/api/transcriptions/" + jobId);
    if (job.status === "completed") {
      setBusy(false);
      show(els.progressBox, false);
      renderResult(job);
      loadRecent();
      return;
    }
    if (job.status === "failed") {
      setBusy(false);
      show(els.progressBox, false);
      showError("Transcription failed: " + (job.error || job.message));
      loadRecent();
      return;
    }
    setProgress(job.message, job.progress, job.elapsed_seconds);
  } catch (err) {
    setProgress("Checking progress… (" + err.message + ")", null, null);
  }
  state.pollTimer = setTimeout(() => pollJob(jobId), 2000);
}

function renderResult(job) {
  els.summary.textContent =
    `${job.filename} · ${formatDuration(job.duration_seconds)} · ${job.speakers} speaker(s) · ` +
    `${(job.languages || []).join(", ") || "no speech"} · ${LABELS[job.speech_provider] || job.speech_provider} · ` +
    `took ${formatDuration(job.elapsed_seconds)}`;
  els.text.textContent = job.transcript || "";
  els.download.href = `/api/transcriptions/${job.job_id}/download`;
  els.download.setAttribute("download", job.download_name || "transcript.txt");
  show(els.resultBox, true);
}

async function loadRecent() {
  try {
    const jobs = await api("/api/transcriptions");
    els.recent.innerHTML = "";
    if (!jobs.length) {
      els.recent.innerHTML = '<li class="muted-text">None yet.</li>';
      return;
    }
    jobs.forEach((job) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = `${job.filename} · ${LABELS[job.speech_provider] || job.speech_provider} · `;
      const status = document.createElement("span");
      status.className = "recent-status recent-" + job.status;
      status.textContent = job.status === "completed"
        ? `${formatDuration(job.duration_seconds)}, ${job.speakers} speaker(s)`
        : job.status;
      li.appendChild(name);
      li.appendChild(status);
      if (job.status === "completed") {
        const view = document.createElement("button");
        view.className = "link-btn";
        view.type = "button";
        view.textContent = "View";
        view.addEventListener("click", async () => renderResult(await api("/api/transcriptions/" + job.job_id)));
        const dl = document.createElement("a");
        dl.href = `/api/transcriptions/${job.job_id}/download`;
        dl.textContent = "Download";
        dl.setAttribute("download", job.download_name || "transcript.txt");
        li.appendChild(view);
        li.appendChild(dl);
      } else if (job.status === "failed" && job.error) {
        li.title = job.error;
      }
      els.recent.appendChild(li);
    });
  } catch (_) { /* list is optional */ }
}

/* ---------- wiring ---------- */

els.provider.addEventListener("change", () => {
  try { localStorage.setItem(PROVIDER_STORAGE_KEY, provider()); } catch (_) { /* optional */ }
  applyProvider();
});
els.file.addEventListener("change", onFileChosen);
els.form.addEventListener("submit", startTranscription);
els.copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(els.text.textContent);
    els.copy.textContent = "Copied";
    setTimeout(() => { els.copy.textContent = "Copy text"; }, 1500);
  } catch (_) { showError("Copy failed; select the text and copy it manually."); }
});
["dragenter", "dragover"].forEach((ev) => els.dropZone.addEventListener(ev, (e) => {
  e.preventDefault();
  if (!state.busy) els.dropZone.classList.add("drag-over");
}));
["dragleave", "drop"].forEach((ev) => els.dropZone.addEventListener(ev, (e) => {
  e.preventDefault();
  els.dropZone.classList.remove("drag-over");
}));
els.dropZone.addEventListener("drop", (e) => {
  if (state.busy || !e.dataTransfer.files.length) return;
  els.file.files = e.dataTransfer.files;
  onFileChosen();
});

loadConfig();
loadRecent();
