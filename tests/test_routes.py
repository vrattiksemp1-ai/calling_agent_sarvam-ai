"""End-to-end API route tests with fully mocked providers."""


def _create_session(client) -> str:
    resp = client.post("/api/sessions")
    assert resp.status_code == 200
    return resp.json()["session_id"]


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "Sarvam Cloud Lead Agent"


def test_index_serves_frontend(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Sarvam Cloud Lead Agent" in resp.text


def test_config_endpoint(client):
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "sarvam-cloud"
    assert body["stt_model"] == "saaras:v3"
    assert body["tts_model"] == "bulbul:v3"


def test_provider_status(client):
    resp = client.get("/api/provider/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_session(client):
    resp = client.post("/api/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"]
    assert body["current_state"] == "greeting"
    assert "greeting" in body
    assert body["audio_base64"]
    assert body["audio_mime"]


def test_get_session_not_found(client):
    resp = client.get("/api/sessions/nonexistent")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_text_conversation_full_flow(client):
    sid = _create_session(client)
    for turn in [
        "my name is rahul sharma",
        "my phone is 9876543210",
        "I need a CRM with voice automation",
        "yes sure you can contact me",
        "yes I confirm",
    ]:
        resp = client.post(f"/api/sessions/{sid}/message", json={"text": turn})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["assistant_message"]
        assert body["audio_base64"]
        assert body["audio_mime"]

    summary = client.get(f"/api/sessions/{sid}/summary").json()
    assert summary["completion_status"] == "completed"
    assert summary["qualification_score"] >= 70
    assert summary["qualification_level"] == "hot"

    lead = client.get(f"/api/sessions/{sid}/lead").json()
    assert lead["fields"]["full_name"] == "Rahul Sharma"
    assert lead["qualification_level"] == "hot"


def test_audio_conversation_flow(client):
    sid = _create_session(client)
    # A tiny fake WebM blob; the mocked provider still returns a transcript.
    blob = b"\x1aE\xdf\xa3\x01fake-webm-data"
    resp = client.post(
        f"/api/sessions/{sid}/audio",
        files={"file": ("recording.webm", blob, "audio/webm")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["transcript"]
    assert body["assistant_message"]
    assert body["metrics"]["audio_duration_ms"] >= 0
    # Audio mode should produce audio bytes and estimate a cloud cost.
    assert body["audio_base64"]
    assert body["metrics"]["estimated_provider_cost"] > 0


def test_empty_audio_rejected(client):
    sid = _create_session(client)
    resp = client.post(
        f"/api/sessions/{sid}/audio",
        files={"file": ("empty.webm", b"", "audio/webm")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_AUDIO"


def test_oversized_audio_rejected(client):
    from backend.audio import validate_upload
    from backend.errors import InvalidAudioError

    big = b"x" * (16 * 1024 * 1024)
    try:
        validate_upload("big.webm", "audio/webm", len(big), 15)
        raise AssertionError("should have raised")
    except InvalidAudioError as exc:
        assert "too large" in exc.message


def test_confirm_endpoint(client):
    sid = _create_session(client)
    resp = client.post(f"/api/sessions/{sid}/confirm", json={"confirmed": True})
    assert resp.status_code == 200
    assert resp.json()["conversation_status"] in {"completed", "in_progress", "active"}


def test_reset_session(client):
    sid = _create_session(client)
    resp = client.post(f"/api/sessions/{sid}/reset")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_state"] == "greeting"
    assert body["lead"]["qualification_score"] == 0


def test_delete_session(client):
    sid = _create_session(client)
    resp = client.delete(f"/api/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_leads_list_and_delete(client):
    sid = _create_session(client)
    client.post(f"/api/sessions/{sid}/message", json={"text": "my name is rahul sharma"})
    leads = client.get("/api/leads").json()
    assert isinstance(leads, list)
    if leads:
        lid = leads[0]["id"]
        resp = client.delete(f"/api/leads/{lid}")
        assert resp.status_code == 200


def test_missing_lead_404(client):
    assert client.get("/api/leads/99999").status_code == 404
    assert client.get("/api/leads/99999/export.json").status_code == 404


def test_error_shape_consistent(client):
    resp = client.get("/api/sessions/missing123")
    body = resp.json()
    assert set(body["error"].keys()) == {"code", "message", "retryable", "details"}
    assert body["error"]["code"] == "SESSION_NOT_FOUND"
