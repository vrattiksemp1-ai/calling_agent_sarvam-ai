"""Pipeline trace helpers: clipping + secret redaction."""

from backend.pipeline_trace import clip, configure_pipeline_trace, summarize_messages, trace


def test_clip_redacts_secret_keys():
    configure_pipeline_trace(enabled=True, max_chars=2000)
    payload = clip(
        {
            "Authorization": "Bearer secret",
            "api_key": "abc",
            "transcript": "hello caller",
            "nested": {"token": "xyz", "ok": 1},
        }
    )
    assert payload["Authorization"] == "[REDACTED]"
    assert payload["api_key"] == "[REDACTED]"
    assert payload["nested"]["token"] == "[REDACTED]"
    assert payload["transcript"] == "hello caller"
    assert payload["nested"]["ok"] == 1


def test_clip_truncates_long_strings():
    # Floor is 200 chars (configure_pipeline_trace clamps low values).
    configure_pipeline_trace(enabled=True, max_chars=200)
    out = clip("x" * 300)
    assert out.startswith("x" * 200)
    assert "truncated" in out


def test_summarize_messages_includes_roles():
    configure_pipeline_trace(enabled=True, max_chars=2000)
    summary = summarize_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert summary["count"] == 2
    assert summary["messages"][0]["role"] == "system"
    assert summary["messages"][1]["content"] == "hi"


def test_trace_disabled_is_noop(caplog):
    configure_pipeline_trace(enabled=False)
    with caplog.at_level("INFO"):
        trace("should.not.log", foo="bar")
    assert "pipeline=" not in caplog.text
    configure_pipeline_trace(enabled=True)
