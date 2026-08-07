import csv
import json

import pytest

from backend.benchmarking import (
    CELLS,
    build_report,
    import_custom_sqlite,
    percentile,
    prepare_run,
    register_cell,
    validate_specs,
)
from backend.config import Settings
from backend.database import create_engine_and_session
from backend.models import LEAD_FIELDS, Message, ProviderEvent, Session
from backend.prompts import build_system_prompt
from backend.telephony.call_manager import fallback_text


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_checked_in_specs_and_corpus_validate():
    assert validate_specs() == []


def test_managed_spec_matches_current_agent_defaults():
    spec = _read_json(
        __import__("pathlib").Path(__file__).parents[1]
        / "benchmark"
        / "managed_pilot.json"
    )
    settings = Settings(_env_file=None)
    assert spec["identity"]["business_name"] == settings.business_name
    assert spec["identity"]["business_description"] == settings.business_description
    assert spec["fields"] == LEAD_FIELDS
    assert spec["voice"]["tts_model"] == settings.sarvam_tts_model
    assert spec["voice"]["speaker"] == settings.sarvam_tts_speaker
    assert spec["voice"]["temperature"] == settings.sarvam_tts_temperature
    assert spec["voice"]["pace"] == settings.sarvam_tts_pace
    assert spec["privacy"]["retain_audio"] == settings.retain_audio is False
    assert "AI assistant" in build_system_prompt("Vrattiks", "technology company")
    assert "AI assistant" in fallback_text("greeting", "en")
    assert "AI assistant" in fallback_text("greeting", "hi")
    assert "AI assistant" in fallback_text("greeting", "gu")


def test_corpus_has_required_coverage_without_audio():
    corpus = _read_json(
        __import__("pathlib").Path(__file__).parents[1] / "benchmark" / "corpus.json"
    )
    assert corpus["retain_audio"] is False
    assert len(corpus["scenarios"]) >= 20
    languages = {row["language"] for row in corpus["scenarios"]}
    assert {"Gujarati", "Hindi", "English", "Gujlish", "Hinglish"} <= languages
    tags = {tag for row in corpus["scenarios"] for tag in row["tags"]}
    assert {
        "gsm_noise",
        "long_pause",
        "cross_talk",
        "barge_in",
        "mid_sentence_switch",
        "consent",
        "lead_extraction",
    } <= tags
    assert corpus["expansion_templates"]["100_calls"]["total_calls"] == 100
    assert corpus["expansion_templates"]["200_calls"]["total_calls"] == 200
    assert all("audio" not in row for row in corpus["scenarios"])


def test_prepare_and_registration_are_safe(tmp_path):
    manifest = prepare_run(
        tmp_path / "run", calls_per_cell=2, run_id="safety-test"
    )
    assert manifest["total_calls"] == 2 * len(CELLS)
    assert all(row["chargeable_call_placed"] is False for row in manifest["registrations"])

    dry_run = register_cell(
        tmp_path / "run",
        cell="managed+twilio",
        destination="+910000000000",
    )
    assert dry_run["status"] == "dry_run_registered"
    assert dry_run["chargeable_call_placed"] is False

    with pytest.raises(ValueError, match="exactly match"):
        register_cell(
            tmp_path / "run",
            cell="managed+twilio",
            destination="+910000000000",
            execute=True,
            confirm_destination="+919999999999",
        )

    approved = register_cell(
        tmp_path / "run",
        cell="managed+twilio",
        destination="+910000000000",
        execute=True,
        confirm_destination="+910000000000",
    )
    assert approved["status"] == "approved_for_external_execution"
    assert approved["chargeable_call_placed"] is False


def test_report_with_no_data_is_insufficient_evidence(tmp_path):
    run_dir = tmp_path / "run"
    prepare_run(run_dir, calls_per_cell=1, run_id="no-data")
    report = build_report(run_dir)
    assert report["selection"]["status"] == "insufficient_evidence"
    assert all(cell["gate_status"] == "insufficient_evidence" for cell in report["cells"])
    assert (run_dir / "report.json").exists()
    with (run_dir / "report.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4


def test_custom_sqlite_import_reads_provider_events_and_messages(tmp_path):
    database = tmp_path / "telemetry.db"
    _, factory = create_engine_and_session(f"sqlite:///{database.as_posix()}")
    with factory() as db:
        session = Session(status="completed")
        db.add(session)
        db.flush()
        db.add(
            Message(
                session_id=session.id,
                role="assistant",
                content="done",
                estimated_provider_cost=3.5,
            )
        )
        for event_type, latency in (
            ("utterance_end", 100),
            ("first_outbound_audio", 1100),
            ("interruption_clear_delay", 120),
        ):
            db.add(
                ProviderEvent(
                    session_id=session.id,
                    provider="twilio",
                    event_type=event_type,
                    latency_ms=latency,
                    request_id="turn-1",
                )
            )
        db.commit()

    run_dir = tmp_path / "run"
    prepare_run(run_dir, calls_per_cell=1, run_id="sqlite")
    assert import_custom_sqlite(
        run_dir, database, cell="custom+twilio"
    ) == 1
    path = run_dir / "evidence" / "custom_twilio-custom_sqlite.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["completed"] is True
    assert row["cost_inr"] == 3.5
    assert row["speech_end_to_first_audio_ms_samples"] == [1000]
    assert row["barge_in_to_silence_ms_samples"] == [120]


def test_report_math_and_exact_plan_gates(tmp_path):
    run_dir = tmp_path / "run"
    prepare_run(run_dir, calls_per_cell=1, run_id="math")
    evidence = run_dir / "evidence"
    evidence.mkdir()
    rows = [
        {
            "cell": "managed+exotel",
            "call_id": "call-1",
            "completed": True,
            "speech_end_to_first_audio_ms": 1000,
            "barge_in_to_silence_ms": 100,
            "false_endpoints": 0,
            "endpoint_opportunities": 50,
            "language_switch_clauses": 1,
            "entity_correct": 9,
            "entity_total": 10,
            "task_success": True,
            "gujarati_native_speaker_rating": 4,
            "cost_inr": 10,
        },
        {
            "cell": "managed+exotel",
            "call_id": "call-2",
            "completed": True,
            "speech_end_to_first_audio_ms": 2000,
            "barge_in_to_silence_ms": 300,
            "false_endpoints": 0,
            "endpoint_opportunities": 50,
            "language_switch_success": True,
            "entity_correct": 8,
            "entity_total": 10,
            "task_success": True,
            "gujarati_native_speaker_rating": 5,
            "cost_inr": 14,
        },
    ]
    (evidence / "managed_exotel-test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    worksheet = _read_json(
        __import__("pathlib").Path(__file__).parents[1]
        / "benchmark"
        / "compliance_residency.json"
    )
    worksheet["cells"]["managed+exotel"] = {
        name: "verified" for name in worksheet["required_fields"]
    }
    compliance = tmp_path / "compliance.json"
    compliance.write_text(json.dumps(worksheet), encoding="utf-8")

    report = build_report(run_dir, compliance_path=compliance)
    cell = next(row for row in report["cells"] if row["cell"] == "managed+exotel")
    assert cell["metrics"]["speech_end_to_first_audio"]["p50_ms"] == 1500
    assert cell["metrics"]["speech_end_to_first_audio"]["p95_ms"] == 1950
    assert cell["metrics"]["barge_in_to_silence"]["p95_ms"] == 290
    assert cell["metrics"]["entity_accuracy"]["value"] == 0.85
    assert cell["metrics"]["task_accuracy"]["value"] == 1
    assert cell["metrics"]["gujarati_native_speaker_rating"]["mean"] == 4.5
    assert cell["metrics"]["cost_per_completed_call_inr"]["value"] == 12
    assert cell["gate_status"] == "passed"
    assert report["selection"]["cell"] == "managed+exotel"


def test_false_endpoint_gate_is_strictly_below_two_percent(tmp_path):
    run_dir = tmp_path / "run"
    prepare_run(run_dir, calls_per_cell=1, run_id="strict")
    evidence = run_dir / "evidence"
    evidence.mkdir()
    row = {
        "cell": "custom+twilio",
        "call_id": "call-1",
        "completed": True,
        "speech_end_to_first_audio_ms": 1000,
        "barge_in_to_silence_ms": 100,
        "false_endpoints": 1,
        "endpoint_opportunities": 50,
        "language_switch_clauses": 1,
        "entity_correct": 1,
        "entity_total": 1,
        "task_success": True,
        "gujarati_native_speaker_rating": 5,
        "cost_inr": 1,
    }
    (evidence / "custom_twilio-test.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    worksheet = _read_json(
        __import__("pathlib").Path(__file__).parents[1]
        / "benchmark"
        / "compliance_residency.json"
    )
    worksheet["cells"]["custom+twilio"] = {
        name: "verified" for name in worksheet["required_fields"]
    }
    compliance = tmp_path / "compliance.json"
    compliance.write_text(json.dumps(worksheet), encoding="utf-8")

    report = build_report(run_dir, compliance_path=compliance)
    cell = next(row for row in report["cells"] if row["cell"] == "custom+twilio")
    assert cell["metrics"]["false_endpoint_rate"]["value"] == 0.02
    assert cell["exact_plan_gates"]["false_endpoint_rate"] is False
    assert cell["gate_status"] == "failed"
    assert report["selection"]["status"] == "insufficient_evidence"


def test_percentile_uses_linear_interpolation():
    assert percentile([], 0.95) is None
    assert percentile([100], 0.95) == 100
    assert percentile([100, 300], 0.95) == 290
