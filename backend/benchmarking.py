"""Offline four-cell voice benchmark preparation, import, and reporting.

The harness never calls a telephony API. ``--execute`` authorizes a schedule
record for a separately controlled external operator only after an exact
destination confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.models import LEAD_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
CELLS = (
    "managed+exotel",
    "managed+twilio",
    "custom+exotel",
    "custom+twilio",
)
MANAGED_CELLS = {"managed+exotel", "managed+twilio"}
CUSTOM_CELLS = {"custom+exotel", "custom+twilio"}
PLAN_GATES = {
    "speech_end_to_first_audio_p50_ms_max": 1500,
    "speech_end_to_first_audio_p95_ms_max": 2500,
    "barge_in_to_silence_p95_ms_max": 300,
    "false_endpoint_rate_max_exclusive": 0.02,
    "language_switch_max_clauses": 1,
}
CANONICAL_EVIDENCE_FIELDS = {
    "cell",
    "scenario_id",
    "call_id",
    "turn_id",
    "completed",
    "speech_end_to_first_audio_ms",
    "speech_end_to_first_audio_ms_samples",
    "barge_in_to_silence_ms",
    "barge_in_to_silence_ms_samples",
    "false_endpoints",
    "endpoint_opportunities",
    "language_switch_clauses",
    "language_switch_success",
    "entity_correct",
    "entity_total",
    "task_success",
    "gujarati_native_speaker_rating",
    "cost_inr",
    "source",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_specs(spec_dir: Path = BENCHMARK_DIR) -> list[str]:
    """Validate the checked-in, credential-free benchmark definitions."""
    errors: list[str] = []
    managed = _load_json(spec_dir / "managed_pilot.json")
    corpus = _load_json(spec_dir / "corpus.json")
    compliance = _load_json(spec_dir / "compliance_residency.json")

    if managed.get("credential_policy", {}).get("contains_secrets") is not False:
        errors.append("managed_pilot must declare contains_secrets=false")
    if managed.get("fields") != LEAD_FIELDS:
        errors.append("managed_pilot fields differ from backend.models.LEAD_FIELDS")
    endpoints = managed.get("provisioning", {}).get("carrier_endpoints", {})
    if endpoints.get("twilio", {}).get("webhook") != (
        "https://apps.sarvam.ai/api/app-runtime/channels/twilio"
    ):
        errors.append("managed Twilio webhook is missing or incorrect")
    if endpoints.get("exotel", {}).get("voicebot_endpoint") != (
        "https://apps.sarvam.ai/api/app-runtime/channels/exotel"
    ):
        errors.append("managed Exotel Voicebot endpoint is missing or incorrect")
    if endpoints.get("exotel", {}).get("mumbai_base_url") != (
        "https://api.in.exotel.com"
    ):
        errors.append("Exotel Mumbai base URL is missing or incorrect")
    if managed.get("privacy", {}).get("recording_enabled_by_default") is not False:
        errors.append("managed recording must default off")
    if managed.get("privacy", {}).get("retain_audio") is not False:
        errors.append("managed retain_audio must be false")
    if managed.get("greeting", {}).get("must_explicitly_identify_ai") is not True:
        errors.append("managed greeting must explicitly identify the AI assistant")

    scenarios = corpus.get("scenarios", [])
    if len(scenarios) < 20:
        errors.append("corpus requires at least 20 smoke scenarios")
    ids = [row.get("id") for row in scenarios]
    if len(ids) != len(set(ids)):
        errors.append("corpus scenario IDs must be unique")
    required_languages = {"Gujarati", "Hindi", "English", "Gujlish", "Hinglish"}
    languages = {row.get("language") for row in scenarios}
    if not required_languages.issubset(languages):
        errors.append("corpus does not cover all required languages")
    for row in scenarios:
        if not row.get("id") or not row.get("script") or not isinstance(
            row.get("expected_entities"), dict
        ):
            errors.append(f"invalid corpus scenario: {row.get('id')!r}")
    if corpus.get("retain_audio") is not False:
        errors.append("corpus retain_audio must be false")
    for name, total in (("100_calls", 100), ("200_calls", 200)):
        if (
            corpus.get("expansion_templates", {})
            .get(name, {})
            .get("total_calls")
            != total
        ):
            errors.append(f"missing reproducible {name} expansion template")

    required = compliance.get("required_fields", [])
    for cell in CELLS:
        values = compliance.get("cells", {}).get(cell)
        if not isinstance(values, dict):
            errors.append(f"missing compliance worksheet for {cell}")
            continue
        missing = set(required) - set(values)
        if missing:
            errors.append(f"{cell} compliance fields missing: {sorted(missing)}")
        if any(value in {"", None} for value in values.values()):
            errors.append(f"{cell} compliance unknowns must be 'unverified'")
    return errors


def prepare_run(
    run_dir: Path,
    *,
    calls_per_cell: int = 25,
    run_id: str | None = None,
) -> dict[str, Any]:
    errors = validate_specs()
    if errors:
        raise ValueError("; ".join(errors))
    if calls_per_cell < 1:
        raise ValueError("calls_per_cell must be positive")
    corpus = _load_json(BENCHMARK_DIR / "corpus.json")
    scenarios = [row["id"] for row in corpus["scenarios"]]
    run_id = run_id or uuid.uuid4().hex
    registrations = []
    for cell in CELLS:
        for index in range(calls_per_cell):
            registrations.append(
                {
                    "registration_id": f"{cell}-{index + 1:03d}",
                    "cell": cell,
                    "scenario_id": scenarios[index % len(scenarios)],
                    "status": "planned",
                    "destination": None,
                    "chargeable_call_placed": False,
                }
            )
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": _now(),
        "retain_audio": False,
        "calls_per_cell": calls_per_cell,
        "total_calls": len(registrations),
        "cells": list(CELLS),
        "registrations": registrations,
        "safety": {
            "places_calls": False,
            "execute_requires_exact_destination_confirmation": True,
            "external_operator_required": True,
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


def register_cell(
    run_dir: Path,
    *,
    cell: str,
    destination: str | None,
    execute: bool = False,
    confirm_destination: str | None = None,
) -> dict[str, Any]:
    """Register or authorize planned rows without invoking any provider."""
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell!r}")
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if execute:
        if not destination:
            raise ValueError("--execute requires --destination")
        if confirm_destination != destination:
            raise ValueError(
                "--confirm-destination must exactly match --destination"
            )
    changed = 0
    for row in manifest["registrations"]:
        if row["cell"] != cell:
            continue
        row["destination"] = destination
        row["status"] = (
            "approved_for_external_execution" if execute else "dry_run_registered"
        )
        row["chargeable_call_placed"] = False
        changed += 1
    if not changed:
        raise ValueError(f"manifest has no registrations for {cell}")
    manifest["updated_at"] = _now()
    _write_json(manifest_path, manifest)
    return {
        "cell": cell,
        "registered": changed,
        "status": (
            "approved_for_external_execution" if execute else "dry_run_registered"
        ),
        "chargeable_call_placed": False,
        "message": "No carrier API was called. An external operator must place approved calls.",
    }


def _records_from_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    payload = _load_json(path)
    if isinstance(payload, list):
        return payload
    for key in ("records", "calls", "rows", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("JSON import must be an array or contain records/calls/rows/data")


def _coerce(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    if text.lower() in {"true", "yes"}:
        return True
    if text.lower() in {"false", "no"}:
        return False
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text


def import_evidence(
    run_dir: Path,
    source_path: Path,
    *,
    cell: str,
    source: str,
) -> int:
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell!r}")
    if source == "managed_export" and cell not in MANAGED_CELLS:
        raise ValueError("managed exports may only be imported into managed cells")
    records = _records_from_file(source_path)
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / f"{cell.replace('+', '_')}-{source}.jsonl"
    with target.open("a", encoding="utf-8") as handle:
        for raw in records:
            unknown = set(raw) - CANONICAL_EVIDENCE_FIELDS
            if unknown:
                raise ValueError(
                    f"unsupported evidence columns: {sorted(unknown)}; map the export first"
                )
            row = {key: _coerce(value) for key, value in raw.items()}
            row["cell"] = cell
            row["source"] = source
            if not row.get("call_id"):
                raise ValueError("every evidence row requires call_id")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(records)


def import_custom_sqlite(run_dir: Path, database: Path, *, cell: str) -> int:
    """Import available custom telemetry without claiming unavailable metrics."""
    if cell not in CUSTOM_CELLS:
        raise ValueError("SQLite telemetry may only be imported into custom cells")
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        sessions = connection.execute(
            "SELECT id, status FROM sessions ORDER BY id"
        ).fetchall()
        events = connection.execute(
            "SELECT session_id, request_id, provider, event_type, latency_ms "
            "FROM provider_events WHERE session_id IS NOT NULL "
            "AND request_id IS NOT NULL ORDER BY id"
        ).fetchall()
        costs = {
            row["session_id"]: row["cost"]
            for row in connection.execute(
                "SELECT session_id, SUM(estimated_provider_cost) AS cost "
                "FROM messages GROUP BY session_id"
            ).fetchall()
        }
    finally:
        connection.close()

    turns: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    session_carriers: dict[str, set[str]] = defaultdict(set)
    for event in events:
        if event["latency_ms"] is not None:
            turns[(event["session_id"], event["request_id"])][
                event["event_type"]
            ] = int(event["latency_ms"])
        if event["event_type"] in {
            "first_outbound_audio",
            "playback_mark",
            "interruption_clear",
        } and event["provider"] in {"twilio", "exotel"}:
            session_carriers[event["session_id"]].add(event["provider"])
    by_session: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"latency": [], "barge": []}
    )
    for (session_id, _), phases in turns.items():
        start = phases.get("utterance_end", phases.get("transcript_received"))
        outbound = phases.get("first_outbound_audio")
        if start is not None and outbound is not None:
            by_session[session_id]["latency"].append(max(0, outbound - start))
        if "interruption_clear_delay" in phases:
            by_session[session_id]["barge"].append(
                phases["interruption_clear_delay"]
            )

    records = []
    target_carrier = cell.split("+", 1)[1]
    for session in sessions:
        session_id = session["id"]
        if target_carrier not in session_carriers.get(session_id, set()):
            continue
        row: dict[str, Any] = {
            "cell": cell,
            "call_id": session_id,
            "completed": session["status"] == "completed",
            "cost_inr": float(costs.get(session_id) or 0),
            "source": "custom_sqlite",
        }
        if by_session[session_id]["latency"]:
            row["speech_end_to_first_audio_ms_samples"] = by_session[session_id][
                "latency"
            ]
        if by_session[session_id]["barge"]:
            row["barge_in_to_silence_ms_samples"] = by_session[session_id][
                "barge"
            ]
        records.append(row)

    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / f"{cell.replace('+', '_')}-custom_sqlite.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(records)


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _evidence_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence_dir = run_dir / "evidence"
    if not evidence_dir.exists():
        return rows
    for path in sorted(evidence_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _merge_calls(rows: list[dict[str, Any]], cell: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("cell") != cell:
            continue
        key = str(row.get("call_id") or f"row-{index}")
        target = merged.setdefault(key, {"call_id": key, "cell": cell})
        for name, value in row.items():
            if value is not None:
                if name.endswith("_samples"):
                    target.setdefault(name, []).extend(value)
                else:
                    target[name] = value
    return list(merged.values())


def _numeric_samples(calls: list[dict], scalar: str, samples: str) -> list[float]:
    values: list[float] = []
    for call in calls:
        if call.get(scalar) is not None:
            values.append(float(call[scalar]))
        values.extend(float(value) for value in call.get(samples, []))
    return values


def _ratio(calls: list[dict], numerator: str, denominator: str) -> dict[str, Any]:
    num = sum(float(call.get(numerator) or 0) for call in calls)
    den = sum(float(call.get(denominator) or 0) for call in calls)
    return {"value": (num / den if den else None), "numerator": num, "denominator": den}


def _residency_status(worksheet: dict, cell: str) -> dict[str, Any]:
    required = worksheet.get("required_fields", [])
    values = worksheet.get("cells", {}).get(cell, {})
    missing = [
        field
        for field in required
        if values.get(field) in {None, "", "unverified"}
    ]
    return {
        "status": "verified" if not missing else "unverified",
        "unverified_fields": missing,
    }


def summarize_cell(
    rows: list[dict[str, Any]],
    cell: str,
    worksheet: dict,
) -> dict[str, Any]:
    calls = _merge_calls(rows, cell)
    latency = _numeric_samples(
        calls,
        "speech_end_to_first_audio_ms",
        "speech_end_to_first_audio_ms_samples",
    )
    barge = _numeric_samples(
        calls, "barge_in_to_silence_ms", "barge_in_to_silence_ms_samples"
    )
    false_endpoint = _ratio(calls, "false_endpoints", "endpoint_opportunities")
    switch_values = []
    for call in calls:
        if call.get("language_switch_clauses") is not None:
            switch_values.append(
                float(call["language_switch_clauses"])
                <= PLAN_GATES["language_switch_max_clauses"]
            )
        elif call.get("language_switch_success") is not None:
            switch_values.append(bool(call["language_switch_success"]))
    entity = _ratio(calls, "entity_correct", "entity_total")
    task_values = [
        bool(call["task_success"])
        for call in calls
        if call.get("task_success") is not None
    ]
    gujarati = [
        float(call["gujarati_native_speaker_rating"])
        for call in calls
        if call.get("gujarati_native_speaker_rating") is not None
    ]
    completed_ids = {
        call["call_id"] for call in calls if call.get("completed") is True
    }
    total_cost = sum(float(call.get("cost_inr") or 0) for call in calls)
    residency = _residency_status(worksheet, cell)
    metrics = {
        "speech_end_to_first_audio": {
            "p50_ms": percentile(latency, 0.50),
            "p95_ms": percentile(latency, 0.95),
            "n": len(latency),
        },
        "barge_in_to_silence": {
            "p95_ms": percentile(barge, 0.95),
            "n": len(barge),
        },
        "false_endpoint_rate": false_endpoint,
        "language_switch_within_one_clause": {
            "value": (
                sum(switch_values) / len(switch_values) if switch_values else None
            ),
            "n": len(switch_values),
        },
        "entity_accuracy": entity,
        "task_accuracy": {
            "value": (
                sum(task_values) / len(task_values) if task_values else None
            ),
            "n": len(task_values),
        },
        "gujarati_native_speaker_rating": {
            "mean": sum(gujarati) / len(gujarati) if gujarati else None,
            "n": len(gujarati),
        },
        "cost_per_completed_call_inr": {
            "value": (
                total_cost / len(completed_ids) if completed_ids else None
            ),
            "completed_calls": len(completed_ids),
            "total_cost_inr": total_cost,
        },
        "residency": residency,
    }
    missing = []
    required_paths = {
        "speech_end_to_first_audio": metrics["speech_end_to_first_audio"]["n"],
        "barge_in_to_silence": metrics["barge_in_to_silence"]["n"],
        "false_endpoint_rate": false_endpoint["denominator"],
        "language_switch": metrics["language_switch_within_one_clause"]["n"],
        "entity_accuracy": entity["denominator"],
        "task_accuracy": metrics["task_accuracy"]["n"],
        "gujarati_native_speaker_rating": metrics[
            "gujarati_native_speaker_rating"
        ]["n"],
        "cost_per_completed_call": metrics["cost_per_completed_call_inr"][
            "completed_calls"
        ],
    }
    missing.extend(name for name, count in required_paths.items() if not count)
    if residency["status"] != "verified":
        missing.append("residency")

    exact_gates = {
        "latency_p50": (
            metrics["speech_end_to_first_audio"]["p50_ms"] is not None
            and metrics["speech_end_to_first_audio"]["p50_ms"]
            <= PLAN_GATES["speech_end_to_first_audio_p50_ms_max"]
        ),
        "latency_p95": (
            metrics["speech_end_to_first_audio"]["p95_ms"] is not None
            and metrics["speech_end_to_first_audio"]["p95_ms"]
            <= PLAN_GATES["speech_end_to_first_audio_p95_ms_max"]
        ),
        "barge_in_p95": (
            metrics["barge_in_to_silence"]["p95_ms"] is not None
            and metrics["barge_in_to_silence"]["p95_ms"]
            <= PLAN_GATES["barge_in_to_silence_p95_ms_max"]
        ),
        "false_endpoint_rate": (
            false_endpoint["value"] is not None
            and false_endpoint["value"]
            < PLAN_GATES["false_endpoint_rate_max_exclusive"]
        ),
        "language_switch": (
            metrics["language_switch_within_one_clause"]["value"] == 1.0
        ),
    }
    if missing:
        gate_status = "insufficient_evidence"
    elif all(exact_gates.values()):
        gate_status = "passed"
    else:
        gate_status = "failed"
    return {
        "cell": cell,
        "calls_with_evidence": len(calls),
        "metrics": metrics,
        "exact_plan_gates": exact_gates,
        "missing_evidence": sorted(missing),
        "gate_status": gate_status,
    }


def build_report(
    run_dir: Path,
    *,
    compliance_path: Path = BENCHMARK_DIR / "compliance_residency.json",
) -> dict[str, Any]:
    rows = _evidence_rows(run_dir)
    worksheet = _load_json(compliance_path)
    cells = [summarize_cell(rows, cell, worksheet) for cell in CELLS]
    candidates = [cell for cell in cells if cell["gate_status"] == "passed"]
    if candidates:
        candidates.sort(
            key=lambda item: (
                item["metrics"]["speech_end_to_first_audio"]["p95_ms"],
                -item["metrics"]["gujarati_native_speaker_rating"]["mean"],
                -item["metrics"]["entity_accuracy"]["value"],
                -item["metrics"]["task_accuracy"]["value"],
                item["metrics"]["cost_per_completed_call_inr"]["value"],
            )
        )
        selection = {
            "status": "selected",
            "cell": candidates[0]["cell"],
            "basis": "Only cells with complete evidence and all exact plan gates passing were ranked.",
        }
    else:
        selection = {
            "status": "insufficient_evidence",
            "cell": None,
            "basis": "No cell has complete real evidence and all exact plan gates passing.",
        }
    report = {
        "schema_version": "1.0",
        "generated_at": _now(),
        "run_id": _load_json(run_dir / "manifest.json").get("run_id"),
        "retain_audio": False,
        "plan_gates": PLAN_GATES,
        "cells": cells,
        "selection": selection,
    }
    _write_json(run_dir / "report.json", report)
    csv_path = run_dir / "report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "cell",
            "gate_status",
            "latency_p50_ms",
            "latency_p95_ms",
            "barge_in_p95_ms",
            "false_endpoint_rate",
            "language_switch_rate",
            "entity_accuracy",
            "task_accuracy",
            "gujarati_rating",
            "cost_per_completed_call_inr",
            "residency_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            metrics = cell["metrics"]
            writer.writerow(
                {
                    "cell": cell["cell"],
                    "gate_status": cell["gate_status"],
                    "latency_p50_ms": metrics["speech_end_to_first_audio"][
                        "p50_ms"
                    ],
                    "latency_p95_ms": metrics["speech_end_to_first_audio"][
                        "p95_ms"
                    ],
                    "barge_in_p95_ms": metrics["barge_in_to_silence"]["p95_ms"],
                    "false_endpoint_rate": metrics["false_endpoint_rate"]["value"],
                    "language_switch_rate": metrics[
                        "language_switch_within_one_clause"
                    ]["value"],
                    "entity_accuracy": metrics["entity_accuracy"]["value"],
                    "task_accuracy": metrics["task_accuracy"]["value"],
                    "gujarati_rating": metrics[
                        "gujarati_native_speaker_rating"
                    ]["mean"],
                    "cost_per_completed_call_inr": metrics[
                        "cost_per_completed_call_inr"
                    ]["value"],
                    "residency_status": metrics["residency"]["status"],
                }
            )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--calls-per-cell", type=int, default=25)
    prepare.add_argument("--run-id")

    register = subparsers.add_parser("register")
    register.add_argument("--run-dir", type=Path, required=True)
    register.add_argument("--cell", choices=CELLS, required=True)
    register.add_argument("--destination")
    register.add_argument("--execute", action="store_true")
    register.add_argument("--confirm-destination")

    managed = subparsers.add_parser("import-managed")
    managed.add_argument("--run-dir", type=Path, required=True)
    managed.add_argument("--cell", choices=sorted(MANAGED_CELLS), required=True)
    managed.add_argument("--input", type=Path, required=True)

    ratings = subparsers.add_parser("import-ratings")
    ratings.add_argument("--run-dir", type=Path, required=True)
    ratings.add_argument("--cell", choices=CELLS, required=True)
    ratings.add_argument("--input", type=Path, required=True)

    custom = subparsers.add_parser("import-custom")
    custom.add_argument("--run-dir", type=Path, required=True)
    custom.add_argument("--cell", choices=sorted(CUSTOM_CELLS), required=True)
    custom.add_argument("--database", type=Path, required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument(
        "--compliance",
        type=Path,
        default=BENCHMARK_DIR / "compliance_residency.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            errors = validate_specs()
            payload = {"status": "ok" if not errors else "invalid", "errors": errors}
        elif args.command == "prepare":
            payload = prepare_run(
                args.run_dir,
                calls_per_cell=args.calls_per_cell,
                run_id=args.run_id,
            )
        elif args.command == "register":
            payload = register_cell(
                args.run_dir,
                cell=args.cell,
                destination=args.destination,
                execute=args.execute,
                confirm_destination=args.confirm_destination,
            )
        elif args.command == "import-managed":
            payload = {
                "imported": import_evidence(
                    args.run_dir,
                    args.input,
                    cell=args.cell,
                    source="managed_export",
                )
            }
        elif args.command == "import-ratings":
            payload = {
                "imported": import_evidence(
                    args.run_dir,
                    args.input,
                    cell=args.cell,
                    source="manual_rating",
                )
            }
        elif args.command == "import-custom":
            payload = {
                "imported": import_custom_sqlite(
                    args.run_dir, args.database, cell=args.cell
                )
            }
        else:
            payload = build_report(args.run_dir, compliance_path=args.compliance)
    except (OSError, ValueError, json.JSONDecodeError, sqlite3.DatabaseError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("status") != "invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
