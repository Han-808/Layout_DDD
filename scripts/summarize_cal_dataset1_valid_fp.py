#!/usr/bin/env python3
"""Compute false-positive rates for the source-valid cal_dataset1 scenes.

Every selected GT event is treated as valid, as requested.  No event is
excluded because of conservative geometry warnings.  Unresolved outcomes are
reported separately and never counted as valid.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("collision", "oob", "support")
SCHEMA_VERSION = "cal_dataset1_source_valid_fp_summary_v1"


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    case_ids = args.case_id or [f"source_valid_{index:03d}" for index in range(1, 11)]
    rows, missing = _collect_rows(dataset_root, run_root, case_ids)
    if missing and not args.allow_partial:
        raise RuntimeError(
            "valid FP run is incomplete: " + "; ".join(missing)
        )
    if not rows:
        raise RuntimeError("no completed source-valid event outcomes found")
    summary = _summary_rows(rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "cal_dataset1",
        "experiment_type": "source_valid_false_positive_rate",
        "validity_assumption": (
            "All source_valid events are counted as valid; conservative geometry "
            "warnings are not exclusions."
        ),
        "denominator_contract": {
            "fp_rate_all": "predicted_invalid / all selected valid GT events",
            "fp_rate_resolved": "predicted_invalid / resolved valid GT events",
            "unresolved": "reported separately and never imputed as valid",
        },
        "case_ids": case_ids,
        "event_count": len(rows),
        "missing": missing,
        "summary": summary,
        "outputs": {
            "per_event": str((out_dir / "per_event.tsv").resolve()),
            "summary": str((out_dir / "summary.tsv").resolve()),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(out_dir / "per_event.tsv", rows)
    _write_tsv(out_dir / "summary.tsv", summary)
    _write_json(out_dir / "summary.json", payload)
    print(json.dumps({
        "case_count": len({row["case_id"] for row in rows}),
        "event_count": len(rows),
        "missing_count": len(missing),
        "summary": str((out_dir / "summary.tsv").resolve()),
        "per_event": str((out_dir / "per_event.tsv").resolve()),
    }, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "Support" / "datasets" / "cal_dataset1"),
    )
    parser.add_argument(
        "--run-root",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp2_valid_fp_gpt56"
        ),
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--allow-partial",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if not args.out_dir:
        args.out_dir = str(Path(args.run_root) / "fp_analysis")
    return args


def _collect_rows(
    dataset_root: Path,
    run_root: Path,
    case_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id in case_ids:
        fixture = dataset_root / "fixtures" / case_id
        gt_path = fixture / "event_gt.json"
        report_path = run_root / "deterministic" / case_id / "evaluation_report.json"
        status_path = report_path.with_name("case_status.json")
        if not gt_path.is_file():
            missing.append(f"{case_id}: missing GT")
            continue
        if not report_path.is_file() or not status_path.is_file():
            missing.append(f"{case_id}: missing evaluation report/status")
            continue
        status = _read_json(status_path)
        if status.get("status") != "completed":
            missing.append(f"{case_id}: status={status.get('status')}")
            continue
        gt = _read_json(gt_path)
        report = _read_json(report_path)
        outcomes = _outcome_index(report)
        events = [
            event for event in gt.get("events") or []
            if isinstance(event, dict) and str(event.get("metric")) in METRICS
        ]
        for event in events:
            semantic = str(event.get("semantic_label"))
            if semantic != "valid":
                raise RuntimeError(
                    f"{case_id} {event.get('metric')}:{event.get('event_id')} "
                    f"is not valid GT: {semantic!r}"
                )
            key = (str(event["metric"]), str(event["event_id"]))
            record = outcomes.get(key)
            if record is None:
                raise RuntimeError(f"report lacks GT event {case_id} {key}")
            verdict = record.get("final_verdict")
            resolved = verdict in {"valid", "invalid"}
            rows.append({
                "case_id": case_id,
                "metric": key[0],
                "event_id": key[1],
                "gt_label": "valid",
                "predicted_label": verdict if resolved else "unresolved",
                "resolved": int(resolved),
                "false_positive": int(verdict == "invalid"),
                "route": record.get("route"),
                "requires_vlm": int(bool(record.get("requires_vlm"))),
                "vlm_adjudicated": int(record.get("route") == "vlm_adjudicated"),
                "adjudication_error": record.get("adjudication_error"),
                "geometry_warning_excluded": 0,
            })
    rows.sort(key=lambda row: (
        str(row["metric"]),
        str(row["case_id"]),
        str(row["event_id"]),
    ))
    return rows, missing


def _outcome_index(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    metrics = report.get("metrics") or {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in (metrics.get("collision") or {}).get("pairs") or []:
        if not isinstance(record, dict):
            continue
        event_id = "|".join(sorted([
            str(record.get("object_a")),
            str(record.get("object_b")),
        ]))
        index[("collision", event_id)] = record
    for metric in ("oob", "support"):
        for record in (metrics.get(metric) or {}).get("objects") or []:
            if isinstance(record, dict):
                index[(metric, str(record.get("object_id")))] = record
    return index


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("overall", "overall", rows)
    ]
    grouped.extend(
        ("metric", metric, [row for row in rows if row["metric"] == metric])
        for metric in METRICS
    )
    routes = sorted({str(row.get("route") or "unresolved") for row in rows})
    grouped.extend(
        (
            "route",
            route,
            [
                row for row in rows
                if str(row.get("route") or "unresolved") == route
            ],
        )
        for route in routes
    )
    summary: list[dict[str, Any]] = []
    for group_type, group, selected in grouped:
        if not selected:
            continue
        resolved = [row for row in selected if row["resolved"]]
        false_positives = [row for row in selected if row["false_positive"]]
        vlm = [row for row in selected if row["vlm_adjudicated"]]
        vlm_resolved = [row for row in vlm if row["resolved"]]
        vlm_fp = [row for row in vlm if row["false_positive"]]
        summary.append({
            "group_type": group_type,
            "group": group,
            "valid_gt_total": len(selected),
            "resolved": len(resolved),
            "unresolved": len(selected) - len(resolved),
            "predicted_valid": sum(
                row["predicted_label"] == "valid" for row in selected
            ),
            "false_positives": len(false_positives),
            "fp_rate_all": len(false_positives) / len(selected),
            "fp_rate_resolved": (
                len(false_positives) / len(resolved) if resolved else None
            ),
            "specificity_all": (
                sum(row["predicted_label"] == "valid" for row in selected)
                / len(selected)
            ),
            "specificity_resolved": (
                sum(row["predicted_label"] == "valid" for row in resolved)
                / len(resolved)
                if resolved else None
            ),
            "vlm_adjudicated": len(vlm),
            "vlm_resolved": len(vlm_resolved),
            "vlm_false_positives": len(vlm_fp),
            "vlm_fp_rate_all": (
                len(vlm_fp) / len(vlm) if vlm else None
            ),
            "vlm_fp_rate_resolved": (
                len(vlm_fp) / len(vlm_resolved) if vlm_resolved else None
            ),
        })
    return summary


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
