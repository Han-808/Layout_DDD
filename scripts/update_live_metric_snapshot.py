#!/usr/bin/env python3
"""Overwrite one full metric-wise snapshot for a running scene-level run."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


METRICS = (
    "scale_consistency",
    "style_consistency",
    "object_pairing_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--snapshot-path", type=Path, default=None)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    snapshot_path = (
        args.snapshot_path.expanduser().resolve()
        if args.snapshot_path is not None
        else output_root / "live_metric_snapshot.json"
    )
    payload = build_snapshot(output_root)
    atomic_write_json(snapshot_path, payload)
    print(
        json.dumps(
            {
                "snapshot_path": str(snapshot_path),
                "generated_at": payload["generated_at"],
                "completed_cases": payload["counts"]["completed"],
                "running_cases": payload["counts"]["running"],
                "resolved_cases": payload["counts"]["resolved"],
                "unresolved_cases": payload["counts"]["unresolved"],
                "infrastructure_failure_cases": payload["counts"][
                    "infrastructure_failure"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_snapshot(output_root: Path) -> dict[str, Any]:
    run_manifest = read_json_if_present(output_root / "run_manifest.json")
    experiment = read_json_if_present(output_root / "experiment_plan.json")
    requested_ids = [
        str(item["case_id"])
        for item in experiment.get("cases", [])
        if isinstance(item, dict) and item.get("case_id")
    ]
    case_root = output_root / "cases"
    discovered_ids = sorted(
        (path.name for path in case_root.glob("[NS][0-9][0-9][0-9]") if path.is_dir()),
        key=lambda value: (value[0], int(value[1:])),
    )
    case_ids = requested_ids or discovered_ids
    cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        root = case_root / case_id
        manifest = read_json_if_present(root / "case_run_manifest.json")
        status = str(manifest.get("status") or "pending")
        record: dict[str, Any] = {
            "case_id": case_id,
            "status": status,
            "started_at": manifest.get("started_at"),
            "completed_at": manifest.get("completed_at"),
            "api_usage": deepcopy(manifest.get("api_usage") or {}),
            "final_result": {
                "final_decision_status": manifest.get("final_decision_status"),
                "l1_decision_status": manifest.get("l1_decision_status"),
                "l3_decision_status": manifest.get("l3_decision_status"),
                "l3_unresolved_metrics": deepcopy(
                    manifest.get("l3_unresolved_metrics") or []
                ),
                "l3_infrastructure_failure_metrics": deepcopy(
                    manifest.get("l3_infrastructure_failure_metrics") or []
                ),
            },
            "metrics": {},
        }
        report = read_json_if_present(root / "scene_quality_report.json")
        comparison = read_json_if_present(root / "scene_comparison.json")
        report_metrics = report.get("metrics")
        report_metrics = report_metrics if isinstance(report_metrics, dict) else {}
        comparison_metrics = comparison.get("metrics")
        comparison_metrics = (
            comparison_metrics if isinstance(comparison_metrics, dict) else {}
        )
        for metric in METRICS:
            metric_report = report_metrics.get(metric)
            metric_report = metric_report if isinstance(metric_report, dict) else {}
            metric_comparison = comparison_metrics.get(metric)
            metric_comparison = (
                metric_comparison if isinstance(metric_comparison, dict) else {}
            )
            model = metric_comparison.get("model")
            model = model if isinstance(model, dict) else {}
            judgement = metric_report.get("judgement")
            judgement = judgement if isinstance(judgement, dict) else {}
            record["metrics"][metric] = {
                "status": metric_report.get("status") or model.get("status"),
                "verdict": model.get("prediction") or judgement.get("verdict"),
                "score": metric_report.get("score", model.get("score")),
                "coverage": deepcopy(metric_report.get("coverage")),
                "reason": judgement.get("reason") or model.get("reason"),
                "defects": collect_defects(metric_report),
                "anomaly_object_ids": deepcopy(
                    model.get("anomaly_object_ids") or []
                ),
                "visual_evidence_paths": collect_evidence_paths(metric_report),
                "judge_call_count": metric_report.get(
                    "judge_call_count", model.get("judge_call_count")
                ),
            }
        cases.append(record)

    counts = {
        "requested": len(case_ids),
        "completed": sum(item["status"] == "complete" for item in cases),
        "running": sum(item["status"] == "running" for item in cases),
        "pending": sum(item["status"] == "pending" for item in cases),
        "resolved": sum(
            item["final_result"]["final_decision_status"] == "resolved"
            for item in cases
        ),
        "unresolved": sum(
            item["final_result"]["final_decision_status"] == "unresolved"
            for item in cases
        ),
        "infrastructure_failure": sum(
            item["final_result"]["final_decision_status"]
            == "infrastructure_failure"
            for item in cases
        ),
    }
    return {
        "schema_version": "live_metric_snapshot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overwrite_policy": "replace_previous_snapshot_atomically",
        "scope": "all_completed_case_reports_available_at_generation_time",
        "in_progress_case_policy": (
            "record progress only; complete metric verdicts are published when "
            "the case report is atomically finalized"
        ),
        "output_root": str(output_root),
        "run_status": run_manifest.get("status"),
        "counts": counts,
        "cases": cases,
    }


def collect_defects(value: Any) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            raw = node.get("defects")
            if isinstance(raw, list):
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
                    if encoded not in seen:
                        seen.add(encoded)
                        defects.append(deepcopy(item))
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return defects


def collect_evidence_paths(value: Any) -> list[str]:
    paths: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in {"evidence_paths", "images_used", "visual_evidence"}:
                    if isinstance(child, list):
                        paths.update(
                            str(item)
                            for item in child
                            if isinstance(item, str) and item
                        )
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return sorted(paths)


def read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
