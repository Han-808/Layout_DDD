#!/usr/bin/env python3
"""Select the first publishable scene-evaluation attempt for every case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.camera_cal_scene_level.persisted_scoring import (  # noqa: E402
    case_scoring_summary,
    run_scoring_aggregate,
)


DEFAULT_CASE_IDS = tuple(f"S{index:03d}" for index in range(100, 110))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-label")
    parser.add_argument("--provider-route", default="ForgeAX API1")
    args = parser.parse_args()

    attempt_roots = tuple(path.expanduser().resolve() for path in args.attempt_root)
    case_ids = tuple(dict.fromkeys(args.case_id or DEFAULT_CASE_IDS))
    selections, pending = first_publishable_attempts(
        attempt_roots=attempt_roots,
        case_ids=case_ids,
    )
    if args.pending_only:
        for case_id in pending:
            print(case_id)
        return
    if pending:
        raise SystemExit(f"nonpublishable cases remain: {','.join(pending)}")
    if args.output_root is None or not args.model_label:
        raise SystemExit("--output-root and --model-label are required for selection")
    result = write_selection(
        output_root=args.output_root.expanduser().resolve(),
        model_label=str(args.model_label),
        provider_route=str(args.provider_route),
        attempt_roots=attempt_roots,
        case_ids=case_ids,
        selections=selections,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def first_publishable_attempts(
    *,
    attempt_roots: tuple[Path, ...],
    case_ids: tuple[str, ...],
) -> tuple[dict[str, Path], list[str]]:
    selections: dict[str, Path] = {}
    pending: list[str] = []
    for case_id in case_ids:
        selected = None
        for attempt_root in attempt_roots:
            case_dir = attempt_root / "cases" / case_id
            if _is_publishable(case_dir):
                selected = case_dir
                break
        if selected is None:
            pending.append(case_id)
        else:
            selections[case_id] = selected
    return selections, pending


def write_selection(
    *,
    output_root: Path,
    model_label: str,
    provider_route: str,
    attempt_roots: tuple[Path, ...],
    case_ids: tuple[str, ...],
    selections: dict[str, Path],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    building = output_root.parent / f".{output_root.name}.building"
    if building.exists():
        raise FileExistsError(f"stale build directory requires review: {building}")
    if set(selections) != set(case_ids):
        raise ValueError("selection must contain every expected case")

    attempt_indices = {root: index for index, root in enumerate(attempt_roots)}
    selected_summaries: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    building.mkdir(parents=True)
    (building / "cases").mkdir()
    try:
        for case_id in case_ids:
            source_case = selections[case_id]
            source_run = source_case.parent.parent
            report = _read_json(source_case / "evaluation_report.json")
            manifest = _read_json(source_case / "case_run_manifest.json")
            coverage = report.get("coverage")
            coverage = coverage if isinstance(coverage, dict) else {}
            target = building / "cases" / case_id
            os.symlink(source_case, target, target_is_directory=True)
            selected_summaries.append(_case_summary(source_case))
            case_records.append(
                {
                    "case_id": case_id,
                    "selected_attempt_index": attempt_indices[source_run],
                    "source_run": str(source_run),
                    "source_case": str(source_case),
                    "storage": "absolute_directory_symlink",
                    "status": manifest.get("status"),
                    "final_decision_status": manifest.get("final_decision_status"),
                    "benchmark_score_100": report.get("benchmark_score_100"),
                    "benchmark_score_status": report.get("benchmark_score_status"),
                    "evaluation_status": report.get("evaluation_status"),
                    "grounded_score_fraction": coverage.get("grounded_score_fraction"),
                    "l1_engineering_failure": manifest.get("l1_engineering_failure"),
                    "case_manifest_sha256": _sha256(
                        source_case / "case_run_manifest.json"
                    ),
                    "evaluation_report_sha256": _sha256(
                        source_case / "evaluation_report.json"
                    ),
                    "l1_report_sha256": _sha256(source_case / "l1_report.json"),
                    "l3_report_sha256": _sha256(
                        source_case / "scene_quality_report.json"
                    ),
                }
            )

        aggregate = run_scoring_aggregate(selected_summaries)
        if aggregate.get("official_score_100") is None:
            raise ValueError("selected full set has no official score")
        if int(aggregate.get("infrastructure_failure_case_count", -1)) != 0:
            raise ValueError("selected full set still has infrastructure failures")

        retry_case_count = sum(
            1 for row in case_records if row["selected_attempt_index"] > 0
        )
        selection_manifest = {
            "schema_version": "scene_level_first_publishable_selection_v1",
            "status": "complete",
            "model_label": model_label,
            "evaluator_model": "gpt-5.6-sol",
            "provider_route": provider_route,
            "case_count": len(case_records),
            "attempt_roots": [str(root) for root in attempt_roots],
            "selection_policy": "first_publishable_attempt_only_no_score_selection",
            "publishability_policy": {
                "case_status": "complete",
                "final_decision_status": "resolved",
                "l1_engineering_failure": False,
                "evaluation_status": "complete",
                "benchmark_score_status": "complete",
                "benchmark_score_100": "finite_number",
            },
            "cases": case_records,
        }
        summary = {
            "schema_version": "selected_scene_level_summary_v1",
            "status": "complete",
            "model_label": model_label,
            "evaluator_model": "gpt-5.6-sol",
            "provider_route": provider_route,
            "totals": {
                "cases": len(case_records),
                "successful": len(case_records),
                "failed": 0,
                "final_unresolved": 0,
                "final_infrastructure_failure": 0,
                "l1_engineering_failure_cases": 0,
                "retry_cases": retry_case_count,
                "baseline_cases": len(case_records) - retry_case_count,
                "attempt_rounds": len(attempt_roots),
            },
            "aggregate": aggregate,
        }
        _write_json(building / "selection_manifest.json", selection_manifest)
        _write_json(building / "run_manifest.json", selection_manifest)
        _write_json(building / "summary.json", summary)
        building.replace(output_root)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise

    return {
        "output_root": str(output_root),
        "case_count": len(case_records),
        "retry_case_count": retry_case_count,
        "attempt_rounds": len(attempt_roots),
        "official_score_100": aggregate["official_score_100"],
        "mean_coverage_fraction": aggregate["mean_combined_coverage_fraction"],
    }


def _is_publishable(case_dir: Path) -> bool:
    manifest_path = case_dir / "case_run_manifest.json"
    report_path = case_dir / "evaluation_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        report = _read_json(report_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    score = report.get("benchmark_score_100")
    return (
        manifest.get("status") == "complete"
        and manifest.get("final_decision_status") == "resolved"
        and manifest.get("l1_engineering_failure") is False
        and report.get("evaluation_status") == "complete"
        and report.get("benchmark_score_status") == "complete"
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
    )


def _case_summary(case_dir: Path) -> dict[str, Any]:
    return case_scoring_summary(
        case_id=case_dir.name,
        case_manifest=_read_json(case_dir / "case_run_manifest.json"),
        l1_report=_read_json(case_dir / "l1_report.json"),
        l3_report=_read_json(case_dir / "scene_quality_report.json"),
        l1_diagnostics=(
            _read_json(case_dir / "l1_diagnostics.json")
            if (case_dir / "l1_diagnostics.json").is_file()
            else {}
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
