#!/usr/bin/env python3
"""Reproject frozen Scene Quality findings without API or renderer access.

The command is intentionally read-only with respect to the source run.  It
reuses persisted Judge findings and recomputes only the deterministic scoring
projection and layer aggregation.  Discovery, evidence, verdicts, severities,
and defects are never invented or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from benchmark.evaluator.scoring import (
    DEFAULT_DEDUCTION_MULTIPLIER,
    DEDUCTION_MULTIPLIER_METRICS,
    L1_METRIC_WEIGHTS,
    L3_METRIC_WEIGHTS,
    MIN_PUBLISHABLE_SCORE_COVERAGE,
    project_incomplete_metric_coverage,
    project_metric_events,
)
from benchmark.scoring_profiles import SCORING_SPEC_VERSION


POSTHOC_SCHEMA_VERSION = "scene_quality_posthoc_rescore_v1"
METRICS = tuple(L3_METRIC_WEIGHTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute frozen metric events under the current scoring "
            "projection without acquiring evidence or calling a model."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--deduction-multiplier",
        type=_positive_float,
        default=DEFAULT_DEDUCTION_MULTIPLIER,
        help=(
            "Multiply deductions for Collision, Support, OOB, Scale, Style, "
            "and Object Pairing (default: 2.0)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = rescore_run(
        input_root=args.input_root,
        output_root=args.output_root,
        deduction_multiplier=args.deduction_multiplier,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "case_count": report["case_count"],
                "input_root": report["input_root"],
                "output_root": str(args.output_root.resolve()),
                "comparison_path": str(
                    (args.output_root / "posthoc_rescore_comparison.json").resolve()
                ),
            },
            indent=2,
        )
    )


def rescore_run(
    *,
    input_root: Path,
    output_root: Path,
    deduction_multiplier: float = DEFAULT_DEDUCTION_MULTIPLIER,
) -> dict[str, Any]:
    resolved_multiplier = _positive_finite(
        deduction_multiplier,
        "deduction_multiplier",
    )
    source = input_root.resolve()
    destination = output_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"input run does not exist: {source}")
    if source == destination or source in destination.parents:
        raise ValueError(
            "output-root must be outside the source run so frozen outputs "
            "remain untouched"
        )
    case_paths = sorted((source / "cases").glob("*/scene_quality_report.json"))
    if not case_paths:
        raise ValueError("input run contains no scene_quality_report.json files")

    plan = _read_object(source / "experiment_plan.json", required=False)
    evaluator_model = str(
        ((plan.get("model_route") or {}).get("model")) or "unknown"
    )
    cases: list[dict[str, Any]] = []
    for report_path in case_paths:
        cases.append(
            _rescore_case(
                report_path,
                evaluator_model=evaluator_model,
                deduction_multiplier=resolved_multiplier,
            )
        )

    models = _aggregate_by(cases, key="generation_model")
    overall = _aggregate_cases(cases)
    report = {
        "schema_version": POSTHOC_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(source),
        "source_run_fingerprint": _run_fingerprint(case_paths),
        "source_scoring_versions": sorted(
            {
                str(case["source_scoring_version"])
                for case in cases
            }
        ),
        "target_scoring_version": SCORING_SPEC_VERSION,
        "deduction_multiplier": resolved_multiplier,
        "deduction_multiplier_metrics": list(
            DEDUCTION_MULTIPLIER_METRICS
        ),
        "evaluator_model": evaluator_model,
        "case_count": len(cases),
        "new_evidence_acquired": False,
        "judgements_reused": True,
        "renderer_invoked": False,
        "api_calls_added": 0,
        "exact_recomputation_scope": [
            "per-object event burden aggregation",
            "metric deductions from persisted events",
            "coverage-conditioned L3 aggregation",
            "intrinsic_validity_v2 L1/L3 aggregate when persisted L1 is numeric",
        ],
        "not_recomputed": [
            "never-discovered Placement candidates",
            "never-judged Placement defects",
            "legacy excluded_function_owned rows",
            "usable-side decoding or directional verdicts",
            "relation-admission decisions",
            "evidence acquisition and grounding states",
            "severity labels or metric verdicts",
        ],
        "track_a": {
            "status": "exact_posthoc_reprojection",
            "change": (
                "Functional and Placement use the strongest persisted event "
                "burden per scoring object; cross-metric suppression is not "
                "replayed or invented."
            ),
        },
        "track_b": {
            "status": "requires_new_evaluation_run",
            "change": (
                "Recall improvements are implemented in discovery-to-Judge "
                "routing but cannot recover claims absent from frozen traces."
            ),
            "score_effect_in_this_report": 0,
        },
        "overall": overall,
        "models": models,
        "cases": cases,
    }

    destination.mkdir(parents=True, exist_ok=True)
    case_dir = destination / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        _write_json(case_dir / f"{case['case_id']}.json", case)
    _write_json(destination / "posthoc_rescore_comparison.json", report)
    return report


def _rescore_case(
    report_path: Path,
    *,
    evaluator_model: str,
    deduction_multiplier: float,
) -> dict[str, Any]:
    scene_report = _read_object(report_path)
    case_dir = report_path.parent
    case_id = case_dir.name
    run_manifest = _read_object(case_dir / "case_run_manifest.json", required=False)
    evaluation_report = _read_object(
        case_dir / "evaluation_report.json", required=False
    )
    ordered_ids = [
        str(item)
        for item in (
            (scene_report.get("scoring") or {}).get(
                "ordered_canonical_object_ids"
            )
            or []
        )
    ]
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError(f"{case_id}: missing or duplicate canonical object IDs")

    metrics: dict[str, Any] = {}
    for metric_name in METRICS:
        metric = (scene_report.get("metrics") or {}).get(metric_name)
        if not isinstance(metric, dict):
            raise ValueError(f"{case_id}: missing metric {metric_name}")
        scoring = metric.get("scoring")
        if not isinstance(scoring, dict):
            raise ValueError(f"{case_id}: missing persisted scoring for {metric_name}")
        events = scoring.get("events")
        if not isinstance(events, list):
            raise ValueError(f"{case_id}: persisted events are unavailable")
        coverage = _coverage_fraction(metric)
        new_raw = project_metric_events(
            metric_name,
            ordered_object_ids=ordered_ids,
            events=events,
            nominal_weight=float(L3_METRIC_WEIGHTS[metric_name]),
            deduction_multiplier=deduction_multiplier,
        )
        new_projected = project_incomplete_metric_coverage(
            new_raw,
            coverage_fraction=coverage,
        )
        old_raw_score = _old_raw_score(scoring, metric)
        old_burdens = scoring.get("capped_object_burdens") or {}
        new_burdens = new_raw.get("effective_object_burdens") or {}
        changed_objects = [
            object_id
            for object_id in ordered_ids
            if not math.isclose(
                float(old_burdens.get(object_id) or 0.0),
                float(new_burdens.get(object_id) or 0.0),
                abs_tol=1.0e-12,
            )
        ]
        metrics[metric_name] = {
            "status": str(metric.get("status") or "unknown"),
            "verdict": (metric.get("judgement") or {}).get("verdict"),
            "coverage_fraction": coverage,
            "event_count": len(events),
            "old": {
                "raw_score": old_raw_score,
                "burden_total": scoring.get("burden_total_b_m"),
                "object_burdens": old_burdens,
                "aggregation": "sum_then_cap_at_one",
            },
            "new": {
                "raw_score": new_raw.get("score"),
                "published_score": new_projected.get("score"),
                "burden_total": new_raw.get("burden_total_b_m"),
                "object_burdens": new_burdens,
                "aggregation": new_raw.get("object_burden_aggregation"),
                "coverage_projection": new_projected.get(
                    "coverage_projection"
                ),
            },
            "score_delta": _difference(new_raw.get("score"), old_raw_score),
            "burden_delta": _difference(
                new_raw.get("burden_total_b_m"),
                scoring.get("burden_total_b_m"),
            ),
            "changed_object_ids": changed_objects,
        }

    old_l3 = _aggregate_l3(metrics, score_branch="old")
    new_l3 = _aggregate_l3(metrics, score_branch="new")
    old_l1_score = _l1_score(evaluation_report)
    l1_reprojection = _reproject_l1(
        case_dir / "l1_report.json",
        ordered_ids=ordered_ids,
        deduction_multiplier=deduction_multiplier,
    )
    new_l1_score = (
        l1_reprojection["score"]
        if l1_reprojection["recomputed"]
        else old_l1_score
    )
    old_benchmark = _benchmark_score(
        old_l1_score,
        old_l3.get("published_score"),
    )
    new_benchmark = _benchmark_score(
        new_l1_score,
        new_l3.get("published_score"),
    )
    generation_model = _generation_model(run_manifest)
    return {
        "case_id": case_id,
        "generation_model": generation_model,
        "evaluator_model": evaluator_model,
        "source_report_path": str(report_path.resolve()),
        "source_report_sha256": _sha256(report_path),
        "source_scoring_version": str(
            next(
                (
                    ((scene_report.get("metrics") or {}).get(name) or {})
                    .get("scoring", {})
                    .get("schema_version")
                    for name in METRICS
                    if isinstance(
                        ((scene_report.get("metrics") or {}).get(name) or {}).get(
                            "scoring"
                        ),
                        dict,
                    )
                ),
                "unknown",
            )
        ),
        "n_scene": len(ordered_ids),
        "l1_score": old_l1_score,
        "old_l1_score": old_l1_score,
        "new_l1_score": new_l1_score,
        "l1_reprojection": l1_reprojection,
        "reported_benchmark_score_100": run_manifest.get(
            "benchmark_score_100"
        ),
        "old_l3": old_l3,
        "new_l3": new_l3,
        "old_benchmark_score_100": old_benchmark,
        "new_benchmark_score_100": new_benchmark,
        "benchmark_delta": _difference(new_benchmark, old_benchmark),
        "metrics": metrics,
    }


def _aggregate_l3(
    metrics: dict[str, dict[str, Any]],
    *,
    score_branch: str,
) -> dict[str, Any]:
    weighted_score = 0.0
    grounded_weight = 0.0
    for metric_name, weight in L3_METRIC_WEIGHTS.items():
        record = metrics[metric_name]
        score = record[score_branch].get("raw_score")
        if record.get("status") != "evaluated" or not _is_number(score):
            continue
        local_weight = float(weight) * float(record["coverage_fraction"])
        weighted_score += float(score) * local_weight
        grounded_weight += local_weight
    required_weight = sum(float(item) for item in L3_METRIC_WEIGHTS.values())
    coverage = grounded_weight / required_weight if required_weight else 0.0
    observed = weighted_score / grounded_weight if grounded_weight else None
    published = (
        observed
        if observed is not None
        and coverage >= MIN_PUBLISHABLE_SCORE_COVERAGE
        else None
    )
    return {
        "observed_score": observed,
        "published_score": published,
        "score_100": published * 100.0 if published is not None else None,
        "grounded_score_fraction": coverage,
        "coverage_threshold_passed": coverage
        >= MIN_PUBLISHABLE_SCORE_COVERAGE,
        "minimum_publishable_coverage": MIN_PUBLISHABLE_SCORE_COVERAGE,
    }


def _aggregate_by(
    cases: Iterable[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case.get(key) or "unknown")].append(case)
    return [
        {key: name, **_aggregate_cases(items)}
        for name, items in sorted(grouped.items())
    ]


def _aggregate_cases(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(cases)
    metric_summary: dict[str, Any] = {}
    for metric_name in METRICS:
        old_scores = [
            case["metrics"][metric_name]["old"]["raw_score"]
            for case in rows
            if _is_number(case["metrics"][metric_name]["old"]["raw_score"])
        ]
        new_scores = [
            case["metrics"][metric_name]["new"]["raw_score"]
            for case in rows
            if _is_number(case["metrics"][metric_name]["new"]["raw_score"])
        ]
        metric_summary[metric_name] = {
            "old_mean_score_100": _mean_100(old_scores),
            "new_mean_score_100": _mean_100(new_scores),
            "mean_delta_points": _difference(
                _mean_100(new_scores), _mean_100(old_scores)
            ),
            "case_count": min(len(old_scores), len(new_scores)),
        }
    old_benchmark = [
        case["old_benchmark_score_100"]
        for case in rows
        if _is_number(case.get("old_benchmark_score_100"))
    ]
    new_benchmark = [
        case["new_benchmark_score_100"]
        for case in rows
        if _is_number(case.get("new_benchmark_score_100"))
    ]
    old_l3 = [
        case["old_l3"]["score_100"]
        for case in rows
        if _is_number(case["old_l3"].get("score_100"))
    ]
    new_l3 = [
        case["new_l3"]["score_100"]
        for case in rows
        if _is_number(case["new_l3"].get("score_100"))
    ]
    return {
        "case_count": len(rows),
        "old_benchmark_score_count": len(old_benchmark),
        "new_benchmark_score_count": len(new_benchmark),
        "old_l3_score_count": len(old_l3),
        "new_l3_score_count": len(new_l3),
        "old_mean_benchmark_score_100": _mean(old_benchmark),
        "new_mean_benchmark_score_100": _mean(new_benchmark),
        "benchmark_delta_points": _difference(
            _mean(new_benchmark), _mean(old_benchmark)
        ),
        "old_mean_l3_score_100": _mean(old_l3),
        "new_mean_l3_score_100": _mean(new_l3),
        "l3_delta_points": _difference(_mean(new_l3), _mean(old_l3)),
        "metrics": metric_summary,
    }


def _coverage_fraction(metric: dict[str, Any]) -> float:
    coverage = metric.get("coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    projection = coverage.get("score_projection")
    projection = projection if isinstance(projection, dict) else {}
    grounding = coverage.get("score_grounding")
    grounding = grounding if isinstance(grounding, dict) else {}
    raw = projection.get("coverage_fraction")
    if not _is_number(raw):
        raw = grounding.get("fraction")
    if not _is_number(raw):
        raw = coverage.get("fraction")
    if not _is_number(raw):
        return 1.0
    return min(1.0, max(0.0, float(raw)))


def _old_raw_score(scoring: dict[str, Any], metric: dict[str, Any]) -> float | None:
    coverage_projection = scoring.get("coverage_projection")
    coverage_projection = (
        coverage_projection if isinstance(coverage_projection, dict) else {}
    )
    value = coverage_projection.get("raw_score_before_coverage_projection")
    if not _is_number(value):
        value = scoring.get("score")
    if not _is_number(value):
        value = metric.get("score")
    return float(value) if _is_number(value) else None


def _l1_score(evaluation_report: dict[str, Any]) -> float | None:
    layer = (evaluation_report.get("layer_reports") or {}).get(
        "l1_physical_plausibility"
    )
    layer = layer if isinstance(layer, dict) else {}
    score = layer.get("score")
    return float(score) if _is_number(score) else None


def _reproject_l1(
    report_path: Path,
    *,
    ordered_ids: list[str],
    deduction_multiplier: float,
) -> dict[str, Any]:
    report = _read_object(report_path, required=False)
    metric_reports = report.get("metrics")
    if not isinstance(metric_reports, dict):
        return {
            "recomputed": False,
            "score": None,
            "reason": "l1_report_unavailable",
            "metrics": {},
        }
    projections: dict[str, Any] = {}
    scores: list[float] = []
    for metric_name, nominal_weight in L1_METRIC_WEIGHTS.items():
        metric = metric_reports.get(metric_name)
        scoring = metric.get("scoring") if isinstance(metric, dict) else None
        events = scoring.get("events") if isinstance(scoring, dict) else None
        if not isinstance(events, list):
            return {
                "recomputed": False,
                "score": None,
                "reason": f"persisted_events_unavailable:{metric_name}",
                "metrics": projections,
            }
        projection = project_metric_events(
            metric_name,
            ordered_object_ids=ordered_ids,
            events=events,
            nominal_weight=float(nominal_weight),
            deduction_multiplier=deduction_multiplier,
        )
        projections[metric_name] = projection
        if not _is_number(projection.get("score")):
            return {
                "recomputed": False,
                "score": None,
                "reason": f"projected_score_unavailable:{metric_name}",
                "metrics": projections,
            }
        scores.append(float(projection["score"]))
    return {
        "recomputed": True,
        "score": mean(scores),
        "reason": None,
        "metrics": projections,
    }


def _benchmark_score(l1: Any, l3: Any) -> float | None:
    if not _is_number(l1) or not _is_number(l3):
        return None
    return 100.0 * (0.30 * float(l1) + 0.70 * float(l3))


def _generation_model(run_manifest: dict[str, Any]) -> str:
    source_root = run_manifest.get("source_case_root")
    if not source_root:
        return "unknown"
    case_manifest = _read_object(
        Path(str(source_root)) / "case_manifest.json",
        required=False,
    )
    return str(((case_manifest.get("source") or {}).get("namespace")) or "unknown")


def _run_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.parent.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_finite(value: Any, label: str) -> float:
    if not _is_number(value) or float(value) <= 0.0:
        raise ValueError(f"{label} must be finite and greater than zero")
    return float(value)


def _positive_float(value: str) -> float:
    try:
        return _positive_finite(float(value), "deduction multiplier")
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(item) for item in values if _is_number(item)]
    return mean(numbers) if numbers else None


def _mean_100(values: Iterable[Any]) -> float | None:
    value = _mean(values)
    return value * 100.0 if value is not None else None


def _difference(left: Any, right: Any) -> float | None:
    if not _is_number(left) or not _is_number(right):
        return None
    return float(left) - float(right)


if __name__ == "__main__":
    main()
