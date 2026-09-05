from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.rescore_scene_quality_posthoc import _reproject_l1, rescore_run


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "metric": "functional_consistency",
        "category": "group_function_failure",
        "severity": "impaired",
        "burden": 0.4,
        "allocations": {"chair": 0.4},
        "target_ids": ["chair"],
        "source_reference": {},
    }


def _scoring(metric: str, *, score: float, events: list[dict]) -> dict:
    burden = 0.8 if metric == "functional_consistency" else 0.0
    return {
        "schema_version": "object_equivalent_burden_v1",
        "metric": metric,
        "ordered_canonical_object_ids": ["chair", "table"],
        "events": events,
        "score": score,
        "burden_total_b_m": burden,
        "capped_object_burdens": {
            "chair": burden,
            "table": 0.0,
        },
    }


def test_posthoc_rescore_is_read_only_and_reprojects_same_object_events(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_run"
    case_dir = source / "cases" / "S001"
    source_case = tmp_path / "dataset" / "S001"
    metrics = {}
    for name in (
        "scale_consistency",
        "style_consistency",
        "object_pairing_consistency",
        "functional_consistency",
        "semantic_placement_consistency",
    ):
        functional = name == "functional_consistency"
        events = [_event("a"), _event("b")] if functional else []
        metrics[name] = {
            "status": "evaluated",
            "score": 0.2 if functional else 1.0,
            "judgement": {
                "verdict": "invalid" if functional else "valid",
            },
            "coverage": {
                "score_grounding": {"fraction": 1.0},
            },
            "scoring": _scoring(
                name,
                score=0.2 if functional else 1.0,
                events=events,
            ),
        }
    scene_report = {
        "scoring": {
            "ordered_canonical_object_ids": ["chair", "table"],
        },
        "metrics": metrics,
    }
    report_path = case_dir / "scene_quality_report.json"
    _write(report_path, scene_report)
    _write(
        case_dir / "evaluation_report.json",
        {
            "layer_reports": {
                "l1_physical_plausibility": {"score": 1.0},
            }
        },
    )
    _write(
        case_dir / "case_run_manifest.json",
        {
            "source_case_root": str(source_case),
            "benchmark_score_100": 75.36,
        },
    )
    _write(
        source_case / "case_manifest.json",
        {"source": {"namespace": "generator_a"}},
    )
    _write(
        source / "experiment_plan.json",
        {"model_route": {"model": "judge_a"}},
    )
    before = hashlib.sha256(report_path.read_bytes()).hexdigest()

    result = rescore_run(
        input_root=source,
        output_root=tmp_path / "analysis",
    )

    case = result["cases"][0]
    function = case["metrics"]["functional_consistency"]
    assert function["old"]["raw_score"] == pytest.approx(0.2)
    assert function["new"]["raw_score"] == pytest.approx(0.6)
    assert function["new"]["burden_total"] == pytest.approx(0.4)
    assert function["new"]["aggregation"] == (
        "per_object_max_across_events"
    )
    assert case["old_l3"]["score_100"] == pytest.approx(58.4)
    assert case["new_l3"]["score_100"] == pytest.approx(79.2)
    assert case["old_benchmark_score_100"] == pytest.approx(70.88)
    assert case["new_benchmark_score_100"] == pytest.approx(85.44)
    assert result["models"][0]["generation_model"] == "generator_a"
    assert result["new_evidence_acquired"] is False
    assert result["track_b"]["status"] == "requires_new_evaluation_run"
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == before


def test_posthoc_l1_reprojection_applies_deduction_multiplier(
    tmp_path: Path,
) -> None:
    object_ids = [f"obj_{index}" for index in range(10)]
    invalid_collision = {
        "event_id": "collision-event",
        "burden": 1.0,
        "allocations": {object_ids[0]: 1.0},
    }
    metrics = {}
    for metric_name in ("collision", "support", "oob"):
        metrics[metric_name] = {
            "scoring": {
                "events": (
                    [invalid_collision]
                    if metric_name == "collision"
                    else []
                )
            }
        }
    report_path = tmp_path / "l1_report.json"
    _write(report_path, {"metrics": metrics})

    scaled = _reproject_l1(
        report_path,
        ordered_ids=object_ids,
        deduction_multiplier=2.0,
    )
    unscaled = _reproject_l1(
        report_path,
        ordered_ids=object_ids,
        deduction_multiplier=1.0,
    )

    assert scaled["recomputed"] is True
    assert unscaled["recomputed"] is True
    assert scaled["metrics"]["collision"]["score"] == pytest.approx(0.5)
    assert unscaled["metrics"]["collision"]["score"] == pytest.approx(0.75)
    assert scaled["score"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
