from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.non_rectangular import (
    L1_LAYER,
    L1_METRICS,
    L3_LAYER,
    L3_METRICS,
    ROOM_REPORT_SCHEMA_VERSION,
    SCORING_PROFILE_SCHEMA_VERSION,
    NonRectangularEvaluationInput,
    NonRectangularReportError,
    build_non_rectangular_evaluation_report,
    execute_non_rectangular_workflow,
    prepare_non_rectangular_evaluation,
    program_coverage_compliance,
    validate_non_rectangular_scoring_profile,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"


def _fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _preflight(
    *,
    target_range: tuple[int, int] | None = None,
    invalid_mapping: bool = False,
    omit_generated_scene: bool = False,
):
    program = _fixture("simple_multi_room_program.json")
    plan = _fixture("simple_multi_room_object_plan.json")
    scene = _fixture("simple_multi_room_scene.json")
    if target_range is not None:
        program["target_total_instances"] = {
            "min": target_range[0],
            "max": target_range[1],
        }
    if invalid_mapping:
        for artifact in (plan, scene):
            artifact["rooms"][0].pop("program_id")
            artifact["rooms"][0].pop("room_type")
    value = NonRectangularEvaluationInput.from_artifacts(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=program,
        object_plan=plan,
        generated_scene=None if omit_generated_scene else scene,
    )
    return prepare_non_rectangular_evaluation(value)


def _profile() -> dict[str, Any]:
    return {
        "schema_version": SCORING_PROFILE_SCHEMA_VERSION,
        "profile_id": "internal_test_profile",
        "layer_weights": {L1_LAYER: 0.3, L3_LAYER: 0.7},
        "metric_weights": {
            L1_LAYER: {
                "collision": 1 / 3,
                "oob": 1 / 3,
                "support": 1 / 3,
            },
            L3_LAYER: {
                "scale_consistency": 0.04,
                "style_consistency": 0.07,
                "object_pairing_consistency": 0.09,
                "functional_consistency": 0.52,
                "semantic_placement_consistency": 0.28,
            },
        },
    }


def _inject_scoreable_mapping_error(preflight, *, rooms: int = 5):
    """Exercise aggregation with one error below the requested half-room stop."""

    preflight.program_mapping["coverage_compliance"] = (
        program_coverage_compliance(
            total_room_count=rooms,
            invalid_room_count=1,
        )
    )
    record = preflight.program_mapping["rooms"]["room_000"]
    record["valid"] = False
    record["failure_reasons"] = ["room_type_mismatch"]
    record["functional_score_override"] = None
    return preflight


class ConfigurableEvaluator:
    def __init__(
        self,
        *,
        scores: dict[str, dict[str, float]] | None = None,
        invalid_counts: dict[str, dict[str, int]] | None = None,
        fail_room: str | None = None,
    ) -> None:
        self.scores = scores or {}
        self.invalid_counts = invalid_counts or {}
        self.fail_room = fail_room

    def evaluate(self, unit) -> dict[str, Any]:
        if unit.room_id == self.fail_room:
            raise RuntimeError("synthetic report failure")
        metrics: dict[str, dict[str, Any]] = {}
        for metric in (*L1_METRICS, *L3_METRICS):
            item: dict[str, Any] = {
                "metric": metric,
                "status": "complete",
                "score": self.scores.get(unit.room_id, {}).get(metric, 1.0),
                "evaluated_object_count": unit.generated_object_count,
                "raw_report": {"source": "configurable_fake"},
            }
            if metric in L1_METRICS:
                item["invalid_count"] = self.invalid_counts.get(
                    unit.room_id, {}
                ).get(metric, 0)
            metrics[metric] = item
        return {
            "schema_version": ROOM_REPORT_SCHEMA_VERSION,
            "room_id": unit.room_id,
            "status": "complete",
            "metrics": metrics,
        }


def _report(
    *,
    preflight=None,
    evaluator=None,
    profile=None,
) -> dict[str, Any]:
    prepared = preflight or _preflight()
    execution = execute_non_rectangular_workflow(
        prepared,
        room_evaluator=evaluator or ConfigurableEvaluator(),
    )
    return build_non_rectangular_evaluation_report(
        execution,
        scoring_profile=profile,
    )


def test_complete_report_without_profile_has_metric_scores_but_no_overall() -> None:
    report = _report()

    assert report["terminal_status"] == "complete"
    assert list(report["rooms"]) == ["room_000", "room_001"]
    assert all(
        item["score"] == 1.0
        for item in report["aggregate"]["metrics"].values()
    )
    assert report["aggregate"]["overall_score"] is None
    assert report["aggregate"]["scoring_status"] == "profile_unconfigured"
    assert report["aggregate"]["official"] is False
    assert report["aggregate"]["publishable"] is False


def test_injected_profile_computes_non_official_layers_and_overall() -> None:
    report = _report(profile=_profile())

    assert report["aggregate"]["layers"][L1_LAYER]["score"] == 1.0
    assert report["aggregate"]["layers"][L3_LAYER]["score"] == 1.0
    assert report["aggregate"]["overall_score"] == 1.0
    assert (
        report["aggregate"]["scoring_status"]
        == "internally_scored_non_official"
    )
    assert report["coverage"]["official_score_eligible"] is False


def test_l1_aggregates_raw_invalid_counts_over_evaluated_objects() -> None:
    evaluator = ConfigurableEvaluator(
        invalid_counts={"room_000": {"collision": 1}}
    )

    report = _report(evaluator=evaluator)
    collision = report["aggregate"]["metrics"]["collision"]

    assert collision["invalid_count"] == 1
    assert collision["total_weight"] == 4
    assert collision["score"] == 0.75


def test_l3_room_scores_use_confirmed_weight_bases() -> None:
    evaluator = ConfigurableEvaluator(
        scores={
            "room_000": {
                "scale_consistency": 0.5,
                "functional_consistency": 0.5,
                "semantic_placement_consistency": 0.5,
            }
        }
    )

    report = _report(evaluator=evaluator)
    metrics = report["aggregate"]["metrics"]

    assert metrics["scale_consistency"]["score"] == 0.75
    assert metrics["functional_consistency"]["score"] == 0.75
    assert metrics["semantic_placement_consistency"]["score"] == 0.75
    assert {
        row["weight_basis"]
        for row in metrics["functional_consistency"]["room_contributions"]
    } == {"planned_instance_count"}
    assert {
        row["weight_basis"]
        for row in metrics["scale_consistency"]["room_contributions"]
    } == {"evaluated_object_count"}


def test_scoreable_mapping_error_applies_linear_scene_functional_factor() -> None:
    report = _report(
        preflight=_inject_scoreable_mapping_error(_preflight())
    )
    functional = report["aggregate"]["metrics"]["functional_consistency"]

    assert functional["score"] == 0.8
    assert functional["room_contributions"][0]["raw_score"] == 1.0
    assert functional["room_contributions"][0]["effective_score"] == 1.0
    assert functional["room_contributions"][0]["functional_score_override"] is None
    assert functional["pre_program_coverage_score"] == 1.0
    assert functional["program_coverage_factor"] == 0.8
    assert functional["post_program_coverage_score"] == 0.8
    assert report["rooms"]["room_000"]["effective_functional_score"] == 1.0
    assert report["rooms"]["room_000"]["functional_score_override"] is None
    assert (
        report["rooms"]["room_000"]["program_mapping_penalty_applied_at"]
        == "scene_functional_aggregate"
    )
    assert report["rooms"]["room_000"]["report"]["metrics"][
        "functional_consistency"
    ]["score"] == 1.0
    for metric in (*L1_METRICS, *L3_METRICS):
        if metric != "functional_consistency":
            assert report["aggregate"]["metrics"][metric]["score"] == 1.0


def test_half_room_mapping_threshold_returns_terminal_zero_without_room_metrics() -> None:
    report = _report(preflight=_preflight(invalid_mapping=True))

    assert report["terminal_status"] == "failed"
    assert report["failure_reason"] == "program_mapping_contract_failed"
    assert report["rooms"] == {}
    assert report["aggregate"]["overall_score"] == 0.0
    assert report["aggregate"]["terminal_case_score"] == 0.0
    assert (
        report["aggregate"]["scoring_status"]
        == "terminal_zero_program_mapping_threshold"
    )
    assert all(
        item["status"] == "not_run"
        for item in report["aggregate"]["metrics"].values()
    )


def test_count_factor_multiplies_only_scene_functional_score() -> None:
    preflight = _preflight(target_range=(5, 6))
    assert preflight.count_compliance["factor"] == pytest.approx((4 / 5) ** 2)

    report = _report(preflight=preflight)
    functional = report["aggregate"]["metrics"]["functional_consistency"]

    assert functional["pre_count_factor_score"] == 1.0
    assert functional["program_coverage_factor"] == 1.0
    assert functional["score"] == pytest.approx((4 / 5) ** 2)
    assert report["aggregate"]["metrics"]["semantic_placement_consistency"][
        "score"
    ] == 1.0


def test_mapping_and_count_factors_multiply_only_scene_functional() -> None:
    preflight = _inject_scoreable_mapping_error(
        _preflight(target_range=(5, 6))
    )
    report = _report(preflight=preflight)
    functional = report["aggregate"]["metrics"]["functional_consistency"]

    assert functional["pre_program_coverage_score"] == 1.0
    assert functional["program_coverage_factor"] == 0.8
    assert functional["post_program_coverage_score"] == 0.8
    assert functional["count_compliance_factor"] == pytest.approx((4 / 5) ** 2)
    assert functional["score"] == pytest.approx(0.8 * (4 / 5) ** 2)
    assert report["aggregate"]["metrics"]["semantic_placement_consistency"][
        "score"
    ] == 1.0


def test_infrastructure_gap_preserves_room_diagnostics_but_nulls_scores() -> None:
    report = _report(
        evaluator=ConfigurableEvaluator(fail_room="room_000"),
        profile=_profile(),
    )

    assert report["terminal_status"] == "incomplete"
    assert report["rooms"]["room_000"]["report"] is None
    assert report["rooms"]["room_001"]["report"] is not None
    assert report["coverage"]["missing_room_ids"] == ["room_000"]
    assert report["aggregate"]["overall_score"] is None
    assert report["aggregate"]["scoring_status"] == "incomplete_coverage"
    assert all(
        item["score"] is None
        for item in report["aggregate"]["metrics"].values()
    )
    assert all(
        item["diagnostic_partial_score"] == 1.0
        for item in report["aggregate"]["metrics"].values()
    )


def test_count_gate_failed_report_contains_no_room_metric_reports() -> None:
    report = _report(
        preflight=_preflight(
            target_range=(7, 8),
            omit_generated_scene=True,
        )
    )

    assert report["terminal_status"] == "failed"
    assert report["failure_reason"] == "object_count_contract_failed"
    assert report["rooms"] == {}
    assert report["aggregate"]["overall_score"] is None
    assert all(
        item["status"] == "not_run"
        for item in report["aggregate"]["metrics"].values()
    )


def test_profile_rejects_l2_or_non_normalized_weights() -> None:
    l2 = _profile()
    l2["layer_weights"]["l2_specification_fidelity"] = 0.0
    with pytest.raises(NonRectangularReportError, match="exactly"):
        validate_non_rectangular_scoring_profile(l2)

    bad_sum = _profile()
    bad_sum["metric_weights"][L3_LAYER]["functional_consistency"] = 0.5
    with pytest.raises(NonRectangularReportError, match="sum to 1.0"):
        validate_non_rectangular_scoring_profile(bad_sum)


def test_report_provenance_contains_hashes_not_artifact_bodies() -> None:
    report = _report()

    assert set(report["provenance"]["artifact_sha256"]) == {
        "room_layout",
        "room_program",
        "object_plan",
        "generated_scene",
    }
    assert "evaluation_input" not in report
    assert report["provenance"]["public_route_connected"] is False
