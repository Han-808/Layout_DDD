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
    run_internal_non_rectangular_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"


def _fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _input(
    *,
    count_failure: bool = False,
    mapping_failure: bool = False,
) -> NonRectangularEvaluationInput:
    program = _fixture("simple_multi_room_program.json")
    plan = _fixture("simple_multi_room_object_plan.json")
    scene = _fixture("simple_multi_room_scene.json")
    if count_failure:
        program["target_total_instances"] = {"min": 7, "max": 8}
    if mapping_failure:
        for artifact in (plan, scene):
            artifact["rooms"][0].pop("program_id")
            artifact["rooms"][0].pop("room_type")
    return NonRectangularEvaluationInput.from_artifacts(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=program,
        object_plan=plan,
        generated_scene=(
            None
            if count_failure
            else scene
        ),
    )


def _profile() -> dict[str, Any]:
    return {
        "schema_version": SCORING_PROFILE_SCHEMA_VERSION,
        "profile_id": "mock_whole_workflow_profile",
        "layer_weights": {L1_LAYER: 0.3, L3_LAYER: 0.7},
        "metric_weights": {
            L1_LAYER: {metric: 1 / 3 for metric in L1_METRICS},
            L3_LAYER: {
                "scale_consistency": 0.04,
                "style_consistency": 0.07,
                "object_pairing_consistency": 0.09,
                "functional_consistency": 0.52,
                "semantic_placement_consistency": 0.28,
            },
        },
    }


class WholeWorkflowMockEvaluator:
    def __init__(self) -> None:
        self.call_order: list[str] = []

    def evaluate(self, unit) -> dict[str, Any]:
        self.call_order.append(unit.room_id)
        score = 0.8 if unit.room_id == "room_000" else 1.0
        metrics: dict[str, dict[str, Any]] = {}
        for metric in (*L1_METRICS, *L3_METRICS):
            item: dict[str, Any] = {
                "metric": metric,
                "status": "complete",
                "score": score,
                "evaluated_object_count": unit.generated_object_count,
                "raw_report": {
                    "mock": True,
                    "room_id": unit.room_id,
                    "object_ids": list(unit.object_ids),
                },
            }
            if metric in L1_METRICS:
                item["invalid_count"] = (
                    1
                    if unit.room_id == "room_000" and metric == "collision"
                    else 0
                )
            metrics[metric] = item
        return {
            "schema_version": ROOM_REPORT_SCHEMA_VERSION,
            "room_id": unit.room_id,
            "status": "complete",
            "metrics": metrics,
        }


def test_mocked_whole_workflow_from_four_artifacts_to_scene_report() -> None:
    evaluator = WholeWorkflowMockEvaluator()

    report = run_internal_non_rectangular_evaluation(
        _input(),
        room_evaluator=evaluator,
        scoring_profile=_profile(),
    )

    assert evaluator.call_order == ["room_000", "room_001"]
    assert report["terminal_status"] == "complete"
    assert list(report["rooms"]) == ["room_000", "room_001"]
    assert report["coverage"]["all_required_rooms_complete"] is True
    assert report["aggregate"]["metrics"]["collision"]["score"] == 0.75
    for metric in L3_METRICS:
        assert report["aggregate"]["metrics"][metric]["score"] == 0.9
    expected_l1 = (0.75 + 1.0 + 1.0) / 3.0
    expected_overall = 0.3 * expected_l1 + 0.7 * 0.9
    assert report["aggregate"]["layers"][L1_LAYER]["score"] == pytest.approx(
        expected_l1
    )
    assert report["aggregate"]["layers"][L3_LAYER]["score"] == pytest.approx(
        0.9
    )
    assert report["aggregate"]["overall_score"] == pytest.approx(
        expected_overall
    )
    assert report["aggregate"]["official"] is False
    assert report["aggregate"]["publishable"] is False
    assert report["provenance"]["public_route_connected"] is False


def test_mocked_whole_workflow_early_stop_has_zero_room_calls() -> None:
    evaluator = WholeWorkflowMockEvaluator()

    report = run_internal_non_rectangular_evaluation(
        _input(count_failure=True),
        room_evaluator=evaluator,
        scoring_profile=_profile(),
    )

    assert evaluator.call_order == []
    assert report["terminal_status"] == "failed"
    assert report["failure_reason"] == "object_count_contract_failed"
    assert report["rooms"] == {}
    assert report["aggregate"]["overall_score"] is None


def test_mocked_whole_workflow_mapping_cutoff_is_zero_with_zero_room_calls() -> None:
    evaluator = WholeWorkflowMockEvaluator()

    report = run_internal_non_rectangular_evaluation(
        _input(mapping_failure=True),
        room_evaluator=evaluator,
        scoring_profile=_profile(),
    )

    assert evaluator.call_order == []
    assert report["terminal_status"] == "failed"
    assert report["failure_reason"] == "program_mapping_contract_failed"
    assert report["rooms"] == {}
    assert report["aggregate"]["overall_score"] == 0.0
    assert report["aggregate"]["terminal_case_score"] == 0.0
    assert (
        report["aggregate"]["scoring_status"]
        == "terminal_zero_program_mapping_threshold"
    )
