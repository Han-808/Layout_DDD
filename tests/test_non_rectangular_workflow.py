from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.non_rectangular import (
    L1_METRICS,
    L3_METRICS,
    ROOM_REPORT_SCHEMA_VERSION,
    NonRectangularEvaluationInput,
    execute_non_rectangular_workflow,
    prepare_non_rectangular_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"


def _fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _preflight(*, failed_count_gate: bool = False):
    program = _fixture("simple_multi_room_program.json")
    if failed_count_gate:
        program["target_total_instances"] = {"min": 7, "max": 8}
    value = NonRectangularEvaluationInput.from_artifacts(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=program,
        object_plan=_fixture("simple_multi_room_object_plan.json"),
        generated_scene=_fixture("simple_multi_room_scene.json"),
    )
    return prepare_non_rectangular_evaluation(value)


def _room_report(room_id: str, object_count: int) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for metric in (*L1_METRICS, *L3_METRICS):
        item: dict[str, Any] = {
            "metric": metric,
            "status": "complete",
            "score": 1.0,
            "evaluated_object_count": object_count,
            "raw_report": {"source": "fake_room_evaluator"},
        }
        if metric in L1_METRICS:
            item["invalid_count"] = 0
        metrics[metric] = item
    return {
        "schema_version": ROOM_REPORT_SCHEMA_VERSION,
        "room_id": room_id,
        "status": "complete",
        "metrics": metrics,
    }


class RecordingEvaluator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate(self, unit) -> dict[str, Any]:
        self.calls.append(unit.room_id)
        return _room_report(unit.room_id, unit.generated_object_count)


def test_metric_inventory_has_no_scene_allocation_plausibility_judge() -> None:
    assert L3_METRICS == (
        "scale_consistency",
        "style_consistency",
        "object_pairing_consistency",
        "functional_consistency",
        "semantic_placement_consistency",
    )
    assert all("allocation" not in metric for metric in L3_METRICS)


def test_workflow_invokes_one_room_evaluator_in_layout_order() -> None:
    evaluator = RecordingEvaluator()

    result = execute_non_rectangular_workflow(
        _preflight(),
        room_evaluator=evaluator,
    )

    assert evaluator.calls == ["room_000", "room_001"]
    assert result.evaluator_call_order == ("room_000", "room_001")
    assert list(result.room_reports) == ["room_000", "room_001"]
    assert result.terminal_status == "complete"
    assert result.complete_room_count == 2
    assert result.infrastructure_failures == ()


def test_count_gate_failure_never_invokes_room_evaluator() -> None:
    evaluator = RecordingEvaluator()

    result = execute_non_rectangular_workflow(
        _preflight(failed_count_gate=True),
        room_evaluator=evaluator,
    )

    assert evaluator.calls == []
    assert result.evaluator_call_order == ()
    assert result.units == ()
    assert result.room_reports == {}
    assert result.terminal_status == "failed"


class FirstRoomRaisesEvaluator(RecordingEvaluator):
    def evaluate(self, unit) -> dict[str, Any]:
        self.calls.append(unit.room_id)
        if unit.room_id == "room_000":
            raise RuntimeError("synthetic evaluator failure")
        return _room_report(unit.room_id, unit.generated_object_count)


def test_infrastructure_failure_keeps_later_room_diagnostics() -> None:
    evaluator = FirstRoomRaisesEvaluator()

    result = execute_non_rectangular_workflow(
        _preflight(),
        room_evaluator=evaluator,
    )

    assert evaluator.calls == ["room_000", "room_001"]
    assert list(result.room_reports) == ["room_001"]
    assert result.terminal_status == "incomplete"
    assert len(result.infrastructure_failures) == 1
    assert result.infrastructure_failures[0].room_id == "room_000"
    assert result.infrastructure_failures[0].failure_type == "RuntimeError"


class MissingMetricEvaluator(RecordingEvaluator):
    def evaluate(self, unit) -> dict[str, Any]:
        self.calls.append(unit.room_id)
        report = _room_report(unit.room_id, unit.generated_object_count)
        report["metrics"].pop("support")
        return report


def test_missing_required_metric_is_infrastructure_failure() -> None:
    result = execute_non_rectangular_workflow(
        _preflight(),
        room_evaluator=MissingMetricEvaluator(),
    )

    assert result.terminal_status == "incomplete"
    assert result.room_reports == {}
    assert len(result.infrastructure_failures) == 2
    assert all(
        item.failure_type == "RoomEvaluatorReportError"
        for item in result.infrastructure_failures
    )


class L2LeakEvaluator(RecordingEvaluator):
    def evaluate(self, unit) -> dict[str, Any]:
        report = _room_report(unit.room_id, unit.generated_object_count)
        report["metrics"]["oor"] = {
            "metric": "oor",
            "status": "complete",
            "score": 1.0,
            "evaluated_object_count": unit.generated_object_count,
            "raw_report": {},
        }
        return report


def test_l2_metric_cannot_leak_into_new_mode_inventory() -> None:
    result = execute_non_rectangular_workflow(
        _preflight(),
        room_evaluator=L2LeakEvaluator(),
    )

    assert result.terminal_status == "incomplete"
    assert result.room_reports == {}


class WrongRoomEvaluator(RecordingEvaluator):
    def evaluate(self, unit) -> dict[str, Any]:
        return _room_report("wrong_room", unit.generated_object_count)


def test_room_report_identity_mismatch_is_infrastructure_failure() -> None:
    result = execute_non_rectangular_workflow(
        _preflight(),
        room_evaluator=WrongRoomEvaluator(),
    )

    assert result.terminal_status == "incomplete"
    assert len(result.infrastructure_failures) == 2
    assert all("room_id must equal" in item.message for item in result.infrastructure_failures)


def test_execution_public_dict_contains_no_room_units_or_input_bodies() -> None:
    result = execute_non_rectangular_workflow(
        _preflight(),
        room_evaluator=RecordingEvaluator(),
    )

    public = result.public_dict()

    assert "units" not in public
    assert "preflight" not in public
    assert public["required_room_count"] == 2
    assert public["complete_room_count"] == 2
    order = public["required_metric_execution_order"]
    assert order.index("functional_consistency") < order.index(
        "semantic_placement_consistency"
    )
