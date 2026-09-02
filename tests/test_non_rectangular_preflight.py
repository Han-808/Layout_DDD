from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmark.non_rectangular import (
    NonRectangularEvaluationInput,
    NonRectangularPreflightError,
    object_count_compliance,
    prepare_non_rectangular_evaluation,
    program_coverage_compliance,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"


def _fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _input(
    *,
    include_generated_scene: bool = True,
    plan_v2: bool = False,
) -> NonRectangularEvaluationInput:
    return NonRectangularEvaluationInput.from_artifacts(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        object_plan=_fixture(
            "simple_multi_room_object_plan_v2.json"
            if plan_v2
            else "simple_multi_room_object_plan.json"
        ),
        generated_scene=(
            _fixture("simple_multi_room_scene.json")
            if include_generated_scene
            else None
        ),
    )


def test_preflight_accepts_consistent_four_artifact_input() -> None:
    result = prepare_non_rectangular_evaluation(_input())

    assert result.layout_id == "fixture_simple_multi_room"
    assert result.room_order == ("room_000", "room_001")
    assert result.terminal_status == "ready"
    assert result.failure_reason is None
    assert result.should_run_room_evaluation is True
    assert result.count_compliance["factor"] == 1.0
    assert result.program_mapping["exact_bijection"] is True
    assert result.program_mapping["coverage_compliance"]["factor"] == 1.0
    assert result.program_mapping["coverage_compliance"]["failed"] is False
    assert set(result.artifact_sha256) == {
        "room_layout",
        "room_program",
        "object_plan",
        "generated_scene",
    }
    assert all(len(value) == 64 for value in result.artifact_sha256.values())


def test_preflight_accepts_simplified_v2_plan_with_same_scene_contract() -> None:
    result = prepare_non_rectangular_evaluation(_input(plan_v2=True))

    assert result.terminal_status == "ready"
    assert result.validation_reports["object_plan"][
        "plan_contract_version"
    ] == "v2"
    assert result.validation_reports["object_plan"]["planned_instance_count"] == 4
    assert result.program_mapping["exact_bijection"] is True


def test_evaluation_input_takes_defensive_copies() -> None:
    room_layout = _fixture("simple_multi_room.json")
    value = NonRectangularEvaluationInput.from_artifacts(
        room_layout=room_layout,
        room_program=_fixture("simple_multi_room_program.json"),
        object_plan=_fixture("simple_multi_room_object_plan.json"),
        generated_scene=_fixture("simple_multi_room_scene.json"),
    )

    room_layout["layout_id"] = "changed_after_construction"

    assert value.room_layout["layout_id"] == "fixture_simple_multi_room"


def test_preflight_rejects_layout_identity_drift() -> None:
    value = _input()
    value.object_plan["layout_id"] = "other_layout"

    with pytest.raises(NonRectangularPreflightError, match="layout_id mismatch"):
        prepare_non_rectangular_evaluation(value)


def test_preflight_rejects_room_coverage_drift() -> None:
    value = _input()
    value.object_plan["rooms"].pop()
    value.object_plan["room_order"] = ["room_000"]

    with pytest.raises(
        NonRectangularPreflightError,
        match="room coverage/order differs",
    ):
        prepare_non_rectangular_evaluation(value)


def test_preflight_rejects_program_cardinality_drift() -> None:
    value = _input()
    value.room_program["programs"].pop()
    value.room_program["program_order"] = ["kitchen_01"]

    with pytest.raises(
        NonRectangularPreflightError,
        match="program count must equal",
    ):
        prepare_non_rectangular_evaluation(value)


def test_half_room_mapping_error_is_terminal_zero_with_lower_integer_bound() -> None:
    value = _input()
    for artifact in (value.object_plan, value.generated_scene):
        artifact["rooms"][0].pop("program_id")
        artifact["rooms"][0].pop("room_type")

    result = prepare_non_rectangular_evaluation(value)

    mapping = result.program_mapping["rooms"]["room_000"]
    assert result.terminal_status == "failed"
    assert result.failure_reason == "program_mapping_contract_failed"
    assert result.should_run_room_evaluation is False
    assert mapping["valid"] is False
    assert mapping["functional_score_override"] is None
    assert mapping["failure_reasons"] == [
        "missing_program_id",
        "missing_room_type",
    ]
    compliance = result.program_mapping["coverage_compliance"]
    assert compliance["invalid_room_count"] == 1
    assert compliance["deduction_numerator"] == 1
    assert compliance["deduction_denominator"] == 2
    assert compliance["deduction"] == 0.5
    assert compliance["factor"] == 0.0
    assert compliance["failure_boundary_invalid_room_count"] == 1
    assert compliance["terminal_case_score"] == 0.0
    assert compliance["failed"] is True


def test_duplicate_program_assignment_can_fail_mapping_threshold() -> None:
    value = _input()
    for artifact in (value.object_plan, value.generated_scene):
        artifact["rooms"][1]["program_id"] = "kitchen_01"
        artifact["rooms"][1]["room_type"] = "kitchen"

    result = prepare_non_rectangular_evaluation(value)

    assert result.terminal_status == "failed"
    assert result.failure_reason == "program_mapping_contract_failed"
    assert result.should_run_room_evaluation is False
    assert result.program_mapping["exact_bijection"] is False
    assert result.program_mapping["invalid_room_count"] == 2
    assert result.program_mapping["coverage_compliance"]["failed"] is True
    assert result.program_mapping["coverage_compliance"]["factor"] == 0.0
    assert "duplicate_program_assignment" in result.program_mapping["rooms"][
        "room_000"
    ]["failure_reasons"]


@pytest.mark.parametrize(
    ("rooms", "invalid", "deduction", "factor", "boundary", "failed"),
    [
        (10, 0, 0.0, 1.0, 5, False),
        (10, 1, 1 / 10, 9 / 10, 5, False),
        (10, 4, 4 / 10, 6 / 10, 5, False),
        (10, 5, 5 / 10, 0.0, 5, True),
        (9, 3, 3 / 9, 6 / 9, 4, False),
        (9, 4, 4 / 9, 0.0, 4, True),
        (5, 1, 1 / 5, 4 / 5, 2, False),
        (5, 2, 2 / 5, 0.0, 2, True),
        (2, 1, 1 / 2, 0.0, 1, True),
        (1, 0, 0.0, 1.0, 1, False),
        (1, 1, 1.0, 0.0, 1, True),
    ],
)
def test_program_mapping_linear_policy_and_half_room_terminal_zero(
    rooms: int,
    invalid: int,
    deduction: float,
    factor: float,
    boundary: int,
    failed: bool,
) -> None:
    report = program_coverage_compliance(
        total_room_count=rooms,
        invalid_room_count=invalid,
    )

    assert report["deduction"] == pytest.approx(deduction)
    assert report["factor"] == pytest.approx(factor)
    assert report["failure_boundary_invalid_room_count"] == boundary
    assert report["failed"] is failed
    assert report["should_run_room_evaluation"] is (not failed)
    assert report["terminal_case_score"] == (0.0 if failed else None)


def test_stage_c_must_preserve_stage_a_mapping_even_when_mapping_is_content() -> None:
    value = _input()
    value.generated_scene["rooms"][0]["room_type"] = "bedroom"

    with pytest.raises(
        NonRectangularPreflightError,
        match="Stage C changed room_000.room_type",
    ):
        prepare_non_rectangular_evaluation(value)


def test_stage_c_slot_coverage_is_exact() -> None:
    value = _input()
    value.generated_scene["rooms"][0]["objects"].pop()

    with pytest.raises(
        NonRectangularPreflightError,
        match="Stage C slot coverage differs for room_000",
    ):
        prepare_non_rectangular_evaluation(value)


@pytest.mark.parametrize(
    ("planned", "minimum", "maximum", "factor", "failed"),
    [
        (40, 40, 50, 1.0, False),
        (50, 40, 50, 1.0, False),
        (35, 40, 50, (35 / 40) ** 2, False),
        (30, 40, 50, (30 / 40) ** 2, True),
        (60, 40, 50, (50 / 60) ** 2, False),
        (65, 40, 50, (50 / 65) ** 2, True),
    ],
)
def test_quadratic_count_compliance(
    planned: int,
    minimum: int,
    maximum: int,
    factor: float,
    failed: bool,
) -> None:
    report = object_count_compliance(
        planned_count=planned,
        minimum=minimum,
        maximum=maximum,
    )

    assert report["factor"] == pytest.approx(factor)
    assert report["failed"] is failed
    assert report["should_run_room_evaluation"] is (not failed)


def test_count_gate_early_failure_is_a_result_not_an_exception() -> None:
    value = _input(include_generated_scene=False)
    value.room_program["target_total_instances"] = {"min": 7, "max": 8}

    result = prepare_non_rectangular_evaluation(value)

    assert result.terminal_status == "failed"
    assert result.failure_reason == "object_count_contract_failed"
    assert result.should_run_room_evaluation is False
    assert result.count_compliance["factor"] == pytest.approx((4 / 7) ** 2)
    assert "generated_scene" not in result.validation_reports
    assert "generated_scene" not in result.artifact_sha256


def test_passing_count_gate_requires_generated_scene() -> None:
    value = _input(include_generated_scene=False)

    with pytest.raises(
        NonRectangularPreflightError,
        match="generated_scene is required",
    ):
        prepare_non_rectangular_evaluation(value)


def test_count_failure_threshold_is_strictly_less_than() -> None:
    report = object_count_compliance(
        planned_count=3,
        minimum=4,
        maximum=5,
        failure_threshold=(3 / 4) ** 2,
    )

    assert report["factor"] == report["failure_threshold"]
    assert report["failed"] is False


def test_preflight_public_dict_does_not_expose_artifact_bodies() -> None:
    report = prepare_non_rectangular_evaluation(_input()).public_dict()

    assert "evaluation_input" not in report
    assert set(report["artifact_sha256"]) == {
        "room_layout",
        "room_program",
        "object_plan",
        "generated_scene",
    }
