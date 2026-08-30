"""Cross-artifact preflight for the additive non-rectangular workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from benchmark.non_rectangular.contracts import (
    NonRectangularContractError,
    validate_multi_room_object_plan,
    validate_multi_room_scene,
    validate_room_program,
)
from benchmark.non_rectangular.room_layout import (
    RoomLayoutValidationError,
    validate_room_layout,
)


COUNT_COMPLIANCE_POLICY = "quadratic_outside_inclusive_range_v1"
COUNT_COMPLIANCE_FAILURE_THRESHOLD = 0.6
PROGRAM_COVERAGE_POLICY = "linear_invalid_room_with_half_room_zero_v1"
PREFLIGHT_SCHEMA_VERSION = "non_rectangular_evaluation_preflight_v1"


class NonRectangularPreflightError(ValueError):
    """Raised when artifacts cannot form one structurally evaluable case."""


@dataclass(frozen=True, slots=True)
class NonRectangularEvaluationInput:
    """Authoritative artifacts; Stage C may be absent after an early count stop."""

    room_layout: dict[str, Any]
    room_program: dict[str, Any]
    object_plan: dict[str, Any]
    generated_scene: dict[str, Any] | None

    @classmethod
    def from_artifacts(
        cls,
        *,
        room_layout: Mapping[str, Any],
        room_program: Mapping[str, Any],
        object_plan: Mapping[str, Any],
        generated_scene: Mapping[str, Any] | None,
    ) -> "NonRectangularEvaluationInput":
        """Take defensive copies so later caller mutation cannot change a run."""

        return cls(
            room_layout=deepcopy(dict(room_layout)),
            room_program=deepcopy(dict(room_program)),
            object_plan=deepcopy(dict(object_plan)),
            generated_scene=(
                deepcopy(dict(generated_scene))
                if generated_scene is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class NonRectangularPreflightResult:
    """Validated input plus deterministic cross-artifact diagnostics."""

    evaluation_input: NonRectangularEvaluationInput
    layout_id: str
    room_order: tuple[str, ...]
    validation_reports: dict[str, dict[str, Any]]
    program_mapping: dict[str, Any]
    count_compliance: dict[str, Any]
    artifact_sha256: dict[str, str]
    terminal_status: str
    failure_reason: str | None

    @property
    def should_run_room_evaluation(self) -> bool:
        return self.terminal_status == "ready"

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "layout_id": self.layout_id,
            "room_order": list(self.room_order),
            "terminal_status": self.terminal_status,
            "failure_reason": self.failure_reason,
            "should_run_room_evaluation": self.should_run_room_evaluation,
            "validation_reports": deepcopy(self.validation_reports),
            "program_mapping": deepcopy(self.program_mapping),
            "count_compliance": deepcopy(self.count_compliance),
            "artifact_sha256": dict(self.artifact_sha256),
        }


def prepare_non_rectangular_evaluation(
    evaluation_input: NonRectangularEvaluationInput,
) -> NonRectangularPreflightResult:
    """Validate artifact contracts, identities, mapping, slots, and count gate."""

    if not isinstance(evaluation_input, NonRectangularEvaluationInput):
        raise NonRectangularPreflightError(
            "evaluation_input must be NonRectangularEvaluationInput"
        )
    try:
        reports: dict[str, dict[str, Any]] = {
            "room_layout": validate_room_layout(evaluation_input.room_layout),
            "room_program": validate_room_program(evaluation_input.room_program),
            "object_plan": validate_multi_room_object_plan(
                evaluation_input.object_plan
            ),
        }
    except (NonRectangularContractError, RoomLayoutValidationError) as exc:
        raise NonRectangularPreflightError(str(exc)) from exc

    layout_ids: dict[str, str] = {
        name: str(report["layout_id"])
        for name, report in reports.items()
    }
    if len(set(layout_ids.values())) != 1:
        raise NonRectangularPreflightError(
            f"artifact layout_id mismatch: {layout_ids!r}"
        )
    layout_id = layout_ids["room_layout"]
    room_order = tuple(str(item) for item in reports["room_layout"]["room_ids"])
    for name in ("object_plan",):
        actual = tuple(str(item) for item in reports[name]["room_ids"])
        if actual != room_order:
            raise NonRectangularPreflightError(
                f"{name} room coverage/order differs from room_layout: "
                f"expected={room_order!r}, actual={actual!r}"
            )
    if int(reports["room_program"]["program_count"]) != len(room_order):
        raise NonRectangularPreflightError(
            "room_program program count must equal room_layout room count"
        )

    plan_rooms = _rooms_by_id(evaluation_input.object_plan["rooms"])
    program_mapping = program_mapping_report(
        room_order=room_order,
        programs=evaluation_input.room_program["programs"],
        plan_rooms=plan_rooms,
    )
    target = reports["room_program"]["target_total_instances"]
    count_compliance = object_count_compliance(
        planned_count=int(reports["object_plan"]["planned_instance_count"]),
        minimum=int(target["min"]),
        maximum=int(target["max"]),
    )
    mapping_failed = bool(program_mapping["coverage_compliance"]["failed"])
    count_failed = bool(count_compliance["failed"])
    artifact_sha256 = {
        "room_layout": _canonical_sha256(evaluation_input.room_layout),
        "room_program": _canonical_sha256(evaluation_input.room_program),
        "object_plan": _canonical_sha256(evaluation_input.object_plan),
    }
    if mapping_failed or count_failed:
        if mapping_failed and count_failed:
            failure_reason = (
                "program_mapping_and_object_count_contract_failed"
            )
        elif mapping_failed:
            failure_reason = "program_mapping_contract_failed"
        else:
            failure_reason = "object_count_contract_failed"
        return NonRectangularPreflightResult(
            evaluation_input=evaluation_input,
            layout_id=layout_id,
            room_order=room_order,
            validation_reports=reports,
            program_mapping=program_mapping,
            count_compliance=count_compliance,
            artifact_sha256=artifact_sha256,
            terminal_status="failed",
            failure_reason=failure_reason,
        )

    if evaluation_input.generated_scene is None:
        raise NonRectangularPreflightError(
            "generated_scene is required when the Stage-A count gate passes"
        )
    try:
        scene_report = validate_multi_room_scene(
            evaluation_input.generated_scene
        )
    except NonRectangularContractError as exc:
        raise NonRectangularPreflightError(str(exc)) from exc
    reports["generated_scene"] = scene_report
    if str(scene_report["layout_id"]) != layout_id:
        raise NonRectangularPreflightError(
            "artifact layout_id mismatch: generated_scene differs from Stage A"
        )
    actual_scene_rooms = tuple(str(item) for item in scene_report["room_ids"])
    if actual_scene_rooms != room_order:
        raise NonRectangularPreflightError(
            "generated_scene room coverage/order differs from room_layout: "
            f"expected={room_order!r}, actual={actual_scene_rooms!r}"
        )
    scene_rooms = _rooms_by_id(evaluation_input.generated_scene["rooms"])
    _require_stage_c_mapping_preserved(
        room_order=room_order,
        plan_rooms=plan_rooms,
        scene_rooms=scene_rooms,
    )
    _require_exact_slot_coverage(
        room_order=room_order,
        planned=reports["object_plan"]["room_slot_counts"],
        generated=scene_report["room_slot_counts"],
    )
    artifact_sha256["generated_scene"] = _canonical_sha256(
        evaluation_input.generated_scene
    )
    return NonRectangularPreflightResult(
        evaluation_input=evaluation_input,
        layout_id=layout_id,
        room_order=room_order,
        validation_reports=reports,
        program_mapping=program_mapping,
        count_compliance=count_compliance,
        artifact_sha256=artifact_sha256,
        terminal_status="ready",
        failure_reason=None,
    )


def object_count_compliance(
    *,
    planned_count: int,
    minimum: int,
    maximum: int,
    failure_threshold: float = COUNT_COMPLIANCE_FAILURE_THRESHOLD,
) -> dict[str, Any]:
    """Return the frozen quadratic count factor and early-stop decision."""

    for name, value in (
        ("planned_count", planned_count),
        ("minimum", minimum),
        ("maximum", maximum),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise NonRectangularPreflightError(
                f"{name} must be a positive integer"
            )
    if minimum > maximum:
        raise NonRectangularPreflightError("minimum must be <= maximum")
    if (
        isinstance(failure_threshold, bool)
        or not isinstance(failure_threshold, (int, float))
        or not math.isfinite(float(failure_threshold))
        or not 0.0 < float(failure_threshold) < 1.0
    ):
        raise NonRectangularPreflightError(
            "failure_threshold must be finite and strictly between 0 and 1"
        )

    if minimum <= planned_count <= maximum:
        factor = 1.0
        deviation = "within_range"
    elif planned_count < minimum:
        factor = (float(planned_count) / float(minimum)) ** 2
        deviation = "below_minimum"
    else:
        factor = (float(maximum) / float(planned_count)) ** 2
        deviation = "above_maximum"
    failed = factor < float(failure_threshold)
    return {
        "policy": COUNT_COMPLIANCE_POLICY,
        "planned_instance_count": planned_count,
        "target_total_instances": {"min": minimum, "max": maximum},
        "deviation": deviation,
        "factor": factor,
        "failure_threshold": float(failure_threshold),
        "threshold_comparison": "factor < failure_threshold",
        "failed": failed,
        "should_run_room_evaluation": not failed,
    }


def program_coverage_compliance(
    *,
    total_room_count: int,
    invalid_room_count: int,
) -> dict[str, Any]:
    """Return the frozen linear mapping deduction and terminal-zero decision.

    With ``k`` invalid room mappings across ``n`` rooms, scoreable cases use
    factor ``1 - k/n``.  At ``max(1, floor(n/2))`` invalid rooms the case
    receives a terminal zero and room evaluation does not run.
    """

    for name, value in (
        ("total_room_count", total_room_count),
        ("invalid_room_count", invalid_room_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise NonRectangularPreflightError(f"{name} must be an integer")
    if total_room_count < 1:
        raise NonRectangularPreflightError(
            "total_room_count must be a positive integer"
        )
    if not 0 <= invalid_room_count <= total_room_count:
        raise NonRectangularPreflightError(
            "invalid_room_count must be within [0, total_room_count]"
        )

    failure_boundary = max(1, total_room_count // 2)
    failed = invalid_room_count >= failure_boundary
    deduction = float(invalid_room_count) / float(total_room_count)
    factor = 0.0 if failed else 1.0 - deduction
    return {
        "policy": PROGRAM_COVERAGE_POLICY,
        "total_room_count": total_room_count,
        "invalid_room_count": invalid_room_count,
        "deduction_numerator": invalid_room_count,
        "deduction_denominator": total_room_count,
        "deduction": deduction,
        "factor": factor,
        "failure_boundary_invalid_room_count": failure_boundary,
        "threshold_comparison": (
            "invalid_room_count >= max(1, floor(total_room_count / 2))"
        ),
        "failed": failed,
        "should_run_room_evaluation": not failed,
        "terminal_case_score": 0.0 if failed else None,
        "affects_metric": "functional_consistency",
    }


def _require_stage_c_mapping_preserved(
    *,
    room_order: tuple[str, ...],
    plan_rooms: Mapping[str, Mapping[str, Any]],
    scene_rooms: Mapping[str, Mapping[str, Any]],
) -> None:
    for room_id in room_order:
        for field in ("program_id", "room_type"):
            planned = plan_rooms[room_id].get(field)
            generated = scene_rooms[room_id].get(field)
            if planned != generated:
                raise NonRectangularPreflightError(
                    f"Stage C changed {room_id}.{field}: "
                    f"planned={planned!r}, generated={generated!r}"
                )


def _require_exact_slot_coverage(
    *,
    room_order: tuple[str, ...],
    planned: Mapping[str, Mapping[str, int]],
    generated: Mapping[str, Mapping[str, int]],
) -> None:
    for room_id in room_order:
        expected = dict(planned[room_id])
        actual = dict(generated[room_id])
        if actual != expected:
            raise NonRectangularPreflightError(
                f"Stage C slot coverage differs for {room_id}: "
                f"expected={expected!r}, actual={actual!r}"
            )


def program_mapping_report(
    *,
    room_order: tuple[str, ...],
    programs: list[Mapping[str, Any]],
    plan_rooms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    programs_by_id = {
        str(item["program_id"]): str(item["room_type"])
        for item in programs
    }
    assignments: dict[str, list[str]] = {}
    coverage_assignments: dict[str, list[str]] = {}
    records: dict[str, dict[str, Any]] = {}
    for room_id in room_order:
        room = plan_rooms[room_id]
        program_id = room.get("program_id")
        room_type = room.get("room_type")
        reasons: list[str] = []
        normalized_program_id = (
            str(program_id)
            if isinstance(program_id, str) and program_id
            else None
        )
        normalized_room_type = (
            str(room_type) if isinstance(room_type, str) and room_type else None
        )
        expected_room_type = (
            programs_by_id.get(normalized_program_id)
            if normalized_program_id is not None
            else None
        )
        if normalized_program_id is None:
            reasons.append("missing_program_id")
        elif expected_room_type is None:
            reasons.append("unknown_program_id")
        else:
            assignments.setdefault(normalized_program_id, []).append(room_id)
        if normalized_room_type is None:
            reasons.append("missing_room_type")
        elif (
            expected_room_type is not None
            and normalized_room_type != expected_room_type
        ):
            reasons.append("room_type_mismatch")
        if (
            normalized_program_id is not None
            and expected_room_type is not None
            and normalized_room_type == expected_room_type
        ):
            coverage_assignments.setdefault(normalized_program_id, []).append(
                room_id
            )
        records[room_id] = {
            "room_id": room_id,
            "program_id": normalized_program_id,
            "room_type": normalized_room_type,
            "expected_room_type": expected_room_type,
            "valid": not reasons,
            "failure_reasons": reasons,
            "functional_score_override": None,
        }

    for program_id, assigned_rooms in assignments.items():
        if len(assigned_rooms) <= 1:
            continue
        for room_id in assigned_rooms:
            record = records[room_id]
            record["failure_reasons"].append("duplicate_program_assignment")
            record["valid"] = False

    used_once = {
        program_id
        for program_id, assigned_rooms in coverage_assignments.items()
        if len(assigned_rooms) == 1
    }
    covered_program_ids = [
        program_id
        for program_id in programs_by_id
        if program_id in coverage_assignments
    ]
    missing_program_ids = [
        program_id
        for program_id in programs_by_id
        if program_id not in coverage_assignments
    ]
    non_bijective_program_ids = [
        program_id
        for program_id in programs_by_id
        if len(coverage_assignments.get(program_id, ())) != 1
    ]
    valid_room_count = sum(bool(record["valid"]) for record in records.values())
    coverage_compliance = program_coverage_compliance(
        total_room_count=len(room_order),
        invalid_room_count=len(room_order) - valid_room_count,
    )
    return {
        "policy": "model_declared_room_program_bijection_v1",
        "rooms": records,
        "valid_room_count": valid_room_count,
        "invalid_room_count": len(room_order) - valid_room_count,
        "covered_program_ids": covered_program_ids,
        "missing_program_ids": missing_program_ids,
        "non_bijective_program_ids": non_bijective_program_ids,
        "exact_bijection": (
            valid_room_count == len(room_order)
            and len(used_once) == len(programs_by_id)
        ),
        "coverage_compliance": coverage_compliance,
        "room_functional_score_overrides_enabled": False,
        "coverage_penalty_applied_at": "scene_functional_aggregate",
        "scoreable_error_affects_only_functional": True,
        "terminal_threshold_ends_case": True,
        "failure_affects_only_functional": False,
    }


def _rooms_by_id(rooms: Any) -> dict[str, Mapping[str, Any]]:
    return {str(room["room_id"]): room for room in rooms}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NonRectangularPreflightError(
            "artifact is not canonical-JSON serializable"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "COUNT_COMPLIANCE_FAILURE_THRESHOLD",
    "COUNT_COMPLIANCE_POLICY",
    "NonRectangularEvaluationInput",
    "NonRectangularPreflightError",
    "NonRectangularPreflightResult",
    "PREFLIGHT_SCHEMA_VERSION",
    "PROGRAM_COVERAGE_POLICY",
    "object_count_compliance",
    "prepare_non_rectangular_evaluation",
    "program_coverage_compliance",
    "program_mapping_report",
]
