"""Fail-closed orchestration around one injected complete-room evaluator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol

from benchmark.non_rectangular.preflight import NonRectangularPreflightResult
from benchmark.non_rectangular.room_unit import (
    RoomEvaluationUnit,
    build_room_evaluation_units,
)


ROOM_REPORT_SCHEMA_VERSION = "non_rectangular_complete_room_report_v1"
WORKFLOW_EXECUTION_SCHEMA_VERSION = "non_rectangular_workflow_execution_v1"

L1_METRICS = ("collision", "oob", "support")
L3_METRICS = (
    "scale_consistency",
    "style_consistency",
    "object_pairing_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)
REQUIRED_METRICS = (*L1_METRICS, *L3_METRICS)
ROOM_METRIC_EXECUTION_ORDER = REQUIRED_METRICS


class RoomEvaluator(Protocol):
    """Future concrete evaluator; Functional must precede Placement."""

    def evaluate(self, unit: RoomEvaluationUnit) -> Mapping[str, Any]: ...


class RoomEvaluatorReportError(ValueError):
    """Raised when one injected evaluator violates the room-report contract."""


@dataclass(frozen=True, slots=True)
class RoomEvaluationInfrastructureFailure:
    room_id: str
    room_index: int
    failure_type: str
    message: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "room_index": self.room_index,
            "failure_type": self.failure_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class NonRectangularWorkflowExecution:
    """Room reports and infrastructure state before scene aggregation."""

    preflight: NonRectangularPreflightResult
    units: tuple[RoomEvaluationUnit, ...]
    room_reports: dict[str, dict[str, Any]]
    infrastructure_failures: tuple[RoomEvaluationInfrastructureFailure, ...]
    evaluator_call_order: tuple[str, ...]
    terminal_status: str

    @property
    def complete_room_count(self) -> int:
        return len(self.room_reports)

    @property
    def required_room_count(self) -> int:
        return len(self.preflight.room_order)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKFLOW_EXECUTION_SCHEMA_VERSION,
            "layout_id": self.preflight.layout_id,
            "terminal_status": self.terminal_status,
            "room_order": list(self.preflight.room_order),
            "required_metric_execution_order": list(
                ROOM_METRIC_EXECUTION_ORDER
            ),
            "evaluator_call_order": list(self.evaluator_call_order),
            "required_room_count": self.required_room_count,
            "complete_room_count": self.complete_room_count,
            "room_reports": deepcopy(self.room_reports),
            "infrastructure_failures": [
                item.public_dict() for item in self.infrastructure_failures
            ],
        }


def execute_non_rectangular_workflow(
    preflight: NonRectangularPreflightResult,
    *,
    room_evaluator: RoomEvaluator,
) -> NonRectangularWorkflowExecution:
    """Run rooms sequentially; continue after infrastructure failures."""

    if not isinstance(preflight, NonRectangularPreflightResult):
        raise TypeError("preflight must be NonRectangularPreflightResult")
    if not preflight.should_run_room_evaluation:
        return NonRectangularWorkflowExecution(
            preflight=preflight,
            units=(),
            room_reports={},
            infrastructure_failures=(),
            evaluator_call_order=(),
            terminal_status="failed",
        )
    evaluate = getattr(room_evaluator, "evaluate", None)
    if not callable(evaluate):
        raise TypeError("room_evaluator must provide callable evaluate(unit)")

    units = build_room_evaluation_units(preflight)
    reports: dict[str, dict[str, Any]] = {}
    failures: list[RoomEvaluationInfrastructureFailure] = []
    call_order: list[str] = []
    for unit in units:
        call_order.append(unit.room_id)
        try:
            raw = evaluate(unit)
            report = validate_complete_room_report(raw, unit=unit)
        except Exception as exc:
            failures.append(
                RoomEvaluationInfrastructureFailure(
                    room_id=unit.room_id,
                    room_index=unit.room_index,
                    failure_type=type(exc).__name__,
                    message=_bounded_message(exc),
                )
            )
            continue
        reports[unit.room_id] = report

    terminal_status = "incomplete" if failures else "complete"
    return NonRectangularWorkflowExecution(
        preflight=preflight,
        units=units,
        room_reports=reports,
        infrastructure_failures=tuple(failures),
        evaluator_call_order=tuple(call_order),
        terminal_status=terminal_status,
    )


def validate_complete_room_report(
    value: Mapping[str, Any],
    *,
    unit: RoomEvaluationUnit,
) -> dict[str, Any]:
    """Validate the normalized report returned by the single room evaluator."""

    if not isinstance(value, Mapping):
        raise RoomEvaluatorReportError("room evaluator report must be an object")
    expected_keys = {"schema_version", "room_id", "status", "metrics"}
    if set(value) != expected_keys:
        raise RoomEvaluatorReportError(
            "room evaluator report keys must be exactly "
            f"{sorted(expected_keys)!r}"
        )
    if value.get("schema_version") != ROOM_REPORT_SCHEMA_VERSION:
        raise RoomEvaluatorReportError("unsupported room evaluator report version")
    if value.get("room_id") != unit.room_id:
        raise RoomEvaluatorReportError(
            f"room evaluator report room_id must equal {unit.room_id!r}"
        )
    if value.get("status") != "complete":
        raise RoomEvaluatorReportError(
            "room evaluator report status must be 'complete'"
        )
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(REQUIRED_METRICS):
        raise RoomEvaluatorReportError(
            "room evaluator metrics must exactly cover the L1/L3 inventory"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for metric in REQUIRED_METRICS:
        raw = metrics[metric]
        if not isinstance(raw, Mapping):
            raise RoomEvaluatorReportError(f"{metric} result must be an object")
        required = {
            "metric",
            "status",
            "score",
            "evaluated_object_count",
            "raw_report",
        }
        if metric in L1_METRICS:
            required.add("invalid_count")
        if set(raw) != required:
            raise RoomEvaluatorReportError(
                f"{metric} result keys must be exactly {sorted(required)!r}"
            )
        if raw.get("metric") != metric:
            raise RoomEvaluatorReportError(f"{metric} result identity mismatch")
        if raw.get("status") != "complete":
            raise RoomEvaluatorReportError(f"{metric} status must be 'complete'")
        score = _score(raw.get("score"), path=f"{metric}.score")
        evaluated_count = _nonnegative_int(
            raw.get("evaluated_object_count"),
            path=f"{metric}.evaluated_object_count",
        )
        if evaluated_count > unit.generated_object_count:
            raise RoomEvaluatorReportError(
                f"{metric}.evaluated_object_count exceeds room generated objects"
            )
        raw_report = raw.get("raw_report")
        if not isinstance(raw_report, Mapping):
            raise RoomEvaluatorReportError(f"{metric}.raw_report must be an object")
        item = {
            "metric": metric,
            "status": "complete",
            "score": score,
            "evaluated_object_count": evaluated_count,
            "raw_report": deepcopy(dict(raw_report)),
        }
        if metric in L1_METRICS:
            invalid_count = _nonnegative_int(
                raw.get("invalid_count"),
                path=f"{metric}.invalid_count",
            )
            item["invalid_count"] = invalid_count
        normalized[metric] = item
    return {
        "schema_version": ROOM_REPORT_SCHEMA_VERSION,
        "room_id": unit.room_id,
        "status": "complete",
        "metrics": normalized,
    }


def _score(value: Any, *, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise RoomEvaluatorReportError(f"{path} must be finite within [0, 1]")
    return float(value)


def _nonnegative_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RoomEvaluatorReportError(f"{path} must be a non-negative integer")
    return value


def _bounded_message(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    return message[:500]


__all__ = [
    "L1_METRICS",
    "L3_METRICS",
    "REQUIRED_METRICS",
    "ROOM_METRIC_EXECUTION_ORDER",
    "ROOM_REPORT_SCHEMA_VERSION",
    "RoomEvaluationInfrastructureFailure",
    "RoomEvaluator",
    "RoomEvaluatorReportError",
    "NonRectangularWorkflowExecution",
    "execute_non_rectangular_workflow",
    "validate_complete_room_report",
]
