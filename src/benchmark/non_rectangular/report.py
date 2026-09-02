"""Scene aggregation and fail-closed reporting for the internal framework."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from benchmark.non_rectangular.contracts import (
    NON_RECTANGULAR_EVALUATION_MODE,
)
from benchmark.non_rectangular.workflow import (
    L1_METRICS,
    L3_METRICS,
    REQUIRED_METRICS,
    NonRectangularWorkflowExecution,
)


EVALUATION_REPORT_SCHEMA_VERSION = "non_rectangular_evaluation_report_v1"
SCORING_PROFILE_SCHEMA_VERSION = "non_rectangular_scoring_profile_v1"
L1_LAYER = "l1_physical_plausibility"
L3_LAYER = "l3_scene_quality"
SCORING_LAYERS = (L1_LAYER, L3_LAYER)
DEFAULT_NON_RECTANGULAR_SCORING_PROFILE = {
    "schema_version": SCORING_PROFILE_SCHEMA_VERSION,
    "profile_id": "non_rectangular_room_weighted_v1",
    "layer_weights": {L1_LAYER: 0.30, L3_LAYER: 0.70},
    "metric_weights": {
        L1_LAYER: {
            "collision": 1.0 / 3.0,
            "oob": 1.0 / 3.0,
            "support": 1.0 / 3.0,
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


class NonRectangularReportError(ValueError):
    """Raised when internal aggregation inputs or a test profile are invalid."""


def build_non_rectangular_evaluation_report(
    execution: NonRectangularWorkflowExecution,
    *,
    scoring_profile: Mapping[str, Any] | None = None,
    public_route_connected: bool = False,
) -> dict[str, Any]:
    """Build the two-level room/scene report without claiming an official score."""

    if not isinstance(execution, NonRectangularWorkflowExecution):
        raise NonRectangularReportError(
            "execution must be NonRectangularWorkflowExecution"
        )
    resolved_profile = (
        validate_non_rectangular_scoring_profile(scoring_profile)
        if scoring_profile is not None
        else None
    )
    preflight = execution.preflight
    if execution.terminal_status == "failed":
        terminal_mapping_zero = bool(
            preflight.program_mapping["coverage_compliance"]["failed"]
        )
        return {
            "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
            "evaluation_mode": NON_RECTANGULAR_EVALUATION_MODE,
            "layout_id": preflight.layout_id,
            "terminal_status": "failed",
            "failure_reason": preflight.failure_reason,
            "preflight": preflight.public_dict(),
            "rooms": {},
            "aggregate": _empty_aggregate(
                scoring_profile=resolved_profile,
                reason=(preflight.failure_reason or "preflight_failed"),
                terminal_zero=terminal_mapping_zero,
            ),
            "coverage": {
                "required_room_count": len(preflight.room_order),
                "complete_room_count": 0,
                "missing_room_ids": list(preflight.room_order),
                "all_required_rooms_complete": False,
                "infrastructure_failure_count": 0,
                "official_score_eligible": False,
            },
            "provenance": {
                "artifact_sha256": dict(preflight.artifact_sha256),
                "public_route_connected": bool(public_route_connected),
                "official_scoring_profile_frozen": False,
            },
        }

    units_by_id = {unit.room_id: unit for unit in execution.units}
    failures_by_room: dict[str, list[dict[str, Any]]] = {}
    for failure in execution.infrastructure_failures:
        failures_by_room.setdefault(failure.room_id, []).append(
            failure.public_dict()
        )
    rooms: dict[str, dict[str, Any]] = {}
    for room_id in preflight.room_order:
        unit = units_by_id[room_id]
        raw_room_report = deepcopy(execution.room_reports.get(room_id))
        raw_functional_score = (
            raw_room_report["metrics"]["functional_consistency"]["score"]
            if raw_room_report is not None
            else None
        )
        effective_functional_score = (
            unit.functional_score_override
            if unit.functional_score_override is not None
            else raw_functional_score
        )
        rooms[room_id] = {
            "room_index": unit.room_index,
            "program_id": unit.program_id,
            "room_type": unit.room_type,
            "program_mapping_valid": unit.mapping_valid,
            "program_mapping_failure_reasons": list(
                unit.mapping_failure_reasons
            ),
            "planned_instance_count": unit.planned_instance_count,
            "generated_object_count": unit.generated_object_count,
            "effective_functional_score": effective_functional_score,
            "functional_score_override": unit.functional_score_override,
            "program_mapping_penalty_applied_at": (
                "scene_functional_aggregate"
            ),
            "report": raw_room_report,
            "infrastructure_failures": failures_by_room.get(room_id, []),
        }

    aggregate_metrics = _aggregate_metrics(execution)
    layer_scores, overall_score, scoring_status = _score_layers(
        aggregate_metrics,
        scoring_profile=resolved_profile,
        execution_complete=execution.terminal_status == "complete",
    )
    missing_room_ids = [
        room_id
        for room_id in preflight.room_order
        if room_id not in execution.room_reports
    ]
    all_rooms_complete = not missing_room_ids and not execution.infrastructure_failures
    return {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "evaluation_mode": NON_RECTANGULAR_EVALUATION_MODE,
        "layout_id": preflight.layout_id,
        "terminal_status": execution.terminal_status,
        "failure_reason": None,
        "preflight": preflight.public_dict(),
        "rooms": rooms,
        "aggregate": {
            "metrics": aggregate_metrics,
            "layers": layer_scores,
            "overall_score": overall_score,
            "terminal_case_score": None,
            "scoring_status": scoring_status,
            "scoring_profile": deepcopy(resolved_profile),
            "official": False,
            "publishable": False,
        },
        "coverage": {
            "required_room_count": len(preflight.room_order),
            "complete_room_count": len(execution.room_reports),
            "missing_room_ids": missing_room_ids,
            "all_required_rooms_complete": all_rooms_complete,
            "infrastructure_failure_count": len(
                execution.infrastructure_failures
            ),
            "official_score_eligible": False,
        },
        "provenance": {
            "artifact_sha256": dict(preflight.artifact_sha256),
            "public_route_connected": bool(public_route_connected),
            "official_scoring_profile_frozen": False,
        },
    }


def validate_non_rectangular_scoring_profile(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an explicitly injected non-official profile for internal tests."""

    if not isinstance(value, Mapping):
        raise NonRectangularReportError("scoring_profile must be an object")
    expected = {
        "schema_version",
        "profile_id",
        "layer_weights",
        "metric_weights",
    }
    if set(value) != expected:
        raise NonRectangularReportError(
            f"scoring_profile keys must be exactly {sorted(expected)!r}"
        )
    if value.get("schema_version") != SCORING_PROFILE_SCHEMA_VERSION:
        raise NonRectangularReportError("unsupported scoring profile version")
    profile_id = value.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise NonRectangularReportError("scoring_profile.profile_id is required")
    layer_weights = _weight_mapping(
        value.get("layer_weights"),
        expected=SCORING_LAYERS,
        path="layer_weights",
    )
    metric_weights = value.get("metric_weights")
    if not isinstance(metric_weights, Mapping) or set(metric_weights) != set(
        SCORING_LAYERS
    ):
        raise NonRectangularReportError(
            "metric_weights must contain exactly L1 and L3"
        )
    normalized_metric_weights = {
        L1_LAYER: _weight_mapping(
            metric_weights[L1_LAYER],
            expected=L1_METRICS,
            path=f"metric_weights.{L1_LAYER}",
        ),
        L3_LAYER: _weight_mapping(
            metric_weights[L3_LAYER],
            expected=L3_METRICS,
            path=f"metric_weights.{L3_LAYER}",
        ),
    }
    return {
        "schema_version": SCORING_PROFILE_SCHEMA_VERSION,
        "profile_id": profile_id.strip(),
        "layer_weights": layer_weights,
        "metric_weights": normalized_metric_weights,
        "official": False,
    }


def _aggregate_metrics(
    execution: NonRectangularWorkflowExecution,
) -> dict[str, dict[str, Any]]:
    units = {unit.room_id: unit for unit in execution.units}
    complete_execution = execution.terminal_status == "complete"
    output: dict[str, dict[str, Any]] = {}
    for metric in REQUIRED_METRICS:
        contributions: list[dict[str, Any]] = []
        invalid_total = 0
        weighted_sum = 0.0
        weight_total = 0
        for room_id in execution.preflight.room_order:
            report = execution.room_reports.get(room_id)
            if report is None:
                continue
            metric_report = report["metrics"][metric]
            unit = units[room_id]
            raw_score = float(metric_report["score"])
            override = (
                unit.functional_score_override
                if metric == "functional_consistency"
                else None
            )
            effective_score = float(override) if override is not None else raw_score
            if metric in {
                "functional_consistency",
                "semantic_placement_consistency",
            }:
                weight = unit.planned_instance_count
                weight_basis = "planned_instance_count"
            else:
                weight = int(metric_report["evaluated_object_count"])
                weight_basis = "evaluated_object_count"
            weighted_sum += effective_score * weight
            weight_total += weight
            if metric in L1_METRICS:
                invalid_total += int(metric_report["invalid_count"])
            contributions.append(
                {
                    "room_id": room_id,
                    "raw_score": raw_score,
                    "effective_score": effective_score,
                    "weight": weight,
                    "weight_basis": weight_basis,
                    "functional_score_override": override,
                }
            )

        if metric in L1_METRICS:
            diagnostic_score = (
                1.0 - min(float(invalid_total) / float(weight_total), 1.0)
                if weight_total > 0
                else None
            )
        else:
            diagnostic_score = (
                weighted_sum / float(weight_total)
                if weight_total > 0
                else None
            )
        pre_program_coverage_score = None
        post_program_coverage_score = None
        pre_count_factor_score = None
        if metric == "functional_consistency" and diagnostic_score is not None:
            pre_program_coverage_score = diagnostic_score
            diagnostic_score *= float(
                execution.preflight.program_mapping[
                    "coverage_compliance"
                ]["factor"]
            )
            post_program_coverage_score = diagnostic_score
            pre_count_factor_score = diagnostic_score
            diagnostic_score *= float(
                execution.preflight.count_compliance["factor"]
            )
        complete = bool(complete_execution and diagnostic_score is not None)
        output[metric] = {
            "metric": metric,
            "status": "complete" if complete else "incomplete",
            "score": diagnostic_score if complete else None,
            "diagnostic_partial_score": (
                diagnostic_score if not complete else None
            ),
            "total_weight": weight_total,
            "invalid_count": invalid_total if metric in L1_METRICS else None,
            "pre_program_coverage_score": pre_program_coverage_score,
            "program_coverage_factor": (
                float(
                    execution.preflight.program_mapping[
                        "coverage_compliance"
                    ]["factor"]
                )
                if metric == "functional_consistency"
                else None
            ),
            "post_program_coverage_score": post_program_coverage_score,
            "pre_count_factor_score": pre_count_factor_score,
            "count_compliance_factor": (
                float(execution.preflight.count_compliance["factor"])
                if metric == "functional_consistency"
                else None
            ),
            "room_contributions": contributions,
        }
    return output


def _score_layers(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    scoring_profile: Mapping[str, Any] | None,
    execution_complete: bool,
) -> tuple[dict[str, dict[str, Any]], float | None, str]:
    empty_layers = {
        layer: {"score": None, "status": "unscored"}
        for layer in SCORING_LAYERS
    }
    if scoring_profile is None:
        return empty_layers, None, "profile_unconfigured"
    if not execution_complete or any(
        metrics[metric].get("score") is None for metric in REQUIRED_METRICS
    ):
        return empty_layers, None, "incomplete_coverage"

    metric_weights = scoring_profile["metric_weights"]
    l1_score = sum(
        float(metrics[metric]["score"])
        * float(metric_weights[L1_LAYER][metric])
        for metric in L1_METRICS
    )
    l3_score = sum(
        float(metrics[metric]["score"])
        * float(metric_weights[L3_LAYER][metric])
        for metric in L3_METRICS
    )
    layers = {
        L1_LAYER: {"score": l1_score, "status": "scored"},
        L3_LAYER: {"score": l3_score, "status": "scored"},
    }
    overall = (
        l1_score * float(scoring_profile["layer_weights"][L1_LAYER])
        + l3_score * float(scoring_profile["layer_weights"][L3_LAYER])
    )
    return layers, overall, "internally_scored_non_official"


def _empty_aggregate(
    *,
    scoring_profile: Mapping[str, Any] | None,
    reason: str,
    terminal_zero: bool = False,
) -> dict[str, Any]:
    return {
        "metrics": {
            metric: {
                "metric": metric,
                "status": "not_run",
                "score": None,
                "reason": reason,
            }
            for metric in REQUIRED_METRICS
        },
        "layers": {
            layer: {"score": None, "status": "not_run"}
            for layer in SCORING_LAYERS
        },
        "overall_score": 0.0 if terminal_zero else None,
        "terminal_case_score": 0.0 if terminal_zero else None,
        "scoring_status": (
            "terminal_zero_program_mapping_threshold"
            if terminal_zero
            else "not_run"
        ),
        "scoring_profile": deepcopy(scoring_profile),
        "official": False,
        "publishable": False,
    }


def _weight_mapping(
    value: Any,
    *,
    expected: tuple[str, ...],
    path: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise NonRectangularReportError(
            f"{path} must contain exactly {list(expected)!r}"
        )
    result: dict[str, float] = {}
    for name in expected:
        weight = value[name]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise NonRectangularReportError(
                f"{path}.{name} must be finite and non-negative"
            )
        result[name] = float(weight)
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1.0e-9):
        raise NonRectangularReportError(f"{path} weights must sum to 1.0")
    return result


__all__ = [
    "DEFAULT_NON_RECTANGULAR_SCORING_PROFILE",
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "L1_LAYER",
    "L3_LAYER",
    "NonRectangularReportError",
    "SCORING_PROFILE_SCHEMA_VERSION",
    "build_non_rectangular_evaluation_report",
    "validate_non_rectangular_scoring_profile",
]
