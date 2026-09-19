"""Polygon geometry adapters and the non-rectangular input contracts.

Every public name is re-exported lazily. Importing the historical evaluation
pipeline eagerly from this package cycles through :mod:`benchmark.evaluator`,
because the shared judge imports a submodule of this package at module level
while ``benchmark.evaluator`` is still initialising. Resolving on first
attribute access keeps both import orders working.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "COUNT_COMPLIANCE_FAILURE_THRESHOLD": "preflight",
    "CanonicalNonRectangularRoomEvaluator": "evaluator",
    "DEFAULT_NON_RECTANGULAR_SCORING_PROFILE": "report",
    "EVALUATION_REPORT_SCHEMA_VERSION": "report",
    "L1_LAYER": "report",
    "L1_METRICS": "workflow",
    "L3_LAYER": "report",
    "L3_METRICS": "workflow",
    "MULTI_ROOM_SCENE_SCHEMA_VERSION": "contracts",
    "NON_RECTANGULAR_EVALUATION_MODE": "contracts",
    "NON_RECTANGULAR_ROOM_EVALUATOR_VERSION": "evaluator",
    "NonRectangularContractError": "contracts",
    "NonRectangularEvaluationInput": "preflight",
    "NonRectangularPreflightError": "preflight",
    "NonRectangularPreflightResult": "preflight",
    "NonRectangularReportError": "report",
    "NonRectangularRoomMetricIncomplete": "evaluator",
    "NonRectangularWorkflowExecution": "workflow",
    "OBJECT_PLAN_SCHEMA_VERSION": "contracts",
    "OBJECT_PLAN_V2_SCHEMA_VERSION": "contracts",
    "POLYGON_OOB_EVALUATOR_VERSION": "oob",
    "POLYGON_ROOM_GEOMETRY_SCHEMA_VERSION": "geometry",
    "POLYGON_ROOM_METADATA_KEY": "geometry",
    "PROGRAM_COVERAGE_POLICY": "preflight",
    "PolygonOOBEvaluationError": "oob",
    "PolygonRoomGeometry": "geometry",
    "PolygonRoomGeometryError": "geometry",
    "REQUIRED_METRICS": "workflow",
    "ROOM_CANONICAL_PROJECTION_VERSION": "projection",
    "ROOM_EVALUATION_UNIT_SCHEMA_VERSION": "room_unit",
    "ROOM_LAYOUT_SCHEMA_VERSION": "room_layout",
    "ROOM_METRIC_EXECUTION_ORDER": "workflow",
    "ROOM_PROGRAM_SCHEMA_VERSION": "contracts",
    "ROOM_REPORT_SCHEMA_VERSION": "workflow",
    "RoomEvaluationInfrastructureFailure": "workflow",
    "RoomEvaluationUnit": "room_unit",
    "RoomEvaluationUnitError": "room_unit",
    "RoomEvaluator": "workflow",
    "RoomEvaluatorReportError": "workflow",
    "RoomLayoutValidationError": "room_layout",
    "SCORING_PROFILE_SCHEMA_VERSION": "report",
    "build_non_rectangular_evaluation_report": "report",
    "build_room_evaluation_units": "room_unit",
    "check_polygon_oob": "oob",
    "execute_non_rectangular_workflow": "workflow",
    "object_count_compliance": "preflight",
    "polygon_geometry_from_scene": "geometry",
    "prepare_non_rectangular_evaluation": "preflight",
    "program_coverage_compliance": "preflight",
    "program_mapping_report": "preflight",
    "project_room_unit_to_canonical_scene": "projection",
    "room_scene_quality_prompt_context": "projection",
    "run_internal_non_rectangular_evaluation": "runner",
    "run_non_rectangular_evaluation": "runner",
    "validate_complete_room_report": "workflow",
    "validate_multi_room_object_plan": "contracts",
    "validate_multi_room_scene": "contracts",
    "validate_non_rectangular_scoring_profile": "report",
    "validate_room_layout": "room_layout",
    "validate_room_program": "contracts",
}


def __getattr__(name: str) -> Any:
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = [
    "COUNT_COMPLIANCE_FAILURE_THRESHOLD",
    "CanonicalNonRectangularRoomEvaluator",
    "DEFAULT_NON_RECTANGULAR_SCORING_PROFILE",
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "L1_LAYER",
    "L1_METRICS",
    "L3_LAYER",
    "L3_METRICS",
    "MULTI_ROOM_SCENE_SCHEMA_VERSION",
    "NON_RECTANGULAR_EVALUATION_MODE",
    "NON_RECTANGULAR_ROOM_EVALUATOR_VERSION",
    "NonRectangularContractError",
    "NonRectangularEvaluationInput",
    "NonRectangularPreflightError",
    "NonRectangularPreflightResult",
    "NonRectangularReportError",
    "NonRectangularRoomMetricIncomplete",
    "NonRectangularWorkflowExecution",
    "OBJECT_PLAN_SCHEMA_VERSION",
    "OBJECT_PLAN_V2_SCHEMA_VERSION",
    "POLYGON_OOB_EVALUATOR_VERSION",
    "POLYGON_ROOM_GEOMETRY_SCHEMA_VERSION",
    "POLYGON_ROOM_METADATA_KEY",
    "PROGRAM_COVERAGE_POLICY",
    "PolygonOOBEvaluationError",
    "PolygonRoomGeometry",
    "PolygonRoomGeometryError",
    "REQUIRED_METRICS",
    "ROOM_CANONICAL_PROJECTION_VERSION",
    "ROOM_EVALUATION_UNIT_SCHEMA_VERSION",
    "ROOM_LAYOUT_SCHEMA_VERSION",
    "ROOM_METRIC_EXECUTION_ORDER",
    "ROOM_PROGRAM_SCHEMA_VERSION",
    "ROOM_REPORT_SCHEMA_VERSION",
    "RoomEvaluationInfrastructureFailure",
    "RoomEvaluationUnit",
    "RoomEvaluationUnitError",
    "RoomEvaluator",
    "RoomEvaluatorReportError",
    "RoomLayoutValidationError",
    "SCORING_PROFILE_SCHEMA_VERSION",
    "build_non_rectangular_evaluation_report",
    "build_room_evaluation_units",
    "check_polygon_oob",
    "execute_non_rectangular_workflow",
    "object_count_compliance",
    "polygon_geometry_from_scene",
    "prepare_non_rectangular_evaluation",
    "program_coverage_compliance",
    "program_mapping_report",
    "project_room_unit_to_canonical_scene",
    "room_scene_quality_prompt_context",
    "run_internal_non_rectangular_evaluation",
    "run_non_rectangular_evaluation",
    "validate_complete_room_report",
    "validate_multi_room_object_plan",
    "validate_multi_room_scene",
    "validate_non_rectangular_scoring_profile",
    "validate_room_layout",
    "validate_room_program",
]
