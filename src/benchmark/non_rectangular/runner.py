"""Internal-only composed entry point for the non-rectangular framework."""

from __future__ import annotations

from typing import Any, Mapping

from benchmark.non_rectangular.preflight import (
    NonRectangularEvaluationInput,
    prepare_non_rectangular_evaluation,
)
from benchmark.non_rectangular.report import (
    build_non_rectangular_evaluation_report,
)
from benchmark.non_rectangular.workflow import (
    RoomEvaluator,
    execute_non_rectangular_workflow,
)


def run_internal_non_rectangular_evaluation(
    evaluation_input: NonRectangularEvaluationInput,
    *,
    room_evaluator: RoomEvaluator,
    scoring_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose preflight, room execution, and reporting without public routing."""

    preflight = prepare_non_rectangular_evaluation(evaluation_input)
    execution = execute_non_rectangular_workflow(
        preflight,
        room_evaluator=room_evaluator,
    )
    return build_non_rectangular_evaluation_report(
        execution,
        scoring_profile=scoring_profile,
    )


__all__ = ["run_internal_non_rectangular_evaluation"]
