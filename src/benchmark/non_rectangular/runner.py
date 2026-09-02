"""Internal-only composed entry point for the non-rectangular framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from benchmark.non_rectangular.evaluator import (
    CanonicalNonRectangularRoomEvaluator,
)

from benchmark.non_rectangular.preflight import (
    NonRectangularEvaluationInput,
    prepare_non_rectangular_evaluation,
)
from benchmark.non_rectangular.report import (
    DEFAULT_NON_RECTANGULAR_SCORING_PROFILE,
    build_non_rectangular_evaluation_report,
)
from benchmark.non_rectangular.workflow import (
    RoomEvaluator,
    execute_non_rectangular_workflow,
)
from benchmark.utils.io import write_json


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


def run_non_rectangular_evaluation(
    evaluation_input: NonRectangularEvaluationInput,
    *,
    out: str | Path,
    room_evaluator: RoomEvaluator | None = None,
    scoring_profile: Mapping[str, Any] | None = None,
    evaluator_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the explicitly selected public non-rectangular workflow."""

    out_path = Path(out).expanduser().resolve()
    report_path = (
        out_path
        if out_path.suffix.lower() == ".json"
        else out_path / "evaluation_report.json"
    )
    output_root = report_path.parent
    evaluator = room_evaluator
    if evaluator is None:
        kwargs = dict(evaluator_kwargs or {})
        kwargs.setdefault("output_root", output_root)
        evaluator = CanonicalNonRectangularRoomEvaluator(**kwargs)
    preflight = prepare_non_rectangular_evaluation(evaluation_input)
    execution = execute_non_rectangular_workflow(
        preflight,
        room_evaluator=evaluator,
    )
    report = build_non_rectangular_evaluation_report(
        execution,
        scoring_profile=(
            scoring_profile
            if scoring_profile is not None
            else DEFAULT_NON_RECTANGULAR_SCORING_PROFILE
        ),
        public_route_connected=True,
    )
    write_json(report_path, report)
    return report


__all__ = [
    "run_internal_non_rectangular_evaluation",
    "run_non_rectangular_evaluation",
]
