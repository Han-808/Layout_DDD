"""Polygon geometry adapters for the shared evaluator.

The historical non-rectangular evaluator and runtime are not re-exported here.
New model rooms run through camera_cal_scene_level.uniform; the remaining
historical modules are imported directly by the few callers that still need
them.

Only the generation-facing input contracts are re-exported, because eagerly
importing the historical evaluation pipeline from this package would cycle
through :mod:`benchmark.evaluator` while that package is still initialising.
"""

from benchmark.non_rectangular.contracts import (
    NON_RECTANGULAR_EVALUATION_MODE,
    NonRectangularContractError,
    validate_room_program,
)
from benchmark.non_rectangular.room_layout import (
    ROOM_LAYOUT_SCHEMA_VERSION,
    RoomLayoutValidationError,
    validate_room_layout,
)
from benchmark.non_rectangular.preflight import (
    NonRectangularEvaluationInput,
    NonRectangularPreflightError,
    prepare_non_rectangular_evaluation,
)


__all__ = [
    "NON_RECTANGULAR_EVALUATION_MODE",
    "NonRectangularContractError",
    "NonRectangularEvaluationInput",
    "NonRectangularPreflightError",
    "ROOM_LAYOUT_SCHEMA_VERSION",
    "RoomLayoutValidationError",
    "prepare_non_rectangular_evaluation",
    "validate_room_layout",
    "validate_room_program",
]
