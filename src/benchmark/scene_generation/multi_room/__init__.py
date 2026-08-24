"""Additive multi-room generation compatibility mode.

The package composes the existing campaign transport, deterministic retrieval,
and frozen two-stage model calls.  It never imports or invokes evaluation code.
"""

from benchmark.scene_generation.multi_room.floor_plan import (
    FloorPlanValidationError,
    LoadedFloorPlan,
    load_floor_plan,
    validate_floor_plan,
)


GENERATION_MODE = "multi_room_with_architecture_v1"
WORKFLOW_PROFILE_ID = "frozen-two-stage-multi-room-with-architecture-v1"


__all__ = [
    "FloorPlanValidationError",
    "GENERATION_MODE",
    "LoadedFloorPlan",
    "WORKFLOW_PROFILE_ID",
    "load_floor_plan",
    "validate_floor_plan",
]
