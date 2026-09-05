"""Lossless per-room projection into the canonical evaluator scene shape."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.evaluator.context_projection import (
    project_scene_for_evaluator_context,
)
from benchmark.evaluator.scene_quality.prompt_context import (
    METRIC_PROMPT_CONTEXT_VERSION,
)
from benchmark.non_rectangular.contracts import (
    NON_RECTANGULAR_EVALUATION_MODE,
)
from benchmark.non_rectangular.geometry import (
    POLYGON_ROOM_GEOMETRY_SCHEMA_VERSION,
    POLYGON_ROOM_METADATA_KEY,
)
from benchmark.non_rectangular.room_unit import RoomEvaluationUnit


ROOM_CANONICAL_PROJECTION_VERSION = "non_rectangular_room_canonical_projection_v1"


def project_room_unit_to_canonical_scene(
    unit: RoomEvaluationUnit,
) -> dict[str, Any]:
    """Project one authoritative room without recentering or plan leakage."""

    if not isinstance(unit, RoomEvaluationUnit):
        raise TypeError("unit must be RoomEvaluationUnit")
    wall_height = max(
        (float(wall["height_m"]) for wall in unit.wall_segments),
        default=2.8,
    )
    ceiling_z = float(unit.floor_z_m + wall_height)
    scene_id = f"{unit.layout_id}::{unit.room_id}"
    scene = {
        "schema_version": "canonical_scene_v1",
        "scene_id": scene_id,
        "request_id": scene_id,
        "scene_type": str(unit.room_type or "unmapped room"),
        "boundary": [list(point) for point in unit.floor_polygon_xy],
        "scene_height": ceiling_z,
        "objects": deepcopy(list(unit.generated_objects)),
        "metadata": {
            "evaluation_mode": NON_RECTANGULAR_EVALUATION_MODE,
            "projection_version": ROOM_CANONICAL_PROJECTION_VERSION,
            "room_scope": "current_room_objects_and_walls_only",
            "room_id": unit.room_id,
            "coordinates_transformed": False,
            POLYGON_ROOM_METADATA_KEY: {
                "schema_version": POLYGON_ROOM_GEOMETRY_SCHEMA_VERSION,
                "room_id": unit.room_id,
                "floor_polygon_xy": [
                    list(point) for point in unit.floor_polygon_xy
                ],
                "wall_segments": deepcopy(list(unit.wall_segments)),
                "floor_z_m": float(unit.floor_z_m),
                "ceiling_z_m": ceiling_z,
                "tolerance_m": 1.0e-6,
            },
        },
    }
    return project_scene_for_evaluator_context(scene)


def room_scene_quality_prompt_context(
    unit: RoomEvaluationUnit,
) -> dict[str, Any]:
    """Expose only the assigned room type to Functional/Placement scoring."""

    room_type = str(unit.room_type or "unmapped room")
    task_summary = f"The evaluated room is assigned the function: {room_type}."
    return {
        "schema_version": METRIC_PROMPT_CONTEXT_VERSION,
        "values": {"task_summary": task_summary},
        "metric_fields": {
            "scale_consistency": ["room_type"],
            "style_consistency": ["room_type"],
            "object_pairing_consistency": ["room_type"],
            "functional_consistency": ["room_type", "task_summary"],
            "semantic_placement_consistency": [
                "room_type",
                "task_summary",
            ],
        },
        "metric_instructions": {
            "functional_consistency": (
                "Assess whether the visible objects and their arrangement let "
                "this room serve its assigned function."
            ),
            "semantic_placement_consistency": (
                "Assess placement plausibility within this assigned room type; "
                "do not re-score collision, OOB, or support."
            ),
        },
        "source": "benchmark_room_program_mapping",
    }


__all__ = [
    "ROOM_CANONICAL_PROJECTION_VERSION",
    "project_room_unit_to_canonical_scene",
    "room_scene_quality_prompt_context",
]
