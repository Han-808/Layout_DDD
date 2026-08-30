"""Deterministic in-memory room projections for non-rectangular evaluation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from benchmark.non_rectangular.preflight import (
    NonRectangularPreflightResult,
)


ROOM_EVALUATION_UNIT_SCHEMA_VERSION = "non_rectangular_room_evaluation_unit_v1"


class RoomEvaluationUnitError(ValueError):
    """Raised when a preflight result cannot be projected into room units."""


@dataclass(frozen=True, slots=True)
class RoomEvaluationUnit:
    """One room's geometry, plan, generated objects, mapping, and provenance."""

    layout_id: str
    room_id: str
    room_index: int
    coordinate_frame: dict[str, Any]
    floor_polygon_xy: tuple[tuple[float, float], ...]
    wall_segments: tuple[dict[str, Any], ...]
    program_id: str | None
    room_type: str | None
    mapping_valid: bool
    mapping_failure_reasons: tuple[str, ...]
    functional_score_override: float | None
    prompt_granularity: str
    scene_description: str
    global_constraints: tuple[str, ...]
    zones: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    planned_objects: tuple[dict[str, Any], ...]
    generated_objects: tuple[dict[str, Any], ...]
    planned_instance_count: int
    generated_object_count: int
    artifact_sha256: dict[str, str]

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(str(item["id"]) for item in self.generated_objects)

    def public_dict(self) -> dict[str, Any]:
        """Return a serializable defensive view without coordinate transforms."""

        return {
            "schema_version": ROOM_EVALUATION_UNIT_SCHEMA_VERSION,
            "layout_id": self.layout_id,
            "room_id": self.room_id,
            "room_index": self.room_index,
            "coordinate_frame": deepcopy(self.coordinate_frame),
            "geometry": {
                "floor_z_m": 0.0,
                "floor_polygon_xy": [list(point) for point in self.floor_polygon_xy],
                "wall_segments": deepcopy(list(self.wall_segments)),
            },
            "program_mapping": {
                "program_id": self.program_id,
                "room_type": self.room_type,
                "valid": self.mapping_valid,
                "failure_reasons": list(self.mapping_failure_reasons),
                "functional_score_override": self.functional_score_override,
            },
            "object_plan": {
                "prompt_granularity": self.prompt_granularity,
                "scene_description": self.scene_description,
                "global_constraints": list(self.global_constraints),
                "zones": deepcopy(list(self.zones)),
                "relations": deepcopy(list(self.relations)),
                "objects": deepcopy(list(self.planned_objects)),
            },
            "generated_objects": deepcopy(list(self.generated_objects)),
            "planned_instance_count": self.planned_instance_count,
            "generated_object_count": self.generated_object_count,
            "object_ids": list(self.object_ids),
            "artifact_sha256": dict(self.artifact_sha256),
            "coordinates_transformed": False,
        }


def build_room_evaluation_units(
    preflight: NonRectangularPreflightResult,
) -> tuple[RoomEvaluationUnit, ...]:
    """Build one unit per room in the authoritative layout order."""

    if not isinstance(preflight, NonRectangularPreflightResult):
        raise RoomEvaluationUnitError(
            "preflight must be NonRectangularPreflightResult"
        )
    if not preflight.should_run_room_evaluation:
        raise RoomEvaluationUnitError(
            "failed preflight cases cannot build room evaluation units"
        )

    source = preflight.evaluation_input
    if source.generated_scene is None:
        raise RoomEvaluationUnitError(
            "ready preflight result must contain generated_scene"
        )
    layout_rooms = _rooms_by_id(source.room_layout["rooms"])
    plan_rooms = _rooms_by_id(source.object_plan["rooms"])
    scene_rooms = _rooms_by_id(source.generated_scene["rooms"])
    mapping_rooms = preflight.program_mapping["rooms"]
    plan_report = preflight.validation_reports["object_plan"]
    scene_report = preflight.validation_reports["generated_scene"]
    coordinate_frame = deepcopy(source.generated_scene["coordinate_frame"])

    units: list[RoomEvaluationUnit] = []
    for room_index, room_id in enumerate(preflight.room_order):
        geometry = layout_rooms[room_id]
        plan = plan_rooms[room_id]
        scene = scene_rooms[room_id]
        mapping = mapping_rooms[room_id]
        units.append(
            RoomEvaluationUnit(
                layout_id=preflight.layout_id,
                room_id=room_id,
                room_index=room_index,
                coordinate_frame=deepcopy(coordinate_frame),
                floor_polygon_xy=tuple(
                    (float(point[0]), float(point[1]))
                    for point in geometry["floor_polygon_xy"]
                ),
                wall_segments=tuple(
                    deepcopy(item) for item in geometry["wall_segments"]
                ),
                program_id=mapping["program_id"],
                room_type=mapping["room_type"],
                mapping_valid=bool(mapping["valid"]),
                mapping_failure_reasons=tuple(mapping["failure_reasons"]),
                functional_score_override=(
                    float(mapping["functional_score_override"])
                    if mapping["functional_score_override"] is not None
                    else None
                ),
                prompt_granularity=str(
                    source.object_plan.get(
                        "prompt_granularity",
                        "simplified",
                    )
                ),
                scene_description=str(
                    plan.get("scene_description")
                    or mapping.get("room_type")
                    or "unmapped room"
                ),
                global_constraints=tuple(
                    str(item)
                    for item in plan.get("global_constraints", [])
                ),
                zones=tuple(
                    deepcopy(item) for item in plan.get("zones", [])
                ),
                relations=tuple(
                    deepcopy(item) for item in plan.get("relations", [])
                ),
                planned_objects=tuple(
                    deepcopy(item) for item in plan["objects"]
                ),
                generated_objects=tuple(
                    deepcopy(item) for item in scene["objects"]
                ),
                planned_instance_count=int(
                    plan_report["room_instance_counts"][room_id]
                ),
                generated_object_count=int(
                    scene_report["room_object_counts"][room_id]
                ),
                artifact_sha256=dict(preflight.artifact_sha256),
            )
        )
    return tuple(units)


def _rooms_by_id(rooms: Any) -> dict[str, Mapping[str, Any]]:
    return {str(room["room_id"]): room for room in rooms}


__all__ = [
    "ROOM_EVALUATION_UNIT_SCHEMA_VERSION",
    "RoomEvaluationUnit",
    "RoomEvaluationUnitError",
    "build_room_evaluation_units",
]
