"""Deterministic polygon floor/wall inventory with exact shared-wall dedup."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from benchmark.non_rectangular import validate_room_layout
from benchmark.resources import runtime_resource_path


COMPILED_ARCHITECTURE_SCHEMA_VERSION = (
    "non_rectangular_compiled_architecture_v1"
)
COMPILED_ARCHITECTURE_SCHEMA_PATH = runtime_resource_path(
    "schemas/non_rectangular/compiled_architecture_v1.schema.json"
)


class NonRectangularArchitectureError(ValueError):
    """Raised when logical walls cannot form an exact physical inventory."""


def build_polygon_architecture(
    room_layout: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile floors and exact coincident walls without geometry repair."""

    report = validate_room_layout(room_layout)
    room_order = [str(item) for item in report["room_ids"]]
    rooms_by_id = {
        str(room["room_id"]): room for room in room_layout["rooms"]
    }
    logical_walls: list[dict[str, Any]] = []
    grouped: dict[
        tuple[tuple[float, float], tuple[float, float]],
        dict[str, Any],
    ] = {}
    floors: list[dict[str, Any]] = []
    rooms: list[dict[str, Any]] = []

    for room_id in room_order:
        room = rooms_by_id[room_id]
        floor_id = f"{room_id}.floor"
        wall_ids: list[str] = []
        floors.append(
            {
                "floor_id": floor_id,
                "room_id": room_id,
                "floor_z_m": 0.0,
                "polygon_xy": deepcopy(room["floor_polygon_xy"]),
            }
        )
        for wall in room["wall_segments"]:
            wall_id = str(wall["wall_id"])
            wall_ids.append(wall_id)
            segment = [
                [float(item) for item in wall["start_xy"]],
                [float(item) for item in wall["end_xy"]],
            ]
            logical_walls.append(
                {
                    "wall_id": wall_id,
                    "room_id": room_id,
                    "segment_global_m": segment,
                    "inward_normal_xy": [
                        float(item) for item in wall["inward_normal_xy"]
                    ],
                    "height_m": float(wall["height_m"]),
                    "thickness_m": float(wall["thickness_m"]),
                }
            )
            key = _segment_key(segment)
            entry = grouped.get(key)
            endpoint = {"room_id": room_id, "wall_id": wall_id}
            normal = tuple(float(item) for item in wall["inward_normal_xy"])
            if entry is None:
                grouped[key] = {
                    "segment": [list(key[0]), list(key[1])],
                    "height_m": float(wall["height_m"]),
                    "thickness_m": float(wall["thickness_m"]),
                    "logical_endpoints": [endpoint],
                    "normals": [normal],
                }
            else:
                if not math.isclose(
                    float(entry["height_m"]),
                    float(wall["height_m"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ) or not math.isclose(
                    float(entry["thickness_m"]),
                    float(wall["thickness_m"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise NonRectangularArchitectureError(
                        "coincident logical walls must share height/thickness"
                    )
                if len(entry["logical_endpoints"]) >= 2:
                    raise NonRectangularArchitectureError(
                        "one physical wall may belong to at most two rooms"
                    )
                if entry["logical_endpoints"][0]["room_id"] == room_id:
                    raise NonRectangularArchitectureError(
                        "duplicate coincident logical walls within one room"
                    )
                prior_normal = entry["normals"][0]
                if not all(
                    math.isclose(
                        prior_normal[index],
                        -normal[index],
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                    for index in range(2)
                ):
                    raise NonRectangularArchitectureError(
                        "shared wall inward normals must oppose"
                    )
                entry["logical_endpoints"].append(endpoint)
                entry["normals"].append(normal)
        rooms.append(
            {
                "room_id": room_id,
                "floor_id": floor_id,
                "logical_wall_ids": wall_ids,
            }
        )

    physical_walls: list[dict[str, Any]] = []
    for index, entry in enumerate(grouped.values()):
        endpoints = deepcopy(entry["logical_endpoints"])
        physical_walls.append(
            {
                "physical_wall_id": f"physical_wall_{index:04d}",
                "segment_global_m": deepcopy(entry["segment"]),
                "height_m": float(entry["height_m"]),
                "thickness_m": float(entry["thickness_m"]),
                "shared": len(endpoints) == 2,
                "logical_endpoints": endpoints,
            }
        )

    artifact = {
        "schema_version": COMPILED_ARCHITECTURE_SCHEMA_VERSION,
        "layout_id": str(room_layout["layout_id"]),
        "coordinate_frame": deepcopy(room_layout["coordinate_frame"]),
        "room_order": room_order,
        "rooms": rooms,
        "floors": floors,
        "logical_walls": logical_walls,
        "physical_walls": physical_walls,
        "excluded_architecture": ["ceiling", "doors", "windows"],
    }
    _validate_schema(artifact)
    return artifact


def _segment_key(
    segment: list[list[float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    start = (float(segment[0][0]), float(segment[0][1]))
    end = (float(segment[1][0]), float(segment[1][1]))
    return (start, end) if start <= end else (end, start)


def _validate_schema(value: Mapping[str, Any]) -> None:
    try:
        schema = json.loads(
            COMPILED_ARCHITECTURE_SCHEMA_PATH.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NonRectangularArchitectureError(
            "cannot load packaged compiled-architecture schema"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise NonRectangularArchitectureError(
            f"compiled-architecture schema failed at {path}: {error.message}"
        )


__all__ = [
    "COMPILED_ARCHITECTURE_SCHEMA_PATH",
    "COMPILED_ARCHITECTURE_SCHEMA_VERSION",
    "NonRectangularArchitectureError",
    "build_polygon_architecture",
]
