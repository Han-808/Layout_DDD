"""Strict geometry validation for the additive room-layout truth contract."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from shapely.geometry import LinearRing, Polygon

from benchmark.resources import runtime_resource_path


ROOM_LAYOUT_SCHEMA_VERSION = "non_rectangular_room_layout_v1"
ROOM_LAYOUT_SCHEMA_PATH = runtime_resource_path(
    "schemas/non_rectangular/room_layout_v1.schema.json"
)
_NORMAL_TOLERANCE = 1.0e-6


class RoomLayoutValidationError(ValueError):
    """Raised when room-layout geometry fails the additive contract."""


def validate_room_layout(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema plus polygon/wall-loop geometry without mutation."""

    if not isinstance(value, Mapping):
        raise RoomLayoutValidationError("room layout must be a JSON object")
    _validate_schema(value)

    tolerance = _finite_number(
        value["geometry_tolerance_m"],
        "geometry_tolerance_m",
    )
    rooms = value["rooms"]
    room_ids = [str(room["room_id"]) for room in rooms]
    if int(value["room_count"]) != len(rooms):
        raise RoomLayoutValidationError(
            "room_count must exactly match rooms"
        )
    if list(value["room_order"]) != room_ids:
        raise RoomLayoutValidationError(
            "room_order must exactly match rooms serialization order"
        )
    if len(room_ids) != len(set(room_ids)):
        raise RoomLayoutValidationError("room IDs must be unique")

    polygons: list[tuple[str, Polygon]] = []
    wall_ids: set[str] = set()
    wall_count = 0
    for room_index, room in enumerate(rooms):
        room_id = str(room["room_id"])
        path = f"rooms[{room_index}]"
        _finite_number(room["floor_z_m"], f"{path}.floor_z_m")

        points = [
            _point(point, f"{path}.floor_polygon_xy[{index}]")
            for index, point in enumerate(room["floor_polygon_xy"])
        ]
        _validate_distinct_vertices(points, tolerance=tolerance, path=path)
        ring = LinearRing(points)
        polygon = Polygon(points)
        if not ring.is_simple or not polygon.is_valid:
            raise RoomLayoutValidationError(
                f"{path}.floor_polygon_xy must be a valid simple polygon"
            )
        if not ring.is_ccw:
            raise RoomLayoutValidationError(
                f"{path}.floor_polygon_xy must use counter-clockwise winding"
            )
        if float(polygon.area) <= tolerance * tolerance:
            raise RoomLayoutValidationError(
                f"{path}.floor_polygon_xy must have positive area"
            )

        walls = room["wall_segments"]
        if len(walls) != len(points):
            raise RoomLayoutValidationError(
                f"{path}.wall_segments must match polygon edge count"
            )
        for wall_index, wall in enumerate(walls):
            wall_path = f"{path}.wall_segments[{wall_index}]"
            wall_id = str(wall["wall_id"])
            if not wall_id.startswith(f"{room_id}."):
                raise RoomLayoutValidationError(
                    f"{wall_path}.wall_id must be scoped by {room_id!r}"
                )
            if wall_id in wall_ids:
                raise RoomLayoutValidationError(
                    f"duplicate scene-global wall ID: {wall_id}"
                )
            wall_ids.add(wall_id)

            expected_start = points[wall_index]
            expected_end = points[(wall_index + 1) % len(points)]
            actual_start = _point(wall["start_xy"], f"{wall_path}.start_xy")
            actual_end = _point(wall["end_xy"], f"{wall_path}.end_xy")
            _require_close(
                actual_start,
                expected_start,
                tolerance=tolerance,
                path=f"{wall_path}.start_xy",
            )
            _require_close(
                actual_end,
                expected_end,
                tolerance=tolerance,
                path=f"{wall_path}.end_xy",
            )
            edge = (
                expected_end[0] - expected_start[0],
                expected_end[1] - expected_start[1],
            )
            length = math.hypot(*edge)
            if length <= tolerance:
                raise RoomLayoutValidationError(
                    f"{wall_path} length must exceed geometry tolerance"
                )
            expected_normal = (-edge[1] / length, edge[0] / length)
            actual_normal = _point(
                wall["inward_normal_xy"],
                f"{wall_path}.inward_normal_xy",
            )
            _require_close(
                actual_normal,
                expected_normal,
                tolerance=_NORMAL_TOLERANCE,
                path=f"{wall_path}.inward_normal_xy",
            )
            _finite_number(wall["height_m"], f"{wall_path}.height_m")
            _finite_number(
                wall["thickness_m"],
                f"{wall_path}.thickness_m",
            )
        wall_count += len(walls)
        polygons.append((room_id, polygon))

    for left_index, (left_id, left) in enumerate(polygons):
        for right_id, right in polygons[left_index + 1 :]:
            overlap_area = float(left.intersection(right).area)
            if overlap_area > tolerance * tolerance:
                raise RoomLayoutValidationError(
                    "room polygon interiors must not overlap: "
                    f"{left_id!r} vs {right_id!r} area={overlap_area}"
                )

    return {
        "schema_version": ROOM_LAYOUT_SCHEMA_VERSION,
        "layout_id": str(value["layout_id"]),
        "valid": True,
        "room_count": len(rooms),
        "room_ids": room_ids,
        "wall_segment_count": wall_count,
        "geometry_tolerance_m": tolerance,
        "total_floor_area_m2": sum(
            float(polygon.area) for _, polygon in polygons
        ),
        "coordinate_frame_shared": True,
        "room_interiors_disjoint": True,
    }


def _validate_schema(value: Mapping[str, Any]) -> None:
    try:
        schema = json.loads(ROOM_LAYOUT_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RoomLayoutValidationError(
            "cannot load packaged room-layout schema"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise RoomLayoutValidationError(
        f"room-layout schema validation failed at {path}: {error.message}"
    )


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoomLayoutValidationError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RoomLayoutValidationError(f"{path} must be finite")
    return result


def _point(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RoomLayoutValidationError(f"{path} must be [x, y]")
    return (
        _finite_number(value[0], f"{path}[0]"),
        _finite_number(value[1], f"{path}[1]"),
    )


def _validate_distinct_vertices(
    points: list[tuple[float, float]],
    *,
    tolerance: float,
    path: str,
) -> None:
    for left_index, left in enumerate(points):
        for right_index, right in enumerate(points[left_index + 1 :], start=left_index + 1):
            if math.dist(left, right) <= tolerance:
                raise RoomLayoutValidationError(
                    f"{path}.floor_polygon_xy vertices {left_index} and "
                    f"{right_index} are duplicated within tolerance"
                )


def _require_close(
    actual: tuple[float, float],
    expected: tuple[float, float],
    *,
    tolerance: float,
    path: str,
) -> None:
    if math.dist(actual, expected) > tolerance:
        raise RoomLayoutValidationError(
            f"{path} differs from ordered polygon geometry: "
            f"expected={expected!r}, actual={actual!r}"
        )


__all__ = [
    "ROOM_LAYOUT_SCHEMA_PATH",
    "ROOM_LAYOUT_SCHEMA_VERSION",
    "RoomLayoutValidationError",
    "validate_room_layout",
]
