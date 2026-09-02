"""Deterministic polygon geometry for the additive non-rectangular evaluator.

This module is deliberately isolated from the canonical rectangular geometry
helpers.  Callers must opt in by carrying the versioned room metadata emitted
by :func:`project_room_unit_to_canonical_scene`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPoint,
    MultiLineString,
    Point,
    Polygon,
)
from shapely.ops import nearest_points, polylabel

from benchmark.evaluator.generic_validity.geometry import get_obb_corners
from benchmark.non_rectangular.contracts import (
    NON_RECTANGULAR_EVALUATION_MODE,
)
from benchmark.scene_io.object_normalization import normalize_object


POLYGON_ROOM_GEOMETRY_SCHEMA_VERSION = "non_rectangular_polygon_room_geometry_v1"
POLYGON_ROOM_METADATA_KEY = "non_rectangular_room_geometry"
DEFAULT_CAMERA_WALL_CLEARANCE_M = 0.08
DEFAULT_CAMERA_VERTICAL_CLEARANCE_M = 0.10


class PolygonRoomGeometryError(ValueError):
    """Raised when an explicitly selected polygon room is unusable."""


@dataclass(frozen=True, slots=True)
class PolygonWall:
    wall_id: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    inward_normal_xy: tuple[float, float]
    height_m: float
    thickness_m: float

    @property
    def tangent_xy(self) -> tuple[float, float]:
        dx = self.end_xy[0] - self.start_xy[0]
        dy = self.end_xy[1] - self.start_xy[1]
        length = math.hypot(dx, dy)
        return (dx / length, dy / length)

    @property
    def line(self) -> LineString:
        return LineString([self.start_xy, self.end_xy])

    def public_dict(self) -> dict[str, Any]:
        return {
            "wall_id": self.wall_id,
            "start_xy": list(self.start_xy),
            "end_xy": list(self.end_xy),
            "inward_normal_xy": list(self.inward_normal_xy),
            "tangent_xy": list(self.tangent_xy),
            "height_m": self.height_m,
            "thickness_m": self.thickness_m,
        }


@dataclass(frozen=True, slots=True)
class PolygonRoomGeometry:
    room_id: str
    floor_polygon_xy: tuple[tuple[float, float], ...]
    walls: tuple[PolygonWall, ...]
    floor_z_m: float
    ceiling_z_m: float
    tolerance_m: float = 1.0e-6

    @classmethod
    def from_metadata(
        cls,
        value: Mapping[str, Any],
    ) -> "PolygonRoomGeometry":
        if not isinstance(value, Mapping):
            raise PolygonRoomGeometryError(
                "non-rectangular room geometry metadata must be an object"
            )
        if value.get("schema_version") != POLYGON_ROOM_GEOMETRY_SCHEMA_VERSION:
            raise PolygonRoomGeometryError(
                "unsupported non-rectangular room geometry version"
            )
        points = tuple(_point(item, "floor_polygon_xy") for item in value.get("floor_polygon_xy", ()))
        polygon = Polygon(points)
        if len(points) < 3 or not polygon.is_valid or polygon.is_empty:
            raise PolygonRoomGeometryError("floor_polygon_xy must be a valid polygon")
        floor_z = _finite(value.get("floor_z_m"), "floor_z_m")
        ceiling_z = _finite(value.get("ceiling_z_m"), "ceiling_z_m")
        if ceiling_z <= floor_z:
            raise PolygonRoomGeometryError("ceiling_z_m must exceed floor_z_m")
        tolerance = _finite(value.get("tolerance_m", 1.0e-6), "tolerance_m")
        if tolerance <= 0.0:
            raise PolygonRoomGeometryError("tolerance_m must be positive")
        raw_walls = value.get("wall_segments")
        if not isinstance(raw_walls, list) or len(raw_walls) != len(points):
            raise PolygonRoomGeometryError(
                "wall_segments must exactly cover polygon edges"
            )
        walls: list[PolygonWall] = []
        for index, raw in enumerate(raw_walls):
            if not isinstance(raw, Mapping):
                raise PolygonRoomGeometryError(
                    f"wall_segments[{index}] must be an object"
                )
            start = _point(raw.get("start_xy"), f"wall_segments[{index}].start_xy")
            end = _point(raw.get("end_xy"), f"wall_segments[{index}].end_xy")
            normal = _point(
                raw.get("inward_normal_xy"),
                f"wall_segments[{index}].inward_normal_xy",
            )
            normal_length = math.hypot(*normal)
            if normal_length <= tolerance:
                raise PolygonRoomGeometryError("wall inward normal cannot be zero")
            walls.append(
                PolygonWall(
                    wall_id=str(raw.get("wall_id") or "").strip(),
                    start_xy=start,
                    end_xy=end,
                    inward_normal_xy=(
                        normal[0] / normal_length,
                        normal[1] / normal_length,
                    ),
                    height_m=_finite(raw.get("height_m"), "wall height"),
                    thickness_m=_finite(raw.get("thickness_m"), "wall thickness"),
                )
            )
        if any(not wall.wall_id for wall in walls):
            raise PolygonRoomGeometryError("wall_id is required")
        return cls(
            room_id=str(value.get("room_id") or "").strip(),
            floor_polygon_xy=points,
            walls=tuple(walls),
            floor_z_m=floor_z,
            ceiling_z_m=ceiling_z,
            tolerance_m=tolerance,
        )

    @property
    def polygon(self) -> Polygon:
        return Polygon(self.floor_polygon_xy)

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        minimum_x, minimum_y, maximum_x, maximum_y = self.polygon.bounds
        return (
            float(minimum_x),
            float(maximum_x),
            float(minimum_y),
            float(maximum_y),
        )

    @property
    def span_m(self) -> float:
        min_x, max_x, min_y, max_y = self.bounds_xy
        return max(max_x - min_x, max_y - min_y)

    @property
    def camera_max_z_m(self) -> float:
        """Bounded camera-search ceiling; the benchmark has no room ceiling."""

        return self.ceiling_z_m + max(5.0, self.span_m * 1.5)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLYGON_ROOM_GEOMETRY_SCHEMA_VERSION,
            "room_id": self.room_id,
            "floor_polygon_xy": [list(point) for point in self.floor_polygon_xy],
            "wall_segments": [wall.public_dict() for wall in self.walls],
            "floor_z_m": self.floor_z_m,
            "ceiling_z_m": self.ceiling_z_m,
            "camera_max_z_m": self.camera_max_z_m,
            "camera_vertical_limit_policy": "no_ceiling_bounded_search_v1",
            "tolerance_m": self.tolerance_m,
        }

    def representative_interior_xy(self) -> tuple[float, float]:
        polygon = self.polygon
        centroid = polygon.centroid
        if polygon.covers(centroid):
            candidate = polylabel(
                polygon,
                tolerance=max(self.tolerance_m, 0.01),
            )
        else:
            candidate = polygon.representative_point()
        return (float(candidate.x), float(candidate.y))

    def contains_xy(
        self,
        point_xy: Iterable[float],
        *,
        wall_clearance_m: float = 0.0,
    ) -> bool:
        point = _point(point_xy, "point_xy")
        region, _ = self._camera_region(wall_clearance_m)
        return bool(region.covers(Point(point)))

    def coerce_xy_inside(
        self,
        point_xy: Iterable[float],
        *,
        wall_clearance_m: float = 0.0,
    ) -> tuple[float, float]:
        point = Point(_point(point_xy, "point_xy"))
        region, _ = self._camera_region(wall_clearance_m)
        if region.covers(point):
            return (float(point.x), float(point.y))
        resolved, _ = nearest_points(region, point)
        return (float(resolved.x), float(resolved.y))

    def distance_to_wall(self, point_xy: Iterable[float]) -> float:
        return float(self.polygon.boundary.distance(Point(_point(point_xy, "point_xy"))))

    def segment_visible_inside_room(
        self,
        start_xy: Iterable[float],
        end_xy: Iterable[float],
    ) -> bool:
        line = LineString(
            [_point(start_xy, "start_xy"), _point(end_xy, "end_xy")]
        )
        return bool(self.polygon.buffer(self.tolerance_m).covers(line))

    def place_on_feasible_ray(
        self,
        *,
        target: Iterable[float],
        direction: Iterable[float],
        desired_distance_m: float,
        wall_clearance_m: float = DEFAULT_CAMERA_WALL_CLEARANCE_M,
        vertical_clearance_m: float = DEFAULT_CAMERA_VERTICAL_CLEARANCE_M,
        minimum_distance_m: float = 0.25,
        relax_wall_clearance: bool = True,
    ) -> tuple[np.ndarray, float, dict[str, Any]] | None:
        target_vector = _vector3(target, "target")
        ray = _vector3(direction, "direction")
        norm = float(np.linalg.norm(ray))
        if norm <= 1.0e-9:
            raise PolygonRoomGeometryError("camera direction cannot be zero")
        ray /= norm
        desired = _positive(desired_distance_m, "desired_distance_m")
        minimum = _positive(minimum_distance_m, "minimum_distance_m")
        region, actual_wall_clearance = self._camera_region(
            wall_clearance_m,
            allow_relaxation=relax_wall_clearance,
        )
        maximum_trace = max(20.0, self.span_m * 4.0, desired * 2.0 + 2.0)
        xy_intervals = self._xy_ray_intervals(
            region,
            target_vector=target_vector,
            direction=ray,
            maximum_trace=maximum_trace,
        )
        z_interval = _z_ray_interval(
            target_z=float(target_vector[2]),
            direction_z=float(ray[2]),
            minimum_z=self.floor_z_m + max(0.0, float(vertical_clearance_m)),
            maximum_z=self.camera_max_z_m,
            maximum_trace=maximum_trace,
        )
        if z_interval is None:
            return None
        feasible: list[tuple[float, float]] = []
        for lower, upper in xy_intervals:
            combined_lower = max(lower, z_interval[0], minimum)
            combined_upper = min(upper, z_interval[1], maximum_trace)
            if combined_upper + self.tolerance_m < combined_lower:
                continue
            feasible.append((combined_lower, combined_upper))
        choices: list[tuple[float, float, float]] = []
        for lower, upper in feasible:
            distance = min(max(desired, lower), upper)
            location = target_vector + ray * distance
            if not self.segment_visible_inside_room(location[:2], target_vector[:2]):
                continue
            choices.append((abs(distance - desired), -distance, distance))
        if not choices:
            return None
        _, _, distance = min(choices)
        location = target_vector + ray * distance
        audit = {
            "method": "polygon_ray_interval_with_los_v1",
            "room_id": self.room_id,
            "requested_distance_m": desired,
            "actual_distance_m": float(distance),
            "distance_truncated": abs(float(distance) - desired) > 1.0e-6,
            "wall_clearance_requested_m": float(wall_clearance_m),
            "wall_clearance_applied_m": actual_wall_clearance,
            "wall_clearance_relaxation_allowed": bool(
                relax_wall_clearance
            ),
            "vertical_clearance_m": float(vertical_clearance_m),
            "point_inside_polygon": True,
            "room_wall_los_valid": True,
            "feasible_distance_intervals_m": [
                [float(lower), float(upper)] for lower, upper in feasible
            ],
        }
        return location, float(distance), audit

    def nearest_wall_measurement(
        self,
        footprint: Polygon,
    ) -> dict[str, Any]:
        measurements = self.wall_measurements(footprint)
        if not measurements:
            raise PolygonRoomGeometryError("room has no wall segments")
        return deepcopy(
            min(
                measurements,
                key=lambda item: (
                    float(item["distance_m"]),
                    str(item["wall_id"]),
                ),
            )
        )

    def wall_measurements(self, footprint: Polygon) -> list[dict[str, Any]]:
        coordinates = list(footprint.exterior.coords)[:-1]
        result: list[dict[str, Any]] = []
        for wall in self.walls:
            normal = np.asarray(wall.inward_normal_xy, dtype=float)
            start = np.asarray(wall.start_xy, dtype=float)
            signed = min(
                float(np.dot(np.asarray(point, dtype=float) - start, normal))
                for point in coordinates
            )
            distance = float(footprint.distance(wall.line))
            result.append(
                {
                    "wall_id": wall.wall_id,
                    "plane": wall.wall_id,
                    "signed_clearance_m": signed,
                    "distance_m": distance,
                    "start_xy": list(wall.start_xy),
                    "end_xy": list(wall.end_xy),
                    "inward_normal_xy": list(wall.inward_normal_xy),
                    "tangent_xy": list(wall.tangent_xy),
                }
            )
        return result

    def violated_walls(
        self,
        footprint: Polygon,
    ) -> list[dict[str, Any]]:
        tolerance_band = max(self.tolerance_m * 4.0, 1.0e-5)
        candidates = [
            measurement
            for measurement in self.wall_measurements(footprint)
            if footprint.intersects(
                self.wall_by_id(str(measurement["wall_id"])).line.buffer(
                    tolerance_band,
                    cap_style="flat",
                )
            )
        ]
        if candidates:
            return candidates
        return [self.nearest_wall_measurement(footprint)]

    def wall_by_id(self, wall_id: str) -> PolygonWall:
        for wall in self.walls:
            if wall.wall_id == wall_id:
                return wall
        raise PolygonRoomGeometryError(f"unknown wall_id {wall_id!r}")

    def _camera_region(
        self,
        clearance_m: float,
        *,
        allow_relaxation: bool = True,
    ) -> tuple[Any, float]:
        requested = max(0.0, float(clearance_m))
        clearances = [requested]
        if allow_relaxation and requested > 0.02:
            clearances.extend([requested * 0.5, 0.02])
        if allow_relaxation:
            clearances.append(0.0)
        for clearance in dict.fromkeys(clearances):
            region = (
                self.polygon.buffer(-clearance, join_style="mitre")
                if clearance > 0.0
                else self.polygon
            )
            if not region.is_empty:
                return region, float(clearance)
        raise PolygonRoomGeometryError("polygon has no usable interior")

    def _xy_ray_intervals(
        self,
        region: Any,
        *,
        target_vector: np.ndarray,
        direction: np.ndarray,
        maximum_trace: float,
    ) -> list[tuple[float, float]]:
        horizontal = np.asarray(direction[:2], dtype=float)
        horizontal_norm_squared = float(np.dot(horizontal, horizontal))
        if horizontal_norm_squared <= 1.0e-12:
            if region.covers(Point(target_vector[:2])):
                return [(0.0, maximum_trace)]
            return []
        endpoint = target_vector[:2] + horizontal * maximum_trace
        intersection = region.intersection(
            LineString([target_vector[:2], endpoint])
        )
        intervals: list[tuple[float, float]] = []
        for line in _line_components(intersection):
            coordinates = list(line.coords)
            if not coordinates:
                # GEOS may retain an empty LineString inside a degenerate
                # tangent/collection intersection. It contributes no feasible
                # interval and is normal geometry exhaustion, not a program
                # failure.
                continue
            projected = [
                float(
                    np.dot(
                        np.asarray(point, dtype=float) - target_vector[:2],
                        horizontal,
                    )
                    / horizontal_norm_squared
                )
                for point in coordinates
            ]
            lower = max(0.0, min(projected))
            upper = min(maximum_trace, max(projected))
            if upper + self.tolerance_m >= lower:
                intervals.append((lower, upper))
        return sorted(intervals)


def polygon_geometry_from_scene(
    scene: Mapping[str, Any] | None,
) -> PolygonRoomGeometry | None:
    if not isinstance(scene, Mapping):
        return None
    metadata = scene.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("evaluation_mode") != NON_RECTANGULAR_EVALUATION_MODE:
        return None
    geometry = metadata.get(POLYGON_ROOM_METADATA_KEY)
    if not isinstance(geometry, Mapping):
        raise PolygonRoomGeometryError(
            "selected non-rectangular scene lacks polygon room metadata"
        )
    return PolygonRoomGeometry.from_metadata(geometry)


def is_non_rectangular_camera_scene(scene: Mapping[str, Any] | None) -> bool:
    return polygon_geometry_from_scene(scene) is not None


def object_footprint_polygon(raw_object: Mapping[str, Any]) -> Polygon:
    normalized = normalize_object(dict(raw_object))
    footprint = MultiPoint(get_obb_corners(normalized)[:, :2]).convex_hull
    if footprint.is_empty or not isinstance(footprint, Polygon):
        raise PolygonRoomGeometryError(
            f"object {normalized.id!r} has no valid footprint"
        )
    return footprint


def object_vertical_interval(raw_object: Mapping[str, Any]) -> tuple[float, float]:
    normalized = normalize_object(dict(raw_object))
    corners = get_obb_corners(normalized)
    return (float(np.min(corners[:, 2])), float(np.max(corners[:, 2])))


def point_inside_object_obb(
    point: Iterable[float],
    raw_object: Mapping[str, Any],
    *,
    clearance_m: float = 0.0,
) -> bool:
    normalized = normalize_object(dict(raw_object))
    value = _vector3(point, "point")
    local = normalized.R.T @ (value - normalized.center)
    return bool(
        np.all(
            np.abs(local)
            <= normalized.half + max(0.0, float(clearance_m))
        )
    )


def _line_components(value: Any) -> list[LineString]:
    if isinstance(value, LineString):
        return [value]
    if isinstance(value, MultiLineString):
        return list(value.geoms)
    if isinstance(value, GeometryCollection):
        result: list[LineString] = []
        for item in value.geoms:
            result.extend(_line_components(item))
        return result
    return []


def _z_ray_interval(
    *,
    target_z: float,
    direction_z: float,
    minimum_z: float,
    maximum_z: float,
    maximum_trace: float,
) -> tuple[float, float] | None:
    if maximum_z <= minimum_z:
        return None
    if abs(direction_z) <= 1.0e-12:
        return (
            (0.0, maximum_trace)
            if minimum_z <= target_z <= maximum_z
            else None
        )
    first = (minimum_z - target_z) / direction_z
    second = (maximum_z - target_z) / direction_z
    lower = max(0.0, min(first, second))
    upper = min(maximum_trace, max(first, second))
    return (lower, upper) if upper >= lower else None


def _point(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        raise PolygonRoomGeometryError(f"{path} must be a 2-vector")
    return (_finite(value[0], path), _finite(value[1], path))


def _vector3(value: Any, path: str) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
        raise PolygonRoomGeometryError(f"{path} must be a 3-vector")
    return np.asarray(
        [_finite(value[index], f"{path}[{index}]") for index in range(3)],
        dtype=float,
    )


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise PolygonRoomGeometryError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PolygonRoomGeometryError(f"{path} must be finite")
    return result


def _positive(value: Any, path: str) -> float:
    result = _finite(value, path)
    if result <= 0.0:
        raise PolygonRoomGeometryError(f"{path} must be positive")
    return result


__all__ = [
    "DEFAULT_CAMERA_VERTICAL_CLEARANCE_M",
    "DEFAULT_CAMERA_WALL_CLEARANCE_M",
    "POLYGON_ROOM_GEOMETRY_SCHEMA_VERSION",
    "POLYGON_ROOM_METADATA_KEY",
    "PolygonRoomGeometry",
    "PolygonRoomGeometryError",
    "PolygonWall",
    "is_non_rectangular_camera_scene",
    "object_footprint_polygon",
    "object_vertical_interval",
    "point_inside_object_obb",
    "polygon_geometry_from_scene",
]
