"""Deterministic geometry facts for functional evidence acquisition.

This module translates trusted local-side hypotheses into world-space and
logical-boundary observations. It never decides metric validity.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


FUNCTIONAL_GEOMETRY_VERSION = "functional_geometry_v2"
_LOCAL_SIDE_AXES = {
    "local_pos_x": (1.0, 0.0, 0.0),
    "local_neg_x": (-1.0, 0.0, 0.0),
    "local_pos_y": (0.0, 1.0, 0.0),
    "local_neg_y": (0.0, -1.0, 0.0),
}
_APPROACH_SAMPLE_DISTANCES_M = (0.5, 1.0)
_FRONTAGE_EPSILON_M = 1.0e-3


def build_functional_geometry_observations(
    scene: dict[str, Any],
    probe: dict[str, Any],
) -> dict[str, Any]:
    """Return read-only target, surface, and logical-boundary geometry."""

    objects = {
        str(item.get("id")): item
        for item in scene.get("objects") or []
        if isinstance(item, dict) and item.get("id")
    }
    target_ids = list(
        dict.fromkeys(
            [
                *[
                    str(item)
                    for item in probe.get("target_ids") or []
                    if str(item)
                ],
                *[
                    str(item)
                    for item in probe.get("related_target_ids") or []
                    if str(item)
                ],
            ]
        )
    )
    target_bounds = _union_bounds(
        [objects[item] for item in target_ids if item in objects]
    )
    surface_contracts = {
        str(item.get("target_id") or ""): item
        for item in probe.get("surface_targets") or []
        if isinstance(item, dict) and item.get("target_id")
    }
    boundary = (
        _boundary_xy(scene)
        if probe.get("logical_boundary_enabled", True) is not False
        else []
    )
    surfaces: list[dict[str, Any]] = []
    for hypothesis in probe.get("usable_surface_hypotheses") or []:
        if not isinstance(hypothesis, dict):
            continue
        target_id = str(hypothesis.get("target_id") or "")
        object_record = objects.get(target_id)
        if object_record is None:
            continue
        surface_contract = surface_contracts.get(target_id, {})
        clearance_applicable = bool(
            surface_contract.get("need_clearance", False)
        )
        center = _vector3(object_record.get("center"))
        size = _vector3(object_record.get("size"))
        rotation = _vector3(object_record.get("rotation") or [0.0, 0.0, 0.0])
        for surface in hypothesis.get("surfaces") or []:
            if not isinstance(surface, dict):
                continue
            side_id = str(surface.get("side_id") or "")
            local_axis = _LOCAL_SIDE_AXES.get(side_id)
            if local_axis is None:
                continue
            world = _rotate_euler(local_axis, rotation)
            horizontal_norm = math.hypot(world[0], world[1])
            if horizontal_norm <= 1.0e-12:
                continue
            direction = (
                world[0] / horizontal_norm,
                world[1] / horizontal_norm,
                0.0,
            )
            support_extent = _obb_support_extent(
                size=size,
                rotation_degrees=rotation,
                direction=direction,
            )
            frontage_origin = (
                center[0]
                + direction[0]
                * (support_extent + _FRONTAGE_EPSILON_M),
                center[1]
                + direction[1]
                * (support_extent + _FRONTAGE_EPSILON_M),
            )
            boundary_facts = _boundary_facts(
                origin=frontage_origin,
                direction=(direction[0], direction[1]),
                boundary=boundary,
            )
            surfaces.append(
                {
                    "target_id": target_id,
                    "status": str(hypothesis.get("status") or ""),
                    "surface_role": surface.get("surface_role"),
                    "side_id": side_id,
                    "descriptor_kind": (
                        "usable_side_world_direction"
                    ),
                    "routing_only": True,
                    "architecture_orientation_applicable": True,
                    "clearance_applicable": clearance_applicable,
                    "world_outward_direction": list(direction),
                    "object_center_xy": [center[0], center[1]],
                    "frontage_origin_xy": list(frontage_origin),
                    "frontage_support_extent_m": support_extent,
                    **boundary_facts,
                }
            )
    return {
        "schema_version": FUNCTIONAL_GEOMETRY_VERSION,
        "decision_authority": "none",
        "scene_access": "read_only",
        "descriptor_role": "camera_and_check_routing_only",
        "metric_verdict_authority": False,
        "target_ids": target_ids,
        "target_bounds": target_bounds,
        "focus_center": (
            [
                (target_bounds[0][axis] + target_bounds[1][axis]) / 2.0
                for axis in range(3)
            ]
            if target_bounds is not None
            else None
        ),
        "extent": (
            [
                target_bounds[1][axis] - target_bounds[0][axis]
                for axis in range(3)
            ]
            if target_bounds is not None
            else None
        ),
        "logical_boundary_available": bool(boundary),
        "surface_observations": surfaces,
        "observation_status": (
            "available"
            if surfaces
            else "ambiguous"
            if any(
                str(item.get("status") or "")
                in {"ambiguous", "insufficient_comparison"}
                for item in probe.get("usable_surface_hypotheses") or []
                if isinstance(item, dict)
            )
            else "unavailable"
        ),
        "source_probe": deepcopy(
            {
                "probe_id": probe.get("probe_id"),
                "kind": probe.get("kind"),
                "required_observations": probe.get(
                    "required_observations"
                ),
            }
        ),
    }


def _boundary_facts(
    *,
    origin: tuple[float, float],
    direction: tuple[float, float],
    boundary: list[tuple[float, float]],
) -> dict[str, Any]:
    if len(boundary) < 3:
        return {
            "nearest_boundary_distance_m": None,
            "outward_ray_boundary_distance_m": None,
            "approach_samples": [],
        }
    nearest = min(
        _point_segment_distance(origin, start, end)
        for start, end in _segments(boundary)
    )
    intersections = [
        distance
        for start, end in _segments(boundary)
        if (
            distance := _ray_segment_distance(
                origin,
                direction,
                start,
                end,
            )
        )
        is not None
    ]
    samples = [
        {
            "distance_m": distance,
            "point_xy": [
                origin[0] + direction[0] * distance,
                origin[1] + direction[1] * distance,
            ],
            "inside_logical_boundary": _point_in_polygon(
                (
                    origin[0] + direction[0] * distance,
                    origin[1] + direction[1] * distance,
                ),
                boundary,
            ),
        }
        for distance in _APPROACH_SAMPLE_DISTANCES_M
    ]
    return {
        "nearest_boundary_distance_m": float(nearest),
        "outward_ray_boundary_distance_m": (
            float(min(intersections)) if intersections else None
        ),
        "approach_samples": samples,
    }


def _union_bounds(
    objects: list[dict[str, Any]],
) -> list[list[float]] | None:
    bounds: list[tuple[list[float], list[float]]] = []
    for item in objects:
        center = _vector3(item.get("center"))
        size = _vector3(item.get("size"))
        rotation = _vector3(item.get("rotation") or [0.0, 0.0, 0.0])
        if any(value <= 0.0 for value in size):
            continue
        half = _rotated_world_half_extents(
            size=size,
            rotation_degrees=rotation,
        )
        bounds.append(
            (
                [center[index] - half[index] for index in range(3)],
                [center[index] + half[index] for index in range(3)],
            )
        )
    if not bounds:
        return None
    return [
        [min(item[0][axis] for item in bounds) for axis in range(3)],
        [max(item[1][axis] for item in bounds) for axis in range(3)],
    ]


def _obb_support_extent(
    *,
    size: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> float:
    if any(value <= 0.0 for value in size):
        return 0.0
    local_axes = (
        _rotate_euler((1.0, 0.0, 0.0), rotation_degrees),
        _rotate_euler((0.0, 1.0, 0.0), rotation_degrees),
        _rotate_euler((0.0, 0.0, 1.0), rotation_degrees),
    )
    return float(
        sum(
            (size[index] / 2.0)
            * abs(
                sum(
                    local_axes[index][axis] * direction[axis]
                    for axis in range(3)
                )
            )
            for index in range(3)
        )
    )


def _rotated_world_half_extents(
    *,
    size: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float],
) -> tuple[float, float, float]:
    axes = (
        _rotate_euler((1.0, 0.0, 0.0), rotation_degrees),
        _rotate_euler((0.0, 1.0, 0.0), rotation_degrees),
        _rotate_euler((0.0, 0.0, 1.0), rotation_degrees),
    )
    half = tuple(value / 2.0 for value in size)
    return tuple(
        sum(
            abs(axes[local_axis][world_axis]) * half[local_axis]
            for local_axis in range(3)
        )
        for world_axis in range(3)
    )


def _boundary_xy(scene: dict[str, Any]) -> list[tuple[float, float]]:
    raw = scene.get("boundary")
    if not isinstance(raw, list):
        room = scene.get("room")
        raw = room.get("boundary") if isinstance(room, dict) else []
    result: list[tuple[float, float]] = []
    for item in raw or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            point = (float(item[0]), float(item[1]))
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in point):
            result.append(point)
    return result


def _rotate_euler(
    vector: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float],
) -> tuple[float, float, float]:
    roll, pitch, yaw = [math.radians(value) for value in rotation_degrees]
    x, y, z = vector
    y, z = y * math.cos(roll) - z * math.sin(roll), (
        y * math.sin(roll) + z * math.cos(roll)
    )
    x, z = x * math.cos(pitch) + z * math.sin(pitch), (
        -x * math.sin(pitch) + z * math.cos(pitch)
    )
    x, y = x * math.cos(yaw) - y * math.sin(yaw), (
        x * math.sin(yaw) + y * math.cos(yaw)
    )
    return x, y, z


def _vector3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return 0.0, 0.0, 0.0
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0
    return result if all(math.isfinite(item) for item in result) else (
        0.0,
        0.0,
        0.0,
    )


def _segments(
    polygon: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return list(zip(polygon, [*polygon[1:], polygon[0]], strict=True))


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
            / denominator,
        ),
    )
    return math.hypot(
        point[0] - (start[0] + t * dx),
        point[1] - (start[1] + t * dy),
    )


def _ray_segment_distance(
    origin: tuple[float, float],
    direction: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float | None:
    segment = (end[0] - start[0], end[1] - start[1])
    determinant = direction[0] * segment[1] - direction[1] * segment[0]
    if abs(determinant) <= 1.0e-12:
        return None
    offset = (start[0] - origin[0], start[1] - origin[1])
    ray_t = (offset[0] * segment[1] - offset[1] * segment[0]) / determinant
    segment_t = (
        offset[0] * direction[1] - offset[1] * direction[0]
    ) / determinant
    if ray_t >= 0.0 and 0.0 <= segment_t <= 1.0:
        return ray_t
    return None


def _point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    inside = False
    x, y = point
    for (x0, y0), (x1, y1) in _segments(polygon):
        if (y0 > y) == (y1 > y):
            continue
        crossing_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
        if x < crossing_x:
            inside = not inside
    return inside
