from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def bbox_z_bounds(obj: dict) -> tuple[float, float]:
    center_z = float(obj["center"][2])
    height = float(obj["size"][2])
    return center_z - height / 2.0, center_z + height / 2.0


def room_polygon(room: dict) -> Polygon | MultiPolygon:
    region_polygons = _region_polygons(room)
    if region_polygons:
        union = unary_union(region_polygons)
        if not union.is_empty:
            return union
    polygon = room.get("floor_polygon") or room.get("boundary")
    if not polygon:
        raise KeyError("room requires floor_polygon or boundary")
    return Polygon(polygon)


def footprint_polygon(obj: dict, yaw_aware: bool = True) -> Polygon:
    cx, cy = float(obj["center"][0]), float(obj["center"][1])
    sx, sy = float(obj["size"][0]), float(obj["size"][1])
    half_x, half_y = sx / 2.0, sy / 2.0
    corners = np.array(
        [
            [-half_x, -half_y],
            [half_x, -half_y],
            [half_x, half_y],
            [-half_x, half_y],
        ],
        dtype=float,
    )
    if yaw_aware:
        theta = math.radians(float(obj.get("yaw", 0.0)))
        rotation = np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )
        corners = corners @ rotation.T
    corners += np.array([cx, cy])
    return Polygon(corners)


def footprint_inside_room(obj: dict, room: dict, tolerance: float = 1.0e-6) -> bool:
    footprint = footprint_polygon(obj)
    polygon = room_polygon(room)
    return polygon.buffer(tolerance).covers(footprint)


def _region_polygons(room: dict) -> list[Polygon]:
    floor_plan = room.get("floor_plan") if isinstance(room.get("floor_plan"), dict) else {}
    regions = floor_plan.get("regions") or room.get("regions") or []
    if not isinstance(regions, list):
        return []
    polygons = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        polygon = region.get("floor_polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        candidate = Polygon(polygon)
        if candidate.is_valid and not candidate.is_empty and candidate.area > 0:
            polygons.append(candidate)
    return polygons


def footprints_intersect(obj_a: dict, obj_b: dict, area_tolerance: float = 1.0e-6) -> bool:
    return footprint_intersection_area(obj_a, obj_b) > area_tolerance


def footprint_intersection_area(obj_a: dict, obj_b: dict) -> float:
    return float(footprint_polygon(obj_a).intersection(footprint_polygon(obj_b)).area)


def horizontal_overlap_ratio(child: dict, parent: dict) -> float:
    child_area = float(footprint_polygon(child).area)
    if child_area <= 0:
        return 0.0
    return footprint_intersection_area(child, parent) / child_area


def vertical_overlap(obj_a: dict, obj_b: dict) -> float:
    a_min, a_max = bbox_z_bounds(obj_a)
    b_min, b_max = bbox_z_bounds(obj_b)
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def center_xy(obj: dict) -> np.ndarray:
    return np.array([float(obj["center"][0]), float(obj["center"][1])], dtype=float)


def distance_xy(obj_a: dict, obj_b: dict) -> float:
    return float(np.linalg.norm(center_xy(obj_a) - center_xy(obj_b)))


def sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})
