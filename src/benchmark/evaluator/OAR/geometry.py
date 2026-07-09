from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from benchmark.evaluator.OOR.geometry import (
    NormalizedObject,
    get_footprint_corners_xy,
    normalize_object,
    point_in_convex_polygon_2d,
    point_segment_distance_2d,
    segments_intersect_2d,
)


EPS = 1.0e-9


@dataclass(frozen=True)
class WallSegment:
    p0: np.ndarray
    p1: np.ndarray
    length: float
    name: str


@dataclass(frozen=True)
class Corner:
    point: np.ndarray
    name: str


@dataclass(frozen=True)
class NormalizedRoom:
    boundary: np.ndarray
    scene_height: float | None
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    centroid: np.ndarray
    wall_segments: list[WallSegment]
    corners: list[Corner]


def get_room_boundary(scene: dict) -> list[list[float]]:
    if not isinstance(scene, dict):
        return []
    boundary = scene.get("boundary")
    if boundary is None and isinstance(scene.get("room"), dict):
        boundary = scene["room"].get("boundary")
    return _boundary_list(boundary)


def get_scene_height(scene: dict) -> float | None:
    if not isinstance(scene, dict):
        return None
    value = scene.get("scene_height")
    if value is None and isinstance(scene.get("room"), dict):
        value = scene["room"].get("height")
    try:
        height = float(value)
    except (TypeError, ValueError):
        return None
    return height if height > 0.0 else None


def normalize_room(scene: dict) -> NormalizedRoom:
    boundary_list = get_room_boundary(scene)
    boundary = np.asarray(boundary_list, dtype=float)
    if boundary.ndim != 2 or boundary.shape[0] < 3 or boundary.shape[1] != 2:
        raise ValueError("scene room boundary must contain at least three [x, y] points")
    min_x, max_x = float(np.min(boundary[:, 0])), float(np.max(boundary[:, 0]))
    min_y, max_y = float(np.min(boundary[:, 1])), float(np.max(boundary[:, 1]))
    centroid = np.mean(boundary, axis=0)
    walls = get_wall_segments(boundary)
    corners = get_corner_points(boundary)
    return NormalizedRoom(
        boundary=boundary,
        scene_height=get_scene_height(scene),
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        centroid=centroid,
        wall_segments=walls,
        corners=corners,
    )


def get_wall_segments(boundary: Iterable[Iterable[float]] | np.ndarray) -> list[WallSegment]:
    poly = np.asarray(boundary, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return []
    centroid = np.mean(poly, axis=0)
    mids = np.array([(poly[i] + poly[(i + 1) % len(poly)]) / 2.0 for i in range(len(poly))])
    axis_aligned_rect = len(poly) == 4 and _mostly_axis_aligned(poly)
    names: list[str] = []
    if axis_aligned_rect:
        east_idx = int(np.argmax(mids[:, 0]))
        west_idx = int(np.argmin(mids[:, 0]))
        north_idx = int(np.argmax(mids[:, 1]))
        south_idx = int(np.argmin(mids[:, 1]))
        by_index = {east_idx: "east", west_idx: "west", north_idx: "north", south_idx: "south"}
        names = [by_index.get(index, _wall_name_from_midpoint(mids[index], centroid)) for index in range(len(poly))]
    else:
        names = [_wall_name_from_midpoint(mid, centroid) for mid in mids]

    walls = []
    for index in range(len(poly)):
        p0 = poly[index].astype(float)
        p1 = poly[(index + 1) % len(poly)].astype(float)
        walls.append(WallSegment(p0=p0, p1=p1, length=float(np.linalg.norm(p1 - p0)), name=names[index]))
    return walls


def object_footprint_points(obj: NormalizedObject | dict) -> np.ndarray:
    normalized = obj if isinstance(obj, NormalizedObject) else normalize_object(obj)
    return get_footprint_corners_xy(normalized)


def min_distance_footprint_to_wall(obj: NormalizedObject | dict, wall_segment: WallSegment) -> float:
    footprint = object_footprint_points(obj)
    if len(footprint) < 3:
        return math.inf
    for index in range(len(footprint)):
        if segments_intersect_2d(footprint[index], footprint[(index + 1) % len(footprint)], wall_segment.p0, wall_segment.p1):
            return 0.0
    distances = [point_segment_distance_2d(point, wall_segment.p0, wall_segment.p1) for point in footprint]
    for endpoint in [wall_segment.p0, wall_segment.p1]:
        distances.extend(point_segment_distance_2d(endpoint, footprint[i], footprint[(i + 1) % len(footprint)]) for i in range(len(footprint)))
    return float(min(distances)) if distances else math.inf


def min_distance_center_to_wall(obj: NormalizedObject | dict, wall_segment: WallSegment) -> float:
    normalized = obj if isinstance(obj, NormalizedObject) else normalize_object(obj)
    return point_segment_distance_2d(normalized.center[:2], wall_segment.p0, wall_segment.p1)


def footprint_inside_boundary_ratio(obj: NormalizedObject | dict, boundary: Iterable[Iterable[float]] | np.ndarray) -> float:
    normalized = obj if isinstance(obj, NormalizedObject) else normalize_object(obj)
    footprint = object_footprint_points(normalized)
    points = list(footprint) + [normalized.center[:2]]
    if not points:
        return 0.0
    poly = np.asarray(boundary, dtype=float)
    inside = sum(1 for point in points if point_in_polygon_2d(point, poly, eps=1.0e-8))
    return float(inside) / float(len(points))


def get_corner_points(boundary: Iterable[Iterable[float]] | np.ndarray) -> list[Corner]:
    poly = np.asarray(boundary, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return []
    centroid = np.mean(poly, axis=0)
    return [Corner(point=point.astype(float), name=_corner_name_from_point(point, centroid)) for point in poly]


def footprint_point_distance(obj: NormalizedObject | dict, point: Iterable[float] | np.ndarray) -> float:
    footprint = object_footprint_points(obj)
    p = np.asarray(point, dtype=float)
    if len(footprint) < 3:
        return math.inf
    if point_in_convex_polygon_2d(p, footprint, eps=1.0e-8):
        return 0.0
    distances = [point_segment_distance_2d(p, footprint[i], footprint[(i + 1) % len(footprint)]) for i in range(len(footprint))]
    return float(min(distances)) if distances else math.inf


def point_in_polygon_2d(point: Iterable[float] | np.ndarray, polygon: np.ndarray, eps: float = 0.0) -> bool:
    p = np.asarray(point, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or len(poly) < 3:
        return False
    for index in range(len(poly)):
        if point_segment_distance_2d(p, poly[index], poly[(index + 1) % len(poly)]) <= eps:
            return True
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        yi, yj = poly[i, 1], poly[j, 1]
        xi, xj = poly[i, 0], poly[j, 0]
        intersects = ((yi > p[1]) != (yj > p[1])) and (p[0] < (xj - xi) * (p[1] - yi) / ((yj - yi) + EPS) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _boundary_list(value: object) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points: list[list[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            continue
    return points


def _mostly_axis_aligned(poly: np.ndarray) -> bool:
    for index in range(len(poly)):
        edge = poly[(index + 1) % len(poly)] - poly[index]
        if abs(float(edge[0])) > 1.0e-6 and abs(float(edge[1])) > 1.0e-6:
            return False
    return True


def _wall_name_from_midpoint(midpoint: np.ndarray, centroid: np.ndarray) -> str:
    delta = midpoint - centroid
    if abs(float(delta[0])) > abs(float(delta[1])):
        return "east" if delta[0] > 0 else "west"
    return "north" if delta[1] > 0 else "south"


def _corner_name_from_point(point: np.ndarray, centroid: np.ndarray) -> str:
    ew = "east" if point[0] >= centroid[0] else "west"
    ns = "north" if point[1] >= centroid[1] else "south"
    return f"{ns}{ew}"
