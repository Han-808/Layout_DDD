from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from benchmark.scene_io.object_normalization import EPS, NormalizedObject, normalize_object, rotation_matrix_from_euler

def get_obb_corners(obj: NormalizedObject) -> np.ndarray:
    return _obb_corners_from_parts(obj.center, obj.half, obj.R)


def get_footprint_corners_xy(obj: NormalizedObject) -> np.ndarray:
    local = np.array(
        [
            [-obj.half[0], -obj.half[1], 0.0],
            [obj.half[0], -obj.half[1], 0.0],
            [obj.half[0], obj.half[1], 0.0],
            [-obj.half[0], obj.half[1], 0.0],
        ]
    )
    return (obj.center + local @ obj.R.T)[:, :2]


def point_in_obb(point: Iterable[float] | np.ndarray, obj: NormalizedObject, eps: float = 0.0) -> bool:
    p = np.asarray(point, dtype=float)
    q = obj.R.T @ (p - obj.center)
    return bool(np.all(np.abs(q) <= obj.half + float(eps)))


def sample_points_in_obb(obj: NormalizedObject, grid: tuple[int, int, int] = (3, 3, 3)) -> np.ndarray:
    xs = _linspace_inside(-obj.half[0], obj.half[0], grid[0])
    ys = _linspace_inside(-obj.half[1], obj.half[1], grid[1])
    zs = _linspace_inside(-obj.half[2], obj.half[2], grid[2])
    local = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)
    if not np.any(np.all(np.isclose(local, 0.0), axis=1)):
        local = np.vstack([local, np.zeros(3)])
    return obj.center + local @ obj.R.T


def sample_front_face_points(obj: NormalizedObject, grid: tuple[int, int] = (3, 3)) -> np.ndarray:
    xs = _linspace_inside(-obj.half[0], obj.half[0], grid[0])
    zs = _linspace_inside(-obj.half[2], obj.half[2], grid[1])
    local = np.array([[x, -obj.half[1], z] for x in xs for z in zs], dtype=float)
    return obj.center + local @ obj.R.T


def footprint_edge_distance(obj_a: NormalizedObject, obj_b: NormalizedObject) -> float:
    return polygon_distance_2d(get_footprint_corners_xy(obj_a), get_footprint_corners_xy(obj_b))


def footprint_overlap_score(obj_a: NormalizedObject, obj_b: NormalizedObject, grid: tuple[int, int] = (5, 5)) -> float:
    xs = _linspace_inside(-obj_a.half[0], obj_a.half[0], grid[0])
    ys = _linspace_inside(-obj_a.half[1], obj_a.half[1], grid[1])
    local = np.array([[x, y, 0.0] for x in xs for y in ys], dtype=float)
    points = (obj_a.center + local @ obj_a.R.T)[:, :2]
    poly_b = get_footprint_corners_xy(obj_b)
    inside = sum(1 for point in points if point_in_convex_polygon_2d(point, poly_b, eps=1.0e-8))
    return float(inside) / float(len(points)) if len(points) else 0.0


def project_obb_to_axis(obj: NormalizedObject, axis: Iterable[float] | np.ndarray) -> tuple[float, float]:
    unit_axis = _unit(np.asarray(axis, dtype=float))
    projections = get_obb_corners(obj) @ unit_axis
    return float(np.min(projections)), float(np.max(projections))


def interval_gap(interval_a: tuple[float, float], interval_b: tuple[float, float]) -> float:
    a_min, a_max = interval_a
    b_min, b_max = interval_b
    if a_max >= b_min and b_max >= a_min:
        return 0.0
    return float(max(b_min - a_max, a_min - b_max))


def interval_overlap_ratio(interval_a: tuple[float, float], interval_b: tuple[float, float]) -> float:
    a_min, a_max = interval_a
    b_min, b_max = interval_b
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    denom = min(max(0.0, a_max - a_min), max(0.0, b_max - b_min))
    return 0.0 if denom <= EPS else float(overlap / denom)


def ray_intersects_obb(
    origin: Iterable[float] | np.ndarray,
    direction: Iterable[float] | np.ndarray,
    obj_b: NormalizedObject,
    max_distance: float | None = None,
) -> bool:
    ray_origin = obj_b.R.T @ (np.asarray(origin, dtype=float) - obj_b.center)
    ray_dir = obj_b.R.T @ _unit(np.asarray(direction, dtype=float))
    t_enter = -math.inf
    t_exit = math.inf
    for index in range(3):
        if abs(ray_dir[index]) <= EPS:
            if ray_origin[index] < -obj_b.half[index] or ray_origin[index] > obj_b.half[index]:
                return False
            continue
        t1 = (-obj_b.half[index] - ray_origin[index]) / ray_dir[index]
        t2 = (obj_b.half[index] - ray_origin[index]) / ray_dir[index]
        t_near, t_far = min(t1, t2), max(t1, t2)
        t_enter = max(t_enter, t_near)
        t_exit = min(t_exit, t_far)
        if t_enter > t_exit:
            return False
    hit_distance = max(t_enter, 0.0)
    if t_exit < hit_distance:
        return False
    return max_distance is None or hit_distance <= float(max_distance)


def angle_diff_degrees(a: float, b: float) -> float:
    diff = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(diff)


def angle_between_degrees(a: Iterable[float] | np.ndarray, b: Iterable[float] | np.ndarray) -> float:
    a_unit = _unit(np.asarray(a, dtype=float))
    b_unit = _unit(np.asarray(b, dtype=float))
    if np.linalg.norm(a_unit) <= EPS or np.linalg.norm(b_unit) <= EPS:
        return 180.0
    cosine = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def point_in_convex_polygon_2d(point: Iterable[float] | np.ndarray, polygon: np.ndarray, eps: float = 0.0) -> bool:
    p = np.asarray(point, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 3:
        return False
    signs = []
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        cross = _cross2(b - a, p - a)
        if abs(cross) <= eps:
            continue
        signs.append(cross > 0)
    return not signs or all(sign == signs[0] for sign in signs)


def segments_intersect_2d(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray, eps: float = 1.0e-9) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return _cross2(q - p, r - p)

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    return (
        abs(o1) <= eps and on_segment(a1, b1, a2)
        or abs(o2) <= eps and on_segment(a1, b2, a2)
        or abs(o3) <= eps and on_segment(b1, a1, b2)
        or abs(o4) <= eps and on_segment(b1, a2, b2)
    )


def point_segment_distance_2d(point: Iterable[float] | np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(point, dtype=float)
    segment = b - a
    denom = float(np.dot(segment, segment))
    if denom <= EPS:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, segment) / denom, 0.0, 1.0))
    projection = a + t * segment
    return float(np.linalg.norm(p - projection))


def polygon_distance_2d(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    a = np.asarray(poly_a, dtype=float)
    b = np.asarray(poly_b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return math.inf
    if any(point_in_convex_polygon_2d(point, b, eps=1.0e-9) for point in a):
        return 0.0
    if any(point_in_convex_polygon_2d(point, a, eps=1.0e-9) for point in b):
        return 0.0
    for i in range(len(a)):
        a1, a2 = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            if segments_intersect_2d(a1, a2, b[j], b[(j + 1) % len(b)]):
                return 0.0
    distances = []
    for point in a:
        distances.extend(point_segment_distance_2d(point, b[j], b[(j + 1) % len(b)]) for j in range(len(b)))
    for point in b:
        distances.extend(point_segment_distance_2d(point, a[i], a[(i + 1) % len(a)]) for i in range(len(a)))
    return float(min(distances)) if distances else math.inf


def center_xy_distance(obj_a: NormalizedObject, obj_b: NormalizedObject) -> float:
    return float(np.linalg.norm(obj_a.center[:2] - obj_b.center[:2]))


def _obb_corners_from_parts(center: np.ndarray, half: np.ndarray, R: np.ndarray) -> np.ndarray:
    local = np.array(
        [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=float,
    )
    return center + local @ R.T


def _linspace_inside(start: float, stop: float, count: int) -> np.ndarray:
    count = max(1, int(count))
    return np.linspace(float(start), float(stop), count)


def _first_present(obj: dict, keys: list[str]) -> object | None:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def _rotation_from_yaw(value: object | None) -> list[float] | None:
    if value is None:
        return None
    return [0.0, 0.0, value]


def _vector3(value: object | None) -> np.ndarray | None:
    if isinstance(value, np.ndarray) and value.shape == (3,):
        return value.astype(float)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)
        except (TypeError, ValueError):
            return None
    return None


def _rotation_vector(value: object | None, *, assume_degrees: bool = False) -> np.ndarray:
    vector = _vector3(value)
    if vector is None:
        vector = np.zeros(3, dtype=float)
    if assume_degrees:
        return vector.astype(float)
    max_abs = float(np.max(np.abs(vector))) if vector.size else 0.0
    if max_abs <= (2.0 * math.pi + 1.0e-6):
        return np.degrees(vector)
    return vector.astype(float)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector, dtype=float)
    return vector.astype(float) / norm


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
