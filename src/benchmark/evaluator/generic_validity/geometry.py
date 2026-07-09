from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from benchmark.evaluator.generic_validity.asset_resolver import resolve_asset_metadata


EPS = 1.0e-9


@dataclass(frozen=True)
class NormalizedObject:
    id: str
    jid: str | None
    category: str | None
    center: np.ndarray
    size: np.ndarray
    half: np.ndarray
    rotation: np.ndarray
    yaw_degrees: float
    R: np.ndarray
    right: np.ndarray
    front: np.ndarray
    up: np.ndarray
    bottom_z: float
    top_z: float
    asset_ref: dict
    asset_proxy: dict
    metadata: dict
    interactive: bool


def load_objects(scene: dict) -> list[dict]:
    if not isinstance(scene, dict):
        return []
    for key in ["objects", "assets"]:
        value = scene.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


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


def normalize_object(obj: dict, *, asset_csv_path: str | None = None, asset_root: str | None = None) -> NormalizedObject:
    if not isinstance(obj, dict):
        raise ValueError("object must be a mapping")
    if asset_csv_path or asset_root:
        obj = resolve_asset_metadata(obj, asset_csv_path=asset_csv_path, asset_root=asset_root)
    pose = obj.get("pose") if isinstance(obj.get("pose"), dict) else {}
    placement = obj.get("placement") if isinstance(obj.get("placement"), dict) else {}
    object_id = _first_present(obj, ["id", "object_id", "asset_id"])
    if object_id is None:
        raise ValueError("object id is missing")
    center = _vector3(_first_present(obj, ["center", "position"]))
    if center is None:
        center = _vector3(_first_present(pose, ["center", "position"]))
    if center is None:
        center = _vector3(_first_present(placement, ["center", "position"]))
    asset_ref = obj.get("asset_ref") if isinstance(obj.get("asset_ref"), dict) else {}
    asset_proxy = obj.get("asset_proxy") if isinstance(obj.get("asset_proxy"), dict) else {}
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    size = _vector3(_first_present(obj, ["size", "dimensions"]))
    if size is None:
        size = _vector3(asset_proxy.get("bbox_size"))
    if size is None:
        size = _vector3(metadata.get("transformed_size"))
    if size is None:
        size = _vector3(_first_present(pose, ["size", "dimensions"]))
    if center is None:
        raise ValueError(f"object {object_id!r} center is missing or invalid")
    if size is None or np.any(size <= 0):
        raise ValueError(f"object {object_id!r} size is missing or invalid")

    rotation, yaw_degrees = _rotation_degrees(obj, pose, placement)
    R = rotation_matrix_from_euler(rotation)
    half = size / 2.0
    corners = _obb_corners_from_parts(center, half, R)
    category = obj.get("category") or obj.get("retrieval_category")
    jid = obj.get("jid") or asset_ref.get("asset_key")
    return NormalizedObject(
        id=str(object_id),
        jid=str(jid) if jid is not None else None,
        category=str(category) if category is not None else None,
        center=center,
        size=size,
        half=half,
        rotation=rotation,
        yaw_degrees=float(yaw_degrees),
        R=R,
        right=_unit(R @ np.array([1.0, 0.0, 0.0])),
        front=_unit(R @ np.array([0.0, -1.0, 0.0])),
        up=_unit(R @ np.array([0.0, 0.0, 1.0])),
        bottom_z=float(np.min(corners[:, 2])),
        top_z=float(np.max(corners[:, 2])),
        asset_ref=asset_ref,
        asset_proxy=asset_proxy,
        metadata=metadata,
        interactive=is_interactive_object(obj),
    )


def normalize_objects(scene: dict, *, asset_csv_path: str | None = None, asset_root: str | None = None) -> tuple[list[NormalizedObject], dict[str, str]]:
    objects = []
    errors = {}
    for raw_obj in load_objects(scene):
        object_id = _first_present(raw_obj, ["id", "object_id", "asset_id"])
        try:
            objects.append(normalize_object(raw_obj, asset_csv_path=asset_csv_path, asset_root=asset_root))
        except ValueError as exc:
            if object_id is not None:
                errors[str(object_id)] = str(exc)
    return objects, errors


def is_interactive_object(obj: dict) -> bool:
    if bool(obj.get("interactive")):
        return True
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    if bool(metadata.get("interactive")):
        return True
    asset_ref = obj.get("asset_ref") if isinstance(obj.get("asset_ref"), dict) else {}
    asset_metadata = asset_ref.get("metadata") if isinstance(asset_ref.get("metadata"), dict) else {}
    if bool(asset_metadata.get("interactive")):
        return True
    tags = obj.get("tags")
    return isinstance(tags, list) and any(str(tag).lower() == "interactive" for tag in tags)


def rotation_matrix_from_yaw(yaw_degrees: float) -> np.ndarray:
    yaw = math.radians(float(yaw_degrees))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotation_matrix_from_euler(rotation_degrees: Iterable[float] | np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rotation_degrees]
    rr, pr, yr = np.radians([roll, pitch, yaw])
    cr, sr = math.cos(rr), math.sin(rr)
    cp, sp = math.cos(pr), math.sin(pr)
    cy, sy = math.cos(yr), math.sin(yr)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def get_obb_corners(obj: NormalizedObject) -> np.ndarray:
    return _obb_corners_from_parts(obj.center, obj.half, obj.R)


def get_footprint_corners_xy(obj: NormalizedObject) -> np.ndarray:
    local = np.array(
        [
            [-obj.half[0], -obj.half[1], 0.0],
            [obj.half[0], -obj.half[1], 0.0],
            [obj.half[0], obj.half[1], 0.0],
            [-obj.half[0], obj.half[1], 0.0],
        ],
        dtype=float,
    )
    return (obj.center + local @ obj.R.T)[:, :2]


def sample_footprint_points(obj: NormalizedObject, grid: tuple[int, int] = (3, 3)) -> np.ndarray:
    xs = np.linspace(-obj.half[0], obj.half[0], max(1, int(grid[0])))
    ys = np.linspace(-obj.half[1], obj.half[1], max(1, int(grid[1])))
    local = np.array([[x, y, 0.0] for x in xs for y in ys], dtype=float)
    return (obj.center + local @ obj.R.T)[:, :2]


def sample_bottom_face_points(obj: NormalizedObject, grid: tuple[int, int] = (3, 3)) -> np.ndarray:
    xs = np.linspace(-obj.half[0], obj.half[0], max(1, int(grid[0])))
    ys = np.linspace(-obj.half[1], obj.half[1], max(1, int(grid[1])))
    local = np.array([[x, y, -obj.half[2]] for x in xs for y in ys], dtype=float)
    return obj.center + local @ obj.R.T


def point_in_polygon_2d(point: Iterable[float] | np.ndarray, polygon: Iterable[Iterable[float]] | np.ndarray, eps: float = 1.0e-9) -> bool:
    p = np.asarray(point, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or len(poly) < 3:
        return False
    for i in range(len(poly)):
        if point_segment_distance_2d(p, poly[i], poly[(i + 1) % len(poly)]) <= eps:
            return True
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        yi, yj = poly[i, 1], poly[j, 1]
        xi, xj = poly[i, 0], poly[j, 0]
        if (yi > p[1]) != (yj > p[1]):
            x_intersection = (xj - xi) * (p[1] - yi) / ((yj - yi) + EPS) + xi
            if p[0] < x_intersection:
                inside = not inside
        j = i
    return inside


def point_segment_distance_2d(point: Iterable[float] | np.ndarray, a: Iterable[float] | np.ndarray, b: Iterable[float] | np.ndarray) -> float:
    p = np.asarray(point, dtype=float)
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    segment = bb - aa
    denom = float(np.dot(segment, segment))
    if denom <= EPS:
        return float(np.linalg.norm(p - aa))
    t = float(np.clip(np.dot(p - aa, segment) / denom, 0.0, 1.0))
    projection = aa + t * segment
    return float(np.linalg.norm(p - projection))


def point_polygon_distance_2d(point: Iterable[float] | np.ndarray, polygon: Iterable[Iterable[float]] | np.ndarray) -> float:
    poly = np.asarray(polygon, dtype=float)
    if point_in_polygon_2d(point, poly):
        return 0.0
    return float(min(point_segment_distance_2d(point, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))) if len(poly) else math.inf


def segments_intersect_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1.0e-9) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return _cross2(q - p, r - p)

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    return (
        abs(o1) <= eps
        and on_segment(a, c, b)
        or abs(o2) <= eps
        and on_segment(a, d, b)
        or abs(o3) <= eps
        and on_segment(c, a, d)
        or abs(o4) <= eps
        and on_segment(c, b, d)
    )


def polygon_intersects_polygon_2d(poly_a: Iterable[Iterable[float]] | np.ndarray, poly_b: Iterable[Iterable[float]] | np.ndarray) -> bool:
    a = np.asarray(poly_a, dtype=float)
    b = np.asarray(poly_b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return False
    if any(point_in_polygon_2d(point, b) for point in a):
        return True
    if any(point_in_polygon_2d(point, a) for point in b):
        return True
    return any(segments_intersect_2d(a[i], a[(i + 1) % len(a)], b[j], b[(j + 1) % len(b)]) for i in range(len(a)) for j in range(len(b)))


def polygon_distance_2d(poly_a: Iterable[Iterable[float]] | np.ndarray, poly_b: Iterable[Iterable[float]] | np.ndarray) -> float:
    a = np.asarray(poly_a, dtype=float)
    b = np.asarray(poly_b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return math.inf
    if polygon_intersects_polygon_2d(a, b):
        return 0.0
    distances = []
    for point in a:
        distances.extend(point_segment_distance_2d(point, b[j], b[(j + 1) % len(b)]) for j in range(len(b)))
    for point in b:
        distances.extend(point_segment_distance_2d(point, a[i], a[(i + 1) % len(a)]) for i in range(len(a)))
    return float(min(distances)) if distances else math.inf


def polygon_area(poly: Iterable[Iterable[float]] | np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    if p.ndim != 2 or len(p) < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def signed_polygon_area(poly: Iterable[Iterable[float]] | np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    if p.ndim != 2 or len(p) < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_polygon_intersection_area(poly_a: Iterable[Iterable[float]] | np.ndarray, poly_b: Iterable[Iterable[float]] | np.ndarray) -> float:
    subject = [np.asarray(point, dtype=float) for point in np.asarray(poly_a, dtype=float)]
    clip = [np.asarray(point, dtype=float) for point in np.asarray(poly_b, dtype=float)]
    if len(subject) < 3 or len(clip) < 3:
        return 0.0
    clip_sign = 1.0 if signed_polygon_area(clip) >= 0.0 else -1.0
    output = subject
    for i in range(len(clip)):
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        prev = input_points[-1]
        for curr in input_points:
            curr_inside = _inside_clip(curr, edge_start, edge_end, clip_sign)
            prev_inside = _inside_clip(prev, edge_start, edge_end, clip_sign)
            if curr_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif prev_inside:
                output.append(_line_intersection(prev, curr, edge_start, edge_end))
            prev = curr
    return polygon_area(np.asarray(output, dtype=float)) if len(output) >= 3 else 0.0


def footprint_overlap_area(obj_a: NormalizedObject, obj_b: NormalizedObject) -> float:
    return convex_polygon_intersection_area(get_footprint_corners_xy(obj_a), get_footprint_corners_xy(obj_b))


def footprint_overlap_ratio(obj_a: NormalizedObject, obj_b: NormalizedObject) -> float:
    area = footprint_overlap_area(obj_a, obj_b)
    denom = min(polygon_area(get_footprint_corners_xy(obj_a)), polygon_area(get_footprint_corners_xy(obj_b)))
    return 0.0 if denom <= EPS else float(area / denom)


def footprint_inside_boundary_ratio(obj: NormalizedObject, boundary: Iterable[Iterable[float]] | np.ndarray) -> float:
    polygon = np.asarray(boundary, dtype=float)
    points = list(get_footprint_corners_xy(obj)) + [obj.center[:2]]
    points.extend(sample_footprint_points(obj, (3, 3)))
    inside = sum(1 for point in points if point_in_polygon_2d(point, polygon))
    return float(inside) / float(len(points)) if points else 0.0


def point_in_obb(point: Iterable[float] | np.ndarray, obj: NormalizedObject, eps: float = 0.0) -> bool:
    p = np.asarray(point, dtype=float)
    local = obj.R.T @ (p - obj.center)
    return bool(np.all(np.abs(local) <= obj.half + float(eps)))


def ray_intersects_obb(origin: Iterable[float] | np.ndarray, direction: Iterable[float] | np.ndarray, obj: NormalizedObject) -> dict | None:
    ray_origin = obj.R.T @ (np.asarray(origin, dtype=float) - obj.center)
    ray_dir = obj.R.T @ _unit(np.asarray(direction, dtype=float))
    t_enter = -math.inf
    t_exit = math.inf
    for index in range(3):
        if abs(ray_dir[index]) <= EPS:
            if ray_origin[index] < -obj.half[index] or ray_origin[index] > obj.half[index]:
                return None
            continue
        t1 = (-obj.half[index] - ray_origin[index]) / ray_dir[index]
        t2 = (obj.half[index] - ray_origin[index]) / ray_dir[index]
        t_near, t_far = min(t1, t2), max(t1, t2)
        t_enter = max(t_enter, t_near)
        t_exit = min(t_exit, t_far)
        if t_enter > t_exit:
            return None
    if t_exit < 0.0:
        return None
    t_hit = max(t_enter, 0.0)
    point = np.asarray(origin, dtype=float) + t_hit * np.asarray(direction, dtype=float)
    return {"hit": True, "t_enter": float(t_hit), "point": point}


def vertical_ray_hit_top_surface(origin_xy: Iterable[float] | np.ndarray, z_start: float, target_obj: NormalizedObject) -> dict | None:
    xy = np.asarray(origin_xy, dtype=float)
    if target_obj.top_z > float(z_start) + EPS:
        return None
    if point_in_polygon_2d(xy, get_footprint_corners_xy(target_obj)):
        return {"hit": True, "z": target_obj.top_z, "point": np.array([xy[0], xy[1], target_obj.top_z]), "object_id": target_obj.id}
    return None


def z_interval_overlap(obj_a: NormalizedObject, obj_b: NormalizedObject) -> float:
    return float(max(0.0, min(obj_a.top_z, obj_b.top_z) - max(obj_a.bottom_z, obj_b.bottom_z)))


def _rotation_degrees(obj: dict, pose: dict, placement: dict) -> tuple[np.ndarray, float]:
    rotation_value = _first_present(obj, ["rotation"])
    if rotation_value is None:
        rotation_value = _first_present(pose, ["rotation"])
    if rotation_value is None:
        yaw_degrees = _first_present(obj, ["yaw_degrees"])
        if yaw_degrees is None:
            yaw_degrees = _first_present(placement, ["yaw_degrees"])
        if yaw_degrees is not None:
            yaw = _safe_float(yaw_degrees, 0.0)
            return np.array([0.0, 0.0, yaw], dtype=float), yaw
        yaw = _first_present(obj, ["yaw"])
        if yaw is None:
            yaw = _first_present(placement, ["yaw"])
        yaw_deg = _angle_to_degrees(_safe_float(yaw, 0.0))
        return np.array([0.0, 0.0, yaw_deg], dtype=float), yaw_deg
    vector = _vector3(rotation_value)
    if vector is None:
        vector = np.zeros(3, dtype=float)
    max_abs = float(np.max(np.abs(vector))) if vector.size else 0.0
    if max_abs <= (2.0 * math.pi + 1.0e-6):
        vector = np.degrees(vector)
    return vector.astype(float), float(vector[2])


def _angle_to_degrees(value: float) -> float:
    return math.degrees(value) if abs(value) <= (2.0 * math.pi + 1.0e-6) else value


def _obb_corners_from_parts(center: np.ndarray, half: np.ndarray, R: np.ndarray) -> np.ndarray:
    local = np.array(
        [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=float,
    )
    return center + local @ R.T


def _inside_clip(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray, clip_sign: float) -> bool:
    return clip_sign * _cross2(edge_end - edge_start, point - edge_start) >= -1.0e-9


def _line_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = q2 - q1
    denom = _cross2(r, s)
    if abs(denom) <= EPS:
        return p2
    t = _cross2(q1 - p1, s) / denom
    return p1 + t * r


def _boundary_list(value: object) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            continue
    return points


def _vector3(value: object | None) -> np.ndarray | None:
    if isinstance(value, np.ndarray) and value.shape == (3,):
        return value.astype(float)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)
        except (TypeError, ValueError):
            return None
    return None


def _first_present(obj: dict, keys: list[str]) -> object | None:
    for key in keys:
        if isinstance(obj, dict) and key in obj and obj[key] is not None:
            return obj[key]
    return None


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector, dtype=float)
    return vector.astype(float) / norm


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
