from __future__ import annotations

import math
from statistics import mean
from typing import Any


DEFAULT_RENDER_CONFIG = {
    "backend": "perspective_bbox_zbuffer",
    "width": 640,
    "height": 480,
    "fov_degrees": 50.0,
    "near": 0.05,
    "far": 200.0,
    "min_visible_pixel_area": 4.0,
    "canvas_margin_px": 20,
    "background_rgba": [248, 249, 246, 255],
    "min_camera_distance_m": 2.5,
    "distance_scale": 1.5,
    "fit_span_scale": 0.75,
    "min_target_z": 0.35,
    "top_look_at_shift_scale": 0.25,
    "side_look_at_shift_scale": 0.3,
    "camera_candidates": [[0.0, 0.0], [0.25, 0.0], [-0.25, 0.0], [0.0, 0.25], [0.0, -0.25]],
}

DEFAULT_VIEW_VALIDATION_CONFIG = {
    "min_foreground_ratio": 0.005,
    "min_visible_object_ratio": 0.4,
    "max_camera_retries": 4,
}

DEFAULT_PHYSICAL_FLAGS_CONFIG = {
    "boundary_tolerance_m": 0.35,
    "boundary_tolerance_rel_object_size": 0.25,
    "floor_contact_tolerance_m": 0.15,
    "floor_contact_tolerance_rel_height": 0.10,
    "above_wall_tolerance_m": 0.20,
    "serious_collision_overlap_ratio": 0.60,
    "serious_collision_min_volume": {
        "abs_min_volume_m3": 0.002,
        "object_volume_ratio": 0.01,
        "scene_volume_ratio": 0.0001,
        "min_cap_m3": 0.002,
        "max_cap_m3": 0.05,
    },
    "floor_z_default": 0.0,
}


def resolve_runtime_evidence_config(
    benchmark_config: dict | None,
    bm_instance: dict,
    layout: dict,
    object_group: dict | None = None,
) -> dict:
    """Resolve evidence-generation defaults from config and current scene scale."""

    render = _merge(DEFAULT_RENDER_CONFIG, _section(benchmark_config, "render"))
    view_validation = _merge(DEFAULT_VIEW_VALIDATION_CONFIG, _section(benchmark_config, "view_validation"))
    physical = _merge(DEFAULT_PHYSICAL_FLAGS_CONFIG, _section(benchmark_config, "physical_flags"))

    room = bm_instance.get("room") if isinstance(bm_instance.get("room"), dict) else {}
    boundary = _room_boundary(room)
    objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
    target_ids = set(object_group.get("object_ids", [])) if isinstance(object_group, dict) else set()
    group_objects = [obj for obj in objects if str(obj.get("object_id") or obj.get("id") or "") in target_ids] if target_ids else objects

    room_extent = _boundary_extent(boundary)
    room_height = _number(room.get("wall_height"), 0.0)
    scene_bounds = _object_bounds(objects, boundary)
    group_bounds = _object_bounds(group_objects, [])
    scene_extent = _extent(scene_bounds)
    group_extent = _extent(group_bounds)
    group_center = _center(group_bounds)
    object_stats = _object_scale_stats(group_objects or objects)

    scale_extent = max(group_extent, object_stats["mean_diag_m"], 1.0)
    camera_distance = max(float(render["min_camera_distance_m"]), scale_extent * float(render["distance_scale"]))
    far = max(float(render["far"]), camera_distance + max(room_extent, scene_extent, group_extent, 1.0) * 2.0)
    near = min(float(render["near"]), max(0.01, camera_distance / 1000.0))
    target_z = max(float(render["min_target_z"]), group_center[2] if group_objects else _center(scene_bounds)[2])

    return {
        "room_extent_m": room_extent,
        "room_height_m": room_height,
        "scene_bbox_extent_m": scene_extent,
        "group_extent_m": group_extent,
        "group_center": _round_list(group_center),
        "object_scale": object_stats,
        "render": {
            **render,
            "near": near,
            "far": far,
            "effective_camera_distance_m": camera_distance,
            "effective_target_z": target_z,
            "camera_candidates": _camera_candidates(render.get("camera_candidates")),
            "background_rgba": _rgba(render.get("background_rgba")),
        },
        "view_validation": view_validation,
        "physical_flags": {
            **physical,
            "effective_floor_contact_tolerance_m": max(
                float(physical["floor_contact_tolerance_m"]),
                object_stats["mean_height_m"] * float(physical["floor_contact_tolerance_rel_height"]),
            ),
            "effective_boundary_tolerance_m": max(
                float(physical["boundary_tolerance_m"]),
                max(object_stats["mean_width_m"], object_stats["mean_depth_m"]) * float(physical["boundary_tolerance_rel_object_size"]),
            ),
        },
    }


def effective_floor_contact_tolerance(physical_config: dict, obj: dict) -> float:
    size = obj.get("size")
    height = float(size[2]) if isinstance(size, list) and len(size) == 3 and isinstance(size[2], (int, float)) else 0.0
    return max(
        float(physical_config["floor_contact_tolerance_m"]),
        height * float(physical_config["floor_contact_tolerance_rel_height"]),
    )


def effective_boundary_tolerance(physical_config: dict, obj: dict) -> float:
    size = obj.get("size")
    width = float(size[0]) if isinstance(size, list) and len(size) == 3 and isinstance(size[0], (int, float)) else 0.0
    depth = float(size[1]) if isinstance(size, list) and len(size) == 3 and isinstance(size[1], (int, float)) else 0.0
    return max(
        float(physical_config["boundary_tolerance_m"]),
        max(width, depth) * float(physical_config["boundary_tolerance_rel_object_size"]),
    )


def effective_scale_aware_min_volume(
    volume_config: dict | None,
    *,
    smaller_object_volume_m3: float | None,
    scene_volume_m3: float | None,
) -> dict:
    cfg = volume_config if isinstance(volume_config, dict) else {}
    abs_min = _nonnegative_number(cfg.get("abs_min_volume_m3"), 0.0)
    object_ratio = _nonnegative_number(cfg.get("object_volume_ratio"), 0.0)
    scene_ratio = _nonnegative_number(cfg.get("scene_volume_ratio"), 0.0)
    min_cap = _nonnegative_number(cfg.get("min_cap_m3"), abs_min)
    max_cap = _nonnegative_number(cfg.get("max_cap_m3"), max(abs_min, min_cap))
    if max_cap < min_cap:
        max_cap = min_cap

    valid_object_volume = _positive_number(smaller_object_volume_m3)
    valid_scene_volume = _positive_number(scene_volume_m3)
    object_term = object_ratio * valid_object_volume if valid_object_volume is not None else None
    scene_term = scene_ratio * valid_scene_volume if valid_scene_volume is not None else None

    candidates = [abs_min]
    if object_term is not None:
        candidates.append(object_term)
    if scene_term is not None:
        candidates.append(scene_term)
    raw_threshold = max(candidates)
    threshold = _clamp(raw_threshold, min_cap, max_cap)
    return {
        "threshold_source": "scale_aware",
        "effective_min_collision_volume_m3": threshold,
        "abs_min_volume_m3": abs_min,
        "object_volume_term_m3": object_term,
        "scene_volume_term_m3": scene_term,
        "smaller_object_volume_m3": valid_object_volume,
        "scene_volume_m3": valid_scene_volume,
        "min_cap_m3": min_cap,
        "max_cap_m3": max_cap,
    }


def scene_volume_m3_from_case(case: dict | None) -> float | None:
    room = (case or {}).get("room") if isinstance((case or {}).get("room"), dict) else {}
    floor_area = room_floor_area_m2(room)
    height = _positive_number(room.get("wall_height")) or _positive_number(room.get("height"))
    if floor_area is None or height is None:
        return None
    return floor_area * height


def room_floor_area_m2(room: dict | None) -> float | None:
    if not isinstance(room, dict):
        return None
    floor_plan = room.get("floor_plan") if isinstance(room.get("floor_plan"), dict) else {}
    regions = floor_plan.get("regions") or room.get("regions") or []
    if isinstance(regions, list):
        area = 0.0
        for region in regions:
            if not isinstance(region, dict):
                continue
            polygon_area = _polygon_area(region.get("floor_polygon"))
            if polygon_area is not None:
                area += polygon_area
        if area > 0:
            return area
    return _polygon_area(room.get("floor_polygon") or room.get("boundary"))


def _section(config: dict | None, name: str) -> dict:
    section = (config or {}).get(name)
    return section if isinstance(section, dict) else {}


def _merge(defaults: dict, overrides: dict) -> dict:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _room_boundary(room: dict) -> list[list[float]]:
    boundary = room.get("floor_polygon") or room.get("boundary") or []
    return [point for point in boundary if isinstance(point, list) and len(point) >= 2] if isinstance(boundary, list) else []


def _boundary_extent(boundary: list[list[float]]) -> float:
    if not boundary:
        return 0.0
    xs = [float(point[0]) for point in boundary]
    ys = [float(point[1]) for point in boundary]
    return max(max(xs) - min(xs), max(ys) - min(ys), 0.0)


def _object_bounds(objects: list[dict], boundary: list[list[float]]) -> tuple[float, float, float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for point in boundary:
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    for obj in objects:
        center = obj.get("center")
        size = obj.get("size")
        if not isinstance(center, list) or not isinstance(size, list) or len(center) != 3 or len(size) != 3:
            continue
        cx, cy, cz = [float(value) for value in center]
        sx, sy, sz = [float(value) for value in size]
        xs.extend([cx - sx / 2.0, cx + sx / 2.0])
        ys.extend([cy - sy / 2.0, cy + sy / 2.0])
        zs.extend([cz - sz / 2.0, cz + sz / 2.0])
    if not xs:
        xs = [0.0, 1.0]
    if not ys:
        ys = [0.0, 1.0]
    if not zs:
        zs = [0.0, 1.0]
    return min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)


def _extent(bounds: tuple[float, float, float, float, float, float]) -> float:
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    return max(max_x - min_x, max_y - min_y, max_z - min_z, 0.0)


def _center(bounds: tuple[float, float, float, float, float, float]) -> list[float]:
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    return [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0]


def _object_scale_stats(objects: list[dict]) -> dict:
    widths = []
    depths = []
    heights = []
    diags = []
    for obj in objects:
        size = obj.get("size")
        if not isinstance(size, list) or len(size) != 3:
            continue
        w, d, h = [float(value) for value in size]
        widths.append(w)
        depths.append(d)
        heights.append(h)
        diags.append(math.sqrt(w * w + d * d + h * h))
    return {
        "count": len(diags),
        "mean_width_m": _mean(widths),
        "mean_depth_m": _mean(depths),
        "mean_height_m": _mean(heights),
        "mean_diag_m": _mean(diags),
        "max_diag_m": max(diags, default=0.0),
    }


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _nonnegative_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number < 0:
        return default
    return number


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _polygon_area(value: object) -> float | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    points = []
    for point in value:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    if len(points) < 3:
        return None
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _camera_candidates(value: object) -> list[list[float]]:
    if not isinstance(value, list):
        return list(DEFAULT_RENDER_CONFIG["camera_candidates"])
    candidates = []
    for item in value:
        if isinstance(item, list) and len(item) == 2 and all(isinstance(v, (int, float)) for v in item):
            candidates.append([float(item[0]), float(item[1])])
    return candidates or list(DEFAULT_RENDER_CONFIG["camera_candidates"])


def _rgba(value: object) -> list[int]:
    if isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return [int(v) for v in value]
    return list(DEFAULT_RENDER_CONFIG["background_rgba"])


def _round_list(values: list[float]) -> list[float]:
    return [round(float(value), 4) for value in values]
