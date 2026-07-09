from __future__ import annotations

import numpy as np

from benchmark.evaluator.generic_validity.geometry import (
    get_footprint_corners_xy,
    get_room_boundary,
    get_scene_height,
    normalize_objects,
    point_in_polygon_2d,
    point_polygon_distance_2d,
    point_segment_distance_2d,
    sample_bottom_face_points,
    vertical_ray_hit_top_surface,
)


def check_support(scene: dict, config: dict | None = None) -> dict:
    cfg = config or {}
    objects, object_errors = normalize_objects(scene)
    if not objects:
        return {
            "metric": "support",
            "status": "not_applicable",
            "score": 1.0,
            "num_objects": 0,
            "supported_count": 0,
            "unsupported_count": 0,
            "objects": [],
            "object_errors": object_errors,
        }

    boundary = np.asarray(get_room_boundary(scene), dtype=float)
    has_boundary = boundary.ndim == 2 and len(boundary) >= 3
    floor_eps = float(cfg.get("floor_eps", 0.05))
    support_gap = float(cfg.get("support_gap", 0.06))
    sink_tolerance = float(cfg.get("sink_tolerance", 0.05))
    grid = cfg.get("bottom_sample_grid", [3, 3])
    sample_grid = (int(grid[0]), int(grid[1])) if isinstance(grid, list) and len(grid) >= 2 else (3, 3)
    support_ratio_threshold = float(cfg.get("support_ratio_threshold", 0.30))
    allow_wall_support = bool(cfg.get("allow_wall_support_proxy", True))
    records = []
    supported_count = 0

    for obj in objects:
        samples = sample_bottom_face_points(obj, sample_grid)
        object_wall_support = _wall_support_proxy(obj, scene, cfg) if allow_wall_support else False
        supported_samples = 0
        support_sources: set[str] = set()
        supporting_objects: set[str] = set()
        sinking = obj.bottom_z < -sink_tolerance

        for sample in samples:
            sample_xy = sample[:2]
            sample_z = float(sample[2])
            sample_supported = False
            if abs(sample_z) <= floor_eps and (not has_boundary or point_in_polygon_2d(sample_xy, boundary)):
                sample_supported = True
                support_sources.add("floor")
            for other in objects:
                if other.id == obj.id:
                    continue
                hit = vertical_ray_hit_top_surface(sample_xy, sample_z, other)
                if hit is not None and sample_z - float(hit["z"]) <= support_gap:
                    sample_supported = True
                    support_sources.add("object")
                    supporting_objects.add(other.id)
                if point_in_polygon_2d(sample_xy, get_footprint_corners_xy(other)) and sample_z < other.top_z - sink_tolerance:
                    sinking = True
            if object_wall_support:
                sample_supported = True
                support_sources.add("wall_proxy")
            if sample_supported:
                supported_samples += 1

        total_samples = len(samples)
        support_ratio = float(supported_samples) / float(max(total_samples, 1))
        supported = support_ratio >= support_ratio_threshold and not sinking
        if supported:
            supported_count += 1
        records.append(
            {
                "object_id": obj.id,
                "supported": bool(supported),
                "support_ratio": support_ratio,
                "support_sources": sorted(support_sources),
                "supporting_objects": sorted(supporting_objects),
                "sinking": bool(sinking),
                "wall_support_proxy": bool(object_wall_support),
            }
        )

    num_objects = len(objects)
    return {
        "metric": "support",
        "status": "checked",
        "score": float(supported_count) / float(max(num_objects, 1)),
        "num_objects": num_objects,
        "supported_count": supported_count,
        "unsupported_count": num_objects - supported_count,
        "objects": records,
        "object_errors": object_errors,
    }


def _wall_support_proxy(obj: object, scene: dict, config: dict) -> bool:
    boundary = np.asarray(get_room_boundary(scene), dtype=float)
    if boundary.ndim != 2 or len(boundary) < 3:
        return False
    wall_support_distance = float(config.get("wall_support_distance", 0.08))
    scene_height = get_scene_height(scene)
    if scene_height is not None and obj.top_z > scene_height + float(config.get("eps_z", 0.05)):
        return False
    footprint = get_footprint_corners_xy(obj)
    min_distance = min(
        point_segment_distance_2d(point, boundary[index], boundary[(index + 1) % len(boundary)])
        for point in footprint
        for index in range(len(boundary))
    )
    return float(min_distance) <= wall_support_distance
