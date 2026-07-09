from __future__ import annotations

import numpy as np

from benchmark.evaluator.generic_validity.geometry import (
    footprint_inside_boundary_ratio,
    get_room_boundary,
    get_scene_height,
    normalize_objects,
)


def check_oob(scene: dict, config: dict | None = None) -> dict:
    cfg = config or {}
    boundary = np.asarray(get_room_boundary(scene), dtype=float)
    if boundary.ndim != 2 or len(boundary) < 3:
        return {
            "metric": "oob",
            "status": "invalid_input",
            "score": 0.0,
            "reason": "scene boundary is missing or invalid",
            "oob_count": 0,
            "oob_rate": 0.0,
            "num_objects": 0,
            "objects": [],
        }
    objects, object_errors = normalize_objects(scene)
    if not objects:
        return {
            "metric": "oob",
            "status": "not_applicable",
            "score": 1.0,
            "oob_count": 0,
            "oob_rate": 0.0,
            "num_objects": 0,
            "objects": [],
            "object_errors": object_errors,
        }

    inside_ratio_threshold = float(cfg.get("inside_ratio_threshold", 0.98))
    floor_eps = float(cfg.get("floor_eps", 0.05))
    check_height = bool(cfg.get("check_height", True))
    height_eps = float(cfg.get("height_eps", 0.05))
    scene_height = get_scene_height(scene)
    records = []
    oob_count = 0
    for obj in objects:
        inside_ratio = footprint_inside_boundary_ratio(obj, boundary)
        boundary_oob = inside_ratio < inside_ratio_threshold
        floor_oob = obj.bottom_z < -floor_eps
        height_oob = bool(check_height and scene_height is not None and obj.top_z > scene_height + height_eps)
        is_oob = boundary_oob or floor_oob or height_oob
        if is_oob:
            oob_count += 1
        records.append(
            {
                "object_id": obj.id,
                "inside_ratio": float(inside_ratio),
                "boundary_oob": bool(boundary_oob),
                "floor_oob": bool(floor_oob),
                "height_oob": bool(height_oob),
                "bottom_z": float(obj.bottom_z),
                "top_z": float(obj.top_z),
            }
        )

    num_objects = len(objects)
    oob_rate = float(oob_count) / float(max(num_objects, 1))
    return {
        "metric": "oob",
        "status": "checked",
        "score": float(1.0 - oob_rate),
        "oob_count": oob_count,
        "oob_rate": oob_rate,
        "num_objects": num_objects,
        "objects": records,
        "object_errors": object_errors,
    }
