from __future__ import annotations

import numpy as np

from benchmark.evaluator.generic_validity.geometry import (
    get_footprint_corners_xy,
    normalize_objects,
    point_polygon_distance_2d,
)
from benchmark.evaluator.generic_validity.navigability import compute_navigability_grid


def check_accessibility(scene: dict, config: dict | None = None, navigability_cache: dict | None = None) -> dict:
    cfg = config or {}
    objects, object_errors = normalize_objects(scene)
    interactive_objects = [obj for obj in objects if obj.interactive]
    if not interactive_objects:
        return {
            "metric": "accessibility",
            "status": "not_applicable",
            "score": 1.0,
            "interactive_count": 0,
            "accessible_count": 0,
            "inaccessible_count": 0,
            "objects": [],
            "object_errors": object_errors,
        }

    cache = navigability_cache or compute_navigability_grid(scene, config)
    if cache.get("status") != "checked":
        return {
            "metric": "accessibility",
            "status": "invalid_input",
            "score": 0.0,
            "reason": cache.get("reason", "navigability grid could not be computed"),
            "interactive_count": len(interactive_objects),
            "accessible_count": 0,
            "inaccessible_count": len(interactive_objects),
            "objects": [],
            "object_errors": object_errors,
        }

    access_radius = float(cfg.get("access_radius", 0.45))
    require_largest = bool(cfg.get("require_largest_component", True))
    x_centers = cache["x_centers"]
    y_centers = cache["y_centers"]
    free = cache["free"]
    labels = cache["component_labels"]
    largest_id = int(cache["largest_component_id"])
    records = []
    accessible_count = 0
    for obj in interactive_objects:
        footprint = get_footprint_corners_xy(obj)
        candidate_count = 0
        accessible = False
        for row, y in enumerate(y_centers):
            for col, x in enumerate(x_centers):
                if not free[row, col]:
                    continue
                if require_largest and labels[row, col] != largest_id:
                    continue
                point = np.array([x, y], dtype=float)
                if point_polygon_distance_2d(point, footprint) <= access_radius:
                    candidate_count += 1
                    accessible = True
        if accessible:
            accessible_count += 1
        records.append({"object_id": obj.id, "accessible": bool(accessible), "candidate_access_cells": candidate_count})

    interactive_count = len(interactive_objects)
    score = float(accessible_count) / float(interactive_count)
    return {
        "metric": "accessibility",
        "status": "checked",
        "score": score,
        "interactive_count": interactive_count,
        "accessible_count": accessible_count,
        "inaccessible_count": interactive_count - accessible_count,
        "objects": records,
        "object_errors": object_errors,
    }
