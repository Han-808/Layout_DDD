from __future__ import annotations

from itertools import combinations

from benchmark.evaluator.generic_validity.geometry import (
    NormalizedObject,
    footprint_overlap_area,
    footprint_overlap_ratio,
    get_obb_corners,
    normalize_objects,
    point_in_obb,
    z_interval_overlap,
)


def check_collision(scene: dict, config: dict | None = None) -> dict:
    cfg = config or {}
    objects, object_errors = normalize_objects(scene)
    num_objects = len(objects)
    if num_objects == 0:
        return {
            "metric": "collision",
            "status": "not_applicable",
            "score": 1.0,
            "collision_count": 0,
            "collision_pair_count": 0,
            "collision_object_count": 0,
            "collision_rate": 0.0,
            "num_objects": 0,
            "pairs": [],
            "object_errors": object_errors,
        }

    z_overlap_eps = float(cfg.get("z_overlap_eps", 0.03))
    xy_overlap_area_eps = float(cfg.get("xy_overlap_area_eps", 0.005))
    ignore_exemptions = bool(cfg.get("ignore_supported_or_contained_pairs", True))
    collision_pairs = []
    collision_object_ids: set[str] = set()

    for obj_a, obj_b in combinations(objects, 2):
        xy_overlap = footprint_overlap_area(obj_a, obj_b)
        z_overlap = z_interval_overlap(obj_a, obj_b)
        if xy_overlap <= xy_overlap_area_eps or z_overlap <= z_overlap_eps:
            continue
        exempted, exemption_reason = _is_exempt_pair(obj_a, obj_b, cfg) if ignore_exemptions else (False, "")
        pair = {
            "object_a": obj_a.id,
            "object_b": obj_b.id,
            "xy_overlap_area": float(xy_overlap),
            "z_overlap": float(z_overlap),
            "exempted": bool(exempted),
        }
        if exemption_reason:
            pair["exemption_reason"] = exemption_reason
        collision_pairs.append(pair)
        if not exempted:
            collision_object_ids.update([obj_a.id, obj_b.id])

    collision_count = sum(1 for pair in collision_pairs if not pair["exempted"])
    collision_rate = min(float(collision_count) / float(max(num_objects, 1)), 1.0)
    return {
        "metric": "collision",
        "status": "checked",
        "score": float(1.0 - collision_rate),
        "collision_count": collision_count,
        "collision_pair_count": collision_count,
        "collision_object_count": len(collision_object_ids),
        "collision_rate": collision_rate,
        "num_objects": num_objects,
        "pairs": collision_pairs,
        "object_errors": object_errors,
    }


def _is_exempt_pair(obj_a: NormalizedObject, obj_b: NormalizedObject, config: dict) -> tuple[bool, str]:
    support_gap = float(config.get("support_gap", 0.06))
    overlap_ratio = footprint_overlap_ratio(obj_a, obj_b)
    if abs(obj_a.bottom_z - obj_b.top_z) <= support_gap and overlap_ratio > 0.1:
        return True, f"{obj_a.id}_supported_by_{obj_b.id}"
    if abs(obj_b.bottom_z - obj_a.top_z) <= support_gap and overlap_ratio > 0.1:
        return True, f"{obj_b.id}_supported_by_{obj_a.id}"
    if _mostly_within(obj_a, obj_b):
        return True, f"{obj_a.id}_mostly_within_{obj_b.id}"
    if _mostly_within(obj_b, obj_a):
        return True, f"{obj_b.id}_mostly_within_{obj_a.id}"
    return False, ""


def _mostly_within(inner: NormalizedObject, outer: NormalizedObject, threshold: float = 0.80) -> bool:
    points = list(get_obb_corners(inner)) + [inner.center]
    inside = sum(1 for point in points if point_in_obb(point, outer, eps=1.0e-6))
    return bool(points) and float(inside) / float(len(points)) >= threshold
