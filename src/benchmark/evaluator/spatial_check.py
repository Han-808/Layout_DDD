from __future__ import annotations

import math
from typing import Callable

import numpy as np

from benchmark.evaluator.geometry import (
    bbox_z_bounds,
    center_xy,
    distance_xy,
    footprint_intersection_area,
    footprint_polygon,
    horizontal_overlap_ratio,
    room_polygon,
    sorted_unique,
)


RelationEvaluator = Callable[[dict, dict | str | None, dict, dict], tuple[bool, str]]


def check_spatial_relations(layout: dict, bm_instance: dict, config: dict | None = None) -> tuple[bool, list[dict], list[str]]:
    cfg = (config or {}).get("spatial", config or {})
    objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
    by_id = {obj["object_id"]: obj for obj in objects}
    by_category: dict[str, list[dict]] = {}
    for obj in objects:
        by_category.setdefault(obj["category"], []).append(obj)

    failures: list[dict] = []
    repair_targets: list[str] = []

    if cfg.get("check_required_objects", True):
        for category in bm_instance.get("required_objects") or []:
            if category not in by_category:
                failures.append(
                    {
                        "type": "missing_required_object",
                        "category": category,
                        "message": f"Required object category '{category}' is missing.",
                    }
                )

    for relation in layout.get("relations", []):
        if not isinstance(relation, dict):
            continue
        source = by_id.get(relation.get("source"))
        if source is None:
            failures.append(
                {
                    "type": "missing_relation_source",
                    "message": f"Relation source {relation.get('source')} is missing.",
                }
            )
            continue
        target = by_id.get(relation.get("target")) or relation.get("target")
        ok, message = _evaluate_relation(relation.get("type"), source, target, bm_instance, cfg)
        if not ok:
            failure = {
                "type": relation.get("type", "unknown_relation"),
                "objects": _relation_objects(source, target),
                "message": message,
                "hard": bool(relation.get("hard", False)),
            }
            failures.append(failure)
            repair_targets.extend(failure["objects"])

    for constraint in bm_instance.get("spatial_constraints") or []:
        source_candidates = by_category.get(constraint.get("source_category"), [])
        target_candidates = _target_candidates(constraint, by_category)
        if not source_candidates:
            failures.append(
                {
                    "type": "missing_constraint_source",
                    "category": constraint.get("source_category"),
                    "message": f"No object satisfies source_category {constraint.get('source_category')}.",
                    "hard": bool(constraint.get("hard", False)),
                }
            )
            continue

        relation_type = constraint["type"]
        passed = False
        last_message = ""
        for source in source_candidates:
            candidates = target_candidates or [constraint.get("target")]
            for target in candidates:
                ok, message = _evaluate_relation(relation_type, source, target, bm_instance, cfg)
                passed = passed or ok
                last_message = message
                if passed:
                    break
            if passed:
                break

        if not passed:
            objects_in_failure = [source_candidates[0]["object_id"]]
            if target_candidates and isinstance(target_candidates[0], dict):
                objects_in_failure.append(target_candidates[0]["object_id"])
            failures.append(
                {
                    "type": relation_type,
                    "objects": objects_in_failure,
                    "message": last_message or f"Constraint {relation_type} is not satisfied.",
                    "hard": bool(constraint.get("hard", False)),
                }
            )
            repair_targets.extend(objects_in_failure)

    failures = _dedupe_failures(failures)
    return not failures, failures, sorted_unique(repair_targets)


def _target_candidates(constraint: dict, by_category: dict[str, list[dict]]) -> list[dict] | None:
    target_category = constraint.get("target_category")
    if target_category:
        return by_category.get(target_category, [])
    return None


def _relation_objects(source: dict, target: dict | str | None) -> list[str]:
    objects = [source["object_id"]]
    if isinstance(target, dict):
        objects.append(target["object_id"])
    return objects


def _evaluate_relation(
    relation_type: str | None,
    source: dict,
    target: dict | str | None,
    bm_instance: dict,
    cfg: dict,
) -> tuple[bool, str]:
    if relation_type == "near":
        if not isinstance(target, dict):
            return False, "near relation requires an object target."
        distance = distance_xy(source, target)
        threshold = float(cfg.get("near_distance", 1.0))
        return distance <= threshold, f"{source['object_id']} is {distance:.3f}m from {target['object_id']}, above near threshold {threshold:.3f}m."

    if relation_type == "far":
        if not isinstance(target, dict):
            return False, "far relation requires an object target."
        distance = distance_xy(source, target)
        threshold = float(cfg.get("far_distance", 2.0))
        return distance >= threshold, f"{source['object_id']} is {distance:.3f}m from {target['object_id']}, below far threshold {threshold:.3f}m."

    if relation_type == "facing":
        if not isinstance(target, dict):
            return False, "facing relation requires an object target."
        angle = _facing_angle_deg(source, target)
        threshold = float(cfg.get("facing_angle_tolerance_deg", 35.0))
        return angle <= threshold, f"{source['object_id']} facing angle to {target['object_id']} is {angle:.1f}deg, above tolerance {threshold:.1f}deg."

    if relation_type == "against_wall":
        return _against_wall(source, str(target or "any_wall"), bm_instance, cfg)

    if relation_type == "on_top_of":
        if not isinstance(target, dict):
            return False, "on_top_of relation requires an object target."
        z_tol = float(cfg.get("on_top_z_tolerance", 0.05))
        min_overlap = float(cfg.get("min_on_top_overlap_ratio", 0.25))
        z_min, _ = bbox_z_bounds(source)
        _, z_target_max = bbox_z_bounds(target)
        overlap_ratio = horizontal_overlap_ratio(source, target)
        ok = abs(z_min - z_target_max) <= z_tol and overlap_ratio >= min_overlap
        return ok, f"{source['object_id']} is not on top of {target['object_id']} within z/overlap thresholds."

    if relation_type in {"left_of", "right_of", "in_front_of", "behind"}:
        if not isinstance(target, dict):
            return False, f"{relation_type} relation requires an object target."
        return _directional_relation(relation_type, source, target, cfg)

    return False, f"Unsupported relation type {relation_type}."


def _facing_angle_deg(source: dict, target: dict) -> float:
    yaw = math.radians(float(source.get("yaw", 0.0)))
    facing = np.array([math.sin(yaw), math.cos(yaw)], dtype=float)
    direction = center_xy(target) - center_xy(source)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        return 180.0
    direction = direction / norm
    dot = float(np.clip(np.dot(facing, direction), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _against_wall(source: dict, target_wall: str, bm_instance: dict, cfg: dict) -> tuple[bool, str]:
    room = bm_instance.get("room")
    if not room:
        return False, "against_wall relation requires a room boundary."

    polygon = room_polygon(room)
    min_x, min_y, max_x, max_y = polygon.bounds
    bounds = footprint_polygon(source).bounds
    distances = {
        "west_wall": abs(bounds[0] - min_x),
        "east_wall": abs(max_x - bounds[2]),
        "south_wall": abs(bounds[1] - min_y),
        "north_wall": abs(max_y - bounds[3]),
    }
    threshold = float(cfg.get("against_wall_distance", cfg.get("wall_contact_tolerance", 0.2)))
    if target_wall == "any_wall":
        distance = min(distances.values())
        ok = distance <= threshold
        return ok, f"{source['object_id']} is {distance:.3f}m from nearest wall, above threshold {threshold:.3f}m."
    if target_wall not in distances:
        return False, f"Unknown wall target {target_wall}."
    distance = distances[target_wall]
    ok = distance <= threshold
    return ok, f"{source['object_id']} is {distance:.3f}m from {target_wall}, above threshold {threshold:.3f}m."


def _directional_relation(relation_type: str, source: dict, target: dict, cfg: dict) -> tuple[bool, str]:
    source_center = center_xy(source)
    target_center = center_xy(target)
    x_delta = source_center[0] - target_center[0]
    y_delta = source_center[1] - target_center[1]
    lr_delta = float(cfg.get("left_right_min_delta", 0.1))
    fb_delta = float(cfg.get("front_back_min_delta", 0.1))
    checks = {
        "left_of": x_delta < -lr_delta,
        "right_of": x_delta > lr_delta,
        "in_front_of": y_delta < -fb_delta,
        "behind": y_delta > fb_delta,
    }
    ok = checks[relation_type]
    return ok, f"{source['object_id']} does not satisfy {relation_type} relative to {target['object_id']}."


def _dedupe_failures(failures: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for failure in failures:
        key = (
            failure.get("type"),
            tuple(failure.get("objects", [])),
            failure.get("category"),
            failure.get("message"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(failure)
    return deduped
