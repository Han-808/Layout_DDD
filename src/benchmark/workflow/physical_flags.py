from __future__ import annotations

from benchmark.evidence_config import (
    effective_boundary_tolerance,
    effective_floor_contact_tolerance,
    effective_scale_aware_min_volume,
    resolve_runtime_evidence_config,
    scene_volume_m3_from_case,
)
from benchmark.evaluator.geometry import bbox_z_bounds, footprint_inside_room, footprint_intersection_area, footprint_polygon, vertical_overlap


def collect_physical_flags(layout: dict, case: dict, config: dict | None = None) -> list[dict]:
    resolved = resolve_runtime_evidence_config(config, case, layout)
    cfg = resolved["physical_flags"]
    policy = resolved.get("physical_flag_policy", {})
    vertical_cfg = resolved.get("vertical_consistency", {})
    collision_ratio_threshold = float(cfg["serious_collision_overlap_ratio"])
    collision_volume_config = cfg.get("serious_collision_min_volume")
    scene_volume = scene_volume_m3_from_case(case)

    room = _room_with_floor_polygon(case.get("room") or {})
    floor_z = float(room.get("floor_z", cfg["floor_z_default"])) if room else float(cfg["floor_z_default"])
    wall_height = float(room["wall_height"]) if room and "wall_height" in room else None
    boundary_source_kind = _boundary_source_kind(room)
    boundary_source_confidence = _source_confidence(boundary_source_kind)
    wall_source_kind = _wall_height_source_kind(room, wall_height)
    wall_source_confidence = _source_confidence(wall_source_kind)
    floor_source_kind = _floor_source_kind(room)
    floor_source_confidence = _source_confidence(floor_source_kind)
    objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
    flags: list[dict] = []

    for obj in objects:
        object_id = _object_id(obj)
        z_min, z_max = bbox_z_bounds(obj)
        floor_contact_tolerance = effective_floor_contact_tolerance(cfg, obj)
        boundary_tolerance = effective_boundary_tolerance(cfg, obj)
        if room and _has_boundary(room) and not footprint_inside_room(obj, room, boundary_tolerance):
            fallback = _is_fallback_source(boundary_source_kind)
            behavior = _fallback_behavior(policy.get("fallback_boundary_behavior"))
            confidence = _downgraded_confidence(boundary_source_confidence, fallback, behavior)
            suppressed = bool(fallback and behavior == "suppress")
            severity = "info" if suppressed else "low" if confidence == "low" else "medium"
            flags.append(
                {
                    "type": "room_boundary",
                    "code": "room_boundary_suppressed" if suppressed else "room_boundary_low_confidence" if confidence == "low" else "room_boundary",
                    "objects": [object_id],
                    "object_ids": [object_id],
                    "severity": severity,
                    "confidence": confidence,
                    "blocking": bool(policy.get("fallback_evidence_blocking", False)) if fallback else False,
                    "suppressed": suppressed,
                    "repair_relevant": not suppressed,
                    "source_kind": boundary_source_kind,
                    "source_confidence": confidence,
                    "boundary_tolerance_m": boundary_tolerance,
                    "message": _source_aware_message(
                        f"{object_id} footprint is outside the room boundary.",
                        fallback,
                        "Boundary is fallback-derived; treat this as a soft repair cue, not absolute room geometry.",
                    ),
                }
            )
        if z_min < floor_z - floor_contact_tolerance:
            flags.append(
                {
                    "type": "below_floor",
                    "code": "below_floor",
                    "objects": [object_id],
                    "object_ids": [object_id],
                    "severity": "high",
                    "confidence": "high",
                    "blocking": False,
                    "source_kind": floor_source_kind,
                    "source_confidence": floor_source_confidence,
                    "floor_z": floor_z,
                    "effective_floor_contact_tolerance_m": floor_contact_tolerance,
                    "message": f"{object_id} extends below floor_z.",
                }
            )
        wall_tolerance = max(floor_contact_tolerance, float(cfg.get("above_wall_tolerance_m", 0.0)))
        if wall_height is not None and z_max > wall_height + wall_tolerance:
            fallback = _is_fallback_source(wall_source_kind)
            behavior = _fallback_behavior(policy.get("fallback_wall_height_behavior"))
            confidence = _downgraded_confidence(wall_source_confidence, fallback, behavior)
            suppressed = bool(fallback and behavior == "suppress")
            severity = "info" if suppressed else "medium" if confidence == "low" else "high"
            flags.append(
                {
                    "type": "above_wall_height",
                    "code": "above_wall_height_suppressed" if suppressed else "above_wall_height_low_confidence" if confidence == "low" else "above_wall_height",
                    "objects": [object_id],
                    "object_ids": [object_id],
                    "severity": severity,
                    "confidence": confidence,
                    "blocking": bool(policy.get("fallback_evidence_blocking", False)) if fallback else False,
                    "suppressed": suppressed,
                    "repair_relevant": not suppressed,
                    "source_kind": wall_source_kind,
                    "source_confidence": confidence,
                    "wall_height": wall_height,
                    "effective_above_wall_tolerance_m": wall_tolerance,
                    "message": _source_aware_message(
                        f"{object_id} extends above wall_height.",
                        fallback,
                        "Wall height is fallback-derived; treat this as lower-confidence evidence.",
                    ),
                }
            )
            impossible = _impossible_height_constraint(obj, floor_z, wall_height, floor_contact_tolerance, wall_tolerance)
            if impossible:
                flags.append(
                    {
                        "type": "impossible_height_constraint",
                        "code": "fallback_metadata_conflict" if fallback else "impossible_height_constraint",
                        "objects": [object_id],
                        "object_ids": [object_id],
                        "severity": "medium" if fallback else "high",
                        "confidence": confidence,
                        "blocking": False,
                        "source_kind": wall_source_kind,
                        "source_confidence": confidence,
                        **impossible,
                    }
                )

    for index, obj_a in enumerate(objects):
        for obj_b in objects[index + 1 :]:
            ratio = serious_collision_ratio(obj_a, obj_b)
            volume = intersection_volume(obj_a, obj_b)
            smaller_volume = _smaller_bbox_volume(obj_a, obj_b)
            threshold = effective_scale_aware_min_volume(
                collision_volume_config if isinstance(collision_volume_config, dict) else {},
                smaller_object_volume_m3=smaller_volume,
                scene_volume_m3=scene_volume,
            )
            collision_volume_threshold = float(threshold["effective_min_collision_volume_m3"])
            if ratio > collision_ratio_threshold and volume > collision_volume_threshold:
                id_a, id_b = _object_id(obj_a), _object_id(obj_b)
                flags.append(
                    {
                        "type": "serious_collision",
                        "code": "serious_collision",
                        "objects": [id_a, id_b],
                        "object_ids": [id_a, id_b],
                        "severity": "high",
                        "confidence": "high",
                        "blocking": False,
                        "overlap_ratio": ratio,
                        "intersection_volume_m3": volume,
                        "smaller_object_volume_m3": threshold["smaller_object_volume_m3"],
                        "threshold": collision_ratio_threshold,
                        "effective_min_collision_volume_m3": collision_volume_threshold,
                        "threshold_source": threshold["threshold_source"],
                        "message": f"{id_a} overlaps above threshold {collision_ratio_threshold:.3f} with {id_b}.",
                    }
                )
    if bool(vertical_cfg.get("enabled", True)):
        flags.extend(_floating_or_vertical_flags(objects, floor_z, vertical_cfg))
    return flags


def serious_collision_ratio(obj_a: dict, obj_b: dict) -> float:
    volume = intersection_volume(obj_a, obj_b)
    if volume <= 0:
        return 0.0
    smaller_volume = min(_bbox_volume(obj_a), _bbox_volume(obj_b))
    if smaller_volume <= 0:
        return 0.0
    return float(volume) / float(smaller_volume)


def intersection_volume(obj_a: dict, obj_b: dict) -> float:
    intersection_area = footprint_intersection_area(obj_a, obj_b)
    z_overlap = vertical_overlap(obj_a, obj_b)
    return max(0.0, float(intersection_area) * float(z_overlap))


def _bbox_volume(obj: dict) -> float:
    try:
        height = float(obj["size"][2])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return float(footprint_polygon(obj).area) * height


def _smaller_bbox_volume(obj_a: dict, obj_b: dict) -> float | None:
    volumes = [volume for volume in [_bbox_volume(obj_a), _bbox_volume(obj_b)] if volume > 0]
    if not volumes:
        return None
    return min(volumes)


def _room_with_floor_polygon(room: dict) -> dict:
    result = dict(room)
    if "floor_polygon" not in result and "boundary" in result:
        result["floor_polygon"] = result["boundary"]
    return result


def _has_boundary(room: dict) -> bool:
    floor_plan = room.get("floor_plan") if isinstance(room.get("floor_plan"), dict) else {}
    regions = floor_plan.get("regions") or room.get("regions") or []
    if isinstance(regions, list) and any(isinstance(region, dict) and region.get("floor_polygon") for region in regions):
        return True
    return bool(room.get("floor_polygon") or room.get("boundary"))


def _object_id(obj: dict) -> str:
    return str(obj.get("object_id") or obj.get("id") or "unknown")


def _boundary_source_kind(room: dict) -> str:
    floor_plan = room.get("floor_plan") if isinstance(room.get("floor_plan"), dict) else {}
    if isinstance(floor_plan.get("regions"), list) and floor_plan.get("regions"):
        return _canonical_source_kind(floor_plan.get("source_kind") or room.get("boundary_source_kind") or "semantic_region")
    return _canonical_source_kind(
        floor_plan.get("source_kind")
        or room.get("boundary_source_kind")
        or room.get("room_boundary_source_kind")
        or room.get("boundary_source")
        or room.get("source_kind")
        or "unknown"
    )


def _wall_height_source_kind(room: dict, wall_height: float | None) -> str:
    explicit = (
        room.get("wall_height_source_kind")
        or room.get("height_source_kind")
        or room.get("wall_height_source")
        or room.get("height_source")
    )
    if explicit:
        return _canonical_source_kind(explicit)
    if wall_height is None:
        return "unknown"
    boundary_kind = _boundary_source_kind(room)
    if boundary_kind in {"object_position_extent_fallback", "fallback_default"} and abs(float(wall_height) - 3.0) < 1.0e-6:
        return "fallback_default"
    if boundary_kind in {"semantic_region", "stage_geometry"}:
        return boundary_kind
    return "room_metadata"


def _floor_source_kind(room: dict) -> str:
    explicit = room.get("floor_z_source_kind") or room.get("floor_source_kind") or room.get("floor_z_source")
    if explicit:
        return _canonical_source_kind(explicit)
    if "floor_z" not in room:
        return "fallback_default"
    boundary_kind = _boundary_source_kind(room)
    if boundary_kind in {"semantic_region", "stage_geometry"}:
        return boundary_kind
    return "room_metadata"


def _canonical_source_kind(value: object) -> str:
    text = str(value or "").lower()
    if "semantic" in text and ("region" in text or "polygon" in text):
        return "semantic_region"
    if "stage" in text or "floor_polygon" in text:
        return "stage_geometry"
    if "object_position_extent" in text or "position_extent" in text:
        return "object_position_extent_fallback"
    if "default" in text or "fallback" in text:
        return "fallback_default"
    if "metadata" in text or "room" in text or "case" in text:
        return "room_metadata"
    return str(value or "unknown")


def _source_confidence(source_kind: str) -> str:
    if source_kind == "semantic_region":
        return "high"
    if source_kind in {"stage_geometry", "room_metadata"}:
        return "medium"
    return "low"


def _is_fallback_source(source_kind: str) -> bool:
    return source_kind in {"object_position_extent_fallback", "fallback_default", "unknown"}


def _fallback_behavior(value: object) -> str:
    behavior = str(value or "downgrade").lower()
    return behavior if behavior in {"downgrade", "suppress", "normal"} else "downgrade"


def _downgraded_confidence(source_confidence: str, fallback: bool, behavior: object) -> str:
    if not fallback:
        return source_confidence
    if _fallback_behavior(behavior) == "normal":
        return source_confidence
    return "low"


def _source_aware_message(base: str, fallback: bool, note: str) -> str:
    return f"{base} {note}" if fallback else base


def _impossible_height_constraint(
    obj: dict,
    floor_z: float,
    wall_height: float,
    floor_margin: float,
    wall_margin: float,
) -> dict | None:
    size = obj.get("size")
    if not isinstance(size, list) or len(size) < 3:
        return None
    try:
        height = float(size[2])
    except (TypeError, ValueError):
        return None
    min_center_z = float(floor_z) + height / 2.0 + max(0.0, float(floor_margin))
    max_center_z = float(wall_height) - height / 2.0 - max(0.0, float(wall_margin))
    if max_center_z >= min_center_z:
        return None
    return {
        "object_id": _object_id(obj),
        "object_height": height,
        "floor_z": floor_z,
        "wall_height": wall_height,
        "min_center_z": min_center_z,
        "max_center_z": max_center_z,
        "message": (
            f"{_object_id(obj)} height exceeds the feasible floor-wall interval. "
            "Do not attempt naive lowering below floor."
        ),
    }


def _floating_or_vertical_flags(objects: list[dict], floor_z: float, cfg: dict) -> list[dict]:
    flags: list[dict] = []
    ignore_tokens = [str(item).lower() for item in cfg.get("ignore_categories", []) if isinstance(item, str)]
    min_gap_abs = float(cfg.get("min_gap_absolute", 0.25))
    min_gap_rel = float(cfg.get("min_gap_relative_to_height", 0.5))
    overlap_threshold = float(cfg.get("min_support_overlap_ratio", 0.15))
    blocking = bool(cfg.get("blocking", False))
    for obj in objects:
        object_id = _object_id(obj)
        category = str(obj.get("category") or "").lower()
        if any(token and token in category for token in ignore_tokens):
            continue
        if obj.get("support_parent"):
            continue
        center = obj.get("center")
        size = obj.get("size")
        if not isinstance(size, list) or len(size) < 3:
            continue
        try:
            height = float(size[2])
        except (TypeError, ValueError):
            continue
        bottom_z, _ = bbox_z_bounds(obj)
        gap_threshold = max(min_gap_abs, height * min_gap_rel)
        support_top = _nearest_support_top(obj, objects, overlap_threshold)
        reference_z = floor_z if support_top is None else support_top
        vertical_gap = bottom_z - reference_z
        if vertical_gap <= gap_threshold:
            continue
        flags.append(
            {
                "type": "floating_or_vertical_inconsistency",
                "code": "floating_or_vertical_inconsistency",
                "objects": [object_id],
                "object_ids": [object_id],
                "object_id": object_id,
                "bottom_z": bottom_z,
                "nearest_support_top_z": support_top,
                "floor_z": floor_z,
                "vertical_gap": vertical_gap,
                "gap_threshold": gap_threshold,
                "severity": "medium",
                "confidence": "medium",
                "blocking": blocking,
                "message": f"{object_id} appears vertically unsupported or floating.",
            }
        )
    return flags


def _nearest_support_top(obj: dict, objects: list[dict], overlap_threshold: float) -> float | None:
    bottom_z, _ = bbox_z_bounds(obj)
    area_obj = float(footprint_polygon(obj).area)
    if area_obj <= 0:
        return None
    candidates: list[float] = []
    for other in objects:
        if other is obj:
            continue
        _, other_top = bbox_z_bounds(other)
        if other_top > bottom_z:
            continue
        area_other = float(footprint_polygon(other).area)
        min_area = min(area_obj, area_other)
        if min_area <= 0:
            continue
        overlap_ratio = footprint_intersection_area(obj, other) / min_area
        if overlap_ratio >= overlap_threshold:
            candidates.append(float(other_top))
    return max(candidates) if candidates else None
