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
    collision_ratio_threshold = float(cfg["serious_collision_overlap_ratio"])
    collision_volume_config = cfg.get("serious_collision_min_volume")
    scene_volume = scene_volume_m3_from_case(case)

    room = _room_with_floor_polygon(case.get("room") or {})
    floor_z = float(room.get("floor_z", cfg["floor_z_default"])) if room else float(cfg["floor_z_default"])
    wall_height = float(room["wall_height"]) if room and "wall_height" in room else None
    objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
    flags: list[dict] = []

    for obj in objects:
        object_id = _object_id(obj)
        z_min, z_max = bbox_z_bounds(obj)
        floor_contact_tolerance = effective_floor_contact_tolerance(cfg, obj)
        boundary_tolerance = effective_boundary_tolerance(cfg, obj)
        if room and _has_boundary(room) and not footprint_inside_room(obj, room, boundary_tolerance):
            flags.append(
                {
                    "type": "room_boundary",
                    "objects": [object_id],
                    "severity": "medium",
                    "boundary_tolerance_m": boundary_tolerance,
                    "message": f"{object_id} footprint is outside the room boundary.",
                }
            )
        if z_min < floor_z - floor_contact_tolerance:
            flags.append(
                {
                    "type": "below_floor",
                    "objects": [object_id],
                    "severity": "high",
                    "floor_z": floor_z,
                    "effective_floor_contact_tolerance_m": floor_contact_tolerance,
                    "message": f"{object_id} extends below floor_z.",
                }
            )
        wall_tolerance = max(floor_contact_tolerance, float(cfg.get("above_wall_tolerance_m", 0.0)))
        if wall_height is not None and z_max > wall_height + wall_tolerance:
            flags.append(
                {
                    "type": "above_wall_height",
                    "objects": [object_id],
                    "severity": "high",
                    "wall_height": wall_height,
                    "effective_above_wall_tolerance_m": wall_tolerance,
                    "message": f"{object_id} extends above wall_height.",
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
                        "objects": [id_a, id_b],
                        "severity": "high",
                        "overlap_ratio": ratio,
                        "intersection_volume_m3": volume,
                        "smaller_object_volume_m3": threshold["smaller_object_volume_m3"],
                        "threshold": collision_ratio_threshold,
                        "effective_min_collision_volume_m3": collision_volume_threshold,
                        "threshold_source": threshold["threshold_source"],
                        "message": f"{id_a} overlaps above threshold {collision_ratio_threshold:.3f} with {id_b}.",
                    }
                )
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
