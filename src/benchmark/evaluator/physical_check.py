from __future__ import annotations

from benchmark.evaluator.geometry import (
    bbox_z_bounds,
    footprint_inside_room,
    footprint_intersection_area,
    horizontal_overlap_ratio,
    sorted_unique,
    vertical_overlap,
)


def check_physical_validity(layout: dict, bm_instance: dict, config: dict | None = None) -> tuple[bool, list[dict], list[str]]:
    cfg = (config or {}).get("geometry", config or {})
    boundary_tolerance = float(cfg.get("boundary_tolerance", 1.0e-6))
    collision_area_tolerance = float(cfg.get("collision_area_tolerance", 1.0e-6))
    vertical_overlap_tolerance = float(cfg.get("vertical_overlap_tolerance", 1.0e-6))
    floor_contact_tolerance = float(cfg.get("floor_contact_tolerance", 0.05))
    support_z_tolerance = float(cfg.get("support_z_tolerance", 0.05))
    min_support_overlap_ratio = float(cfg.get("min_support_overlap_ratio", 0.25))

    room = bm_instance.get("room")
    floor_z = float(room.get("floor_z", 0.0)) if isinstance(room, dict) else 0.0
    wall_height = float(room["wall_height"]) if isinstance(room, dict) and "wall_height" in room else None
    objects = layout.get("objects", [])
    object_by_id = {obj.get("object_id"): obj for obj in objects if isinstance(obj, dict)}

    failures: list[dict] = []
    repair_targets: list[str] = []

    for obj in objects:
        object_id = obj["object_id"]
        z_min, z_max = bbox_z_bounds(obj)
        if room and not footprint_inside_room(obj, room, boundary_tolerance):
            failures.append(
                {
                    "type": "boundary",
                    "objects": [object_id],
                    "message": f"{object_id} footprint is outside the room boundary.",
                }
            )
            repair_targets.append(object_id)
        if z_min < floor_z - floor_contact_tolerance:
            failures.append(
                {
                    "type": "below_floor",
                    "objects": [object_id],
                    "message": f"{object_id} extends below floor_z.",
                }
            )
            repair_targets.append(object_id)
        if wall_height is not None and z_max > wall_height + floor_contact_tolerance:
            failures.append(
                {
                    "type": "above_wall_height",
                    "objects": [object_id],
                    "message": f"{object_id} extends above wall_height.",
                }
            )
            repair_targets.append(object_id)

    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            if vertical_overlap(obj_a, obj_b) <= vertical_overlap_tolerance:
                continue
            if footprint_intersection_area(obj_a, obj_b) <= collision_area_tolerance:
                continue
            id_a, id_b = obj_a["object_id"], obj_b["object_id"]
            failures.append(
                {
                    "type": "collision",
                    "objects": [id_a, id_b],
                    "message": f"{id_a} collides with {id_b}.",
                }
            )
            repair_targets.extend([id_a, id_b])

    for obj in objects:
        object_id = obj["object_id"]
        z_min, _ = bbox_z_bounds(obj)
        support_parent = obj.get("support_parent")
        if support_parent == "floor":
            if abs(z_min - floor_z) > floor_contact_tolerance:
                failures.append(
                    {
                        "type": "floating",
                        "objects": [object_id],
                        "message": f"{object_id} is marked as floor-supported but z_min is not near floor_z.",
                    }
                )
                repair_targets.append(object_id)
        elif isinstance(support_parent, str) and support_parent:
            parent = object_by_id.get(support_parent)
            if parent is None:
                failures.append(
                    {
                        "type": "missing_support_parent",
                        "objects": [object_id],
                        "message": f"{object_id} references missing support_parent {support_parent}.",
                    }
                )
                repair_targets.append(object_id)
                continue
            _, parent_z_max = bbox_z_bounds(parent)
            overlap_ratio = horizontal_overlap_ratio(obj, parent)
            if abs(z_min - parent_z_max) > support_z_tolerance or overlap_ratio < min_support_overlap_ratio:
                failures.append(
                    {
                        "type": "invalid_support",
                        "objects": [object_id, support_parent],
                        "message": f"{object_id} is not sufficiently supported by {support_parent}.",
                    }
                )
                repair_targets.extend([object_id, support_parent])
        elif z_min > floor_z + floor_contact_tolerance:
            failures.append(
                {
                    "type": "floating",
                    "objects": [object_id],
                    "message": f"{object_id} has no support_parent and is above the floor.",
                }
            )
            repair_targets.append(object_id)

    return not failures, failures, sorted_unique(repair_targets)
