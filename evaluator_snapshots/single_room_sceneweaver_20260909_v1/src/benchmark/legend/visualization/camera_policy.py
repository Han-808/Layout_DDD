from __future__ import annotations


def room_camera_policy() -> list[dict]:
    return [
        {"id": "topdown_room", "projection": "xy", "description": "Top-down room-level bbox view."},
        {"id": "front_room", "projection": "xz", "description": "Front elevation room-level bbox view."},
        {"id": "corner_room", "projection": "oblique", "description": "Simple oblique room-level bbox view."},
    ]


def global_camera_policy() -> list[dict]:
    return [
        {"id": "topdown_global_xy", "projection": "xy", "description": "Global top-down room-level bbox view."},
    ]


def group_camera_policy(group_id: str) -> list[dict]:
    return [
        {"id": f"{group_id}_xy", "group_id": group_id, "projection": "xy", "description": "Group top-down bbox view."},
        {"id": f"{group_id}_yz", "group_id": group_id, "projection": "yz", "description": "Group side elevation bbox view."},
        {"id": f"{group_id}_xz", "group_id": group_id, "projection": "xz", "description": "Group front elevation bbox view."},
    ]


def pair_camera_policy(spec_id: str) -> list[dict]:
    return [
        {"id": "pair_top", "spec_id": spec_id, "projection": "xy", "description": "Top-down pair-local view."},
        {"id": "pair_side", "spec_id": spec_id, "projection": "xz", "description": "Side/elevation pair-local view."},
        {"id": "pair_oblique", "spec_id": spec_id, "projection": "oblique", "description": "Oblique pair-local view."},
    ]
