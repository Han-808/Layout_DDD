from __future__ import annotations

import math
from typing import Any

from benchmark.evaluator.OAR.geometry import (
    NormalizedRoom,
    WallSegment,
    min_distance_center_to_wall,
    min_distance_footprint_to_wall,
)
from benchmark.evaluator.OOR.geometry import NormalizedObject


def check_against_wall(subject: NormalizedObject, room: NormalizedRoom, wall_name: str | None = None, config: dict | None = None) -> dict:
    cfg = config or {}
    selection = _select_wall(subject, room, wall_name)
    if selection is None:
        return _invalid_result("against_wall", subject, f"wall {wall_name!r} was not found" if wall_name else "room has no wall segments")
    wall, distance = selection
    min_size_xy = float(min(subject.size[0], subject.size[1]))
    threshold = min(
        max(float(cfg.get("eps_wall", 0.08)), float(cfg.get("wall_ratio", 0.15)) * min_size_xy),
        float(cfg.get("max_against_distance", 0.20)),
    )
    passed = distance <= threshold
    return _checked_result(
        "against_wall",
        subject,
        wall,
        passed,
        {
            "wall": wall.name,
            "requested_wall": wall_name,
            "distance_to_wall": distance,
            "center_distance_to_wall": min_distance_center_to_wall(subject, wall),
            "threshold": threshold,
        },
    )


def check_near_wall(subject: NormalizedObject, room: NormalizedRoom, wall_name: str | None = None, config: dict | None = None) -> dict:
    cfg = config or {}
    selection = _select_wall(subject, room, wall_name)
    if selection is None:
        return _invalid_result("near_wall", subject, f"wall {wall_name!r} was not found" if wall_name else "room has no wall segments")
    wall, distance = selection
    room_diag = math.hypot(room.max_x - room.min_x, room.max_y - room.min_y)
    threshold = max(float(cfg.get("near_wall_min", 0.30)), float(cfg.get("near_wall_ratio", 0.10)) * room_diag)
    threshold = min(threshold, float(cfg.get("near_wall_max", 0.80)))
    passed = distance <= threshold
    return _checked_result(
        "near_wall",
        subject,
        wall,
        passed,
        {
            "wall": wall.name,
            "requested_wall": wall_name,
            "distance_to_wall": distance,
            "center_distance_to_wall": min_distance_center_to_wall(subject, wall),
            "room_diagonal": room_diag,
            "threshold": threshold,
        },
    )


def check_below_wall(subject: NormalizedObject, room: NormalizedRoom, wall_name: str | None = None, config: dict | None = None) -> dict:
    cfg = config or {}
    near_result = check_near_wall(subject, room, wall_name, cfg)
    if near_result.get("status") != "checked":
        return {
            "relation": "below_wall",
            "category": "wall",
            "subject_id": subject.id,
            "passed": False,
            "score": 0.0,
            "evidence": {"near_wall_result": near_result.get("evidence", {})},
            "status": "invalid_input",
        }
    eps_z = float(cfg.get("eps_z", 0.05))
    top = float(subject.top)
    scene_height = room.scene_height
    height_ok = True if scene_height is None else top <= float(scene_height) + eps_z
    passed = bool(near_result.get("passed")) and height_ok
    evidence = {
        "wall": (near_result.get("evidence") or {}).get("wall"),
        "requested_wall": wall_name,
        "near_wall_passed": bool(near_result.get("passed")),
        "distance_to_wall": (near_result.get("evidence") or {}).get("distance_to_wall"),
        "near_wall_threshold": (near_result.get("evidence") or {}).get("threshold"),
        "top_z": top,
        "scene_height": scene_height,
        "eps_z": eps_z,
        "height_ok": bool(height_ok),
    }
    return {
        "relation": "below_wall",
        "category": "wall",
        "subject_id": subject.id,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "evidence": evidence,
        "status": "checked",
    }


def _select_wall(subject: NormalizedObject, room: NormalizedRoom, wall_name: str | None) -> tuple[WallSegment, float] | None:
    walls = room.wall_segments if isinstance(room, NormalizedRoom) else []
    if not walls:
        return None
    target_name = _normalize_name(wall_name)
    candidates = [wall for wall in walls if wall.name == target_name] if target_name else list(walls)
    if not candidates:
        return None
    scored = [(wall, min_distance_footprint_to_wall(subject, wall)) for wall in candidates]
    return min(scored, key=lambda item: item[1])


def _checked_result(relation: str, subject: NormalizedObject, wall: WallSegment, passed: bool, evidence: dict[str, Any]) -> dict:
    return {
        "relation": relation,
        "category": "wall",
        "subject_id": subject.id,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "evidence": evidence,
        "status": "checked",
    }


def _invalid_result(relation: str, subject: NormalizedObject, reason: str) -> dict:
    return {
        "relation": relation,
        "category": "wall",
        "subject_id": getattr(subject, "id", ""),
        "passed": False,
        "score": 0.0,
        "evidence": {"reason": reason},
        "status": "invalid_input",
    }


def _normalize_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ")
    for name in ["east", "west", "north", "south"]:
        if name in text.split() or name in text:
            return name
    return text or None
