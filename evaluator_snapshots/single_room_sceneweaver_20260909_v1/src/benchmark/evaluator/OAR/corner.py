from __future__ import annotations

import math
from typing import Any

import numpy as np

from benchmark.evaluator.OAR.geometry import Corner, NormalizedRoom, footprint_point_distance
from benchmark.evaluator.OOR.geometry import NormalizedObject


def check_at_corner(subject: NormalizedObject, room: NormalizedRoom, corner_name: str | None = None, config: dict | None = None) -> dict:
    cfg = config or {}
    selection = _select_corner(subject, room, corner_name)
    if selection is None:
        return _invalid_result(subject, f"corner {corner_name!r} was not found" if corner_name else "room has no corners")
    corner, distance = selection
    center_distance = float(np.linalg.norm(subject.center[:2] - corner.point))
    object_xy_diagonal = math.hypot(float(subject.size[0]), float(subject.size[1]))
    threshold = max(float(cfg.get("corner_min", 0.20)), float(cfg.get("corner_ratio", 0.50)) * object_xy_diagonal)
    threshold = min(threshold, float(cfg.get("corner_max", 0.80)))
    passed = distance <= threshold
    return {
        "relation": "at_corner",
        "category": "corner",
        "subject_id": subject.id,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "evidence": {
            "corner": corner.name,
            "requested_corner": corner_name,
            "distance_to_corner": distance,
            "center_distance_to_corner": center_distance,
            "object_xy_diagonal": object_xy_diagonal,
            "threshold": threshold,
        },
        "status": "checked",
    }


def _select_corner(subject: NormalizedObject, room: NormalizedRoom, corner_name: str | None) -> tuple[Corner, float] | None:
    corners = room.corners if isinstance(room, NormalizedRoom) else []
    if not corners:
        return None
    target_name = _normalize_corner(corner_name)
    candidates = [corner for corner in corners if corner.name == target_name] if target_name else list(corners)
    if not candidates:
        return None
    scored = [(corner, footprint_point_distance(subject, corner.point)) for corner in candidates]
    return min(scored, key=lambda item: item[1])


def _invalid_result(subject: NormalizedObject, reason: str) -> dict[str, Any]:
    return {
        "relation": "at_corner",
        "category": "corner",
        "subject_id": getattr(subject, "id", ""),
        "passed": False,
        "score": 0.0,
        "evidence": {"reason": reason},
        "status": "invalid_input",
    }


def _normalize_corner(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if "north" in text and "east" in text:
        return "northeast"
    if "north" in text and "west" in text:
        return "northwest"
    if "south" in text and "east" in text:
        return "southeast"
    if "south" in text and "west" in text:
        return "southwest"
    for name in ["northeast", "northwest", "southeast", "southwest"]:
        if name in text:
            return name
    return text or None
