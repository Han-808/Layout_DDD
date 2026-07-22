from __future__ import annotations

import math
from typing import Any

import numpy as np

from benchmark.evaluator.OAR.geometry import (
    Corner,
    NormalizedRoom,
    WallSegment,
    footprint_point_distance,
    min_distance_footprint_to_wall,
)
from benchmark.evaluator.OOR.geometry import NormalizedObject


OAR_ATTACHMENT_DETECTOR_VERSION = "oar_attachment_proxy_v2"


def check_near_corner(
    subject: NormalizedObject,
    room: NormalizedRoom,
    corner_name: str | None,
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    selection = _select_corner(subject, room, corner_name)
    if selection is None:
        return _invalid("near_corner", "corner", subject.id, "requested corner was not found")
    corner, distance = selection
    object_diag = math.hypot(float(subject.size[0]), float(subject.size[1]))
    threshold = _clamp(
        float(cfg.get("near_corner_ratio", 1.0)) * object_diag,
        float(cfg.get("near_corner_min", 0.40)),
        float(cfg.get("near_corner_max", 1.50)),
    )
    return _result(
        "near_corner",
        "corner",
        subject,
        distance <= threshold,
        {
            "corner": corner.name,
            "requested_corner": corner_name,
            "distance_to_corner": distance,
            "threshold": threshold,
        },
    )


def check_room_center(
    subject: NormalizedObject,
    room: NormalizedRoom,
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    room_diag = math.hypot(room.max_x - room.min_x, room.max_y - room.min_y)
    distance = float(np.linalg.norm(subject.center[:2] - room.centroid))
    threshold = max(
        float(cfg.get("minimum_radius", 0.25)),
        float(cfg.get("radius_ratio", 0.15)) * room_diag,
    )
    threshold = min(threshold, float(cfg.get("maximum_radius", 1.50)))
    return _result(
        "room_center",
        "room_region",
        subject,
        distance <= threshold,
        {
            "room_centroid": [float(value) for value in room.centroid],
            "center_distance": distance,
            "room_diagonal": room_diag,
            "threshold": threshold,
        },
    )


def check_room_region(
    subject: NormalizedObject,
    room: NormalizedRoom,
    region: str | None,
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    canonical = _normalize_region(region)
    if canonical == "center":
        result = check_room_center(subject, room, cfg.get("center"))
        result["relation"] = "room_region"
        result["evidence"] = {**result.get("evidence", {}), "requested_region": region, "region": canonical}
        return result
    if canonical is None:
        return _invalid("room_region", "room_region", subject.id, "room_region requires a known region")

    width = room.max_x - room.min_x
    depth = room.max_y - room.min_y
    if width <= 0.0 or depth <= 0.0:
        return _invalid("room_region", "room_region", subject.id, "room dimensions must be positive")
    u = (float(subject.center[0]) - room.min_x) / width
    v = (float(subject.center[1]) - room.min_y) / depth
    lower = float(cfg.get("third_lower", 1.0 / 3.0))
    upper = float(cfg.get("third_upper", 2.0 / 3.0))
    tests = {
        "west": u <= lower,
        "east": u >= upper,
        "south": v <= lower,
        "north": v >= upper,
        "southwest": u <= lower and v <= lower,
        "southeast": u >= upper and v <= lower,
        "northwest": u <= lower and v >= upper,
        "northeast": u >= upper and v >= upper,
    }
    if canonical not in tests:
        return _invalid("room_region", "room_region", subject.id, f"unsupported room region {canonical!r}")
    return _result(
        "room_region",
        "room_region",
        subject,
        bool(tests[canonical]),
        {
            "requested_region": region,
            "region": canonical,
            "normalized_center": [u, v],
            "third_thresholds": [lower, upper],
            "coordinate_frame": "west=-x,east=+x,south/front=-y,north/back=+y",
        },
    )


def check_along_wall(
    subject: NormalizedObject,
    room: NormalizedRoom,
    wall_name: str | None,
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    selection = _select_wall(subject, room, wall_name)
    if selection is None:
        return _invalid("along_wall", "wall", subject.id, "requested wall was not found")
    wall, distance = selection
    room_diag = math.hypot(room.max_x - room.min_x, room.max_y - room.min_y)
    distance_threshold = _clamp(
        float(cfg.get("distance_ratio", 0.10)) * room_diag,
        float(cfg.get("distance_min", 0.30)),
        float(cfg.get("distance_max", 0.80)),
    )
    wall_angle = math.degrees(math.atan2(float(wall.p1[1] - wall.p0[1]), float(wall.p1[0] - wall.p0[0])))
    object_angle = float(subject.rotation[2]) + (90.0 if float(subject.size[1]) > float(subject.size[0]) else 0.0)
    angle_error = _axis_angle_diff(object_angle, wall_angle)
    angle_threshold = float(cfg.get("angle_threshold_degrees", 20.0))
    passed = distance <= distance_threshold and angle_error <= angle_threshold
    return _result(
        "along_wall",
        "wall",
        subject,
        passed,
        {
            "wall": wall.name,
            "requested_wall": wall_name,
            "distance_to_wall": distance,
            "distance_threshold": distance_threshold,
            "object_long_axis_degrees": object_angle % 180.0,
            "wall_axis_degrees": wall_angle % 180.0,
            "axis_angle_error_degrees": angle_error,
            "angle_threshold_degrees": angle_threshold,
        },
    )


def check_mounted_on_wall(
    subject: NormalizedObject,
    room: NormalizedRoom,
    wall_name: str | None,
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    selection = _select_wall(subject, room, wall_name)
    if selection is None:
        return _invalid("mounted_on_wall", "wall_attachment", subject.id, "requested wall was not found")
    wall, distance = selection
    contact_threshold = float(cfg.get("contact_threshold", 0.12))
    floor_clearance = float(cfg.get("minimum_floor_clearance", 0.10))
    ceiling_tolerance = float(cfg.get("ceiling_tolerance", 0.05))
    within_height = room.scene_height is None or float(subject.top) <= room.scene_height + ceiling_tolerance
    proxy_checks = {
        "within_wall_contact_threshold": distance <= contact_threshold,
        "above_floor_clearance": float(subject.bottom) > floor_clearance,
        "within_room_height": within_height,
    }
    return _vlm_candidate(
        "mounted_on_wall",
        "wall_attachment",
        subject,
        {
            "wall": wall.name,
            "requested_wall": wall_name,
            "distance_to_wall": distance,
            "contact_threshold": contact_threshold,
            "bottom_z": float(subject.bottom),
            "minimum_floor_clearance": floor_clearance,
            "top_z": float(subject.top),
            "scene_height": room.scene_height,
            "within_room_height": within_height,
            "proxy": "obb_to_wall_attachment",
            "proxy_checks": proxy_checks,
            "proxy_checks_passed": all(proxy_checks.values()),
        },
    )


def check_ceiling_attachment(
    subject: NormalizedObject,
    room: NormalizedRoom,
    relation: str,
    config: dict | None = None,
) -> dict[str, Any]:
    if relation not in {"attached_to_ceiling", "hung_from_ceiling"}:
        return _invalid(relation, "ceiling_attachment", subject.id, "unsupported ceiling relation")
    if room.scene_height is None:
        return _vlm_candidate(
            relation,
            "ceiling_attachment",
            subject,
            {
                "top_z": float(subject.top),
                "scene_height": None,
                "proxy": "obb_to_ceiling_attachment",
                "proxy_checks": {"scene_height_available": False},
                "proxy_checks_passed": None,
                "reason": "scene_height_unavailable",
            },
        )
    cfg = config or {}
    signed_clearance = float(room.scene_height) - float(subject.top)
    gap = abs(signed_clearance)
    threshold = float(cfg.get("top_gap_threshold", 0.10))
    floor_tolerance = float(cfg.get("floor_tolerance", 0.05))
    proxy_checks = {
        "within_ceiling_gap_threshold": gap <= threshold,
        "not_below_floor": float(subject.bottom) >= -floor_tolerance,
    }
    return _vlm_candidate(
        relation,
        "ceiling_attachment",
        subject,
        {
            "top_z": float(subject.top),
            "scene_height": float(room.scene_height),
            "top_gap": gap,
            "signed_top_clearance": signed_clearance,
            "top_gap_threshold": threshold,
            "bottom_z": float(subject.bottom),
            "floor_tolerance": floor_tolerance,
            "proxy": "obb_to_ceiling_attachment",
            "proxy_checks": proxy_checks,
            "proxy_checks_passed": all(proxy_checks.values()),
        },
    )


def _select_wall(
    subject: NormalizedObject,
    room: NormalizedRoom,
    wall_name: str | None,
) -> tuple[WallSegment, float] | None:
    target = _normalize_wall(wall_name)
    candidates = [wall for wall in room.wall_segments if wall.name == target] if target else list(room.wall_segments)
    if not candidates:
        return None
    return min(
        ((wall, min_distance_footprint_to_wall(subject, wall)) for wall in candidates),
        key=lambda item: item[1],
    )


def _select_corner(
    subject: NormalizedObject,
    room: NormalizedRoom,
    corner_name: str | None,
) -> tuple[Corner, float] | None:
    target = _normalize_corner(corner_name)
    candidates = [corner for corner in room.corners if corner.name == target] if target else list(room.corners)
    if not candidates:
        return None
    return min(
        ((corner, footprint_point_distance(subject, corner.point)) for corner in candidates),
        key=lambda item: item[1],
    )


def _result(
    relation: str,
    category: str,
    subject: NormalizedObject,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "relation": relation,
        "category": category,
        "subject_id": subject.id,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "status": "checked",
        "backend": "deterministic",
        "evidence": evidence,
    }


def _vlm_candidate(
    relation: str,
    category: str,
    subject: NormalizedObject,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Return OBB proxy evidence without treating it as attachment truth."""

    return {
        "relation": relation,
        "category": category,
        "subject_id": subject.id,
        "passed": None,
        "score": None,
        "status": "requires_vlm",
        "route": "requires_vlm",
        "backend": "deterministic",
        "evidence": {
            "detector": OAR_ATTACHMENT_DETECTOR_VERSION,
            "routing_has_invalid_prior": False,
            **evidence,
        },
    }


def _invalid(relation: str, category: str, subject_id: str, reason: str) -> dict[str, Any]:
    return {
        "relation": relation,
        "category": category,
        "subject_id": subject_id,
        "passed": False,
        "score": 0.0,
        "status": "invalid_input",
        "backend": "deterministic",
        "evidence": {"reason": reason},
    }


def _normalize_wall(value: str | None) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "right": "east",
        "right_wall": "east",
        "east_wall": "east",
        "left": "west",
        "left_wall": "west",
        "west_wall": "west",
        "back": "north",
        "back_wall": "north",
        "rear_wall": "north",
        "north_wall": "north",
        "front": "south",
        "front_wall": "south",
        "south_wall": "south",
    }
    if text in aliases:
        return aliases[text]
    return text if text in {"east", "west", "north", "south"} else None


def _normalize_corner(value: str | None) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = text.removesuffix("_corner")
    aliases = {
        "front_left": "southwest",
        "front_right": "southeast",
        "back_left": "northwest",
        "back_right": "northeast",
    }
    text = aliases.get(text, text)
    return text if text in {"northeast", "northwest", "southeast", "southwest"} else None


def _normalize_region(value: str | None) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    for suffix in ("_region", "_area", "_side"):
        text = text.removesuffix(suffix)
    aliases = {
        "middle": "center",
        "centre": "center",
        "left": "west",
        "right": "east",
        "front": "south",
        "back": "north",
        "rear": "north",
        "front_left": "southwest",
        "front_right": "southeast",
        "back_left": "northwest",
        "back_right": "northeast",
    }
    text = aliases.get(text, text)
    allowed = {"center", "west", "east", "south", "north", "southwest", "southeast", "northwest", "northeast"}
    return text if text in allowed else None


def _axis_angle_diff(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
