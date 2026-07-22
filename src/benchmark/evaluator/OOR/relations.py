from __future__ import annotations

import math
from typing import Any

import numpy as np

from benchmark.evaluator.OOR.geometry import NormalizedObject, footprint_edge_distance, point_segment_distance_2d


def check_far(subject: NormalizedObject, anchor: NormalizedObject, config: dict | None = None) -> dict[str, Any]:
    cfg = config or {}
    distance = footprint_edge_distance(subject, anchor)
    anchor_diag = math.hypot(float(anchor.size[0]), float(anchor.size[1]))
    threshold = _clamp(
        float(cfg.get("alpha", 1.5)) * anchor_diag,
        float(cfg.get("min_threshold", 0.30)),
        float(cfg.get("max_threshold", 1.50)),
    )
    return _result(
        "far",
        "proximity",
        subject,
        anchor,
        distance > threshold,
        {"footprint_edge_distance": distance, "threshold": threshold},
    )


def check_orientation(
    subject: NormalizedObject,
    anchor: NormalizedObject,
    relation: str,
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    threshold = float(cfg.get("angle_threshold_degrees", 20.0))
    yaw_diff = _axis_angle_diff(float(subject.rotation[2]), float(anchor.rotation[2]))
    if relation == "parallel":
        error = yaw_diff
    elif relation == "perpendicular":
        error = abs(90.0 - yaw_diff)
    else:
        return _invalid(relation, "orientation", [subject.id, anchor.id], "unsupported orientation relation")
    return _result(
        relation,
        "orientation",
        subject,
        anchor,
        error <= threshold,
        {
            "yaw_axis_difference_degrees": yaw_diff,
            "target_angle_degrees": 0.0 if relation == "parallel" else 90.0,
            "angle_error_degrees": error,
            "angle_threshold_degrees": threshold,
        },
    )


def check_alignment(
    subject: NormalizedObject,
    anchor: NormalizedObject,
    spec: dict,
    config: dict | None = None,
) -> dict[str, Any]:
    """Check whether two object centers share a row, column, or vertical line.

    Alignment is positional. Relative orientation belongs to the separate
    parallel/perpendicular predicates.
    """

    cfg = config or {}
    axis = str(spec.get("axis") or spec.get("alignment_axis") or "").strip().lower()
    axis_aliases = {
        "horizontal": "x",
        "left_right": "x",
        "row": "x",
        "vertical": "y",
        "front_back": "y",
        "column": "y",
        "height": "z",
    }
    axis = axis_aliases.get(axis, axis)
    tolerance_ratio = float(cfg.get("center_tolerance_ratio", 0.25))
    minimum_tolerance = float(cfg.get("minimum_tolerance", 0.05))

    candidates: dict[str, dict[str, float]] = {}
    # Aligned along X means that the centers share a Y row, and vice versa.
    candidates["x"] = {
        "error": abs(float(subject.center[1] - anchor.center[1])),
        "tolerance": max(
            minimum_tolerance,
            tolerance_ratio * min(float(subject.size[1]), float(anchor.size[1])),
        ),
    }
    candidates["y"] = {
        "error": abs(float(subject.center[0] - anchor.center[0])),
        "tolerance": max(
            minimum_tolerance,
            tolerance_ratio * min(float(subject.size[0]), float(anchor.size[0])),
        ),
    }
    xy_error = float(np.linalg.norm(subject.center[:2] - anchor.center[:2]))
    candidates["z"] = {
        "error": xy_error,
        "tolerance": max(
            minimum_tolerance,
            tolerance_ratio * min(float(np.linalg.norm(subject.size[:2])), float(np.linalg.norm(anchor.size[:2]))),
        ),
    }
    if axis in candidates:
        selected_axis = axis
    elif not axis:
        selected_axis = min(candidates, key=lambda name: candidates[name]["error"] / candidates[name]["tolerance"])
    else:
        return _invalid("aligned", "alignment", [subject.id, anchor.id], f"unsupported alignment axis {axis!r}")
    selected = candidates[selected_axis]
    return _result(
        "aligned",
        "alignment",
        subject,
        anchor,
        selected["error"] <= selected["tolerance"],
        {
            "alignment_axis": selected_axis,
            "axis_was_explicit": bool(axis),
            "center_line_error": selected["error"],
            "center_line_tolerance": selected["tolerance"],
            "candidate_axes": candidates,
        },
    )


def check_between(
    subject: NormalizedObject,
    anchors: list[NormalizedObject],
    config: dict | None = None,
) -> dict[str, Any]:
    if len(anchors) != 2:
        return _invalid("between", "multi_object", [subject.id, *(item.id for item in anchors)], "between requires two anchors")
    cfg = config or {}
    a = anchors[0].center[:2]
    b = anchors[1].center[:2]
    p = subject.center[:2]
    segment = b - a
    length = float(np.linalg.norm(segment))
    if length <= 1.0e-9:
        return _invalid("between", "multi_object", [subject.id, anchors[0].id, anchors[1].id], "anchor centers coincide")
    t = float(np.dot(p - a, segment) / np.dot(segment, segment))
    line_distance = point_segment_distance_2d(p, a, b)
    scale = max(float(np.linalg.norm(subject.size[:2])), 0.25)
    threshold = min(
        float(cfg.get("max_line_distance", 1.0)),
        max(float(cfg.get("min_line_distance", 0.20)), float(cfg.get("line_distance_ratio", 0.50)) * scale),
    )
    projection_tolerance = float(cfg.get("projection_tolerance", 0.05))
    passed = -projection_tolerance <= t <= 1.0 + projection_tolerance and line_distance <= threshold
    return _group_result(
        "between",
        [subject.id, anchors[0].id, anchors[1].id],
        passed,
        {
            "segment_projection": t,
            "line_distance": line_distance,
            "line_distance_threshold": threshold,
            "projection_tolerance": projection_tolerance,
        },
    )


def check_ordered(objects: list[NormalizedObject], spec: dict, config: dict | None = None) -> dict[str, Any]:
    if len(objects) < 2:
        return _invalid("ordered", "multi_object", [item.id for item in objects], "ordered requires at least two objects")
    cfg = config or {}
    axis_and_sign = _ordered_axis(spec)
    if axis_and_sign is None:
        return _invalid(
            "ordered",
            "multi_object",
            [item.id for item in objects],
            "ordered requires an explicit axis or direction",
        )
    axis, sign = axis_and_sign
    margin = float(cfg.get("minimum_center_margin", 0.02))
    values = [float(item.center[axis]) * sign for item in objects]
    deltas = [right - left for left, right in zip(values, values[1:])]
    passed = all(delta > margin for delta in deltas)
    return _group_result(
        "ordered",
        [item.id for item in objects],
        passed,
        {
            "axis": ["x", "y", "z"][axis],
            "direction_sign": sign,
            "projected_centers": values,
            "pairwise_deltas": deltas,
            "minimum_center_margin": margin,
        },
    )


def check_around(
    members: list[NormalizedObject],
    anchor: NormalizedObject,
    config: dict | None = None,
) -> dict[str, Any]:
    if len(members) < 2:
        return _invalid("around", "multi_object", [item.id for item in members], "around requires at least two members")
    cfg = config or {}
    vectors = np.array([item.center[:2] - anchor.center[:2] for item in members], dtype=float)
    radii = np.linalg.norm(vectors, axis=1)
    if np.any(radii <= 1.0e-9):
        return _group_result(
            "around",
            [item.id for item in members],
            False,
            {"reason": "one or more member centers coincide with the anchor center"},
            anchor_id=anchor.id,
        )
    angles = np.sort(np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * math.pi))
    wrapped = np.concatenate([angles, [angles[0] + 2.0 * math.pi]])
    max_gap_degrees = math.degrees(float(np.max(np.diff(wrapped))))
    radial_cv = float(np.std(radii) / np.mean(radii)) if float(np.mean(radii)) > 0 else math.inf
    max_allowed_gap = float(cfg.get("max_angular_gap_degrees", 200.0 if len(members) == 2 else 180.0))
    max_radial_cv = float(cfg.get("max_radial_cv", 0.65))
    passed = max_gap_degrees <= max_allowed_gap and radial_cv <= max_radial_cv
    return _group_result(
        "around",
        [item.id for item in members],
        passed,
        {
            "anchor_id": anchor.id,
            "member_count": len(members),
            "radii": [float(value) for value in radii],
            "radial_cv": radial_cv,
            "max_radial_cv": max_radial_cv,
            "max_angular_gap_degrees": max_gap_degrees,
            "max_allowed_angular_gap_degrees": max_allowed_gap,
        },
        anchor_id=anchor.id,
    )


def _result(
    relation: str,
    category: str,
    subject: NormalizedObject,
    anchor: NormalizedObject,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "relation": relation,
        "category": category,
        "subject_id": subject.id,
        "object_id": anchor.id,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "evidence": evidence,
        "status": "checked",
    }


def _group_result(
    relation: str,
    object_ids: list[str],
    passed: bool,
    evidence: dict[str, Any],
    *,
    anchor_id: str | None = None,
) -> dict[str, Any]:
    result = {
        "relation": relation,
        "category": "multi_object",
        "object_ids": object_ids,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "evidence": evidence,
        "status": "checked",
    }
    if anchor_id is not None:
        result["anchor_id"] = anchor_id
    return result


def _invalid(relation: str, category: str, object_ids: list[str], reason: str) -> dict[str, Any]:
    return {
        "relation": relation,
        "category": category,
        "object_ids": object_ids,
        "passed": False,
        "score": 0.0,
        "evidence": {"reason": reason},
        "status": "invalid_input",
    }


def _axis_angle_diff(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def _ordered_axis(spec: dict) -> tuple[int, float] | None:
    direction = str(spec.get("direction") or "").strip().lower().replace("-", "_").replace(" ", "_")
    direction_map = {
        "left_to_right": (0, 1.0),
        "right_to_left": (0, -1.0),
        "front_to_back": (1, 1.0),
        "back_to_front": (1, -1.0),
        "bottom_to_top": (2, 1.0),
        "top_to_bottom": (2, -1.0),
    }
    if direction in direction_map:
        return direction_map[direction]
    axis_name = str(spec.get("axis") or "").strip().lower()
    if axis_name not in {"x", "y", "z"}:
        return None
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]
    sign = -1.0 if str(spec.get("order") or "ascending").strip().lower() in {"descending", "reverse"} else 1.0
    return axis, sign


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
