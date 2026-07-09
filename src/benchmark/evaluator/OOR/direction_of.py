from __future__ import annotations

import math

import numpy as np

from benchmark.evaluator.OOR.geometry import (
    NormalizedObject,
    angle_diff_degrees,
    center_xy_distance,
    footprint_overlap_score,
    sample_points_in_obb,
)


SIDE_RELATIONS = {"left", "right", "in_front", "behind"}
DIRECTION_RELATIONS = SIDE_RELATIONS | {"above", "below", "aligned_with"}


def check_direction(
    subject: NormalizedObject,
    anchor: NormalizedObject,
    relation_type: str,
    config: dict | None = None,
) -> dict:
    cfg = config or {}
    relation = str(relation_type)
    try:
        if relation in SIDE_RELATIONS:
            return _check_side_relation(subject, anchor, relation, cfg)
        if relation in {"above", "below"}:
            return _check_above_below(subject, anchor, relation, cfg)
        if relation == "aligned_with":
            return _check_aligned_with(subject, anchor, cfg)
        return _invalid_result(relation, subject, anchor, f"unsupported direction relation: {relation}")
    except (TypeError, ValueError, AttributeError) as exc:
        return _invalid_result(relation, subject, anchor, str(exc))


def _check_side_relation(subject: NormalizedObject, anchor: NormalizedObject, relation: str, cfg: dict) -> dict:
    points = sample_points_in_obb(subject, grid=(3, 3, 3))
    q = (points - anchor.center) @ anchor.R
    hx, hy, hz = [float(value) for value in anchor.half]
    anchor_diag = math.hypot(float(anchor.size[0]), float(anchor.size[1]))
    margin_xy = float(cfg.get("margin_xy_ratio", 0.25)) * max(float(anchor.size[0]), float(anchor.size[1]))
    max_side_distance = _clamp(
        float(cfg.get("side_alpha", 1.5)) * anchor_diag,
        float(cfg.get("side_min_distance", 0.5)),
        float(cfg.get("side_max_distance", 1.5)),
    )
    vertical_margin = float(cfg.get("vertical_margin_ratio", 0.5)) * max(float(subject.size[2]), float(anchor.size[2]))
    if relation == "left":
        mask = (q[:, 0] < -hx) & (np.abs(q[:, 1]) <= hy + margin_xy) & (np.abs(q[:, 2]) <= hz + vertical_margin) & (np.abs(q[:, 0] + hx) <= max_side_distance)
    elif relation == "right":
        mask = (q[:, 0] > hx) & (np.abs(q[:, 1]) <= hy + margin_xy) & (np.abs(q[:, 2]) <= hz + vertical_margin) & (np.abs(q[:, 0] - hx) <= max_side_distance)
    elif relation == "in_front":
        mask = (q[:, 1] < -hy) & (np.abs(q[:, 0]) <= hx + margin_xy) & (np.abs(q[:, 2]) <= hz + vertical_margin) & (np.abs(q[:, 1] + hy) <= max_side_distance)
    else:
        mask = (q[:, 1] > hy) & (np.abs(q[:, 0]) <= hx + margin_xy) & (np.abs(q[:, 2]) <= hz + vertical_margin) & (np.abs(q[:, 1] - hy) <= max_side_distance)
    score = float(np.count_nonzero(mask)) / float(len(points)) if len(points) else 0.0
    threshold = float(cfg.get("side_score_threshold", 0.5))
    return {
        "relation": relation,
        "category": "direction_of",
        "subject_id": subject.id,
        "object_id": anchor.id,
        "passed": score >= threshold,
        "score": float(score),
        "evidence": {
            "relation_type": relation,
            "score": float(score),
            "side_score_threshold": threshold,
            "num_sample_points": int(len(points)),
            "num_satisfying_points": int(np.count_nonzero(mask)),
            "margin_xy": float(margin_xy),
            "max_side_distance": float(max_side_distance),
            "vertical_margin": float(vertical_margin),
        },
        "status": "checked",
    }


def _check_above_below(subject: NormalizedObject, anchor: NormalizedObject, relation: str, cfg: dict) -> dict:
    eps_z = float(cfg.get("eps_z", 0.05))
    min_xy_overlap = float(cfg.get("min_xy_overlap", 0.2))
    anchor_diag = math.hypot(float(anchor.size[0]), float(anchor.size[1]))
    xy_threshold = 0.75 * anchor_diag
    xy_overlap = footprint_overlap_score(subject, anchor)
    center_distance = center_xy_distance(subject, anchor)
    xy_ok = xy_overlap >= min_xy_overlap or center_distance <= xy_threshold
    if relation == "above":
        z_ok = subject.center[2] > anchor.center[2] and subject.bottom >= anchor.top - eps_z
    else:
        z_ok = subject.center[2] < anchor.center[2] and subject.top <= anchor.bottom + eps_z
    score = 1.0 if z_ok and xy_ok else 0.0
    return {
        "relation": relation,
        "category": "direction_of",
        "subject_id": subject.id,
        "object_id": anchor.id,
        "passed": score >= 0.5,
        "score": score,
        "evidence": {
            "relation_type": relation,
            "xy_overlap": float(xy_overlap),
            "center_distance": float(center_distance),
            "above_below_xy_threshold": float(xy_threshold),
            "min_xy_overlap": min_xy_overlap,
            "eps_z": eps_z,
            "z_condition": bool(z_ok),
            "xy_condition": bool(xy_ok),
        },
        "status": "checked",
    }


def _check_aligned_with(subject: NormalizedObject, anchor: NormalizedObject, cfg: dict) -> dict:
    yaw_diff = angle_diff_degrees(float(subject.rotation[2]), float(anchor.rotation[2]))
    threshold = float(cfg.get("yaw_threshold_degrees", 20))
    score = 1.0 if yaw_diff <= threshold else 0.0
    return {
        "relation": "aligned_with",
        "category": "direction_of",
        "subject_id": subject.id,
        "object_id": anchor.id,
        "passed": score >= 0.5,
        "score": score,
        "evidence": {
            "relation_type": "aligned_with",
            "yaw_diff": float(yaw_diff),
            "yaw_threshold_degrees": threshold,
        },
        "status": "checked",
    }


def _invalid_result(relation: str, subject: object, anchor: object, reason: str) -> dict:
    return {
        "relation": relation,
        "category": "direction_of",
        "subject_id": getattr(subject, "id", ""),
        "object_id": getattr(anchor, "id", ""),
        "passed": False,
        "score": 0.0,
        "evidence": {"relation_type": relation, "reason": reason},
        "status": "invalid_input",
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
