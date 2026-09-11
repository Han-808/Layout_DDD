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
    grid = _pairwise_grid(cfg.get("pairwise_grid", (5, 5, 3)))
    epsilon = float(cfg.get("pairwise_epsilon", 0.02))
    valid_threshold = float(cfg.get("pairwise_valid_threshold", 0.60))
    invalid_threshold = float(cfg.get("pairwise_invalid_threshold", 0.40))
    if epsilon < 0.0:
        raise ValueError("pairwise_epsilon must be non-negative")
    if not 0.0 <= invalid_threshold < valid_threshold <= 1.0:
        raise ValueError("pairwise thresholds must satisfy 0 <= invalid < valid <= 1")

    axis, sign, axis_name, positive_direction = {
        "left": (0, -1.0, "x", "-x"),
        "right": (0, 1.0, "x", "+x"),
        "in_front": (1, -1.0, "y", "-y"),
        "behind": (1, 1.0, "y", "+y"),
    }[relation]
    subject_points = sample_points_in_obb(subject, grid=grid)
    anchor_points = sample_points_in_obb(anchor, grid=grid)
    subject_projection = sign * subject_points[:, axis]
    anchor_projection = sign * anchor_points[:, axis]
    pairwise_delta = subject_projection[:, None] - anchor_projection[None, :]
    ordered = pairwise_delta > epsilon
    tied = np.abs(pairwise_delta) <= epsilon
    num_pairs = int(pairwise_delta.size)
    num_ordered = int(np.count_nonzero(ordered))
    num_tied = int(np.count_nonzero(tied))
    score = (float(num_ordered) + 0.5 * float(num_tied)) / float(num_pairs) if num_pairs else 0.5
    if score >= valid_threshold:
        ordering_state = "valid"
    elif score <= invalid_threshold:
        ordering_state = "invalid"
    else:
        ordering_state = "boundary"
    return {
        "relation": relation,
        "category": "direction_of",
        "subject_id": subject.id,
        "object_id": anchor.id,
        "passed": ordering_state == "valid",
        "score": float(score),
        "evidence": {
            "relation_type": relation,
            "reference_frame": "room",
            "axis": axis_name,
            "positive_direction": positive_direction,
            "ordering_state": ordering_state,
            "ordering_score": float(score),
            "signed_center_delta": float(sign * (subject.center[axis] - anchor.center[axis])),
            "pairwise_epsilon": epsilon,
            "pairwise_valid_threshold": valid_threshold,
            "pairwise_invalid_threshold": invalid_threshold,
            "pairwise_grid": list(grid),
            "subject_projection_range": [float(np.min(subject_projection)), float(np.max(subject_projection))],
            "anchor_projection_range": [float(np.min(anchor_projection)), float(np.max(anchor_projection))],
            "num_subject_sample_points": int(len(subject_points)),
            "num_anchor_sample_points": int(len(anchor_points)),
            "num_pairwise_comparisons": num_pairs,
            "num_ordered_pairs": num_ordered,
            "num_tied_pairs": num_tied,
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


def _pairwise_grid(value: object) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("pairwise_grid must contain exactly three integers")
    grid = tuple(int(item) for item in value)
    if any(item < 1 for item in grid):
        raise ValueError("pairwise_grid values must be positive")
    return grid
