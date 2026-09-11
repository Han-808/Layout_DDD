from __future__ import annotations

from benchmark.evaluator.OOR.geometry import NormalizedObject, point_in_obb, sample_points_in_obb


def check_within(subject: NormalizedObject, anchor: NormalizedObject, config: dict | None = None) -> dict:
    cfg = config or {}
    threshold = float(cfg.get("inside_ratio_threshold", 0.80))
    try:
        inside_ratio = _inside_ratio(subject, anchor)
        score = 1.0 if inside_ratio >= threshold else 0.0
        return {
            "relation": "within",
            "category": "containment",
            "subject_id": subject.id,
            "object_id": anchor.id,
            "passed": score >= 0.5,
            "score": score,
            "evidence": {"inside_ratio": float(inside_ratio), "threshold": threshold},
            "status": "checked",
        }
    except (TypeError, ValueError, AttributeError) as exc:
        return _invalid_result("within", subject, anchor, str(exc))


def check_out_of(subject: NormalizedObject, anchor: NormalizedObject, config: dict | None = None) -> dict:
    cfg = config or {}
    threshold = float(cfg.get("out_of_ratio_threshold", 0.10))
    try:
        inside_ratio = _inside_ratio(subject, anchor)
        score = 1.0 if inside_ratio <= threshold else 0.0
        return {
            "relation": "out_of",
            "category": "containment",
            "subject_id": subject.id,
            "object_id": anchor.id,
            "passed": score >= 0.5,
            "score": score,
            "evidence": {"inside_ratio": float(inside_ratio), "threshold": threshold},
            "status": "checked",
        }
    except (TypeError, ValueError, AttributeError) as exc:
        return _invalid_result("out_of", subject, anchor, str(exc))


def _inside_ratio(subject: NormalizedObject, anchor: NormalizedObject) -> float:
    points = sample_points_in_obb(subject, grid=(3, 3, 3))
    if len(points) == 0:
        return 0.0
    inside = sum(1 for point in points if point_in_obb(point, anchor, eps=1.0e-8))
    return float(inside) / float(len(points))


def _invalid_result(relation: str, subject: object, anchor: object, reason: str) -> dict:
    return {
        "relation": relation,
        "category": "containment",
        "subject_id": getattr(subject, "id", ""),
        "object_id": getattr(anchor, "id", ""),
        "passed": False,
        "score": 0.0,
        "evidence": {"reason": reason},
        "status": "invalid_input",
    }
