from __future__ import annotations

import math

from benchmark.evaluator.OOR.geometry import NormalizedObject, footprint_edge_distance


def check_near(subject: NormalizedObject, anchor: NormalizedObject, config: dict | None = None) -> dict:
    cfg = config or {}
    try:
        dist = footprint_edge_distance(subject, anchor)
        anchor_diag = math.hypot(float(anchor.size[0]), float(anchor.size[1]))
        threshold = _clamp(
            float(cfg.get("alpha", 1.5)) * anchor_diag,
            float(cfg.get("min_threshold", 0.30)),
            float(cfg.get("max_threshold", 1.50)),
        )
        score = 1.0 if dist <= threshold else 0.0
        return {
            "relation": "near",
            "category": "proximity",
            "subject_id": subject.id,
            "object_id": anchor.id,
            "passed": score >= 0.5,
            "score": score,
            "evidence": {
                "footprint_edge_distance": float(dist),
                "anchor_xy_diagonal": float(anchor_diag),
                "threshold": float(threshold),
            },
            "status": "checked",
        }
    except (TypeError, ValueError, AttributeError) as exc:
        return _invalid_result("near", subject, anchor, str(exc))


def _invalid_result(relation: str, subject: object, anchor: object, reason: str) -> dict:
    return {
        "relation": relation,
        "category": "proximity",
        "subject_id": getattr(subject, "id", ""),
        "object_id": getattr(anchor, "id", ""),
        "passed": False,
        "score": 0.0,
        "evidence": {"reason": reason},
        "status": "invalid_input",
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
