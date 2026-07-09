from __future__ import annotations

import numpy as np

from benchmark.evaluator.OOR.geometry import NormalizedObject, angle_between_degrees, ray_intersects_obb, sample_front_face_points


def check_face_to(subject: NormalizedObject, anchor: NormalizedObject, config: dict | None = None) -> dict:
    cfg = config or {}
    hit_rate_threshold = float(cfg.get("hit_rate_threshold", 0.30))
    angle_threshold = float(cfg.get("angle_threshold_degrees", 20))
    max_distance = float(cfg.get("max_distance", 5.0))
    try:
        origins = sample_front_face_points(subject, grid=(3, 3))
        hits = sum(1 for origin in origins if ray_intersects_obb(origin, subject.front, anchor, max_distance=max_distance))
        hit_rate = float(hits) / float(len(origins)) if len(origins) else 0.0
        target_vector_xy = anchor.center[:2] - subject.center[:2]
        front_xy = subject.front[:2]
        angle = angle_between_degrees(front_xy, target_vector_xy)
        passed = hit_rate >= hit_rate_threshold or angle <= angle_threshold
        score = 1.0 if passed else 0.0
        return {
            "relation": "face_to",
            "category": "facing",
            "subject_id": subject.id,
            "object_id": anchor.id,
            "passed": bool(passed),
            "score": score,
            "evidence": {
                "hit_rate": float(hit_rate),
                "num_rays": int(len(origins)),
                "hits": int(hits),
                "angle_degrees": float(angle),
                "hit_rate_threshold": hit_rate_threshold,
                "angle_threshold_degrees": angle_threshold,
                "max_distance": max_distance,
            },
            "status": "checked",
        }
    except (TypeError, ValueError, AttributeError) as exc:
        return {
            "relation": "face_to",
            "category": "facing",
            "subject_id": getattr(subject, "id", ""),
            "object_id": getattr(anchor, "id", ""),
            "passed": False,
            "score": 0.0,
            "evidence": {"reason": str(exc)},
            "status": "invalid_input",
        }
