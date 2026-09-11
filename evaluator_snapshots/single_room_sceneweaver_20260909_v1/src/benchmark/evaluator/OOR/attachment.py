from __future__ import annotations

import numpy as np

from benchmark.evaluator.OOR.geometry import NormalizedObject, interval_gap, interval_overlap_ratio, project_obb_to_axis


def check_contact(subject: NormalizedObject, anchor: NormalizedObject, config: dict | None = None) -> dict:
    cfg = config or {}
    eps_contact = float(cfg.get("eps_contact", 0.05))
    min_projected_overlap = float(cfg.get("min_projected_overlap", 0.15))
    try:
        candidates = []
        axes = [
            ("right", anchor.right, anchor.front, anchor.up),
            ("left", -anchor.right, anchor.front, anchor.up),
            ("front", anchor.front, anchor.right, anchor.up),
            ("back", -anchor.front, anchor.right, anchor.up),
            ("up", anchor.up, anchor.right, anchor.front),
            ("down", -anchor.up, anchor.right, anchor.front),
        ]
        for label, normal, axis_1, axis_2 in axes:
            interval_a = project_obb_to_axis(subject, normal)
            interval_b = project_obb_to_axis(anchor, normal)
            gap = interval_gap(interval_a, interval_b)
            overlap_1 = interval_overlap_ratio(project_obb_to_axis(subject, axis_1), project_obb_to_axis(anchor, axis_1))
            overlap_2 = interval_overlap_ratio(project_obb_to_axis(subject, axis_2), project_obb_to_axis(anchor, axis_2))
            plane_overlap = min(overlap_1, overlap_2)
            passes = gap <= eps_contact and plane_overlap >= min_projected_overlap
            candidates.append(
                {
                    "label": label,
                    "normal": np.asarray(normal, dtype=float),
                    "gap": float(gap),
                    "projected_overlap": float(plane_overlap),
                    "passes": bool(passes),
                }
            )
        best = sorted(candidates, key=lambda item: (not item["passes"], item["gap"], -item["projected_overlap"]))[0]
        score = 1.0 if best["passes"] else 0.0
        return {
            "relation": "contact",
            "category": "attachment",
            "subject_id": subject.id,
            "object_id": anchor.id,
            "passed": score >= 0.5,
            "score": score,
            "evidence": {
                "best_gap": float(best["gap"]),
                "best_normal": str(best["label"]),
                "best_projected_overlap": float(best["projected_overlap"]),
                "eps_contact": eps_contact,
                "min_projected_overlap": min_projected_overlap,
                "proxy": "obb_bbox_contact_only",
            },
            "status": "checked",
        }
    except (TypeError, ValueError, AttributeError) as exc:
        return {
            "relation": "contact",
            "category": "attachment",
            "subject_id": getattr(subject, "id", ""),
            "object_id": getattr(anchor, "id", ""),
            "passed": False,
            "score": 0.0,
            "evidence": {"reason": str(exc), "proxy": "obb_bbox_contact_only"},
            "status": "invalid_input",
        }
