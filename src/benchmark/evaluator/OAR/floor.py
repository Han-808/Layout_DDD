from __future__ import annotations

from typing import Any

from benchmark.evaluator.OOR.geometry import NormalizedObject


def check_on_floor(subject: NormalizedObject, room: object | None = None, config: dict | None = None) -> dict:
    cfg = config or {}
    eps_floor = float(cfg.get("eps_floor", 0.05))
    try:
        bottom = float(subject.bottom)
    except (AttributeError, TypeError, ValueError):
        return _invalid_result("subject object is missing valid bottom geometry")
    floor_z = 0.0
    gap = abs(bottom - floor_z)
    passed = gap <= eps_floor
    return {
        "relation": "on_floor",
        "category": "floor",
        "subject_id": subject.id,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "evidence": {
            "bottom_z": bottom,
            "floor_z": floor_z,
            "gap": gap,
            "eps_floor": eps_floor,
        },
        "status": "checked",
    }


def _invalid_result(reason: str, subject_id: str = "") -> dict[str, Any]:
    return {
        "relation": "on_floor",
        "category": "floor",
        "subject_id": subject_id,
        "passed": False,
        "score": 0.0,
        "evidence": {"reason": reason},
        "status": "invalid_input",
    }
