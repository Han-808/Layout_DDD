"""P0b out-of-bounds metric: exact OBB versus six room planes with conservative VLM adjudication."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from benchmark.scene_io.object_normalization import (
    get_room_boundary,
    get_scene_height,
    normalize_objects,
)
from benchmark.visual_judge.p0b import LocalViewProvider, adjudicate_p0b_event


OOB_EVALUATOR_VERSION = "oob_p0b_v1"
DEFAULT_OOB_CONFIG = {
    "enabled": True,
    "official_mode": False,
    "detector_only": False,
    "numerical_eps": 1.0e-6,
    "score_mode": "invalid_object_count_over_objects",
}

OOB_NOTES = [
    "OOB is object-versus-room-plane only; object-architecture penetration is a separate P0b family.",
    "Exact OBB intervals over the six room planes supply evidence; a detector flag alone never reduces the score.",
    "Only objects whose final VLM verdict is invalid count as OOB.",
    "Objects inside all six planes pass directly; a positive flag never auto-fails and routes to semantic adjudication.",
]


class OOBEvaluationError(RuntimeError):
    """Raised when official OOB evaluation cannot complete required adjudication."""


def check_oob(
    scene: dict,
    config: dict | None = None,
    *,
    prompt: str | None = None,
    relationships: list[dict] | dict | None = None,
    render_evidence: list[str] | None = None,
    vlm_judge: object | None = None,
    local_view_provider: LocalViewProvider | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_OOB_CONFIG, **(config or {})}
    _validate_oob_config(cfg)
    room = _resolve_room(scene)
    if room is None:
        return _invalid_input_report()

    objects, object_errors = normalize_objects(scene)
    num_objects = len(objects)
    if num_objects == 0:
        return _empty_oob_report(object_errors, room)

    eps = float(cfg.get("numerical_eps", 1.0e-6))
    official_mode = bool(cfg.get("official_mode"))
    detector_only = bool(cfg.get("detector_only"))

    records: list[dict[str, Any]] = []
    requires_vlm_count = 0
    adjudication_failures: list[str] = []

    for obj in objects:
        record = _evaluate_object(
            scene=scene,
            obj=obj,
            room=room,
            eps=eps,
            cfg=cfg,
            prompt=prompt,
            relationships=relationships,
            render_evidence=render_evidence,
            vlm_judge=vlm_judge,
            local_view_provider=local_view_provider,
        )
        records.append(record)
        if record.get("requires_vlm"):
            requires_vlm_count += 1
        if record.get("adjudication_error"):
            adjudication_failures.append(str(record["adjudication_error"]))

    candidate_count = sum(1 for record in records if record.get("candidate_oob"))
    invalid_count = sum(1 for record in records if record.get("final_verdict") == "invalid")
    resolved = [record for record in records if record.get("final_verdict") in {"valid", "invalid"}]
    unresolved_vlm_count = sum(
        1
        for record in records
        if record.get("requires_vlm") and record.get("final_verdict") not in {"valid", "invalid"}
    )

    if adjudication_failures and official_mode:
        raise OOBEvaluationError("; ".join(adjudication_failures))
    if requires_vlm_count and official_mode and vlm_judge is None:
        raise OOBEvaluationError(
            "OOB events require P0b VLM adjudication in official mode, but no judge is configured"
        )

    if detector_only:
        score = None
        status = "detector_only"
    elif unresolved_vlm_count:
        score = None
        status = "requires_vlm"
    else:
        oob_rate = min(float(invalid_count) / float(max(num_objects, 1)), 1.0)
        score = float(1.0 - oob_rate)
        status = "checked"

    direct_valid = sum(1 for record in records if str(record.get("route") or "").startswith("direct_valid"))
    vlm_adjudicated = sum(1 for record in records if record.get("route") == "vlm_adjudicated")
    return {
        "metric": "oob",
        "evaluator_version": OOB_EVALUATOR_VERSION,
        "status": status,
        "score": score,
        "official_mode": official_mode,
        "detector_only": detector_only,
        "score_mode": str(cfg["score_mode"]),
        "num_objects": num_objects,
        "candidate_oob_count": candidate_count,
        "oob_count": invalid_count,
        "invalid_object_count": invalid_count,
        "oob_rate": None if score is None else float(1.0 - score),
        "requires_vlm_count": requires_vlm_count,
        "resolved_object_count": len(resolved),
        "objects": records,
        "object_errors": object_errors,
        "room": room["report"],
        "coverage": {
            "object_count": num_objects,
            "direct_valid_objects": direct_valid,
            "vlm_adjudicated_objects": vlm_adjudicated,
            "candidate_oob_objects": candidate_count,
            "evidence_level": "obb",
        },
        "notes": list(OOB_NOTES),
    }


def _evaluate_object(
    *,
    scene: dict,
    obj,
    room: dict[str, Any],
    eps: float,
    cfg: dict[str, Any],
    prompt: str | None,
    relationships: list[dict] | dict | None,
    render_evidence: list[str] | None,
    vlm_judge: object | None,
    local_view_provider: LocalViewProvider | None,
) -> dict[str, Any]:
    center = np.asarray(obj.center, dtype=float)
    # radius_axis = sum_i half[i] * |R[axis, i]| ; columns of R are local axes in world.
    radius = np.abs(np.asarray(obj.R, dtype=float)) @ np.asarray(obj.half, dtype=float)
    min_axis = center - radius
    max_axis = center + radius
    intervals = {
        "x": [float(min_axis[0]), float(max_axis[0])],
        "y": [float(min_axis[1]), float(max_axis[1])],
        "z": [float(min_axis[2]), float(max_axis[2])],
    }
    ceiling_z = room["ceiling_z"]
    plane_flags = {
        "west_oob": bool(min_axis[0] < room["min_x"] - eps),
        "east_oob": bool(max_axis[0] > room["max_x"] + eps),
        "south_oob": bool(min_axis[1] < room["min_y"] - eps),
        "north_oob": bool(max_axis[1] > room["max_y"] + eps),
        "floor_oob": bool(min_axis[2] < room["floor_z"] - eps),
        "ceiling_oob": bool(ceiling_z is not None and max_axis[2] > ceiling_z + eps),
    }
    candidate_oob = any(plane_flags.values())

    record: dict[str, Any] = {
        "object_id": obj.id,
        "obb_intervals": intervals,
        "plane_flags": plane_flags,
        "candidate_oob": candidate_oob,
        "requires_vlm": False,
        "route": None,
        "final_verdict": None,
        "affects_oob_score": False,
        "judge_result": None,
        "adjudication_error": None,
    }

    if not candidate_oob:
        record.update(
            {
                "route": "direct_valid_inside",
                "final_verdict": "valid",
                "affects_oob_score": True,
            }
        )
        return record

    record["requires_vlm"] = True
    if bool(cfg.get("detector_only")):
        return record
    if vlm_judge is None:
        return record

    event = {
        "object_id": obj.id,
        "object_ids": [obj.id],
        "architecture_element": "room_bounds",
        "plane_flags": plane_flags,
    }
    detector_evidence = {
        "detector": OOB_EVALUATOR_VERSION,
        "plane_flags": plane_flags,
        "obb_intervals": intervals,
        "room": room["report"],
        "numerical_eps": eps,
        "object": {
            "id": obj.id,
            "category": obj.category,
            "description": obj.desc,
            "center": [float(value) for value in center],
            "size": [float(value) for value in np.asarray(obj.size, dtype=float)],
            "rotation_degrees": [float(value) for value in np.asarray(obj.rotation, dtype=float)],
            "geometry_provenance": _geometry_provenance(scene, obj.id),
        },
        "extracted_relationships_are_claims_only": True,
    }
    try:
        judge_result = adjudicate_p0b_event(
            metric="oob",
            event=event,
            prompt=str(prompt or ""),
            relationships=relationships,
            scene=scene,
            detector_evidence=detector_evidence,
            judge=vlm_judge,
            object_ids=[obj.id],
            overview_render_evidence=list(render_evidence or []),
            local_view_provider=local_view_provider,
        )
    except Exception as exc:
        record["adjudication_error"] = f"{type(exc).__name__}: {exc}"
        record["route"] = "vlm_adjudication_failed"
        if bool(cfg.get("official_mode")):
            raise OOBEvaluationError(record["adjudication_error"]) from exc
        return record

    verdict = str(judge_result.get("verdict"))
    record.update(
        {
            "route": "vlm_adjudicated",
            "final_verdict": verdict,
            "affects_oob_score": True,
            "judge_result": deepcopy(judge_result),
        }
    )
    return record


def _resolve_room(scene: dict) -> dict[str, Any] | None:
    points = np.asarray(get_room_boundary(scene), dtype=float)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        return None
    min_x, max_x = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    min_y, max_y = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    if not (max_x > min_x and max_y > min_y):
        return None
    expected = np.asarray(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
        dtype=float,
    )
    if not all(np.any(np.all(np.isclose(points, corner, atol=1.0e-6, rtol=0.0), axis=1)) for corner in expected):
        return None
    scene_height = get_scene_height(scene)
    if scene_height is None or not math.isfinite(float(scene_height)):
        return None
    ceiling_z = float(scene_height)
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "floor_z": 0.0,
        "ceiling_z": ceiling_z,
        "report": {
            "boundary": [[float(point[0]), float(point[1])] for point in points],
            "width": max_x - min_x,
            "depth": max_y - min_y,
            "height": ceiling_z,
            "floor_z": 0.0,
        },
    }


def _geometry_provenance(scene: dict, object_id: str) -> Any:
    for item in scene.get("objects", []) if isinstance(scene, dict) else []:
        if isinstance(item, dict) and str(item.get("id")) == str(object_id):
            return item.get("geometry_provenance")
    return None


def _invalid_input_report() -> dict[str, Any]:
    return {
        "metric": "oob",
        "evaluator_version": OOB_EVALUATOR_VERSION,
        "status": "invalid_input",
        "score": 0.0,
        "reason": "scene boundary is missing or not a resolvable rectangular room",
        "num_objects": 0,
        "candidate_oob_count": 0,
        "oob_count": 0,
        "invalid_object_count": 0,
        "requires_vlm_count": 0,
        "resolved_object_count": 0,
        "objects": [],
        "object_errors": {},
        "coverage": _empty_coverage(),
        "notes": list(OOB_NOTES),
    }


def _empty_oob_report(object_errors: dict[str, str], room: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric": "oob",
        "evaluator_version": OOB_EVALUATOR_VERSION,
        "status": "not_applicable",
        "score": None,
        "reason": "no_physical_objects",
        "num_objects": 0,
        "candidate_oob_count": 0,
        "oob_count": 0,
        "invalid_object_count": 0,
        "requires_vlm_count": 0,
        "resolved_object_count": 0,
        "objects": [],
        "object_errors": object_errors,
        "room": room["report"],
        "coverage": _empty_coverage(),
        "notes": list(OOB_NOTES),
    }


def _validate_oob_config(config: dict[str, Any]) -> None:
    if bool(config.get("official_mode")) and bool(config.get("detector_only")):
        raise ValueError("oob.official_mode and oob.detector_only are mutually exclusive")
    eps = float(config.get("numerical_eps", 1.0e-6))
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("oob.numerical_eps must be a finite non-negative number")
    if config.get("score_mode") != "invalid_object_count_over_objects":
        raise ValueError("oob.score_mode must be 'invalid_object_count_over_objects'")


def _empty_coverage() -> dict[str, Any]:
    return {
        "object_count": 0,
        "direct_valid_objects": 0,
        "vlm_adjudicated_objects": 0,
        "candidate_oob_objects": 0,
        "evidence_level": "obb",
    }
