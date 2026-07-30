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
from benchmark.visual_judge.runtime import EvidenceControlUnresolvedError


OOB_EVALUATOR_VERSION = "oob_p0b_v2"
# ``numerical_eps`` is computation robustness only; it governs the wall and
# ceiling planes. ``floor_contact_tolerance_m`` is a separate semantic contact
# tolerance for the floor plane: ordinary floor-standing geometry may sink below
# the nominal floor by this much because of asset bounds, placement, or contact
# modelling, so a floor crossing within this band is clearly safe and bypasses
# VLM adjudication rather than being routed as a candidate violation.
DEFAULT_FLOOR_CONTACT_TOLERANCE_M = 0.005
DEFAULT_OOB_CONFIG = {
    "enabled": True,
    "official_mode": False,
    "detector_only": False,
    "numerical_eps": 1.0e-6,
    "floor_contact_tolerance_m": DEFAULT_FLOOR_CONTACT_TOLERANCE_M,
    "score_mode": "invalid_object_count_over_objects",
}

# Canonical order of the six room planes for the raw penetration measurement.
OOB_PLANES = ("west_oob", "east_oob", "south_oob", "north_oob", "floor_oob", "ceiling_oob")

OOB_NOTES = [
    "OOB is object-versus-room-plane only; object-architecture penetration is a separate P0b family.",
    "Exact OBB intervals over the six room planes supply evidence; a detector flag alone never reduces the score.",
    "Only objects whose final VLM verdict is invalid count as OOB.",
    "Objects inside all six planes pass directly; a positive flag never auto-fails and routes to semantic adjudication.",
    "numerical_eps is computation robustness for the wall and ceiling planes; the floor plane uses the separate "
    "floor_contact_tolerance_m semantic contact tolerance so ordinary sub-centimetre floor sink stays clearly valid.",
    "plane_penetration_m records the raw non-negative crossing depth for all six planes independently of the "
    "semantic plane_flags, so an accepted shallow floor sink is never erased from the diagnostics.",
    "A positive floor sink that stays within floor_contact_tolerance_m routes direct_valid_floor_contact_tolerance; "
    "an OBB with no meaningful crossing routes direct_valid_inside.",
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
    floor_tol = float(cfg.get("floor_contact_tolerance_m", DEFAULT_FLOOR_CONTACT_TOLERANCE_M))
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
            floor_tol=floor_tol,
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
    floor_tol: float,
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
    # Raw, non-negative penetration past each nominal plane, computed independently
    # of the semantic plane_flags so an accepted shallow floor sink is preserved.
    plane_penetration = _plane_penetration(min_axis, max_axis, room, ceiling_z)
    floor_penetration = plane_penetration["floor_oob"]
    # Wall and ceiling planes use ``numerical_eps`` for floating-point robustness.
    # The floor plane uses the separate semantic ``floor_contact_tolerance_m`` so
    # ordinary shallow floor sink is treated as valid contact rather than a
    # candidate violation routed to the VLM. Numerical robustness stays additive
    # so exactly ``floor_contact_tolerance_m`` (with float noise) is still accepted.
    plane_flags = {
        "west_oob": bool(plane_penetration["west_oob"] > eps),
        "east_oob": bool(plane_penetration["east_oob"] > eps),
        "south_oob": bool(plane_penetration["south_oob"] > eps),
        "north_oob": bool(plane_penetration["north_oob"] > eps),
        "floor_oob": bool(floor_penetration > floor_tol + eps),
        "ceiling_oob": bool(ceiling_z is not None and plane_penetration["ceiling_oob"] > eps),
    }
    candidate_oob = any(plane_flags.values())
    # A positive floor sink beyond numerical noise that does not meaningfully
    # exceed the semantic tolerance. An OBB flush on the floor is not "within
    # tolerance" because it has no actual penetration.
    within_floor_contact_tolerance = bool(
        floor_penetration > eps and not plane_flags["floor_oob"]
    )
    # Flagged-only view retained as a backward-compatible alias of the raw field.
    crossing_depths = {
        plane: plane_penetration[plane] for plane in OOB_PLANES if plane_flags[plane]
    }

    record: dict[str, Any] = {
        "object_id": obj.id,
        "obb_intervals": intervals,
        "plane_flags": plane_flags,
        "plane_penetration_m": plane_penetration,
        "within_floor_contact_tolerance": within_floor_contact_tolerance,
        "floor_contact_tolerance_m": floor_tol,
        "numerical_eps": eps,
        "crossing_depths_m": crossing_depths,
        "floor_penetration_m": floor_penetration,
        "candidate_oob": candidate_oob,
        "requires_vlm": False,
        "route": None,
        "final_verdict": None,
        "affects_oob_score": False,
        "judge_result": None,
        "adjudication_error": None,
    }

    if not candidate_oob:
        # Both routes are direct-valid; the tolerance route makes explicit that the
        # raw OBB is not strictly inside but sinks only within floor-contact tolerance.
        route = (
            "direct_valid_floor_contact_tolerance"
            if within_floor_contact_tolerance
            else "direct_valid_inside"
        )
        record.update(
            {
                "route": route,
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
        "floor_contact_tolerance_m": floor_tol,
        "plane_penetration_m": plane_penetration,
        "within_floor_contact_tolerance": within_floor_contact_tolerance,
        "crossing_depths_m": crossing_depths,
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
    except EvidenceControlUnresolvedError as exc:
        record["route"] = "unresolved"
        record["evidence_control"] = exc.result.to_dict()
        return record
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


def _plane_penetration(
    min_axis: np.ndarray,
    max_axis: np.ndarray,
    room: dict[str, Any],
    ceiling_z: float | None,
) -> dict[str, float]:
    """Raw, non-negative penetration depth (metres) past each of the six planes.

    Depths are measured against the nominal room boundary and clamped at zero, so
    they are reported for every plane regardless of the semantic plane_flags. This
    preserves the true measured crossing as a fact for the VLM and the diagnostics,
    and never erases an accepted shallow floor sink. When the room has no ceiling
    the ceiling penetration is reported as ``0.0`` rather than an error sentinel.
    """

    ceiling_penetration = (
        max(float(max_axis[2]) - float(ceiling_z), 0.0) if ceiling_z is not None else 0.0
    )
    return {
        "west_oob": max(room["min_x"] - float(min_axis[0]), 0.0),
        "east_oob": max(float(max_axis[0]) - room["max_x"], 0.0),
        "south_oob": max(room["min_y"] - float(min_axis[1]), 0.0),
        "north_oob": max(float(max_axis[1]) - room["max_y"], 0.0),
        "floor_oob": max(room["floor_z"] - float(min_axis[2]), 0.0),
        "ceiling_oob": ceiling_penetration,
    }


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
    floor_tol = float(config.get("floor_contact_tolerance_m", DEFAULT_FLOOR_CONTACT_TOLERANCE_M))
    if not math.isfinite(floor_tol) or floor_tol < 0.0:
        raise ValueError("oob.floor_contact_tolerance_m must be a finite non-negative number")
    if floor_tol < eps:
        raise ValueError("oob.floor_contact_tolerance_m must be >= oob.numerical_eps")
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
