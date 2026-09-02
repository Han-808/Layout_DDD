"""Polygon-aware OOB metric used only by the non-rectangular evaluator."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points

from benchmark.evaluator.generic_validity.geometry import get_obb_corners
from benchmark.non_rectangular.geometry import (
    PolygonRoomGeometryError,
    polygon_geometry_from_scene,
)
from benchmark.scene_io.object_normalization import normalize_objects
from benchmark.visual_judge.contracts import (
    response_schema_audit_from_exception,
)
from benchmark.visual_judge.p0b import LocalViewProvider, adjudicate_p0b_event
from benchmark.visual_judge.runtime import EvidenceControlUnresolvedError


POLYGON_OOB_EVALUATOR_VERSION = "non_rectangular_polygon_oob_p0b_v2"
DEFAULT_FLOOR_CONTACT_TOLERANCE_M = 0.005
DEFAULT_POLYGON_OOB_CONFIG = {
    "official_mode": False,
    "detector_only": False,
    "numerical_eps": 1.0e-6,
    "floor_contact_tolerance_m": DEFAULT_FLOOR_CONTACT_TOLERANCE_M,
    "score_mode": "invalid_object_count_over_objects",
}


class PolygonOOBEvaluationError(RuntimeError):
    """Raised when required non-rectangular OOB adjudication cannot finish."""


def check_polygon_oob(
    scene: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    prompt: str | None = None,
    relationships: list[dict] | dict | None = None,
    render_evidence: list[str] | None = None,
    vlm_judge: object | None = None,
    local_view_provider: LocalViewProvider | None = None,
) -> dict[str, Any]:
    """Evaluate exact OBB footprint difference against one room polygon.

    The scoring unit remains one object even when the footprint crosses more
    than one wall.  Edge-scoped records are detector/camera evidence only.
    """

    cfg = {**DEFAULT_POLYGON_OOB_CONFIG, **(config or {})}
    _validate_config(cfg)
    try:
        geometry = polygon_geometry_from_scene(scene)
    except PolygonRoomGeometryError as exc:
        return _invalid_input_report(str(exc))
    if geometry is None:
        return _invalid_input_report(
            "scene is not explicitly projected for non-rectangular evaluation"
        )
    objects, object_errors = normalize_objects(scene)
    num_objects = len(objects)
    if num_objects == 0:
        return {
            **_base_report(geometry=geometry, object_errors=object_errors),
            "status": "not_applicable",
            "score": None,
            "reason": "no_physical_objects",
        }

    raw_by_id = {
        str(item.get("id")): item
        for item in scene.get("objects", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    records: list[dict[str, Any]] = []
    for obj in objects:
        records.append(
            _evaluate_object(
                scene=scene,
                raw_object=raw_by_id.get(obj.id, {}),
                obj=obj,
                geometry=geometry,
                cfg=cfg,
                prompt=prompt,
                relationships=relationships,
                render_evidence=render_evidence,
                vlm_judge=vlm_judge,
                local_view_provider=local_view_provider,
            )
        )

    failures = [
        str(item["adjudication_error"])
        for item in records
        if item.get("adjudication_error")
    ]
    if failures and bool(cfg["official_mode"]):
        raise PolygonOOBEvaluationError("; ".join(failures))
    requires_vlm_count = sum(bool(item["requires_vlm"]) for item in records)
    unresolved_count = sum(
        bool(item["requires_vlm"])
        and item.get("final_verdict") not in {"valid", "invalid"}
        for item in records
    )
    invalid_count = sum(
        item.get("final_verdict") == "invalid" for item in records
    )
    if bool(cfg["detector_only"]):
        status, score = "detector_only", None
    elif unresolved_count:
        status, score = "requires_vlm", None
    else:
        status = "checked"
        score = 1.0 - min(
            float(invalid_count) / float(max(num_objects, 1)),
            1.0,
        )
    candidate_count = sum(bool(item["candidate_oob"]) for item in records)
    return {
        "metric": "oob",
        "evaluator_version": POLYGON_OOB_EVALUATOR_VERSION,
        "status": status,
        "score": score,
        "official_mode": bool(cfg["official_mode"]),
        "detector_only": bool(cfg["detector_only"]),
        "score_mode": str(cfg["score_mode"]),
        "num_objects": num_objects,
        "candidate_oob_count": candidate_count,
        "oob_count": invalid_count,
        "invalid_object_count": invalid_count,
        "oob_rate": None if score is None else 1.0 - float(score),
        "requires_vlm_count": requires_vlm_count,
        "resolved_object_count": sum(
            item.get("final_verdict") in {"valid", "invalid"}
            for item in records
        ),
        "objects": records,
        "object_errors": object_errors,
        "room": geometry.public_dict(),
        "coverage": {
            "object_count": num_objects,
            "direct_valid_objects": sum(
                str(item.get("route") or "").startswith("direct_valid")
                for item in records
            ),
            "vlm_adjudicated_objects": sum(
                item.get("route") == "vlm_adjudicated" for item in records
            ),
            "candidate_oob_objects": candidate_count,
            "evidence_level": "obb_footprint_polygon_difference",
        },
        "notes": [
            "Horizontal OOB is measured as projected OBB footprint minus the exact room polygon.",
            "One object remains one scoring event even when multiple wall edges are crossed.",
            "Wall-edge records use layout-provided inward normals for evidence acquisition only.",
            "Ceiling is outside the frozen non-rectangular benchmark scope.",
        ],
    }


def _evaluate_object(
    *,
    scene: dict[str, Any],
    raw_object: dict[str, Any],
    obj: Any,
    geometry: Any,
    cfg: dict[str, Any],
    prompt: str | None,
    relationships: list[dict] | dict | None,
    render_evidence: list[str] | None,
    vlm_judge: object | None,
    local_view_provider: LocalViewProvider | None,
) -> dict[str, Any]:
    eps = float(cfg["numerical_eps"])
    floor_tolerance = float(cfg["floor_contact_tolerance_m"])
    footprint = MultiPoint(get_obb_corners(obj)[:, :2]).convex_hull
    accepted_room = geometry.polygon.buffer(eps, join_style="mitre")
    outside = footprint.difference(accepted_room)
    outside_area = float(outside.area)
    area_threshold = max(eps * eps, float(footprint.area) * 1.0e-10)
    horizontal_oob = outside_area > area_threshold
    corners = get_obb_corners(obj)
    minimum_z = float(np.min(corners[:, 2]))
    floor_penetration = max(float(geometry.floor_z_m) - minimum_z, 0.0)
    floor_oob = floor_penetration > floor_tolerance + eps
    within_floor_tolerance = bool(
        floor_penetration > eps and not floor_oob
    )
    wall_records = (
        geometry.violated_walls(footprint) if horizontal_oob else []
    )
    edge_records = [
        _edge_violation_record(
            geometry=geometry,
            footprint=footprint,
            measurement=measurement,
        )
        for measurement in wall_records
    ]
    record: dict[str, Any] = {
        "object_id": obj.id,
        "candidate_oob": bool(horizontal_oob or floor_oob),
        "horizontal_oob": horizontal_oob,
        "floor_oob": floor_oob,
        "outside_area_m2": outside_area,
        "outside_area_ratio": (
            outside_area / float(footprint.area)
            if float(footprint.area) > eps * eps
            else 0.0
        ),
        "maximum_horizontal_penetration_m": _maximum_outside_distance(
            outside,
            geometry.polygon.boundary,
        ),
        "floor_penetration_m": floor_penetration,
        "within_floor_contact_tolerance": within_floor_tolerance,
        "floor_contact_tolerance_m": floor_tolerance,
        "numerical_eps": eps,
        "violated_wall_ids": [
            str(item["wall_id"]) for item in edge_records
        ],
        "violated_edges": edge_records,
        "footprint_xy": [
            [float(x), float(y)]
            for x, y in list(footprint.exterior.coords)[:-1]
        ],
        "requires_vlm": False,
        "route": None,
        "final_verdict": None,
        "affects_oob_score": False,
        "judge_result": None,
        "adjudication_error": None,
    }
    if not record["candidate_oob"]:
        record.update(
            route=(
                "direct_valid_floor_contact_tolerance"
                if within_floor_tolerance
                else "direct_valid_inside_polygon"
            ),
            final_verdict="valid",
            affects_oob_score=True,
        )
        return record

    record["requires_vlm"] = True
    if bool(cfg["detector_only"]) or vlm_judge is None:
        return record
    event = {
        "object_id": obj.id,
        "object_ids": [obj.id],
        "architecture_element": "room_polygon_and_floor",
        "violated_wall_ids": list(record["violated_wall_ids"]),
        "violated_edges": deepcopy(edge_records),
        "floor_oob": floor_oob,
        "plane_flags": {"floor_oob": floor_oob},
    }
    detector_evidence = {
        "detector": POLYGON_OOB_EVALUATOR_VERSION,
        "room_geometry": geometry.public_dict(),
        "outside_area_m2": outside_area,
        "outside_area_ratio": record["outside_area_ratio"],
        "maximum_horizontal_penetration_m": record[
            "maximum_horizontal_penetration_m"
        ],
        "floor_penetration_m": floor_penetration,
        "plane_flags": {"floor_oob": floor_oob},
        "floor_contact_tolerance_m": floor_tolerance,
        "violated_wall_ids": list(record["violated_wall_ids"]),
        "violated_edges": deepcopy(edge_records),
        "object": {
            "id": obj.id,
            "category": obj.category,
            "description": obj.desc,
            "center": [float(value) for value in obj.center],
            "size": [float(value) for value in obj.size],
            "rotation_degrees": [float(value) for value in obj.rotation],
            "geometry_provenance": raw_object.get(
                "geometry_provenance"
            ),
        },
        "extracted_relationships_are_claims_only": True,
    }
    try:
        judgement = adjudicate_p0b_event(
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
        record["route"] = "vlm_adjudication_failed"
        record["adjudication_error"] = f"{type(exc).__name__}: {exc}"
        schema_audit = response_schema_audit_from_exception(exc)
        if schema_audit is not None:
            record["adjudication_failure_audit"] = schema_audit
        if bool(cfg["official_mode"]):
            raise PolygonOOBEvaluationError(
                str(record["adjudication_error"])
            ) from exc
        return record
    verdict = str(judgement.get("verdict"))
    record.update(
        route="vlm_adjudicated",
        final_verdict=verdict,
        affects_oob_score=True,
        judge_result=deepcopy(judgement),
    )
    return record


def _edge_violation_record(
    *,
    geometry: Any,
    footprint: Polygon,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    wall = geometry.wall_by_id(str(measurement["wall_id"]))
    crossing = footprint.boundary.intersection(wall.line)
    if crossing.is_empty:
        wall_point, footprint_point = nearest_points(wall.line, footprint)
        focus = footprint_point
        wall_focus = wall_point
    else:
        focus = crossing.centroid
        wall_focus = crossing.centroid
    return {
        **deepcopy(measurement),
        "focus_xy": [float(focus.x), float(focus.y)],
        "wall_focus_xy": [float(wall_focus.x), float(wall_focus.y)],
        "edge_local_frame": {
            "inward_normal_xy": list(wall.inward_normal_xy),
            "tangent_xy": list(wall.tangent_xy),
        },
    }


def _maximum_outside_distance(outside: Any, boundary: Any) -> float:
    if outside.is_empty:
        return 0.0
    coordinates: list[tuple[float, float]] = []

    def collect(value: Any) -> None:
        exterior = getattr(value, "exterior", None)
        if exterior is not None:
            coordinates.extend(
                (float(x), float(y)) for x, y in exterior.coords
            )
        for item in getattr(value, "geoms", ()):
            collect(item)

    collect(outside)
    return max(
        (float(Point(point).distance(boundary)) for point in coordinates),
        default=0.0,
    )


def _base_report(*, geometry: Any, object_errors: dict[str, str]) -> dict[str, Any]:
    return {
        "metric": "oob",
        "evaluator_version": POLYGON_OOB_EVALUATOR_VERSION,
        "num_objects": 0,
        "candidate_oob_count": 0,
        "oob_count": 0,
        "invalid_object_count": 0,
        "requires_vlm_count": 0,
        "resolved_object_count": 0,
        "objects": [],
        "object_errors": object_errors,
        "room": geometry.public_dict(),
        "coverage": {
            "object_count": 0,
            "direct_valid_objects": 0,
            "vlm_adjudicated_objects": 0,
            "candidate_oob_objects": 0,
            "evidence_level": "obb_footprint_polygon_difference",
        },
        "notes": [],
    }


def _invalid_input_report(reason: str) -> dict[str, Any]:
    return {
        "metric": "oob",
        "evaluator_version": POLYGON_OOB_EVALUATOR_VERSION,
        "status": "invalid_input",
        "score": 0.0,
        "reason": reason,
        "num_objects": 0,
        "candidate_oob_count": 0,
        "oob_count": 0,
        "invalid_object_count": 0,
        "requires_vlm_count": 0,
        "resolved_object_count": 0,
        "objects": [],
        "object_errors": {},
        "coverage": {
            "object_count": 0,
            "direct_valid_objects": 0,
            "vlm_adjudicated_objects": 0,
            "candidate_oob_objects": 0,
            "evidence_level": "unavailable",
        },
        "notes": [],
    }


def _validate_config(config: dict[str, Any]) -> None:
    if bool(config["official_mode"]) and bool(config["detector_only"]):
        raise ValueError("official_mode and detector_only are mutually exclusive")
    for name in ("numerical_eps", "floor_contact_tolerance_m"):
        value = config.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if float(config["floor_contact_tolerance_m"]) < float(config["numerical_eps"]):
        raise ValueError(
            "floor_contact_tolerance_m must be >= numerical_eps"
        )
    if config.get("score_mode") != "invalid_object_count_over_objects":
        raise ValueError(
            "score_mode must be 'invalid_object_count_over_objects'"
        )


__all__ = [
    "DEFAULT_POLYGON_OOB_CONFIG",
    "POLYGON_OOB_EVALUATOR_VERSION",
    "PolygonOOBEvaluationError",
    "check_polygon_oob",
]
