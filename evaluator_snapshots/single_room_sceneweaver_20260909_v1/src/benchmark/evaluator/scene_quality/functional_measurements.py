"""Deterministic, check-scoped measurements for Functional judgement.

The bank is built from the accepted Functional check ledger before any
camera unit is scheduled.  It records spatial facts but has no authority to
create checks, choose cameras, or decide metric validity.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from benchmark.evaluator.scene_quality.functional_geometry import (
    build_functional_geometry_observations,
)


FUNCTIONAL_MEASUREMENT_BANK_VERSION = "functional_measurement_bank_v1"
FUNCTIONAL_MEASUREMENT_STATUSES = frozenset(
    {"complete", "partial", "unavailable"}
)

_FORBIDDEN_EXTENSION_FIELDS = frozenset(
    {
        "verdict",
        "validity",
        "score",
        "defect",
        "defects",
        "is_invalid",
        "conclusion",
        "scene_mutation",
        "camera_pose",
    }
)
_DIRECTION_REQUIRED_CHECK_TYPES = frozenset(
    {"architecture_orientation", "directional_correspondence"}
)


def build_functional_measurement_bank(
    *,
    scene: dict[str, Any] | None,
    functional_check_ledger: dict[str, Any],
    discovery: dict[str, Any],
) -> dict[str, Any]:
    """Measure every accepted check independently of camera scheduling."""

    normalized_scene = scene if isinstance(scene, dict) else {}
    objects = {
        str(item.get("id")): item
        for item in normalized_scene.get("objects") or []
        if isinstance(item, dict) and item.get("id")
    }
    directed = {
        str(item.get("target_id")): item
        for item in discovery.get("directed_surface_targets") or []
        if isinstance(item, dict) and item.get("target_id")
    }
    hypotheses: dict[str, dict[str, Any]] = {}
    for target_id, record in directed.items():
        hypothesis = record.get("precomputed_usable_surface_hypothesis")
        if isinstance(hypothesis, dict):
            hypotheses[target_id] = deepcopy(hypothesis)
    checks = [
        item
        for item in functional_check_ledger.get("checks") or []
        if isinstance(item, dict) and item.get("check_id")
    ]
    measurements = [
        _measure_check(
            scene=normalized_scene,
            objects=objects,
            directed=directed,
            hypotheses=hypotheses,
            check=check,
        )
        for check in checks
    ]
    status_counts = {
        status: sum(item["status"] == status for item in measurements)
        for status in sorted(FUNCTIONAL_MEASUREMENT_STATUSES)
    }
    return {
        "schema_version": FUNCTIONAL_MEASUREMENT_BANK_VERSION,
        "generation_stage": "accepted_checks_before_camera_scheduling",
        "measurement_role": "deterministic_spatial_evidence_not_verdict",
        "decision_authority": "none",
        "scene_access": "read_only",
        "accepted_check_count": len(checks),
        "check_measurement_count": len(measurements),
        "check_measurements": measurements,
        "coverage": {
            "unit": "accepted_functional_check",
            "eligible_count": len(checks),
            "recorded_count": len(measurements),
            "complete_count": status_counts["complete"],
            "partial_count": status_counts["partial"],
            "unavailable_count": status_counts["unavailable"],
            "record_coverage_complete": len(measurements) == len(checks),
        },
        "extension_contract": {
            "field": "measurement_extensions",
            "keyed_by": "check_id_then_namespace",
            "decision_fields_forbidden": sorted(
                _FORBIDDEN_EXTENSION_FIELDS
            ),
        },
    }


def attach_functional_measurement_extension(
    bank: dict[str, Any],
    *,
    namespace: str,
    by_check_id: dict[str, dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    """Attach a deterministic measurement family without changing checks.

    This is the integration hook for independently implemented measurement
    families such as direction-aware clearance.  Unknown check IDs and any
    semantic-decision field are rejected.
    """

    normalized_namespace = str(namespace or "").strip()
    if not normalized_namespace:
        raise ValueError("functional measurement extension requires namespace")
    if not isinstance(by_check_id, dict):
        raise TypeError("functional measurement extension must be keyed by check ID")
    result = deepcopy(bank)
    _validate_bank_shape(result)
    rows = {
        str(item["check_id"]): item
        for item in result["check_measurements"]
    }
    unknown = sorted(set(str(item) for item in by_check_id) - set(rows))
    if unknown:
        raise ValueError(
            "functional measurement extension references unknown checks: "
            f"{unknown}"
        )
    for raw_check_id, payload in by_check_id.items():
        check_id = str(raw_check_id)
        if not isinstance(payload, dict):
            raise TypeError(
                "functional measurement extension payload must be an object; "
                f"check_id={check_id!r}"
            )
        _reject_decision_fields(payload)
        extensions = rows[check_id].setdefault("measurement_extensions", {})
        if normalized_namespace in extensions:
            raise ValueError(
                "functional measurement extension namespace already exists; "
                f"check_id={check_id!r}, namespace={normalized_namespace!r}"
            )
        extensions[normalized_namespace] = {
            "source": str(source or "deterministic"),
            "decision_authority": "none",
            "measurement": deepcopy(payload),
        }
    return result


def compact_functional_measurements_for_checks(
    bank: dict[str, Any] | None,
    check_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Return the bounded Judge-facing subset for exact required checks."""

    if not isinstance(bank, dict):
        return {
            "schema_version": FUNCTIONAL_MEASUREMENT_BANK_VERSION,
            "status": "unavailable",
            "measurement_role": "deterministic_spatial_evidence_not_verdict",
            "decision_authority": "none",
            "check_measurements": [],
        }
    _validate_bank_shape(bank)
    requested = list(
        dict.fromkeys(str(item) for item in check_ids if str(item).strip())
    )
    by_id = {
        str(item["check_id"]): item
        for item in bank["check_measurements"]
    }
    rows = [
        _compact_check_measurement(by_id[check_id])
        for check_id in requested
        if check_id in by_id
    ]
    return {
        "schema_version": FUNCTIONAL_MEASUREMENT_BANK_VERSION,
        "status": (
            "complete"
            if len(rows) == len(requested)
            else "partial"
            if rows
            else "unavailable"
        ),
        "measurement_role": "deterministic_spatial_evidence_not_verdict",
        "measurement_semantics": (
            "Directions, bearings, distances, and boundary facts are "
            "deterministically derived from the canonical scene after a "
            "trusted usable-side hypothesis is bound. They are spatial "
            "evidence, not a validity threshold or metric verdict. Images "
            "remain authoritative for visible asset semantics and context."
        ),
        "decision_authority": "none",
        "requested_check_ids": requested,
        "check_measurements": rows,
    }


def _measure_check(
    *,
    scene: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    directed: dict[str, dict[str, Any]],
    hypotheses: dict[str, dict[str, Any]],
    check: dict[str, Any],
) -> dict[str, Any]:
    check_id = str(check.get("check_id") or "")
    check_type = str(check.get("check_type") or "")
    target_ids = list(
        dict.fromkeys(
            str(item)
            for item in check.get("target_ids") or []
            if str(item).strip()
        )
    )
    check_hypotheses = [
        deepcopy(hypotheses[target_id])
        for target_id in target_ids
        if target_id in hypotheses
    ]
    geometry_by_target = {
        target_id: _validated_object_geometry(objects.get(target_id))
        for target_id in target_ids
    }
    usable_surface_geometry_ids = {
        target_id
        for target_id, geometry in geometry_by_target.items()
        if geometry is not None and geometry[2] is not None
    }
    pseudo_probe = {
        "probe_id": f"measurement::{check_id}",
        "kind": _probe_kind_for_check(check_type),
        "target_ids": target_ids,
        "related_target_ids": [],
        "surface_targets": [
            {
                "target_id": target_id,
                "need_clearance": bool(
                    check.get("need_clearance", False)
                    and target_id in directed
                ),
            }
            for target_id in target_ids
            if target_id in directed
        ],
        "usable_surface_hypotheses": check_hypotheses,
        "logical_boundary_enabled": True,
        "required_observations": list(
            check.get("required_observations") or []
        ),
    }
    geometry = build_functional_geometry_observations(scene, pseudo_probe)
    surfaces_by_target: dict[str, list[dict[str, Any]]] = {}
    for surface in geometry.get("surface_observations") or []:
        if not isinstance(surface, dict):
            continue
        target_id = str(surface.get("target_id") or "")
        # ``functional_geometry`` also serves legacy routing code and therefore
        # retains a permissive zero-vector fallback.  The Measurement Bank may
        # not turn that fallback into canonical evidence: only observations
        # backed by finite center/size/rotation values survive this boundary.
        if target_id not in usable_surface_geometry_ids:
            continue
        surfaces_by_target.setdefault(
            target_id, []
        ).append(_surface_measurement(surface))
    relation_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for relation in geometry.get("relation_observations") or []:
        if not isinstance(relation, dict):
            continue
        endpoint_id = str(relation.get("endpoint_id") or "")
        counterpart_id = str(relation.get("counterpart_id") or "")
        if (
            endpoint_id not in usable_surface_geometry_ids
            or geometry_by_target.get(counterpart_id) is None
        ):
            continue
        relation_by_identity.setdefault(
            (
                endpoint_id,
                counterpart_id,
            ),
            [],
        ).append(relation)

    unavailable: list[dict[str, str]] = []
    target_measurements: list[dict[str, Any]] = []
    for target_id in target_ids:
        record = objects.get(target_id)
        validated_geometry = geometry_by_target.get(target_id)
        if record is None or validated_geometry is None:
            unavailable.append(
                {
                    "field": f"targets.{target_id}.scene_geometry",
                    "reason": (
                        "target_not_available_in_canonical_scene"
                        if record is None
                        else "target_geometry_is_not_finite_positive_xyz"
                    ),
                }
            )
            target_measurements.append(
                {
                    "target_id": target_id,
                    "geometry_status": "unavailable",
                    "directionality": (
                        "directed" if target_id in directed else "non_directed"
                    ),
                    "usable_surfaces": [],
                }
            )
            continue
        center, size, rotation = validated_geometry
        directed_target = target_id in directed
        usable_surfaces = surfaces_by_target.get(target_id, [])
        if directed_target and rotation is None:
            unavailable.append(
                {
                    "field": f"targets.{target_id}.rotation",
                    "reason": "target_rotation_is_not_finite_xyz",
                }
            )
        if directed_target and not usable_surfaces:
            unavailable.append(
                {
                    "field": f"targets.{target_id}.usable_surface",
                    "reason": _surface_unavailable_reason(
                        hypotheses.get(target_id)
                    ),
                }
            )
        target_measurements.append(
            {
                "target_id": target_id,
                "geometry_status": "available",
                "directionality": (
                    "directed" if directed_target else "non_directed"
                ),
                "center_xyz": list(center),
                "size_xyz": list(size),
                "usable_surface_status": (
                    str((hypotheses.get(target_id) or {}).get("status") or "")
                    if directed_target
                    else "not_applicable"
                ),
                "usable_surfaces": usable_surfaces,
            }
        )

    pair_measurements: list[dict[str, Any]] = []
    for endpoint_id in target_ids:
        for counterpart_id in target_ids:
            if endpoint_id == counterpart_id:
                continue
            pair = _pair_measurement(
                endpoint_id=endpoint_id,
                counterpart_id=counterpart_id,
                objects=objects,
                surface_relations=relation_by_identity.get(
                    (endpoint_id, counterpart_id), []
                ),
            )
            pair_measurements.append(pair)
            if pair["geometry_status"] != "available":
                unavailable.append(
                    {
                        "field": (
                            f"pairs.{endpoint_id}->{counterpart_id}.distance"
                        ),
                        "reason": "pair_endpoint_geometry_unavailable",
                    }
                )

    required_missing = _required_unavailable_fields(
        check_type=check_type,
        target_ids=target_ids,
        directed=directed,
        objects=objects,
        surfaces_by_target=surfaces_by_target,
        pair_measurements=pair_measurements,
        geometry_by_target=geometry_by_target,
    )
    unavailable = _stable_unique_dicts([*unavailable, *required_missing])
    any_geometry = bool(
        any(item.get("geometry_status") == "available" for item in target_measurements)
        or any(item.get("geometry_status") == "available" for item in pair_measurements)
    )
    status = (
        "complete"
        if not required_missing
        else "partial"
        if any_geometry
        else "unavailable"
    )
    return {
        "check_id": check_id,
        "check_type": check_type,
        "owner_stage": check.get("owner_stage"),
        "target_ids": target_ids,
        "status": status,
        "target_measurements": target_measurements,
        "pair_measurements": pair_measurements,
        "unavailable_fields": unavailable,
        "measurement_extensions": {},
        "provenance": {
            "source": "canonical_scene_and_precomputed_usable_side",
            "functional_geometry_schema_version": geometry.get(
                "schema_version"
            ),
            "surface_hypothesis_count": len(check_hypotheses),
            "camera_schedule_consulted": False,
            "render_artifact_consulted": False,
        },
        "decision_authority": "none",
    }


def _surface_measurement(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value.get(key))
        for key in (
            "surface_role",
            "side_id",
            "surface_confidence",
            "world_outward_direction",
            "frontage_origin_xy",
            "frontage_support_extent_m",
            "nearest_boundary_distance_m",
            "outward_ray_boundary_distance_m",
            "approach_samples",
        )
        if key in value
    }


def _pair_measurement(
    *,
    endpoint_id: str,
    counterpart_id: str,
    objects: dict[str, dict[str, Any]],
    surface_relations: list[dict[str, Any]],
) -> dict[str, Any]:
    endpoint = objects.get(endpoint_id)
    counterpart = objects.get(counterpart_id)
    if endpoint is None or counterpart is None:
        return {
            "endpoint_id": endpoint_id,
            "counterpart_id": counterpart_id,
            "geometry_status": "unavailable",
            "surface_relations": [],
        }
    start = _vector3(endpoint.get("center"))
    end = _vector3(counterpart.get("center"))
    if start is None or end is None:
        return {
            "endpoint_id": endpoint_id,
            "counterpart_id": counterpart_id,
            "geometry_status": "unavailable",
            "surface_relations": [],
        }
    delta = (end[0] - start[0], end[1] - start[1])
    distance = math.hypot(*delta)
    bearing = (
        [delta[0] / distance, delta[1] / distance, 0.0]
        if distance > 1.0e-12
        else None
    )
    return {
        "endpoint_id": endpoint_id,
        "counterpart_id": counterpart_id,
        "geometry_status": "available",
        "center_distance_m": float(distance),
        "endpoint_to_counterpart_direction": bearing,
        "surface_relations": [
            {
                key: deepcopy(item.get(key))
                for key in (
                    "surface_role",
                    "side_id",
                    "surface_hypothesis_status",
                    "surface_confidence",
                    "endpoint_world_outward_direction",
                    "facing_angle_to_counterpart_degrees",
                )
                if key in item
            }
            for item in surface_relations
        ],
    }


def _required_unavailable_fields(
    *,
    check_type: str,
    target_ids: list[str],
    directed: dict[str, dict[str, Any]],
    objects: dict[str, dict[str, Any]],
    surfaces_by_target: dict[str, list[dict[str, Any]]],
    pair_measurements: list[dict[str, Any]],
    geometry_by_target: dict[
        str,
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float] | None,
        ]
        | None,
    ],
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for target_id in target_ids:
        if target_id not in objects or geometry_by_target.get(target_id) is None:
            missing.append(
                {
                    "field": f"targets.{target_id}.scene_geometry",
                    "reason": "required_target_geometry_unavailable",
                }
            )
    direction_required_ids = (
        [target_id for target_id in target_ids if target_id in directed]
        if check_type in _DIRECTION_REQUIRED_CHECK_TYPES
        or check_type == "clearance"
        else []
    )
    for target_id in direction_required_ids:
        if not surfaces_by_target.get(target_id):
            missing.append(
                {
                    "field": f"targets.{target_id}.usable_surface",
                    "reason": "required_usable_side_not_resolved",
                }
            )
    if len(target_ids) > 1 and not any(
        item.get("geometry_status") == "available"
        for item in pair_measurements
    ):
        missing.append(
            {
                "field": "pair_geometry",
                "reason": "required_pair_geometry_unavailable",
            }
        )
    return _stable_unique_dicts(missing)


def _compact_check_measurement(value: dict[str, Any]) -> dict[str, Any]:
    targets = []
    for target in value.get("target_measurements") or []:
        if not isinstance(target, dict):
            continue
        targets.append(
            {
                key: deepcopy(target.get(key))
                for key in (
                    "target_id",
                    "geometry_status",
                    "directionality",
                    "usable_surface_status",
                    "usable_surfaces",
                )
                if key in target
            }
        )
    return {
        "check_id": str(value.get("check_id") or ""),
        "check_type": str(value.get("check_type") or ""),
        "target_ids": [str(item) for item in value.get("target_ids") or []],
        "status": str(value.get("status") or "unavailable"),
        "target_measurements": targets,
        "pair_measurements": deepcopy(value.get("pair_measurements") or []),
        "unavailable_fields": deepcopy(value.get("unavailable_fields") or []),
        "measurement_extensions": deepcopy(
            value.get("measurement_extensions") or {}
        ),
        "decision_authority": "none",
    }


def _probe_kind_for_check(check_type: str) -> str:
    if check_type in {"directional_correspondence", "relative_use_geometry"}:
        return "functional_correspondence"
    if check_type == "clearance":
        return "approach_clearance"
    return "functional_frontage"


def _surface_unavailable_reason(hypothesis: dict[str, Any] | None) -> str:
    if not isinstance(hypothesis, dict):
        return "usable_surface_hypothesis_not_available"
    status = str(hypothesis.get("status") or "")
    return f"usable_surface_status_{status or 'unknown'}"


def _validated_object_geometry(
    value: dict[str, Any] | None,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float] | None,
] | None:
    if not isinstance(value, dict):
        return None
    center = _vector3(value.get("center"))
    size = _vector3(value.get("size"))
    if center is None or size is None or any(item <= 0.0 for item in size):
        return None
    rotation = _vector3(value.get("rotation") or [0.0, 0.0, 0.0])
    return center, size, rotation


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result: list[float] = []
    for item in value[:3]:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        result.append(number)
    return result[0], result[1], result[2]


def _validate_bank_shape(value: dict[str, Any]) -> None:
    if value.get("schema_version") != FUNCTIONAL_MEASUREMENT_BANK_VERSION:
        raise ValueError("unsupported functional measurement bank schema")
    rows = value.get("check_measurements")
    if not isinstance(rows, list) or any(
        not isinstance(item, dict) or not item.get("check_id")
        for item in rows
    ):
        raise ValueError("functional measurement bank requires check rows")
    check_ids = [str(item["check_id"]) for item in rows]
    if len(check_ids) != len(set(check_ids)):
        raise ValueError("functional measurement bank check IDs must be unique")


def _reject_decision_fields(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = set(value) & _FORBIDDEN_EXTENSION_FIELDS
        if forbidden:
            raise ValueError(
                "functional measurement extension may not contain decision "
                f"fields: {sorted(forbidden)}"
            )
        for child in value.values():
            _reject_decision_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_decision_fields(child)


def _stable_unique_dicts(
    values: list[dict[str, str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        identity = (str(value.get("field") or ""), str(value.get("reason") or ""))
        if identity in seen:
            continue
        seen.add(identity)
        result.append(value)
    return result
