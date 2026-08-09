"""Acquire usable-side direction facts without metric judgement.

The historical module name is retained for compatibility.  The prepass now
serves every directed object: it localizes the usable side and builds a
deterministic world-direction descriptor for camera/check routing.  Logical
boundary measurements remain optional clearance evidence and never become a
metric verdict.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from benchmark.visual_judge.usable_surface import (
    USABLE_SURFACE_STATUSES,
    USABLE_SURFACE_SIDE_IDS,
    validate_usable_surface_response,
)


FUNCTIONAL_BOUNDARY_EVIDENCE_VERSION = (
    "functional_boundary_evidence_v3"
)
_FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "verdict",
        "validity",
        "score",
        "defect",
        "defects",
        "is_invalid",
        "camera_pose",
        "scene_mutation",
    }
)


def acquire_functional_boundary_evidence(
    *,
    provider: Any,
    scene: dict[str, Any],
    discovery: dict[str, Any],
    architecture_context: dict[str, Any],
) -> dict[str, Any]:
    """Decode every directed usable side, then obtain routing geometry."""

    targets = qualifying_direction_surface_targets(discovery)
    audit: dict[str, Any] = {
        "schema_version": FUNCTIONAL_BOUNDARY_EVIDENCE_VERSION,
        "status": "not_applicable",
        "decision_authority": "none",
        "scene_access": "read_only",
        "purpose": (
            "usable_side_localization_and_direction_routing_descriptor"
        ),
        "direction_descriptor_role": "routing_only",
        "logical_boundary_enabled": bool(
            architecture_context.get("logical_boundary_enabled")
        ),
        "requested_surface_targets": list(deepcopy(targets)),
        "requested_target_ids": [
            str(item["target_id"]) for item in targets
        ],
        "usable_surface_hypotheses": [],
        "functional_geometry": {},
        "decoder_audit": {},
        "decoder_calls": 0,
        "cache_hits": 0,
        "preview_render_count": 0,
        "provider_invoked": False,
    }
    if not targets:
        audit.update(
            status="complete_no_targets",
            reason="no_directed_surface_target",
        )
        return audit
    call = getattr(
        provider,
        "provide_functional_boundary_evidence",
        None,
    )
    if not callable(call):
        audit.update(
            status="not_configured",
            reason="functional_boundary_evidence_provider_not_configured",
        )
        return audit
    try:
        audit["provider_invoked"] = True
        raw = call(
            {
                "category": "functional_boundary_evidence_request",
                "metric": "functional_consistency",
                "scene": deepcopy(scene),
                "architecture_context": deepcopy(architecture_context),
                "surface_targets": list(deepcopy(targets)),
                "decision_authority": "none",
                "scene_access": "read_only",
            }
        )
        validated = validate_functional_boundary_evidence_response(
            raw,
            requested_targets=targets,
        )
    except Exception as exc:
        audit.update(
            status="failed",
            reason="functional_boundary_evidence_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return audit
    audit.update(deepcopy(validated))
    decoder = (
        audit.get("decoder_audit")
        if isinstance(audit.get("decoder_audit"), dict)
        else {}
    )
    audit["decoder_calls"] = _nonnegative_int(
        decoder.get("decoder_calls", 0)
    )
    audit["cache_hits"] = _nonnegative_int(
        decoder.get("cache_hits", 0)
    )
    audit["preview_render_count"] = _nonnegative_int(
        decoder.get("preview_render_count", 0)
    )
    return audit


def qualifying_direction_surface_targets(
    discovery: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Select every directed object with a declared usable-surface role."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in discovery.get("directed_surface_targets") or []:
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("target_id") or "").strip()
        directionality = str(
            item.get("directionality") or "directed"
        ).strip()
        need_clearance = item.get("need_clearance")
        if not isinstance(need_clearance, bool):
            raise ValueError(
                "directed surface target need_clearance must be a boolean; "
                f"target_id={target_id!r}"
            )
        roles = list(
            dict.fromkeys(
                str(role).strip()
                for role in item.get("surface_roles") or []
                if str(role).strip()
            )
        )
        if (
            not target_id
            or target_id in seen
            or directionality != "directed"
            or not roles
        ):
            continue
        seen.add(target_id)
        result.append(
            {
                "target_id": target_id,
                "directionality": "directed",
                "surface_roles": roles,
                "need_clearance": need_clearance,
            }
        )
    return tuple(result)


def qualifying_boundary_surface_targets(
    discovery: dict[str, Any],
    *,
    architecture_context: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Compatibility selector for the former clearance-only prepass.

    New orchestration should use :func:`qualifying_direction_surface_targets`.
    This narrower helper remains stable for callers that explicitly ask which
    directed objects also require logical-boundary clearance measurements.
    """

    if not bool(architecture_context.get("logical_boundary_enabled")):
        return ()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in discovery.get("directed_surface_targets") or []:
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("target_id") or "").strip()
        directionality = str(
            item.get("directionality") or "directed"
        ).strip()
        need_clearance = item.get("need_clearance")
        if not isinstance(need_clearance, bool):
            raise ValueError(
                "directed surface target need_clearance must be a boolean; "
                f"target_id={target_id!r}"
            )
        roles = list(
            dict.fromkeys(
                str(role).strip()
                for role in item.get("surface_roles") or []
                if str(role).strip()
            )
        )
        if (
            not target_id
            or target_id in seen
            or directionality != "directed"
            or not need_clearance
            or not roles
        ):
            continue
        seen.add(target_id)
        result.append(
            {
                "target_id": target_id,
                "directionality": directionality,
                "surface_roles": roles,
                "need_clearance": True,
            }
        )
    return tuple(result)


def discovery_with_boundary_hypotheses(
    discovery: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Attach prevalidated hypotheses only to the planner's private copy."""

    result = deepcopy(discovery)
    hypotheses = {
        str(item.get("target_id")): deepcopy(item)
        for item in evidence.get("usable_surface_hypotheses") or []
        if isinstance(item, dict) and item.get("target_id")
    }
    for item in result.get("directed_surface_targets") or []:
        if not isinstance(item, dict):
            continue
        hypothesis = hypotheses.get(str(item.get("target_id") or ""))
        if hypothesis is not None:
            item["precomputed_usable_surface_hypothesis"] = hypothesis
    return result


def boundary_evidence_for_targets(
    evidence: dict[str, Any],
    target_ids: set[str],
) -> dict[str, Any]:
    """Return a compact, group-safe Judge subset without audit payloads."""

    requested = [
        deepcopy(item)
        for item in evidence.get("requested_surface_targets") or []
        if isinstance(item, dict)
        and str(item.get("target_id") or "") in target_ids
    ]
    hypotheses = [
        _compact_surface_hypothesis(item)
        for item in evidence.get("usable_surface_hypotheses") or []
        if isinstance(item, dict)
        and str(item.get("target_id") or "") in target_ids
    ]
    geometry = (
        evidence.get("functional_geometry")
        if isinstance(evidence.get("functional_geometry"), dict)
        else {}
    )
    observations = [
        _compact_boundary_observation(item)
        for item in geometry.get("surface_observations") or []
        if isinstance(item, dict)
        and str(item.get("target_id") or "") in target_ids
    ]
    return {
        "schema_version": FUNCTIONAL_BOUNDARY_EVIDENCE_VERSION,
        "status": str(evidence.get("status") or "not_available"),
        "decision_authority": "none",
        "scene_access": "read_only",
        "measurement_semantics": (
            "VLM-decoded trusted local-side hypotheses plus deterministic "
            "scene geometry. Direction descriptors are routing-only; optional "
            "clearance measurements are evidence, not a universal validity "
            "threshold."
        ),
        "direction_descriptor_role": "routing_only",
        "requested_surface_targets": requested,
        "usable_surface_hypotheses": hypotheses,
        "functional_geometry": {
            "schema_version": geometry.get("schema_version"),
            "decision_authority": "none",
            "scene_access": "read_only",
            "logical_boundary_available": bool(
                geometry.get("logical_boundary_available")
            ),
            "surface_observations": observations,
            "observation_status": (
                "available"
                if observations
                else str(geometry.get("observation_status") or "unavailable")
            ),
        },
    }


def _compact_surface_hypothesis(
    value: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target_id": str(value.get("target_id") or ""),
        "status": str(value.get("status") or ""),
        "surfaces": [
            {
                key: deepcopy(surface.get(key))
                for key in (
                    "surface_role",
                    "side_id",
                    "visual_cues",
                    "confidence",
                )
                if key in surface
            }
            for surface in value.get("surfaces") or []
            if isinstance(surface, dict)
        ],
        "bank_complete": bool(value.get("bank_complete", False)),
        "available_side_ids": [
            str(item)
            for item in value.get("available_side_ids") or []
            if str(item)
        ],
    }


def _compact_boundary_observation(
    value: dict[str, Any],
) -> dict[str, Any]:
    # World vectors and exact object transforms are intentionally retained in
    # the audit artifact but withheld from the Judge-facing compact packet.
    # They route the camera; they must not become a structured shortcut for the
    # architecture-orientation verdict.
    allowed = (
        "target_id",
        "status",
        "surface_role",
        "side_id",
        "descriptor_kind",
        "routing_only",
        "architecture_orientation_applicable",
        "clearance_applicable",
    )
    result = {
        key: deepcopy(value.get(key))
        for key in allowed
        if key in value
    }
    clearance_applicable = value.get("clearance_applicable")
    if clearance_applicable is None:
        clearance_applicable = any(
            key in value
            for key in (
                "nearest_boundary_distance_m",
                "outward_ray_boundary_distance_m",
                "approach_samples",
            )
        )
        result["clearance_applicable"] = bool(clearance_applicable)
    if clearance_applicable is True:
        for key in (
            "nearest_boundary_distance_m",
            "outward_ray_boundary_distance_m",
            "approach_samples",
        ):
            if key in value:
                result[key] = deepcopy(value.get(key))
    return result


def validate_functional_boundary_evidence_response(
    value: Any,
    *,
    requested_targets: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            "functional boundary evidence response must be an object"
        )
    _reject_decision_fields(value)
    allowed = {
        "schema_version",
        "status",
        "decision_authority",
        "scene_access",
        "surface_targets",
        "usable_surface_hypotheses",
        "functional_geometry",
        "decoder_audit",
        "provenance",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "functional boundary evidence contains unsupported fields: "
            f"{sorted(unknown)}"
        )
    if value.get("decision_authority") not in (None, "none"):
        raise ValueError(
            "functional boundary evidence has no decision authority"
        )
    requested_by_id = {
        str(item.get("target_id") or ""): deepcopy(item)
        for item in requested_targets
        if isinstance(item, dict) and item.get("target_id")
    }
    known = set(requested_by_id)
    targets = _unique_target_objects(
        value.get("surface_targets"),
        known=known,
        label="surface_targets",
    )
    if {str(item.get("target_id") or "") for item in targets} != known:
        raise ValueError(
            "functional boundary evidence must echo every requested target"
        )
    for item in targets:
        requested = requested_by_id[str(item["target_id"])]
        if (
            list(item.get("surface_roles") or [])
            != list(requested.get("surface_roles") or [])
            or str(item.get("directionality") or "")
            != str(requested.get("directionality") or "")
            or item.get("need_clearance")
            is not requested.get("need_clearance")
        ):
            raise ValueError(
                "functional boundary evidence may not alter the requested "
                "surface contract"
            )
    hypotheses = _unique_target_objects(
        value.get("usable_surface_hypotheses"),
        known=known,
        label="usable_surface_hypotheses",
        allow_missing=True,
    )
    normalized_hypotheses: list[dict[str, Any]] = []
    for item in hypotheses:
        status = str(item.get("status") or "")
        if status not in USABLE_SURFACE_STATUSES:
            raise ValueError(
                "functional boundary evidence has unsupported surface status"
            )
        target_id = str(item["target_id"])
        provenance = (
            item.get("provenance")
            if isinstance(item.get("provenance"), dict)
            else {}
        )
        available = set(
            str(side_id)
            for side_id in (
                item.get("available_side_ids")
                or provenance.get("trusted_side_ids")
                or USABLE_SURFACE_SIDE_IDS
            )
            if str(side_id)
        )
        bank_complete = bool(
            item.get(
                "bank_complete",
                provenance.get(
                    "bank_complete",
                    available == set(USABLE_SURFACE_SIDE_IDS),
                ),
            )
        )
        validated_surface = validate_usable_surface_response(
            {
                key: deepcopy(item.get(key))
                for key in ("status", "surfaces", "reason")
            },
            allowed_surface_roles=set(
                requested_by_id[target_id].get("surface_roles") or []
            ),
            available_side_ids=available,
            bank_complete=bank_complete,
        )
        normalized_hypotheses.append(
            {
                **deepcopy(item),
                **validated_surface,
                "target_id": target_id,
                "bank_complete": bank_complete,
                "available_side_ids": sorted(available),
                "decision_authority": "none",
            }
        )
    geometry = value.get("functional_geometry")
    if not isinstance(geometry, dict):
        raise ValueError(
            "functional boundary evidence requires functional_geometry"
        )
    observations = geometry.get("surface_observations")
    if not isinstance(observations, list) or any(
        not isinstance(item, dict) for item in observations
    ):
        raise ValueError(
            "functional_geometry.surface_observations must be a list"
        )
    for item in observations:
        target_id = str(item.get("target_id") or "")
        if target_id not in known:
            raise ValueError(
                "functional geometry references an unknown target"
            )
        for key in (
            "nearest_boundary_distance_m",
            "outward_ray_boundary_distance_m",
        ):
            number = item.get(key)
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or float(number) < 0.0
            ):
                raise ValueError(
                    f"functional geometry {key} must be non-negative"
                )
    decoder = value.get("decoder_audit")
    if not isinstance(decoder, dict):
        raise ValueError(
            "functional boundary evidence requires decoder_audit"
        )
    status = str(value.get("status") or "").strip()
    if status not in {"complete", "partial", "failed"}:
        raise ValueError(
            "functional boundary evidence status is unsupported"
        )
    return {
        "schema_version": FUNCTIONAL_BOUNDARY_EVIDENCE_VERSION,
        "status": status,
        "decision_authority": "none",
        "scene_access": "read_only",
        "surface_targets": targets,
        "usable_surface_hypotheses": normalized_hypotheses,
        "functional_geometry": deepcopy(geometry),
        "decoder_audit": deepcopy(decoder),
        "provenance": deepcopy(value.get("provenance") or {}),
    }


def _unique_target_objects(
    value: Any,
    *,
    known: set[str],
    label: str,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    if value is None and allow_missing:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, dict) for item in value
    ):
        raise ValueError(f"{label} must be a list of objects")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        target_id = str(item.get("target_id") or "").strip()
        if not target_id or target_id not in known or target_id in seen:
            raise ValueError(
                f"{label} must reference unique requested target IDs"
            )
        seen.add(target_id)
        result.append(deepcopy(item))
    return result


def _reject_decision_fields(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = set(value) & _FORBIDDEN_DECISION_FIELDS
        if forbidden:
            raise ValueError(
                "functional boundary evidence may not return decision fields"
            )
        for child in value.values():
            _reject_decision_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_decision_fields(child)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value
