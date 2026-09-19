"""Auditable structured-evidence fallback for binary evaluator decisions.

The evaluator prefers rendered evidence, but a camera failure must not turn a
metric into a permanently partial result when the canonical scene still
contains usable object geometry.  This module defines the shared terminal
contract used by L1 and L3:

* visual evidence when available;
* otherwise a geometry-only deterministic or VLM decision;
* otherwise an explicit policy-default valid decision.

The last branch is deliberately labelled as policy resolution rather than
empirical evidence.  Consumers can therefore publish a binary result without
misrepresenting how it was obtained.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from benchmark.evaluator.context_projection import (
    project_scene_for_evaluator_context,
)


STRUCTURED_FALLBACK_POLICY = "structured_geometry_then_valid_v1"
GEOMETRY_ONLY_VLM_MODE = "geometry_only_vlm"
DETERMINISTIC_GEOMETRY_MODE = "deterministic_geometry"
POLICY_DEFAULT_VALID_MODE = "policy_default_valid_no_evidence"


def structured_geometry_packet(
    scene: dict[str, Any],
    *,
    metric: str,
    target_ids: list[str] | tuple[str, ...],
    trigger_reason: str,
) -> dict[str, Any] | None:
    """Return a compact, task-intent-free geometry packet when usable.

    Every requested target must have a finite canonical centre and positive
    finite size.  Rotation, boundary, and room height are retained when
    available but are not required for the packet itself.
    """

    if not isinstance(scene, dict):
        return None
    projected = project_scene_for_evaluator_context(scene)
    by_id = {
        str(item.get("id") or ""): item
        for item in projected.get("objects") or []
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    resolved_ids = list(
        dict.fromkeys(str(item) for item in target_ids if str(item).strip())
    )
    if not resolved_ids:
        return None
    objects: list[dict[str, Any]] = []
    for object_id in resolved_ids:
        obj = by_id.get(object_id)
        if not isinstance(obj, dict):
            return None
        center = _finite_vector3(obj.get("center"), positive=False)
        size = _finite_vector3(
            obj.get("size")
            or (
                obj.get("asset_proxy", {}).get("bbox_size")
                if isinstance(obj.get("asset_proxy"), dict)
                else None
            ),
            positive=True,
        )
        if center is None or size is None:
            return None
        rotation = _finite_vector3(obj.get("rotation"), positive=False)
        objects.append(
            {
                "id": object_id,
                "category": (
                    obj.get("category")
                    or obj.get("retrieval_category")
                    or "unknown"
                ),
                "description": obj.get("description") or obj.get("desc"),
                "center": center,
                "size": size,
                "rotation": rotation,
            }
        )
    room_height = projected.get("scene_height")
    room_height = (
        float(room_height)
        if _finite_number(room_height) and float(room_height) > 0.0
        else None
    )
    return {
        "schema_version": "structured_geometry_finalization_v1",
        "policy": STRUCTURED_FALLBACK_POLICY,
        "mode": GEOMETRY_ONLY_VLM_MODE,
        "metric": str(metric),
        "trigger_reason": str(trigger_reason),
        "target_ids": resolved_ids,
        "objects": objects,
        "room": {
            "boundary": deepcopy(projected.get("boundary")),
            "scene_height": room_height,
        },
        "visual_evidence_count": 0,
        "allowed_verdicts": ["valid", "invalid"],
        "fallback_on_judge_failure": POLICY_DEFAULT_VALID_MODE,
        "generator_private_intent_removed": True,
    }


def configure_geometry_only_request(
    request: dict[str, Any],
    packet: dict[str, Any],
) -> None:
    """Mark a Judge request as a terminal geometry-only forced choice."""

    request["structured_geometry_finalization"] = deepcopy(packet)
    request["structured_context_policy"] = {
        "object_fields": [
            "id",
            "category",
            "description",
            "center",
            "size",
            "rotation",
        ],
        "room_fields": ["boundary", "scene_height"],
        "generator_private_intent": "excluded",
        "decision_rule": (
            "Use structured geometry and asset-grounded object semantics for "
            "one terminal valid/invalid choice. Do not request visual evidence."
        ),
    }
    contract = request.get("response_contract")
    if isinstance(contract, dict):
        contract["verdict"] = ["valid", "invalid"]
        evidence_request = contract.get("evidence_request")
        if isinstance(evidence_request, dict):
            evidence_request["required_when_insufficient"] = False


def structured_fallback_record(
    *,
    mode: str,
    trigger_reason: str,
    geometry_packet: dict[str, Any] | None = None,
    policy_resolved: bool = True,
    empirically_grounded: bool | None = None,
) -> dict[str, Any]:
    """Build the stable provenance record shared by metric scopes/checks."""

    if empirically_grounded is None:
        empirically_grounded = mode != POLICY_DEFAULT_VALID_MODE
    return {
        "schema_version": "structured_fallback_resolution_v1",
        "policy": STRUCTURED_FALLBACK_POLICY,
        "mode": str(mode),
        "trigger_reason": str(trigger_reason),
        "policy_resolved": bool(policy_resolved),
        "empirically_grounded": bool(empirically_grounded),
        "geometry_available": geometry_packet is not None,
        "geometry_packet": deepcopy(geometry_packet),
    }


def apply_policy_default_valid(
    record: dict[str, Any],
    *,
    reason: str,
    geometry_packet: dict[str, Any] | None = None,
    recovery_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one terminal scope as explicit zero-confidence valid."""

    fallback = structured_fallback_record(
        mode=POLICY_DEFAULT_VALID_MODE,
        trigger_reason=reason,
        geometry_packet=geometry_packet,
        empirically_grounded=False,
    )
    judgement = {
        "evidence_status": "insufficient",
        "verdict": "valid",
        "confidence": 0.0,
        "reason": (
            "No visual or structured adjudication produced an invalid "
            "finding; the explicit terminal policy returns valid."
        ),
        "missing_evidence": [],
        "defects": [],
        "evidence_request": None,
        "evidence_ambiguous": True,
        "forced_binary": True,
        "defaulted": True,
        "decision_source": POLICY_DEFAULT_VALID_MODE,
        "structured_fallback": deepcopy(fallback),
    }
    if recovery_failure is not None:
        judgement["recovery_failure"] = deepcopy(recovery_failure)
    record.update(
        status="evaluated",
        score=1.0,
        reason=None,
        judgement=judgement,
        structured_fallback=deepcopy(fallback),
        evidence_coverage={
            "grounded": True,
            "empirically_grounded": False,
            "policy_resolved": True,
            "coverage_kind": POLICY_DEFAULT_VALID_MODE,
            "grounding_fraction": 1.0,
        },
    )
    return record


def fallback_resolution(record: Any) -> dict[str, Any] | None:
    """Return normalized structured fallback provenance from a scope."""

    if not isinstance(record, dict):
        return None
    value = record.get("structured_fallback")
    if isinstance(value, dict):
        return value
    judgement = record.get("judgement")
    if isinstance(judgement, dict) and isinstance(
        judgement.get("structured_fallback"), dict
    ):
        return judgement["structured_fallback"]
    return None


def policy_resolved(record: Any) -> bool:
    value = fallback_resolution(record)
    return bool(value and value.get("policy_resolved") is True)


def has_inferred_binary_rows(
    judgement: Any,
    *,
    row_keys: tuple[str, ...],
) -> bool:
    if not isinstance(judgement, dict):
        return False
    return any(
        isinstance(row, dict)
        and row.get("observation_status") == "inferred_under_budget"
        and row.get("conclusion")
        in {"valid", "invalid", "excluded_function_owned"}
        for key in row_keys
        for row in judgement.get(key) or []
    )


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_vector3(
    value: Any,
    *,
    positive: bool,
) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if not all(_finite_number(item) for item in value):
        return None
    result = [float(item) for item in value]
    if positive and any(item <= 0.0 for item in result):
        return None
    return result


__all__ = [
    "DETERMINISTIC_GEOMETRY_MODE",
    "GEOMETRY_ONLY_VLM_MODE",
    "POLICY_DEFAULT_VALID_MODE",
    "STRUCTURED_FALLBACK_POLICY",
    "apply_policy_default_valid",
    "configure_geometry_only_request",
    "fallback_resolution",
    "has_inferred_binary_rows",
    "policy_resolved",
    "structured_fallback_record",
    "structured_geometry_packet",
]
