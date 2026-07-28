from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from benchmark.rendering.camera_pose import (
    CAMERA_ACTION_PARAMETERS,
    CAMERA_ACTION_PROTOCOL_VERSION,
    apply_camera_action,
)


ACTIVE_CORRECTIVE_PROPOSAL_VERSION = "metric_corrective_camera_proposal_v2"
SELECTOR_SAFE_CAMERA_DEFICIENCY_CODES = frozenset(
    {
        "measured_local_visibility_insufficient",
        "required_local_view_count_missing",
        "required_entities_not_jointly_visible",
        "focus_region_out_of_frame",
        "target_occluded_or_too_small",
        "focus_region_too_small",
        "architecture_plane_not_visible",
        "redundant_local_views",
    }
)


def generate_corrective_camera_proposals(
    *,
    metric: str,
    candidates: list[dict[str, Any]],
    deficiency: dict[str, Any] | None,
    history: list[dict[str, Any]] | None = None,
    request: dict[str, Any] | None = None,
    max_proposals: int = 12,
) -> list[dict[str, Any]]:
    """Generate feasible, metric-specific bounded camera repairs.

    The VLM chooses among these proposals; it never supplies a free-form pose.
    Every proposal is preflighted through the same deterministic action
    protocol that will execute it.
    """

    metric_name = _metric_family(metric)
    codes = _camera_deficiency_codes(deficiency)
    if not codes:
        return []
    executed = {
        (
            str(item.get("parent_view_id") or ""),
            str(item.get("action_primitive") or ""),
        )
        for item in history or []
        if isinstance(item, dict)
    }
    proposal_specs = _proposal_specs(metric_name, codes)
    eligible_candidates = [
        candidate
        for candidate in candidates
        if not (
            metric_name == "collision"
            and str(candidate.get("focus_kind") or "") == "pair_context"
        )
    ]
    scene = (
        request.get("scene")
        if isinstance(request, dict)
        and isinstance(request.get("scene"), dict)
        else None
    )
    proposals: list[dict[str, Any]] = []
    # Round-robin by action family.  With a bounded proposal list this gives
    # every candidate one corrective option before early candidates receive
    # their second/third options, avoiding generator-order bias.
    for spec in proposal_specs:
        for candidate in eligible_candidates:
            parent_id = str(candidate.get("id") or "")
            if not parent_id:
                continue
            action = str(spec["action"])
            if (parent_id, action) in executed:
                continue
            try:
                result_pose = apply_camera_action(
                    candidate,
                    action,
                    scene=scene,
                )
            except (TypeError, ValueError):
                continue
            validation = (
                result_pose.get("active_action_validation")
                if isinstance(result_pose.get("active_action_validation"), dict)
                else {}
            )
            proposal_id = f"proposal_{len(proposals):02d}"
            proposal = {
                "schema_version": ACTIVE_CORRECTIVE_PROPOSAL_VERSION,
                "proposal_id": proposal_id,
                "metric": metric_name,
                "parent_view_id": parent_id,
                "family": str(spec["family"]),
                "action_primitive": action,
                "quantized_parameters": deepcopy(
                    CAMERA_ACTION_PARAMETERS[action]
                ),
                "action_protocol": CAMERA_ACTION_PROTOCOL_VERSION,
                "look_at_role": str(
                    candidate.get("focus_kind") or spec["look_at_role"]
                ),
                "target_evidence": str(spec["objective"]),
                "repairs_deficiency_codes": list(codes),
                "room_feasible": bool(
                    validation.get("room_interior_checked")
                    or not candidate.get("room_bounds")
                ),
                "free_space_checked": bool(
                    validation.get("canonical_obb_clearance_checked")
                ),
                "proxy_framing_checked": bool(
                    validation.get("proxy_framing_checked")
                ),
                "result_pose": result_pose,
                "result_pose_fingerprint": pose_fingerprint(result_pose),
            }
            proposal["proposal_fingerprint"] = _canonical_sha256(
                {
                    "version": ACTIVE_CORRECTIVE_PROPOSAL_VERSION,
                    "parent_view_id": parent_id,
                    "family": proposal["family"],
                    "action": action,
                    "result_pose_fingerprint": proposal[
                        "result_pose_fingerprint"
                    ],
                }
            )
            proposals.append(proposal)
            if len(proposals) >= max(1, int(max_proposals)):
                return proposals
    return proposals


def pose_fingerprint(pose: dict[str, Any]) -> str:
    payload = {
        "location": _quantized_vector(pose.get("location")),
        "target": _quantized_vector(pose.get("target")),
        "lens_mm": _quantized_number(pose.get("lens_mm")),
        "camera_type": str(pose.get("camera_type") or "PERSP"),
    }
    return _canonical_sha256(payload)


def selector_safe_proposals(
    proposals: list[dict[str, Any]],
    *,
    internal_to_alias: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Create request-local proposal aliases and retain a private lookup."""

    outbound: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        parent = str(proposal.get("parent_view_id") or "")
        parent_alias = internal_to_alias.get(parent)
        if parent_alias is None:
            continue
        alias = f"proposal_{len(outbound):02d}"
        outbound.append(
            {
                "id": alias,
                "source_candidate_id": parent_alias,
                "family": str(proposal.get("family") or ""),
                "action": str(proposal.get("action_primitive") or ""),
                "target_evidence": str(proposal.get("target_evidence") or ""),
                "repairs": [
                    str(value)
                    for value in proposal.get(
                        "repairs_deficiency_codes", []
                    )
                    if str(value) in SELECTOR_SAFE_CAMERA_DEFICIENCY_CODES
                ],
                "room_feasible": bool(proposal.get("room_feasible")),
            }
        )
        lookup[alias] = proposal
    return outbound, lookup


def _proposal_specs(metric: str, codes: list[str]) -> list[dict[str, str]]:
    focus_first = any(
        code in {"focus_region_too_small", "focus_region_out_of_frame"}
        for code in codes
    )
    if metric == "collision":
        specs = [
            _spec(
                "pair_focus_dolly",
                "dolly_in",
                "collision_focus",
                "increase contact/intersection pixel scale",
            ),
            _spec(
                "pair_side_orbit_left",
                "orbit_left",
                "collision_focus",
                "expose both target surfaces and relative depth",
            ),
            _spec(
                "pair_side_orbit_right",
                "orbit_right",
                "collision_focus",
                "expose both target surfaces and relative depth",
            ),
            _spec(
                "contact_oblique_high",
                "elevate",
                "collision_focus",
                "separate the contact boundary from occluding surfaces",
            ),
            _spec(
                "contact_oblique_low",
                "lower",
                "collision_focus",
                "expose a complementary contact boundary",
            ),
        ]
    elif metric == "oob":
        specs = [
            _spec(
                "boundary_focus_dolly",
                "dolly_in",
                "object_plane_boundary",
                "increase object-plane boundary pixel scale",
            ),
            _spec(
                "plane_tangent_left",
                "orbit_left",
                "object_plane_boundary",
                "separate object silhouette from the violated plane",
            ),
            _spec(
                "plane_tangent_right",
                "orbit_right",
                "object_plane_boundary",
                "separate object silhouette from the violated plane",
            ),
            _spec(
                "interior_normal_retreat",
                "dolly_out",
                "violated_architecture_plane",
                "recover local plane context without leaving the room",
            ),
        ]
    else:
        specs = [
            _spec(
                "base_support_focus_dolly",
                "dolly_in",
                "support_gap",
                "increase base/support gap pixel scale",
            ),
            _spec(
                "low_angle_orbit_left",
                "orbit_left",
                "support_gap",
                "remove occlusion along the base/support interface",
            ),
            _spec(
                "low_angle_orbit_right",
                "orbit_right",
                "support_gap",
                "remove occlusion along the base/support interface",
            ),
            _spec(
                "lower_contact_view",
                "lower",
                "support_gap",
                "make vertical clearance visible against the support surface",
            ),
            _spec(
                "medium_oblique_topology",
                "elevate",
                "support_topology",
                "expose partial support and overhang topology",
            ),
        ]
    if focus_first:
        return specs
    return specs[1:] + specs[:1]


def _spec(
    family: str,
    action: str,
    look_at_role: str,
    objective: str,
) -> dict[str, str]:
    return {
        "family": family,
        "action": action,
        "look_at_role": look_at_role,
        "objective": objective,
    }


def _camera_deficiency_codes(value: dict[str, Any] | None) -> list[str]:
    source = value if isinstance(value, dict) else {}
    deficiencies = source.get("deficiencies")
    codes = [
        str(item.get("code") or "")
        for item in deficiencies
        if isinstance(item, dict)
        and item.get("repairability") == "camera"
        and item.get("code")
    ] if isinstance(deficiencies, list) else []
    if not codes and source.get("camera_repairable") is True:
        codes = [
            str(value)
            for value in source.get("reason_codes", [])
            if value
        ]
    return list(dict.fromkeys(codes))


def _metric_family(value: str) -> str:
    metric = str(value or "").strip().lower()
    if metric == "object_architecture_penetration":
        return "oob"
    if metric not in {"collision", "oob", "support"}:
        raise ValueError(f"unsupported active camera metric {metric!r}")
    return metric


def _quantized_vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return [round(float(component), 6) for component in value]


def _quantized_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(float(value), 6)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
