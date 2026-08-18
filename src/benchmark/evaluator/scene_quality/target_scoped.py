"""Independent Judge episodes for target-centred fallback scopes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from benchmark.evaluator.scene_quality.claim_identity import (
    match_final_defects_to_routed_claims,
)
from benchmark.evaluator.scene_quality.functional_checks import (
    canonicalize_typed_invalid_envelope,
)
from benchmark.evaluator.scene_quality.placement_checks import (
    canonicalize_placement_defect_linkage,
    validate_placement_check_results,
)
from benchmark.evaluator.scene_quality.target_scope import (
    TargetCameraScope,
    build_target_camera_scope,
)
from benchmark.evaluator.scene_quality.terminal import (
    scope_was_defaulted,
    terminalize_required_scope,
)
from benchmark.visual_judge.contracts import (
    response_schema_audit_from_exception,
)
from benchmark.visual_judge.evidence_gate import (
    DeterministicEvidenceGate,
)
from benchmark.visual_judge.interfaces import EvidenceGateRequest
from benchmark.visual_judge.orchestration.audit import (
    evidence_artifact_refs,
)
from benchmark.visual_judge.orchestration.budget import (
    extend_acquisition_ledger,
)


_HARD_EVIDENCE_FAILURE_TOKENS = (
    "blank",
    "corrupt",
    "undecodable",
    "missing",
    "integrity",
    "endpoint",
    "http",
    "connection",
    "timeout",
    "authentication",
    "authorization",
    "rate_limit",
    "ratelimit",
    "scene_contract",
    "input_contract",
)


def resolve_target_evidence_packets(
    value: list[str] | dict[str, Any] | None,
    *,
    metric_name: str,
    policy: dict[str, Any],
    scene: dict[str, Any],
    prompt: str | None,
    targets: list[dict[str, Any]],
    camera_evidence_provider: Any,
    resolve_metric_evidence: Callable[..., tuple[list[str], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Acquire one bounded target-local packet per explicit target."""

    packets: list[dict[str, Any]] = []
    for item in targets:
        target_id = str(item.get("target_id") or "").strip()
        explicit_context_ids = [
            str(value)
            for value in item.get("context_ids") or []
            if str(value).strip() and str(value) != target_id
        ]
        try:
            scope = build_target_camera_scope(
                scene,
                target_id=target_id,
                metric=metric_name,
                explicit_context_ids=explicit_context_ids,
                # A target scope is never allowed to become an isolated crop:
                # the global anchor is part of this fallback's evidence
                # contract even when a caller omitted the local-policy flag.
                include_global_context=True,
            )
        except Exception as exc:
            packets.append(
                {
                    "target_id": target_id,
                    "target_scope": None,
                    "paths": [],
                    "resolution": {
                        "scope_satisfied": False,
                        "source": "target_scope_failure",
                        "provider_invoked": False,
                        "provider_status": "failed",
                        "provider_reason": "target_camera_scope_invalid",
                        "missing_paths": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    "routed_candidate_claims": deepcopy(
                        item.get("routed_candidate_claims") or []
                    ),
                    "required_placement_checks": deepcopy(
                        item.get("required_placement_checks") or []
                    ),
                }
            )
            continue
        packet_value = _target_packet_value(
            value,
            metric_name=metric_name,
            target_id=target_id,
        )
        paths, resolution = resolve_metric_evidence(
            packet_value,
            metric_name=metric_name,
            policy=policy,
            scene=scene,
            prompt=prompt,
            selected_object_ids=list(scope.framing_ids),
            selected_group_ids=[],
            selected_groups=[],
            camera_evidence_provider=camera_evidence_provider,
            target_scope=scope,
        )
        packets.append(
            {
                "target_id": target_id,
                "context_ids": list(scope.context_ids),
                "framing_ids": list(scope.framing_ids),
                "target_scope": scope,
                "paths": paths,
                "resolution": resolution,
                "routed_candidate_claims": deepcopy(
                    item.get("routed_candidate_claims") or []
                ),
                "required_placement_checks": deepcopy(
                    item.get("required_placement_checks") or []
                ),
            }
        )
    return packets


def evaluate_target_scoped_judgements(
    *,
    metric_name: str,
    scene: dict[str, Any],
    prompt: str | None,
    packets: list[dict[str, Any]],
    vlm_judge: Any,
    authorized_deviations: list[dict[str, Any]],
    visual_style_spec: dict[str, Any] | None,
    build_judge_request: Callable[..., dict[str, Any]],
    call_judge: Callable[[Any, dict[str, Any]], dict[str, Any]],
    apply_prompt_exemptions: Callable[..., dict[str, Any]],
    normalize_judgement: Callable[..., dict[str, Any]],
    evidence_phase: str = "target_local_confirmation",
    placement_discovery: dict[str, Any] | None = None,
    functional_ownership_ledger: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Judge each target independently with target-only defect ownership."""

    records: list[dict[str, Any]] = []
    for packet in packets:
        target_id = str(packet.get("target_id") or "")
        scope = packet.get("target_scope")
        resolution = packet.get("resolution") or {}
        required_checks = [
            deepcopy(item)
            for item in packet.get("required_placement_checks") or []
            if isinstance(item, dict)
        ]
        record: dict[str, Any] = {
            "scope_kind": "target_centered_context",
            "scope_id": (
                scope.scope_id
                if isinstance(scope, TargetCameraScope)
                else f"target_scope_{target_id}"
            ),
            "target_id": target_id,
            "context_ids": list(packet.get("context_ids") or []),
            "framing_ids": list(packet.get("framing_ids") or [target_id]),
            "target_scope": (
                scope.to_dict()
                if isinstance(scope, TargetCameraScope)
                else None
            ),
            "evidence_paths": list(packet.get("paths") or []),
            "evidence_resolution": deepcopy(resolution),
            "status": "unresolved",
            "terminal_state": "pending",
            "score": None,
            "reason": resolution.get("provider_reason"),
            "vlm_invoked": False,
            "judge_episode_count": 0,
            "judgement": None,
            "routed_candidate_claims": deepcopy(
                packet.get("routed_candidate_claims") or []
            ),
            "required_placement_checks": required_checks,
            "placement_check_resolution": None,
            "claim_correspondence": [],
            "attribution_policy": {
                "default_target_ids": [target_id],
                "context_ids_are_non_owning": True,
            },
        }
        normal_packet_ready = bool(
            isinstance(scope, TargetCameraScope)
            and resolution.get("scope_satisfied")
            and packet.get("paths")
        )
        (
            retained_global_fallback,
            retained_global_integrity,
        ) = _retained_global_fallback_allowed(
            packet,
            metric_name=metric_name,
        )
        if retained_global_integrity is not None:
            resolution["retained_global_integrity_gate"] = deepcopy(
                retained_global_integrity
            )
            if retained_global_integrity.get("ready") is not True:
                resolution["provider_reason"] = (
                    _integrity_failure_reason(retained_global_integrity)
                )
            record["evidence_resolution"] = deepcopy(resolution)
        if (
            not isinstance(scope, TargetCameraScope)
            or (not normal_packet_ready and not retained_global_fallback)
        ):
            record["reason"] = (
                resolution.get("provider_reason")
                or "target_local_render_evidence_unavailable"
            )
            records.append(
                _attach_default_placement_row(
                    terminalize_required_scope(
                        record,
                        phase=f"target_local:{target_id}",
                    )
                )
            )
            continue

        if retained_global_fallback:
            record["retained_global_forced_final"] = True
            record["evidence_coverage"] = {
                "grounded": False,
                "coverage_kind": "retained_global_anchor_only",
                "required_components": [
                    "global_anchor",
                    "target_local",
                ],
                "observed_components": ["global_anchor"],
                "missing_components": ["target_local"],
                "grounding_fraction": 0.5,
            }
            record["evidence_degradation"] = {
                "reason": (
                    resolution.get("provider_reason")
                    or "target_local_acquisition_unavailable"
                ),
                "provider_status": resolution.get("provider_status"),
                "provider_error": resolution.get("provider_error"),
                "forced_final_judge": True,
                "ambiguity_retained": True,
            }
        else:
            record["evidence_coverage"] = {
                "grounded": True,
                "coverage_kind": "global_anchor_and_target_local",
                "required_components": [
                    "global_anchor",
                    "target_local",
                ],
                "observed_components": [
                    "global_anchor",
                    "target_local",
                ],
                "missing_components": [],
                "grounding_fraction": 1.0,
            }

        request = build_judge_request(
            metric_name=metric_name,
            scene=scene,
            prompt=prompt,
            render_evidence=list(packet["paths"]),
            selected_object_ids=list(scope.framing_ids),
            selected_group_ids=[],
            groups=[],
            authorized_deviations=authorized_deviations,
            visual_style_spec=visual_style_spec,
            evidence_phase=evidence_phase,
            decision_mode="final",
            routed_screen_claims=record["routed_candidate_claims"],
            placement_discovery=placement_discovery,
            required_placement_checks=required_checks,
            functional_ownership_ledger=(
                functional_ownership_ledger
                if metric_name == "semantic_placement_consistency"
                else None
            ),
            target_scope=scope,
            attribution_target_ids=[target_id],
            context_object_ids=list(scope.context_ids),
        )
        if retained_global_fallback:
            request["budget_exhaustion_finalization"] = {
                "required": True,
                "trigger_stop_reason": (
                    resolution.get("provider_reason")
                    or "target_local_acquisition_unavailable"
                ),
                "ambiguity_before_forcing": True,
                "previous_missing_observations": [
                    observation
                    for observation in scope.required_observations
                    if observation != "global_context_preserved"
                ],
                "previous_evidence_request": {
                    "target_ids": [target_id],
                    "missing_observations": [
                        observation
                        for observation in scope.required_observations
                        if observation != "global_context_preserved"
                    ],
                    "view_goal": (
                        "A target-local view was requested but acquisition "
                        "failed; make the most educated binary choice from "
                        "the retained global anchor and preserve ambiguity."
                    ),
                    "metadata": {
                        "scope_id": scope.scope_id,
                        "fallback": "retained_global_anchor_only",
                    },
                },
            }
        request["camera_acquisition_ledger"] = extend_acquisition_ledger(
            None,
            artifact_ids=evidence_artifact_refs(packet["paths"]),
        )
        record["vlm_invoked"] = True
        record["judge_episode_count"] = 1
        audit_records = getattr(vlm_judge, "audit_records", None)
        audit_start = len(audit_records) if isinstance(audit_records, list) else None
        try:
            raw = call_judge(vlm_judge, request)
            adjusted = apply_prompt_exemptions(
                raw,
                metric_name=metric_name,
                authorized_deviations=authorized_deviations,
            )
            if retained_global_fallback:
                adjusted = deepcopy(adjusted)
                adjusted.update(
                    evidence_ambiguous=True,
                    forced_binary=True,
                    decision_source=(
                        "forced_final_with_retained_global_anchor"
                    ),
                )
            placement_resolution = None
            if metric_name == "semantic_placement_consistency" and required_checks:
                adjusted = canonicalize_typed_invalid_envelope(adjusted)
                adjusted = canonicalize_placement_defect_linkage(
                    adjusted,
                    required_checks=required_checks,
                )
                if retained_global_fallback:
                    for row in adjusted.get("placement_check_results") or []:
                        if not isinstance(row, dict):
                            continue
                        row["observation_status"] = (
                            "inferred_under_budget"
                        )
                placement_resolution = validate_placement_check_results(
                    adjusted,
                    required_checks=required_checks,
                    function_events=list(
                        (
                            functional_ownership_ledger
                            if isinstance(
                                functional_ownership_ledger,
                                dict,
                            )
                            else {}
                        ).get("events")
                        or []
                    ),
                )
            outcome = normalize_judgement(
                adjusted,
                metric_name=metric_name,
                valid_object_ids={target_id},
            )
            record.update(
                status=outcome["status"],
                score=outcome["score"],
                reason=outcome["reason"],
                judgement=adjusted,
                placement_check_resolution=placement_resolution,
                claim_correspondence=match_final_defects_to_routed_claims(
                    metric_name,
                    adjusted.get("defects") or [],
                    record["routed_candidate_claims"],
                ),
            )
        except Exception as exc:
            schema_audit = response_schema_audit_from_exception(exc)
            record.update(
                status="failed",
                reason="vlm_judge_failed",
                judgement={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **(
                        {"response_schema_audit": schema_audit}
                        if schema_audit is not None
                        else {}
                    ),
                },
            )
        if (
            audit_start is not None
            and isinstance(audit_records, list)
            and len(audit_records) > audit_start
        ):
            record["camera_control_audit"] = deepcopy(audit_records[-1])
        records.append(
            _attach_default_placement_row(
                terminalize_required_scope(
                    record,
                    phase=f"target_local:{target_id}",
                )
            )
        )
    return records


def _retained_global_fallback_allowed(
    packet: dict[str, Any],
    *,
    metric_name: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Allow a final Judge only for a real retained global anchor.

    The local acquisition truth remains ``scope_satisfied=False``.  This
    helper merely distinguishes recoverable camera/acquisition loss from
    missing, corrupt, transport, configuration, or scene-contract failures.
    """

    scope = packet.get("target_scope")
    resolution = packet.get("resolution")
    if not isinstance(scope, TargetCameraScope) or not isinstance(
        resolution, dict
    ):
        return False, None
    if not scope.require_global_anchor:
        return False, None
    if resolution.get("global_anchor_satisfied") is not True:
        return False, None
    if resolution.get("local_scope_satisfied") is not False:
        return False, None
    if resolution.get("provider_invoked") is not True:
        return False, None
    if str(resolution.get("provider_status") or "").lower() not in {
        "failed",
        "insufficient",
    }:
        return False, None
    if resolution.get("missing_paths"):
        return False, None
    if not packet.get("paths"):
        return False, None
    reason_and_error = " ".join(
        str(resolution.get(key) or "").lower()
        for key in ("provider_reason", "provider_error")
    )
    if any(token in reason_and_error for token in _HARD_EVIDENCE_FAILURE_TOKENS):
        return False, None
    gate_result = DeterministicEvidenceGate().check(
        EvidenceGateRequest(
            task="scene_quality",
            metric=metric_name,
            target_ids=(scope.target_id,),
            scene={},
            visual_evidence=tuple(packet.get("paths") or []),
            evidence_goal={
                "scope_id": scope.scope_id,
                "role": "retained_global_anchor",
            },
        )
    )
    audit = {
        "ready": gate_result.ready,
        "reason_codes": list(gate_result.reason_codes),
        "deficiencies": [
            deepcopy(item) for item in gate_result.deficiencies
        ],
        "provenance": deepcopy(gate_result.provenance),
    }
    return gate_result.ready, audit


def _integrity_failure_reason(audit: dict[str, Any]) -> str:
    codes = {
        str(item)
        for item in audit.get("reason_codes") or []
        if str(item)
    }
    if "blank_render" in codes:
        return "blank_evidence"
    if codes & {"undecodable_render", "corrupt_render_evidence"}:
        return "corrupt_evidence"
    if codes & {
        "visual_evidence_missing",
        "evidence_path_missing",
        "evidence_file_missing",
        "empty_render_file",
        "evidence_file_unreadable",
    }:
        return "evidence_missing"
    return "evidence_integrity_failure"


def target_packet_audit(packet: dict[str, Any]) -> dict[str, Any]:
    scope = packet.get("target_scope")
    return {
        "scope_kind": "target_centered_context",
        "target_id": str(packet.get("target_id") or ""),
        "context_ids": list(packet.get("context_ids") or []),
        "target_scope": (
            scope.to_dict() if isinstance(scope, TargetCameraScope) else None
        ),
        "paths": list(packet.get("paths") or []),
        "resolution": deepcopy(packet.get("resolution") or {}),
    }


def _target_packet_value(
    value: list[str] | dict[str, Any] | None,
    *,
    metric_name: str,
    target_id: str,
) -> list[str] | dict[str, Any] | None:
    if not isinstance(value, dict):
        return value
    metric_value = value.get(metric_name)
    if isinstance(metric_value, dict) and target_id in metric_value:
        return {
            "global": deepcopy(
                value.get("global") or value.get("global_context") or []
            ),
            "object_local": deepcopy(metric_value[target_id]),
        }
    return value


def _attach_default_placement_row(record: dict[str, Any]) -> dict[str, Any]:
    if not scope_was_defaulted(record):
        return record
    required = record.get("required_placement_checks") or []
    if not required:
        return record
    judgement = (
        deepcopy(record.get("judgement"))
        if isinstance(record.get("judgement"), dict)
        else {}
    )
    judgement["placement_check_results"] = [
        {
            "check_id": str(check["check_id"]),
            "subject_id": str(check["subject_id"]),
            "context_ids": sorted(
                str(item) for item in check.get("context_ids") or []
            ),
            "observation_status": "inferred_under_budget",
            "conclusion": "valid",
            "reason": (
                "The target-centred episode defaulted valid after a non-hard "
                "failure; ambiguity is retained in audit."
            ),
        }
        for check in required
    ]
    record["judgement"] = judgement
    record["placement_check_resolution"] = validate_placement_check_results(
        judgement,
        required_checks=required,
    )
    return record
