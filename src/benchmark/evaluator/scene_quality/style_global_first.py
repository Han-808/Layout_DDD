"""Conditional global-first Style screening and local confirmation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from benchmark.evaluator.scene_quality.claim_identity import (
    claim_records,
    object_level_finding_records,
)
from benchmark.visual_judge.orchestration.audit import (
    evidence_artifact_refs,
)
from benchmark.visual_judge.orchestration.budget import (
    extend_acquisition_ledger,
)
from benchmark.visual_judge.contracts import (
    response_schema_audit_from_exception,
)


def evaluate_style_global_then_group_local(
    *,
    base: dict[str, Any],
    metric_config: dict[str, Any],
    scene: dict[str, Any],
    object_ids: list[str],
    groups: list[dict[str, Any]] | None,
    grouping_report: dict[str, Any] | None,
    global_evidence: list[str],
    render_evidence: list[str] | dict[str, Any] | None,
    camera_evidence_provider: Any,
    vlm_judge: Any,
    prompt: str | None,
    visual_style_spec: dict[str, Any] | None,
    authorized_deviations: list[dict[str, Any]],
    build_judge_request: Callable[..., dict[str, Any]],
    call_judge: Callable[[Any, dict[str, Any]], dict[str, Any]],
    apply_prompt_exemptions: Callable[..., dict[str, Any]],
    normalize_judgement: Callable[..., dict[str, Any]],
    resolve_group_evidence_packets: Callable[
        ..., list[dict[str, Any]]
    ],
    resolve_metric_evidence: Callable[
        ..., tuple[list[str], dict[str, Any]]
    ],
    group_packet_audit: Callable[[dict[str, Any]], dict[str, Any]],
    evaluate_group_scoped_judgements: Callable[
        ..., dict[str, Any]
    ],
) -> dict[str, Any]:
    """Use the global Style pass only as a verdict or local-review router."""

    plan = (
        metric_config.get("evidence_plan")
        if isinstance(metric_config.get("evidence_plan"), dict)
        else {}
    )
    selected_global_evidence = _global_packet(
        global_evidence,
        plan=plan,
    )
    base["camera_acquisition_ledger"] = extend_acquisition_ledger(
        None,
        artifact_ids=evidence_artifact_refs(selected_global_evidence),
    )
    global_request = build_judge_request(
        metric_name="style_consistency",
        scene=scene,
        prompt=prompt,
        render_evidence=selected_global_evidence,
        selected_object_ids=[],
        selected_group_ids=[],
        groups=groups or [],
        authorized_deviations=authorized_deviations,
        visual_style_spec=visual_style_spec,
        evidence_phase="global_screen",
        decision_mode="screen",
    )
    global_request["camera_acquisition_ledger"] = deepcopy(
        base["camera_acquisition_ledger"]
    )
    base["evidence_request"]["vlm_invoked"] = True
    base["evidence_request"]["evidence_phase"] = "global_screen"
    base["vlm_invoked"] = True
    base["judge_call_count"] = 1
    audit_records = getattr(vlm_judge, "audit_records", None)
    audit_start = (
        len(audit_records)
        if isinstance(audit_records, list)
        else None
    )
    try:
        global_raw = call_judge(vlm_judge, global_request)
        global_adjusted = apply_prompt_exemptions(
            global_raw,
            metric_name="style_consistency",
            authorized_deviations=authorized_deviations,
        )
        global_outcome = normalize_judgement(
            global_adjusted,
            metric_name="style_consistency",
            valid_object_ids=set(object_ids),
        )
    except Exception as exc:
        schema_audit = response_schema_audit_from_exception(exc)
        base.update(
            status="unresolved",
            reason="vlm_global_style_screen_failed",
            route="global_screen_failed",
            local_review={
                "requested": False,
                "reason": "global_screen_failed",
            },
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
        return base

    global_audit = _latest_audit(
        audit_records,
        start=audit_start,
    )
    if global_audit is not None:
        base["global_camera_control_audit"] = deepcopy(global_audit)
        base["camera_acquisition_ledger"] = (
            _ledger_from_audit(global_audit)
            or deepcopy(base["camera_acquisition_ledger"])
        )
        _add_controller_render_counts(base, global_audit)

    global_screen = {
        **deepcopy(global_adjusted),
        "decision_role": "style_router_or_clear_verdict",
        "final_metric_verdict": bool(
            global_outcome.get("status") == "evaluated"
            and float(global_outcome.get("score")) == 1.0
        ),
    }
    base["global_screen"] = global_screen
    base["global_context_evidence_paths"] = list(
        selected_global_evidence
    )
    base["global_evidence_paths"] = list(selected_global_evidence)

    if (
        global_outcome.get("status") == "evaluated"
        and float(global_outcome.get("score")) == 1.0
    ):
        base.update(
            status="evaluated",
            reason=None,
            score=1.0,
            route="global_clear",
            router_state="clear",
            judgement=deepcopy(global_adjusted),
            selected_group_ids=[],
            selected_object_ids=[],
            local_evidence_paths=[],
            group_results=[],
            final_defect_claims=[],
            final_object_findings=[],
            object_level_attribution=_object_attribution([]),
            local_review={
                "requested": False,
                "reason": "global_style_screen_clear",
                "routed_group_ids": [],
                "routing_fallback": None,
            },
        )
        base["coverage"] = {
            "eligible_count": 1,
            "resolved_count": 1,
            "fraction": 1.0,
            "complete": True,
        }
        return base

    router_state = (
        "insufficient_evidence"
        if (
            global_outcome.get("status") != "evaluated"
            or global_adjusted.get("evidence_status") == "insufficient"
        )
        else "suspicious"
    )
    candidate_defects = [
        deepcopy(item)
        for item in global_adjusted.get("defects") or []
        if isinstance(item, dict)
    ]
    routed_claims = claim_records(
        "style_consistency",
        candidate_defects,
        source_phase="global_screen",
        claim_status="suspicious_candidate",
    )
    base["global_screen_candidate_claims"] = deepcopy(routed_claims)
    base["router_state"] = router_state
    base["route"] = "global_screen_then_group_local"

    if not groups:
        return _unresolved_without_grouping(
            base,
            global_adjusted=global_adjusted,
            router_state=router_state,
        )

    eligible_groups, skipped_groups = _eligible_groups(
        groups,
        valid_object_ids=set(object_ids),
        minimum_members=_minimum_group_members(plan),
    )
    all_trusted_groups, _ = _eligible_groups(
        groups,
        valid_object_ids=set(object_ids),
        minimum_members=1,
    )
    localized = _localized_scope(global_adjusted)
    if localized["localized"]:
        selected_groups = suspicious_groups(
            all_trusted_groups,
            judgement=global_adjusted,
            include_all_when_unlocalized=False,
        )
        routing_fallback = None
    else:
        selected_groups = deepcopy(eligible_groups)
        routing_fallback = "all_eligible_non_singleton_groups"

    if not selected_groups:
        base.update(
            status="unresolved",
            reason=(
                "style_suspicion_not_mapped_to_eligible_group"
                if localized["localized"]
                else "no_eligible_style_groups_for_local_confirmation"
            ),
            score=None,
            selected_group_ids=[],
            selected_object_ids=[],
            group_filter={
                "minimum_group_members": _minimum_group_members(plan),
                "eligible_group_ids": [
                    str(group.get("group_id") or "")
                    for group in eligible_groups
                ],
                "skipped_groups": skipped_groups,
            },
            local_review={
                "requested": True,
                "completed": False,
                "reason": "required_local_confirmation_unavailable",
                "routed_group_ids": [],
                "routing_fallback": routing_fallback,
            },
            judgement={
                "evidence_status": "insufficient",
                "verdict": "ambiguous",
                "confidence": 0.0,
                "reason": (
                    "The global Style screen requires local confirmation, "
                    "but no eligible trusted group can provide it."
                ),
                "missing_evidence": [
                    "style_group_local_confirmation"
                ],
                "defects": [],
                "global_screen_candidate_claims": deepcopy(
                    routed_claims
                ),
            },
        )
        return base

    selected_group_ids = [
        str(group["group_id"]) for group in selected_groups
    ]
    baseline_eligible_group_ids = {
        str(group.get("group_id") or "") for group in eligible_groups
    }
    forced_singleton_group_ids = [
        group_id
        for group_id in selected_group_ids
        if group_id not in baseline_eligible_group_ids
    ]
    selected_object_ids = list(
        dict.fromkeys(
            str(member)
            for group in selected_groups
            for member in group.get("object_ids") or []
        )
    )
    local_policy = _local_policy(
        plan,
        selected_global_count=len(selected_global_evidence),
    )
    local_input = (
        deepcopy(render_evidence)
        if isinstance(render_evidence, dict)
        else {}
    )
    local_input["global"] = list(selected_global_evidence)
    packets = resolve_group_evidence_packets(
        local_input,
        metric_name="style_consistency",
        policy=local_policy,
        scene=scene,
        prompt=prompt,
        groups=selected_groups,
        grouping_report=grouping_report,
        camera_evidence_provider=camera_evidence_provider,
        resolve_metric_evidence=resolve_metric_evidence,
        initial_acquisition_ledger=base.get(
            "camera_acquisition_ledger"
        ),
        max_total_images=_resolved_total_image_budget(vlm_judge),
    )
    for packet in packets:
        member_ids = {
            str(item)
            for item in packet["group"].get("object_ids") or []
        }
        packet["routed_candidate_claims"] = [
            deepcopy(claim)
            for claim in routed_claims
            if member_ids.intersection(claim.get("target_ids") or [])
        ]
    packet_ledgers = [
        packet.get("metric_camera_acquisition_ledger_after")
        for packet in packets
        if isinstance(
            packet.get("metric_camera_acquisition_ledger_after"),
            dict,
        )
    ]
    if packet_ledgers:
        base["camera_acquisition_ledger"] = deepcopy(
            packet_ledgers[-1]
        )

    base["selected_group_ids"] = selected_group_ids
    base["selected_object_ids"] = selected_object_ids
    base["evidence_paths"] = list(
        dict.fromkeys(
            path
            for packet in packets
            for path in packet["paths"]
        )
    )
    base["local_evidence_paths"] = list(
        dict.fromkeys(
            path
            for packet in packets
            for path in packet["paths"]
            if path not in selected_global_evidence
        )
    )
    base["group_filter"] = {
        "minimum_group_members": _minimum_group_members(plan),
        "eligible_group_ids": [
            str(group.get("group_id") or "")
            for group in eligible_groups
        ],
        "explicit_singleton_group_ids": forced_singleton_group_ids,
        "routed_group_ids": selected_group_ids,
        "skipped_groups": skipped_groups,
    }
    base["local_review"] = {
        "requested": True,
        "completed": False,
        "reason": (
            "global_style_screen_suspicious"
            if router_state == "suspicious"
            else "global_style_screen_insufficient"
        ),
        "routed_group_ids": selected_group_ids,
        "routing_fallback": routing_fallback,
        "localized_target_ids": localized["target_ids"],
        "localized_group_ids": localized["group_ids"],
    }
    base["evidence_request"].update(
        {
            "camera_scope": "group_local",
            "image_budget": local_policy["image_budget"],
            "global_image_budget": local_policy[
                "global_image_budget"
            ],
            "scoped_image_budget": local_policy[
                "scoped_image_budget"
            ],
            "image_order": list(local_policy["image_order"]),
            "evidence_phase": "local_confirmation",
            "provider_invoked": any(
                packet["resolution"].get("provider_invoked") is True
                for packet in packets
            ),
            "provider_status": _packet_status(packets),
            "provider_reason": next(
                (
                    packet["resolution"].get("provider_reason")
                    for packet in packets
                    if packet["resolution"].get("provider_reason")
                ),
                None,
            ),
            "target_object_ids": selected_object_ids,
            "target_group_ids": selected_group_ids,
            "group_requests": [
                group_packet_audit(packet) for packet in packets
            ],
        }
    )
    initial_judge_calls = int(base.get("judge_call_count") or 0)
    result = evaluate_group_scoped_judgements(
        base=base,
        metric_name="style_consistency",
        scene=scene,
        prompt=prompt,
        packets=packets,
        vlm_judge=vlm_judge,
        authorized_deviations=authorized_deviations,
        visual_style_spec=visual_style_spec,
        build_judge_request=build_judge_request,
        call_judge=call_judge,
        apply_prompt_exemptions=apply_prompt_exemptions,
        normalize_judgement=normalize_judgement,
        evidence_phase="local_confirmation",
        decision_mode="final",
    )
    result["judge_call_count"] = initial_judge_calls + int(
        result.get("judge_call_count") or 0
    )
    result["vlm_invoked"] = True
    result["evidence_request"]["vlm_invoked"] = True
    result["global_screen"] = global_screen
    result["global_screen_candidate_claims"] = deepcopy(routed_claims)
    result["router_state"] = router_state
    result["route"] = "global_screen_then_group_local"
    result["local_review"] = {
        **deepcopy(base["local_review"]),
        "completed": bool(
            result.get("coverage", {}).get("complete")
        ),
    }
    _attach_style_object_findings(result)
    return result


def suspicious_groups(
    groups: list[dict[str, Any]],
    *,
    judgement: dict[str, Any],
    include_all_when_unlocalized: bool,
) -> list[dict[str, Any]]:
    """Map localized targets onto immutable trusted evidence groups."""

    localized = _localized_scope(judgement)
    selected = [
        deepcopy(group)
        for group in groups
        if str(group.get("group_id") or "") in localized["group_ids"]
        or bool(
            set(
                str(member)
                for member in group.get("object_ids") or []
            ).intersection(localized["target_ids"])
        )
    ]
    if selected or not include_all_when_unlocalized:
        return selected
    return [deepcopy(group) for group in groups]


def _localized_scope(judgement: dict[str, Any]) -> dict[str, Any]:
    target_ids = {
        str(target_id)
        for defect in judgement.get("defects") or []
        if isinstance(defect, dict)
        for target_id in defect.get("target_ids") or []
        if str(target_id) and str(target_id) != "scene"
    }
    group_ids: set[str] = set()
    evidence_request = judgement.get("evidence_request")
    if isinstance(evidence_request, dict):
        target_ids.update(
            str(target_id)
            for target_id in evidence_request.get("target_ids") or []
            if str(target_id) and str(target_id) != "scene"
        )
        metadata = evidence_request.get("metadata")
        if isinstance(metadata, dict):
            group_ids.update(
                str(group_id)
                for group_id in metadata.get(
                    "suspicious_group_ids"
                )
                or []
                if str(group_id)
            )
    return {
        "localized": bool(target_ids or group_ids),
        "target_ids": sorted(target_ids),
        "group_ids": sorted(group_ids),
    }


def _global_packet(
    paths: list[str],
    *,
    plan: dict[str, Any],
) -> list[str]:
    global_plan = (
        plan.get("global_policy")
        if isinstance(plan.get("global_policy"), dict)
        else {}
    )
    budget = max(1, int(global_plan.get("image_budget") or 1))
    return list(dict.fromkeys(paths))[:budget]


def _local_policy(
    plan: dict[str, Any],
    *,
    selected_global_count: int,
) -> dict[str, Any]:
    local_plan = (
        plan.get("local_policy")
        if isinstance(plan.get("local_policy"), dict)
        else {}
    )
    local_budget = max(1, int(local_plan.get("image_budget") or 1))
    global_budget = min(
        selected_global_count,
        max(
            1,
            int(
                local_plan.get("global_context_image_budget")
                or 1
            ),
        ),
    )
    max_packet = max(
        global_budget + local_budget,
        int(
            local_plan.get("max_packet_images")
            or global_budget + local_budget
        ),
    )
    image_order = local_plan.get("image_order")
    if not isinstance(image_order, list) or not image_order:
        image_order = ["global_context", "group_local"]
    return {
        "camera_scope": "group_local",
        "camera_mode": "metric_local",
        "selector": "deterministic",
        "image_budget": max_packet,
        "global_image_budget": global_budget,
        "scoped_image_budget": local_budget,
        "presentation": "raw",
        "image_order": list(image_order),
        "include_global_context": global_budget > 0,
        "camera_pose_mode": "visibility_ranked",
    }


def _minimum_group_members(plan: dict[str, Any]) -> int:
    local_plan = (
        plan.get("local_policy")
        if isinstance(plan.get("local_policy"), dict)
        else {}
    )
    return max(2, int(local_plan.get("minimum_group_members") or 2))


def _eligible_groups(
    groups: list[dict[str, Any]],
    *,
    valid_object_ids: set[str],
    minimum_members: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for group in groups:
        members = list(
            dict.fromkeys(
                str(member)
                for member in group.get("object_ids") or []
                if str(member) in valid_object_ids
            )
        )
        normalized = {**deepcopy(group), "object_ids": members}
        if len(members) >= minimum_members:
            eligible.append(normalized)
        else:
            skipped.append(
                {
                    "group_id": str(group.get("group_id") or ""),
                    "member_ids": members,
                    "member_count": len(members),
                    "reason": (
                        "singleton_group"
                        if len(members) == 1
                        else "empty_group"
                    ),
                }
            )
    return eligible, skipped


def _unresolved_without_grouping(
    base: dict[str, Any],
    *,
    global_adjusted: dict[str, Any],
    router_state: str,
) -> dict[str, Any]:
    base.update(
        status="unresolved",
        reason="object_grouping_unavailable_for_style_confirmation",
        score=None,
        selected_group_ids=[],
        selected_object_ids=[],
        local_review={
            "requested": True,
            "completed": False,
            "reason": "trusted_grouping_unavailable",
            "routed_group_ids": [],
            "routing_fallback": None,
        },
        judgement={
            "evidence_status": "insufficient",
            "verdict": "ambiguous",
            "confidence": 0.0,
            "reason": (
                "The global Style screen requires local confirmation, but "
                "trusted object grouping is unavailable."
            ),
            "missing_evidence": ["style_group_local_confirmation"],
            "defects": [],
            "router_state": router_state,
            "global_screen": deepcopy(global_adjusted),
        },
    )
    return base


def _latest_audit(
    records: Any,
    *,
    start: int | None,
) -> dict[str, Any] | None:
    if (
        start is None
        or not isinstance(records, list)
        or len(records) <= start
    ):
        return None
    return deepcopy(records[-1])


def _ledger_from_audit(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    audit = (
        record.get("audit")
        if isinstance(record.get("audit"), dict)
        else record
    )
    camera = (
        audit.get("camera_acquisition")
        if isinstance(audit, dict)
        and isinstance(audit.get("camera_acquisition"), dict)
        else {}
    )
    ledger = camera.get("ledger")
    return deepcopy(ledger) if isinstance(ledger, dict) else None


def _add_controller_render_counts(
    base: dict[str, Any],
    record: dict[str, Any],
) -> None:
    audit = (
        record.get("audit")
        if isinstance(record.get("audit"), dict)
        else record
    )
    telemetry = (
        audit.get("experiment_telemetry")
        if isinstance(audit, dict)
        and isinstance(audit.get("experiment_telemetry"), dict)
        else {}
    )
    preview = int(telemetry.get("preview_render_count") or 0)
    full = int(telemetry.get("full_render_count") or 0)
    base["preview_render_count"] = int(
        base.get("preview_render_count") or 0
    ) + preview
    base["final_render_count"] = int(
        base.get("final_render_count") or 0
    ) + full
    base["preview_renderer_invoked"] = bool(
        base.get("preview_renderer_invoked") or preview
    )
    base["renderer_invoked"] = bool(
        base.get("renderer_invoked") or full
    )


def _resolved_total_image_budget(judge: Any) -> int | None:
    control = getattr(judge, "control", None)
    value = getattr(control, "max_total_images", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return None


def _attach_style_object_findings(
    result: dict[str, Any],
) -> None:
    observations = [
        ("group_local_review", defect)
        for group_result in result.get("group_results") or []
        if isinstance(group_result, dict)
        and group_result.get("status") == "evaluated"
        and float(group_result.get("score") or 0.0) == 0.0
        for defect in (
            (group_result.get("judgement") or {}).get("defects")
            or []
        )
        if isinstance(defect, dict)
    ]
    findings = object_level_finding_records(
        "style_consistency",
        observations,
    )
    raw_object_observation_count = sum(
        int(finding.get("observation_count") or 0)
        for finding in findings
    )
    result["final_object_findings"] = findings
    result["object_level_attribution"] = _object_attribution(
        findings,
        raw_count=raw_object_observation_count,
    )
    judgement = result.get("judgement")
    if isinstance(judgement, dict):
        judgement["object_findings"] = deepcopy(findings)
        judgement["object_penalty_count"] = len(findings)
        judgement["object_penalty_policy"] = (
            "one_per_metric_object_after_required_local_confirmation"
        )


def _object_attribution(
    findings: list[dict[str, Any]],
    *,
    raw_count: int = 0,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "unit": "object",
        "deduplication_key": ["metric", "object_id"],
        "cross_phase_deduplication": True,
        "cross_metric_deduplication": False,
        "raw_defect_observation_count": raw_count,
        "unique_object_count": len(findings),
        "merged_duplicate_observation_count": max(
            0,
            raw_count - len(findings),
        ),
        "penalty_unit_count": len(findings),
    }


def _packet_status(packets: list[dict[str, Any]]) -> str:
    statuses = {
        str(packet["resolution"].get("provider_status") or "unknown")
        for packet in packets
    }
    if not statuses:
        return "not_requested"
    return next(iter(statuses)) if len(statuses) == 1 else "mixed"
