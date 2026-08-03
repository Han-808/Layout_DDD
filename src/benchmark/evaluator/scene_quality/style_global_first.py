"""Global-first style screening with group-scoped local confirmation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable


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
    resolve_group_evidence_packets: Callable[..., list[dict[str, Any]]],
    resolve_metric_evidence: Callable[..., tuple[list[str], dict[str, Any]]],
    group_packet_audit: Callable[[dict[str, Any]], dict[str, Any]],
    evaluate_group_scoped_judgements: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Run one global screen and only then inspect implicated groups."""

    global_request = build_judge_request(
        metric_name="style_consistency",
        scene=scene,
        prompt=prompt,
        render_evidence=global_evidence,
        selected_object_ids=[],
        selected_group_ids=[],
        groups=groups or [],
        authorized_deviations=authorized_deviations,
        visual_style_spec=visual_style_spec,
        evidence_phase="global_screen",
        decision_mode="screen",
    )
    base["evidence_request"]["vlm_invoked"] = True
    base["evidence_request"]["evidence_phase"] = "global_screen"
    base["vlm_invoked"] = True
    base["judge_call_count"] = 1
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
        base.update(
            status="unresolved",
            reason="vlm_global_style_screen_failed",
            route="global_screen_failed",
            judgement={
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return base

    base["global_screen"] = deepcopy(global_adjusted)
    if (
        global_outcome["status"] == "evaluated"
        and global_outcome["score"] == 1.0
    ):
        base.update(
            status="evaluated",
            reason=None,
            score=1.0,
            route="global_screen_resolved",
            router_state="not_suspicious",
            judgement=global_adjusted,
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
        if global_adjusted.get("evidence_status") == "insufficient"
        else "suspicious"
    )
    compatibility_without_grouping = not groups
    candidate_groups = groups or _compatibility_target_groups(
        object_ids,
        judgement=global_adjusted,
        include_all_when_unlocalized=(
            router_state == "insufficient_evidence"
        ),
    )
    selected_groups = suspicious_groups(
        candidate_groups,
        judgement=global_adjusted,
        include_all_when_unlocalized=router_state == "insufficient_evidence",
    )
    if not selected_groups:
        base.update(
            status="unresolved",
            reason=(
                "object_grouping_unavailable_for_style_drilldown"
                if compatibility_without_grouping
                else "style_suspicion_not_mapped_to_group"
            ),
            route=(
                "global_screen_then_local"
                if compatibility_without_grouping
                else "global_screen_then_group_local"
            ),
            router_state=router_state,
            judgement=global_adjusted,
        )
        return base

    plan = metric_config.get("evidence_plan") or {}
    local_plan = (
        plan.get("local_policy")
        if isinstance(plan.get("local_policy"), dict)
        else {}
    )
    local_scope = (
        str(local_plan.get("camera_scope") or "object_local")
        if compatibility_without_grouping
        else "group_local"
    )
    global_plan = (
        plan.get("global_policy")
        if isinstance(plan.get("global_policy"), dict)
        else {}
    )
    global_image_budget = max(
        0,
        int(global_plan.get("image_budget") or 1),
    )
    selected_global_evidence = list(
        dict.fromkeys(global_evidence)
    )[:global_image_budget]
    local_image_budget = int(local_plan.get("image_budget") or 1)
    max_packet_images = max(
        local_image_budget + len(selected_global_evidence),
        int(
            local_plan.get("max_packet_images")
            or local_image_budget + len(selected_global_evidence)
        ),
    )
    image_order = local_plan.get("image_order")
    if not isinstance(image_order, list) or not image_order:
        image_order = ["global_context", local_scope]
    local_policy = {
        "camera_scope": local_scope,
        "camera_mode": "metric_local",
        "selector": "deterministic",
        # The local budget controls corrective views. Reused global anchors are
        # reserved separately so packet truncation cannot remove required
        # scene context.
        "image_budget": max_packet_images,
        "global_image_budget": len(selected_global_evidence),
        "scoped_image_budget": local_image_budget,
        "presentation": "raw",
        "image_order": list(image_order),
        "include_global_context": True,
        # Style remains global-first. This request-level override applies only
        # to the suspicious/insufficient local-confirmation phase and prevents
        # the metric-wide ``auto`` default (global_only) from suppressing the
        # required group-local render.
        "camera_pose_mode": "visibility_ranked",
    }
    local_evidence_input: dict[str, Any]
    if isinstance(render_evidence, dict):
        local_evidence_input = deepcopy(render_evidence)
    else:
        local_evidence_input = {}
    local_evidence_input["global"] = selected_global_evidence
    packets = resolve_group_evidence_packets(
        local_evidence_input,
        metric_name="style_consistency",
        policy=local_policy,
        scene=scene,
        prompt=prompt,
        groups=selected_groups,
        grouping_report=(
            None
            if compatibility_without_grouping
            else grouping_report
        ),
        camera_evidence_provider=camera_evidence_provider,
        resolve_metric_evidence=resolve_metric_evidence,
    )
    selected_group_ids = [
        str(group["group_id"]) for group in selected_groups
    ]
    selected_object_ids = list(
        dict.fromkeys(
            str(member)
            for group in selected_groups
            for member in group["object_ids"]
        )
    )
    base["router_state"] = router_state
    route = (
        "global_screen_then_local"
        if compatibility_without_grouping
        else "global_screen_then_group_local"
    )
    base["route"] = route
    base["compatibility_scope_fallback"] = (
        "target_local_without_grouping"
        if compatibility_without_grouping
        else None
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
    base["evidence_request"].update(
        {
            "camera_scope": local_scope,
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
    result["judge_call_count"] = 1 + int(
        result.get("judge_call_count") or 0
    )
    result["vlm_invoked"] = True
    result["evidence_request"]["vlm_invoked"] = True
    result["global_screen"] = deepcopy(global_adjusted)
    result["router_state"] = router_state
    result["route"] = route
    result["compatibility_scope_fallback"] = (
        "target_local_without_grouping"
        if compatibility_without_grouping
        else None
    )
    return result


def suspicious_groups(
    groups: list[dict[str, Any]],
    *,
    judgement: dict[str, Any],
    include_all_when_unlocalized: bool,
) -> list[dict[str, Any]]:
    """Map Judge-localized objects/groups onto immutable grouping scopes."""

    target_ids: set[str] = {
        str(target_id)
        for defect in judgement.get("defects") or []
        if isinstance(defect, dict)
        for target_id in defect.get("target_ids") or []
        if str(target_id)
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
                for group_id in metadata.get("suspicious_group_ids") or []
                if str(group_id)
            )
    selected = [
        deepcopy(group)
        for group in groups
        if str(group.get("group_id")) in group_ids
        or bool(
            target_ids.intersection(
                str(member) for member in group.get("object_ids") or []
            )
        )
    ]
    if selected or not include_all_when_unlocalized:
        return selected
    return [deepcopy(group) for group in groups]


def _compatibility_target_groups(
    object_ids: list[str],
    *,
    judgement: dict[str, Any],
    include_all_when_unlocalized: bool,
) -> list[dict[str, Any]]:
    """Keep the pre-grouping public style route usable without merging targets."""

    valid_ids = set(object_ids)
    target_ids = list(
        dict.fromkeys(
            str(target_id)
            for defect in judgement.get("defects") or []
            if isinstance(defect, dict)
            for target_id in defect.get("target_ids") or []
            if str(target_id) in valid_ids
        )
    )
    evidence_request = judgement.get("evidence_request")
    if isinstance(evidence_request, dict):
        target_ids.extend(
            str(target_id)
            for target_id in evidence_request.get("target_ids") or []
            if str(target_id) in valid_ids
            and str(target_id) not in target_ids
        )
    if not target_ids and include_all_when_unlocalized:
        target_ids = list(object_ids)
    return [
        {
            "group_id": f"compat_target_{target_id}",
            "object_ids": [target_id],
        }
        for target_id in target_ids
    ]


def _packet_status(packets: list[dict[str, Any]]) -> str:
    statuses = {
        str(packet["resolution"].get("provider_status") or "unknown")
        for packet in packets
    }
    return next(iter(statuses)) if len(statuses) == 1 else "mixed"
