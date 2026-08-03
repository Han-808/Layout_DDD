"""JSON-first L3 screening with visual confirmation on demand.

The screening pass is a routing decision, not a final invalid verdict.  It
lets scale and object-pairing checks use canonical object metadata before
paying for camera selection and rendering.  Suspicious or insufficient
screens are confirmed through the existing group-scoped Judge/Controller
path, so ``need_more_evidence`` keeps using the normal camera repair loop.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from benchmark.evaluator.scene_quality.style_global_first import (
    suspicious_groups,
)


def evaluate_json_screen_then_group_visual(
    *,
    base: dict[str, Any],
    metric_name: str,
    metric_config: dict[str, Any],
    scene: dict[str, Any],
    object_ids: list[str],
    groups: list[dict[str, Any]] | None,
    grouping_report: dict[str, Any] | None,
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
    """Screen canonical JSON, then visually confirm only routed groups."""

    screen_groups = groups or []
    screen_group_ids = [
        str(group["group_id"])
        for group in screen_groups
        if group.get("group_id") is not None
    ]
    screen_request = build_judge_request(
        metric_name=metric_name,
        scene=scene,
        prompt=prompt,
        render_evidence=[],
        selected_object_ids=list(object_ids),
        selected_group_ids=screen_group_ids,
        groups=screen_groups,
        authorized_deviations=authorized_deviations,
        visual_style_spec=visual_style_spec,
        evidence_phase="json_screen",
        decision_mode="screen",
    )
    base["evidence_request"]["vlm_invoked"] = True
    base["evidence_request"]["evidence_phase"] = "json_screen"
    base["vlm_invoked"] = True
    base["judge_call_count"] = 1
    try:
        screen_raw = call_judge(vlm_judge, screen_request)
        screen_adjusted = apply_prompt_exemptions(
            screen_raw,
            metric_name=metric_name,
            authorized_deviations=authorized_deviations,
        )
        screen_outcome = normalize_judgement(
            screen_adjusted,
            metric_name=metric_name,
            valid_object_ids=set(object_ids),
        )
    except Exception as exc:
        base.update(
            status="unresolved",
            reason="vlm_json_screen_failed",
            route="json_screen_failed",
            judgement={
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return base

    base["json_screen"] = deepcopy(screen_adjusted)
    if (
        screen_outcome["status"] == "evaluated"
        and screen_outcome["score"] == 1.0
    ):
        eligible_count = int(
            (base.get("coverage") or {}).get("eligible_count") or 1
        )
        base.update(
            status="evaluated",
            reason=None,
            score=1.0,
            route="json_screen_resolved",
            router_state="not_suspicious",
            judgement=screen_adjusted,
        )
        base["dependencies"].update(
            {
                "render_evidence": (
                    "not_required_json_screen_clear"
                ),
                "requested_evidence_scope": "json_screen",
                "evidence_scope_satisfied": True,
                "evidence_source": "structured_scene_json",
                "provider_status": "not_requested",
            }
        )
        base["unavailable_reason"] = None
        base["evidence_request"].update(
            {
                "evidence_phase": "json_screen",
                "provider_invoked": False,
                "provider_status": "not_requested",
                "provider_reason": None,
                "evidence_source": "structured_scene_json",
                "scope_satisfied": True,
                "missing_paths": [],
            }
        )
        base["coverage"] = {
            "eligible_count": eligible_count,
            "resolved_count": eligible_count,
            "fraction": 1.0,
            "complete": True,
        }
        return base

    router_state = (
        "insufficient_evidence"
        if screen_adjusted.get("evidence_status") == "insufficient"
        else "suspicious"
    )
    compatibility_without_grouping = not groups
    candidate_groups = groups or _compatibility_target_groups(
        object_ids,
        judgement=screen_adjusted,
        include_all_when_unlocalized=(
            router_state == "insufficient_evidence"
        ),
    )
    selected_groups = suspicious_groups(
        candidate_groups,
        judgement=screen_adjusted,
        include_all_when_unlocalized=(
            router_state == "insufficient_evidence"
        ),
    )
    if not selected_groups:
        base.update(
            status="unresolved",
            reason=(
                "object_grouping_unavailable_for_visual_confirmation"
                if compatibility_without_grouping
                else "json_suspicion_not_mapped_to_group"
            ),
            route="json_screen_then_visual",
            router_state=router_state,
            judgement=screen_adjusted,
        )
        return base

    plan = metric_config.get("evidence_plan") or {}
    global_plan = (
        plan.get("global_policy")
        if isinstance(plan.get("global_policy"), dict)
        else {}
    )
    local_plan = (
        plan.get("local_policy")
        if isinstance(plan.get("local_policy"), dict)
        else {}
    )
    global_budget = max(0, int(global_plan.get("image_budget") or 1))
    local_budget = max(1, int(local_plan.get("image_budget") or 1))
    max_packet_images = max(
        global_budget + local_budget,
        int(
            local_plan.get("max_packet_images")
            or (
                metric_config.get("evidence_policy")
                if isinstance(
                    metric_config.get("evidence_policy"),
                    dict,
                )
                else {}
            ).get("image_budget")
            or global_budget + local_budget
        ),
    )
    local_scope = (
        str(local_plan.get("camera_scope") or "object_local")
        if compatibility_without_grouping
        else "group_local"
    )
    visual_policy = {
        "camera_scope": local_scope,
        "camera_mode": "metric_local",
        "selector": "deterministic",
        # These are maximums, not required cardinalities.  A smaller packet may
        # still resolve when the Judge reports that the evidence is sufficient.
        "image_budget": max_packet_images,
        "global_image_budget": global_budget,
        "scoped_image_budget": local_budget,
        "presentation": "raw",
        "image_order": ["global_context", local_scope],
        "include_global_context": global_budget > 0,
        "camera_pose_mode": "visibility_ranked",
    }
    visual_input: dict[str, Any]
    if isinstance(render_evidence, dict):
        visual_input = deepcopy(render_evidence)
    else:
        visual_input = {}
    visual_input["global"] = _global_evidence_paths(
        render_evidence
    )[:global_budget]

    packets = resolve_group_evidence_packets(
        visual_input,
        metric_name=metric_name,
        policy=visual_policy,
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
            for member in group.get("object_ids") or []
        )
    )
    packet_scope_satisfied = bool(packets) and all(
        packet["resolution"].get("scope_satisfied") is True
        for packet in packets
    )
    provider_invoked = any(
        packet["resolution"].get("provider_invoked") is True
        for packet in packets
    )
    provider_status = _packet_status(packets)
    provider_reason = next(
        (
            packet["resolution"].get("provider_reason")
            for packet in packets
            if packet["resolution"].get("provider_reason")
        ),
        None,
    )
    missing_paths = list(
        dict.fromkeys(
            str(path)
            for packet in packets
            for path in packet["resolution"].get("missing_paths") or []
        )
    )
    route = (
        "json_screen_then_object_visual"
        if compatibility_without_grouping
        else "json_screen_then_group_visual"
    )
    base["router_state"] = router_state
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
    global_paths = set(visual_input["global"])
    base["local_evidence_paths"] = [
        path
        for path in base["evidence_paths"]
        if path not in global_paths
    ]
    base["resolved_evidence_policy"] = deepcopy(visual_policy)
    base["dependencies"].update(
        {
            "render_evidence": (
                "available"
                if base["evidence_paths"]
                else "unavailable"
            ),
            "requested_evidence_scope": local_scope,
            "evidence_scope_satisfied": packet_scope_satisfied,
            "evidence_source": "per_group_camera_evidence",
            "provider_status": provider_status,
        }
    )
    base["unavailable_reason"] = (
        None
        if packet_scope_satisfied
        else provider_reason
        or "visual_confirmation_evidence_unavailable"
    )
    base["evidence_request"].update(
        {
            "camera_scope": local_scope,
            "image_budget": visual_policy["image_budget"],
            "global_image_budget": global_budget,
            "scoped_image_budget": local_budget,
            "image_order": list(visual_policy["image_order"]),
            "include_global_context": visual_policy[
                "include_global_context"
            ],
            "evidence_phase": "visual_confirmation",
            "provider_invoked": provider_invoked,
            "provider_status": provider_status,
            "provider_reason": provider_reason,
            "evidence_source": "per_group_camera_evidence",
            "scope_satisfied": packet_scope_satisfied,
            "missing_paths": missing_paths,
            "target_object_ids": selected_object_ids,
            "target_group_ids": selected_group_ids,
            "group_requests": [
                group_packet_audit(packet) for packet in packets
            ],
        }
    )
    result = evaluate_group_scoped_judgements(
        base=base,
        metric_name=metric_name,
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
        evidence_phase="visual_confirmation",
        decision_mode="final",
    )
    result["judge_call_count"] = 1 + int(
        result.get("judge_call_count") or 0
    )
    result["vlm_invoked"] = True
    result["evidence_request"]["vlm_invoked"] = True
    result["json_screen"] = deepcopy(screen_adjusted)
    result["router_state"] = router_state
    result["route"] = route
    return result


def _global_evidence_paths(
    value: list[str] | dict[str, Any] | None,
) -> list[str]:
    if isinstance(value, dict):
        candidate = (
            value.get("global")
            or value.get("global_context")
            or value.get("default")
            or value.get("all")
        )
    else:
        candidate = value
    if not isinstance(candidate, (list, tuple)):
        return []
    paths: list[str] = []
    for item in candidate:
        if isinstance(item, (str, Path)) and str(item).strip():
            path = str(item)
        elif isinstance(item, dict):
            raw = item.get("path") or item.get("image_path")
            path = str(raw) if raw is not None else ""
        else:
            path = ""
        if path and path not in paths:
            paths.append(path)
    return paths


def _compatibility_target_groups(
    object_ids: list[str],
    *,
    judgement: dict[str, Any],
    include_all_when_unlocalized: bool,
) -> list[dict[str, Any]]:
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
    if not statuses:
        return "not_requested"
    return next(iter(statuses)) if len(statuses) == 1 else "mixed"
