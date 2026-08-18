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

from benchmark.evaluator.scene_quality.claim_identity import (
    claim_records,
    deduplicate_defects,
)
from benchmark.evaluator.scene_quality.style_global_first import (
    suspicious_groups,
)
from benchmark.evaluator.scene_quality.terminal import (
    infrastructure_failure_from_scope,
    scope_was_defaulted,
    terminalize_required_scope,
)
from benchmark.evaluator.scene_quality.target_scope import (
    localized_target_ids,
)
from benchmark.evaluator.scene_quality.target_scoped import (
    evaluate_target_scoped_judgements,
    resolve_target_evidence_packets,
    target_packet_audit,
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
    audit_records = getattr(vlm_judge, "audit_records", None)
    audit_start = (
        len(audit_records)
        if isinstance(audit_records, list)
        else None
    )
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
        _attach_json_screen_audit(
            base,
            audit_records=audit_records,
            audit_start=audit_start,
        )
        return terminalize_required_scope(
            base,
            phase=f"{metric_name}.json_screen",
        )

    _attach_json_screen_audit(
        base,
        audit_records=audit_records,
        audit_start=audit_start,
    )
    screen_record = _json_screen_record(
        screen_adjusted,
        metric_name=metric_name,
    )
    base["json_screen"] = screen_record
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
            judgement=screen_record,
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
        return terminalize_required_scope(
            base,
            phase=f"{metric_name}.json_screen",
        )

    router_state = (
        "insufficient_evidence"
        if screen_adjusted.get("evidence_status") == "insufficient"
        else "suspicious"
    )
    routed_candidate_claims = claim_records(
        metric_name,
        screen_adjusted.get("defects") or [],
        source_phase="json_screen",
        claim_status="suspicious_candidate",
    )
    base["routed_candidate_claims"] = deepcopy(
        routed_candidate_claims
    )
    compatibility_without_grouping = bool(
        not groups and metric_name == "scale_consistency"
    )
    candidate_groups = (
        groups
        or (
            _compatibility_target_groups(
                object_ids,
                judgement=screen_adjusted,
                include_all_when_unlocalized=(
                    router_state == "insufficient_evidence"
                ),
            )
            if compatibility_without_grouping
            else []
        )
    )
    selected_groups = suspicious_groups(
        candidate_groups,
        judgement=screen_adjusted,
        include_all_when_unlocalized=(
            router_state == "insufficient_evidence"
        ),
    )
    localized_ids = localized_target_ids(
        screen_adjusted,
        valid_object_ids=object_ids,
    )
    grouped_selected_ids = {
        str(member)
        for group in selected_groups
        for member in group.get("object_ids") or []
    }
    target_fallback_ids = (
        [
            target_id
            for target_id in localized_ids
            if target_id not in grouped_selected_ids
        ]
        if metric_name == "object_pairing_consistency"
        else []
    )
    if not selected_groups and not target_fallback_ids:
        base.update(
            status="unresolved",
            reason=(
                "object_grouping_unavailable_for_visual_confirmation"
                if compatibility_without_grouping
                else "json_suspicion_not_localized_for_target_confirmation"
            ),
            route="json_screen_then_visual",
            router_state=router_state,
            judgement=screen_record,
        )
        return terminalize_required_scope(
            base,
            phase=f"{metric_name}.visual_confirmation_routing",
        )

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
    group_local_scope = (
        str(local_plan.get("camera_scope") or "object_local")
        if compatibility_without_grouping
        else "group_local"
    )
    group_policy = {
        "camera_scope": group_local_scope,
        "camera_mode": "metric_local",
        "selector": "deterministic",
        # These are maximums, not required cardinalities.  A smaller packet may
        # still resolve when the Judge reports that the evidence is sufficient.
        "image_budget": max_packet_images,
        "global_image_budget": global_budget,
        "scoped_image_budget": local_budget,
        "presentation": "raw",
        "image_order": ["global_context", group_local_scope],
        "include_global_context": global_budget > 0,
        "camera_pose_mode": "visibility_ranked",
    }
    target_policy = {
        **deepcopy(group_policy),
        "camera_scope": "object_local",
        "image_order": ["global_context", "object_local"],
    }
    visual_input: dict[str, Any]
    if isinstance(render_evidence, dict):
        visual_input = deepcopy(render_evidence)
    else:
        visual_input = {}
    visual_input["global"] = _global_evidence_paths(
        render_evidence
    )[:global_budget]

    packets = (
        resolve_group_evidence_packets(
            visual_input,
            metric_name=metric_name,
            policy=group_policy,
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
        if selected_groups
        else []
    )
    for packet in packets:
        member_ids = {
            str(item)
            for item in packet["group"].get("object_ids") or []
        }
        packet["routed_candidate_claims"] = [
            deepcopy(claim)
            for claim in routed_candidate_claims
            if member_ids.intersection(
                str(item)
                for item in claim.get("target_ids") or []
            )
        ]
    target_specs: list[dict[str, Any]] = []
    for target_id in target_fallback_ids:
        matching_claims = [
            deepcopy(claim)
            for claim in routed_candidate_claims
            if target_id
            in {str(item) for item in claim.get("target_ids") or []}
        ]
        explicit_context_ids = list(
            dict.fromkeys(
                str(item)
                for claim in matching_claims
                for item in claim.get("target_ids") or []
                if str(item) != target_id and str(item) in set(object_ids)
            )
        )
        target_specs.append(
            {
                "target_id": target_id,
                "context_ids": explicit_context_ids,
                "routed_candidate_claims": matching_claims,
            }
        )
    target_packets = resolve_target_evidence_packets(
        visual_input,
        metric_name=metric_name,
        policy=target_policy,
        scene=scene,
        prompt=prompt,
        targets=target_specs,
        camera_evidence_provider=camera_evidence_provider,
        resolve_metric_evidence=resolve_metric_evidence,
    )
    selected_group_ids = [
        str(group["group_id"]) for group in selected_groups
    ]
    selected_object_ids = list(
        dict.fromkeys(
            [
                *[
                    str(member)
                    for group in selected_groups
                    for member in group.get("object_ids") or []
                ],
                *target_fallback_ids,
            ]
        )
    )
    all_packets = [*packets, *target_packets]
    packet_scope_satisfied = bool(all_packets) and all(
        packet["resolution"].get("scope_satisfied") is True
        for packet in all_packets
    )
    provider_invoked = any(
        packet["resolution"].get("provider_invoked") is True
        for packet in all_packets
    )
    provider_status = _packet_status(all_packets)
    provider_reason = next(
        (
            packet["resolution"].get("provider_reason")
            for packet in all_packets
            if packet["resolution"].get("provider_reason")
        ),
        None,
    )
    missing_paths = list(
        dict.fromkeys(
            str(path)
            for packet in all_packets
            for path in packet["resolution"].get("missing_paths") or []
        )
    )
    route = (
        "json_screen_then_object_visual"
        if compatibility_without_grouping
        else "json_screen_then_group_and_target_visual"
        if packets and target_packets
        else "json_screen_then_target_visual"
        if target_packets
        else "json_screen_then_group_visual"
    )
    base["router_state"] = router_state
    base["route"] = route
    base["compatibility_scope_fallback"] = (
        "target_local_without_grouping"
        if compatibility_without_grouping
        else "target_centered_context_without_group_redefinition"
        if target_packets
        else None
    )
    base["selected_group_ids"] = selected_group_ids
    base["selected_object_ids"] = selected_object_ids
    base["evidence_paths"] = list(
        dict.fromkeys(
            path
            for packet in all_packets
            for path in packet["paths"]
        )
    )
    global_paths = set(visual_input["global"])
    base["local_evidence_paths"] = [
        path
        for path in base["evidence_paths"]
        if path not in global_paths
    ]
    base["resolved_evidence_policy"] = {
        "group_policy": deepcopy(group_policy),
        "target_policy": deepcopy(target_policy),
    }
    base["dependencies"].update(
        {
            "render_evidence": (
                "available"
                if base["evidence_paths"]
                else "unavailable"
            ),
            "requested_evidence_scope": (
                "mixed_group_and_target_local"
                if packets and target_packets
                else "target_local"
                if target_packets
                else group_local_scope
            ),
            "evidence_scope_satisfied": packet_scope_satisfied,
            "evidence_source": (
                "group_and_target_camera_evidence"
                if packets and target_packets
                else "target_centered_camera_evidence"
                if target_packets
                else "per_group_camera_evidence"
            ),
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
            "camera_scope": (
                "mixed_group_and_target_local"
                if packets and target_packets
                else "object_local"
                if target_packets
                else group_local_scope
            ),
            "image_budget": max_packet_images,
            "global_image_budget": global_budget,
            "scoped_image_budget": local_budget,
            "image_order": ["global_context", "scoped_local"],
            "include_global_context": global_budget > 0,
            "evidence_phase": "visual_confirmation",
            "provider_invoked": provider_invoked,
            "provider_status": provider_status,
            "provider_reason": provider_reason,
            "evidence_source": (
                "group_and_target_camera_evidence"
                if packets and target_packets
                else "target_centered_camera_evidence"
                if target_packets
                else "per_group_camera_evidence"
            ),
            "scope_satisfied": packet_scope_satisfied,
            "missing_paths": missing_paths,
            "target_object_ids": selected_object_ids,
            "target_group_ids": selected_group_ids,
            "group_requests": [
                group_packet_audit(packet) for packet in packets
            ],
            "target_requests": [
                target_packet_audit(packet) for packet in target_packets
            ],
        }
    )
    grouped_result = (
        evaluate_group_scoped_judgements(
            base=deepcopy(base),
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
        if packets
        else deepcopy(base)
    )
    target_results = evaluate_target_scoped_judgements(
        metric_name=metric_name,
        scene=scene,
        prompt=prompt,
        packets=target_packets,
        vlm_judge=vlm_judge,
        authorized_deviations=authorized_deviations,
        visual_style_spec=visual_style_spec,
        build_judge_request=build_judge_request,
        call_judge=call_judge,
        apply_prompt_exemptions=apply_prompt_exemptions,
        normalize_judgement=normalize_judgement,
        evidence_phase="target_local_confirmation",
    )
    result = _merge_json_visual_scopes(
        base=base,
        grouped_result=grouped_result,
        target_results=target_results,
        metric_name=metric_name,
    )
    result["judge_call_count"] = 1 + int(result["judge_call_count"])
    result["vlm_invoked"] = True
    result["evidence_request"]["vlm_invoked"] = True
    result["json_screen"] = screen_record
    result["routed_candidate_claims"] = deepcopy(
        routed_candidate_claims
    )
    result["router_state"] = router_state
    result["route"] = route
    return terminalize_required_scope(
        result,
        phase=f"{metric_name}.visual_confirmation",
    )


def _merge_json_visual_scopes(
    *,
    base: dict[str, Any],
    grouped_result: dict[str, Any],
    target_results: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    """Aggregate real group scopes and non-group target scopes together."""

    result = deepcopy(grouped_result if grouped_result else base)
    group_results = [
        deepcopy(item)
        for item in result.get("group_results") or []
        if isinstance(item, dict)
    ]
    result["group_results"] = group_results
    result["target_scope_results"] = deepcopy(target_results)
    result["target_scope_policy"] = {
        "scope_kind": "target_centered_context",
        "creates_group": False,
        "redefines_group_membership": False,
        "context_objects_are_defect_owners": False,
        "judge_episode": "independent_per_target",
    }
    scopes = [*group_results, *target_results]
    evaluated = [item for item in scopes if item.get("status") == "evaluated"]
    grounded = [
        item
        for item in evaluated
        if not scope_was_defaulted(item)
        and not (
            isinstance(item.get("evidence_coverage"), dict)
            and item["evidence_coverage"].get("grounded") is False
        )
    ]
    invalid = [item for item in evaluated if item.get("score") == 0.0]
    failures: list[dict[str, Any]] = []
    for item in group_results:
        failure = infrastructure_failure_from_scope(
            item,
            phase="group_local",
            scope_id=str(item.get("group_id") or "") or None,
        )
        if failure is not None:
            failures.append(failure)
    for item in target_results:
        failure = infrastructure_failure_from_scope(
            item,
            phase="target_local",
            scope_id=str(item.get("target_id") or "") or None,
        )
        if failure is not None:
            failures.append(failure)

    result["judge_call_count"] = sum(
        int(item.get("judge_episode_count") or 0) for item in scopes
    )
    result["vlm_invoked"] = bool(result["judge_call_count"])
    result.setdefault("evidence_request", {})["vlm_invoked"] = result[
        "vlm_invoked"
    ]
    result["coverage"] = {
        "eligible_count": len(scopes),
        "resolved_count": len(grounded),
        "terminal_resolved_count": len(evaluated),
        "fraction": len(grounded) / len(scopes) if scopes else None,
        "complete": bool(scopes) and len(grounded) == len(scopes),
        "group_scope_count": len(group_results),
        "target_scope_count": len(target_results),
        "score_grounding": {
            "unit": "frozen_evaluation_obligation",
            "eligible_count": len(scopes),
            "grounded_count": len(grounded),
            "defaulted_count": len(scopes) - len(grounded),
            "fraction": (
                len(grounded) / len(scopes) if scopes else 0.0
            ),
            "complete": bool(scopes) and len(grounded) == len(scopes),
            "defaulted_units": [
                {
                    "unit_id": (
                        "target_local:"
                        + str(item.get("target_id"))
                        if item.get("scope_kind")
                        == "target_centered_context"
                        else "group_local:"
                        + str(item.get("group_id"))
                    ),
                    "unit_type": "judge_episode",
                    "grounded": False,
                    "defaulted": scope_was_defaulted(item),
                }
                for item in evaluated
                if item not in grounded
            ],
        },
    }
    if failures:
        result["infrastructure_failures"] = deepcopy(failures)
        result.update(
            status="failed",
            terminal_state="infrastructure_failure",
            reason="required_scope_infrastructure_failure",
            score=None,
            judgement={
                "evidence_status": "unavailable",
                "verdict": None,
                "confidence": 0.0,
                "reason": (
                    "One or more required visual confirmation scopes failed "
                    "for an engineering reason."
                ),
                "missing_evidence": [
                    f"{item.get('phase')}:{item.get('scope_id')}"
                    for item in failures
                ],
                "defects": [],
                "aggregation": "fail_closed_on_required_scope_failure",
                "group_judgements": deepcopy(group_results),
                "target_scope_judgements": deepcopy(target_results),
            },
        )
        return result

    defects = deduplicate_defects(
        metric_name,
        (
            defect
            for item in invalid
            for defect in (item.get("judgement") or {}).get("defects") or []
        ),
    )
    result["final_defect_claims"] = claim_records(
        metric_name,
        defects,
        source_phase="visual_confirmation",
        claim_status="final",
    )
    degraded = any(
        item.get("terminal_state") == "evaluated_degraded"
        for item in evaluated
    )
    if invalid:
        confidence = min(
            float((item.get("judgement") or {}).get("confidence") or 0.0)
            for item in invalid
        )
        verdict = "invalid"
        score = 0.0
        reason = "At least one required visual confirmation scope is invalid."
    else:
        confidence = min(
            (
                float((item.get("judgement") or {}).get("confidence") or 0.0)
                for item in evaluated
            ),
            default=0.0,
        )
        verdict = "valid"
        score = 1.0
        reason = "All required visual confirmation scopes resolved valid."
    result.update(
        status="evaluated",
        terminal_state=("evaluated_degraded" if degraded else "evaluated"),
        reason=None,
        score=score,
        judgement={
            "evidence_status": "sufficient",
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "missing_evidence": [],
            "defects": defects,
            "aggregation": "invalid_if_any_visual_scope_invalid",
            "group_judgements": deepcopy(group_results),
            "target_scope_judgements": deepcopy(target_results),
        },
    )
    return result


def _attach_json_screen_audit(
    report: dict[str, Any],
    *,
    audit_records: Any,
    audit_start: int | None,
) -> None:
    """Persist the already-produced audit without affecting routing."""

    if (
        audit_start is not None
        and isinstance(audit_records, list)
        and len(audit_records) > audit_start
        and isinstance(audit_records[-1], dict)
    ):
        report["json_screen_camera_control_audit"] = deepcopy(
            audit_records[-1]
        )


def _json_screen_record(
    response: dict[str, Any],
    *,
    metric_name: str,
) -> dict[str, Any]:
    """Persist routing output without presenting a candidate as final invalid."""

    record = deepcopy(response)
    verdict = str(record.get("verdict") or "")
    if verdict == "invalid":
        candidate_defects = deepcopy(record.get("defects") or [])
        record.update(
            decision_role="router_only",
            final_metric_verdict=False,
            screen_status="suspicious_candidate",
            screen_state="material_candidate",
            model_response_verdict="invalid",
            verdict="candidate",
            candidate_defects=candidate_defects,
            candidate_claims=claim_records(
                metric_name,
                candidate_defects,
                source_phase="json_screen",
                claim_status="suspicious_candidate",
            ),
            defects=[],
        )
        return record
    if verdict == "valid":
        record.update(
            decision_role="final_metric_verdict",
            final_metric_verdict=True,
            screen_status="clear",
            screen_state="clear",
        )
        return record
    record.update(
        decision_role="router_only",
        final_metric_verdict=False,
        screen_status="insufficient",
        screen_state="review_required",
    )
    return record


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
    """Retain Scale's established target-local direct-call fallback."""

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
