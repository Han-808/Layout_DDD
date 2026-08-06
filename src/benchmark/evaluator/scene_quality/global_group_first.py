"""Scene-global discovery followed by mandatory multi-object group review.

Style, functional, and semantic-placement consistency need two complementary
visual scopes. The scene-global pass owns scene-wide and cross-group claims;
the group-local pass inspects every non-singleton evidence group for defects
that need closer context. Neither pass is a router for the other, and
metric/object penalty units are deduplicated during aggregation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from benchmark.evaluator.scene_quality.claim_identity import (
    canonical_claim_key,
    canonical_target_ids,
    claim_record,
    claim_records,
    deduplicate_defects,
    object_level_finding_records,
)
from benchmark.evaluator.scene_quality.functional_probe import (
    acquire_functional_probe_evidence,
    functional_probe_judge_packet,
)
from benchmark.visual_judge.placement_discovery import (
    placement_groups_to_confirm,
)
from benchmark.visual_judge.orchestration.audit import (
    evidence_artifact_refs,
)


_SUPPORTED_METRICS = {
    "style_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
}


def evaluate_global_discovery_then_group_local(
    *,
    base: dict[str, Any],
    metric_name: str,
    metric_config: dict[str, Any],
    scene: dict[str, Any],
    object_ids: list[str],
    groups: list[dict[str, Any]] | None,
    grouping_report: dict[str, Any] | None,
    global_evidence: list[str],
    render_evidence: list[str] | dict[str, Any] | None,
    camera_evidence_provider: Any,
    functional_evidence_planner: Any = None,
    functional_probe_evidence_provider: Any = None,
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
    """Evaluate one scene-global scope and every eligible group-local scope."""

    if metric_name not in _SUPPORTED_METRICS:
        raise ValueError(
            "global/group evaluation only supports style, functional, and "
            f"semantic placement consistency, got {metric_name!r}"
        )

    plan = (
        metric_config.get("evidence_plan")
        if isinstance(metric_config.get("evidence_plan"), dict)
        else {}
    )
    global_plan = (
        plan.get("global_policy")
        if isinstance(plan.get("global_policy"), dict)
        else {}
    )
    global_budget = max(
        1,
        int(global_plan.get("image_budget") or len(global_evidence) or 1),
    )
    selected_global_evidence = _select_angled_global_context(
        _global_evidence_candidates(
            render_evidence,
            fallback=global_evidence,
        ),
        limit=global_budget,
    )
    if (
        metric_name == "semantic_placement_consistency"
        and selected_global_evidence
    ):
        placement_call = getattr(
            functional_evidence_planner,
            "discover_placement_evidence",
            None,
        )
        if callable(placement_call):
            try:
                placement_discovery = placement_call(
                    {
                        "metric": metric_name,
                        "scene_id": scene.get("scene_id"),
                        "scene_type": scene.get("scene_type"),
                        "global_image_path": selected_global_evidence[0],
                        "objects": _minimal_discovery_objects(scene),
                    }
                )
                base["placement_discovery"] = deepcopy(
                    placement_discovery
                )
            except Exception as exc:
                base["placement_discovery"] = {
                    "schema_version": "placement_discovery_v1",
                    "status": "failed",
                    "decision_authority": "none",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
    global_judge_evidence = list(selected_global_evidence)
    functional_probe_packet: dict[str, Any] | None = None
    functional_probe_budget = _functional_probe_budget(
        plan,
        judge=vlm_judge,
        global_image_count=len(selected_global_evidence),
        provider=functional_probe_evidence_provider,
    )
    base["functional_probe_budget"] = {
        "requested": _configured_functional_probe_units(plan),
        "effective": functional_probe_budget,
        "source": (
            "evidence_plan.prejudgement_probe_policy.max_probe_units"
            if _has_configured_functional_probe_units(plan)
            else "default"
        ),
        "total_image_budget": _resolved_total_image_budget(vlm_judge),
        "judge_packet_max_images": getattr(vlm_judge, "max_images", None),
    }
    if (
        metric_name == "functional_consistency"
        and selected_global_evidence
        and _functional_probe_enabled(plan)
    ):
        probe_paths, probe_audit = acquire_functional_probe_evidence(
            planner=functional_evidence_planner,
            provider=functional_probe_evidence_provider,
            scene=scene,
            global_image_path=selected_global_evidence[0],
            max_probe_units=functional_probe_budget,
            groups=groups,
            grouping_report=grouping_report,
        )
        cross_group_probe_paths = (
            list(probe_audit.get("cross_group_evidence_paths") or [])
            if probe_audit.get("planner_mode")
            == "functional_discovery_v3"
            else list(probe_paths)
        )
        global_judge_evidence = list(
            dict.fromkeys(
                [
                    *selected_global_evidence,
                    *cross_group_probe_paths,
                ]
            )
        )
        functional_probe_packet = functional_probe_judge_packet(
            global_paths=selected_global_evidence,
            probe_paths=cross_group_probe_paths,
            acquisition_audit=probe_audit,
        )
        base["functional_probe_acquisition"] = deepcopy(
            probe_audit
        )
        base["functional_probe_evidence_paths"] = list(probe_paths)
        base["functional_cross_group_evidence_paths"] = list(
            cross_group_probe_paths
        )
        base["functional_group_evidence_paths"] = deepcopy(
            probe_audit.get("group_evidence_paths") or {}
        )
        base["functional_discovery"] = deepcopy(
            probe_audit.get("functional_discovery")
        )
        base["functional_acquisition_plan"] = deepcopy(
            probe_audit.get("functional_acquisition_plan")
        )
        base["functional_probe_judge_packet"] = deepcopy(
            functional_probe_packet
        )
        base["evidence_paths"] = list(
            dict.fromkeys(
                [
                    *base.get("evidence_paths", []),
                    *probe_paths,
                ]
            )
        )
        probe_usages = [
            item.get("provider_usage")
            for item in probe_audit.get("probe_results") or []
            if isinstance(item, dict)
            and isinstance(item.get("provider_usage"), dict)
        ]
        probe_preview_count = sum(
            int(usage.get("preview_render_count") or 0)
            for usage in probe_usages
            if usage.get("cache_hit") is not True
        )
        boundary_evidence = (
            probe_audit.get("functional_boundary_evidence")
            if isinstance(
                probe_audit.get("functional_boundary_evidence"),
                dict,
            )
            else {}
        )
        boundary_preview_count = int(
            boundary_evidence.get("preview_render_count") or 0
        )
        probe_preview_count += boundary_preview_count
        probe_final_count = sum(
            int(usage.get("final_render_count") or 0)
            for usage in probe_usages
            if usage.get("cache_hit") is not True
        )
        if probe_paths:
            base["renderer_invoked"] = True
            base["final_render_count"] = int(
                base.get("final_render_count") or 0
            ) + (probe_final_count or len(probe_paths))
        if probe_preview_count:
            base["preview_renderer_invoked"] = True
            base["preview_render_count"] = int(
                base.get("preview_render_count") or 0
            ) + probe_preview_count
        probe_provider_invoked = any(
            isinstance(item, dict)
            for item in probe_audit.get("probe_results") or []
        ) or bool(boundary_evidence.get("provider_invoked"))
        base["evidence_request"].update(
            {
                "functional_probe_planner_invoked": (
                    probe_audit.get("status")
                    not in {"not_configured"}
                ),
                "functional_probe_provider_invoked": (
                    probe_provider_invoked
                ),
                "functional_probe_status": probe_audit.get(
                    "status"
                ),
                "functional_probe_count": len(probe_paths),
                "functional_probe_presentation": "raw_rgb_only",
                "provider_invoked": bool(
                    base["evidence_request"].get("provider_invoked")
                    or probe_provider_invoked
                ),
            }
        )

    pre_judge_artifact_paths = [
        *selected_global_evidence,
        *(
            list(
                (
                    base.get("functional_probe_acquisition") or {}
                ).get("acquired_artifact_paths")
                or []
            )
            if isinstance(
                base.get("functional_probe_acquisition"),
                dict,
            )
            else []
        ),
    ]
    base["camera_acquisition_ledger"] = (
        _initial_camera_acquisition_ledger(pre_judge_artifact_paths)
    )

    (
        global_record,
        global_outcome,
        global_audit,
    ) = _evaluate_global_scope(
        base=base,
        metric_name=metric_name,
        scene=scene,
        object_ids=object_ids,
        groups=groups,
        global_evidence=global_judge_evidence,
        functional_probe_packet=functional_probe_packet,
        vlm_judge=vlm_judge,
        prompt=prompt,
        visual_style_spec=visual_style_spec,
        authorized_deviations=authorized_deviations,
        build_judge_request=build_judge_request,
        call_judge=call_judge,
        apply_prompt_exemptions=apply_prompt_exemptions,
        normalize_judgement=normalize_judgement,
    )
    base["global_discovery"] = deepcopy(global_record)
    base["global_context_evidence_paths"] = list(
        selected_global_evidence
    )
    base["global_evidence_paths"] = list(global_judge_evidence)
    if global_audit is not None:
        base["global_camera_control_audit"] = deepcopy(global_audit)
    base["camera_acquisition_ledger"] = (
        _camera_acquisition_ledger_from_audit(global_audit)
        or deepcopy(base.get("camera_acquisition_ledger"))
        or _initial_camera_acquisition_ledger(global_judge_evidence)
    )

    global_defects = (
        deepcopy(global_record.get("defects") or [])
        if _is_invalid_outcome(global_outcome)
        else []
    )
    scene_claims = claim_records(
        metric_name,
        global_defects,
        source_phase="global_discovery",
        claim_status="final",
    )
    base["global_scene_claims"] = deepcopy(scene_claims)

    minimum_members = _minimum_group_members(plan)
    forced_group_ids = {
        str(item)
        for item in (
            (
                base.get("functional_probe_acquisition") or {}
            ).get("forced_group_ids")
            if isinstance(
                base.get("functional_probe_acquisition"),
                dict,
            )
            else []
        )
        or []
    }
    if (
        metric_name == "semantic_placement_consistency"
        and isinstance(base.get("placement_discovery"), dict)
    ):
        forced_group_ids.update(
            placement_groups_to_confirm(
                base["placement_discovery"],
                groups=groups or [],
            )
        )
    eligible_groups, skipped_groups = _filter_groups(
        groups,
        valid_object_ids=set(object_ids),
        minimum_members=minimum_members,
        forced_group_ids=forced_group_ids,
    )
    covered_ids = {
        str(member)
        for group in groups or []
        for member in group.get("object_ids") or []
    }
    singleton_only_partition = bool(groups) and not eligible_groups and (
        covered_ids == set(object_ids)
        and all(
            int(item["member_count"]) < minimum_members
            for item in skipped_groups
        )
    )
    group_phase_required = not singleton_only_partition
    if len(object_ids) < minimum_members:
        group_phase_required = False

    base["group_filter"] = {
        "policy": "skip_groups_below_minimum_members",
        "minimum_group_members": minimum_members,
        "force_for_eligible_groups": True,
        "eligible_group_ids": [
            str(group["group_id"]) for group in eligible_groups
        ],
        "functional_confirmation_forced_group_ids": sorted(
            forced_group_ids
        ),
        "skipped_groups": deepcopy(skipped_groups),
    }

    global_preview_count = int(base.get("preview_render_count") or 0)
    global_final_count = int(base.get("final_render_count") or 0)
    global_renderer_invoked = bool(base.get("renderer_invoked"))
    global_provider_invoked = bool(
        base.get("evidence_request", {}).get("provider_invoked")
    )

    packets: list[dict[str, Any]] = []
    if eligible_groups:
        local_policy = _local_evidence_policy(
            plan,
            selected_global_count=len(selected_global_evidence),
        )
        local_global_context = _select_angled_global_context(
            selected_global_evidence,
            limit=int(local_policy["global_image_budget"]),
        )
        local_input: dict[str, Any]
        if isinstance(render_evidence, dict):
            local_input = deepcopy(render_evidence)
        else:
            local_input = {}
        local_input["global"] = list(local_global_context)
        functional_acquisition = (
            base.get("functional_probe_acquisition")
            if isinstance(
                base.get("functional_probe_acquisition"),
                dict,
            )
            else {}
        )
        group_probe_paths = (
            functional_acquisition.get("group_evidence_paths")
            if metric_name == "functional_consistency"
            and isinstance(
                functional_acquisition.get("group_evidence_paths"),
                dict,
            )
            else {}
        )
        if group_probe_paths:
            existing_metric = local_input.get(metric_name)
            existing_metric = (
                deepcopy(existing_metric)
                if isinstance(existing_metric, dict)
                else {}
            )
            for group_id, paths in group_probe_paths.items():
                if isinstance(paths, list) and paths:
                    existing_paths = existing_metric.get(str(group_id))
                    existing_paths = (
                        existing_paths
                        if isinstance(existing_paths, list)
                        else []
                    )
                    existing_metric[str(group_id)] = list(
                        dict.fromkeys(
                            [
                                *[
                                    str(path)
                                    for path in existing_paths
                                    if str(path).strip()
                                ],
                                *[
                                    str(path)
                                    for path in paths
                                    if str(path).strip()
                                ],
                            ]
                        )
                    )
            local_input[metric_name] = existing_metric
            packet_capacity = _judge_packet_capacity(vlm_judge)
            largest_scoped_packet = max(
                (
                    len(paths)
                    for paths in existing_metric.values()
                    if isinstance(paths, list)
                ),
                default=int(local_policy["scoped_image_budget"]),
            )
            available_scoped_slots = max(
                0,
                packet_capacity
                - int(local_policy["global_image_budget"]),
            )
            local_policy["scoped_image_budget"] = min(
                largest_scoped_packet,
                available_scoped_slots,
            )
            local_policy["image_budget"] = (
                int(local_policy["global_image_budget"])
                + int(local_policy["scoped_image_budget"])
            )
        packets = resolve_group_evidence_packets(
            local_input,
            metric_name=metric_name,
            policy=local_policy,
            scene=scene,
            prompt=prompt,
            groups=eligible_groups,
            grouping_report=grouping_report,
            camera_evidence_provider=camera_evidence_provider,
            resolve_metric_evidence=resolve_metric_evidence,
            initial_acquisition_ledger=base.get(
                "camera_acquisition_ledger"
            ),
            max_total_images=_resolved_total_image_budget(vlm_judge),
            camera_target_ids_by_group=(
                _placement_camera_targets_by_group(
                    base.get("placement_discovery"),
                    groups=groups or [],
                )
                if metric_name
                == "semantic_placement_consistency"
                else None
            ),
        )
        packet_ledgers = [
            packet.get("camera_acquisition_ledger_after")
            for packet in packets
            if isinstance(
                packet.get("camera_acquisition_ledger_after"),
                dict,
            )
        ]
        if packet_ledgers:
            base["camera_acquisition_ledger"] = deepcopy(
                packet_ledgers[-1]
            )
        group_probe_packets = (
            functional_acquisition.get("group_probe_packets")
            if isinstance(
                functional_acquisition.get("group_probe_packets"),
                dict,
            )
            else {}
        )
        for packet in packets:
            group_id = str(packet["group"].get("group_id") or "")
            if group_id in group_probe_packets:
                packet["functional_probe_evidence"] = deepcopy(
                    group_probe_packets[group_id]
                )
            if (
                metric_name == "semantic_placement_consistency"
                and isinstance(base.get("placement_discovery"), dict)
            ):
                members = {
                    str(item)
                    for item in packet["group"].get("object_ids") or []
                }
                packet["placement_discovery"] = {
                    **deepcopy(base["placement_discovery"]),
                    "candidates": [
                        deepcopy(item)
                        for item in (
                            base["placement_discovery"].get("candidates")
                            or []
                        )
                        if isinstance(item, dict)
                        and str(item.get("subject_id")) in members
                    ],
                }
        _update_local_evidence_metadata(
            base,
            packets=packets,
            local_policy=local_policy,
            global_discovery_evidence=selected_global_evidence,
            local_global_context=local_global_context,
            group_packet_audit=group_packet_audit,
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
            evidence_phase="group_local_review",
            decision_mode="final",
        )
    else:
        result = base
        result["group_results"] = []
        result["selected_group_ids"] = []
        result["selected_object_ids"] = []
        result["local_evidence_paths"] = []
        result["evidence_request"]["group_requests"] = []

    group_results = result.get("group_results")
    group_results = group_results if isinstance(group_results, list) else []
    for record in group_results:
        judgement = (
            record.get("judgement")
            if isinstance(record.get("judgement"), dict)
            else {}
        )
        record["scene_claim_correspondence"] = (
            _compare_group_defects_to_scene_claims(
                metric_name,
                judgement.get("defects") or [],
                scene_claims,
            )
        )

    local_preview_count = (
        int(result.get("preview_render_count") or 0)
        if eligible_groups
        else 0
    )
    local_final_count = (
        int(result.get("final_render_count") or 0)
        if eligible_groups
        else 0
    )
    result["preview_render_count"] = (
        global_preview_count + local_preview_count
    )
    result["final_render_count"] = (
        global_final_count + local_final_count
    )
    result["preview_renderer_invoked"] = bool(
        result["preview_render_count"]
    )
    result["renderer_invoked"] = bool(
        global_renderer_invoked
        or result.get("renderer_invoked")
        or result["final_render_count"]
    )
    result["evidence_request"]["renderer_invoked"] = result[
        "renderer_invoked"
    ]
    result["evidence_request"]["provider_invoked"] = bool(
        global_provider_invoked
        or result["evidence_request"].get("provider_invoked")
    )

    result["judge_call_count"] = 1 + sum(
        1 for item in group_results if item.get("vlm_invoked")
    )
    result["vlm_invoked"] = True
    result["evidence_request"]["vlm_invoked"] = True
    result["global_discovery"] = deepcopy(global_record)
    result["global_scene_claims"] = deepcopy(scene_claims)
    result["route"] = "global_discovery_then_forced_group_local"
    result["aggregation_policy"] = (
        "invalid_if_scene_global_or_any_eligible_group_invalid"
    )

    group_phase_available = groups is not None
    group_phase_complete = (
        not group_phase_required
        or (
            bool(eligible_groups)
            and len(
                [
                    item
                    for item in group_results
                    if item.get("status") == "evaluated"
                ]
            )
            == len(eligible_groups)
        )
    )
    result["group_phase"] = {
        "required": group_phase_required,
        "grouping_available": group_phase_available,
        "status": (
            "not_required_singleton_only"
            if not group_phase_required
            else "complete"
            if group_phase_complete
            else "unresolved"
        ),
        "eligible_group_count": len(eligible_groups),
        "resolved_group_count": len(
            [
                item
                for item in group_results
                if item.get("status") == "evaluated"
            ]
        ),
    }
    aggregated = _aggregate_global_and_group_results(
        result,
        metric_name=metric_name,
        global_record=global_record,
        global_outcome=global_outcome,
        scene_claims=scene_claims,
        group_results=group_results,
        group_phase_required=group_phase_required,
        group_phase_complete=group_phase_complete,
    )
    return _apply_functional_acquisition_budget_status(
        aggregated,
        metric_name=metric_name,
    )


def _apply_functional_acquisition_budget_status(
    result: dict[str, Any],
    *,
    metric_name: str,
) -> dict[str, Any]:
    if metric_name != "functional_consistency":
        return result
    acquisition = result.get("functional_probe_acquisition")
    if (
        not isinstance(acquisition, dict)
        or acquisition.get("budget_exhausted") is not True
    ):
        return result
    result["functional_acquisition_coverage_complete"] = False
    result["functional_acquisition_budget_exhausted"] = True
    unscheduled = [
        deepcopy(item)
        for item in acquisition.get("unscheduled_discovery_items") or []
        if isinstance(item, dict)
    ]
    result["functional_acquisition_coverage"] = {
        "coverage_complete": False,
        "budget_exhausted": True,
        "unscheduled_acquisition_ids": [
            deepcopy(item.get("acquisition_identity"))
            for item in unscheduled
        ],
        "undelivered_observation_goals": [
            str(item.get("observation_goal") or "")
            for item in unscheduled
            if str(item.get("observation_goal") or "").strip()
        ],
        "undelivered_target_ids": list(
            dict.fromkeys(
                str(target_id)
                for item in unscheduled
                for target_id in item.get("target_ids") or []
                if str(target_id).strip()
            )
        ),
        "decision_authority": "none",
        "verdict_preserved": True,
    }
    return result


def _evaluate_global_scope(
    *,
    base: dict[str, Any],
    metric_name: str,
    scene: dict[str, Any],
    object_ids: list[str],
    groups: list[dict[str, Any]] | None,
    global_evidence: list[str],
    functional_probe_packet: dict[str, Any] | None,
    vlm_judge: Any,
    prompt: str | None,
    visual_style_spec: dict[str, Any] | None,
    authorized_deviations: list[dict[str, Any]],
    build_judge_request: Callable[..., dict[str, Any]],
    call_judge: Callable[[Any, dict[str, Any]], dict[str, Any]],
    apply_prompt_exemptions: Callable[..., dict[str, Any]],
    normalize_judgement: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    request = build_judge_request(
        metric_name=metric_name,
        scene=scene,
        prompt=prompt,
        render_evidence=global_evidence,
        selected_object_ids=[],
        selected_group_ids=[],
        groups=groups or [],
        authorized_deviations=authorized_deviations,
        visual_style_spec=visual_style_spec,
        evidence_phase="global_discovery",
        decision_mode="final",
        functional_probe_evidence=functional_probe_packet,
    )
    if isinstance(base.get("camera_acquisition_ledger"), dict):
        request["camera_acquisition_ledger"] = deepcopy(
            base["camera_acquisition_ledger"]
        )
    if isinstance(base.get("placement_discovery"), dict):
        request["placement_discovery"] = deepcopy(
            base["placement_discovery"]
        )
    base["evidence_request"]["vlm_invoked"] = True
    base["evidence_request"]["evidence_phase"] = "global_discovery"
    base["vlm_invoked"] = True
    audit_records = getattr(vlm_judge, "audit_records", None)
    audit_start = (
        len(audit_records) if isinstance(audit_records, list) else None
    )
    try:
        raw = call_judge(vlm_judge, request)
        adjusted = apply_prompt_exemptions(
            raw,
            metric_name=metric_name,
            authorized_deviations=authorized_deviations,
        )
        outcome = normalize_judgement(
            adjusted,
            metric_name=metric_name,
            valid_object_ids=set(object_ids),
        )
        record = deepcopy(adjusted)
        evaluated = outcome.get("status") == "evaluated"
        invalid = evaluated and float(outcome.get("score")) == 0.0
        record.update(
            scope_level="scene_global",
            decision_role=(
                "final_scene_scope_verdict"
                if evaluated
                else "required_scene_scope_unresolved"
            ),
            final_metric_verdict=evaluated,
            global_status=(
                "confirmed_invalid"
                if invalid
                else "clear"
                if evaluated
                else "insufficient"
            ),
            does_not_short_circuit_group_review=True,
        )
    except Exception as exc:
        record = {
            "scope_level": "scene_global",
            "decision_role": "required_scene_scope_unresolved",
            "final_metric_verdict": False,
            "global_status": "failed",
            "does_not_short_circuit_group_review": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "defects": [],
        }
        outcome = {
            "status": "unresolved",
            "score": None,
            "reason": "vlm_global_discovery_failed",
        }
    audit = None
    if (
        audit_start is not None
        and isinstance(audit_records, list)
        and len(audit_records) > audit_start
    ):
        audit = deepcopy(audit_records[-1])
    return record, outcome, audit


def _minimal_discovery_objects(
    scene: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "id": str(item.get("id")),
            "category": str(
                item.get("category")
                or item.get("retrieval_category")
                or "unknown"
            ),
        }
        for item in scene.get("objects") or []
        if isinstance(item, dict) and item.get("id")
    ]


def _placement_camera_targets_by_group(
    discovery: Any,
    *,
    groups: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Keep defect ownership local while framing stated placement context."""

    if not isinstance(discovery, dict):
        return {}
    owner_by_object = {
        str(object_id): str(group.get("group_id") or "")
        for group in groups
        if isinstance(group, dict)
        for object_id in group.get("object_ids") or []
        if str(object_id).strip()
    }
    known_ids = set(owner_by_object)
    result: dict[str, list[str]] = {}
    for candidate in discovery.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        subject_id = str(candidate.get("subject_id") or "").strip()
        group_id = owner_by_object.get(subject_id)
        if not subject_id or not group_id:
            continue
        targets = result.setdefault(group_id, [])
        for object_id in [
            subject_id,
            *[
                str(item).strip()
                for item in candidate.get("context_ids") or []
            ],
        ]:
            if (
                object_id
                and object_id in known_ids
                and object_id not in targets
            ):
                targets.append(object_id)
    return result


def _functional_probe_enabled(plan: dict[str, Any]) -> bool:
    policy = (
        plan.get("prejudgement_probe_policy")
        if isinstance(plan.get("prejudgement_probe_policy"), dict)
        else {}
    )
    return bool(policy.get("enabled", True))


def _camera_acquisition_ledger_from_audit(
    record: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    audit = (
        record.get("audit")
        if isinstance(record.get("audit"), dict)
        else record
    )
    acquisition = (
        audit.get("camera_acquisition")
        if isinstance(audit, dict)
        and isinstance(audit.get("camera_acquisition"), dict)
        else {}
    )
    ledger = acquisition.get("ledger")
    return deepcopy(ledger) if isinstance(ledger, dict) else None


def _initial_camera_acquisition_ledger(
    paths: list[str],
) -> dict[str, Any]:
    artifact_ids = list(
        dict.fromkeys(evidence_artifact_refs(list(paths)))
    )
    return {
        "schema_version": "metric_camera_acquisition_ledger_v1",
        "artifact_ids": artifact_ids,
        "total_images_acquired": len(artifact_ids),
        "evidence_rounds": 0,
        "selector_calls": 0,
        "camera_actions": 0,
        "deterministic_rounds": 0,
        "vlm_rounds": 0,
    }


def _functional_probe_budget(
    plan: dict[str, Any],
    *,
    judge: Any,
    global_image_count: int,
    provider: Any,
) -> int:
    configured = _configured_functional_probe_units(plan)
    total_capacity = _resolved_total_image_budget(judge)
    if total_capacity is None:
        return configured
    remaining_artifacts = max(
        0,
        total_capacity - max(0, int(global_image_count)),
    )
    artifacts_per_probe = getattr(
        provider,
        "functional_probe_full_artifacts_per_selected_view",
        1,
    )
    if (
        isinstance(artifacts_per_probe, bool)
        or not isinstance(artifacts_per_probe, int)
        or artifacts_per_probe < 1
    ):
        artifacts_per_probe = 1
    return min(
        configured,
        remaining_artifacts // artifacts_per_probe,
    )


def _configured_functional_probe_units(
    plan: dict[str, Any],
) -> int:
    policy = (
        plan.get("prejudgement_probe_policy")
        if isinstance(plan.get("prejudgement_probe_policy"), dict)
        else {}
    )
    value = policy.get("max_probe_units")
    return max(0, int(4 if value is None else value))


def _has_configured_functional_probe_units(
    plan: dict[str, Any],
) -> bool:
    policy = plan.get("prejudgement_probe_policy")
    return isinstance(policy, dict) and "max_probe_units" in policy


def _resolved_total_image_budget(judge: Any) -> int | None:
    control = getattr(judge, "control", None)
    value = getattr(control, "max_total_images", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return None


def _judge_packet_capacity(judge: Any) -> int:
    value = getattr(judge, "max_images", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, value)
    return 6


def _minimum_group_members(plan: dict[str, Any]) -> int:
    local_plan = (
        plan.get("local_policy")
        if isinstance(plan.get("local_policy"), dict)
        else {}
    )
    return max(2, int(local_plan.get("minimum_group_members") or 2))


def _filter_groups(
    groups: list[dict[str, Any]] | None,
    *,
    valid_object_ids: set[str],
    minimum_members: int,
    forced_group_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    forced = set(forced_group_ids or set())
    for group in groups or []:
        members = list(
            dict.fromkeys(
                str(member)
                for member in group.get("object_ids") or []
                if str(member) in valid_object_ids
            )
        )
        normalized = {**deepcopy(group), "object_ids": members}
        group_id = str(group.get("group_id") or "")
        if len(members) >= minimum_members or group_id in forced:
            if group_id in forced:
                normalized["functional_confirmation_forced"] = True
            eligible.append(normalized)
            continue
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


def _local_evidence_policy(
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
    global_context_budget = min(
        selected_global_count,
        max(
            1,
            int(
                local_plan.get("global_context_image_budget")
                or 1
            ),
        ),
    )
    max_packet_images = max(
        global_context_budget + local_budget,
        int(
            local_plan.get("max_packet_images")
            or global_context_budget + local_budget
        ),
    )
    image_order = local_plan.get("image_order")
    if not isinstance(image_order, list) or not image_order:
        image_order = ["global_context", "group_local"]
    return {
        "camera_scope": "group_local",
        "camera_mode": "metric_local",
        "selector": "deterministic",
        "image_budget": max_packet_images,
        "global_image_budget": global_context_budget,
        "scoped_image_budget": local_budget,
        "presentation": "raw",
        "image_order": list(image_order),
        "include_global_context": global_context_budget > 0,
        "global_context_view_family": "angled_perspective",
        "camera_pose_mode": str(
            local_plan.get("camera_pose_mode") or "visibility_ranked"
        ),
    }


def _select_angled_global_context(
    paths: list[str],
    *,
    limit: int,
) -> list[str]:
    """Prefer an angled overview over a top-down image for local context."""

    if limit < 1:
        return []
    unique = list(dict.fromkeys(str(path) for path in paths if path))

    def rank(path: str) -> int:
        lowered = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if any(
            token in lowered
            for token in ("perspective", "oblique", "angled")
        ):
            return 0
        if any(
            token in lowered
            for token in ("top", "overhead", "birdseye", "bird_eye")
        ):
            return 2
        return 1

    return sorted(unique, key=rank)[:limit]


def _global_evidence_candidates(
    render_evidence: list[str] | dict[str, Any] | None,
    *,
    fallback: list[str],
) -> list[str]:
    """Recover all supplied overview candidates before applying the budget.

    The generic evidence resolver enforces the one-image budget in input
    order.  Camera-cal packets place the top view after the perspective view,
    but callers are not required to preserve that order.  Inspecting the raw
    global packet here lets this metric-specific route select by view family
    before truncation without changing the shared L1 evidence packet.
    """

    raw: Any = (
        render_evidence.get("global")
        if isinstance(render_evidence, dict)
        else render_evidence
    )
    candidates: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                path = item.get("path") or item.get("image_path")
            else:
                path = item
            if path:
                candidates.append(str(path))
    candidates.extend(str(path) for path in fallback if path)
    return list(dict.fromkeys(candidates))


def _update_local_evidence_metadata(
    base: dict[str, Any],
    *,
    packets: list[dict[str, Any]],
    local_policy: dict[str, Any],
    global_discovery_evidence: list[str],
    local_global_context: list[str],
    group_packet_audit: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    selected_group_ids = [
        str(packet["group"]["group_id"]) for packet in packets
    ]
    selected_object_ids = list(
        dict.fromkeys(
            str(member)
            for packet in packets
            for member in packet["group"].get("object_ids") or []
        )
    )
    all_paths = list(
        dict.fromkeys(
            [
                *global_discovery_evidence,
                *(
                    path
                    for packet in packets
                    for path in packet.get("paths") or []
                ),
            ]
        )
    )
    global_set = set(global_discovery_evidence)
    base["selected_group_ids"] = selected_group_ids
    base["selected_object_ids"] = selected_object_ids
    base["evidence_paths"] = all_paths
    base["local_evidence_paths"] = [
        path for path in all_paths if path not in global_set
    ]
    base["resolved_global_evidence_policy"] = deepcopy(
        base.get("resolved_evidence_policy")
    )
    base["resolved_local_evidence_policy"] = deepcopy(local_policy)
    base["dependencies"].update(
        {
            "object_grouping": "available",
            "requested_evidence_scope": "group_local",
            "evidence_scope_satisfied": bool(packets)
            and all(
                packet["resolution"].get("scope_satisfied") is True
                for packet in packets
            ),
            "evidence_source": "global_then_per_group_camera_evidence",
            "provider_status": _packet_status(packets),
        }
    )
    base["evidence_request"].update(
        {
            "camera_scope": "group_local",
            "image_budget": local_policy["image_budget"],
            "global_image_budget": local_policy["global_image_budget"],
            "scoped_image_budget": local_policy["scoped_image_budget"],
            "global_context_paths": list(local_global_context),
            "image_order": list(local_policy["image_order"]),
            "include_global_context": local_policy[
                "include_global_context"
            ],
            "evidence_phase": "group_local_review",
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
            "evidence_source": "global_then_per_group_camera_evidence",
            "scope_satisfied": bool(packets)
            and all(
                packet["resolution"].get("scope_satisfied") is True
                for packet in packets
            ),
            "missing_paths": list(
                dict.fromkeys(
                    str(path)
                    for packet in packets
                    for path in packet["resolution"].get(
                        "missing_paths"
                    )
                    or []
                )
            ),
            "target_object_ids": selected_object_ids,
            "target_group_ids": selected_group_ids,
            "group_requests": [
                group_packet_audit(packet) for packet in packets
            ],
        }
    )


def _aggregate_global_and_group_results(
    base: dict[str, Any],
    *,
    metric_name: str,
    global_record: dict[str, Any],
    global_outcome: dict[str, Any],
    scene_claims: list[dict[str, Any]],
    group_results: list[dict[str, Any]],
    group_phase_required: bool,
    group_phase_complete: bool,
) -> dict[str, Any]:
    evaluated_groups = [
        item
        for item in group_results
        if item.get("status") == "evaluated"
    ]
    invalid_groups = [
        item for item in evaluated_groups if item.get("score") == 0.0
    ]
    global_evaluated = global_outcome.get("status") == "evaluated"
    global_invalid = _is_invalid_outcome(global_outcome)
    global_valid = (
        global_evaluated
        and float(global_outcome.get("score")) == 1.0
    )

    local_defects = [
        deepcopy(defect)
        for item in invalid_groups
        for defect in (
            (item.get("judgement") or {}).get("defects") or []
        )
        if isinstance(defect, dict)
    ]
    global_defects = [
        deepcopy(defect)
        for defect in global_record.get("defects") or []
        if global_invalid and isinstance(defect, dict)
    ]
    defects = deduplicate_defects(
        metric_name,
        [*global_defects, *local_defects],
    )
    object_findings = object_level_finding_records(
        metric_name,
        [
            *[
                ("global_discovery", defect)
                for defect in global_defects
            ],
            *[
                (
                    f"group_local_review:{item.get('group_id')}",
                    defect,
                )
                for item in invalid_groups
                for defect in (
                    (item.get("judgement") or {}).get("defects") or []
                )
            ],
        ],
    )
    raw_observation_count = sum(
        int(finding.get("observation_count") or 0)
        for finding in object_findings
    )
    base["final_object_findings"] = deepcopy(object_findings)
    base["object_level_attribution"] = {
        "enabled": True,
        "unit": "object",
        "deduplication_key": ["metric", "object_id"],
        "cross_phase_deduplication": True,
        "cross_metric_deduplication": False,
        "raw_defect_observation_count": raw_observation_count,
        "unique_object_count": len(object_findings),
        "merged_duplicate_observation_count": max(
            0,
            raw_observation_count - len(object_findings),
        ),
        "penalty_unit_count": len(object_findings),
    }

    final_claims = list(scene_claims)
    seen_claim_ids = {
        str(claim.get("claim_id"))
        for claim in final_claims
        if claim.get("claim_id")
    }
    for item in invalid_groups:
        for defect in (item.get("judgement") or {}).get("defects") or []:
            if not isinstance(defect, dict):
                continue
            claim = claim_record(
                metric_name,
                defect,
                source_phase=(
                    f"group_local_review:{item.get('group_id')}"
                ),
                claim_status="final",
            )
            claim_id = str(claim["claim_id"])
            if claim_id in seen_claim_ids:
                continue
            seen_claim_ids.add(claim_id)
            final_claims.append(claim)
    base["final_defect_claims"] = final_claims

    missing_evidence: list[str] = []
    if not global_evaluated:
        missing_evidence.append("scene_global_judgement")
    missing_evidence.extend(
        f"group_local_judgement:{item.get('group_id')}"
        for item in group_results
        if item.get("status") != "evaluated"
    )
    if group_phase_required and not group_results:
        missing_evidence.append("eligible_group_partition")

    required_group_units = (
        len(group_results)
        if group_results
        else 1
        if group_phase_required
        else 0
    )
    eligible_count = 1 + required_group_units
    resolved_count = int(global_evaluated) + len(evaluated_groups)
    coverage_complete = bool(
        global_evaluated and group_phase_complete
    )
    base["coverage"] = {
        "eligible_count": eligible_count,
        "resolved_count": resolved_count,
        "fraction": (
            resolved_count / eligible_count if eligible_count else None
        ),
        "complete": coverage_complete,
        "scene_global_resolved": global_evaluated,
        "group_phase_complete": group_phase_complete,
    }

    if global_invalid or invalid_groups:
        invalid_judgements: list[dict[str, Any]] = []
        if global_invalid:
            invalid_judgements.append(global_record)
        invalid_judgements.extend(
            item.get("judgement") or {}
            for item in invalid_groups
        )
        confidence = min(
            (
                float(item.get("confidence") or 0.0)
                for item in invalid_judgements
            ),
            default=0.0,
        )
        judgement = {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": confidence,
            "reason": (
                "At least one scene-global or eligible group-local scope has "
                "a significant in-scope defect."
            ),
            "missing_evidence": [],
            "unresolved_scopes": missing_evidence,
            "defects": defects,
            "object_findings": deepcopy(object_findings),
            "object_penalty_count": len(object_findings),
            "object_penalty_policy": (
                "one_per_metric_object_across_global_and_local"
            ),
            "aggregation": (
                "invalid_if_scene_global_or_any_eligible_group_invalid"
            ),
            "scene_global_judgement": deepcopy(global_record),
            "group_judgements": deepcopy(group_results),
        }
        base.update(
            status="evaluated",
            reason=None,
            score=0.0,
            judgement=judgement,
        )
        return base

    all_groups_valid = all(
        item.get("score") == 1.0 for item in evaluated_groups
    )
    if global_valid and group_phase_complete and all_groups_valid:
        confidence_values = [
            float(global_record.get("confidence") or 0.0),
            *[
                float(
                    (item.get("judgement") or {}).get("confidence")
                    or 0.0
                )
                for item in evaluated_groups
            ],
        ]
        base.update(
            status="evaluated",
            reason=None,
            score=1.0,
            judgement={
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": min(confidence_values),
                "reason": (
                    "The scene-global scope and every eligible multi-object "
                    "group resolved without an in-scope defect."
                ),
                "missing_evidence": [],
                "defects": [],
                "object_findings": [],
                "object_penalty_count": 0,
                "object_penalty_policy": (
                    "one_per_metric_object_across_global_and_local"
                ),
                "aggregation": (
                    "scene_global_and_all_eligible_groups_must_resolve_valid"
                ),
                "scene_global_judgement": deepcopy(global_record),
                "group_judgements": deepcopy(group_results),
            },
        )
        return base

    base.update(
        status="unresolved",
        reason="one_or_more_required_visual_scopes_unresolved",
        score=None,
        judgement={
            "evidence_status": "insufficient",
            "verdict": "ambiguous",
            "confidence": 0.0,
            "reason": (
                "The metric cannot resolve valid because one or more required "
                "scene-global or group-local scopes remain unresolved."
            ),
            "missing_evidence": missing_evidence,
            "defects": [],
            "object_findings": [],
            "object_penalty_count": 0,
            "object_penalty_policy": (
                "one_per_metric_object_across_global_and_local"
            ),
            "aggregation": (
                "unresolved_without_complete_global_and_group_coverage"
            ),
            "scene_global_judgement": deepcopy(global_record),
            "group_judgements": deepcopy(group_results),
        },
    )
    return base


def _compare_group_defects_to_scene_claims(
    metric_name: str,
    defects: list[Any],
    scene_claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    scene_claims_by_object: dict[str, list[dict[str, Any]]] = {}
    for claim in scene_claims:
        for object_id in canonical_target_ids(claim):
            scene_claims_by_object.setdefault(object_id, []).append(claim)
    for defect in deduplicate_defects(metric_name, defects):
        final_claim = claim_record(
            metric_name,
            defect,
            source_phase="group_local_review",
            claim_status="final",
        )
        exact = next(
            (
                claim
                for claim in scene_claims
                if canonical_claim_key(metric_name, claim)
                == canonical_claim_key(metric_name, defect)
            ),
            None,
        )
        same_targets = exact or next(
            (
                claim
                for claim in scene_claims
                if canonical_claim_key(metric_name, claim)[:3]
                == canonical_claim_key(metric_name, defect)[:3]
            ),
            None,
        )
        group_object_ids = set(canonical_target_ids(defect))
        overlapping_object_ids = sorted(
            group_object_ids & set(scene_claims_by_object)
        )
        new_object_ids = sorted(
            group_object_ids - set(overlapping_object_ids)
        )
        overlapping_claim_ids = list(
            dict.fromkeys(
                str(claim.get("claim_id"))
                for object_id in overlapping_object_ids
                for claim in scene_claims_by_object.get(object_id, [])
                if claim.get("claim_id")
            )
        )
        comparisons.append(
            {
                "group_claim": final_claim,
                "scene_claim_id": (
                    str(same_targets.get("claim_id"))
                    if isinstance(same_targets, dict)
                    else None
                ),
                "relationship": (
                    "duplicate_of_scene_claim"
                    if exact is not None
                    else "partially_overlaps_scene_object_finding"
                    if overlapping_object_ids and new_object_ids
                    else "same_metric_object_already_flagged_global"
                    if overlapping_object_ids
                    else "same_targets_distinct_relation"
                    if same_targets is not None
                    else "new_group_claim"
                ),
                "object_level_deduplication": {
                    "deduplication_key": ["metric", "object_id"],
                    "overlapping_object_ids": overlapping_object_ids,
                    "new_object_ids": new_object_ids,
                    "scene_claim_ids": overlapping_claim_ids,
                    "duplicate_penalty_suppressed": bool(
                        overlapping_object_ids
                    ),
                    "cross_metric_deduplication": False,
                },
            }
        )
    return comparisons


def _packet_status(packets: list[dict[str, Any]]) -> str:
    statuses = {
        str(packet["resolution"].get("provider_status") or "unknown")
        for packet in packets
    }
    if not statuses:
        return "not_requested"
    return next(iter(statuses)) if len(statuses) == 1 else "mixed"


def _is_invalid_outcome(outcome: dict[str, Any]) -> bool:
    return (
        outcome.get("status") == "evaluated"
        and float(outcome.get("score")) == 0.0
    )
