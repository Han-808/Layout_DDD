"""Scene-global, cross-group relation, and multi-object group review.

Functional and semantic-placement consistency retain two complementary visual
scopes. Functional consistency adds a bounded middle stage in which every
discovered cross-group target set receives its own Judge episode. Atomic
relation predicates sharing that target set receive separate result rows. The
scene-global pass owns only overall scene-level claims, while the group-local
pass inspects every ordinarily eligible group and any singleton group that owns
an explicit routed check. Metric/object penalty units are deduplicated during
aggregation. Conditional Style routing lives only in ``style_global_first``.
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
from benchmark.evaluator.scene_quality.cross_group_relations import (
    _camera_acquisition_ledger_from_audit,
    _cross_group_relation_episode_specs,
    _discovered_cross_group_target_sets,
    _evaluate_cross_group_relation_scopes,
    _forbidden_cross_group_defects,
    _initial_camera_acquisition_ledger,
    _relation_episode_defect_violations,
    _relation_schedule_audit,
)
from benchmark.evaluator.scene_quality.functional_prejudgement import (
    FunctionalPrejudgementEvidenceRequest,
    FunctionalPrejudgementEvidenceResult,
    resolve_functional_prejudgement_evidence_source,
)
from benchmark.evaluator.scene_quality.functional_checks import (
    FUNCTIONAL_CHECK_LEDGER_VERSION,
    apply_functional_check_judgements,
    build_functional_check_ledger,
    canonicalize_typed_invalid_envelope,
    checks_for_group,
    forced_group_ids_from_checks,
)
from benchmark.evaluator.scene_quality.functional_ownership import (
    build_cross_metric_ownership_audit,
    build_functional_ownership_ledger,
    validate_functional_ownership_ledger,
)
from benchmark.evaluator.scene_quality.placement_checks import (
    PLACEMENT_CHECK_LEDGER_VERSION,
    apply_placement_check_judgements,
    build_placement_check_ledger,
    canonicalize_placement_defect_linkage,
    forced_group_ids_from_placement_checks,
    merge_placement_checks,
    normalize_judge_originated_placement_results,
    placement_camera_targets_by_group,
    placement_checks_for_group,
    placement_global_checks,
    validate_placement_check_results,
)
from benchmark.evaluator.scene_quality.functional_planner_adapter import (
    is_functional_discovery_planner_mode,
)
from benchmark.evaluator.scene_quality.functional_probe import (
    FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION,
    functional_probe_judge_packet,
)
from benchmark.evaluator.scene_quality.placement_severity import (
    placement_severity_summary,
)
from benchmark.visual_judge.functional_evidence import (
    FUNCTIONAL_PROBE_DEFAULT_UNITS,
    FUNCTIONAL_PROBE_MAX_UNITS,
)
from benchmark.visual_judge.orchestration.budget import (
    merge_acquisition_ledger_delta,
)
from benchmark.visual_judge.contracts import (
    response_schema_audit_from_exception,
)


_SUPPORTED_METRICS = {
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
    functional_prejudgement_evidence_source: Any = None,
    functional_prejudgement_evidence_config: dict[str, Any] | None = None,
    discovery_identity_image_path: str | None = None,
    discovery_identity_legend: dict[str, str] | None = None,
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
    functional_ownership_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one scene-global scope and every eligible group-local scope."""

    if metric_name not in _SUPPORTED_METRICS:
        raise ValueError(
            "mandatory global/group evaluation only supports functional "
            "and semantic placement consistency, got "
            f"{metric_name!r}"
        )
    if functional_ownership_ledger is not None:
        functional_ownership_ledger = (
            validate_functional_ownership_ledger(
                functional_ownership_ledger,
                known_object_ids=object_ids,
            )
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
                        "identity_image_path": (
                            discovery_identity_image_path
                        ),
                        "identity_legend": deepcopy(
                            discovery_identity_legend or {}
                        ),
                        "objects": _minimal_discovery_objects(scene),
                    }
                )
                base["placement_discovery"] = deepcopy(
                    placement_discovery
                )
                base["placement_check_ledger"] = (
                    build_placement_check_ledger(
                        placement_discovery,
                        groups=groups,
                    )
                )
            except Exception as exc:
                schema_audit = response_schema_audit_from_exception(exc)
                base["placement_discovery"] = {
                    "schema_version": "placement_discovery_v2",
                    "status": "failed",
                    "decision_authority": "none",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **(
                        {"response_schema_audit": schema_audit}
                        if schema_audit is not None
                        else {}
                    ),
                }
                base["placement_check_ledger"] = {
                    "schema_version": PLACEMENT_CHECK_LEDGER_VERSION,
                    "checks": [],
                    "accepted_check_count": 0,
                    "decision_authority": "none",
                }
    if metric_name == "semantic_placement_consistency":
        base.setdefault(
            "placement_check_ledger",
            {
                "schema_version": PLACEMENT_CHECK_LEDGER_VERSION,
                "checks": [],
                "accepted_check_count": 0,
                "decision_authority": "none",
            },
        )
        base["functional_ownership_ledger"] = deepcopy(
            functional_ownership_ledger
        )
    global_judge_evidence = list(selected_global_evidence)
    functional_probe_packet: dict[str, Any] | None = None
    cross_group_relation_specs: list[dict[str, Any]] = []
    discovered_cross_group_target_sets: list[tuple[str, ...]] = []
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
        "budget_enforcement_scope": (
            "probe_units_across_separate_judge_episodes"
        ),
        "judge_episode_image_cap_is_metric_wide_authority": False,
    }
    if metric_name == "functional_consistency":
        base["functional_prejudgement_evidence_mode"] = str(
            getattr(
                functional_prejudgement_evidence_source,
                "mode",
                (functional_prejudgement_evidence_config or {}).get(
                    "mode",
                    "runtime",
                ),
            )
        )
        base["functional_prejudgement_evidence_source"] = {
            "mode": base[
                "functional_prejudgement_evidence_mode"
            ],
            "status": "not_executed",
            "decision_authority": "none",
        }
        base["prejudgement_functional_stage"] = {
            "planner_calls": 0,
            "usable_surface_detector_calls": 0,
            "selector_calls": 0,
            "preview_render_count": 0,
            "full_render_count": 0,
            "judge_facing_image_count": 0,
            "cache_hits": 0,
        }
    if (
        metric_name == "functional_consistency"
        and selected_global_evidence
        and _functional_probe_enabled(plan)
        and _functional_prejudgement_should_run(
            functional_prejudgement_evidence_config,
            planner=functional_evidence_planner,
            provider=functional_probe_evidence_provider,
            injected_source=functional_prejudgement_evidence_source,
        )
    ):
        prejudgement_source = (
            resolve_functional_prejudgement_evidence_source(
                functional_prejudgement_evidence_config,
                planner=functional_evidence_planner,
                provider=functional_probe_evidence_provider,
                injected_source=(
                    functional_prejudgement_evidence_source
                ),
            )
        )
        prejudgement_result = (
            prejudgement_source.prepare_functional_evidence(
                FunctionalPrejudgementEvidenceRequest.create(
                    scene=scene,
                    global_image_path=selected_global_evidence[0],
                    max_probe_units=functional_probe_budget,
                    groups=groups,
                    grouping_report=grouping_report,
                    identity_image_path=(
                        discovery_identity_image_path
                    ),
                    identity_legend=discovery_identity_legend,
                )
            )
        )
        if not isinstance(
            prejudgement_result,
            FunctionalPrejudgementEvidenceResult,
        ):
            raise TypeError(
                "functional prejudgement source must return "
                "FunctionalPrejudgementEvidenceResult"
            )
        probe_paths = list(
            prejudgement_result.selected_judge_probe_paths
        )
        probe_audit = deepcopy(
            prejudgement_result.runtime_audit
        )
        # Consume the validated source contract as authoritative.  In
        # particular, frozen mode must not depend on a stale or independently
        # reconstructed copy embedded in its legacy runtime audit.
        probe_audit["selected_raw_rgb_paths"] = list(
            prejudgement_result.selected_judge_probe_paths
        )
        probe_audit["cross_group_evidence_paths"] = list(
            prejudgement_result.cross_group_probe_paths
        )
        probe_audit["group_probe_packets"] = deepcopy(
            prejudgement_result.group_owned_probe_packets
        )
        if prejudgement_result.functional_discovery is not None:
            probe_audit["functional_discovery"] = deepcopy(
                prejudgement_result.functional_discovery
            )
        if (
            prejudgement_result.functional_boundary_evidence
            is not None
        ):
            probe_audit["functional_boundary_evidence"] = deepcopy(
                prejudgement_result.functional_boundary_evidence
            )
        if prejudgement_result.acquisition_plan is not None:
            probe_audit["functional_acquisition_plan"] = deepcopy(
                prejudgement_result.acquisition_plan
            )
        probe_audit["unscheduled_discovery_items"] = list(
            deepcopy(
                prejudgement_result.unscheduled_discovery_items
            )
        )
        functional_check_ledger = _functional_check_ledger_from_audit(
            probe_audit,
            groups=groups or [],
            scene=scene,
        )
        probe_audit["functional_check_ledger"] = deepcopy(
            functional_check_ledger
        )
        base["functional_prejudgement_evidence"] = (
            prejudgement_result.to_dict()
        )
        base["prejudgement_functional_stage"] = deepcopy(
            prejudgement_result.telemetry
        )
        base["functional_prejudgement_evidence_source"] = (
            deepcopy(prejudgement_source.manifest())
        )
        base["functional_prejudgement_evidence_mode"] = str(
            prejudgement_source.mode
        )
        cross_group_probe_paths = list(
            prejudgement_result.cross_group_probe_paths
        )
        if (
            not cross_group_probe_paths
            and not is_functional_discovery_planner_mode(
                probe_audit.get("planner_mode")
            )
        ):
            cross_group_probe_paths = list(probe_paths)
        cross_group_packet: dict[str, Any] | None = None
        if prejudgement_source.mode != "disabled":
            cross_group_packet = deepcopy(
                prejudgement_result.cross_group_probe_packet
            )
            if cross_group_packet is None:
                cross_group_packet = functional_probe_judge_packet(
                    global_paths=selected_global_evidence,
                    probe_paths=cross_group_probe_paths,
                    acquisition_audit=probe_audit,
                )
        if is_functional_discovery_planner_mode(
            probe_audit.get("planner_mode")
        ):
            cross_group_relation_specs = (
                _cross_group_relation_episode_specs(
                    acquisition_audit=probe_audit,
                    groups=groups or [],
                    global_paths=selected_global_evidence,
                )
            )
            discovered_cross_group_target_sets = (
                _discovered_cross_group_target_sets(probe_audit)
            )
            # Typed discovery gives each cross-group target set an isolated
            # Judge episode. The scene-global Judge receives no relation
            # probes.
            global_judge_evidence = list(selected_global_evidence)
            functional_probe_packet = None
        else:
            # Preserve the legacy planner's undifferentiated global packet.
            global_judge_evidence = list(
                dict.fromkeys(
                    [
                        *selected_global_evidence,
                        *cross_group_probe_paths,
                    ]
                )
            )
            functional_probe_packet = deepcopy(cross_group_packet)
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
        base["functional_check_ledger"] = deepcopy(
            functional_check_ledger
        )
        base["functional_probe_judge_packet"] = deepcopy(
            cross_group_packet
        )
        base["functional_cross_group_relation_schedule"] = [
            _relation_schedule_audit(spec)
            for spec in cross_group_relation_specs
        ]
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
    global_episode_ledger = _initial_camera_acquisition_ledger(
        global_judge_evidence
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
        camera_acquisition_ledger=global_episode_ledger,
        forbidden_cross_group_target_sets=(
            discovered_cross_group_target_sets
        ),
        required_placement_checks=(
            placement_global_checks(base.get("placement_check_ledger"))
            if metric_name == "semantic_placement_consistency"
            else []
        ),
        functional_ownership_ledger=(
            functional_ownership_ledger
            if metric_name == "semantic_placement_consistency"
            else None
        ),
    )
    base["global_discovery"] = deepcopy(global_record)
    base["global_context_evidence_paths"] = list(
        selected_global_evidence
    )
    base["global_evidence_paths"] = list(global_judge_evidence)
    if global_audit is not None:
        base["global_camera_control_audit"] = deepcopy(global_audit)
    global_episode_after = _camera_acquisition_ledger_from_audit(
        global_audit
    )
    if global_episode_after is not None:
        base["camera_acquisition_ledger"] = (
            merge_acquisition_ledger_delta(
                base.get("camera_acquisition_ledger"),
                episode_before=global_episode_ledger,
                episode_after=global_episode_after,
            )
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
    relation_results = (
        _evaluate_cross_group_relation_scopes(
            specs=cross_group_relation_specs,
            metric_name=metric_name,
            scene=scene,
            global_evidence=selected_global_evidence,
            vlm_judge=vlm_judge,
            prompt=prompt,
            visual_style_spec=visual_style_spec,
            authorized_deviations=authorized_deviations,
            build_judge_request=build_judge_request,
            call_judge=call_judge,
            apply_prompt_exemptions=apply_prompt_exemptions,
            normalize_judgement=normalize_judgement,
        )
        if metric_name == "functional_consistency"
        else []
    )
    for relation_result in relation_results:
        episode = relation_result.get("camera_acquisition_episode")
        episode = episode if isinstance(episode, dict) else {}
        episode_before = episode.get("ledger_before_judge")
        episode_after = episode.get("ledger_after_judge")
        if isinstance(episode_before, dict) and isinstance(
            episode_after,
            dict,
        ):
            base["camera_acquisition_ledger"] = (
                merge_acquisition_ledger_delta(
                    base.get("camera_acquisition_ledger"),
                    episode_before=episode_before,
                    episode_after=episode_after,
                )
            )
    relation_claims: list[dict[str, Any]] = []
    relation_claim_ids: set[str] = set()
    for relation_result in relation_results:
        if relation_result.get("score") != 0.0:
            continue
        for defect in (
            (relation_result.get("judgement") or {}).get("defects")
            or []
        ):
            if not isinstance(defect, dict):
                continue
            claim = claim_record(
                metric_name,
                defect,
                source_phase=(
                    "cross_group_relation_review:"
                    f"{relation_result.get('relation_id')}"
                ),
                claim_status="final",
            )
            claim_id = str(claim.get("claim_id") or "")
            if claim_id in relation_claim_ids:
                continue
            relation_claim_ids.add(claim_id)
            relation_claims.append(claim)
    relation_phase_complete = all(
        item.get("status") == "evaluated"
        for item in relation_results
    )
    base["cross_group_relation_results"] = deepcopy(
        relation_results
    )
    base["cross_group_relation_claims"] = deepcopy(
        relation_claims
    )
    base["cross_group_relation_phase"] = {
        "required": bool(cross_group_relation_specs),
        "scheduled_relation_count": len(
            cross_group_relation_specs
        ),
        "judge_eligible_relation_count": len(
            [
                item
                for item in cross_group_relation_specs
                if item.get("pair_specific_evidence_available") is True
            ]
        ),
        "skipped_missing_pair_evidence_count": len(
            [
                item
                for item in cross_group_relation_specs
                if item.get("pair_specific_evidence_available") is not True
            ]
        ),
        "resolved_relation_count": len(
            [
                item
                for item in relation_results
                if item.get("status") == "evaluated"
            ]
        ),
        "status": (
            "not_required_no_discovered_cross_group_relation"
            if not cross_group_relation_specs
            else "complete"
            if relation_phase_complete
            else "unresolved"
        ),
        "max_probe_units": functional_probe_budget,
    }
    upstream_claims = [*scene_claims, *relation_claims]

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
    if metric_name == "functional_consistency":
        forced_group_ids.update(
            forced_group_ids_from_checks(
                base.get("functional_check_ledger")
                if isinstance(
                    base.get("functional_check_ledger"),
                    dict,
                )
                else None
            )
        )
    if (
        metric_name == "semantic_placement_consistency"
        and isinstance(base.get("placement_check_ledger"), dict)
    ):
        forced_group_ids.update(
            forced_group_ids_from_placement_checks(
                base["placement_check_ledger"],
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
                placement_camera_targets_by_group(
                    base.get("placement_check_ledger"),
                )
                if metric_name
                == "semantic_placement_consistency"
                else None
            ),
        )
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
            required_checks = (
                checks_for_group(
                    base.get("functional_check_ledger"),
                    group_id,
                )
                if metric_name == "functional_consistency"
                and isinstance(
                    base.get("functional_check_ledger"),
                    dict,
                )
                else []
            )
            if group_id in group_probe_packets or required_checks:
                functional_packet = deepcopy(
                    group_probe_packets.get(group_id)
                    or {
                        "schema_version": (
                            FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION
                        ),
                        "planning_role": (
                            "visual_evidence_only_no_metric_verdict"
                        ),
                        "probe_inclusion_is_invalidity_prior": False,
                        "group_id": group_id,
                        "observation_requests": [],
                        "image_order": [],
                        "decision_authority": "none",
                    }
                )
                if required_checks:
                    functional_packet["required_checks"] = deepcopy(
                        required_checks
                    )
                    functional_packet["required_check_ids"] = [
                        str(item["check_id"])
                        for item in required_checks
                    ]
                    functional_packet["required_check_count"] = len(
                        required_checks
                    )
                    functional_packet["observation_complete"] = False
                    functional_packet["coverage_complete"] = False
                    functional_packet["coverage_semantics"] = (
                        "acquisition_and_required_check_resolution"
                    )
                    _bind_architecture_orientation_evidence(
                        functional_packet,
                        packet_paths=list(packet.get("paths") or []),
                        angled_global_paths=local_global_context,
                    )
                packet["functional_probe_evidence"] = functional_packet
            if (
                metric_name == "semantic_placement_consistency"
                and isinstance(base.get("placement_check_ledger"), dict)
            ):
                required_placement_checks = placement_checks_for_group(
                    base["placement_check_ledger"],
                    group_id,
                )
                packet["placement_discovery"] = deepcopy(
                    base.get("placement_discovery")
                )
                packet["required_placement_checks"] = (
                    required_placement_checks
                )
                packet["functional_ownership_ledger"] = deepcopy(
                    functional_ownership_ledger
                )
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
    functional_check_coverage: dict[str, Any] | None = None
    if (
        metric_name == "functional_consistency"
        and isinstance(result.get("functional_check_ledger"), dict)
    ):
        (
            result["functional_check_ledger"],
            functional_check_coverage,
        ) = apply_functional_check_judgements(
            result["functional_check_ledger"],
            relation_results=relation_results,
            group_results=group_results,
        )
        result["functional_check_coverage"] = deepcopy(
            functional_check_coverage
        )
    placement_check_coverage: dict[str, Any] | None = None
    if (
        metric_name == "semantic_placement_consistency"
        and isinstance(result.get("placement_check_ledger"), dict)
    ):
        (
            result["placement_check_ledger"],
            placement_check_coverage,
        ) = apply_placement_check_judgements(
            result["placement_check_ledger"],
            global_record=global_record,
            group_results=group_results,
        )
        result["placement_check_coverage"] = deepcopy(
            placement_check_coverage
        )
        result["cross_metric_ownership_audit"] = (
            build_cross_metric_ownership_audit(
                functional_ownership_ledger=(
                    functional_ownership_ledger
                ),
                placement_check_ledger=result[
                    "placement_check_ledger"
                ],
            )
        )
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
                upstream_claims,
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
        1
        for item in relation_results
        if item.get("vlm_invoked")
    ) + sum(
        1 for item in group_results if item.get("vlm_invoked")
    )
    result["vlm_invoked"] = True
    result["evidence_request"]["vlm_invoked"] = True
    result["global_discovery"] = deepcopy(global_record)
    result["global_scene_claims"] = deepcopy(scene_claims)
    result["route"] = (
        "global_then_cross_group_relations_then_group_local"
        if metric_name == "functional_consistency"
        else "global_discovery_then_forced_group_local"
    )
    result["aggregation_policy"] = (
        "invalid_if_global_relation_or_group_scope_invalid"
        if metric_name == "functional_consistency"
        else "invalid_if_scene_global_or_any_eligible_group_invalid"
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
    functional_check_phase_complete = bool(
        metric_name != "functional_consistency"
        or functional_check_coverage is None
        or functional_check_coverage.get("complete")
    )
    if metric_name == "functional_consistency":
        result["functional_check_phase"] = {
            "required": bool(
                (functional_check_coverage or {}).get(
                    "required_check_count"
                )
            ),
            "status": (
                "complete"
                if functional_check_phase_complete
                else "unresolved"
            ),
            **deepcopy(functional_check_coverage or {}),
        }
    placement_check_phase_complete = bool(
        metric_name != "semantic_placement_consistency"
        or placement_check_coverage is None
        or placement_check_coverage.get("complete")
    )
    if metric_name == "semantic_placement_consistency":
        result["placement_check_phase"] = {
            "required": bool(
                (placement_check_coverage or {}).get(
                    "required_check_count"
                )
            ),
            "status": (
                "complete"
                if placement_check_phase_complete
                else "unresolved"
            ),
            **deepcopy(placement_check_coverage or {}),
        }
    controller_audits = [
        audit
        for audit in [
            global_audit,
            *[
                item.get("camera_control_audit")
                for item in relation_results
                if isinstance(item, dict)
            ],
            *[
                item.get("camera_control_audit")
                for item in group_results
                if isinstance(item, dict)
            ],
        ]
        if isinstance(audit, dict)
    ]
    result["judge_triggered_camera_stage"] = (
        _judge_triggered_stage_telemetry(controller_audits)
    )
    result["combined_evidence_budget"] = {
        "accounting": (
            "per_judge_episode_limit_with_metric_aggregate_audit"
        ),
        "budget_enforcement_scope": "judge_episode",
        "episode_seed_counting": "judge_facing_evidence",
        "judge_triggered_render_counting": "physical_artifacts",
        "max_images_per_judge_episode": _resolved_total_image_budget(
            vlm_judge
        ),
        "camera_acquisition_ledger": deepcopy(
            result.get("camera_acquisition_ledger")
        ),
        "metric_aggregate_counting": "physical_artifacts",
        "metric_aggregate_is_budget_authority": False,
        "group_iteration_order": [
            str(group.get("group_id") or "")
            for group in eligible_groups
        ],
        "cross_group_relation_iteration_order": [
            str(item.get("relation_id") or "")
            for item in relation_results
        ],
    }
    result["combined"] = {
        "total_budget_accounting": deepcopy(
            result["combined_evidence_budget"]
        )
    }
    aggregated = _aggregate_global_and_group_results(
        result,
        metric_name=metric_name,
        global_record=global_record,
        global_outcome=global_outcome,
        scene_claims=scene_claims,
        relation_claims=relation_claims,
        relation_results=relation_results,
        relation_phase_complete=relation_phase_complete,
        group_results=group_results,
        group_phase_required=group_phase_required,
        group_phase_complete=group_phase_complete,
        functional_check_phase_complete=(
            functional_check_phase_complete
        ),
        placement_check_phase_complete=placement_check_phase_complete,
    )
    aggregated = _apply_functional_acquisition_budget_status(
        aggregated,
        metric_name=metric_name,
    )
    if metric_name == "functional_consistency":
        resolved = aggregated.get("status") == "evaluated"
        aggregated["functional_ownership_ledger"] = (
            build_functional_ownership_ledger(
                scene_object_ids=object_ids,
                global_record=(global_record if resolved else None),
                relation_results=(relation_results if resolved else []),
                group_results=(group_results if resolved else []),
                functional_check_ledger=(
                    result.get("functional_check_ledger")
                    if resolved
                    and isinstance(
                        result.get("functional_check_ledger"),
                        dict,
                    )
                    else None
                ),
            )
        )
    return aggregated


def _judge_triggered_stage_telemetry(
    audit_records: list[dict[str, Any]],
) -> dict[str, int]:
    """Summarize Controller work without re-counting initial evidence.

    Controller experiment telemetry covers acquisitions triggered after a
    Judge asks for more evidence. Each Judge episode enforces its own camera
    budget; the metric ledger only aggregates cost and audit telemetry.
    """

    telemetry_records: list[dict[str, Any]] = []
    for record in audit_records:
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
        telemetry_records.append(telemetry)
    return {
        "selector_calls": sum(
            int(item.get("deterministic_selector_calls") or 0)
            + int(item.get("vlm_selector_calls") or 0)
            for item in telemetry_records
        ),
        "evidence_rounds": sum(
            int(item.get("deterministic_rounds") or 0)
            + int(item.get("vlm_rounds") or 0)
            for item in telemetry_records
        ),
        "camera_actions": sum(
            int(item.get("selected_view_count") or 0)
            for item in telemetry_records
        ),
        "preview_render_count": sum(
            int(item.get("preview_render_count") or 0)
            for item in telemetry_records
        ),
        "full_render_count": sum(
            int(item.get("full_render_count") or 0)
            for item in telemetry_records
        ),
        "judge_facing_image_count": sum(
            _judge_facing_acquired_image_count(record)
            for record in audit_records
        ),
    }


def _judge_facing_acquired_image_count(
    record: dict[str, Any],
) -> int:
    audit = (
        record.get("audit")
        if isinstance(record.get("audit"), dict)
        else record
    )
    trace = (
        audit.get("trace")
        if isinstance(audit, dict)
        and isinstance(audit.get("trace"), list)
        else []
    )
    return sum(
        len((item.get("result") or {}).get("visual_evidence") or [])
        for item in trace
        if isinstance(item, dict)
        and item.get("stage") == "render"
        and item.get("status") == "completed"
        and isinstance(item.get("result"), dict)
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
    camera_acquisition_ledger: dict[str, Any],
    forbidden_cross_group_target_sets: list[tuple[str, ...]],
    required_placement_checks: list[dict[str, Any]],
    functional_ownership_ledger: dict[str, Any] | None,
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
        required_placement_checks=required_placement_checks,
        functional_ownership_ledger=functional_ownership_ledger,
    )
    request["camera_acquisition_ledger"] = deepcopy(
        camera_acquisition_ledger
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
        if required_placement_checks:
            adjusted = canonicalize_typed_invalid_envelope(adjusted)
        placement_resolution = None
        if metric_name == "semantic_placement_consistency":
            registered_checks = (
                _registered_placement_checks_from_controller_audit(
                    audit_records,
                    audit_start=audit_start,
                )
            )
            if registered_checks:
                base["placement_check_ledger"] = (
                    merge_placement_checks(
                        base["placement_check_ledger"],
                        registered_checks,
                    )
                )
            current_stage_registered_checks = [
                item
                for item in registered_checks
                if item.get("owner_stage") == "scene_global"
                and item.get("handoff_status")
                != "deferred_to_group_local"
            ]
            adjusted, judge_originated_checks = (
                normalize_judge_originated_placement_results(
                    adjusted,
                    known_ids=set(object_ids),
                    groups=groups,
                    existing_checks=list(
                        (
                            base.get("placement_check_ledger") or {}
                        ).get("checks")
                        or []
                    ),
                    expected_owner_stage="scene_global",
                )
            )
            if judge_originated_checks:
                base["placement_check_ledger"] = (
                    merge_placement_checks(
                        base["placement_check_ledger"],
                        judge_originated_checks,
                    )
                )
            phase_check_ids = {
                str(item.get("check_id") or "")
                for item in [
                    *required_placement_checks,
                    *current_stage_registered_checks,
                    *judge_originated_checks,
                ]
                if isinstance(item, dict) and item.get("check_id")
            }
            phase_checks = [
                deepcopy(check)
                for check in (
                    base.get("placement_check_ledger") or {}
                ).get("checks")
                or []
                if str(check.get("check_id") or "")
                in phase_check_ids
            ]
            adjusted = canonicalize_placement_defect_linkage(
                adjusted,
                required_checks=phase_checks,
            )
            placement_resolution = validate_placement_check_results(
                adjusted,
                required_checks=phase_checks,
                function_events=list(
                    (functional_ownership_ledger or {}).get("events")
                    or []
                ),
            )
        outcome = normalize_judgement(
            adjusted,
            metric_name=metric_name,
            valid_object_ids=set(object_ids),
        )
        if metric_name == "functional_consistency":
            phase_violations = (
                _forbidden_cross_group_defects(
                    adjusted.get("defects") or [],
                    forbidden_target_sets=(
                        forbidden_cross_group_target_sets
                    ),
                )
            )
            if phase_violations:
                raise ValueError(
                    "scene-global functional Judge returned a relation "
                    "owned by the cross-group relation stage: "
                    f"{phase_violations}"
                )
        record = deepcopy(adjusted)
        if placement_resolution is not None:
            record["placement_check_resolution"] = deepcopy(
                placement_resolution
            )
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
        if (
            metric_name == "semantic_placement_consistency"
            and isinstance(base.get("placement_check_ledger"), dict)
        ):
            (
                base["placement_check_ledger"],
                _,
            ) = apply_placement_check_judgements(
                base["placement_check_ledger"],
                global_record=record,
                group_results=[],
            )
    except Exception as exc:
        schema_audit = response_schema_audit_from_exception(exc)
        record = {
            "scope_level": "scene_global",
            "decision_role": "required_scene_scope_unresolved",
            "final_metric_verdict": False,
            "global_status": "failed",
            "does_not_short_circuit_group_review": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            **(
                {"response_schema_audit": schema_audit}
                if schema_audit is not None
                else {}
            ),
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


def _registered_placement_checks_from_controller_audit(
    audit_records: Any,
    *,
    audit_start: int | None,
) -> list[dict[str, Any]]:
    if (
        audit_start is None
        or not isinstance(audit_records, list)
        or len(audit_records) <= audit_start
    ):
        return []
    latest = audit_records[-1]
    payload = (
        latest.get("audit")
        if isinstance(latest, dict)
        and isinstance(latest.get("audit"), dict)
        else latest
    )
    request = (
        payload.get("judge_request")
        if isinstance(payload, dict)
        and isinstance(payload.get("judge_request"), dict)
        else {}
    )
    context = (
        request.get("context")
        if isinstance(request.get("context"), dict)
        else {}
    )
    checks = [
        *(
            context.get("required_placement_checks")
            if isinstance(
                context.get("required_placement_checks"), list
            )
            else []
        ),
        *(
            context.get("deferred_placement_checks")
            if isinstance(
                context.get("deferred_placement_checks"), list
            )
            else []
        ),
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for item in checks:
        if not isinstance(item, dict):
            raise TypeError(
                "controller placement check handoff must contain objects"
            )
        check_id = str(item.get("check_id") or "").strip()
        if not check_id:
            raise ValueError(
                "controller placement check handoff requires check IDs"
            )
        prior = by_id.get(check_id)
        if prior is not None and prior != item:
            raise ValueError(
                "controller placement check handoff contains conflicting "
                f"records for {check_id!r}"
            )
        by_id[check_id] = deepcopy(item)
    return [by_id[check_id] for check_id in sorted(by_id)]


def _placement_checks_for_result(
    result: dict[str, Any],
    *,
    ledger: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    returned_ids = {
        str(item.get("check_id") or "")
        for item in result.get("placement_check_results") or []
        if isinstance(item, dict) and item.get("check_id")
    }
    checks = [
        deepcopy(check)
        for check in (ledger or {}).get("checks") or []
        if isinstance(check, dict)
        and str(check.get("check_id") or "") in returned_ids
    ]
    if len(checks) != len(returned_ids):
        missing = sorted(
            returned_ids
            - {
                str(check.get("check_id") or "")
                for check in checks
            }
        )
        raise ValueError(
            f"placement Judge returned unknown required checks: {missing}"
        )
    return checks


def _functional_check_ledger_from_audit(
    acquisition_audit: dict[str, Any],
    *,
    groups: list[dict[str, Any]],
    scene: dict[str, Any] | None = None,
) -> dict[str, Any]:
    direct = acquisition_audit.get("functional_check_ledger")
    if isinstance(direct, dict):
        return deepcopy(direct)
    plan = acquisition_audit.get("functional_acquisition_plan")
    if isinstance(plan, dict) and isinstance(
        plan.get("functional_check_ledger"),
        dict,
    ):
        return deepcopy(plan["functional_check_ledger"])
    discovery = acquisition_audit.get("functional_discovery")
    if isinstance(discovery, dict):
        return build_functional_check_ledger(
            discovery,
            groups=groups,
            scene=scene,
        )
    return {
        "schema_version": FUNCTIONAL_CHECK_LEDGER_VERSION,
        "checks": [],
        "accepted_check_count": 0,
        "decision_authority": "none",
    }


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


def _functional_probe_budget(
    plan: dict[str, Any],
    *,
    judge: Any,
    global_image_count: int,
    provider: Any,
) -> int:
    configured = _configured_functional_probe_units(plan)
    # The proactive units are routed across isolated cross-group and
    # group-local Judge episodes. Per-episode image limits remain enforced by
    # each Judge packet and Controller; they are not a metric-wide authority
    # over how many distinct obligations may receive one probe.
    _ = (judge, global_image_count, provider)
    return configured


def _configured_functional_probe_units(
    plan: dict[str, Any],
) -> int:
    policy = (
        plan.get("prejudgement_probe_policy")
        if isinstance(plan.get("prejudgement_probe_policy"), dict)
        else {}
    )
    value = policy.get("max_probe_units")
    requested = max(
        0,
        int(
            FUNCTIONAL_PROBE_DEFAULT_UNITS
            if value is None
            else value
        ),
    )
    return min(requested, FUNCTIONAL_PROBE_MAX_UNITS)


def _functional_prejudgement_should_run(
    config: dict[str, Any] | None,
    *,
    planner: Any,
    provider: Any,
    injected_source: Any,
) -> bool:
    if injected_source is not None:
        return True
    mode = str((config or {}).get("mode") or "runtime").strip()
    if mode in {"disabled", "frozen", "precomputed"}:
        return True
    # Preserve the pre-refactor default: the runtime proactive stage was
    # skipped unless both existing dependencies were configured.
    return planner is not None and provider is not None


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


def _bind_architecture_orientation_evidence(
    functional_packet: dict[str, Any],
    *,
    packet_paths: list[str],
    angled_global_paths: list[str],
) -> None:
    """Bind each orientation check to one global and its reused local view."""

    checks = [
        item
        for item in functional_packet.get("required_checks") or []
        if isinstance(item, dict)
    ]
    orientation_checks = [
        item
        for item in checks
        if item.get("check_type") == "architecture_orientation"
    ]
    if not orientation_checks:
        return
    normalized_paths = list(
        dict.fromkeys(str(path) for path in packet_paths if str(path))
    )
    aliases = {
        path: f"image_{index:02d}"
        for index, path in enumerate(normalized_paths)
    }
    global_path = next(
        (
            str(path)
            for path in angled_global_paths
            if str(path) in aliases
        ),
        normalized_paths[0] if normalized_paths else None,
    )
    image_records = [
        item
        for item in functional_packet.get("image_order") or []
        if isinstance(item, dict)
    ]
    bindings: list[dict[str, Any]] = []
    for check in orientation_checks:
        check_id = str(check.get("check_id") or "")
        target_ids = {
            str(item) for item in check.get("target_ids") or []
        }
        side_path = next(
            (
                str(item.get("artifact_id"))
                for item in image_records
                if check_id
                in {
                    str(value)
                    for value in item.get("check_ids") or []
                }
                and str(item.get("artifact_id") or "") in aliases
                and str(item.get("artifact_id") or "") != global_path
            ),
            None,
        )
        if side_path is None:
            side_path = next(
                (
                    str(path)
                    for path in check.get("evidence_refs") or []
                    if str(path) in aliases and str(path) != global_path
                ),
                None,
            )
        shared_clearance_ids = [
            str(item.get("check_id") or "")
            for item in checks
            if item.get("check_type") == "clearance"
            and {
                str(value) for value in item.get("target_ids") or []
            }
            == target_ids
        ]
        bindings.append(
            {
                "check_id": check_id,
                "target_ids": sorted(target_ids),
                "angled_global_image_alias": (
                    aliases.get(global_path) if global_path else None
                ),
                "side_conditioned_local_image_alias": (
                    aliases.get(side_path) if side_path else None
                ),
                "reused_by_clearance_check_ids": shared_clearance_ids,
                "status": (
                    "complete"
                    if global_path and side_path
                    else "pending_more_evidence"
                ),
            }
        )
    policy = (
        functional_packet.get("architecture_orientation_policy")
        if isinstance(
            functional_packet.get("architecture_orientation_policy"),
            dict,
        )
        else {}
    )
    functional_packet["architecture_orientation_policy"] = {
        **deepcopy(policy),
        "global_anchor_policy": "exactly_one_angled_global",
        "local_view_policy": "reuse_side_conditioned_probe",
        "evidence_bindings": bindings,
        "need_more_evidence_loop": "controller_managed",
    }


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
    relation_claims: list[dict[str, Any]],
    relation_results: list[dict[str, Any]],
    relation_phase_complete: bool,
    group_results: list[dict[str, Any]],
    group_phase_required: bool,
    group_phase_complete: bool,
    functional_check_phase_complete: bool,
    placement_check_phase_complete: bool,
) -> dict[str, Any]:
    evaluated_groups = [
        item
        for item in group_results
        if item.get("status") == "evaluated"
    ]
    invalid_groups = [
        item for item in evaluated_groups if item.get("score") == 0.0
    ]
    evaluated_relations = [
        item
        for item in relation_results
        if item.get("status") == "evaluated"
    ]
    invalid_relations = [
        item
        for item in evaluated_relations
        if item.get("score") == 0.0
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
    relation_defects = [
        deepcopy(defect)
        for item in invalid_relations
        for defect in (
            (item.get("judgement") or {}).get("defects") or []
        )
        if isinstance(defect, dict)
    ]
    defects = deduplicate_defects(
        metric_name,
        [*global_defects, *relation_defects, *local_defects],
    )
    placement_summary = (
        placement_severity_summary(defects)
        if metric_name == "semantic_placement_consistency"
        else None
    )
    if placement_summary is not None:
        base["placement_severity"] = deepcopy(placement_summary)
    object_findings = object_level_finding_records(
        metric_name,
        [
            *[
                ("global_discovery", defect)
                for defect in global_defects
            ],
            *[
                (
                    "cross_group_relation_review:"
                    f"{item.get('relation_id')}",
                    defect,
                )
                for item in invalid_relations
                for defect in (
                    (item.get("judgement") or {}).get("defects") or []
                )
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

    final_claims = [*scene_claims, *relation_claims]
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
        "cross_group_relation_judgement:"
        f"{item.get('relation_id')}"
        for item in relation_results
        if item.get("status") != "evaluated"
    )
    missing_evidence.extend(
        f"group_local_judgement:{item.get('group_id')}"
        for item in group_results
        if item.get("status") != "evaluated"
    )
    if group_phase_required and not group_results:
        missing_evidence.append("eligible_group_partition")
    functional_check_coverage = (
        base.get("functional_check_coverage")
        if isinstance(base.get("functional_check_coverage"), dict)
        else {}
    )
    missing_evidence.extend(
        f"functional_check:{check_id}"
        for check_id in (
            functional_check_coverage.get("unresolved_check_ids") or []
        )
    )
    placement_check_coverage = (
        base.get("placement_check_coverage")
        if isinstance(base.get("placement_check_coverage"), dict)
        else {}
    )
    missing_evidence.extend(
        f"placement_check:{check_id}"
        for check_id in (
            placement_check_coverage.get("unresolved_check_ids") or []
        )
    )

    required_group_units = (
        len(group_results)
        if group_results
        else 1
        if group_phase_required
        else 0
    )
    eligible_count = (
        1 + len(relation_results) + required_group_units
    )
    resolved_count = (
        int(global_evaluated)
        + len(evaluated_relations)
        + len(evaluated_groups)
    )
    coverage_complete = bool(
        global_evaluated
        and relation_phase_complete
        and group_phase_complete
        and functional_check_phase_complete
        and placement_check_phase_complete
    )
    base["coverage"] = {
        "eligible_count": eligible_count,
        "resolved_count": resolved_count,
        "fraction": (
            resolved_count / eligible_count if eligible_count else None
        ),
        "complete": coverage_complete,
        "scene_global_resolved": global_evaluated,
        "cross_group_relation_phase_complete": (
            relation_phase_complete
        ),
        "scheduled_cross_group_relation_count": len(
            relation_results
        ),
        "group_phase_complete": group_phase_complete,
        "functional_check_phase_complete": (
            functional_check_phase_complete
        ),
        "placement_check_phase_complete": (
            placement_check_phase_complete
        ),
    }

    if global_invalid or invalid_relations or invalid_groups:
        invalid_judgements: list[dict[str, Any]] = []
        if global_invalid:
            invalid_judgements.append(global_record)
        invalid_judgements.extend(
            item.get("judgement") or {}
            for item in invalid_relations
        )
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
                "At least one scene-global, cross-group relation, or eligible "
                "group-local scope has a significant in-scope defect."
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
                "invalid_if_global_relation_or_group_scope_invalid"
            ),
            **(
                {"placement_severity": deepcopy(placement_summary)}
                if placement_summary is not None
                else {}
            ),
            "scene_global_judgement": deepcopy(global_record),
            "cross_group_relation_judgements": deepcopy(
                relation_results
            ),
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
    all_relations_valid = all(
        item.get("score") == 1.0
        for item in evaluated_relations
    )
    if (
        global_valid
        and relation_phase_complete
        and all_relations_valid
        and group_phase_complete
        and all_groups_valid
        and functional_check_phase_complete
        and placement_check_phase_complete
    ):
        confidence_values = [
            float(global_record.get("confidence") or 0.0),
            *[
                float(
                    (item.get("judgement") or {}).get("confidence")
                    or 0.0
                )
                for item in evaluated_relations
            ],
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
                    "The scene-global scope, every routed cross-group "
                    "relation, and every eligible or explicitly routed group "
                    "resolved without an in-scope defect."
                ),
                "missing_evidence": [],
                "defects": [],
                "object_findings": [],
                "object_penalty_count": 0,
                "object_penalty_policy": (
                    "one_per_metric_object_across_global_and_local"
                ),
                "aggregation": (
                    "global_relations_and_groups_must_resolve_valid"
                ),
                **(
                    {"placement_severity": deepcopy(placement_summary)}
                    if placement_summary is not None
                    else {}
                ),
                "scene_global_judgement": deepcopy(global_record),
                "cross_group_relation_judgements": deepcopy(
                    relation_results
                ),
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
                "scene-global, cross-group relation, or group-local scopes "
                "remain unresolved."
            ),
            "missing_evidence": missing_evidence,
            "defects": [],
            "object_findings": [],
            "object_penalty_count": 0,
            "object_penalty_policy": (
                "one_per_metric_object_across_global_and_local"
            ),
            "aggregation": (
                "unresolved_without_complete_global_relation_group_coverage"
            ),
            **(
                {"placement_severity": deepcopy(placement_summary)}
                if placement_summary is not None
                else {}
            ),
            "scene_global_judgement": deepcopy(global_record),
            "cross_group_relation_judgements": deepcopy(
                relation_results
            ),
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
