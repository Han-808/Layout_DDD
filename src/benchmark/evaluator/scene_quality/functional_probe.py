"""Acquire bounded pre-judgement evidence for global functional review."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.architecture_policy import architecture_contract_from_scene
from benchmark.evaluator.scene_quality.functional_acquisition import (
    FUNCTIONAL_ACQUISITION_PLAN_VERSION,
    build_functional_acquisition_plan,
)
from benchmark.evaluator.scene_quality.functional_boundary_evidence import (
    FUNCTIONAL_BOUNDARY_EVIDENCE_VERSION,
    acquire_functional_boundary_evidence,
    boundary_evidence_for_targets,
    discovery_with_boundary_hypotheses,
)
from benchmark.evaluator.scene_quality.functional_checks import (
    FUNCTIONAL_CHECK_LEDGER_VERSION,
    checks_for_group,
    forced_group_ids_from_checks,
    update_functional_check_evidence,
)
from benchmark.evaluator.scene_quality.functional_measurements import (
    compact_functional_measurements_for_checks,
)
from benchmark.evaluator.scene_quality.functional_planner_adapter import (
    FUNCTIONAL_DISCOVERY_PLANNER_MODE,
    FunctionalEvidencePlannerAdapter,
    LEGACY_FUNCTIONAL_PROBE_PLANNER_MODE,
)
from benchmark.evaluator.scene_quality.terminal import (
    recoverable_validation_failure,
)
from benchmark.rendering.camera_pose import (
    FUNCTIONAL_PROBE_CANDIDATE_BUDGETS,
)
from benchmark.visual_judge.functional_discovery import (
    FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
)
from benchmark.visual_judge.functional_evidence import (
    FUNCTIONAL_PROBE_DEFAULT_UNITS,
    FUNCTIONAL_PROBE_MAX_UNITS,
    FUNCTIONAL_PROBE_PLAN_VERSION,
)


FUNCTIONAL_PROBE_ACQUISITION_VERSION = (
    "functional_probe_acquisition_v5"
)
FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION = (
    "functional_probe_judge_packet_v6"
)


def acquire_functional_probe_evidence(
    *,
    planner: Any,
    provider: Any,
    scene: dict[str, Any],
    global_image_path: str,
    max_probe_units: int = FUNCTIONAL_PROBE_DEFAULT_UNITS,
    groups: list[dict[str, Any]] | None = None,
    grouping_report: dict[str, Any] | None = None,
    identity_image_path: str | None = None,
    identity_legend: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Plan and render raw probe images without issuing a metric verdict."""

    base_evidence_available = Path(global_image_path).expanduser().is_file()

    audit: dict[str, Any] = {
        "schema_version": FUNCTIONAL_PROBE_ACQUISITION_VERSION,
        "plan_schema_version": FUNCTIONAL_PROBE_PLAN_VERSION,
        "discovery_schema_version": FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
        "acquisition_plan_schema_version": (
            FUNCTIONAL_ACQUISITION_PLAN_VERSION
        ),
        "status": "not_configured",
        "decision_authority": "none",
        "planner_role": "visual_evidence_only",
        "source_global_image": str(global_image_path),
        "max_probe_units": int(max_probe_units),
        "probe_units": [],
        "probe_results": [],
        "selected_raw_rgb_paths": [],
        "cross_group_evidence_paths": [],
        "group_evidence_paths": {},
        "group_probe_packets": {},
        "forced_group_ids": [],
        "functional_boundary_evidence": {
            "schema_version": FUNCTIONAL_BOUNDARY_EVIDENCE_VERSION,
            "status": "not_applicable",
            "decision_authority": "none",
            "scene_access": "read_only",
            "reason": "functional_discovery_not_run",
        },
        "judge_image_policy": (
            "global_receives_cross_group_only_groups_receive_owned_probes"
        ),
        "source_scene_modified": False,
    }
    planner_adapter = FunctionalEvidencePlannerAdapter(planner)
    if not planner_adapter.configured:
        audit["reason"] = "functional_evidence_planner_not_configured"
        return [], audit
    provider_call = getattr(
        provider,
        "provide_scene_quality_evidence",
        None,
    )
    if not callable(provider_call) and callable(provider):
        provider_call = provider

    minimal_objects = _minimal_object_list(scene)
    architecture_context = _functional_architecture_context(scene)
    planning_request = {
        "metric": "functional_consistency",
        "scene_id": scene.get("scene_id"),
        "scene_type": scene.get("scene_type"),
        "global_image_path": str(global_image_path),
        "architecture_context": deepcopy(architecture_context),
        "objects": minimal_objects,
        "groups": _minimal_group_list(groups),
        "max_probe_units": int(max_probe_units),
        "identity_image_path": identity_image_path,
        "identity_legend": deepcopy(identity_legend or {}),
    }
    audit["planner_input"] = {
        "metric": "functional_consistency",
        "scene_id": scene.get("scene_id"),
        "scene_type": scene.get("scene_type"),
        "global_image_alias": "scene_global",
        "architecture_context": deepcopy(architecture_context),
        "objects": deepcopy(minimal_objects),
        "groups": _minimal_group_list(groups),
        "max_probe_units": int(max_probe_units),
        "identity_grounding": {
            "image_path": identity_image_path,
            "legend": deepcopy(identity_legend or {}),
        },
        "excluded_fields": [
            "center",
            "size",
            "rotation",
            "description",
            "grouping_reason",
        ],
    }
    boundary_evidence: dict[str, Any] | None = None

    def _build_plan_from_discovery(
        discovery: Any,
    ) -> dict[str, Any]:
        nonlocal boundary_evidence
        boundary_evidence = acquire_functional_boundary_evidence(
            provider=provider,
            scene=scene,
            discovery=discovery,
            architecture_context=architecture_context,
        )
        return build_functional_acquisition_plan(
            discovery_with_boundary_hypotheses(
                discovery,
                boundary_evidence,
            ),
            max_probe_units=max_probe_units,
            groups=groups,
            scene=scene,
        )

    try:
        planner_execution = planner_adapter.execute(
            planning_request,
            build_plan_from_discovery=_build_plan_from_discovery,
        )
        plan = planner_execution.plan
        discovery = planner_execution.discovery
        if (
            planner_execution.mode
            == FUNCTIONAL_DISCOVERY_PLANNER_MODE
        ):
            audit["functional_boundary_evidence"] = deepcopy(
                boundary_evidence
            )
            audit["functional_discovery"] = deepcopy(discovery)
            discovery_coverage = (
                discovery.get("coverage")
                if isinstance(discovery, dict)
                and isinstance(discovery.get("coverage"), dict)
                else None
            )
            if discovery_coverage is not None:
                # Bubble the frozen per-object/relation obligations up to the
                # acquisition component consumed by metric scoring. Keeping
                # coverage only inside ``functional_discovery`` would make a
                # salvaged bad row invisible to the final projection.
                audit["coverage"] = deepcopy(discovery_coverage)
            audit["functional_acquisition_plan"] = deepcopy(plan)
            if isinstance(plan, dict):
                audit["functional_check_ledger"] = deepcopy(
                    plan.get("functional_check_ledger")
                )
                audit["functional_measurement_bank"] = deepcopy(
                    plan.get("functional_measurement_bank")
                )
        audit["planner_mode"] = planner_execution.mode
    except Exception as exc:
        recoverable = bool(
            base_evidence_available
            and recoverable_validation_failure(exc)
        )
        audit.update(
            status=(
                "degraded_no_probes" if recoverable else "failed"
            ),
            reason=(
                "functional_probe_planner_item_failure_isolated"
                if recoverable
                else "functional_probe_planner_failed"
            ),
            error_type=type(exc).__name__,
            error=str(exc),
            coverage={
                "unit": "planner_contract_obligation",
                "eligible_count": 1,
                "grounded_count": 0,
                "fraction": 0.0,
                "complete": False,
            },
            fallback={
                "policy": "no_specialized_probes_keep_base_evidence_v1",
                "defaulted_probe_count": 1,
                "base_group_and_global_judges_continue": recoverable,
            },
        )
        schema_audit = getattr(exc, "schema_audit", None)
        if isinstance(schema_audit, dict):
            audit["response_schema_validation"] = deepcopy(
                schema_audit
            )
        return [], audit
    units = (
        plan.get("probe_units")
        if isinstance(plan, dict)
        and isinstance(plan.get("probe_units"), list)
        else []
    )
    backfill_units = (
        plan.get("backfill_probe_units")
        if isinstance(plan, dict)
        and isinstance(plan.get("backfill_probe_units"), list)
        else []
    )
    audit["planner_request_metadata"] = deepcopy(
        planner_execution.request_metadata
    )
    audit["planner_reason"] = planner_execution.reason
    if isinstance(plan, dict):
        audit["group_confirmations"] = deepcopy(
            plan.get("group_confirmations") or []
        )
        audit["object_evidence_policy"] = deepcopy(
            plan.get("object_evidence_policy") or {}
        )
        audit["unscheduled_discovery_items"] = deepcopy(
            plan.get("unscheduled_discovery_items") or []
        )
        audit["coverage_complete"] = bool(
            plan.get("coverage_complete", True)
        )
        audit["budget_exhausted"] = bool(
            plan.get("budget_exhausted", False)
        )
        audit["forced_group_ids"] = list(
            dict.fromkeys(
                [
                    *[
                        str(item.get("owning_group_id"))
                        for item in [
                            *units,
                            *backfill_units,
                            *(
                                plan.get("unscheduled_discovery_items") or []
                            ),
                        ]
                        if isinstance(item, dict)
                        and item.get("route_scope") == "group_local"
                        and item.get("owning_group_id")
                    ],
                    *forced_group_ids_from_checks(
                        plan.get("functional_check_ledger")
                    ),
                ]
            )
        )
    audit["probe_units"] = deepcopy(units)
    audit["backfill_probe_units"] = deepcopy(backfill_units)
    if not units:
        _attach_boundary_evidence_to_group_packets(
            audit,
            groups=groups,
        )
        _attach_required_checks_to_group_packets(audit, groups=groups)
        _summarize_usable_surface_usage(audit)
        audit.update(
            rendered_probe_count=0,
            failed_probe_count=0,
            planned_probe_count=0,
        )
        audit.update(
            status="complete_no_probes",
            reason="planner_identified_no_probe_worthy_functional_units",
        )
        return [], audit
    if not callable(provider_call):
        _attach_boundary_evidence_to_group_packets(
            audit,
            groups=groups,
        )
        _attach_required_checks_to_group_packets(audit, groups=groups)
        _summarize_usable_surface_usage(audit)
        audit.update(
            status=(
                "degraded_no_probes"
                if base_evidence_available
                else "failed"
            ),
            reason="functional_probe_provider_not_configured",
            rendered_probe_count=0,
            failed_probe_count=len(units),
            planned_probe_count=len(units),
            acquisition_coverage={
                "unit": "planned_functional_probe",
                "eligible_count": len(units),
                "grounded_count": 0,
                "fraction": 0.0,
                "complete": False,
            },
            fallback={
                "policy": "no_specialized_probes_keep_base_evidence_v1",
                "defaulted_probe_count": len(units),
                "base_group_and_global_judges_continue": bool(
                    base_evidence_available
                ),
            },
        )
        return [], audit

    categories = {
        str(item["id"]): str(item["category"])
        for item in minimal_objects
    }
    selected_paths: list[str] = []
    failures = 0
    successful_probe_count = 0
    attempted_backfill_count = 0
    backfill_attempt_quota: int | None = None
    failed_discovery_items: list[dict[str, Any]] = []
    candidate_units = [*units, *backfill_units]
    primary_probe_target = len(units)
    for candidate_index, unit in enumerate(candidate_units):
        is_backfill = candidate_index >= len(units)
        if is_backfill:
            if backfill_attempt_quota is None:
                backfill_attempt_quota = max(
                    0,
                    primary_probe_target - successful_probe_count,
                )
            if (
                successful_probe_count >= primary_probe_target
                or attempted_backfill_count >= backfill_attempt_quota
            ):
                break
        if not isinstance(unit, dict):
            failures += 1
            continue
        if is_backfill:
            attempted_backfill_count += 1
            unit_identity = _probe_unit_identity_record(unit)
            audit["unscheduled_discovery_items"] = [
                item
                for item in (
                    audit.get("unscheduled_discovery_items") or []
                )
                if not (
                    isinstance(item, dict)
                    and item.get("acquisition_identity")
                    == unit_identity
                )
            ]
        target_ids = list(
            dict.fromkeys(
                [
                    *[
                        str(item)
                        for item in unit.get("target_ids") or []
                        if str(item).strip()
                    ],
                    *[
                        str(item)
                        for item in unit.get("related_target_ids") or []
                        if str(item).strip()
                    ],
                ]
            )
        )
        scope = (
            "pair_local" if len(target_ids) > 1 else "object_local"
        )
        route_scope = str(
            "cross_group"
            if audit.get("planner_mode")
            == LEGACY_FUNCTIONAL_PROBE_PLANNER_MODE
            else unit.get("route_scope")
            or (
                "cross_group"
                if len(
                    {
                        _group_for_object(
                            groups,
                            object_id,
                        )
                        for object_id in target_ids
                        if _group_for_object(groups, object_id)
                    }
                )
                > 1
                else "group_local"
            )
        )
        owning_group_id = (
            str(unit.get("owning_group_id"))
            if unit.get("owning_group_id")
            else _single_group_for_targets(groups, target_ids)
        )
        owning_group = _group_by_id(groups, owning_group_id)
        group_member_ids = (
            [
                str(item)
                for item in owning_group.get("object_ids") or []
                if str(item).strip()
            ]
            if route_scope == "group_local"
            and isinstance(owning_group, dict)
            else []
        )
        probe = {
            **deepcopy(unit),
            "target_categories": {
                object_id: categories[object_id]
                for object_id in target_ids
                if object_id in categories
            },
            "view_goal": _probe_view_goal(unit),
            "group_id": (
                owning_group_id
                if route_scope == "group_local"
                else None
            ),
            "group_member_ids": group_member_ids,
            "camera_scope_composition": (
                "specific_target_focus_plus_owning_group_context"
                if group_member_ids
                else "specific_cross_group_targets"
            ),
            "logical_boundary_enabled": bool(
                architecture_context.get("logical_boundary_enabled")
            ),
        }
        scene_summary_ids = set(group_member_ids or target_ids)
        provider_request = {
            "category": "functional_probe_evidence_request",
            "metric": "functional_consistency",
            "event": {
                "type": "functional_probe",
                "probe_id": unit.get("probe_id"),
                "probe_kind": unit.get("kind"),
                "object_ids": target_ids,
            },
            "object_ids": target_ids,
            "group_ids": (
                [owning_group_id]
                if route_scope == "group_local"
                and owning_group_id
                else []
            ),
            "object_groups": (
                [deepcopy(owning_group)]
                if route_scope == "group_local"
                and isinstance(owning_group, dict)
                else []
            ),
            "scene": deepcopy(scene),
            "architecture_context": deepcopy(architecture_context),
            "scene_summary": {
                "scene_id": scene.get("scene_id"),
                "scene_type": scene.get("scene_type"),
                "objects": [
                    deepcopy(item)
                    for item in minimal_objects
                    if str(item["id"]) in scene_summary_ids
                ],
            },
            "natural_language_prompt": None,
            "evidence_scope": scope,
            "evidence_policy": {
                "camera_scope": scope,
                "camera_mode": "metric_local",
                "selector": "vlm_selector",
                "image_budget": 1,
                "global_image_budget": 0,
                "scoped_image_budget": 1,
                "presentation": "raw",
                "image_order": ["metric_local"],
                "include_global_context": False,
                "camera_pose_mode": "query_cov",
            },
            "functional_probe": probe,
            "evidence_goal": {
                "probe_kind": unit.get("kind"),
                "required_observations": deepcopy(
                    unit.get("required_observations") or []
                ),
                "view_goal": probe["view_goal"],
                "observation_goals": deepcopy(
                    unit.get("observation_goals")
                    or [probe["view_goal"]]
                ),
            },
            "existing_global_evidence": [],
            "global_context_mode": "not_requested",
            "selection_role": (
                "visual_evidence_only_do_not_judge_metric"
            ),
            "presentation_invariant": {
                "judge_receives": "raw_rgb",
                "overlay_allowed_in_judge_packet": False,
                "scene_pixels_modified": False,
            },
            "functional_acquisition_route": {
                "route_scope": route_scope,
                "owning_group_id": owning_group_id,
                "discovery_ids": deepcopy(
                    unit.get("discovery_ids") or []
                ),
                "acquisition_trigger": unit.get(
                    "acquisition_trigger"
                ),
                "group_member_ids": group_member_ids,
            },
        }
        result_record = {
            "probe_id": unit.get("probe_id"),
            "kind": unit.get("kind"),
            "target_ids": deepcopy(unit.get("target_ids") or []),
            "related_target_ids": deepcopy(
                unit.get("related_target_ids") or []
            ),
            "required_observations": deepcopy(
                unit.get("required_observations") or []
            ),
            "view_goal": probe["view_goal"],
            "evidence_scope": scope,
            "route_scope": route_scope,
            "owning_group_id": owning_group_id,
            "group_member_ids": group_member_ids,
            "discovery_ids": deepcopy(
                unit.get("discovery_ids") or []
            ),
            "check_ids": deepcopy(unit.get("check_ids") or []),
            "functional_measurements": deepcopy(
                unit.get("functional_measurements") or {}
            ),
            "acquisition_trigger": unit.get("acquisition_trigger"),
            "acquisition_triggers": deepcopy(
                unit.get("acquisition_triggers") or []
            ),
            "observation_goals": deepcopy(
                unit.get("observation_goals") or []
            ),
            "observation_kinds": deepcopy(
                unit.get("observation_kinds") or []
            ),
            "relation_predicates": deepcopy(
                unit.get("relation_predicates") or []
            ),
            "surface_targets": deepcopy(
                unit.get("surface_targets") or []
            ),
            "evidence_reuse": deepcopy(
                unit.get("evidence_reuse") or {}
            ),
            "scheduling": {
                "mode": (
                    "deterministic_backfill"
                    if is_backfill
                    else "primary"
                ),
                "candidate_index": candidate_index,
            },
            "camera_policy": {
                "height_policy": "near_interaction_height",
                "elevation_range_degrees": [8.0, 16.0],
                "preferred_lens_mm": 32.0,
                "context_margin_m": 1.25,
                "candidate_count": (
                    FUNCTIONAL_PROBE_CANDIDATE_BUDGETS.get(
                        str(unit.get("kind") or ""),
                        max(FUNCTIONAL_PROBE_CANDIDATE_BUDGETS.values()),
                    )
                ),
                "judge_presentation": "raw_rgb_only",
            },
            "status": "failed",
            "evidence_paths": [],
        }
        try:
            raw = provider_call(provider_request)
            probe_paths = _raw_probe_paths(raw)
            missing = [
                path
                for path in probe_paths
                if not Path(path).expanduser().is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "functional probe provider returned missing paths: "
                    f"{missing}"
                )
            if not probe_paths:
                raise RuntimeError(
                    "functional probe provider returned no raw RGB evidence"
                )
            path = probe_paths[0]
            if path not in selected_paths:
                selected_paths.append(path)
            successful_probe_count += 1
            provider_usage = deepcopy(
                getattr(provider, "last_call_usage", None)
            )
            result_record.update(
                status="available",
                evidence_paths=[path],
                acquired_artifact_paths=list(
                    dict.fromkeys(
                        str(item)
                        for item in (
                            (
                                provider_usage.get(
                                    "acquired_artifact_paths"
                                )
                                if isinstance(provider_usage, dict)
                                else None
                            )
                            or [path]
                        )
                        if str(item).strip()
                    )
                ),
                provider_usage=provider_usage,
                evidence_coverage=deepcopy(
                    (
                        provider_usage.get("functional_evidence_coverage")
                        if isinstance(provider_usage, dict)
                        else None
                    )
                    or {}
                ),
                usable_surface_audit=deepcopy(
                    (
                        getattr(provider, "last_call_usage", None)
                        or {}
                    ).get("usable_surface")
                ),
                functional_geometry=deepcopy(
                    (
                        getattr(provider, "last_call_usage", None)
                        or {}
                    ).get("functional_geometry")
                ),
            )
        except Exception as exc:
            failures += 1
            provider_usage = deepcopy(
                getattr(provider, "last_call_usage", None)
            )
            result_record.update(
                error_type=type(exc).__name__,
                error=str(exc),
                acquired_artifact_paths=list(
                    dict.fromkeys(
                        str(item)
                        for item in (
                            (
                                provider_usage.get(
                                    "acquired_artifact_paths"
                                )
                                if isinstance(provider_usage, dict)
                                else None
                            )
                            or []
                        )
                        if str(item).strip()
                    )
                ),
                provider_usage=provider_usage,
            )
            failed_discovery_items.append(
                {
                    "acquisition_identity": (
                        _probe_unit_identity_record(unit)
                    ),
                    "discovery_ids": deepcopy(
                        unit.get("discovery_ids") or []
                    ),
                    "check_ids": deepcopy(unit.get("check_ids") or []),
                    "target_ids": target_ids,
                    "route_scope": route_scope,
                    "owning_group_id": owning_group_id,
                    "acquisition_trigger": unit.get(
                        "acquisition_trigger"
                    ),
                    "observation_kinds": deepcopy(
                        unit.get("observation_kinds") or []
                    ),
                    "relation_predicates": deepcopy(
                        unit.get("relation_predicates") or []
                    ),
                    "observation_goal": probe["view_goal"],
                    "reason": "probe_acquisition_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        audit["probe_results"].append(result_record)

    audit["failed_discovery_items"] = failed_discovery_items
    audit["acquisition_coverage_complete"] = bool(
        not audit.get("unscheduled_discovery_items")
        and not failed_discovery_items
    )
    audit["coverage_complete"] = audit[
        "acquisition_coverage_complete"
    ]
    audit["coverage_semantics"] = "artifact_acquisition_only"
    audit["budget_exhausted"] = any(
        isinstance(item, dict)
        and item.get("reason") == "max_probe_units_exhausted"
        for item in audit.get("unscheduled_discovery_items") or []
    )
    audit["selected_raw_rgb_paths"] = list(selected_paths)
    audit["acquired_artifact_paths"] = list(
        dict.fromkeys(
            str(path)
            for result in audit["probe_results"]
            if isinstance(result, dict)
            for path in result.get("acquired_artifact_paths") or []
            if str(path).strip()
        )
    )
    cross_group_paths = [
        str(path)
        for result in audit["probe_results"]
        if isinstance(result, dict)
        and result.get("status") == "available"
        and result.get("route_scope") == "cross_group"
        for path in result.get("evidence_paths") or []
    ]
    group_paths: dict[str, list[str]] = {}
    group_packets: dict[str, dict[str, Any]] = {}
    for result in audit["probe_results"]:
        if (
            not isinstance(result, dict)
            or result.get("status") != "available"
            or result.get("route_scope") != "group_local"
            or not result.get("owning_group_id")
        ):
            continue
        group_id = str(result["owning_group_id"])
        group_paths.setdefault(group_id, [])
        for path in result.get("evidence_paths") or []:
            if str(path) not in group_paths[group_id]:
                group_paths[group_id].append(str(path))
        packet = group_packets.setdefault(
            group_id,
            {
                "schema_version": FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION,
                "planning_role": (
                    "visual_evidence_only_no_metric_verdict"
                ),
                "probe_inclusion_is_invalidity_prior": False,
                "group_id": group_id,
                "observation_requests": [],
                "image_order": [],
            },
        )
        neutral_goal = str(
            next(
                (
                    item.get("neutral_observation_goal")
                    for item in audit.get("group_confirmations") or []
                    if isinstance(item, dict)
                    and str(item.get("group_id")) == group_id
                    and set(item.get("target_ids") or [])
                    == set(
                        [
                            *result.get("target_ids", []),
                            *result.get("related_target_ids", []),
                        ]
                    )
                ),
                result.get("kind") or "functional group observation",
            )
        )
        observation_record = {
            "probe_id": result.get("probe_id"),
            "check_ids": deepcopy(result.get("check_ids") or []),
            "probe_kind": result.get("kind"),
            "target_ids": deepcopy(result.get("target_ids") or []),
            "related_target_ids": deepcopy(
                result.get("related_target_ids") or []
            ),
            "required_observations": deepcopy(
                result.get("required_observations") or []
            ),
            "observation_kinds": deepcopy(
                result.get("observation_kinds") or []
            ),
            "relation_predicates": deepcopy(
                result.get("relation_predicates") or []
            ),
            "observation_goals": deepcopy(
                result.get("observation_goals") or []
            ),
            "neutral_observation_goal": neutral_goal,
            "usable_surface": deepcopy(
                result.get("usable_surface_audit")
            ),
            "functional_geometry": deepcopy(
                result.get("functional_geometry")
            ),
            "functional_measurements": deepcopy(
                result.get("functional_measurements") or {}
            ),
            "evidence_coverage": deepcopy(
                result.get("evidence_coverage") or {}
            ),
            "evidence_reuse": deepcopy(
                result.get("evidence_reuse") or {}
            ),
        }
        packet["observation_requests"].append(observation_record)
        for path in result.get("evidence_paths") or []:
            packet["image_order"].append(
                {
                    "image_alias": (
                        f"group_probe_{len(packet['image_order']):02d}"
                    ),
                    "role": "functional_probe",
                    **deepcopy(observation_record),
                    "presentation": "raw_rgb",
                    "artifact_id": str(path),
                }
            )
    audit["cross_group_evidence_paths"] = list(
        dict.fromkeys(cross_group_paths)
    )
    unscheduled = [
        item
        for item in audit.get("unscheduled_discovery_items") or []
        if isinstance(item, dict)
    ]
    undelivered = [
        *unscheduled,
        *[
            item
            for item in audit.get("failed_discovery_items") or []
            if isinstance(item, dict)
        ],
    ]
    for item in undelivered:
        if (
            item.get("route_scope") != "group_local"
            or not item.get("owning_group_id")
        ):
            continue
        group_id = str(item["owning_group_id"])
        group_packets.setdefault(
            group_id,
            {
                "schema_version": FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION,
                "planning_role": (
                    "visual_evidence_only_no_metric_verdict"
                ),
                "probe_inclusion_is_invalidity_prior": False,
                "group_id": group_id,
                "observation_requests": [],
                "image_order": [],
            },
        )
    audit["group_probe_packets"] = group_packets
    _attach_boundary_evidence_to_group_packets(
        audit,
        groups=groups,
    )
    audit["functional_check_ledger"] = update_functional_check_evidence(
        audit.get("functional_check_ledger")
        or {
            "schema_version": FUNCTIONAL_CHECK_LEDGER_VERSION,
            "checks": [],
            "decision_authority": "none",
        },
        probe_results=[
            item
            for item in audit.get("probe_results") or []
            if isinstance(item, dict)
        ],
    )
    _attach_required_checks_to_group_packets(audit, groups=groups)
    group_packets = audit["group_probe_packets"]
    for group_id, packet in group_packets.items():
        group_unscheduled = [
            item
            for item in undelivered
            if item.get("route_scope") == "group_local"
            and str(item.get("owning_group_id") or "") == group_id
        ]
        packet["acquisition_coverage_complete"] = not group_unscheduled
        packet["observation_complete"] = False
        packet["coverage_complete"] = bool(
            packet["acquisition_coverage_complete"]
            and not packet.get("required_checks")
        )
        packet["coverage_semantics"] = (
            "acquisition_and_required_check_resolution"
        )
        packet["undelivered_observation_goals"] = [
            str(item.get("observation_goal") or "")
            for item in group_unscheduled
            if str(item.get("observation_goal") or "").strip()
        ]
        packet["undelivered_target_ids"] = list(
            dict.fromkeys(
                str(target_id)
                for item in group_unscheduled
                for target_id in item.get("target_ids") or []
                if str(target_id).strip()
            )
        )
        packet["budget_state"] = {
            "max_probe_units": audit.get("max_probe_units"),
            "budget_exhausted": any(
                item.get("reason") == "max_probe_units_exhausted"
                for item in group_unscheduled
            ),
            "acquisition_failed": any(
                item.get("reason") == "probe_acquisition_failed"
                for item in group_unscheduled
            ),
        }
    audit["group_evidence_paths"] = group_paths
    audit["group_probe_packets"] = group_packets
    _summarize_usable_surface_usage(audit)
    audit["rendered_probe_count"] = len(selected_paths)
    audit["failed_probe_count"] = failures
    audit["planned_probe_count"] = len(units)
    audit["attempted_probe_count"] = len(audit["probe_results"])
    audit["attempted_backfill_count"] = attempted_backfill_count
    audit["backfill_attempt_quota"] = int(
        backfill_attempt_quota or 0
    )
    audit["successful_probe_count"] = successful_probe_count
    audit["acquisition_coverage"] = {
        "unit": "planned_functional_probe",
        "eligible_count": len(units),
        "grounded_count": successful_probe_count,
        "fraction": (
            successful_probe_count / len(units) if units else 1.0
        ),
        "complete": successful_probe_count == len(units),
    }
    audit["status"] = (
        "complete"
        if (
            selected_paths
            and failures == 0
            and audit.get("coverage_complete")
        )
        else "partial"
        if selected_paths
        else "degraded_no_probes"
        if base_evidence_available
        else "failed"
    )
    audit["reason"] = (
        None
        if audit["status"] == "complete"
        else "functional_probe_failures_and_budget_exhaustion"
        if failures and audit.get("budget_exhausted")
        else "one_or_more_functional_probes_failed"
        if failures
        else "functional_acquisition_budget_exhausted"
        if audit.get("budget_exhausted")
        else "no_functional_probe_evidence_available"
    )
    return selected_paths, audit


def functional_probe_judge_packet(
    *,
    global_paths: list[str],
    probe_paths: list[str],
    acquisition_audit: dict[str, Any],
) -> dict[str, Any]:
    """Describe image roles without exposing local paths to the remote Judge."""

    image_roles: list[dict[str, Any]] = [
        {
            "image_index": index,
            "image_alias": f"image_{index:02d}",
            "role": "scene_global",
        }
        for index, _ in enumerate(global_paths)
    ]
    path_to_result = {
        str(path): result
        for result in acquisition_audit.get("probe_results") or []
        if isinstance(result, dict)
        for path in result.get("evidence_paths") or []
    }
    for offset, path in enumerate(
        probe_paths,
        start=len(global_paths),
    ):
        result = path_to_result.get(str(path), {})
        image_roles.append(
            {
                "image_index": offset,
                "image_alias": f"image_{offset:02d}",
                "role": "functional_probe",
                "probe_id": result.get("probe_id"),
                "check_ids": deepcopy(result.get("check_ids") or []),
                "probe_kind": result.get("kind"),
                "target_ids": deepcopy(
                    result.get("target_ids") or []
                ),
                "related_target_ids": deepcopy(
                    result.get("related_target_ids") or []
                ),
                "required_observations": deepcopy(
                    result.get("required_observations") or []
                ),
                "observation_kinds": deepcopy(
                    result.get("observation_kinds") or []
                ),
                "relation_predicates": deepcopy(
                    result.get("relation_predicates") or []
                ),
                "observation_goals": deepcopy(
                    result.get("observation_goals") or []
                ),
                "neutral_observation_goal": str(
                    result.get("view_goal")
                    or result.get("kind")
                    or "functional observation"
                ),
                "usable_surface": deepcopy(
                    result.get("usable_surface_audit")
                ),
                "functional_geometry": deepcopy(
                    result.get("functional_geometry")
                ),
                "functional_measurements": deepcopy(
                    result.get("functional_measurements") or {}
                ),
                "evidence_coverage": deepcopy(
                    result.get("evidence_coverage") or {}
                ),
                "evidence_reuse": deepcopy(
                    result.get("evidence_reuse") or {}
                ),
                "presentation": "raw_rgb",
            }
        )
    boundary_evidence = (
        acquisition_audit.get("functional_boundary_evidence")
        if isinstance(
            acquisition_audit.get("functional_boundary_evidence"),
            dict,
        )
        else {}
    )
    boundary_target_ids = {
        str(item.get("target_id") or "")
        for item in boundary_evidence.get(
            "requested_surface_targets"
        )
        or []
        if isinstance(item, dict) and item.get("target_id")
    }
    undelivered_cross_group = [
        item
        for item in [
            *(
                acquisition_audit.get("unscheduled_discovery_items")
                or []
            ),
            *(
                acquisition_audit.get("failed_discovery_items")
                or []
            ),
        ]
        if isinstance(item, dict)
        and item.get("route_scope") == "cross_group"
    ]
    relation_observation_requests = [
        {
            "probe_id": result.get("probe_id"),
            "check_ids": deepcopy(result.get("check_ids") or []),
            "discovery_ids": deepcopy(
                result.get("discovery_ids") or []
            ),
            "scope": "cross_group",
            "target_ids": list(
                dict.fromkeys(
                    [
                        *[
                            str(item)
                            for item in result.get("target_ids") or []
                        ],
                        *[
                            str(item)
                            for item in (
                                result.get("related_target_ids") or []
                            )
                        ],
                    ]
                )
            ),
            "observation_kinds": deepcopy(
                result.get("observation_kinds") or []
            ),
            "relation_predicates": deepcopy(
                result.get("relation_predicates") or []
            ),
            "observation_goals": deepcopy(
                result.get("observation_goals")
                or [result.get("view_goal")]
            ),
            "evidence_image_aliases": [
                str(item.get("image_alias"))
                for item in image_roles
                if item.get("probe_id") == result.get("probe_id")
            ],
            "instructional_role": "relation_to_inspect",
            "evidence_reuse": deepcopy(
                result.get("evidence_reuse") or {}
            ),
            "decision_authority": "none",
        }
        for result in acquisition_audit.get("probe_results") or []
        if isinstance(result, dict)
        and result.get("status") == "available"
        and result.get("route_scope") == "cross_group"
        and result.get("kind") == "functional_correspondence"
    ]
    required_checks = [
        deepcopy(check)
        for check in (
            (
                acquisition_audit.get("functional_check_ledger")
                or {}
            ).get("checks")
            or []
        )
        if isinstance(check, dict)
        and check.get("owner_stage") == "cross_group_relation"
    ]
    required_check_ids = [
        str(check.get("check_id") or "") for check in required_checks
    ]
    functional_measurements = _judge_measurements_for_checks(
        acquisition_audit,
        required_check_ids,
    )
    return {
        "schema_version": FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION,
        "planning_role": "visual_evidence_only_no_metric_verdict",
        "probe_inclusion_is_invalidity_prior": False,
        "boundary_clearance_evidence": (
            boundary_evidence_for_targets(
                boundary_evidence,
                boundary_target_ids,
            )
            if boundary_target_ids
            else {
                "schema_version": FUNCTIONAL_BOUNDARY_EVIDENCE_VERSION,
                "status": str(
                    boundary_evidence.get("status")
                    or "not_applicable"
                ),
                "decision_authority": "none",
                "scene_access": "read_only",
                "requested_surface_targets": [],
                "usable_surface_hypotheses": [],
                "functional_geometry": {
                    "surface_observations": [],
                    "observation_status": "unavailable",
                    "decision_authority": "none",
                    "scene_access": "read_only",
                },
            }
        ),
        "relation_observation_requests": (
            relation_observation_requests
        ),
        "required_checks": required_checks,
        "required_check_ids": required_check_ids,
        "required_check_count": len(required_checks),
        "functional_measurements": functional_measurements,
        "image_order": image_roles,
        "acquisition_coverage_complete": not undelivered_cross_group,
        "observation_complete": False,
        "coverage_complete": bool(
            not undelivered_cross_group and not required_checks
        ),
        "coverage_semantics": (
            "acquisition_and_required_check_resolution"
        ),
        "undelivered_observation_goals": [
            str(item.get("observation_goal") or "")
            for item in undelivered_cross_group
            if str(item.get("observation_goal") or "").strip()
        ],
        "undelivered_target_ids": list(
            dict.fromkeys(
                str(target_id)
                for item in undelivered_cross_group
                for target_id in item.get("target_ids") or []
                if str(target_id).strip()
            )
        ),
        "budget_state": {
            "max_probe_units": acquisition_audit.get("max_probe_units"),
            "planned_probe_count": acquisition_audit.get(
                "planned_probe_count"
            ),
            "rendered_probe_count": acquisition_audit.get(
                "rendered_probe_count"
            ),
            "budget_exhausted": bool(
                any(
                    item.get("reason") == "max_probe_units_exhausted"
                    for item in undelivered_cross_group
                )
            ),
            "acquisition_failed": bool(
                any(
                    item.get("reason") == "probe_acquisition_failed"
                    for item in undelivered_cross_group
                )
            ),
        },
        "source_scene_pixels_modified": False,
    }


def functional_relation_judge_packet(
    *,
    global_paths: list[str],
    probe_result: dict[str, Any],
    required_checks: list[dict[str, Any]] | None = None,
    required_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one packet for atomic checks sharing a cross-group target set.

    The caller starts the isolated Judge episode only after pair-specific
    evidence is available. The packet still records acquisition state so a
    missing artifact remains explicit in schedule/audit data instead of being
    silently treated as sufficient global evidence.
    """

    if probe_result.get("route_scope") != "cross_group":
        raise ValueError(
            "functional relation Judge packet requires cross_group scope"
        )
    if probe_result.get("kind") != "functional_correspondence":
        raise ValueError(
            "functional relation Judge packet requires a correspondence"
        )
    probe_paths = [
        str(path)
        for path in probe_result.get("evidence_paths") or []
        if str(path).strip()
    ]
    pair_specific_available = (
        probe_result.get("status") == "available"
        and bool(probe_paths)
    )
    if required_checks is not None and required_check is not None:
        raise ValueError(
            "supply required_checks or required_check, not both"
        )
    normalized_required_checks = (
        [
            deepcopy(check)
            for check in required_checks
            if isinstance(check, dict)
        ]
        if required_checks is not None
        else [deepcopy(required_check)]
        if isinstance(required_check, dict)
        else []
    )
    if required_checks is not None and len(normalized_required_checks) != len(
        required_checks
    ):
        raise TypeError("required_checks must contain JSON objects")
    if probe_result.get("status") == "available" and not probe_paths:
        raise ValueError(
            "functional relation Judge packet requires pair-specific evidence"
        )
    target_ids = list(
        dict.fromkeys(
            [
                *[
                    str(item)
                    for item in probe_result.get("target_ids") or []
                ],
                *[
                    str(item)
                    for item in (
                        probe_result.get("related_target_ids") or []
                    )
                ],
            ]
        )
    )
    if pair_specific_available:
        packet = functional_probe_judge_packet(
            global_paths=global_paths,
            probe_paths=probe_paths,
            acquisition_audit={
                "probe_results": [deepcopy(probe_result)],
                "unscheduled_discovery_items": [],
                "failed_discovery_items": [],
                "functional_check_ledger": {
                    "schema_version": FUNCTIONAL_CHECK_LEDGER_VERSION,
                    "checks": normalized_required_checks,
                    "decision_authority": "none",
                },
                "max_probe_units": 1,
                "planned_probe_count": 1,
                "rendered_probe_count": 1,
            },
        )
    else:
        image_roles = [
            {
                "image_index": index,
                "image_alias": f"image_{index:02d}",
                "role": "scene_global_relation_fallback",
            }
            for index, _ in enumerate(global_paths)
        ]
        packet = {
            "schema_version": FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION,
            "planning_role": "visual_evidence_only_no_metric_verdict",
            "probe_inclusion_is_invalidity_prior": False,
            "boundary_clearance_evidence": {
                "schema_version": FUNCTIONAL_BOUNDARY_EVIDENCE_VERSION,
                "status": "not_applicable",
                "decision_authority": "none",
                "scene_access": "read_only",
                "requested_surface_targets": [],
                "usable_surface_hypotheses": [],
                "functional_geometry": {
                    "surface_observations": [],
                    "observation_status": "unavailable",
                    "decision_authority": "none",
                    "scene_access": "read_only",
                },
            },
            "relation_observation_requests": [
                {
                    "probe_id": probe_result.get("probe_id"),
                    "check_ids": [
                        str(check.get("check_id") or "")
                        for check in normalized_required_checks
                    ],
                    "discovery_ids": deepcopy(
                        probe_result.get("discovery_ids") or []
                    ),
                    "scope": "cross_group",
                    "target_ids": target_ids,
                    "observation_kinds": deepcopy(
                        probe_result.get("observation_kinds") or []
                    ),
                    "relation_predicates": deepcopy(
                        probe_result.get("relation_predicates") or []
                    ),
                    "observation_goals": deepcopy(
                        probe_result.get("observation_goals")
                        or [probe_result.get("view_goal")]
                    ),
                    "evidence_image_aliases": [
                        item["image_alias"] for item in image_roles
                    ],
                    "instructional_role": "relation_to_inspect",
                    "decision_authority": "none",
                    "pair_specific_evidence_available": False,
                }
            ],
            "image_order": image_roles,
            "required_checks": normalized_required_checks,
            "required_check_ids": [
                str(check.get("check_id") or "")
                for check in normalized_required_checks
            ],
            "required_check_count": len(normalized_required_checks),
            "functional_measurements": deepcopy(
                probe_result.get("functional_measurements") or {}
            ),
            "coverage_complete": False,
            "acquisition_coverage_complete": False,
            "observation_complete": False,
            "coverage_semantics": (
                "acquisition_and_required_check_resolution"
            ),
            "undelivered_observation_goals": deepcopy(
                probe_result.get("observation_goals")
                or [probe_result.get("view_goal")]
            ),
            "undelivered_target_ids": target_ids,
            "budget_state": {
                "max_probe_units": 1,
                "planned_probe_count": 1,
                "rendered_probe_count": 0,
                "budget_exhausted": (
                    probe_result.get("status") == "not_scheduled"
                ),
                "acquisition_failed": (
                    probe_result.get("status") != "not_scheduled"
                ),
            },
            "source_scene_pixels_modified": False,
        }
    packet.update(
        episode_scope="single_cross_group_relation_target_set",
        relation_id=str(
            (probe_result.get("discovery_ids") or [None])[0]
            or probe_result.get("probe_id")
            or ""
        ),
        relation_predicates=deepcopy(
            probe_result.get("relation_predicates") or []
        ),
        probe_id=probe_result.get("probe_id"),
        allowed_defect_target_ids=target_ids,
        defect_target_policy="non_empty_subset_offending_objects_only",
        pair_specific_evidence_available=pair_specific_available,
        artifact_rendered=pair_specific_available,
        view_coverage_complete=False,
        observation_complete=False,
        machine_observation_complete=(
            _machine_observation_complete(
                probe_result,
                target_ids=target_ids,
            )
        ),
        acquisition_status=str(
            probe_result.get("status") or "failed"
        ),
        acquisition_error=(
            {
                "error_type": probe_result.get("error_type"),
                "error": probe_result.get("error"),
            }
            if not pair_specific_available
            else None
        ),
        decision_authority="none",
    )
    return packet


def _judge_measurements_for_checks(
    acquisition_audit: dict[str, Any],
    check_ids: list[str],
) -> dict[str, Any]:
    """Resolve compact measurements from the pre-scheduling bank.

    Older frozen audits may contain only the compact subset copied onto a
    probe result.  That fallback reuses recorded facts; it never recomputes
    measurements after camera scheduling.
    """

    bank = acquisition_audit.get("functional_measurement_bank")
    if isinstance(bank, dict):
        return compact_functional_measurements_for_checks(bank, check_ids)
    requested = {
        str(item) for item in check_ids if str(item).strip()
    }
    rows: dict[str, dict[str, Any]] = {}
    template: dict[str, Any] | None = None
    for result in acquisition_audit.get("probe_results") or []:
        if not isinstance(result, dict):
            continue
        context = result.get("functional_measurements")
        if not isinstance(context, dict):
            continue
        template = context
        for row in context.get("check_measurements") or []:
            if not isinstance(row, dict):
                continue
            check_id = str(row.get("check_id") or "")
            if check_id in requested:
                rows.setdefault(check_id, deepcopy(row))
    if template is None:
        return compact_functional_measurements_for_checks(None, check_ids)
    ordered = [
        rows[check_id]
        for check_id in check_ids
        if check_id in rows
    ]
    return {
        "schema_version": template.get("schema_version"),
        "status": (
            "complete"
            if len(ordered) == len(requested)
            else "partial"
            if ordered
            else "unavailable"
        ),
        "measurement_role": template.get(
            "measurement_role",
            "deterministic_spatial_evidence_not_verdict",
        ),
        "measurement_semantics": template.get("measurement_semantics"),
        "decision_authority": "none",
        "requested_check_ids": list(check_ids),
        "check_measurements": ordered,
    }


def _machine_observation_complete(
    probe_result: dict[str, Any],
    *,
    target_ids: list[str],
) -> bool | None:
    """Report decoder coverage separately from rendered-artifact availability."""

    requested_surface_ids = {
        str(item.get("target_id") or "")
        for item in probe_result.get("surface_targets") or []
        if isinstance(item, dict) and item.get("target_id")
    }
    if not requested_surface_ids:
        return None
    usable_surface = (
        probe_result.get("usable_surface_audit")
        if isinstance(probe_result.get("usable_surface_audit"), dict)
        else {}
    )
    functional_geometry = (
        probe_result.get("functional_geometry")
        if isinstance(probe_result.get("functional_geometry"), dict)
        else {}
    )
    observed_ids = {
        str(item.get("target_id") or "")
        for item in [
            *(usable_surface.get("hypotheses") or []),
            *(functional_geometry.get("surface_observations") or []),
        ]
        if isinstance(item, dict) and item.get("target_id")
        and (
            item.get("status") == "identified"
            or bool(item.get("side_id"))
        )
    }
    trusted_targets = {str(item) for item in target_ids}
    return bool(
        requested_surface_ids
        and requested_surface_ids <= trusted_targets
        and requested_surface_ids <= observed_ids
    )


def _probe_unit_identity_record(unit: dict[str, Any]) -> list[Any]:
    return [
        str(unit.get("kind") or ""),
        sorted(
            {
                *[
                    str(item)
                    for item in unit.get("target_ids") or []
                ],
                *[
                    str(item)
                    for item in unit.get("related_target_ids") or []
                ],
            }
        ),
        sorted(
            str(item)
            for item in unit.get("required_observations") or []
        ),
    ]


def _attach_required_checks_to_group_packets(
    audit: dict[str, Any],
    *,
    groups: list[dict[str, Any]] | None,
) -> None:
    """Route every accepted group-local obligation to its owning Judge."""

    ledger = (
        audit.get("functional_check_ledger")
        if isinstance(audit.get("functional_check_ledger"), dict)
        else {}
    )
    packets = (
        audit.get("group_probe_packets")
        if isinstance(audit.get("group_probe_packets"), dict)
        else {}
    )
    forced_group_ids = [
        str(item)
        for item in audit.get("forced_group_ids") or []
        if str(item).strip()
    ]
    trusted_group_ids = {
        str(group.get("group_id") or "")
        for group in groups or []
        if isinstance(group, dict) and group.get("group_id")
    }
    for group_id in forced_group_ids_from_checks(ledger):
        if trusted_group_ids and group_id not in trusted_group_ids:
            raise ValueError(
                "functional check references an unknown owning group "
                f"{group_id!r}"
            )
        required_checks = checks_for_group(ledger, group_id)
        if not required_checks:
            continue
        packet = packets.setdefault(
            group_id,
            {
                "schema_version": FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION,
                "planning_role": (
                    "visual_evidence_only_no_metric_verdict"
                ),
                "probe_inclusion_is_invalidity_prior": False,
                "group_id": group_id,
                "observation_requests": [],
                "image_order": [],
            },
        )
        packet["required_checks"] = required_checks
        packet["required_check_ids"] = [
            str(check["check_id"]) for check in required_checks
        ]
        packet["required_check_count"] = len(required_checks)
        packet["functional_measurements"] = (
            _judge_measurements_for_checks(
                audit,
                packet["required_check_ids"],
            )
        )
        architecture_orientation_checks = [
            check
            for check in required_checks
            if check.get("check_type") == "architecture_orientation"
        ]
        clearance_checks = [
            check
            for check in required_checks
            if check.get("check_type") == "clearance"
        ]
        packet["architecture_orientation_policy"] = {
            "check_ids": [
                str(check["check_id"])
                for check in architecture_orientation_checks
            ],
            "predicate": (
                "usable_side_points_toward_plausible_accessible_interior"
            ),
            "deterministic_direction_descriptor": (
                "camera_routing_and_measurement_bank_judge_evidence"
            ),
            "judge_evidence": [
                "one_angled_global_view",
                "same_side_conditioned_local_view",
            ],
            "independent_from_clearance": True,
            "decision_authority": "judge",
        }
        packet["clearance_policy"] = {
            "check_ids": [
                str(check["check_id"]) for check in clearance_checks
            ],
            "predicate": (
                "required_approach_opening_or_operation_space_is_available"
            ),
            "architecture_is_a_possible_blocker_not_a_separate_check": True,
            "independent_from_architecture_orientation": True,
            "decision_authority": "judge",
        }
        packet["observation_complete"] = False
        packet["decision_authority"] = "none"
        if group_id not in forced_group_ids:
            forced_group_ids.append(group_id)
    audit["group_probe_packets"] = packets
    audit["forced_group_ids"] = forced_group_ids


def _attach_boundary_evidence_to_group_packets(
    audit: dict[str, Any],
    *,
    groups: list[dict[str, Any]] | None,
) -> None:
    """Route structured facts to trusted owning groups without reownership."""

    evidence = audit.get("functional_boundary_evidence")
    if not isinstance(evidence, dict):
        return
    requested_ids = {
        str(item.get("target_id") or "")
        for item in evidence.get("requested_surface_targets") or []
        if isinstance(item, dict) and item.get("target_id")
    }
    if not requested_ids:
        return
    packets = (
        audit.get("group_probe_packets")
        if isinstance(audit.get("group_probe_packets"), dict)
        else {}
    )
    forced_group_ids = [
        str(item)
        for item in audit.get("forced_group_ids") or []
        if str(item).strip()
    ]
    for group in groups or []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id") or "").strip()
        member_ids = {
            str(item)
            for item in group.get("object_ids") or []
            if str(item).strip()
        }
        target_ids = sorted(requested_ids & member_ids)
        if not group_id or not target_ids:
            continue
        subset = boundary_evidence_for_targets(
            evidence,
            set(target_ids),
        )
        packet = packets.setdefault(
            group_id,
            {
                "schema_version": FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION,
                "planning_role": (
                    "visual_evidence_only_no_metric_verdict"
                ),
                "probe_inclusion_is_invalidity_prior": False,
                "group_id": group_id,
                "observation_requests": [],
                "image_order": [],
            },
        )
        packet["boundary_clearance_evidence"] = subset
        packet["observation_requests"].append(
            {
                "probe_id": "functional_boundary_prepass",
                "probe_kind": "usable_side_direction_prepass",
                "target_ids": target_ids,
                "related_target_ids": [],
                "required_observations": [
                    "interaction_side_visible",
                    "front_back_disambiguated",
                ],
                "neutral_observation_goal": (
                    "Use the decoded usable-side hypothesis to bind the "
                    "correct local view. Use the compact Functional "
                    "Measurement Bank heading and relation facts as "
                    "deterministic spatial evidence, while the angled global "
                    "and side-conditioned local visuals confirm side semantics "
                    "and usage context. Optional clearance measurements are "
                    "evidence, not thresholds."
                ),
                "usable_surface": deepcopy(
                    subset.get("usable_surface_hypotheses") or []
                ),
                "functional_geometry": deepcopy(
                    subset.get("functional_geometry") or {}
                ),
                "decision_authority": "none",
            }
        )
        if group_id not in forced_group_ids:
            forced_group_ids.append(group_id)
    audit["group_probe_packets"] = packets
    audit["forced_group_ids"] = forced_group_ids


def _summarize_usable_surface_usage(
    audit: dict[str, Any],
) -> None:
    surface_audits = [
        result.get("usable_surface_audit")
        for result in audit.get("probe_results") or []
        if isinstance(result, dict)
        and isinstance(result.get("usable_surface_audit"), dict)
    ]
    boundary = (
        audit.get("functional_boundary_evidence")
        if isinstance(
            audit.get("functional_boundary_evidence"),
            dict,
        )
        else {}
    )
    boundary_decoder = (
        boundary.get("decoder_audit")
        if isinstance(boundary.get("decoder_audit"), dict)
        else {}
    )
    audit["usable_surface_decoder_calls"] = int(
        boundary.get("decoder_calls") or 0
    ) + sum(
        int(item.get("decoder_calls") or 0)
        for item in surface_audits
    )
    audit["usable_surface_catalog_contract_hits"] = int(
        boundary.get("catalog_contract_hits") or 0
    ) + sum(
        int(item.get("catalog_contract_hits") or 0)
        for item in surface_audits
    )
    audit["usable_surface_catalog_contract_misses"] = int(
        boundary.get("catalog_contract_misses") or 0
    ) + sum(
        int(item.get("catalog_contract_misses") or 0)
        for item in surface_audits
    )
    audit["usable_surface_cache_hits"] = int(
        boundary.get("cache_hits") or 0
    ) + sum(
        int(item.get("cache_hits") or 0)
        for item in surface_audits
    )
    audit["usable_surface_precomputed_reuses"] = sum(
        int(item.get("precomputed_hypotheses") or 0)
        for item in surface_audits
    )
    audit["usable_surface_preview_render_count"] = int(
        boundary.get("preview_render_count") or 0
    ) + sum(
        int(item.get("preview_render_count") or 0)
        for item in surface_audits
    )
    hypotheses_by_target: dict[str, dict[str, Any]] = {}
    for hypothesis in [
        *(
            boundary.get("usable_surface_hypotheses") or []
        ),
        *[
            hypothesis
            for item in surface_audits
            for hypothesis in item.get("hypotheses") or []
        ],
    ]:
        if not isinstance(hypothesis, dict):
            continue
        target_id = str(hypothesis.get("target_id") or "").strip()
        if target_id:
            hypotheses_by_target[target_id] = deepcopy(hypothesis)
    audit["usable_surface_hypotheses"] = list(
        hypotheses_by_target.values()
    )
    audit["functional_boundary_decoder_audit"] = deepcopy(
        boundary_decoder
    )


def _minimal_object_list(scene: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in scene.get("objects") or []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("id") or "").strip()
        category = str(
            item.get("category")
            or item.get("retrieval_category")
            or ""
        ).strip()
        if object_id and category:
            result.append({"id": object_id, "category": category})
    return result


def _minimal_group_list(
    groups: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [
        {
            "group_id": str(group.get("group_id") or ""),
            "object_ids": [
                str(item)
                for item in group.get("object_ids") or []
                if str(item).strip()
            ],
        }
        for group in groups or []
        if isinstance(group, dict) and str(group.get("group_id") or "").strip()
    ]


def _group_for_object(
    groups: list[dict[str, Any]] | None,
    object_id: str,
) -> str | None:
    return next(
        (
            str(group.get("group_id"))
            for group in groups or []
            if isinstance(group, dict)
            and object_id
            in {
                str(item)
                for item in group.get("object_ids") or []
            }
        ),
        None,
    )


def _single_group_for_targets(
    groups: list[dict[str, Any]] | None,
    target_ids: list[str],
) -> str | None:
    group_ids = {
        _group_for_object(groups, object_id)
        for object_id in target_ids
    }
    group_ids.discard(None)
    return next(iter(group_ids)) if len(group_ids) == 1 else None


def _group_by_id(
    groups: list[dict[str, Any]] | None,
    group_id: str | None,
) -> dict[str, Any] | None:
    if not group_id:
        return None
    return next(
        (
            group
            for group in groups or []
            if isinstance(group, dict)
            and str(group.get("group_id") or "") == str(group_id)
        ),
        None,
    )


def _functional_architecture_context(
    scene: dict[str, Any],
) -> dict[str, Any]:
    metadata = (
        scene.get("metadata")
        if isinstance(scene.get("metadata"), dict)
        else {}
    )
    explicit_contract = metadata.get("architecture_contract")
    boundary = scene.get("boundary")
    has_explicit_boundary = bool(
        isinstance(boundary, list)
        and len(boundary) >= 3
    )
    if not isinstance(explicit_contract, dict) and not has_explicit_boundary:
        return {
            "source": "unavailable",
            "logical_boundary_enabled": False,
            "logical_boundary_xy": [],
            "physical_walls_rendered": None,
            "physical_wall_ids": [],
        }
    contract = architecture_contract_from_scene(scene)
    logical = (
        contract.get("logical_boundary")
        if isinstance(contract.get("logical_boundary"), dict)
        else {}
    )
    physical = (
        contract.get("physical_walls")
        if isinstance(contract.get("physical_walls"), dict)
        else {}
    )
    wall_ids = [
        str(item)
        for item in physical.get("active_wall_ids") or []
        if str(item).strip()
    ]
    return {
        "source": (
            "scene_architecture_contract"
            if isinstance(explicit_contract, dict)
            else "scene_boundary_adapter"
        ),
        "logical_boundary_enabled": bool(logical.get("enabled")),
        "logical_boundary_xy": deepcopy(
            logical.get("boundary") or []
        ),
        "physical_walls_rendered": bool(wall_ids),
        "physical_wall_ids": wall_ids,
    }


def _probe_view_goal(unit: dict[str, Any]) -> str:
    explicit_goal = str(unit.get("view_goal") or "").strip()
    if explicit_goal:
        return explicit_goal[:1000]
    kind = str(unit.get("kind") or "")
    observations = {
        str(item)
        for item in unit.get("required_observations") or []
    }
    if "architecture_plane_visible" in observations:
        return (
            "Show the object's visually decoded usable or control side together "
            "with the nearest authoritative logical room boundary or visible "
            "architectural constraint and the interior-side user approach and "
            "operating region. Physical wall geometry may be absent; do not "
            "treat rendered background beyond the logical room boundary as "
            "accessible interior space."
        )
    if kind == "functional_correspondence":
        predicates = {
            str(item)
            for item in unit.get("relation_predicates") or []
        }
        if predicates == {"relative_use_geometry"}:
            return (
                "Show all related objects together at a scale that preserves "
                "their relative position, distance, reach, coordinated-use, or "
                "operational-connection region. Do not require a usable-side view unless "
                "another supplied observation explicitly requests it."
            )
        return (
            "From a relevant usable-side half-space, show the related objects "
            "together at near interaction height. Keep their usable faces, "
            "relative directions, relative-use geometry, and the context "
            "required by every supplied atomic predicate visually decodable."
        )
    if kind == "approach_clearance":
        return (
            "From the usable-side or approach-side half-space, keep the "
            "object's usable face visible together with the wider user approach "
            "or operating-clearance zone and its nearby architectural "
            "constraints."
        )
    return (
        "From the usable-side half-space, show the object's functional "
        "frontage at near interaction height while preserving the wider "
        "outward space and context that frontage faces."
    )


def _raw_probe_paths(value: Any) -> list[str]:
    raw: Any = value
    if isinstance(value, dict):
        status = str(value.get("status") or "available").lower()
        if status in {
            "failed",
            "error",
            "insufficient",
            "unavailable",
            "not_available",
        } or value.get("error"):
            return []
        raw = (
            value.get("render_evidence_items")
            or value.get("paths")
            or value.get("render_evidence")
            or []
        )
    if not isinstance(raw, (list, tuple)):
        return []
    result: list[str] = []
    for item in raw:
        if isinstance(item, (str, Path)):
            path = str(item)
        elif isinstance(item, dict):
            role = str(item.get("role") or "").lower()
            style = str(item.get("evidence_style") or "raw").lower()
            transform = str(item.get("image_transform") or "none").lower()
            if (
                "highlight" in role
                or "overlay" in role
                or "global" in role
                or style not in {"raw", "rgb", ""}
                or transform not in {"none", ""}
            ):
                continue
            path = str(
                item.get("path") or item.get("image_path") or ""
            )
        else:
            continue
        if path and path not in result:
            result.append(path)
    return result
