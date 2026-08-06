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
from benchmark.rendering.camera_pose import (
    FUNCTIONAL_PROBE_CANDIDATE_BUDGETS,
)
from benchmark.visual_judge.functional_discovery import (
    FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
)
from benchmark.visual_judge.functional_evidence import (
    FUNCTIONAL_PROBE_MAX_UNITS,
    FUNCTIONAL_PROBE_PLAN_VERSION,
)


FUNCTIONAL_PROBE_ACQUISITION_VERSION = (
    "functional_probe_acquisition_v3"
)
FUNCTIONAL_PROBE_JUDGE_PACKET_VERSION = (
    "functional_probe_judge_packet_v4"
)


def acquire_functional_probe_evidence(
    *,
    planner: Any,
    provider: Any,
    scene: dict[str, Any],
    global_image_path: str,
    max_probe_units: int = FUNCTIONAL_PROBE_MAX_UNITS,
    groups: list[dict[str, Any]] | None = None,
    grouping_report: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Plan and render raw probe images without issuing a metric verdict."""

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
    discovery_call = getattr(
        planner,
        "discover_functional_evidence",
        None,
    )
    legacy_plan_call = getattr(
        planner,
        "plan_functional_evidence",
        None,
    )
    if not callable(discovery_call) and not callable(legacy_plan_call):
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
        "excluded_fields": [
            "center",
            "size",
            "rotation",
            "description",
            "grouping_reason",
        ],
    }
    try:
        if callable(discovery_call):
            discovery = discovery_call(planning_request)
            boundary_evidence = acquire_functional_boundary_evidence(
                provider=provider,
                scene=scene,
                discovery=discovery,
                architecture_context=architecture_context,
            )
            audit["functional_boundary_evidence"] = deepcopy(
                boundary_evidence
            )
            plan = build_functional_acquisition_plan(
                discovery_with_boundary_hypotheses(
                    discovery,
                    boundary_evidence,
                ),
                max_probe_units=max_probe_units,
            )
            audit["functional_discovery"] = deepcopy(discovery)
            audit["functional_acquisition_plan"] = deepcopy(plan)
            audit["planner_mode"] = "functional_discovery_v3"
        else:
            plan = legacy_plan_call(planning_request)
            audit["planner_mode"] = "legacy_functional_probe_plan_v2"
    except Exception as exc:
        audit.update(
            status="failed",
            reason="functional_probe_planner_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return [], audit
    units = (
        plan.get("probe_units")
        if isinstance(plan, dict)
        and isinstance(plan.get("probe_units"), list)
        else []
    )
    audit["planner_request_metadata"] = deepcopy(
        (
            discovery.get("provenance", {}).get("request_metadata")
            if callable(discovery_call)
            and isinstance(discovery, dict)
            else plan.get("request_metadata")
            if isinstance(plan, dict)
            else None
        )
    )
    audit["planner_reason"] = (
        str(discovery.get("reason") or "")
        if callable(discovery_call) and isinstance(discovery, dict)
        else str(plan.get("reason") or "")
        if isinstance(plan, dict)
        else ""
    )
    if isinstance(plan, dict):
        audit["group_confirmations"] = deepcopy(
            plan.get("group_confirmations") or []
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
                str(item.get("owning_group_id"))
                for item in [
                    *units,
                    *(
                        plan.get("unscheduled_discovery_items") or []
                    ),
                ]
                if isinstance(item, dict)
                and item.get("route_scope") == "group_local"
                and item.get("owning_group_id")
            )
        )
    audit["probe_units"] = deepcopy(units)
    if not units:
        _attach_boundary_evidence_to_group_packets(
            audit,
            groups=groups,
        )
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
        _summarize_usable_surface_usage(audit)
        audit.update(
            status="failed",
            reason="functional_probe_provider_not_configured",
            rendered_probe_count=0,
            failed_probe_count=len(units),
            planned_probe_count=len(units),
        )
        return [], audit

    categories = {
        str(item["id"]): str(item["category"])
        for item in minimal_objects
    }
    selected_paths: list[str] = []
    failures = 0
    for unit in units:
        if not isinstance(unit, dict):
            failures += 1
            continue
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
            == "legacy_functional_probe_plan_v2"
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
            "acquisition_trigger": unit.get("acquisition_trigger"),
            "surface_targets": deepcopy(
                unit.get("surface_targets") or []
            ),
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
        audit["probe_results"].append(result_record)

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
            "probe_kind": result.get("kind"),
            "target_ids": deepcopy(result.get("target_ids") or []),
            "related_target_ids": deepcopy(
                result.get("related_target_ids") or []
            ),
            "required_observations": deepcopy(
                result.get("required_observations") or []
            ),
            "neutral_observation_goal": neutral_goal,
            "usable_surface": deepcopy(
                result.get("usable_surface_audit")
            ),
            "functional_geometry": deepcopy(
                result.get("functional_geometry")
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
    for item in unscheduled:
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
    group_packets = audit["group_probe_packets"]
    for group_id, packet in group_packets.items():
        group_unscheduled = [
            item
            for item in unscheduled
            if item.get("route_scope") == "group_local"
            and str(item.get("owning_group_id") or "") == group_id
        ]
        packet["coverage_complete"] = not group_unscheduled
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
            "budget_exhausted": bool(group_unscheduled),
        }
    audit["group_evidence_paths"] = group_paths
    audit["group_probe_packets"] = group_packets
    _summarize_usable_surface_usage(audit)
    audit["rendered_probe_count"] = len(selected_paths)
    audit["failed_probe_count"] = failures
    audit["planned_probe_count"] = len(units)
    audit["status"] = (
        "complete"
        if (
            selected_paths
            and failures == 0
            and not audit.get("budget_exhausted")
        )
        else "partial"
        if selected_paths
        else "failed"
    )
    audit["reason"] = (
        None
        if audit["status"] == "complete"
        else "functional_acquisition_budget_exhausted"
        if audit.get("budget_exhausted")
        else "one_or_more_functional_probes_failed"
        if audit["status"] == "partial"
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
        "image_order": image_roles,
        "coverage_complete": not any(
            isinstance(item, dict)
            and item.get("route_scope") == "cross_group"
            for item in (
                acquisition_audit.get("unscheduled_discovery_items")
                or []
            )
        ),
        "undelivered_observation_goals": [
            str(item.get("observation_goal") or "")
            for item in (
                acquisition_audit.get("unscheduled_discovery_items") or []
            )
            if isinstance(item, dict)
            and item.get("route_scope") == "cross_group"
            and str(item.get("observation_goal") or "").strip()
        ],
        "undelivered_target_ids": list(
            dict.fromkeys(
                str(target_id)
                for item in (
                    acquisition_audit.get(
                        "unscheduled_discovery_items"
                    )
                    or []
                )
                if isinstance(item, dict)
                and item.get("route_scope") == "cross_group"
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
                    isinstance(item, dict)
                    and item.get("route_scope") == "cross_group"
                    and item.get("reason") == "max_probe_units_exhausted"
                    for item in (
                        acquisition_audit.get(
                            "unscheduled_discovery_items"
                        )
                        or []
                    )
                )
            ),
        },
        "source_scene_pixels_modified": False,
    }


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
                "probe_kind": "usable_side_boundary_clearance",
                "target_ids": target_ids,
                "related_target_ids": [],
                "required_observations": [
                    "interaction_side_visible",
                    "approach_zone_visible",
                    "architecture_plane_visible",
                ],
                "neutral_observation_goal": (
                    "Consider the decoded usable-side hypothesis together "
                    "with deterministic logical-boundary clearance facts. "
                    "The measurements are evidence, not a validity threshold."
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
            "floor extent and the interior-side user approach and operating "
            "region. Physical wall geometry may be absent; do not treat image "
            "background outside the room footprint as usable floor space."
        )
    if kind == "functional_correspondence":
        return (
            "From a relevant usable-side half-space, show the related objects "
            "together at near interaction height. Keep their usable faces, "
            "mutual interaction orientation, and the outward context those "
            "faces address visually decodable."
        )
    if kind == "approach_clearance":
        return (
            "From the usable-side or approach-side half-space, keep the "
            "object's usable face visible together with a wider floor-level "
            "user approach or operating-clearance zone."
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
