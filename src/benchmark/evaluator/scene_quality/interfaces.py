"""L3 Scene Quality evaluation.

L3 Scene Quality reframes the former narrow "Visual Quality" layer into two
subfamilies:

- **L3a Semantic Coherence** — ``scale_consistency`` and
  ``object_pairing_consistency``;
- **L3b Perceptual Visual Quality** — ``style_consistency``;
- **L3c Functional Validity** — ``functional_consistency``;
- **L3d Semantic Placement** —
  ``semantic_placement_consistency``.

``object_pairing_consistency`` is evaluated only after the configured grouping
algorithm supplies object groups. Its verdict covers target category and role
compatibility with both the scene and local group context. Object position,
distance, angle, orientation, access, and functional arrangement are not
pairing defects: prompt-specified local function belongs to L2
``functional_semantic_fidelity`` and explicit relations belong to L2 OOR/OAR.

The module consumes prepared visual evidence and an injected VLM judge. When a
local metric lacks scope-correct evidence, it may request a packet from an
injected camera-evidence provider; that provider remains selection/rendering
infrastructure and never supplies the metric verdict. This module does not own
camera policy, grouping, or prompt parsing. Missing evidence, a missing judge,
pending applicability, malformed responses, and missing grouping for Object
Pairing are all explicit unresolved states.

Prompt-authorized deviations are passed to the judge with target/relation scope.
When a judge returns structured defects, defects covered by an exact exemption
are removed before scoring. Exemptions never disable an entire metric.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.architecture_policy import architecture_contract_from_scene
from benchmark.evaluator.scene_quality.authorized_deviations import (
    deviation_matches,
    deviations_for_metric,
    validate_authorized_deviations,
)
from benchmark.evaluator.scene_quality.config import (
    SceneQualityInterfaceConfigError,
    _as_object,
    _deep_merge,
    _normalize_layer,
    _validate_evidence_policy,
    _validate_metric_flags,
    _validate_top_level,
    resolve_scene_quality_config,
)
from benchmark.evaluator.scene_quality.definitions import (
    CAMERA_MODES,
    CAMERA_SCOPES,
    DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG,
    EVIDENCE_SELECTORS,
    EXPERIMENTAL_NON_SCORING,
    EXPERIMENTAL_SCENE_QUALITY_METRICS,
    FUNCTIONAL_VALIDITY,
    FUNCTIONAL_VALIDITY_METRICS,
    IMAGE_ORDER_TOKENS,
    JUDGMENT_SCOPE_BY_METRIC,
    METRIC_RUBRICS,
    PERCEPTUAL_VISUAL_QUALITY,
    PERCEPTUAL_VISUAL_QUALITY_METRICS,
    PRESENTATIONS,
    SCENE_QUALITY_INTERFACE_METRICS,
    SCENE_QUALITY_INTERFACE_NAMESPACE,
    SCENE_QUALITY_INTERFACE_VERSION,
    SEMANTIC_COHERENCE,
    SEMANTIC_COHERENCE_METRICS,
    SEMANTIC_PLACEMENT,
    SEMANTIC_PLACEMENT_METRICS,
    SUBFAMILY_BY_METRIC,
    SUPPORTED_SCENE_QUALITY_METRICS,
    _GROUP_SCOPES,
    _RETIRED_CONFIG_NAMESPACES,
    _RETIRED_METRIC_NAMES,
)
from benchmark.evaluator.scene_quality.group_scoped import (
    evaluate_group_scoped_judgements as _evaluate_group_scoped_judgements,
    group_evidence_resolution_summary as _group_evidence_resolution_summary,
    group_packet_audit as _group_packet_audit,
    resolve_group_evidence_packets as _resolve_group_evidence_packets,
)
from benchmark.evaluator.scene_quality.global_group_first import (
    evaluate_global_discovery_then_group_local as _evaluate_global_discovery_then_group_local,
)
from benchmark.evaluator.scene_quality.functional_prejudgement import (
    validate_functional_prejudgement_evidence_config,
)
from benchmark.evaluator.scene_quality.functional_ownership import (
    validate_functional_ownership_ledger,
)
from benchmark.evaluator.scene_quality.placement_severity import (
    PLACEMENT_SEVERITY_LEVELS,
    validate_placement_defect_severity,
)
from benchmark.evaluator.scene_quality.json_screen_first import (
    evaluate_json_screen_then_group_visual as _evaluate_json_screen_then_group_visual,
)
from benchmark.evaluator.scene_quality.style_global_first import (
    evaluate_style_global_then_group_local as _evaluate_style_global_then_group_local,
)
from benchmark.evaluator.evidence_contract import (
    EVIDENCE_STRATEGIES,
    FINAL_VLM_CONTEXT_CONTRACT,
    LOCAL_TRIGGERS,
    ROUTER_STATES,
    ROUTER_TRIGGER_STATES,
    canonical_hierarchy,
    grouping_policy_provenance,
)
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)
from benchmark.visual_judge.group_scope import (
    GroupCameraScope,
    group_scope_evidence_goal,
)
from benchmark.visual_judge.l3_prompts import (
    L3_METRIC_BOUNDARY_RULES,
    L3_METRIC_PROMPT_VERSION,
)


def evaluate_scene_quality_interfaces(
    scene: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    object_grouping_report: dict[str, Any] | list[dict[str, Any]] | None = None,
    render_evidence: list[str] | dict[str, Any] | None = None,
    camera_evidence_provider: Any = None,
    functional_evidence_planner: Any = None,
    functional_probe_evidence_provider: Any = None,
    functional_prejudgement_evidence_source: Any = None,
    discovery_identity_image_path: str | None = None,
    discovery_identity_legend: dict[str, str] | None = None,
    vlm_judge: Any = None,
    authorized_deviations: Any = None,
    metric_applicability: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | None = None,
    prompt: str | None = None,
    visual_style_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the five benchmark L3 metrics from prepared visual evidence.

    Rendering, camera selection, object grouping, and applicability remain
    external responsibilities. This function may call the injected evidence
    provider for the requested scope, but the provider is kept separate from
    the final judge and cannot return a metric verdict. Their absence is never
    treated as valid. The injected judge may expose
    ``adjudicate_scene_quality`` or ``evaluate``, or be directly callable.
    """

    if not isinstance(scene, dict):
        raise TypeError("scene quality interface scene must be a JSON object")
    resolved = resolve_scene_quality_config(
        config, profile=profile, run_overrides=run_overrides
    )
    deviations = validate_authorized_deviations(
        authorized_deviations,
        metric_normalizer=str,
        allowed_metrics=SUPPORTED_SCENE_QUALITY_METRICS,
    )

    object_ids = _scene_object_ids(scene)
    groups = _normalize_groups(
        object_grouping_report,
        valid_object_ids=set(object_ids),
    )
    grouping_available = groups is not None
    if metric_applicability is not None and not isinstance(metric_applicability, dict):
        raise TypeError("metric_applicability must be a JSON object or null")
    applicability = metric_applicability if isinstance(metric_applicability, dict) else {}
    unknown_applicability = sorted(
        set(applicability) - set(SUPPORTED_SCENE_QUALITY_METRICS)
    )
    if unknown_applicability:
        raise ValueError(
            "metric_applicability contains unknown metrics: "
            f"{unknown_applicability}"
        )
    if isinstance(render_evidence, dict):
        retired_evidence_keys = sorted(
            set(render_evidence) & set(_RETIRED_METRIC_NAMES)
        )
        if retired_evidence_keys:
            raise ValueError(
                "render_evidence uses retired L3 metric keys "
                f"{retired_evidence_keys}; use canonical metric names"
            )
    top_enabled = bool(resolved["enabled"])
    functional_prejudgement_config = (
        validate_functional_prejudgement_evidence_config(
            resolved.get("functional_prejudgement_evidence")
        )
    )

    metric_reports: dict[str, dict[str, Any]] = {}
    for metric_name in SUPPORTED_SCENE_QUALITY_METRICS:
        metric_config = resolved["metrics"][metric_name]
        metric_report = _evaluate_metric(
            metric_name=metric_name,
            metric_config=metric_config,
            top_enabled=top_enabled,
            scene=scene,
            object_ids=object_ids,
            groups=groups,
            grouping_available=grouping_available,
            grouping_report=(
                object_grouping_report
                if isinstance(object_grouping_report, dict)
                else None
            ),
            render_evidence=render_evidence,
            camera_evidence_provider=camera_evidence_provider,
            functional_evidence_planner=functional_evidence_planner,
            functional_probe_evidence_provider=(
                functional_probe_evidence_provider
            ),
            functional_prejudgement_evidence_source=(
                functional_prejudgement_evidence_source
            ),
            functional_prejudgement_evidence_config=(
                functional_prejudgement_config
            ),
            discovery_identity_image_path=(
                discovery_identity_image_path
            ),
            discovery_identity_legend=discovery_identity_legend,
            vlm_judge=vlm_judge,
            prompt=prompt,
            visual_style_spec=visual_style_spec,
            authorized_deviations=deviations_for_metric(deviations, metric_name),
            applicability=(
                applicability.get(metric_name)
                if metric_name in applicability
                else {
                    "applicability": "pending",
                    "reason": "metric_missing_from_declared_applicability_map",
                }
                if metric_applicability is not None
                else {
                    "applicability": "pending",
                    "reason": "metric_applicability_not_declared",
                }
            ),
            prior_metric_reports=metric_reports,
        )
        _attach_metric_forced_choice_audit(metric_report)
        metric_reports[metric_name] = metric_report

    active = [entry for entry in metric_reports.values() if entry["affects_score"]]
    resolved_entries = [
        entry
        for entry in active
        if entry["status"] == "evaluated" and isinstance(entry["score"], (int, float))
    ]
    resolved_score = _weighted_metric_score(resolved_entries)
    complete = bool(active) and len(resolved_entries) == len(active)
    score = resolved_score if complete else None
    if not top_enabled:
        status, reason = "not_applicable", "disabled_by_configuration"
    elif not active:
        status, reason = "not_applicable", "no_applicable_scene_quality_metrics"
    elif complete:
        status, reason = "evaluated", None
    elif resolved_entries:
        status, reason = "partial", "one_or_more_scene_quality_metrics_unresolved"
    else:
        status, reason = "unresolved", "scene_quality_metrics_unresolved"

    eligible_count = len(active)
    resolved_count = len(resolved_entries)
    active_metric_names = [
        name
        for name in SUPPORTED_SCENE_QUALITY_METRICS
        if metric_reports[name]["affects_score"]
    ]
    resolved_metric_names = [
        name
        for name in active_metric_names
        if metric_reports[name]["status"] == "evaluated"
        and isinstance(metric_reports[name]["score"], (int, float))
    ]
    return {
        "category": SCENE_QUALITY_INTERFACE_NAMESPACE,
        "interface_version": str(resolved.get("version") or SCENE_QUALITY_INTERFACE_VERSION),
        "metric_prompt_version": L3_METRIC_PROMPT_VERSION,
        "level": "l3_scene_quality",
        "implemented": True,
        "enabled": top_enabled,
        "status": status,
        "reason": reason,
        "score": score,
        "resolved_score": resolved_score,
        "affects_score": bool(active),
        "affects_aggregation": bool(active),
        "renderer_invoked": any(
            bool(entry.get("renderer_invoked"))
            for entry in metric_reports.values()
        ),
        "preview_renderer_invoked": any(
            bool(entry.get("preview_renderer_invoked"))
            for entry in metric_reports.values()
        ),
        "preview_render_count": sum(
            int(entry.get("preview_render_count") or 0)
            for entry in metric_reports.values()
        ),
        "final_render_count": sum(
            int(entry.get("final_render_count") or 0)
            for entry in metric_reports.values()
        ),
        "camera_evidence_provider_invoked": any(
            bool(entry["evidence_request"]["provider_invoked"])
            for entry in metric_reports.values()
        ),
        "vlm_invoked": any(entry["vlm_invoked"] for entry in metric_reports.values()),
        "coverage": {
            "eligible_count": eligible_count,
            "resolved_count": resolved_count,
            "fraction": (
                resolved_count / eligible_count if eligible_count else None
            ),
            "complete": complete,
        },
        "active_metrics": active_metric_names,
        "resolved_metrics": resolved_metric_names,
        # Retained as an empty wire-compatible field. All five metrics are
        # benchmark metrics in the v2 profile.
        "experimental_metrics": {},
        "active_metric_signature": (
            "+".join(active_metric_names) if active_metric_names else "none"
        ),
        "subfamilies": {
            SEMANTIC_COHERENCE: list(SEMANTIC_COHERENCE_METRICS),
            PERCEPTUAL_VISUAL_QUALITY: list(PERCEPTUAL_VISUAL_QUALITY_METRICS),
            FUNCTIONAL_VALIDITY: list(FUNCTIONAL_VALIDITY_METRICS),
            SEMANTIC_PLACEMENT: list(SEMANTIC_PLACEMENT_METRICS),
        },
        "grouping_policy": grouping_policy_provenance(),
        "functional_prejudgement_evidence_config": deepcopy(
            functional_prejudgement_config
        ),
        "final_vlm_context_contract": list(FINAL_VLM_CONTEXT_CONTRACT),
        "evidence_workflow_vocabulary": {
            "evidence_strategy": list(EVIDENCE_STRATEGIES),
            "router_state": list(ROUTER_STATES),
            "local_trigger": list(LOCAL_TRIGGERS),
            "router_trigger_state": list(ROUTER_TRIGGER_STATES),
        },
        "hierarchy": canonical_hierarchy(),
        "metrics": metric_reports,
        "judgment_contract": {
            "evidence_first": True,
            "insufficient_evidence_result": "unresolved",
            "invalid_requires": (
                "one_or_more_significant_explicitly_identified_visible_metric_scoped_defects"
            ),
            "minor_variation_or_subjective_preference": "valid",
            "otherwise_when_evidence_sufficient": "valid",
            "self_reported_confidence": "diagnostic_uncalibrated",
            "semantic_placement_severity": {
                "levels": list(PLACEMENT_SEVERITY_LEVELS),
                "metric_verdict_and_score_unchanged": True,
            },
        },
        "authorized_deviations": deepcopy(deviations),
        "authorized_deviation_precedence": (
            "Prompt specification takes precedence over generic Scene Quality priors. "
            "If an apparent inconsistency is explicitly requested by the prompt, L2 evaluates "
            "whether the request was satisfied and L3 must not penalize that same requested deviation."
        ),
        "l2_l3_boundary": {
            "l3_canonical_semantic_coherence": list(SEMANTIC_COHERENCE_METRICS),
            "l3_canonical_perceptual_visual_quality": list(PERCEPTUAL_VISUAL_QUALITY_METRICS),
            "l3_canonical_functional_validity": list(
                FUNCTIONAL_VALIDITY_METRICS
            ),
            "l3_canonical_semantic_placement": list(
                SEMANTIC_PLACEMENT_METRICS
            ),
            "l3_namespace": SCENE_QUALITY_INTERFACE_NAMESPACE,
            "reuses_l2_evidence": False,
            "l2_question": "Did the scene follow the prompt?",
            "l3_question": "Is the scene coherent, except for prompt-authorized deviations?",
            "oor_oar_owner": "L2 specification fidelity (not L1 physical plausibility)",
            "l1_physical_owner": "collision, oob, support, navigability, accessibility",
            "functional_semantic_owner": (
                "L2 functional_semantic_fidelity; local functionality only "
                "when explicitly specified by the prompt"
            ),
            "object_pairing_scope": (
                "L3 scene_and_group category_and_role_compatibility_only; "
                "excludes position, distance, angle, orientation, and "
                "arrangement"
            ),
            "semantic_placement_owner": (
                "L3 semantic placement judges direction-independent semantic "
                "location plausibility; L1 remains the owner of collision, "
                "OOB, and support, while functional usability remains owned "
                "by L3 functional consistency"
            ),
            "functional_placement_adjacency_boundary": (
                "action-required adjacency belongs to functional consistency; "
                "context-only adjacency belongs to semantic placement"
            ),
        },
        "double_count_guard": {
            "affects_aggregate_score": bool(active),
            "reason": (
                "Scale, grouped Object Pairing, Style, Functional Consistency, "
                "and Semantic Placement are owned and scored only by current "
                "L3 Scene Quality."
            ),
        },
        "notes": [
            "Semantic Coherence, Perceptual Visual Quality, Functional Validity, and Semantic Placement are distinct L3 subfamilies.",
            "Object pairing runs after external grouping and judges target category/role compatibility with both scene and local-group context.",
            "Semantic placement is an active benchmark metric for scene- and local-context location plausibility; it excludes collision, physical support, and functional operability.",
            "Prompt-specified local functionality is owned by L2; explicit position/angle relations are owned by OOR/OAR.",
            "An L3 invalid verdict requires a significant, explicitly identified, visible metric-scoped defect; otherwise sufficient evidence resolves valid.",
            "Camera/evidence policies are configurable defaults resolved from a unified layered configuration.",
            "Missing images, judge, applicability, or required object grouping is unresolved and never scored as valid.",
            "Judge normal scene consistency except where an apparent inconsistency is explicitly "
            "requested by the prompt; do not penalize an authorized deviation and do not extend it to unrelated objects.",
        ],
    }


def _evaluate_metric(
    *,
    metric_name: str,
    metric_config: dict[str, Any],
    top_enabled: bool,
    scene: dict[str, Any],
    object_ids: list[str],
    groups: list[dict[str, Any]] | None,
    grouping_available: bool,
    grouping_report: dict[str, Any] | None,
    render_evidence: list[str] | dict[str, Any] | None,
    camera_evidence_provider: Any,
    functional_evidence_planner: Any,
    functional_probe_evidence_provider: Any,
    functional_prejudgement_evidence_source: Any,
    functional_prejudgement_evidence_config: dict[str, Any],
    discovery_identity_image_path: str | None,
    discovery_identity_legend: dict[str, str] | None,
    vlm_judge: Any,
    prompt: str | None,
    visual_style_spec: dict[str, Any] | None,
    authorized_deviations: list[dict[str, Any]],
    applicability: Any,
    prior_metric_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    policy = deepcopy(metric_config["evidence_policy"])
    evidence_plan = (
        metric_config.get("evidence_plan")
        if isinstance(metric_config.get("evidence_plan"), dict)
        else {}
    )
    json_screen_first = bool(
        metric_name
        in {"scale_consistency", "object_pairing_consistency"}
        and evidence_plan.get("evidence_strategy")
        == "json_screen_then_visual"
    )
    global_discovery_then_group_local = bool(
        metric_name
        in {
            "functional_consistency",
            "semantic_placement_consistency",
        }
        and evidence_plan.get("evidence_strategy")
        == "global_discovery_then_group_local"
    )
    style_global_screen_then_local = bool(
        metric_name == "style_consistency"
        and evidence_plan.get("evidence_strategy")
        in {
            "global_screen_then_local",
            "global_discovery_then_group_local",
        }
    )
    declared_scope = str(policy["camera_scope"])
    # Existing direct callers can still adjudicate a scale packet without
    # supplying the new grouping dependency. Canonical runs provide grouping
    # and therefore take the group-scoped branch below.
    legacy_scale_scope = bool(
        metric_name == "scale_consistency"
        and declared_scope in _GROUP_SCOPES
        and not grouping_available
    )
    if legacy_scale_scope:
        policy.update(
            camera_scope="object_local",
            include_global_context=False,
            image_order=None,
        )
    scope = str(policy["camera_scope"])
    enabled = bool(metric_config["enabled"])

    selected_object_ids: list[str] = []
    selected_group_ids: list[str] = []
    selected_groups_for_judge: list[dict[str, Any]] = []
    if scope == "global":
        eligible_count = 1 if object_ids else 0
    elif scope == "object_local":
        selected_object_ids = list(object_ids)
        eligible_count = len(object_ids)
    else:  # group_local / pair_local
        if grouping_available and groups is not None:
            eligible_groups: list[dict[str, Any]] = []
            scene_ids = set(object_ids)
            minimum_members = (
                2
                if metric_name
                in {
                    "object_pairing_consistency",
                    "style_consistency",
                    "functional_consistency",
                    "semantic_placement_consistency",
                }
                else 1
            )
            for group in groups:
                members = list(
                    dict.fromkeys(
                        str(member)
                        for member in group.get("object_ids") or []
                        if str(member) in scene_ids
                    )
                )
                if len(members) >= minimum_members:
                    eligible_groups.append({**group, "object_ids": members})
            selected_groups_for_judge = deepcopy(eligible_groups)
            selected_group_ids = [
                str(group.get("group_id"))
                for group in eligible_groups
                if group.get("group_id")
            ]
            member_ids: list[str] = []
            for group in eligible_groups:
                for member in group.get("object_ids") or []:
                    member_ids.append(str(member))
            selected_object_ids = list(dict.fromkeys(member_ids))
            eligible_count = len(selected_group_ids)
        else:
            eligible_count = 0

    applicable_state, applicability_record = _applicability_state(applicability)
    should_acquire_evidence = bool(
        top_enabled
        and enabled
        and float(metric_config.get("weight", 1.0)) > 0.0
        and applicable_state == "relevant"
        and eligible_count > 0
        and vlm_judge is not None
    )
    group_evidence_packets: list[dict[str, Any]] = []
    if scope in _GROUP_SCOPES:
        group_evidence_packets = _resolve_group_evidence_packets(
            render_evidence,
            metric_name=metric_name,
            policy=policy,
            scene=scene,
            prompt=prompt,
            groups=selected_groups_for_judge,
            grouping_report=grouping_report,
            camera_evidence_provider=(
                camera_evidence_provider
                if should_acquire_evidence and not json_screen_first
                else None
            ),
            resolve_metric_evidence=_resolve_metric_evidence,
        )
        resolved_evidence = list(
            dict.fromkeys(
                path
                for packet in group_evidence_packets
                for path in packet["paths"]
            )
        )
        evidence_resolution = _group_evidence_resolution_summary(
            group_evidence_packets
        )
    else:
        (
            resolved_evidence,
            evidence_resolution,
        ) = _resolve_metric_evidence(
            render_evidence,
            metric_name=metric_name,
            policy=policy,
            scene=scene,
            prompt=prompt,
            selected_object_ids=selected_object_ids,
            selected_group_ids=selected_group_ids,
            selected_groups=selected_groups_for_judge,
            camera_evidence_provider=(
                camera_evidence_provider
                if should_acquire_evidence and not json_screen_first
                else None
            ),
        )
    evidence_available = bool(resolved_evidence)
    available, unavailable_reason, dependencies = _dependency_state(
        scope=scope,
        grouping_available=grouping_available,
        evidence_available=evidence_available,
        provider_available=camera_evidence_provider is not None,
        evidence_resolution=evidence_resolution,
    )

    base: dict[str, Any] = {
        "metric": metric_name,
        "namespace": SCENE_QUALITY_INTERFACE_NAMESPACE,
        "family": SUBFAMILY_BY_METRIC[metric_name],
        "judgment_scope": deepcopy(JUDGMENT_SCOPE_BY_METRIC[metric_name]),
        "interface_version": SCENE_QUALITY_INTERFACE_VERSION,
        "metric_prompt_version": L3_METRIC_PROMPT_VERSION,
        "implemented": True,
        "enabled": enabled,
        "metric_status": str(
            metric_config.get("metric_status") or "canonical_scoring"
        ),
        "activation_policy": str(
            metric_config.get("activation_policy")
            or "profile_and_applicability"
        ),
        "included_in_canonical_aggregate": bool(
            metric_config.get("included_in_canonical_aggregate", True)
        ),
        "weight": float(metric_config.get("weight", 1.0)),
        "status": "unresolved",
        "reason": None,
        "score": None,
        "affects_score": False,
        "renderer_invoked": _evidence_renderer_invoked(
            evidence_resolution
        ),
        "preview_renderer_invoked": False,
        "preview_render_count": 0,
        "final_render_count": (
            _provider_render_count(evidence_resolution)
        ),
        "requested_camera_scope": scope,
        "declared_camera_scope": declared_scope,
        "compatibility_scope_fallback": (
            "scale_object_local_without_grouping"
            if legacy_scale_scope
            else None
        ),
        "resolved_evidence_policy": deepcopy(policy),
        "evidence_plan": deepcopy(metric_config.get("evidence_plan")),
        "grouping_policy": (
            grouping_policy_provenance()
            if _plan_uses_grouping(metric_config.get("evidence_plan"))
            else None
        ),
        "selected_object_ids": selected_object_ids,
        "selected_group_ids": selected_group_ids,
        "evidence_paths": list(resolved_evidence),
        "evidence_handles": [],
        "evidence_request": {
            "camera_scope": scope,
            "camera_mode": policy["camera_mode"],
            "selector": policy["selector"],
            "image_budget": policy["image_budget"],
            "global_image_budget": policy.get(
                "global_image_budget"
            ),
            "scoped_image_budget": policy.get(
                "scoped_image_budget"
            ),
            "presentation": policy["presentation"],
            "image_order": policy["image_order"],
            "include_global_context": policy["include_global_context"],
            "camera_pose_mode": policy.get("camera_pose_mode"),
            "target_object_ids": selected_object_ids,
            "target_group_ids": selected_group_ids,
            "renderer_invoked": _evidence_renderer_invoked(
                evidence_resolution
            ),
            "provider_invoked": bool(evidence_resolution["provider_invoked"]),
            "provider_status": evidence_resolution["provider_status"],
            "provider_reason": evidence_resolution["provider_reason"],
            "evidence_source": evidence_resolution["source"],
            "scope_satisfied": bool(evidence_resolution["scope_satisfied"]),
            "missing_paths": list(evidence_resolution.get("missing_paths") or []),
            "vlm_invoked": False,
            "group_requests": [
                _group_packet_audit(packet)
                for packet in group_evidence_packets
            ],
        },
        "authorized_deviations": authorized_deviations,
        "applicability": applicability_record,
        "dependencies": dependencies,
        "unavailable_reason": unavailable_reason,
        "coverage": {
            "eligible_count": eligible_count,
            "resolved_count": 0,
            "fraction": None,
            "complete": False,
        },
        "vlm_invoked": False,
        "judgement": None,
    }

    if not top_enabled or not enabled:
        base.update(status="not_applicable", reason="disabled_by_configuration")
        return base
    if float(base["weight"]) <= 0.0:
        base.update(status="not_applicable", reason="zero_metric_weight")
        return base
    if applicable_state == "not_relevant":
        base.update(status="not_applicable", reason="metric_not_relevant_for_asset_policy")
        return base
    base["affects_score"] = bool(
        base["included_in_canonical_aggregate"]
        and metric_name in SCENE_QUALITY_INTERFACE_METRICS
    )
    if applicable_state == "pending":
        base.update(status="unresolved", reason="metric_applicability_pending")
        return base
    if eligible_count == 0:
        if scope in _GROUP_SCOPES and not grouping_available:
            base.update(status="unresolved", reason="object_grouping_unavailable")
        else:
            base.update(status="not_applicable", reason="no_eligible_targets", affects_score=False)
        return base
    if vlm_judge is None:
        base.update(status="unresolved", reason="vlm_judge_not_configured")
        return base
    if json_screen_first:
        return _evaluate_json_screen_then_group_visual(
            base=base,
            metric_name=metric_name,
            metric_config=metric_config,
            scene=scene,
            object_ids=object_ids,
            groups=(
                selected_groups_for_judge
                if grouping_available
                else None
            ),
            grouping_report=grouping_report,
            render_evidence=render_evidence,
            camera_evidence_provider=camera_evidence_provider,
            vlm_judge=vlm_judge,
            prompt=prompt,
            visual_style_spec=visual_style_spec,
            authorized_deviations=authorized_deviations,
            build_judge_request=_judge_request,
            call_judge=_call_scene_quality_judge,
            apply_prompt_exemptions=_apply_prompt_exemptions,
            normalize_judgement=_normalize_judgement,
            resolve_group_evidence_packets=(
                _resolve_group_evidence_packets
            ),
            resolve_metric_evidence=_resolve_metric_evidence,
            group_packet_audit=_group_packet_audit,
            evaluate_group_scoped_judgements=(
                _evaluate_group_scoped_judgements
            ),
        )
    if scope in _GROUP_SCOPES:
        return _evaluate_group_scoped_judgements(
            base=base,
            metric_name=metric_name,
            scene=scene,
            prompt=prompt,
            packets=group_evidence_packets,
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            visual_style_spec=visual_style_spec,
            build_judge_request=_judge_request,
            call_judge=_call_scene_quality_judge,
            apply_prompt_exemptions=_apply_prompt_exemptions,
            normalize_judgement=_normalize_judgement,
            evidence_phase=(
                "initial_visual"
                if metric_name
                in {
                    "functional_consistency",
                    "semantic_placement_consistency",
                }
                else "final"
            ),
            decision_mode="final",
        )
    if not available:
        base.update(status="unresolved", reason=unavailable_reason)
        return base

    if style_global_screen_then_local:
        return _evaluate_style_global_then_group_local(
            base=base,
            metric_config=metric_config,
            scene=scene,
            object_ids=object_ids,
            groups=groups if grouping_available else None,
            grouping_report=grouping_report,
            global_evidence=resolved_evidence,
            render_evidence=render_evidence,
            camera_evidence_provider=camera_evidence_provider,
            vlm_judge=vlm_judge,
            prompt=prompt,
            visual_style_spec=visual_style_spec,
            authorized_deviations=authorized_deviations,
            build_judge_request=_judge_request,
            call_judge=_call_scene_quality_judge,
            apply_prompt_exemptions=_apply_prompt_exemptions,
            normalize_judgement=_normalize_judgement,
            resolve_group_evidence_packets=(
                _resolve_group_evidence_packets
            ),
            resolve_metric_evidence=_resolve_metric_evidence,
            group_packet_audit=_group_packet_audit,
            evaluate_group_scoped_judgements=(
                _evaluate_group_scoped_judgements
            ),
        )

    if global_discovery_then_group_local:
        return _evaluate_global_discovery_then_group_local(
            base=base,
            metric_name=metric_name,
            metric_config=metric_config,
            scene=scene,
            object_ids=object_ids,
            groups=groups,
            grouping_report=grouping_report,
            global_evidence=resolved_evidence,
            render_evidence=render_evidence,
            camera_evidence_provider=camera_evidence_provider,
            functional_evidence_planner=functional_evidence_planner,
            functional_probe_evidence_provider=(
                functional_probe_evidence_provider
            ),
            functional_prejudgement_evidence_source=(
                functional_prejudgement_evidence_source
            ),
            functional_prejudgement_evidence_config=(
                functional_prejudgement_evidence_config
            ),
            discovery_identity_image_path=(
                discovery_identity_image_path
            ),
            discovery_identity_legend=discovery_identity_legend,
            vlm_judge=vlm_judge,
            prompt=prompt,
            visual_style_spec=visual_style_spec,
            authorized_deviations=authorized_deviations,
            build_judge_request=_judge_request,
            call_judge=_call_scene_quality_judge,
            apply_prompt_exemptions=_apply_prompt_exemptions,
            normalize_judgement=_normalize_judgement,
            resolve_group_evidence_packets=(
                _resolve_group_evidence_packets
            ),
            resolve_metric_evidence=_resolve_metric_evidence,
            group_packet_audit=_group_packet_audit,
            evaluate_group_scoped_judgements=(
                _evaluate_group_scoped_judgements
            ),
            functional_ownership_ledger=(
                _resolved_functional_ownership_for_placement(
                    prior_metric_reports,
                    object_ids=object_ids,
                )
            ),
        )

    request = _judge_request(
        metric_name=metric_name,
        scene=scene,
        prompt=prompt,
        render_evidence=resolved_evidence,
        selected_object_ids=selected_object_ids,
        selected_group_ids=selected_group_ids,
        groups=selected_groups_for_judge,
        authorized_deviations=authorized_deviations,
        visual_style_spec=visual_style_spec,
    )
    base["evidence_request"]["vlm_invoked"] = True
    base["vlm_invoked"] = True
    audit_records = getattr(vlm_judge, "audit_records", None)
    audit_start = (
        len(audit_records)
        if isinstance(audit_records, list)
        else None
    )
    try:
        raw = _call_scene_quality_judge(vlm_judge, request)
        if (
            audit_start is not None
            and isinstance(audit_records, list)
            and len(audit_records) > audit_start
        ):
            _apply_controller_render_audit(
                base,
                audit_records[-1],
            )
        adjusted = _apply_prompt_exemptions(
            raw,
            metric_name=metric_name,
            authorized_deviations=authorized_deviations,
        )
        outcome = _normalize_judgement(
            adjusted,
            metric_name=metric_name,
            valid_object_ids=set(object_ids),
        )
    except Exception as exc:
        base.update(
            status="unresolved",
            reason="vlm_judge_failed",
            judgement={
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return base

    base["judgement"] = adjusted
    base["status"] = outcome["status"]
    base["reason"] = outcome["reason"]
    base["score"] = outcome["score"]
    if outcome["status"] == "evaluated":
        base["coverage"] = {
            "eligible_count": eligible_count,
            "resolved_count": eligible_count,
            "fraction": 1.0,
            "complete": True,
        }
    return base


def _resolved_functional_ownership_for_placement(
    prior_metric_reports: dict[str, dict[str, Any]],
    *,
    object_ids: list[str],
) -> dict[str, Any] | None:
    """Expose ownership only after the whole Function metric is final."""

    report = prior_metric_reports.get("functional_consistency")
    if not isinstance(report, dict) or report.get("status") != "evaluated":
        return None
    ledger = report.get("functional_ownership_ledger")
    if not isinstance(ledger, dict):
        return None
    return validate_functional_ownership_ledger(
        ledger,
        known_object_ids=object_ids,
    )


def _attach_metric_forced_choice_audit(
    report: dict[str, Any],
) -> None:
    """Expose forced binary conclusions without parsing free-form reasons."""

    events: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    key == "budget_exhaustion_forced_choice"
                    and isinstance(item, dict)
                    and item.get("applied") is True
                ):
                    events.append(deepcopy(item))
                    continue
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(report)
    unique: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for event in events:
        fingerprint = json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        unique.append(event)
    if not unique:
        report["budget_exhaustion_forced_choice"] = {
            "applied": False
        }
        return
    report["budget_exhaustion_forced_choice"] = {
        **deepcopy(unique[-1]),
        "occurrence_count": len(unique),
        "events": unique,
    }


def _dependency_state(
    *,
    scope: str,
    grouping_available: bool,
    evidence_available: bool,
    provider_available: bool,
    evidence_resolution: dict[str, Any],
) -> tuple[bool, str | None, dict[str, Any]]:
    dependencies: dict[str, Any] = {
        "render_evidence": "available" if evidence_available else "unavailable",
        "camera_evidence_provider": "available" if provider_available else "unavailable",
        "requested_evidence_scope": scope,
        "evidence_scope_satisfied": bool(evidence_resolution["scope_satisfied"]),
        "evidence_source": evidence_resolution["source"],
        "provider_status": evidence_resolution["provider_status"],
    }
    if scope in _GROUP_SCOPES:
        dependencies["object_grouping"] = "available" if grouping_available else "unavailable"
        if not grouping_available:
            return False, "object_grouping_unavailable", dependencies
        if not evidence_resolution["scope_satisfied"]:
            return (
                False,
                evidence_resolution["provider_reason"]
                or "group_local_camera_evidence_unavailable",
                dependencies,
            )
        return True, None, dependencies

    dependencies["object_grouping"] = "not_required"
    if scope == "object_local":
        if not evidence_resolution["scope_satisfied"]:
            return (
                False,
                evidence_resolution["provider_reason"]
                or "object_local_render_evidence_unavailable",
                dependencies,
            )
        return True, None, dependencies
    # global scope
    if not evidence_resolution["scope_satisfied"]:
        return (
            False,
            evidence_resolution["provider_reason"]
            or "global_render_evidence_unavailable",
            dependencies,
        )
    return True, None, dependencies


def _resolve_metric_evidence(
    value: list[str] | dict[str, Any] | None,
    *,
    metric_name: str,
    policy: dict[str, Any],
    scene: dict[str, Any],
    prompt: str | None,
    selected_object_ids: list[str],
    selected_group_ids: list[str],
    selected_groups: list[dict[str, Any]],
    camera_evidence_provider: Any,
    group_scope: GroupCameraScope | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Resolve evidence without confusing a global overview for local proof.

    A flat path list is the harness overview packet and therefore satisfies only
    a global scope. Local metrics require either a metric/scope-keyed packet or
    a successful camera-evidence-provider call. This keeps camera selection
    separate from the final judge while preventing a convenient global image
    from being silently relabeled as object/group-local evidence.
    """

    image_budget = int(policy["image_budget"])
    scope = str(policy["camera_scope"])
    global_paths: list[str] = []
    scoped_paths: list[str] = []
    source = "none"

    if isinstance(value, dict):
        selected = value.get(metric_name)
        if selected is not None:
            scoped_paths = _clean_evidence_paths(selected)
            source = "metric_keyed_input"
        else:
            selected = value.get(scope)
            if selected is not None:
                scoped_paths = _clean_evidence_paths(selected)
                source = "scope_keyed_input"
        global_paths = _clean_evidence_paths(
            value.get("global")
            or value.get("global_context")
            or value.get("default")
            or value.get("all")
        )
        if scope == "global" and not scoped_paths and global_paths:
            scoped_paths = list(global_paths)
            source = "global_keyed_input"
    else:
        global_paths = _clean_evidence_paths(value)
        if scope == "global":
            scoped_paths = list(global_paths)
            source = "flat_global_input"

    provider_invoked = False
    provider_status = "not_needed" if scoped_paths else "not_configured"
    provider_reason: str | None = None
    if not scoped_paths and camera_evidence_provider is not None:
        provider_invoked = True
        provider_result = _request_scene_quality_evidence(
            camera_evidence_provider,
            metric_name=metric_name,
            policy=policy,
            scene=scene,
            prompt=prompt,
            selected_object_ids=selected_object_ids,
            selected_group_ids=selected_group_ids,
            selected_groups=selected_groups,
            group_scope=group_scope,
            existing_global_paths=global_paths,
        )
        provider_status = provider_result["status"]
        provider_reason = provider_result["reason"]
        provider_usage = deepcopy(
            provider_result.get("provider_usage")
        )
        scoped_paths = provider_result["paths"]
        provider_global_paths = provider_result.get("global_paths") or []
        for path in provider_global_paths:
            if path not in global_paths:
                global_paths.append(path)
        if scoped_paths:
            source = "camera_evidence_provider"
    elif not scoped_paths:
        provider_reason = f"{scope}_render_evidence_unavailable"
        provider_usage = None
    else:
        provider_usage = None

    scoped_image_budget = policy.get("scoped_image_budget")
    if scoped_image_budget is not None:
        scoped_limit = int(scoped_image_budget)
        if scoped_limit < 0:
            raise ValueError(
                "scoped_image_budget must be non-negative"
            )
        scoped_paths = scoped_paths[:scoped_limit]
    global_image_budget = policy.get("global_image_budget")
    if global_image_budget is not None:
        global_limit = int(global_image_budget)
        if global_limit < 0:
            raise ValueError(
                "global_image_budget must be non-negative"
            )
        global_paths = global_paths[:global_limit]

    missing_paths = [
        path
        for path in list(dict.fromkeys([*global_paths, *scoped_paths]))
        if not Path(path).expanduser().is_file()
    ]
    if missing_paths:
        return [], {
            "scope_satisfied": False,
            "source": source,
            "provider_invoked": provider_invoked,
            "provider_status": (
                "failed" if provider_invoked else provider_status
            ),
            "provider_reason": "render_evidence_path_missing",
            "global_context_count": len(global_paths),
            "scoped_evidence_count": len(scoped_paths),
            "missing_paths": missing_paths,
            "provider_usage": provider_usage,
        }

    resolved: list[str] = []
    order = policy.get("image_order")
    include_global = bool(policy.get("include_global_context"))
    if scope == "global":
        resolved.extend(scoped_paths)
    elif include_global and isinstance(order, list) and order and str(order[0]).startswith("global"):
        resolved.extend(global_paths)
        resolved.extend(scoped_paths)
    else:
        resolved.extend(scoped_paths)
        if include_global:
            resolved.extend(global_paths)

    resolved = list(dict.fromkeys(resolved))[:image_budget]
    return resolved, {
        "scope_satisfied": bool(scoped_paths),
        "source": source,
        "provider_invoked": provider_invoked,
        "provider_status": provider_status,
        "provider_reason": provider_reason,
        "global_context_count": len(global_paths),
        "scoped_evidence_count": len(scoped_paths),
        "missing_paths": [],
        "provider_usage": provider_usage,
    }


def _clean_evidence_paths(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    clean: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip() and item not in clean:
            clean.append(item)
        elif isinstance(item, Path):
            path = str(item)
            if path and path not in clean:
                clean.append(path)
        elif isinstance(item, dict):
            path = item.get("path") or item.get("image_path")
            if isinstance(path, (str, Path)) and str(path).strip() and str(path) not in clean:
                clean.append(str(path))
    return clean


def _evidence_renderer_invoked(
    resolution: dict[str, Any],
) -> bool:
    usage = resolution.get("provider_usage")
    if not isinstance(usage, dict):
        return False
    # A cached packet is evidence reuse, not a renderer invocation in this
    # metric call. Opaque providers do not get guessed into render telemetry.
    return (
        resolution.get("provider_invoked") is True
        and usage.get("cache_hit") is False
        and bool(usage.get("evidence_refs"))
    )


def _provider_render_count(resolution: dict[str, Any]) -> int:
    if not _evidence_renderer_invoked(resolution):
        return 0
    usage = resolution.get("provider_usage")
    refs = usage.get("evidence_refs") if isinstance(usage, dict) else []
    return len(refs) if isinstance(refs, list) else 0


def _apply_controller_render_audit(
    report: dict[str, Any],
    record: Any,
) -> None:
    audit = (
        record.get("audit")
        if isinstance(record, dict)
        else None
    )
    telemetry = (
        audit.get("experiment_telemetry")
        if isinstance(audit, dict)
        else None
    )
    if not isinstance(telemetry, dict):
        return
    preview_count = int(
        telemetry.get("preview_render_count") or 0
    )
    final_count = int(telemetry.get("full_render_count") or 0)
    report["renderer_invoked"] = bool(
        report.get("renderer_invoked") or final_count
    )
    report["preview_renderer_invoked"] = preview_count > 0
    report["preview_render_count"] = preview_count
    report["final_render_count"] = final_count
    report["evidence_request"]["renderer_invoked"] = report[
        "renderer_invoked"
    ]


def _request_scene_quality_evidence(
    provider: Any,
    *,
    metric_name: str,
    policy: dict[str, Any],
    scene: dict[str, Any],
    prompt: str | None,
    selected_object_ids: list[str],
    selected_group_ids: list[str],
    selected_groups: list[dict[str, Any]],
    group_scope: GroupCameraScope | None = None,
    existing_global_paths: list[str] | None = None,
) -> dict[str, Any]:
    request = {
        "category": "scene_quality_evidence_request",
        "metric": metric_name,
        "event": {
            "type": metric_name,
            "object_ids": list(selected_object_ids),
            "group_ids": list(selected_group_ids),
        },
        "object_ids": list(selected_object_ids),
        "group_ids": list(selected_group_ids),
        "object_groups": deepcopy(selected_groups),
        "scene": deepcopy(scene),
        "scene_summary": {
            "scene_id": scene.get("scene_id"),
            "scene_type": scene.get("scene_type"),
            "object_count": len(scene.get("objects") or []),
            "architecture": architecture_contract_from_scene(scene),
        },
        "natural_language_prompt": prompt,
        "evidence_scope": str(policy["camera_scope"]),
        "evidence_policy": deepcopy(policy),
        "existing_global_evidence": list(
            existing_global_paths or []
        ),
        "global_context_mode": (
            "reuse_existing"
            if existing_global_paths
            else "not_available"
        ),
        "selection_role": "visual_evidence_only_do_not_judge_metric",
    }
    if group_scope is not None:
        scope_value = group_scope.to_dict()
        request.update(
            {
                "group_scope": scope_value,
                "member_ids": list(group_scope.member_ids),
                "target_bounds": deepcopy(
                    scope_value["target_bounds"]
                ),
                "focus_center": list(group_scope.focus_center),
                "target_extent": list(group_scope.extent),
                "evidence_goal": group_scope_evidence_goal(
                    group_scope
                ),
                "grouping_role": (
                    "primary_visual_evidence_decomposition"
                ),
            }
        )
        request["event"]["group_id"] = group_scope.group_id
        request["event"]["focus_region"] = deepcopy(
            scope_value["target_bounds"]
        )
    call = getattr(provider, "provide_scene_quality_evidence", None)
    if not callable(call) and callable(provider):
        call = provider
    if not callable(call):
        return {
            "status": "failed",
            "reason": "camera_evidence_provider_not_callable",
            "paths": [],
        }
    try:
        raw = call(request)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": "camera_evidence_provider_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "paths": [],
        }
    provider_usage = deepcopy(
        getattr(provider, "last_call_usage", None)
    )
    if isinstance(raw, dict):
        status = str(raw.get("status") or "available").strip().lower()
        if status in {"failed", "error"} or raw.get("error"):
            return {
                "status": "failed",
                "reason": "camera_evidence_provider_reported_failure",
                "error": str(raw.get("error") or status),
                "paths": [],
                "provider_usage": provider_usage,
            }
        if status in {"insufficient", "unavailable", "not_available"}:
            return {
                "status": "insufficient",
                "reason": str(raw.get("reason") or "local_render_evidence_not_available"),
                "paths": [],
                "provider_usage": provider_usage,
            }
        paths, global_paths = _split_provider_evidence(
            raw.get("render_evidence_items")
            or raw.get("paths")
            or raw.get("render_evidence"),
            requested_scope=str(policy["camera_scope"]),
        )
    else:
        paths, global_paths = _split_provider_evidence(
            [raw] if isinstance(raw, (str, Path)) else raw,
            requested_scope=str(policy["camera_scope"]),
        )
    if not paths:
        return {
            "status": "insufficient",
            "reason": "camera_evidence_provider_returned_no_evidence",
            "paths": [],
            "provider_usage": provider_usage,
        }
    return {
        "status": "available",
        "reason": None,
        "paths": paths,
        "global_paths": global_paths,
        "provider_usage": provider_usage,
    }


def _split_provider_evidence(
    value: Any,
    *,
    requested_scope: str,
) -> tuple[list[str], list[str]]:
    if not isinstance(value, (list, tuple)):
        return [], []
    scoped: list[str] = []
    global_paths: list[str] = []
    for item in value:
        if isinstance(item, dict):
            path = item.get("path") or item.get("image_path")
            role = str(item.get("role") or "").strip().lower()
            if not isinstance(path, (str, Path)) or not str(path).strip():
                continue
            destination = (
                global_paths
                if requested_scope != "global" and "global" in role
                else scoped
            )
            if str(path) not in destination:
                destination.append(str(path))
        elif isinstance(item, (str, Path)) and str(item).strip():
            if str(item) not in scoped:
                scoped.append(str(item))
    return scoped, global_paths


def _applicability_state(value: Any) -> tuple[str, dict[str, Any]]:
    if value is None:
        return "relevant", {
            "applicability": "not_declared",
            "reason": "no_metric_applicability_record",
        }
    if isinstance(value, bool):
        state = "relevant" if value else "not_relevant"
        return state, {"applicability": state, "source": "boolean_compatibility"}
    if not isinstance(value, dict):
        return "pending", {
            "applicability": "pending",
            "reason": "malformed_metric_applicability_record",
        }
    record = deepcopy(value)
    # Older asset-policy metadata described the then-placeholder evaluator.
    # Those fields cannot override this module's implementation/scoring state.
    record.pop("implemented", None)
    record.pop("affects_score", None)
    record["decision_role"] = "applicability_only"
    raw = record.get("applicability", record.get("status"))
    if raw is True or raw in ("relevant", "applicable"):
        return "relevant", record
    if raw is False or raw in ("not_relevant", "not_applicable"):
        return "not_relevant", record
    if raw in ("pending", "unknown", "unresolved"):
        return "pending", record
    return "pending", {
        **record,
        "applicability": "pending",
        "reason": record.get("reason") or "unrecognized_metric_applicability",
    }


def _weighted_metric_score(entries: list[dict[str, Any]]) -> float | None:
    if not entries:
        return None
    weighted = 0.0
    total_weight = 0.0
    for entry in entries:
        weight = float(entry.get("weight", 1.0))
        weighted += weight * float(entry["score"])
        total_weight += weight
    if total_weight <= 0.0:
        return None
    return weighted / total_weight


def _judge_request(
    *,
    metric_name: str,
    scene: dict[str, Any],
    prompt: str | None,
    render_evidence: list[str],
    selected_object_ids: list[str],
    selected_group_ids: list[str],
    groups: list[dict[str, Any]] | None,
    authorized_deviations: list[dict[str, Any]],
    visual_style_spec: dict[str, Any] | None,
    evidence_phase: str = "final",
    decision_mode: str = "final",
    group_scope: GroupCameraScope | None = None,
    routed_screen_claims: list[dict[str, Any]] | None = None,
    functional_probe_evidence: dict[str, Any] | None = None,
    placement_discovery: dict[str, Any] | None = None,
    required_placement_checks: list[dict[str, Any]] | None = None,
    functional_ownership_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    functional_visual_context = bool(
        metric_name == "functional_consistency"
        and evidence_phase
        in {
            "global_discovery",
            "cross_group_relation_review",
            "group_local_review",
            "initial_visual",
        }
    )
    objects = [
        (
            _compact_functional_object(item)
            if functional_visual_context
            else _compact_object(item)
        )
        for item in scene.get("objects", [])
        if isinstance(item, dict)
        and (
            not selected_object_ids
            or str(item.get("id")) in set(selected_object_ids)
        )
    ]
    selected_groups = [
        (
            {
                "group_id": str(group.get("group_id") or ""),
                "object_ids": [
                    str(item)
                    for item in group.get("object_ids") or []
                    if str(item).strip()
                ],
            }
            if functional_visual_context
            else deepcopy(group)
        )
        for group in groups or []
        if not selected_group_ids or str(group.get("group_id")) in set(selected_group_ids)
    ]
    required_functional_checks = (
        (
            functional_probe_evidence.get("required_checks")
            if isinstance(functional_probe_evidence, dict)
            else None
        )
        or []
    )
    placement_checks = [
        deepcopy(item)
        for item in required_placement_checks or []
        if isinstance(item, dict)
    ]
    allow_scene_wide_functional_ownership = bool(
        metric_name == "functional_consistency"
        and any(
            item.get("check_type") == "clearance"
            for item in required_functional_checks
            if isinstance(item, dict)
        )
    )
    allowed_defect_target_ids = list(
        dict.fromkeys(
            str(item)
            for item in (
                (
                    [
                        object_record.get("id")
                        for object_record in scene.get("objects") or []
                        if isinstance(object_record, dict)
                    ]
                    if allow_scene_wide_functional_ownership
                    else selected_object_ids
                )
                or [
                    object_record.get("id")
                    for object_record in scene.get("objects") or []
                    if isinstance(object_record, dict)
                ]
            )
            if str(item).strip()
        )
    )
    request = {
        "category": SCENE_QUALITY_INTERFACE_NAMESPACE,
        "metric": metric_name,
        "evidence_phase": evidence_phase,
        "decision_mode": decision_mode,
        "metric_prompt_version": L3_METRIC_PROMPT_VERSION,
        "metric_boundary_rules": list(L3_METRIC_BOUNDARY_RULES),
        "metric_rubric": METRIC_RUBRICS[metric_name],
        "judgment_scope": deepcopy(JUDGMENT_SCOPE_BY_METRIC[metric_name]),
        "event": {
            "type": metric_name,
            "object_ids": list(selected_object_ids),
            "group_ids": list(selected_group_ids),
        },
        "prompt": prompt,
        "natural_language_prompt": prompt,
        "camera_scene_context": deepcopy(scene),
        "scene_summary": {
            "scene_id": scene.get("scene_id"),
            "scene_type": scene.get("scene_type"),
            "boundary": deepcopy(scene.get("boundary")),
            "scene_height": scene.get("scene_height"),
            "architecture": architecture_contract_from_scene(scene),
            "object_count": len(scene.get("objects") or []),
            "objects": objects,
        },
        "target_object_ids": list(selected_object_ids),
        "target_group_ids": list(selected_group_ids),
        "object_groups": selected_groups,
        "functional_probe_evidence": deepcopy(
            functional_probe_evidence
        ),
        "placement_discovery": deepcopy(placement_discovery),
        "required_placement_checks": placement_checks,
        "functional_ownership_ledger": deepcopy(
            functional_ownership_ledger
        ),
        "structured_context_policy": (
            {
                "object_fields": ["id", "category"],
                "group_fields": ["group_id", "object_ids"],
                "excluded_object_fields": [
                    "center",
                    "size",
                    "rotation",
                    "description",
                ],
                "reason": (
                    "functional visual grounding must not be replaced by "
                    "transform or grouping-prose shortcuts"
                ),
            }
            if functional_visual_context
            else None
        ),
        "defect_attribution": (
            {
                "unit": "object",
                "target_ids_semantics": (
                    "exact scoring owner objects only; for Functional "
                    "clearance this is the validated causal blocker or the "
                    "affected object for self-layout, and for Placement this "
                    "is the typed check subject; never use the whole evidence "
                    "group as shorthand"
                ),
                "cross_phase_deduplication_key": [
                    "metric",
                    "object_id",
                ],
                "cross_metric_deduplication": False,
            }
            if metric_name
            in {
                "style_consistency",
                "functional_consistency",
                "semantic_placement_consistency",
            }
            else None
        ),
        "routed_screen_claims": deepcopy(
            routed_screen_claims or []
        ),
        "authorized_deviations": deepcopy(authorized_deviations),
        "visual_style_spec": (
            deepcopy(visual_style_spec)
            if metric_name == "style_consistency" and isinstance(visual_style_spec, dict)
            else None
        ),
        "render_evidence": list(render_evidence),
        "response_contract": {
            "evidence_status": ["sufficient", "insufficient"],
            "verdict": ["valid", "invalid", "ambiguous"],
            "invalid_requires_significant_metric_scoped_defect": True,
            "insufficient_requires_ambiguous": True,
            "allowed_target_ids": allowed_defect_target_ids,
            "defect_attribution_unit": (
                "object"
                if metric_name
                in {
                    "style_consistency",
                    "functional_consistency",
                    "semantic_placement_consistency",
                }
                else "claim"
            ),
            "defects": {
                "required_when_invalid": True,
                "fields": ["scope", "target_ids", "relation", "reason"],
                "allowed_scopes": list(
                    JUDGMENT_SCOPE_BY_METRIC[metric_name]["included"]
                ),
                "allowed_target_ids": allowed_defect_target_ids,
            },
        },
        **vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.CANONICAL_METRIC,
            judge_method="adjudicate_scene_quality",
        ),
    }
    if (
        metric_name == "functional_consistency"
        and required_functional_checks
    ):
        request["required_functional_checks"] = deepcopy(
            required_functional_checks
        )
        request["response_contract"]["functional_check_results"] = {
            "required": True,
            "exact_check_ids": [
                str(item.get("check_id") or "")
                for item in required_functional_checks
                if isinstance(item, dict)
            ],
            "fields": [
                "check_id",
                "target_ids",
                "observation_status",
                "conclusion",
                "reason",
            ],
            "conditional_invalid_clearance_fields": [
                "affected_object_ids",
                "cause_kind",
                "causal_object_ids",
                "scoring_target_ids",
            ],
            "observation_status": [
                "observed",
                "inferred_under_budget",
                "missing",
            ],
            "conclusion": ["valid", "invalid", "unresolved"],
            "invalid_defect_linkage": {
                "field": "check_refs",
                "coverage": "every_invalid_check_exactly_once",
                "multiple_refs_allowed_only_for_one_physical_defect": True,
            },
        }
    if metric_name == "semantic_placement_consistency":
        request["response_contract"]["defects"]["fields"].extend(
            [
                "check_id",
                "placement_check_type",
                "severity",
            ]
        )
        request["response_contract"]["defects"][
            "allowed_field_values"
        ] = {
            "severity": list(PLACEMENT_SEVERITY_LEVELS),
        }
        request["placement_severity_policy"] = {
            "schema_version": "semantic_placement_severity_v1",
            "levels": list(PLACEMENT_SEVERITY_LEVELS),
            "metric_verdict_unchanged": True,
        }
        request["placement_check_policy"] = {
            "schema_version": "placement_check_results_v1",
            "allowed_check_types": [
                "support_and_height",
                "scene_zone",
                "contextual_anchor",
            ],
            "discovery_is_routing_prior_only": True,
            "baseline_judge_may_register_discovery_miss": True,
            "defect_owner": "subject_id_only",
            "context_ids_are_non_owning": True,
            "function_exclusion_requires_exact_event_ref": True,
        }
        request["response_contract"]["placement_check_results"] = {
            "required": bool(placement_checks),
            "exact_check_ids": [
                str(item.get("check_id") or "")
                for item in placement_checks
            ],
            "fields": [
                "check_id",
                "subject_id",
                "context_ids",
                "observation_status",
                "conclusion",
                "reason",
            ],
            "observation_status": [
                "observed",
                "inferred_under_budget",
                "missing",
            ],
            "conclusion": [
                "valid",
                "invalid",
                "excluded_function_owned",
                "unresolved",
            ],
        }
        request["response_contract"][
            "judge_originated_placement_results"
        ] = {
            "purpose": "strictly_typed_discovery_miss_recovery",
            "same_call_resolution_requires_current_evidence": True,
            "insufficient_evidence_requires": (
                "evidence_request.metadata.placement_check_proposal"
            ),
        }
    if allow_scene_wide_functional_ownership:
        request["allowed_external_evidence_target_ids"] = list(
            allowed_defect_target_ids
        )
        request["causal_object_catalog"] = [
            {
                "id": str(item.get("id") or ""),
                "category": str(
                    item.get("category")
                    or item.get("retrieval_category")
                    or "unknown"
                ),
            }
            for item in scene.get("objects") or []
            if isinstance(item, dict) and item.get("id")
        ]
    if group_scope is not None:
        scope_value = group_scope.to_dict()
        request["scene_summary"]["group_scope"] = deepcopy(
            {
                "group_id": scope_value.get("group_id"),
                "member_ids": deepcopy(
                    scope_value.get("member_ids") or []
                ),
            }
            if functional_visual_context
            else scope_value
        )
        request.update(
            {
                "group_scope": scope_value,
                "member_ids": list(group_scope.member_ids),
                "target_bounds": deepcopy(
                    scope_value["target_bounds"]
                ),
                "focus_center": list(group_scope.focus_center),
                "target_extent": list(group_scope.extent),
                "evidence_goal": group_scope_evidence_goal(
                    group_scope
                ),
                "grouping_role": (
                    "primary_visual_evidence_decomposition"
                ),
            }
        )
        request["event"]["group_id"] = group_scope.group_id
        request["event"]["focus_region"] = deepcopy(
            scope_value["target_bounds"]
        )
    return request


def _compact_object(value: dict[str, Any]) -> dict[str, Any]:
    proxy = value.get("asset_proxy") if isinstance(value.get("asset_proxy"), dict) else {}
    return {
        "id": value.get("id"),
        "category": value.get("category") or value.get("retrieval_category"),
        "description": value.get("description") or value.get("desc"),
        "center": deepcopy(value.get("center")),
        "size": deepcopy(value.get("size") or proxy.get("bbox_size")),
        "rotation": deepcopy(value.get("rotation")),
    }


def _compact_functional_object(
    value: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": value.get("id"),
        "category": (
            value.get("category")
            or value.get("retrieval_category")
        ),
    }


def _call_scene_quality_judge(judge: Any, request: dict[str, Any]) -> dict[str, Any]:
    call = None
    if str(request.get("decision_mode") or "").lower() == "screen":
        call = getattr(judge, "screen_scene_quality", None)
    if not callable(call):
        call = getattr(judge, "adjudicate_scene_quality", None)
    if not callable(call):
        call = getattr(judge, "evaluate", judge)
    if not callable(call):
        raise TypeError(
            "vlm_judge must be callable or expose "
            "adjudicate_scene_quality(request)/evaluate(request)"
        )
    result = call(request)
    if not isinstance(result, dict):
        raise ValueError("scene-quality VLM response must be a JSON object")
    return deepcopy(result)


def _apply_prompt_exemptions(
    judgement: dict[str, Any],
    *,
    metric_name: str,
    authorized_deviations: list[dict[str, Any]],
) -> dict[str, Any]:
    adjusted = deepcopy(judgement)
    defects = adjusted.get("defects")
    if not isinstance(defects, list):
        return adjusted

    retained: list[Any] = []
    exempted: list[dict[str, Any]] = []
    out_of_scope: list[dict[str, Any]] = []
    excluded_scopes = set(JUDGMENT_SCOPE_BY_METRIC[metric_name].get("excluded") or [])
    for raw_defect in defects:
        if not isinstance(raw_defect, dict):
            retained.append(raw_defect)
            continue
        scope = raw_defect.get("scope") or raw_defect.get("type")
        if metric_name == "object_pairing_consistency" and scope in excluded_scopes:
            out_of_scope.append(deepcopy(raw_defect))
            continue
        target_ids = raw_defect.get("target_ids")
        relation = raw_defect.get("relation")
        if (
            isinstance(target_ids, list)
            and target_ids
            and isinstance(relation, str)
            and relation
            and any(
                deviation_matches(
                    deviation,
                    metric=metric_name,
                    target_ids=[str(item) for item in target_ids],
                    relation=relation,
                )
                for deviation in authorized_deviations
            )
        ):
            exempted.append(deepcopy(raw_defect))
            continue
        retained.append(deepcopy(raw_defect))

    adjusted["defects"] = retained
    if exempted:
        adjusted["prompt_authorized_defects"] = exempted
    if out_of_scope:
        adjusted["out_of_scope_defects"] = out_of_scope
    removed_defects = [*exempted, *out_of_scope]
    if removed_defects:
        functional_rows = adjusted.get("functional_check_results")
        if isinstance(functional_rows, list):
            removed_check_refs = {
                str(check_ref)
                for defect in removed_defects
                if isinstance(defect, dict)
                for check_ref in defect.get("check_refs") or []
                if str(check_ref).strip()
            }
            legacy_removed_target_sets = {
                tuple(
                    sorted(
                        str(item)
                        for item in defect.get("target_ids") or []
                    )
                )
                for defect in removed_defects
                if isinstance(defect, dict)
                and not defect.get("check_refs")
            }
            for row in functional_rows:
                if not isinstance(row, dict) or row.get(
                    "conclusion"
                ) != "invalid":
                    continue
                check_id = str(row.get("check_id") or "")
                legacy_match = any(
                    removed_targets
                    and set(removed_targets)
                    <= {
                        str(item)
                        for item in row.get("target_ids") or []
                    }
                    for removed_targets in legacy_removed_target_sets
                )
                if check_id in removed_check_refs or legacy_match:
                    row["conclusion"] = "valid"
                    row["reason"] = (
                        "The observed condition is covered by an authorized "
                        "deviation or excluded metric scope."
                    )
        placement_rows = adjusted.get("placement_check_results")
        if isinstance(placement_rows, list):
            removed_check_ids = {
                str(defect.get("check_id"))
                for defect in removed_defects
                if isinstance(defect, dict) and defect.get("check_id")
            }
            for row in placement_rows:
                if (
                    isinstance(row, dict)
                    and str(row.get("check_id") or "")
                    in removed_check_ids
                    and row.get("conclusion") == "invalid"
                ):
                    row["conclusion"] = "valid"
                    row["reason"] = (
                        "The observed condition is covered by an authorized "
                        "deviation or excluded metric scope."
                    )
    if (
        adjusted.get("verdict") == "invalid"
        and not retained
        and bool(exempted or out_of_scope)
    ):
        adjusted["original_verdict"] = "invalid"
        adjusted["verdict"] = "valid"
        adjusted["reason"] = (
            "No significant in-scope defect remains after applying exact "
            "prompt-authorized deviations and metric boundaries."
        )
    return adjusted


def _normalize_judgement(
    value: dict[str, Any],
    *,
    metric_name: str,
    valid_object_ids: set[str],
) -> dict[str, Any]:
    confidence = value.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("scene-quality VLM confidence must be between 0 and 1")

    evidence_status = value.get("evidence_status")
    if evidence_status not in {"sufficient", "insufficient"}:
        raise ValueError(
            "scene-quality VLM evidence_status must be 'sufficient' or 'insufficient'"
        )
    defects = value.get("defects")
    if not isinstance(defects, list):
        raise ValueError("scene-quality VLM defects must be a JSON list")
    missing_evidence = value.get("missing_evidence")
    if not isinstance(missing_evidence, list):
        raise ValueError(
            "scene-quality VLM missing_evidence must be a JSON list"
        )
    if evidence_status == "insufficient":
        if value.get("verdict") != "ambiguous":
            raise ValueError(
                "insufficient scene-quality evidence requires verdict='ambiguous'"
            )
        if not missing_evidence or any(
            not isinstance(item, str) or not item.strip()
            for item in missing_evidence
        ):
            raise ValueError(
                "insufficient scene-quality evidence must name missing evidence"
            )
        if defects:
            raise ValueError(
                "insufficient scene-quality evidence cannot assert visible defects"
            )
        return {
            "status": "unresolved",
            "score": None,
            "reason": "insufficient_visual_evidence",
        }

    verdict = value.get("verdict")
    if evidence_status == "sufficient" and missing_evidence:
        raise ValueError(
            "sufficient scene-quality evidence cannot retain missing_evidence"
        )
    if verdict == "ambiguous":
        return {
            "status": "unresolved",
            "score": None,
            "reason": "ambiguous_scene_quality_judgement",
        }
    if verdict in {"valid", "invalid"}:
        if verdict == "valid" and defects:
            raise ValueError(
                "a valid scene-quality verdict cannot retain defect records"
            )
        if verdict == "invalid":
            if not str(value.get("reason") or "").strip():
                raise ValueError(
                    "an invalid scene-quality verdict must explicitly identify a "
                    "significant metric-scoped defect"
                )
            if not defects:
                raise ValueError(
                    "an invalid scene-quality verdict requires one or more "
                    "structured metric-scoped defects"
                )
            for defect in defects:
                if not isinstance(defect, dict):
                    raise ValueError(
                        "scene-quality VLM defects must contain JSON objects"
                    )
                if not str(defect.get("scope") or "").strip():
                    raise ValueError(
                        "scene-quality VLM defects must identify their metric scope"
                    )
                if not str(defect.get("reason") or "").strip():
                    raise ValueError(
                        "scene-quality VLM defects must explain the significant defect"
                    )
                target_ids = defect.get("target_ids")
                if (
                    not isinstance(target_ids, list)
                    or not target_ids
                    or any(
                        not isinstance(item, str) or not item.strip()
                        for item in target_ids
                    )
                ):
                    raise ValueError(
                        "scene-quality VLM defects must identify non-empty target_ids"
                    )
                unknown_targets = sorted(set(target_ids) - valid_object_ids)
                if unknown_targets:
                    raise ValueError(
                        "scene-quality VLM defects reference unknown target IDs "
                        f"{unknown_targets}"
                    )
                if not str(defect.get("relation") or "").strip():
                    raise ValueError(
                        "scene-quality VLM defects must identify the defective relation"
                    )
                allowed_scopes = set(
                    JUDGMENT_SCOPE_BY_METRIC[metric_name].get("included") or []
                )
                if defect.get("scope") not in allowed_scopes:
                    raise ValueError(
                        "scene-quality VLM defect scope is outside the canonical "
                        f"{metric_name} boundary"
                    )
                if metric_name == "semantic_placement_consistency":
                    validate_placement_defect_severity(defect)
        return {
            "status": "evaluated",
            "score": 1.0 if verdict == "valid" else 0.0,
            "reason": None,
        }
    raise ValueError(
        "scene-quality VLM verdict must be valid, invalid, or ambiguous"
    )


def _scene_object_ids(scene: dict[str, Any]) -> list[str]:
    objects = scene.get("objects")
    if not isinstance(objects, list):
        return []
    ids: list[str] = []
    for item in objects:
        if isinstance(item, dict) and item.get("id") is not None:
            ids.append(str(item["id"]))
    return ids


def _normalize_groups(
    object_grouping_report: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    valid_object_ids: set[str],
) -> list[dict[str, Any]] | None:
    """Accept an object-grouping report or a bare list of groups.

    Returns ``None`` when grouping is unavailable so callers can record an
    explicit unavailable state. Grouping is consumed read-only; this module never
    re-implements grouping.
    """

    if object_grouping_report is None:
        return None
    if isinstance(object_grouping_report, dict):
        if (
            object_grouping_report.get("status") == "unavailable"
            and object_grouping_report.get("object_groups") is None
        ):
            return None
        groups = object_grouping_report.get("object_groups")
    else:
        groups = object_grouping_report
    if not isinstance(groups, list):
        raise ValueError(
            "object_grouping_report must contain an object_groups list"
        )
    normalized: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    assigned_ids: set[str] = set()
    for index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise ValueError(
                f"object_grouping_report.object_groups[{index - 1}] must be an object"
            )
        members = group.get("object_ids")
        if not isinstance(members, list):
            raise ValueError(
                f"object_grouping_report.object_groups[{index - 1}].object_ids must be a list"
            )
        group_id = str(group.get("group_id") or "").strip()
        if not group_id:
            raise ValueError(
                f"object_grouping_report.object_groups[{index - 1}].group_id is required"
            )
        if group_id in seen_group_ids:
            raise ValueError(
                f"object_grouping_report contains duplicate group_id {group_id!r}"
            )
        seen_group_ids.add(group_id)
        clean_members = [
            str(member)
            for member in members
            if isinstance(member, (str, int)) and str(member)
        ]
        if len(clean_members) != len(members) or len(clean_members) != len(set(clean_members)):
            raise ValueError(
                f"object_grouping_report group {group_id!r} contains malformed or duplicate object IDs"
            )
        unknown = sorted(set(clean_members) - valid_object_ids)
        if unknown:
            raise ValueError(
                f"object_grouping_report group {group_id!r} references unknown object IDs {unknown}"
            )
        overlap = sorted(set(clean_members) & assigned_ids)
        if overlap:
            raise ValueError(
                f"object_grouping_report assigns object IDs to multiple groups: {overlap}"
            )
        assigned_ids.update(clean_members)
        normalized.append(
            {
                **deepcopy(group),
                "group_id": group_id,
                "object_ids": clean_members,
            }
        )
    missing = sorted(valid_object_ids - assigned_ids)
    if missing:
        raise ValueError(
            "object_grouping_report must assign every scene object exactly once; "
            f"missing {missing}"
        )
    return normalized


def _plan_uses_grouping(plan: dict[str, Any] | None) -> bool:
    return bool(isinstance(plan, dict) and isinstance(plan.get("local_policy"), dict))
