"""Canonical L3 Scene Quality evaluation.

L3 Scene Quality reframes the former narrow "Visual Quality" layer into two
subfamilies:

- **L3a Semantic Coherence** — ``scale_consistency`` and
  ``object_pairing_consistency``;
- **L3b Perceptual Visual Quality** — ``style_consistency``.
- **L3c Functional Validity** — optional ``functional_consistency``;
- **L3d Semantic Placement** — optional
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

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.architecture_policy import architecture_contract_from_scene
from benchmark.evaluator.scene_quality.authorized_deviations import (
    deviation_matches,
    deviations_for_metric,
    validate_authorized_deviations,
)
from benchmark.evaluator.scene_quality.group_scoped import (
    evaluate_group_scoped_judgements as _evaluate_group_scoped_judgements,
    group_evidence_resolution_summary as _group_evidence_resolution_summary,
    group_packet_audit as _group_packet_audit,
    resolve_group_evidence_packets as _resolve_group_evidence_packets,
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
    GROUPING_POLICY_ID,
    LOCAL_TRIGGERS,
    ROUTER_STATES,
    ROUTER_TRIGGER_STATES,
    canonical_hierarchy,
    grouping_policy_provenance,
    validate_evidence_plan,
)
from benchmark.rendering.camera_pose import CAMERA_POSE_MODES
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
    L3_METRIC_RUBRICS,
)


SCENE_QUALITY_INTERFACE_VERSION = "scene_quality_v3"

# Canonical L3 output namespace.
SCENE_QUALITY_INTERFACE_NAMESPACE = "l3_scene_quality"
_RETIRED_CONFIG_NAMESPACES = ("scene_quality_interfaces", "visual_quality_interfaces")
_RETIRED_METRIC_NAMES = ("object_coexistence_consistency",)

# Subfamilies of L3 Scene Quality.
SEMANTIC_COHERENCE = "semantic_coherence"
PERCEPTUAL_VISUAL_QUALITY = "perceptual_visual_quality"
FUNCTIONAL_VALIDITY = "functional_validity"
SEMANTIC_PLACEMENT = "semantic_placement"
EXPERIMENTAL_NON_SCORING = "experimental_non_scoring"
SEMANTIC_COHERENCE_METRICS = ("scale_consistency", "object_pairing_consistency")
PERCEPTUAL_VISUAL_QUALITY_METRICS = ("style_consistency",)
FUNCTIONAL_VALIDITY_METRICS = ("functional_consistency",)
SEMANTIC_PLACEMENT_METRICS = ("semantic_placement_consistency",)
# The frozen profile and aggregate keep the original three L3 metrics. Functional
# and semantic-placement consistency are additive, explicitly enabled diagnostic
# interfaces until the benchmark hierarchy assigns them aggregate ownership.
SCENE_QUALITY_INTERFACE_METRICS = (
    "style_consistency",
    "scale_consistency",
    "object_pairing_consistency",
)
SUPPORTED_SCENE_QUALITY_METRICS = (
    *SCENE_QUALITY_INTERFACE_METRICS,
    "functional_consistency",
    "semantic_placement_consistency",
)
EXPERIMENTAL_SCENE_QUALITY_METRICS = (
    "functional_consistency",
    "semantic_placement_consistency",
)
SUBFAMILY_BY_METRIC = {
    "style_consistency": PERCEPTUAL_VISUAL_QUALITY,
    "scale_consistency": SEMANTIC_COHERENCE,
    "object_pairing_consistency": SEMANTIC_COHERENCE,
    "functional_consistency": FUNCTIONAL_VALIDITY,
    "semantic_placement_consistency": SEMANTIC_PLACEMENT,
}
JUDGMENT_SCOPE_BY_METRIC = {
    "style_consistency": {
        "included": ["significant_visible_style_incompatibility"],
        "excluded": ["minor_variation", "subjective_preference"],
    },
    "scale_consistency": {
        "included": ["significant_visible_category_relative_scale_incoherence"],
        "excluded": ["ordinary_product_size_variation", "minor_ratio_difference"],
    },
    "object_pairing_consistency": {
        "included": [
            "scene_member_category_compatibility",
            "scene_member_role_compatibility",
            "group_member_category_compatibility",
            "group_member_role_compatibility",
        ],
        "excluded": [
            "position",
            "distance",
            "angle",
            "orientation",
            "access",
            "functional_arrangement",
        ],
        "prerequisite": "object_grouping_report",
    },
    "functional_consistency": {
        "included": [
            "group_real_world_usability",
            "interaction_side_accessibility",
            "opening_clearance",
            "orientation_for_use",
            "ensemble_operability",
        ],
        "excluded": [
            "prompt_fidelity",
            "category_pairing",
            "style",
            "scale",
            "exact_relation_fidelity",
        ],
        "prerequisite": "object_grouping_report",
    },
    "semantic_placement_consistency": {
        "included": [
            "semantically_inappropriate_support_surface",
            "implausible_placement_height",
            "semantically_inappropriate_scene_zone",
            "implausible_local_context",
        ],
        "excluded": [
            "collision",
            "penetration",
            "out_of_bounds",
            "physical_support",
            "contact_stability",
            "prompt_fidelity",
            "category_pairing",
            "style",
            "scale",
            "orientation_for_use",
            "accessibility",
            "opening_clearance",
            "ensemble_operability",
            "exact_relation_fidelity",
        ],
        "prerequisite": "object_grouping_report",
    },
}

METRIC_RUBRICS = L3_METRIC_RUBRICS

# Camera-policy vocabularies. Reuse the repository's evidence vocabulary rather
# than inventing a parallel abstraction. ``camera_pose_mode`` (optional) bridges
# to the renderer via the existing ``CAMERA_POSE_MODES`` enum without hardcoding
# any pose here.
CAMERA_SCOPES = ("global", "object_local", "group_local", "pair_local")
CAMERA_MODES = ("global_top", "global_oblique", "metric_local")
EVIDENCE_SELECTORS = ("deterministic", "vlm_selector")
PRESENTATIONS = ("raw", "highlight")
IMAGE_ORDER_TOKENS = (
    "global_context",
    "global_top",
    "global_oblique",
    "metric_local",
    "object_local",
    "group_local",
    "pair_local",
)

# Local scopes whose evidence request is scoped to one or more object groups.
_GROUP_SCOPES = ("group_local", "pair_local")

# Recommended initial per-metric default policies. Defaults only; every field is
# overridable through the layered config resolution below.
DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "implemented": True,
    "version": SCENE_QUALITY_INTERFACE_VERSION,
    "evidence_policy_defaults": {},
    "metrics": {
        "style_consistency": {
            "enabled": True,
            "implemented": True,
            "weight": 1.0 / 3.0,
            "evidence_policy": {
                "camera_scope": "global",
                "camera_mode": "global_top",
                "selector": "deterministic",
                "image_budget": 1,
                "presentation": "raw",
                "image_order": None,
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": "global_screen_then_local",
                "global_policy": {
                    "view_family": "global_top",
                    "image_budget": 1,
                    "top_down": True,
                },
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": GROUPING_POLICY_ID,
                    "image_budget": 1,
                    "max_packet_images": 3,
                    "image_order": [
                        "global_context",
                        "group_local",
                    ],
                    "trigger_states": [
                        "suspicious",
                        "insufficient_evidence",
                    ],
                },
                "router_options": {
                    "global_screen_then_local": {
                        "router": "vlm_global_screen",
                        "trigger_states": [
                            "suspicious",
                            "insufficient_evidence",
                        ],
                    },
                },
                "text_context": [
                    "original_prompt",
                    "parsed_prompt_requirements",
                    "authorized_deviations",
                    "asset_policy",
                ],
            },
        },
        "scale_consistency": {
            "enabled": True,
            "implemented": True,
            "weight": 1.0 / 3.0,
            "evidence_policy": {
                "camera_scope": "group_local",
                "camera_mode": "metric_local",
                "selector": "deterministic",
                "image_budget": 3,
                "global_image_budget": 1,
                "scoped_image_budget": 1,
                "presentation": "raw",
                "image_order": ["global_context", "group_local"],
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": "json_screen_then_visual",
                "global_policy": {
                    "view_family": "global_perspective",
                    "image_budget": 1,
                    "top_down": False,
                },
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": GROUPING_POLICY_ID,
                    "image_budget": 1,
                    "max_packet_images": 3,
                    "trigger_states": [
                        "suspicious",
                        "insufficient_evidence",
                    ],
                },
                "router_options": {
                    "json_screen_then_visual": {
                        "router": "vlm_json_screen",
                        "trigger_states": [
                            "suspicious",
                            "insufficient_evidence",
                        ],
                    },
                },
                "text_context": [
                    "original_prompt",
                    "parsed_prompt_requirements",
                    "authorized_deviations",
                    "asset_policy",
                    "object_grouping_report",
                ],
            },
        },
        "object_pairing_consistency": {
            "enabled": True,
            "implemented": True,
            "weight": 1.0 / 3.0,
            "evidence_policy": {
                "camera_scope": "group_local",
                "camera_mode": "metric_local",
                "selector": "deterministic",
                "image_budget": 3,
                "global_image_budget": 1,
                "scoped_image_budget": 1,
                "presentation": "raw",
                "image_order": ["global_context", "group_local"],
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": "json_screen_then_visual",
                "global_policy": {
                    "view_family": "wall_occlusion_aware_room_perspective",
                    "image_budget": 1,
                    "top_down": False,
                },
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": GROUPING_POLICY_ID,
                    "image_budget": 1,
                    "max_packet_images": 3,
                    "trigger_states": [
                        "suspicious",
                        "insufficient_evidence",
                    ],
                },
                "router_options": {
                    "json_screen_then_visual": {
                        "router": "vlm_json_screen",
                        "trigger_states": [
                            "suspicious",
                            "insufficient_evidence",
                        ],
                    },
                },
                "text_context": [
                    "original_prompt",
                    "parsed_prompt_requirements",
                    "authorized_deviations",
                    "asset_policy",
                    "object_grouping_report",
                ],
            },
        },
        # Additive experimental metric. It is implemented and can be enabled
        # explicitly without changing the frozen three-metric L3 weights.
        "functional_consistency": {
            "enabled": False,
            "implemented": True,
            "metric_status": EXPERIMENTAL_NON_SCORING,
            "activation_policy": "explicit_config_only",
            "included_in_canonical_aggregate": False,
            # A positive local weight lets explicit diagnostic runs execute.
            # The evaluator still excludes this optional interface from the
            # frozen canonical aggregate below.
            "weight": 1.0,
            "evidence_policy": {
                "camera_scope": "group_local",
                "camera_mode": "metric_local",
                "selector": "deterministic",
                "image_budget": 3,
                "global_image_budget": 1,
                "scoped_image_budget": 1,
                "presentation": "raw",
                "image_order": ["global_context", "group_local"],
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": "global_and_local",
                "global_policy": {
                    "view_family": "global_perspective",
                    "image_budget": 1,
                    "top_down": False,
                },
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": GROUPING_POLICY_ID,
                    "image_budget": 1,
                },
                "router_options": None,
                "text_context": [
                    "original_prompt",
                    "authorized_deviations",
                    "asset_policy",
                    "object_grouping_report",
                ],
            },
        },
        # Additive experimental metric. This concerns semantic location only;
        # L1 remains the sole owner of collision and physical support.
        "semantic_placement_consistency": {
            "enabled": False,
            "implemented": True,
            "metric_status": EXPERIMENTAL_NON_SCORING,
            "activation_policy": "explicit_config_only",
            "included_in_canonical_aggregate": False,
            "weight": 1.0,
            "evidence_policy": {
                "camera_scope": "group_local",
                "camera_mode": "metric_local",
                "selector": "deterministic",
                "image_budget": 3,
                "global_image_budget": 1,
                "scoped_image_budget": 1,
                "presentation": "raw",
                "image_order": ["global_context", "group_local"],
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": "global_and_local",
                "global_policy": {
                    "view_family": "global_perspective",
                    "image_budget": 1,
                    "top_down": False,
                },
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": GROUPING_POLICY_ID,
                    "image_budget": 1,
                },
                "router_options": None,
                "text_context": [
                    "original_prompt",
                    "authorized_deviations",
                    "asset_policy",
                    "object_grouping_report",
                ],
            },
        },
    },
}


class SceneQualityInterfaceConfigError(ValueError):
    """Raised when a Scene Quality interface configuration is malformed."""


def resolve_scene_quality_config(
    config: dict[str, Any] | None = None,
    *,
    profile: dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the layered Scene Quality interface configuration.

    Precedence, lowest to highest: built-in defaults, canonical downstream
    profile override (``profile['l3_scene_quality']``), explicit canonical
    config, and per-run override. Retired namespace and metric aliases are
    rejected rather than silently normalized, so canonical configuration has
    one unambiguous input vocabulary. Unknown, future-compatible fields are
    preserved while every known field is validated.
    """

    layers: list[dict[str, Any]] = []
    if isinstance(profile, dict):
        retired_namespaces = sorted(set(profile) & set(_RETIRED_CONFIG_NAMESPACES))
        if retired_namespaces:
            raise SceneQualityInterfaceConfigError(
                "retired L3 profile namespaces are not accepted by the canonical "
                f"resolver: {retired_namespaces}; use {SCENE_QUALITY_INTERFACE_NAMESPACE!r}"
            )
        profile_section = profile.get(SCENE_QUALITY_INTERFACE_NAMESPACE)
        if profile_section is not None:
            layers.append(_normalize_layer(profile_section, "profile override"))
    if config is not None:
        layers.append(_normalize_layer(config, "config override"))
    if run_overrides is not None:
        layers.append(_normalize_layer(run_overrides, "run override"))

    resolved = deepcopy(DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG)
    for layer in layers:
        resolved = _deep_merge(resolved, layer)

    # The global evidence-policy override sits between the built-in per-metric
    # defaults (lowest) and an explicit per-metric override (highest), so it is
    # accumulated separately rather than collapsed into the built-in defaults.
    global_defaults: dict[str, Any] = {}
    explicit_metric_policies: dict[str, dict[str, Any]] = {
        name: {} for name in SUPPORTED_SCENE_QUALITY_METRICS
    }
    for layer in layers:
        layer_defaults = layer.get("evidence_policy_defaults")
        if layer_defaults is not None:
            global_defaults = _deep_merge(
                global_defaults, _as_object(layer_defaults, "evidence_policy_defaults")
            )
        layer_metrics = layer.get("metrics")
        if layer_metrics is not None:
            layer_metrics = _as_object(layer_metrics, "metrics")
            for name in SUPPORTED_SCENE_QUALITY_METRICS:
                metric_patch = layer_metrics.get(name)
                if isinstance(metric_patch, dict) and metric_patch.get("evidence_policy") is not None:
                    explicit_metric_policies[name] = _deep_merge(
                        explicit_metric_policies[name],
                        _as_object(metric_patch["evidence_policy"], f"metrics.{name}.evidence_policy"),
                    )

    _validate_top_level(resolved)
    if not isinstance(resolved.get("evidence_policy_defaults"), dict):
        raise SceneQualityInterfaceConfigError(
            "l3_scene_quality.evidence_policy_defaults must be a JSON object"
        )
    metrics = resolved.get("metrics")
    if not isinstance(metrics, dict):
        raise SceneQualityInterfaceConfigError(
            "l3_scene_quality.metrics must be a JSON object"
        )
    for metric_name in SUPPORTED_SCENE_QUALITY_METRICS:
        metric_config = metrics.get(metric_name)
        if not isinstance(metric_config, dict):
            raise SceneQualityInterfaceConfigError(
                f"l3_scene_quality.metrics.{metric_name} must be a JSON object"
            )
        _validate_metric_flags(metric_name, metric_config)
        if metric_name in EXPERIMENTAL_SCENE_QUALITY_METRICS:
            metric_config.update(
                {
                    "metric_status": EXPERIMENTAL_NON_SCORING,
                    "activation_policy": "explicit_config_only",
                    "included_in_canonical_aggregate": False,
                }
            )
        built_in = DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG["metrics"][metric_name]["evidence_policy"]
        policy = _deep_merge(deepcopy(built_in), global_defaults)
        policy = _deep_merge(policy, explicit_metric_policies[metric_name])
        metric_config["evidence_policy"] = _validate_evidence_policy(metric_name, policy)
        plan = metric_config.get("evidence_plan")
        if plan is not None:
            try:
                validate_evidence_plan(
                    plan, where=f"l3_scene_quality.metrics.{metric_name}.evidence_plan"
                )
            except Exception as exc:  # normalize to this module's error type
                raise SceneQualityInterfaceConfigError(str(exc)) from exc
    return resolved


def evaluate_scene_quality_interfaces(
    scene: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    object_grouping_report: dict[str, Any] | list[dict[str, Any]] | None = None,
    render_evidence: list[str] | dict[str, Any] | None = None,
    camera_evidence_provider: Any = None,
    vlm_judge: Any = None,
    authorized_deviations: Any = None,
    metric_applicability: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | None = None,
    prompt: str | None = None,
    visual_style_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the three canonical L3 metrics from prepared visual evidence.

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

    metric_reports: dict[str, dict[str, Any]] = {}
    for metric_name in SUPPORTED_SCENE_QUALITY_METRICS:
        metric_config = resolved["metrics"][metric_name]
        metric_reports[metric_name] = _evaluate_metric(
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
        )

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
        "experimental_metrics": {
            metric_name: {
                "status": EXPERIMENTAL_NON_SCORING,
                "enabled": bool(
                    resolved["metrics"][
                        metric_name
                    ]["enabled"]
                ),
                "activation_policy": "explicit_config_only",
                "included_in_canonical_aggregate": False,
            }
            for metric_name in EXPERIMENTAL_SCENE_QUALITY_METRICS
        },
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
                "Optional non-scoring L3 semantic placement judges whether an "
                "otherwise physically possible location makes sense in the "
                "scene; L1 remains the owner of collision, OOB, and support"
            ),
        },
        "double_count_guard": {
            "affects_aggregate_score": bool(active),
            "reason": (
                "Scale, grouped Object Pairing, and Style are owned and scored only "
                "by canonical L3 Scene Quality."
            ),
        },
        "notes": [
            "Semantic Coherence (scale/pairing) and Perceptual Visual Quality (style) are distinct subfamilies.",
            "Object pairing runs after external grouping and judges target category/role compatibility with both scene and local-group context.",
            "Semantic placement is an opt-in non-scoring diagnostic for scene- and local-context location plausibility; it excludes collision and physical support.",
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
    vlm_judge: Any,
    prompt: str | None,
    visual_style_spec: dict[str, Any] | None,
    authorized_deviations: list[dict[str, Any]],
    applicability: Any,
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
                if metric_name == "object_pairing_consistency"
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
        "metric_status": (
            EXPERIMENTAL_NON_SCORING
            if metric_name in EXPERIMENTAL_SCENE_QUALITY_METRICS
            else "canonical_scoring"
        ),
        "activation_policy": (
            "explicit_config_only"
            if metric_name in EXPERIMENTAL_SCENE_QUALITY_METRICS
            else "profile_and_applicability"
        ),
        "included_in_canonical_aggregate": (
            metric_name in SCENE_QUALITY_INTERFACE_METRICS
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
    base["affects_score"] = (
        metric_name in SCENE_QUALITY_INTERFACE_METRICS
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

    if (
        metric_name == "style_consistency"
        and isinstance(metric_config.get("evidence_plan"), dict)
        and metric_config["evidence_plan"].get("evidence_strategy")
        == "global_screen_then_local"
    ):
        return _evaluate_style_global_then_group_local(
            base=base,
            metric_config=metric_config,
            scene=scene,
            object_ids=object_ids,
            groups=groups,
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
) -> dict[str, Any]:
    objects = [
        _compact_object(item)
        for item in scene.get("objects", [])
        if isinstance(item, dict)
        and (
            not selected_object_ids
            or str(item.get("id")) in set(selected_object_ids)
        )
    ]
    selected_groups = [
        deepcopy(group)
        for group in groups or []
        if not selected_group_ids or str(group.get("group_id")) in set(selected_group_ids)
    ]
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
            "defects": {
                "required_when_invalid": True,
                "fields": ["scope", "target_ids", "relation", "reason"],
                "allowed_scopes": list(
                    JUDGMENT_SCOPE_BY_METRIC[metric_name]["included"]
                ),
            },
        },
        **vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.CANONICAL_METRIC,
            judge_method="adjudicate_scene_quality",
        ),
    }
    if group_scope is not None:
        scope_value = group_scope.to_dict()
        request["scene_summary"]["group_scope"] = deepcopy(
            scope_value
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
        return {
            "status": "evaluated",
            "score": 1.0 if verdict == "valid" else 0.0,
            "reason": None,
        }
    raise ValueError(
        "scene-quality VLM verdict must be valid, invalid, or ambiguous"
    )


def _validate_top_level(config: dict[str, Any]) -> None:
    for flag in ("enabled", "implemented"):
        if not isinstance(config.get(flag), bool):
            raise SceneQualityInterfaceConfigError(
                f"l3_scene_quality.{flag} must be boolean"
            )
    if config.get("implemented") is not True:
        raise SceneQualityInterfaceConfigError(
            "canonical l3_scene_quality implemented must remain true"
        )
    version = config.get("version")
    if not isinstance(version, str) or not version.strip():
        raise SceneQualityInterfaceConfigError(
            "l3_scene_quality.version must be a non-empty string"
        )


def _validate_metric_flags(metric_name: str, metric_config: dict[str, Any]) -> None:
    for flag in ("enabled", "implemented"):
        if not isinstance(metric_config.get(flag), bool):
            raise SceneQualityInterfaceConfigError(
                f"l3_scene_quality.metrics.{metric_name}.{flag} must be boolean"
            )
    if metric_config.get("implemented") is not True:
        raise SceneQualityInterfaceConfigError(
            f"canonical l3_scene_quality metric {metric_name} implemented must remain true"
        )
    weight = metric_config.get("weight", 1.0)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) < 0.0
    ):
        raise SceneQualityInterfaceConfigError(
            f"l3_scene_quality.metrics.{metric_name}.weight must be non-negative"
        )


def _validate_evidence_policy(metric_name: str, policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise SceneQualityInterfaceConfigError(
            f"l3_scene_quality.metrics.{metric_name}.evidence_policy must be a JSON object"
        )
    scope = policy.get("camera_scope")
    if scope not in CAMERA_SCOPES:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.camera_scope must be one of {list(CAMERA_SCOPES)}, got {scope!r}"
        )
    mode = policy.get("camera_mode")
    if mode not in CAMERA_MODES:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.camera_mode must be one of {list(CAMERA_MODES)}, got {mode!r}"
        )
    selector = policy.get("selector")
    if selector not in EVIDENCE_SELECTORS:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.selector must be one of {list(EVIDENCE_SELECTORS)}, got {selector!r}"
        )
    presentation = policy.get("presentation")
    if presentation not in PRESENTATIONS:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.presentation must be one of {list(PRESENTATIONS)}, got {presentation!r}"
        )
    budget = policy.get("image_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.image_budget must be a positive integer, got {budget!r}"
        )
    for quota_name in (
        "global_image_budget",
        "scoped_image_budget",
    ):
        quota = policy.get(quota_name)
        if quota is None:
            continue
        if (
            isinstance(quota, bool)
            or not isinstance(quota, int)
            or quota < 0
        ):
            raise SceneQualityInterfaceConfigError(
                f"{metric_name}.evidence_policy.{quota_name} "
                "must be a non-negative integer"
            )
        if quota > budget:
            raise SceneQualityInterfaceConfigError(
                f"{metric_name}.evidence_policy.{quota_name} "
                "cannot exceed image_budget"
            )
    if not isinstance(policy.get("include_global_context"), bool):
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.include_global_context must be boolean"
        )
    image_order = policy.get("image_order")
    if image_order is not None:
        if not isinstance(image_order, list) or not image_order:
            raise SceneQualityInterfaceConfigError(
                f"{metric_name}.evidence_policy.image_order must be null or a non-empty list"
            )
        for token in image_order:
            if token not in IMAGE_ORDER_TOKENS:
                raise SceneQualityInterfaceConfigError(
                    f"{metric_name}.evidence_policy.image_order token {token!r} must be one of "
                    f"{list(IMAGE_ORDER_TOKENS)}"
                )
    camera_pose_mode = policy.get("camera_pose_mode")
    if camera_pose_mode is not None and camera_pose_mode not in CAMERA_POSE_MODES:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.camera_pose_mode must be null or one of "
            f"{list(CAMERA_POSE_MODES)}, got {camera_pose_mode!r}"
        )
    return policy


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


def _normalize_layer(layer: Any, label: str) -> dict[str, Any]:
    """Copy one canonical config layer and reject retired surface names."""

    obj = _as_object(layer, label)
    retired_namespaces = sorted(set(obj) & set(_RETIRED_CONFIG_NAMESPACES))
    if retired_namespaces:
        raise SceneQualityInterfaceConfigError(
            "retired L3 config namespaces are not accepted: "
            f"{retired_namespaces}; pass the canonical config directly"
        )
    result = deepcopy(obj)
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        retired_metrics = sorted(set(metrics) & set(_RETIRED_METRIC_NAMES))
        if retired_metrics:
            raise SceneQualityInterfaceConfigError(
                "retired L3 metric names are not accepted: "
                f"{retired_metrics}; use 'object_pairing_consistency'"
            )
    return result


def _plan_uses_grouping(plan: dict[str, Any] | None) -> bool:
    return bool(isinstance(plan, dict) and isinstance(plan.get("local_policy"), dict))


def _as_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SceneQualityInterfaceConfigError(
            f"l3_scene_quality {label} must be a JSON object"
        )
    return value


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise SceneQualityInterfaceConfigError(
            "l3_scene_quality config patch must be a JSON object"
        )
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
