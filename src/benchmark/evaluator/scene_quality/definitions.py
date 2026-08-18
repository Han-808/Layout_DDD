"""Stable L3 Scene Quality metric definitions and built-in defaults."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.assets.facing import (
    CATALOG_FACING_CONTRACT_VERSION,
    DEFAULT_DIRECTED_FUNCTIONAL_SIDE,
)
from benchmark.evaluator.evidence_contract import GROUPING_POLICY_ID
from benchmark.evaluator.scene_quality.functional_prejudgement import (
    DEFAULT_FUNCTIONAL_PREJUDGEMENT_EVIDENCE_CONFIG,
)
from benchmark.evaluator.scene_quality.placement_severity import (
    PLACEMENT_SEVERITY_LEVELS,
)
from benchmark.scoring_profiles import DEFAULT_L3_METRIC_WEIGHTS
from benchmark.visual_judge.l3_prompts import (
    L3_METRIC_RUBRICS,
)
from benchmark.visual_judge.functional_evidence import (
    FUNCTIONAL_PROBE_DEFAULT_UNITS,
)
from benchmark.visual_judge.usable_surface import (
    CATALOG_CONTRACT_USABLE_SURFACE_DETECTOR_BACKEND,
)

SCENE_QUALITY_INTERFACE_VERSION = "scene_quality_v7"

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
SCENE_QUALITY_INTERFACE_METRICS = (
    "style_consistency",
    "scale_consistency",
    "object_pairing_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)
SUPPORTED_SCENE_QUALITY_METRICS = SCENE_QUALITY_INTERFACE_METRICS
# Kept as an empty compatibility export for callers that imported the symbol.
EXPERIMENTAL_SCENE_QUALITY_METRICS: tuple[str, ...] = ()
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
            "scene_real_world_usability",
            "group_real_world_usability",
            "facing_and_interaction_direction",
            "interaction_side_accessibility",
            "opening_clearance",
            "reachability",
            "circulation",
            "action_required_operational_connection",
            "orientation_for_use",
            "ensemble_operability",
        ],
        "excluded": [
            "prompt_fidelity",
            "category_pairing",
            "style",
            "scale",
            "context_only_adjacency",
            "exact_relation_fidelity",
        ],
        "prerequisite": "object_grouping_report",
    },
    "semantic_placement_consistency": {
        "included": [
            "semantically_inappropriate_support_surface",
            "implausible_placement_height",
            "semantically_inappropriate_scene_zone",
            "implausible_cross_group_context",
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
            "action_required_adjacency",
            "functional_correspondence",
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
    "functional_prejudgement_evidence": deepcopy(
        DEFAULT_FUNCTIONAL_PREJUDGEMENT_EVIDENCE_CONFIG
    ),
    "metrics": {
        "style_consistency": {
            "enabled": True,
            "implemented": True,
            "metric_status": "canonical_scoring",
            "activation_policy": "profile_and_applicability",
            "included_in_canonical_aggregate": True,
            "weight": DEFAULT_L3_METRIC_WEIGHTS["style_consistency"],
            "evidence_policy": {
                "camera_scope": "global",
                "camera_mode": "global_oblique",
                "selector": "deterministic",
                "image_budget": 1,
                "presentation": "raw",
                "image_order": None,
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": (
                    "global_screen_then_local"
                ),
                "global_policy": {
                    "view_family": "canonical_overview_perspective",
                    "image_budget": 1,
                    "top_down": False,
                    "perspective_diversity_required": False,
                },
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": GROUPING_POLICY_ID,
                    "image_budget": 1,
                    "global_context_image_budget": 1,
                    "max_packet_images": 2,
                    "image_order": [
                        "global_context",
                        "group_local",
                    ],
                    "minimum_group_members": 2,
                    "force_for_eligible_groups": False,
                },
                "router_options": None,
                "text_context": [
                    "metric_prompt_context.room_type",
                    "visual_style_spec",
                    "authorized_deviations",
                    "asset_policy",
                ],
            },
        },
        "scale_consistency": {
            "enabled": True,
            "implemented": True,
            "metric_status": "canonical_scoring",
            "activation_policy": "profile_and_applicability",
            "included_in_canonical_aggregate": True,
            "weight": DEFAULT_L3_METRIC_WEIGHTS["scale_consistency"],
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
                    "metric_prompt_context.none",
                    "authorized_deviations",
                    "asset_policy",
                    "object_grouping_report",
                ],
            },
        },
        "object_pairing_consistency": {
            "enabled": True,
            "implemented": True,
            "metric_status": "canonical_scoring",
            "activation_policy": "profile_and_applicability",
            "included_in_canonical_aggregate": True,
            "weight": DEFAULT_L3_METRIC_WEIGHTS[
                "object_pairing_consistency"
            ],
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
                    "metric_prompt_context.room_type",
                    "authorized_deviations",
                    "asset_policy",
                    "object_grouping_report",
                ],
            },
        },
        "functional_consistency": {
            "enabled": True,
            "implemented": True,
            "metric_status": "canonical_scoring",
            "activation_policy": "profile_and_applicability",
            "included_in_canonical_aggregate": True,
            "weight": DEFAULT_L3_METRIC_WEIGHTS[
                "functional_consistency"
            ],
            "group_local_check_granularity": "per_check",
            "group_local_evidence_policy": "shared_group_bank",
            "group_local_active_window_max_images": 6,
            "evidence_policy": {
                "camera_scope": "global",
                "camera_mode": "global_oblique",
                "selector": "deterministic",
                "image_budget": 1,
                "presentation": "raw",
                "image_order": None,
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": (
                    "global_discovery_then_group_local"
                ),
                "global_policy": {
                    "view_family": "canonical_overview_perspective",
                    "image_budget": 1,
                    "top_down": False,
                    "perspective_diversity_required": False,
                },
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": GROUPING_POLICY_ID,
                    "image_budget": 1,
                    "global_context_image_budget": 1,
                    "max_packet_images": 2,
                    "image_order": [
                        "global_context",
                        "group_local",
                    ],
                    "minimum_group_members": 2,
                    "force_for_eligible_groups": True,
                },
                "prejudgement_probe_policy": {
                    "enabled": True,
                    "discovery": {
                        "backend": "vlm",
                        "decision_authority": "none",
                        "complete_object_coverage_required": True,
                        "group_normalization": "deterministic",
                        "unusual_confirmation_scope": "group_local",
                    },
                    "usable_surface": {
                        "backend": (
                            CATALOG_CONTRACT_USABLE_SURFACE_DETECTOR_BACKEND
                        ),
                        "catalog_default_directed_side": (
                            DEFAULT_DIRECTED_FUNCTIONAL_SIDE
                        ),
                        "catalog_contract_version": (
                            CATALOG_FACING_CONTRACT_VERSION
                        ),
                        "trusted_side_ids": [
                            "local_pos_x",
                            "local_neg_x",
                            "local_pos_y",
                            "local_neg_y",
                        ],
                        "decode_scope": (
                            "directed_or_uncertain_clearance_targets_"
                            "before_probe_budget"
                        ),
                        "fallback": (
                            "vlm_trusted_side_ids_then_existing_geometry_"
                            "local_camera"
                        ),
                        "scene_access": "read_only",
                    },
                    "planner_input": (
                        "one_global_image_plus_id_category_groups_boundary"
                    ),
                    "max_probe_units": FUNCTIONAL_PROBE_DEFAULT_UNITS,
                    "candidate_count_by_probe_kind": {
                        "functional_frontage": 4,
                        "functional_correspondence": 4,
                        "approach_clearance": 4,
                    },
                    "selected_views_per_unit": 1,
                    "preferred_lens_mm": 32.0,
                    "elevation_range_degrees": [8.0, 16.0],
                    "context_margin_m": 1.25,
                    "judge_presentation": "raw_rgb_only",
                    "decision_authority": "none",
                },
                "router_options": None,
                "text_context": [
                    "metric_prompt_context.none",
                    "authorized_deviations",
                    "asset_policy",
                    "object_grouping_report",
                ],
            },
        },
        # Semantic location only; L1 remains the sole owner of collision and
        # physical support.
        "semantic_placement_consistency": {
            "enabled": True,
            "implemented": True,
            "metric_status": "canonical_scoring",
            "activation_policy": "profile_and_applicability",
            "included_in_canonical_aggregate": True,
            "weight": DEFAULT_L3_METRIC_WEIGHTS[
                "semantic_placement_consistency"
            ],
            "severity_policy": {
                "schema_version": "object_equivalent_burden_v1",
                "levels": list(PLACEMENT_SEVERITY_LEVELS),
                "strict_level": "implausible",
                "extended_level": "atypical",
                "affects_existing_metric_score": True,
            },
            # Experimental post-typed holistic review.  It stays disabled in
            # the generic library default so existing API callers retain
            # their call graph; the camera-cal runner and canonical v2
            # profile explicitly enable it for the current experiment.
            "residual_global_review": {
                "enabled": False,
                "placement_weight": 0.20,
                "image_budget": 3,
                "allowed_check_types": [
                    "scene_zone",
                    "contextual_anchor",
                ],
            },
            "evidence_policy": {
                "camera_scope": "global",
                "camera_mode": "global_oblique",
                "selector": "deterministic",
                "image_budget": 1,
                "presentation": "raw",
                "image_order": None,
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": (
                    "global_discovery_then_group_local"
                ),
                "global_policy": {
                    "view_family": "canonical_overview_perspective",
                    "image_budget": 1,
                    "top_down": False,
                    "perspective_diversity_required": False,
                },
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": GROUPING_POLICY_ID,
                    "image_budget": 1,
                    "global_context_image_budget": 1,
                    "max_packet_images": 2,
                    "image_order": [
                        "global_context",
                        "group_local",
                    ],
                    "minimum_group_members": 2,
                    "force_for_eligible_groups": True,
                },
                "router_options": None,
                "text_context": [
                    "metric_prompt_context.none",
                    "authorized_deviations",
                    "asset_policy",
                    "object_grouping_report",
                ],
            },
        },
    },
}
