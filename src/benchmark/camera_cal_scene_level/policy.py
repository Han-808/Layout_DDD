"""Promptless camera-cal policy and profile projections.

These functions are the canonical policy leaf used by the compatibility
runner.  They intentionally import only package-level evaluator/scoring/CLI
constants and never import the historical script or Evaluation Campaign.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.camera_cal_scene_level.cli import ANNOTATED_L3_METRICS
from benchmark.evaluator.profile import (
    DEFAULT_EVALUATION_PROFILE,
    L1,
    L2,
    L3,
    L4,
)
from benchmark.evaluator.scoring import L3_METRIC_WEIGHTS


CANONICAL_L3_METRICS = ANNOTATED_L3_METRICS
# Empty compatibility export: all annotated L3 metrics are benchmark metrics.
EXPERIMENTAL_L3_METRICS: tuple[str, ...] = ()


def promptless_scene_request(
    scene: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_id": scene.get("request_id"),
        "instruction": "",
        "scene_type": scene.get("scene_type") or case.get("scene_type"),
        "prompt_granularity": "fine_grained",
        "metadata": {
            "promptless_camera_cal": True,
            "generation_prompt_withheld_from_evaluator": True,
        },
    }


def promptless_l1_l3_profile() -> dict[str, Any]:
    profile = deepcopy(DEFAULT_EVALUATION_PROFILE)
    profile["layer_weights"] = {
        L1: 0.30,
        L2: 0.0,
        L3: 0.70,
        L4: 0.0,
    }
    profile[L2]["enabled"] = False
    for metric in profile[L2]["metrics"].values():
        metric["enabled"] = False
        metric["weight"] = 0.0
    return profile


def promptless_l3_only_profile() -> dict[str, Any]:
    """Return an audit-explicit recovery profile that executes only L3."""

    profile = promptless_l1_l3_profile()
    profile["layer_weights"] = {
        L1: 0.0,
        L2: 0.0,
        L3: 1.0,
        L4: 0.0,
    }
    profile[L1]["enabled"] = False
    for metric in profile[L1]["metrics"].values():
        metric["enabled"] = False
        metric["weight"] = 0.0
    return profile


def scene_quality_config(
    metrics: tuple[str, ...],
    *,
    functional_group_local_granularity: str = "per_check",
    functional_group_local_evidence_policy: str = "shared_group_bank",
) -> dict[str, Any]:
    if functional_group_local_granularity not in {
        "per_check",
        "batched",
    }:
        raise ValueError(
            "functional_group_local_granularity must be exactly "
            "'per_check' or 'batched'"
        )
    if functional_group_local_evidence_policy not in {
        "isolated_episode",
        "shared_group_bank",
    }:
        raise ValueError(
            "functional_group_local_evidence_policy must be exactly "
            "'isolated_episode' or 'shared_group_bank'"
        )
    if (
        functional_group_local_evidence_policy == "shared_group_bank"
        and functional_group_local_granularity != "per_check"
    ):
        raise ValueError(
            "shared_group_bank requires "
            "functional_group_local_granularity='per_check'"
        )
    selected = set(metrics)
    return {
        "enabled": True,
        "metrics": {
            metric: {
                "enabled": metric in selected,
                "weight": (
                    L3_METRIC_WEIGHTS[metric]
                    if metric in selected
                    else 0.0
                ),
                **(
                    {
                        "group_local_check_granularity": (
                            functional_group_local_granularity
                        ),
                        "group_local_evidence_policy": (
                            functional_group_local_evidence_policy
                        ),
                        "group_local_active_window_max_images": 6,
                    }
                    if metric == "functional_consistency"
                    else {}
                ),
                **(
                    {
                        "residual_global_review": {
                            "enabled": True,
                            "placement_weight": 0.20,
                            "image_budget": 3,
                            "allowed_check_types": [
                                "scene_zone",
                                "contextual_anchor",
                            ],
                        }
                    }
                    if metric == "semantic_placement_consistency"
                    else {}
                ),
            }
            for metric in ANNOTATED_L3_METRICS
        },
    }


def camera_cal_asset_policy() -> dict[str, Any]:
    return {
        "mode": "fixed_catalog_selection",
        "identity_owner": "benchmark",
        "category_selection_owner": "generator",
        "scale_owner": "generator",
        "appearance_owner": "generator",
        "arrangement_owner": "generator",
        "source": "camera_cal_experiment_protocol",
    }


__all__ = [
    "ANNOTATED_L3_METRICS",
    "CANONICAL_L3_METRICS",
    "EXPERIMENTAL_L3_METRICS",
    "L1",
    "L2",
    "L3",
    "L4",
    "camera_cal_asset_policy",
    "promptless_l1_l3_profile",
    "promptless_l3_only_profile",
    "promptless_scene_request",
    "scene_quality_config",
]
