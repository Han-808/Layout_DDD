"""Versioned benchmark scoring profiles shared across pipeline boundaries."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


SCORING_SPEC_VERSION = "object_equivalent_burden_v3"
LEGACY_SCORING_SPEC_VERSION = "legacy_metric_scoring_compat"
PREVIOUS_INTRINSIC_VALIDITY_PROFILE_ID = "intrinsic_validity_v1"
PREVIOUS_PROMPT_CONDITIONED_QUALITY_PROFILE_ID = (
    "prompt_conditioned_quality_v1"
)
INTRINSIC_VALIDITY_PROFILE_ID = "intrinsic_validity_v2"
PROMPT_CONDITIONED_QUALITY_PROFILE_ID = "prompt_conditioned_quality_v2"
DEFAULT_DEDUCTION_MULTIPLIER = 2.0
DEDUCTION_MULTIPLIER_METRICS = (
    "collision",
    "support",
    "oob",
    "scale_consistency",
    "style_consistency",
    "object_pairing_consistency",
)

PREVIOUS_L3_METRIC_WEIGHTS = {
    "scale_consistency": 0.12,
    "style_consistency": 0.12,
    "object_pairing_consistency": 0.12,
    "functional_consistency": 0.44,
    "semantic_placement_consistency": 0.20,
}

# Function and Placement are the benchmark's primary semantic contribution.
# Scale remains scored at low weight for the catalog-observed track rather
# than being deleted: it is still useful for auditing asset selection and for
# replaying the same evaluator on native/open-asset generators.
DEFAULT_L3_METRIC_WEIGHTS = {
    "scale_consistency": 0.04,
    "style_consistency": 0.07,
    "object_pairing_consistency": 0.09,
    "functional_consistency": 0.52,
    "semantic_placement_consistency": 0.28,
}

SCORING_PROFILES: dict[str, dict[str, Any]] = {
    PREVIOUS_INTRINSIC_VALIDITY_PROFILE_ID: {
        "layer_weights": {
            "l1_physical_plausibility": 0.30,
            "l2_specification_fidelity": 0.0,
            "l3_scene_quality": 0.70,
            "l4_downstream_task_functionality": 0.0,
        },
        "l3_metric_weights": PREVIOUS_L3_METRIC_WEIGHTS,
        "requires_l2_task": False,
        "deduction_multiplier": DEFAULT_DEDUCTION_MULTIPLIER,
        "deduction_multiplier_metrics": list(
            DEDUCTION_MULTIPLIER_METRICS
        ),
    },
    PREVIOUS_PROMPT_CONDITIONED_QUALITY_PROFILE_ID: {
        "layer_weights": {
            "l1_physical_plausibility": 0.20,
            "l2_specification_fidelity": 0.20,
            "l3_scene_quality": 0.60,
            "l4_downstream_task_functionality": 0.0,
        },
        "l3_metric_weights": PREVIOUS_L3_METRIC_WEIGHTS,
        "requires_l2_task": True,
        "deduction_multiplier": DEFAULT_DEDUCTION_MULTIPLIER,
        "deduction_multiplier_metrics": list(
            DEDUCTION_MULTIPLIER_METRICS
        ),
    },
    INTRINSIC_VALIDITY_PROFILE_ID: {
        "layer_weights": {
            "l1_physical_plausibility": 0.30,
            "l2_specification_fidelity": 0.0,
            "l3_scene_quality": 0.70,
            "l4_downstream_task_functionality": 0.0,
        },
        "l3_metric_weights": DEFAULT_L3_METRIC_WEIGHTS,
        "requires_l2_task": False,
        "deduction_multiplier": DEFAULT_DEDUCTION_MULTIPLIER,
        "deduction_multiplier_metrics": list(
            DEDUCTION_MULTIPLIER_METRICS
        ),
    },
    PROMPT_CONDITIONED_QUALITY_PROFILE_ID: {
        "layer_weights": {
            "l1_physical_plausibility": 0.20,
            "l2_specification_fidelity": 0.20,
            "l3_scene_quality": 0.60,
            "l4_downstream_task_functionality": 0.0,
        },
        "l3_metric_weights": DEFAULT_L3_METRIC_WEIGHTS,
        "requires_l2_task": True,
        "deduction_multiplier": DEFAULT_DEDUCTION_MULTIPLIER,
        "deduction_multiplier_metrics": list(
            DEDUCTION_MULTIPLIER_METRICS
        ),
    },
}


def resolve_scoring_profile(profile_id: str) -> dict[str, Any]:
    """Return a validated defensive copy of a leaderboard profile."""

    try:
        profile = deepcopy(SCORING_PROFILES[str(profile_id)])
    except KeyError as exc:
        raise ValueError(f"unknown scoring profile {profile_id!r}") from exc
    weights = profile["layer_weights"]
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0):
        raise RuntimeError(f"scoring profile {profile_id!r} is not normalized")
    metric_weights = profile.get("l3_metric_weights")
    if not isinstance(metric_weights, dict) or set(metric_weights) != set(
        DEFAULT_L3_METRIC_WEIGHTS
    ):
        raise RuntimeError(
            f"scoring profile {profile_id!r} has an invalid L3 metric inventory"
        )
    if not math.isclose(
        sum(float(value) for value in metric_weights.values()),
        1.0,
    ):
        raise RuntimeError(
            f"scoring profile {profile_id!r} L3 weights are not normalized"
        )
    profile.update(
        {
            "scoring_profile_id": str(profile_id),
            "scoring_spec_version": SCORING_SPEC_VERSION,
        }
    )
    return profile


def scoring_profile_for_run(*, has_l2_task: bool) -> dict[str, Any]:
    return resolve_scoring_profile(
        PROMPT_CONDITIONED_QUALITY_PROFILE_ID
        if has_l2_task
        else INTRINSIC_VALIDITY_PROFILE_ID
    )


__all__ = [
    "DEFAULT_DEDUCTION_MULTIPLIER",
    "DEFAULT_L3_METRIC_WEIGHTS",
    "DEDUCTION_MULTIPLIER_METRICS",
    "INTRINSIC_VALIDITY_PROFILE_ID",
    "LEGACY_SCORING_SPEC_VERSION",
    "PREVIOUS_INTRINSIC_VALIDITY_PROFILE_ID",
    "PREVIOUS_L3_METRIC_WEIGHTS",
    "PREVIOUS_PROMPT_CONDITIONED_QUALITY_PROFILE_ID",
    "PROMPT_CONDITIONED_QUALITY_PROFILE_ID",
    "SCORING_PROFILES",
    "SCORING_SPEC_VERSION",
    "resolve_scoring_profile",
    "scoring_profile_for_run",
]
