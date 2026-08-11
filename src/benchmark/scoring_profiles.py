"""Versioned benchmark scoring profiles shared across pipeline boundaries."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


SCORING_SPEC_VERSION = "object_equivalent_burden_v1"
LEGACY_SCORING_SPEC_VERSION = "legacy_metric_scoring_compat"
INTRINSIC_VALIDITY_PROFILE_ID = "intrinsic_validity_v1"
PROMPT_CONDITIONED_QUALITY_PROFILE_ID = "prompt_conditioned_quality_v1"

SCORING_PROFILES: dict[str, dict[str, Any]] = {
    INTRINSIC_VALIDITY_PROFILE_ID: {
        "layer_weights": {
            "l1_physical_plausibility": 0.30,
            "l2_specification_fidelity": 0.0,
            "l3_scene_quality": 0.70,
            "l4_downstream_task_functionality": 0.0,
        },
        "requires_l2_task": False,
    },
    PROMPT_CONDITIONED_QUALITY_PROFILE_ID: {
        "layer_weights": {
            "l1_physical_plausibility": 0.20,
            "l2_specification_fidelity": 0.20,
            "l3_scene_quality": 0.60,
            "l4_downstream_task_functionality": 0.0,
        },
        "requires_l2_task": True,
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
    "INTRINSIC_VALIDITY_PROFILE_ID",
    "LEGACY_SCORING_SPEC_VERSION",
    "PROMPT_CONDITIONED_QUALITY_PROFILE_ID",
    "SCORING_PROFILES",
    "SCORING_SPEC_VERSION",
    "resolve_scoring_profile",
    "scoring_profile_for_run",
]
