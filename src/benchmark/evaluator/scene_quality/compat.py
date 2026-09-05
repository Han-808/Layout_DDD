"""Narrow compatibility adapters for historical Scene Quality profiles.

The active benchmark profile owns five L3 metrics.  The frozen v1 profile owned
only three; reading that profile must not silently activate metrics that were
absent from it.  Keeping this translation outside the evaluator makes the
historical exception explicit and prevents it from shaping the current
interface.
"""

from __future__ import annotations

from typing import Any


PREVIOUS_SCENE_EVALUATION_PROFILE_VERSION = "canonical_scene_evaluation_v1"


def historical_scene_quality_profile_override(
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the exact v1 exclusions, or an empty override for current input."""

    if (
        not isinstance(profile, dict)
        or profile.get("profile_version")
        != PREVIOUS_SCENE_EVALUATION_PROFILE_VERSION
    ):
        return {}
    excluded = {
        "enabled": False,
        "metric_status": "historical_profile_excluded",
        "activation_policy": "profile_excluded",
        "included_in_canonical_aggregate": False,
    }
    return {
        "metrics": {
            "functional_consistency": dict(excluded),
            "semantic_placement_consistency": dict(excluded),
        }
    }
