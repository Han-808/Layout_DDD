from __future__ import annotations

from typing import Any


FUNCTIONAL_GROUPING_EVALUATOR_VERSION = "functional_grouping_placeholder_v0"
DEFAULT_FUNCTIONAL_GROUPING_CONFIG: dict[str, Any] = {
    "enabled": False,
    "implemented": False,
}


def evaluate_functional_grouping(config: dict[str, Any]) -> dict[str, Any]:
    enabled = config.get("enabled", False)
    implemented = config.get("implemented", False)
    if not isinstance(enabled, bool) or not isinstance(implemented, bool):
        raise ValueError("functional_grouping.enabled and implemented must be boolean")
    if implemented:
        raise ValueError(
            "functional_grouping is a deferred placeholder; implemented=true is unsupported"
        )
    if enabled:
        raise ValueError(
            "functional_grouping is not implemented; enabled=true cannot silently activate it"
        )
    return {
        "metric": "functional_grouping",
        "evaluator_version": FUNCTIONAL_GROUPING_EVALUATOR_VERSION,
        "implemented": False,
        "enabled": False,
        "status": "not_implemented",
        "reason": "deferred_by_metric_plan",
        "score": None,
        "partial_score": None,
        "affects_score": False,
        "coverage": {
            "eligible_count": 0,
            "resolved_count": 0,
            "unknown_count": 0,
            "vlm_pending_count": 0,
            "fraction": None,
            "complete": False,
        },
        "routing": {
            "direct_valid": 0,
            "requires_vlm": 0,
            "vlm_adjudicated": 0,
            "vlm_adjudication_failed": 0,
            "unknown": 0,
        },
        "checks": [],
        "notes": [
            "Future Functional Grouping may evaluate semantic groups and relative placement/proximity.",
            "Operability, navigation, accessibility, human factors, and task completion remain outside this placeholder.",
        ],
    }

