"""Versioned post-hoc scoring for the scene-generation leaderboard.

The canonical evaluator owns evidence collection and persists metric ledgers.
The public leaderboard applies a separate, provisional weighting profile to
those frozen ledgers.  Keeping that projection separate prevents a leaderboard
weight adjustment from changing Judge behavior or invalidating evaluation
artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from benchmark.resources import runtime_resource_path


PROFILE_SCHEMA_VERSION = "scene_generation_leaderboard_scoring_profile_v1"
PROFILE_RESOURCE = (
    "configs/evaluation/scene_generation_leaderboard_scoring_v1.json"
)
METRICS = (
    "collision",
    "oob",
    "support",
    "style_consistency",
    "scale_consistency",
    "object_pairing_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)
CATEGORIES = (
    "physical_plausibility",
    "functional_semantics",
    "visual_coherence",
)

# Historical evaluator reports used 2x for L1 and visual-coherence metrics;
# functional/placement ledgers persisted their unscaled 1x scores.  These are
# only used when an older report lacks ``base_metric_deduction``.
RECORDED_DEDUCTION_MULTIPLIERS = {
    "collision": 2.0,
    "oob": 2.0,
    "support": 2.0,
    "style_consistency": 2.0,
    "scale_consistency": 2.0,
    "object_pairing_consistency": 2.0,
    "functional_consistency": 1.0,
    "semantic_placement_consistency": 1.0,
}


def load_leaderboard_scoring_profile(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the package-owned leaderboard profile."""

    source = (
        runtime_resource_path(PROFILE_RESOURCE)
        if path is None
        else Path(path).expanduser().resolve()
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("leaderboard scoring profile must be a JSON object")
    return validate_leaderboard_scoring_profile(value)


def validate_leaderboard_scoring_profile(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a normalized profile or fail closed on weight drift."""

    profile = dict(value)
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("leaderboard scoring profile schema mismatch")
    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("leaderboard scoring profile_id is required")

    category_weights = _positive_number_map(
        profile.get("category_weights"),
        expected=frozenset(CATEGORIES),
        label="category_weights",
    )
    metric_weights = _positive_number_map(
        profile.get("metric_weights"),
        expected=frozenset(METRICS),
        label="metric_weights",
    )
    deduction_multipliers = _positive_number_map(
        profile.get("deduction_multipliers"),
        expected=frozenset(METRICS),
        label="deduction_multipliers",
    )
    if not math.isclose(
        sum(category_weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("leaderboard category weights must sum to 1.0")
    if not math.isclose(
        sum(metric_weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("leaderboard metric weights must sum to 1.0")

    raw_groups = profile.get("metric_groups")
    if not isinstance(raw_groups, Mapping) or set(raw_groups) != set(CATEGORIES):
        raise ValueError("leaderboard metric group inventory mismatch")
    metric_groups: dict[str, list[str]] = {}
    grouped: set[str] = set()
    for category in CATEGORIES:
        raw_metrics = raw_groups[category]
        if not isinstance(raw_metrics, list) or not raw_metrics:
            raise ValueError(f"leaderboard metric group is invalid: {category}")
        metrics = [str(metric) for metric in raw_metrics]
        if len(metrics) != len(set(metrics)) or any(
            metric not in METRICS for metric in metrics
        ):
            raise ValueError(f"leaderboard metric group is invalid: {category}")
        group_weight = sum(metric_weights[metric] for metric in metrics)
        if not math.isclose(
            group_weight,
            category_weights[category],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                f"leaderboard metric group weight mismatch: {category}"
            )
        grouped.update(metrics)
        metric_groups[category] = metrics
    if grouped != set(METRICS):
        raise ValueError("leaderboard metric groups do not cover every metric")

    normalized = {
        **profile,
        "category_weights": category_weights,
        "metric_weights": metric_weights,
        "metric_groups": metric_groups,
        "deduction_multipliers": deduction_multipliers,
    }
    normalized["profile_sha256"] = _json_sha256(
        {key: child for key, child in normalized.items() if key != "profile_sha256"}
    )
    return normalized


def rescore_scene_generation_case(
    *,
    l1_report: Mapping[str, Any],
    l3_report: Mapping[str, Any],
    recorded_metrics: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Apply the web profile to one frozen evaluator result.

    Returned scores use the same 0--100 scale as the webpage.  Missing metrics
    are renormalized within their category exactly as in the current web
    implementation; all three categories must remain scoreable.
    """

    resolved_profile = validate_leaderboard_scoring_profile(
        profile if profile is not None else load_leaderboard_scoring_profile()
    )
    l1_rows = l1_report.get("metrics")
    l1_rows = l1_rows if isinstance(l1_rows, Mapping) else {}
    l3_rows = l3_report.get("metrics")
    l3_rows = l3_rows if isinstance(l3_rows, Mapping) else {}
    fallback = recorded_metrics if isinstance(recorded_metrics, Mapping) else {}
    multipliers = resolved_profile["deduction_multipliers"]

    scores: dict[str, float | None] = {}
    for metric in METRICS:
        rows = l1_rows if metric in {"collision", "oob", "support"} else l3_rows
        row = rows.get(metric)
        score = (
            _rescore_metric(
                row,
                multiplier=multipliers[metric],
                recorded_multiplier=RECORDED_DEDUCTION_MULTIPLIERS[metric],
            )
            if isinstance(row, Mapping)
            else None
        )
        if score is None:
            recorded = _number(fallback.get(metric))
            if recorded is not None:
                base = (
                    1.0 - recorded / 100.0
                ) / RECORDED_DEDUCTION_MULTIPLIERS[metric]
                score = 1.0 - min(1.0, multipliers[metric] * base)
        scores[metric] = score

    weights = resolved_profile["metric_weights"]
    groups = resolved_profile["metric_groups"]
    category_scores = {
        category: _weighted_score(
            scores,
            {metric: weights[metric] for metric in groups[category]},
        )
        for category in CATEGORIES
    }
    if any(category_scores[category] is None for category in CATEGORIES):
        return None
    category_weights = resolved_profile["category_weights"]
    overall = sum(
        category_weights[category] * float(category_scores[category])
        for category in CATEGORIES
    )
    l3_score = _weighted_score(
        scores,
        {
            metric: weights[metric]
            for category in ("visual_coherence", "functional_semantics")
            for metric in groups[category]
        },
    )
    return {
        "profile_id": resolved_profile["profile_id"],
        "profile_sha256": resolved_profile["profile_sha256"],
        "overall_100": overall * 100.0,
        "physical_plausibility_100": (
            float(category_scores["physical_plausibility"]) * 100.0
        ),
        "functional_semantics_100": (
            float(category_scores["functional_semantics"]) * 100.0
        ),
        "visual_coherence_100": (
            float(category_scores["visual_coherence"]) * 100.0
        ),
        "l3_100": None if l3_score is None else l3_score * 100.0,
        "fine_metrics_100": {
            metric: None if score is None else score * 100.0
            for metric, score in scores.items()
        },
    }


def _rescore_metric(
    row: Mapping[str, Any],
    *,
    multiplier: float,
    recorded_multiplier: float,
) -> float | None:
    scoring = row.get("scoring")
    if not isinstance(scoring, Mapping):
        recorded = _number(row.get("score"))
        if recorded is None:
            return None
        base = (1.0 - recorded) / recorded_multiplier
        return 1.0 - min(1.0, multiplier * base)

    components = scoring.get("placement_components")
    component_weights = scoring.get("placement_component_weights")
    if isinstance(components, Mapping) and isinstance(component_weights, Mapping):
        weighted = 0.0
        resolved_weight = 0.0
        for name, raw_weight in component_weights.items():
            component = components.get(name)
            weight = _number(raw_weight)
            if not isinstance(component, Mapping) or weight is None:
                continue
            score = _score_from_base(component, multiplier)
            if score is None:
                continue
            weighted += weight * score
            resolved_weight += weight
        if resolved_weight > 0.0:
            return weighted / resolved_weight
    return _score_from_base(scoring, multiplier)


def _score_from_base(scoring: Mapping[str, Any], multiplier: float) -> float | None:
    base = _number(scoring.get("base_metric_deduction"))
    if base is None:
        return _number(scoring.get("score"))
    return 1.0 - min(1.0, multiplier * base)


def _weighted_score(
    scores: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> float | None:
    available = [
        (float(scores[metric]), float(weight))
        for metric, weight in weights.items()
        if _number(scores.get(metric)) is not None
    ]
    denominator = sum(weight for _, weight in available)
    return (
        sum(score * weight for score, weight in available) / denominator
        if denominator > 0.0
        else None
    )


def _positive_number_map(
    value: Any,
    *,
    expected: frozenset[str],
    label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"leaderboard {label} inventory mismatch")
    result = {str(key): _number(child) for key, child in value.items()}
    if any(child is None or child <= 0.0 for child in result.values()):
        raise ValueError(f"leaderboard {label} must contain positive numbers")
    return {key: float(child) for key, child in result.items() if child is not None}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CATEGORIES",
    "METRICS",
    "PROFILE_RESOURCE",
    "PROFILE_SCHEMA_VERSION",
    "RECORDED_DEDUCTION_MULTIPLIERS",
    "load_leaderboard_scoring_profile",
    "rescore_scene_generation_case",
    "validate_leaderboard_scoring_profile",
]
