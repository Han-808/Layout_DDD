"""Pure scoring summaries for persisted camera-calibration results.

This module contains the scoring extraction/aggregation logic used by the
local evidence viewer, without importing the viewer, HTML helpers, paths, or
I/O.  The canonical publishability threshold is owned by the evaluator
scoring contract and is imported rather than duplicated here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.evaluator.scoring import MIN_PUBLISHABLE_SCORE_COVERAGE


SCORING_METRIC_ORDER = (
    ("L1", "collision", "Collision"),
    ("L1", "support", "Support"),
    ("L1", "oob", "Out of bounds"),
    ("L3", "scale_consistency", "Scale"),
    ("L3", "style_consistency", "Style"),
    (
        "L3",
        "object_pairing_consistency",
        "Object pairing",
    ),
    ("L3", "functional_consistency", "Function"),
    (
        "L3",
        "semantic_placement_consistency",
        "Placement",
    ),
)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def case_scoring_summary(
    *,
    case_id: str,
    case_manifest: dict[str, Any],
    l1_report: dict[str, Any],
    l3_report: dict[str, Any],
    l1_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the persisted scoring contract without re-scoring a case."""

    profile = case_manifest.get("scoring_profile")
    profile = profile if isinstance(profile, dict) else {}
    layer_weights = profile.get("layer_weights")
    layer_weights = layer_weights if isinstance(layer_weights, dict) else {}
    l1_layer_weight = _numeric(
        layer_weights.get("l1_physical_plausibility")
    )
    l3_layer_weight = _numeric(layer_weights.get("l3_scene_quality"))
    l1_layer_weight = 0.0 if l1_layer_weight is None else l1_layer_weight
    l3_layer_weight = 0.0 if l3_layer_weight is None else l3_layer_weight

    denominator = case_manifest.get("canonical_object_denominator")
    denominator = denominator if isinstance(denominator, dict) else {}
    reliability = case_manifest.get("scoring_reliability")
    reliability = reliability if isinstance(reliability, dict) else {}

    l1_metrics = l1_report.get("metrics")
    l1_metrics = l1_metrics if isinstance(l1_metrics, dict) else {}
    l3_metrics = l3_report.get("metrics")
    l3_metrics = l3_metrics if isinstance(l3_metrics, dict) else {}
    l1_scoring = l1_report.get("scoring")
    l1_scoring = l1_scoring if isinstance(l1_scoring, dict) else {}
    l1_backend = l1_report.get("backend_report")
    l1_backend = l1_backend if isinstance(l1_backend, dict) else {}
    if not l1_scoring:
        l1_scoring = l1_backend.get("scoring")
        l1_scoring = l1_scoring if isinstance(l1_scoring, dict) else {}
    l3_scoring = l3_report.get("scoring")
    l3_scoring = l3_scoring if isinstance(l3_scoring, dict) else {}
    l1_metric_weights = l1_scoring.get("metric_weights")
    l1_metric_weights = (
        l1_metric_weights if isinstance(l1_metric_weights, dict) else {}
    )
    l3_metric_weights = l3_scoring.get("metric_weights")
    l3_metric_weights = (
        l3_metric_weights if isinstance(l3_metric_weights, dict) else {}
    )

    metric_records: list[dict[str, Any]] = []
    for layer, metric, label in SCORING_METRIC_ORDER:
        source = l1_metrics if layer == "L1" else l3_metrics
        metric_report = source.get(metric)
        metric_report = (
            metric_report if isinstance(metric_report, dict) else {}
        )
        scoring = metric_report.get("scoring")
        scoring = scoring if isinstance(scoring, dict) else {}
        local_weight = _numeric(
            (
                l1_metric_weights.get(metric)
                if layer == "L1"
                else l3_metric_weights.get(metric)
            )
        )
        if local_weight is None:
            local_weight = _numeric(scoring.get("nominal_metric_weight"))
        if local_weight is None and layer == "L3":
            local_weight = _numeric(metric_report.get("weight"))
        if local_weight is None and layer == "L1":
            local_weight = 1.0 / 3.0
        local_weight = 0.0 if local_weight is None else local_weight
        layer_weight = l1_layer_weight if layer == "L1" else l3_layer_weight
        persisted_score = _numeric(metric_report.get("score"))
        coverage_projection = scoring.get("coverage_projection")
        coverage_projection = (
            coverage_projection
            if isinstance(coverage_projection, dict)
            else {}
        )
        observed_score = persisted_score
        if observed_score is None:
            observed_score = _numeric(
                coverage_projection.get(
                    "raw_score_before_coverage_projection"
                )
            )
        metric_coverage = metric_report.get("coverage")
        metric_coverage = (
            metric_coverage if isinstance(metric_coverage, dict) else {}
        )
        score_grounding = metric_coverage.get("score_grounding")
        score_grounding = (
            score_grounding if isinstance(score_grounding, dict) else {}
        )
        coverage_fraction = _numeric(score_grounding.get("fraction"))
        if coverage_fraction is None:
            coverage_fraction = _numeric(metric_coverage.get("fraction"))
        coverage_complete = (
            score_grounding.get("complete")
            if isinstance(score_grounding.get("complete"), bool)
            else metric_coverage.get("complete")
            if isinstance(metric_coverage.get("complete"), bool)
            else None
        )
        if coverage_fraction is None:
            coverage_fraction = 1.0 if observed_score is not None else 0.0
        coverage_fraction = min(1.0, max(0.0, coverage_fraction))
        coverage_threshold_passed = (
            observed_score is not None
            and coverage_fraction >= MIN_PUBLISHABLE_SCORE_COVERAGE
        )
        score = observed_score if coverage_threshold_passed else None
        score_status = (
            "complete"
            if score is not None and coverage_fraction >= 1.0 - 1.0e-12
            else "partial_coverage"
            if score is not None
            else "failed_coverage_threshold"
            if observed_score is not None
            else "insufficient_metric_coverage"
        )
        events = scoring.get("events")
        events = (
            [deepcopy(event) for event in events if isinstance(event, dict)]
            if isinstance(events, list)
            else []
        )
        judgement = metric_report.get("judgement")
        judgement = judgement if isinstance(judgement, dict) else {}
        placement_component_weights = scoring.get(
            "placement_component_weights"
        )
        placement_component_weights = (
            deepcopy(placement_component_weights)
            if isinstance(placement_component_weights, dict)
            else None
        )
        placement_components = scoring.get("placement_components")
        placement_components = (
            {
                str(name): {
                    "score": _numeric(component.get("score")),
                    "deduction": _numeric(
                        component.get("metric_deduction")
                    ),
                    "event_count": int(
                        component.get("event_count") or 0
                    ),
                }
                for name, component in placement_components.items()
                if isinstance(component, dict)
            }
            if isinstance(placement_components, dict)
            else None
        )
        metric_records.append(
            {
                "layer": layer,
                "metric": metric,
                "label": label,
                "status": str(metric_report.get("status") or "not_recorded"),
                "verdict": judgement.get("verdict"),
                "reason": str(
                    metric_report.get("reason")
                    or judgement.get("reason")
                    or ""
                ),
                "score": score,
                "observed_score": observed_score,
                "score_status": score_status,
                "coverage_fraction": coverage_fraction,
                "coverage_complete": coverage_complete,
                "coverage_threshold_passed": coverage_threshold_passed,
                "coverage": deepcopy(metric_coverage),
                "local_weight": local_weight,
                "overall_weight": layer_weight * local_weight,
                "grounded_overall_weight": (
                    layer_weight * local_weight * coverage_fraction
                    if observed_score is not None
                    else 0.0
                ),
                "weighted_points": (
                    observed_score
                    * layer_weight
                    * local_weight
                    * coverage_fraction
                    * 100.0
                    if observed_score is not None
                    else None
                ),
                "coefficient": _numeric(scoring.get("coefficient_n_m")),
                "burden": _numeric(scoring.get("burden_total_b_m")),
                "p_max": _numeric(scoring.get("p_max")),
                "deduction": _numeric(scoring.get("metric_deduction")),
                "effective_factor": _numeric(
                    scoring.get("effective_local_factor_w_m_n_m")
                ),
                "ledger_available": bool(scoring),
                "event_count": int(scoring.get("event_count") or 0),
                "events": events,
                "placement_component_weights": placement_component_weights,
                "placement_components": placement_components,
            }
        )

    scoreable_records = [
        item
        for item in metric_records
        if _numeric(item.get("overall_weight")) not in (None, 0.0)
    ]
    l1_coverage = l1_report.get("coverage")
    l1_coverage = l1_coverage if isinstance(l1_coverage, dict) else {}
    l3_coverage = l3_report.get("coverage")
    l3_coverage = l3_coverage if isinstance(l3_coverage, dict) else {}
    l1_records = [item for item in scoreable_records if item["layer"] == "L1"]
    l3_records = [item for item in scoreable_records if item["layer"] == "L3"]

    def covered_layer(
        records: list[dict[str, Any]],
    ) -> tuple[float | None, float | None, float, int]:
        required = sum(float(item.get("local_weight") or 0.0) for item in records)
        grounded = sum(
            float(item.get("local_weight") or 0.0)
            * float(item.get("coverage_fraction") or 0.0)
            for item in records
            if _numeric(item.get("observed_score")) is not None
        )
        points = sum(
            float(item["observed_score"])
            * float(item.get("local_weight") or 0.0)
            * float(item.get("coverage_fraction") or 0.0)
            for item in records
            if _numeric(item.get("observed_score")) is not None
        )
        observed_score = points / grounded if grounded > 0.0 else None
        fraction = grounded / required if required > 0.0 else 0.0
        fraction = min(1.0, max(0.0, fraction))
        score = (
            observed_score
            if observed_score is not None
            and fraction >= MIN_PUBLISHABLE_SCORE_COVERAGE
            else None
        )
        resolved = sum(
            _numeric(item.get("observed_score")) is not None
            and float(item.get("coverage_fraction") or 0.0) > 0.0
            for item in records
        )
        return score, observed_score, fraction, resolved

    (
        l1_score,
        l1_observed_score,
        l1_fraction,
        l1_resolved_count,
    ) = covered_layer(l1_records)
    (
        l3_score,
        l3_observed_score,
        l3_fraction,
        l3_resolved_count,
    ) = covered_layer(l3_records)
    l1_coverage = {
        **l1_coverage,
        "eligible_count": len(l1_records),
        "resolved_count": l1_resolved_count,
        "fraction": l1_fraction,
        "grounded_score_fraction": l1_fraction,
        "observed_score": l1_observed_score,
        "earned_score_mass": (
            l1_observed_score * l1_fraction
            if l1_observed_score is not None
            else None
        ),
        "minimum_publishable_coverage": MIN_PUBLISHABLE_SCORE_COVERAGE,
        "coverage_threshold_passed": (
            l1_fraction >= MIN_PUBLISHABLE_SCORE_COVERAGE
        ),
        "complete": bool(l1_records) and l1_fraction >= 1.0 - 1.0e-12,
    }
    l3_coverage = {
        **l3_coverage,
        "eligible_count": len(l3_records),
        "resolved_count": l3_resolved_count,
        "fraction": l3_fraction,
        "grounded_score_fraction": l3_fraction,
        "observed_score": l3_observed_score,
        "earned_score_mass": (
            l3_observed_score * l3_fraction
            if l3_observed_score is not None
            else None
        ),
        "minimum_publishable_coverage": MIN_PUBLISHABLE_SCORE_COVERAGE,
        "coverage_threshold_passed": (
            l3_fraction >= MIN_PUBLISHABLE_SCORE_COVERAGE
        ),
        "complete": bool(l3_records) and l3_fraction >= 1.0 - 1.0e-12,
    }
    layer_values = (
        (l1_observed_score, l1_layer_weight, l1_fraction),
        (l3_observed_score, l3_layer_weight, l3_fraction),
    )
    combined_grounded_weight = sum(
        weight * fraction
        for score, weight, fraction in layer_values
        if score is not None
    )
    combined_required_weight = l1_layer_weight + l3_layer_weight
    benchmark_observed_score = (
        sum(
            float(score) * weight * fraction
            for score, weight, fraction in layer_values
            if score is not None
        )
        / combined_grounded_weight
        if combined_grounded_weight > 0.0
        else None
    )
    combined_coverage_fraction = (
        combined_grounded_weight / combined_required_weight
        if combined_required_weight > 0.0
        else 0.0
    )
    benchmark_score = (
        benchmark_observed_score
        if benchmark_observed_score is not None
        and combined_coverage_fraction >= MIN_PUBLISHABLE_SCORE_COVERAGE
        else None
    )
    benchmark_score_100 = (
        benchmark_score * 100.0 if benchmark_score is not None else None
    )
    benchmark_status = (
        "complete"
        if benchmark_score is not None
        and combined_coverage_fraction >= 1.0 - 1.0e-12
        else "partial_coverage"
        if benchmark_score is not None
        else "failed_coverage_threshold"
        if benchmark_observed_score is not None
        and combined_coverage_fraction < MIN_PUBLISHABLE_SCORE_COVERAGE
        else "insufficient_metric_coverage"
    )
    l1_diagnostics = (
        l1_diagnostics if isinstance(l1_diagnostics, dict) else {}
    )
    engineering_failures = l1_diagnostics.get("engineering_failures")
    engineering_failures = (
        [
            deepcopy(failure)
            for failure in engineering_failures
            if isinstance(failure, dict)
        ]
        if isinstance(engineering_failures, list)
        else []
    )
    unique_failure_keys: set[tuple[str, str]] = set()
    unique_engineering_failures: list[dict[str, Any]] = []
    for failure in engineering_failures:
        key = (
            str(failure.get("metric") or "unknown"),
            str(failure.get("error") or failure.get("route") or "unknown"),
        )
        if key in unique_failure_keys:
            continue
        unique_failure_keys.add(key)
        unique_engineering_failures.append(failure)

    return {
        "case_id": case_id,
        "profile_id": str(
            profile.get("scoring_profile_id") or "not persisted"
        ),
        "spec_version": str(
            profile.get("scoring_spec_version") or "not persisted"
        ),
        "deduction_multiplier": profile.get("deduction_multiplier"),
        "layer_weights": deepcopy(layer_weights),
        "n_scene": denominator.get("n_scene"),
        "object_ids": list(denominator.get("ordered_object_ids") or []),
        "combined_score_100": benchmark_score_100,
        "combined_observed_score_100": (
            benchmark_observed_score * 100.0
            if benchmark_observed_score is not None
            else None
        ),
        "combined_status": benchmark_status,
        "combined_coverage_fraction": combined_coverage_fraction,
        "final_decision_status": str(
            case_manifest.get("final_decision_status") or "unknown"
        ),
        "layers": [
            {
                "layer": "L1",
                "label": "Physical plausibility",
                "status": str(l1_report.get("status") or "not_recorded"),
                "score": l1_score,
                "observed_score": l1_observed_score,
                "score_status": (
                    "complete"
                    if l1_score is not None
                    and l1_fraction >= 1.0 - 1.0e-12
                    else "partial_coverage"
                    if l1_score is not None
                    else "failed_coverage_threshold"
                    if l1_observed_score is not None
                    else "insufficient_metric_coverage"
                ),
                "weight": l1_layer_weight,
                "coverage": deepcopy(l1_coverage),
            },
            {
                "layer": "L3",
                "label": "Implicit scene validity",
                "status": str(l3_report.get("status") or "not_recorded"),
                "score": l3_score,
                "observed_score": l3_observed_score,
                "score_status": (
                    "complete"
                    if l3_score is not None
                    and l3_fraction >= 1.0 - 1.0e-12
                    else "partial_coverage"
                    if l3_score is not None
                    else "failed_coverage_threshold"
                    if l3_observed_score is not None
                    else "insufficient_metric_coverage"
                ),
                "weight": l3_layer_weight,
                "coverage": deepcopy(l3_coverage),
            },
        ],
        "metrics": metric_records,
        "reliability": deepcopy(reliability),
        "engineering_failure_record_count": len(engineering_failures),
        "engineering_failures": unique_engineering_failures,
    }


def _mean_numeric(values: list[Any]) -> float | None:
    numbers = [value for value in (_numeric(item) for item in values) if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def run_scoring_aggregate(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize persisted case scores without filling missing coverage."""

    total_cases = len(summaries)
    published_combined = [
        value
        for value in (
            _numeric(item.get("combined_score_100")) for item in summaries
        )
        if value is not None
    ]
    observed_combined = [
        value
        for value in (
            _numeric(item.get("combined_observed_score_100"))
            for item in summaries
        )
        if value is not None
    ]
    official_score = (
        sum(published_combined) / total_cases
        if total_cases and len(published_combined) == total_cases
        else None
    )
    metric_summaries: list[dict[str, Any]] = []
    for layer, metric, label in SCORING_METRIC_ORDER:
        records = [
            next(
                (
                    value
                    for value in item.get("metrics") or []
                    if value.get("metric") == metric
                ),
                {},
            )
            for item in summaries
        ]
        published_scores = [
            value * 100.0
            for value in (_numeric(record.get("score")) for record in records)
            if value is not None
        ]
        observed_scores = [
            value * 100.0
            for value in (
                _numeric(record.get("observed_score")) for record in records
            )
            if value is not None
        ]
        coverage_values = [
            value
            for value in (
                _numeric(record.get("coverage_fraction"))
                for record in records
            )
            if value is not None
        ]
        overall_weight = next(
            (
                value
                for value in (
                    _numeric(record.get("overall_weight"))
                    for record in records
                )
                if value is not None
            ),
            None,
        )
        metric_summaries.append(
            {
                "layer": layer,
                "metric": metric,
                "label": label,
                "overall_weight": overall_weight,
                "published_case_count": len(published_scores),
                "mean_score_100": _mean_numeric(published_scores),
                "observed_case_count": len(observed_scores),
                "mean_observed_score_100": _mean_numeric(observed_scores),
                "mean_coverage_fraction": _mean_numeric(coverage_values),
            }
        )
    return {
        "case_count": total_cases,
        "published_case_count": len(published_combined),
        "official_score_100": official_score,
        "diagnostic_observed_score_100": _mean_numeric(observed_combined),
        "mean_combined_coverage_fraction": _mean_numeric(
            [item.get("combined_coverage_fraction") for item in summaries]
        ),
        "infrastructure_failure_case_count": sum(
            str(item.get("final_decision_status"))
            == "infrastructure_failure"
            for item in summaries
        ),
        "metrics": metric_summaries,
    }


__all__ = [
    "MIN_PUBLISHABLE_SCORE_COVERAGE",
    "SCORING_METRIC_ORDER",
    "case_scoring_summary",
    "run_scoring_aggregate",
]
