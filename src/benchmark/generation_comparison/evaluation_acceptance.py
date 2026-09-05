"""Experiment acceptance of unchanged evaluator reports; never scoring.

The opt-in FrozenAssets policy exempts only the two asset-selection metrics
that factual frozen ownership makes inapplicable. It cannot waive missing
evidence, infrastructure failures, or any other applicable metric.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from benchmark.evaluator.profile import L0, L2, L3, SCORING_LAYERS
from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.scene_io.validate import ArtifactValidationError


COMPLETE_SCORE = "complete_score_v1"
FROZEN_REQUIRED_METRICS = "frozen_assets_required_metrics_v1"
EXEMPT_METRICS = frozenset({"object_pairing_consistency", "style_consistency"})
MINIMUM_L3_COVERAGE = 0.8
FROZEN_OWNERSHIP = {
    "mode": "benchmark_provided", "identity_owner": "benchmark",
    "category_selection_owner": "benchmark", "scale_owner": "benchmark",
    "appearance_owner": "benchmark", "arrangement_owner": "generator",
}


def acceptance_policy_name(policy: Mapping[str, Any], *, mode: str | None = None) -> str:
    name = policy.get("acceptance_policy", COMPLETE_SCORE)
    if not isinstance(name, str) or name not in {COMPLETE_SCORE, FROZEN_REQUIRED_METRICS}:
        raise ArtifactValidationError("unsupported experiment evaluation acceptance policy")
    if name == FROZEN_REQUIRED_METRICS and mode not in {None, "frozen_assets"}:
        raise ArtifactValidationError("frozen evaluation acceptance requires frozen_assets protocol")
    return name


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _names(value: Any) -> list[str] | None:
    if (not isinstance(value, list) or any(not isinstance(item, str) for item in value)
            or len(value) != len(set(value))):
        return None
    return value


def _inventory_matches_plan(report: Mapping[str, Any]) -> bool:
    """Check the existing evaluator's plan; do not define a second metric set."""
    plan = _map(_map(report.get("evaluation_plan")).get("layers"))
    coverage = _map(report.get("coverage"))
    active = _map(coverage.get("active_metrics_by_layer"))
    resolved = _map(coverage.get("resolved_metrics_by_layer"))
    layers = _map(report.get("layer_reports"))
    if set(active) != set(SCORING_LAYERS) or set(resolved) != set(SCORING_LAYERS):
        return False
    expected_layers = []
    for layer in SCORING_LAYERS:
        specification = _map(plan.get(layer))
        metrics = specification.get("metrics")
        if not isinstance(metrics, Mapping) or not _finite(specification.get("weight")):
            return False
        expected = set()
        for metric, value in metrics.items():
            definition = _map(value)
            weight = definition.get("weight")
            if not isinstance(metric, str) or not _finite(weight):
                return False
            if (specification.get("enabled") is True and definition.get("enabled") is True
                    and weight > 0 and (layer != L2 or definition.get("applicable") is True)):
                expected.add(metric)
        names = _names(active.get(layer))
        completed = _names(resolved.get(layer))
        envelope = _map(layers.get(layer))
        if (names is None or completed is None or set(names) != expected
                or not set(completed).issubset(expected)
                or names != envelope.get("active_metrics", [])
                or completed != envelope.get("resolved_metrics", [])):
            return False
        if expected and specification["weight"] > 0:
            expected_layers.append(layer)
    return (L3 in expected_layers
            and _names(coverage.get("active_layers")) == expected_layers
            and _names(coverage.get("covered_layers")) == expected_layers)


def evaluate_report_acceptance(
    report: Mapping[str, Any], policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an auditable gate decision without mutating/reweighting a report.

    Strict complete-score behavior remains the default. Opt-in decisions require
    canonical v2 coverage/reliability evidence as well as the 0.8 L3 floor.
    A low but fully evaluated score is accepted; an ungrounded high score is not.
    """
    name = acceptance_policy_name(policy)
    status = report.get("benchmark_score_status")
    complete = _finite(report.get("benchmark_score")) and status == "complete"
    reasons: list[str] = []
    exempted: list[str] = []
    required: list[str] = []
    if name == COMPLETE_SCORE:
        if not complete:
            reasons.append("complete_benchmark_score_required")
    else:
        if report.get("report_schema_version") != "scene_evaluation_report_v2" or report.get("workflow") != "canonical_l0_l4":
            reasons.append("unsupported_evaluation_report")
        if not _finite(report.get("benchmark_score")) or status not in {"complete", "partial_coverage"}:
            reasons.append("benchmark_score_unavailable")
        config = _map(report.get("evaluation_config"))
        if (_map(policy.get("static_kwargs")).get("asset_policy") != FROZEN_OWNERSHIP
                or config.get("asset_policy") != FROZEN_OWNERSHIP):
            reasons.append("frozen_asset_ownership_required")
        coverage = _map(report.get("coverage"))
        reliability = _map(report.get("scoring_reliability"))
        layers = _map(report.get("layer_reports"))
        l3_fraction = _map(_map(layers.get(L3)).get("coverage")).get("fraction")
        if not _finite(l3_fraction) or not MINIMUM_L3_COVERAGE <= l3_fraction <= 1:
            reasons.append("l3_coverage_below_0_8_or_missing")
        if _map(layers.get(L0)).get("status") != "passed":
            reasons.append("l0_not_passed")
        if (coverage.get("score_resolution_complete") is not True
                or coverage.get("coverage_threshold_passed") is not True
                or not coverage.get("comparability_signature")):
            reasons.append("canonical_score_coverage_incomplete")
        if reliability.get("schema_version") != "scoring_reliability_v2":
            reasons.append("reliability_evidence_missing")
        if reliability.get("infrastructure_failures") != []:
            reasons.append("infrastructure_failure_or_missing_audit")
        if reliability.get("unresolved_claims") != []:
            reasons.append("unresolved_claims_or_missing_audit")
        unresolved_ids = reliability.get("unresolved_metric_ids")
        if not isinstance(unresolved_ids, list) or any(value != "scoring_coverage" for value in unresolved_ids):
            reasons.append("unresolved_required_metrics")
        active = _map(coverage.get("active_metrics_by_layer"))
        resolved = _map(coverage.get("resolved_metrics_by_layer"))
        if not _inventory_matches_plan(report):
            reasons.append("metric_inventory_differs_from_evaluation_plan")
        records = reliability.get("metrics")
        records = records if isinstance(records, list) else []
        by_id = {record["metric_id"]: record for record in records
                 if isinstance(record, Mapping) and isinstance(record.get("metric_id"), str)}
        if len(by_id) != len(records):
            reasons.append("malformed_or_duplicate_reliability_metric")
        # Cross-check the reliability and coverage inventories so that dropping
        # an unresolved metric from one envelope cannot bypass the gate.
        coverage_ids = {f"{layer}.{metric}" for layer, names in active.items()
                        if _names(names) is not None for metric in names}
        reliability_ids = {key for key, record in by_id.items() if record.get("active") is True}
        if not coverage_ids or coverage_ids != reliability_ids:
            reasons.append("metric_inventory_evidence_mismatch")
        applicability = _map(_map(config.get("metric_applicability")).get(L3))
        for layer, names in active.items():
            if _names(names) is None:
                reasons.append(f"invalid_metric_inventory:{layer}")
                continue
            envelope = _map(layers.get(layer))
            metrics = _map(envelope.get("metrics"))
            if names and layer != L3 and _map(envelope.get("coverage")).get("complete") is not True:
                reasons.append(f"required_layer_incomplete:{layer}")
            for metric in names:
                metric_id = f"{layer}.{metric}"
                raw = _map(metrics.get(metric))
                record = _map(by_id.get(metric_id))
                metric_coverage = _map(raw.get("coverage"))
                if (layer == L3 and metric in EXEMPT_METRICS
                        and _map(applicability.get(metric)).get("applicability") == "not_relevant"
                        and _map(raw.get("applicability")).get("applicability") == "not_relevant"):
                    exempted.append(metric_id)
                    # Inapplicability is the only permissible defaulted source.
                    grounding = _map(metric_coverage.get("score_grounding"))
                    defaults = grounding.get("defaulted_units")
                    if (not isinstance(defaults, list) or not defaults or any(
                        _map(unit).get("unit_id") != "metric_not_relevant_for_asset_policy" for unit in defaults
                    ) or grounding.get("defaulted_count") != len(defaults)
                            or record.get("infrastructure_failure") is not False):
                        reasons.append(f"exempt_metric_has_other_failure:{metric_id}")
                    continue
                required.append(metric_id)
                if (record.get("active") is not True or record.get("scientifically_complete") is not True
                        or record.get("infrastructure_failure") is not False
                        or not raw or metric not in (_names(resolved.get(layer)) or [])):
                    reasons.append(f"required_metric_unresolved:{metric_id}")
                if layer == L3:
                    if (_map(applicability.get(metric)).get("applicability") != "relevant"
                            or _map(raw.get("applicability")).get("applicability") != "relevant"
                            or metric_coverage.get("complete") is not True
                            or metric_coverage.get("fraction") != 1):
                        reasons.append(f"required_metric_ungrounded:{metric_id}")
                grounding = _map(metric_coverage.get("score_grounding"))
                if layer == L3 and metric in {"functional_consistency", "semantic_placement_consistency"} and not grounding:
                    reasons.append(f"required_metric_grounding_missing:{metric_id}")
                if grounding and (grounding.get("complete") is not True
                                  or grounding.get("fraction") != 1
                                  or grounding.get("defaulted_count") != 0
                                  or grounding.get("defaulted_units", []) != []):
                    reasons.append(f"required_metric_defaulted:{metric_id}")
        if status == "partial_coverage" and not exempted:
            reasons.append("no_inapplicability_explains_partial_coverage")
    return {
        "schema_version": "experiment_evaluation_acceptance_v1", "policy": name,
        "accepted": not reasons, "evaluation_complete": complete,
        "accepted_partial_coverage": not reasons and status == "partial_coverage",
        "raw_benchmark_score_status": status,
        "minimum_l3_coverage": MINIMUM_L3_COVERAGE if name == FROZEN_REQUIRED_METRICS else None,
        "report_sha256": canonical_json_sha256(report),
        "required_metric_ids": sorted(required), "exempt_metric_ids": sorted(exempted),
        "reasons": sorted(set(reasons)), "scoring_modified": False,
        "comparability_signature": _map(report.get("coverage")).get("comparability_signature"),
    }
