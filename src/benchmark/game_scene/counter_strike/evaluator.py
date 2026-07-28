"""Counter-Strike metric orchestration and complete-only aggregation.

This module is deliberately an adapter, not another scene-evaluation
workflow.  The canonical evaluator remains authoritative for L1 Collision,
L1 Navigability, and L3 Style Consistency.  Counter-Strike-specific code owns
only its five frozen L4 metrics and the secondary composite that joins those
canonical results with L4.

Every active metric must resolve before its layer or the final composite can
receive a score.  Missing or failed metrics are never removed from the
denominator and their weights are never redistributed.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from benchmark.evaluator.profile import L0, L1, L2, L3, L4

from .judge import CounterStrikeVisualMetricResult
from .loader import CounterStrikeBenchmarkConfig, CounterStrikeCaseContract
from .topology import (
    CounterStrikeTopology,
    analyze_counter_strike_static_geometry,
)
from .visualization import render_counter_strike_topology_diagram


COUNTER_STRIKE_L4_EVALUATOR_VERSION = "counter_strike_l4_evaluator_v1"
COUNTER_STRIKE_INTEGRATED_REPORT_VERSION = (
    "counter_strike_integrated_evaluation_report_v1"
)

CANONICAL_L1_METRICS = ("collision", "navigability")
CANONICAL_L3_METRICS = ("style_consistency",)
COUNTER_STRIKE_L4_METRICS = (
    "zone_clarity",
    "route_structure",
    "spawn_balance",
    "landmark_legibility",
    "cover_diversity",
)
_TOPOLOGY_METRICS = (
    "zone_clarity",
    "route_structure",
    "spawn_balance",
    "cover_diversity",
)
_VISUAL_L4_METRICS = (
    "zone_clarity",
    "landmark_legibility",
    "cover_diversity",
)


class CounterStrikeEvaluationError(RuntimeError):
    """Raised when trusted inputs violate the CS integration contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"Counter-Strike evaluation failed [{code}]: {message}")


def evaluate_counter_strike_l4(
    scene: dict[str, Any],
    *,
    case_contract: CounterStrikeCaseContract,
    benchmark_config: CounterStrikeBenchmarkConfig,
    visual_judge: Any | None,
    frozen_evidence: Any,
    out_dir: str | Path,
    topology_analyzer: Callable[..., tuple[CounterStrikeTopology, dict[str, Any]]]
    = analyze_counter_strike_static_geometry,
    diagram_renderer: Callable[..., Path]
    = render_counter_strike_topology_diagram,
) -> dict[str, Any]:
    """Evaluate exactly five CS L4 metrics over one already-captured scene.

    Topology is constructed once and shared by the deterministic metrics and
    the disclosed topology diagram.  Each perceptual metric is called
    independently, so one model/schema failure cannot erase its sibling or the
    deterministic results.  Provider exceptions are recorded by safe class
    name only because an upstream HTTP exception may echo request material.
    """

    if not isinstance(scene, dict):
        raise CounterStrikeEvaluationError(
            "scene_invalid",
            "scene must be the canonical scene JSON object",
        )
    if not isinstance(case_contract, CounterStrikeCaseContract):
        raise CounterStrikeEvaluationError(
            "case_contract_unvalidated",
            "case_contract must come from load_counter_strike_case_contract",
        )
    if not isinstance(benchmark_config, CounterStrikeBenchmarkConfig):
        raise CounterStrikeEvaluationError(
            "benchmark_config_unvalidated",
            "benchmark_config must come from "
            "load_counter_strike_benchmark_config",
        )

    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, Any]] = {}
    topology: CounterStrikeTopology | None = None
    topology_summary: dict[str, Any] | None = None
    topology_diagram: Path | None = None
    judge_observation_diagram: Path | None = None
    audit_diagram_failure: dict[str, Any] | None = None

    try:
        topology, deterministic = topology_analyzer(
            scene,
            case_contract=case_contract,
            benchmark_config=benchmark_config,
        )
        if not isinstance(topology, CounterStrikeTopology):
            raise TypeError("topology analyzer returned an unexpected topology")
        if not isinstance(deterministic, dict) or set(deterministic) != set(
            _TOPOLOGY_METRICS
        ):
            raise ValueError(
                "topology analyzer must return exactly the four deterministic "
                "metric records"
            )
        topology_summary = topology.summary()
        for metric in _TOPOLOGY_METRICS:
            metrics[metric] = _normalize_metric_record(
                metric,
                deterministic[metric],
                expected_statuses={
                    "checked",
                    "checked_deterministic_component",
                },
                failure_stage="deterministic_topology",
            )
    except Exception as exc:
        failure = _safe_metric_failure(
            stage="deterministic_topology",
            exc=exc,
        )
        for metric in _TOPOLOGY_METRICS:
            metrics[metric] = {"metric": metric, **failure}

    if topology is not None:
        try:
            topology_diagram = Path(
                diagram_renderer(
                    topology,
                    out_path=destination / "topology_diagram.png",
                    title=f"Counter-Strike topology · {case_contract.case_id}",
                    deterministic_metrics={
                        name: metrics[name]
                        for name in _TOPOLOGY_METRICS
                        if _is_score(metrics[name].get("score"))
                    },
                )
            ).expanduser().resolve()
            if not topology_diagram.is_file():
                raise FileNotFoundError(
                    "topology diagram renderer did not create its output"
                )
        except Exception as exc:
            topology_diagram = None
            audit_diagram_failure = _safe_metric_failure(
                stage="topology_diagram",
                exc=exc,
            )
        try:
            judge_observation_diagram = Path(
                diagram_renderer(
                    topology,
                    out_path=destination / "judge_observation_diagram.png",
                    title="Neutral static occupancy and declared spawn observation",
                    deterministic_metrics=None,
                    mode="neutral_judge",
                )
            ).expanduser().resolve()
            if not judge_observation_diagram.is_file():
                raise FileNotFoundError(
                    "neutral judge diagram renderer did not create its output"
                )
            judge_diagram_failure = None
        except Exception as exc:
            judge_observation_diagram = None
            judge_diagram_failure = _safe_metric_failure(
                stage="judge_observation_diagram",
                exc=exc,
            )
    else:
        judge_diagram_failure = {
            "status": "metric_failed",
            "score": None,
            "verdict": "ambiguous",
            "reason": "topology_dependency_unavailable",
            "failure": {
                "stage": "judge_observation_diagram",
                "error_type": "TopologyUnavailable",
            },
        }

    visual_results: dict[str, dict[str, Any]] = {}
    if judge_observation_diagram is None:
        for metric in _VISUAL_L4_METRICS:
            visual_results[metric] = {
                "metric": metric,
                **deepcopy(judge_diagram_failure),
            }
    elif visual_judge is None or not callable(
        getattr(visual_judge, "judge_metric", None)
    ):
        for metric in _VISUAL_L4_METRICS:
            visual_results[metric] = {
                "metric": metric,
                "status": "unresolved",
                "score": None,
                "verdict": "ambiguous",
                "reason": "counter_strike_visual_judge_not_configured",
            }
    else:
        for metric in _VISUAL_L4_METRICS:
            try:
                raw = visual_judge.judge_metric(
                    metric,
                    evidence=frozen_evidence,
                    topology_diagram=judge_observation_diagram,
                    topology_context=_neutral_visual_context(metric),
                )
                visual_results[metric] = _normalize_visual_result(metric, raw)
            except Exception as exc:
                visual_results[metric] = {
                    "metric": metric,
                    **_safe_metric_failure(
                        stage=f"visual_judge.{metric}",
                        exc=exc,
                    ),
                }

    metrics["zone_clarity"] = _merge_zone_clarity(
        deterministic=metrics["zone_clarity"],
        perceptual=visual_results["zone_clarity"],
        config=benchmark_config.raw["l4_metrics"]["zone_clarity"],
    )
    metrics["landmark_legibility"] = _finalize_visual_metric(
        "landmark_legibility",
        visual_results["landmark_legibility"],
    )
    metrics["cover_diversity"] = _merge_cover_diversity(
        deterministic=metrics["cover_diversity"],
        perceptual=visual_results["cover_diversity"],
        config=benchmark_config.raw["l4_metrics"]["cover_diversity"],
    )
    metrics = {
        name: metrics[name]
        for name in COUNTER_STRIKE_L4_METRICS
    }

    metric_weights = {
        name: float(benchmark_config.raw["l4_metrics"][name]["weight"])
        for name in COUNTER_STRIKE_L4_METRICS
    }
    score = _complete_weighted_score(
        metrics,
        metric_weights,
        expected_metrics=COUNTER_STRIKE_L4_METRICS,
    )
    resolved = [
        name
        for name in COUNTER_STRIKE_L4_METRICS
        if _metric_resolved(metrics[name])
    ]
    return {
        "report_schema_version": "counter_strike_l4_report_v1",
        "evaluator_version": COUNTER_STRIKE_L4_EVALUATOR_VERSION,
        "layer": L4,
        "category": "counter_strike_static_spatial_design",
        "status": "evaluated" if score is not None else "incomplete",
        "score": score,
        "metrics": metrics,
        "active_metrics": list(COUNTER_STRIKE_L4_METRICS),
        "resolved_metrics": resolved,
        "active_metric_signature": "+".join(COUNTER_STRIKE_L4_METRICS),
        "coverage": {
            "complete": score is not None,
            "resolved_metric_count": len(resolved),
            "required_metric_count": len(COUNTER_STRIKE_L4_METRICS),
            "unresolved_metrics": [
                name
                for name in COUNTER_STRIKE_L4_METRICS
                if name not in resolved
            ],
            "aggregation": "complete_only_no_missing_metric_reweight",
        },
        "metric_weights": metric_weights,
        "topology": topology_summary,
        "topology_diagram": (
            topology_diagram.as_posix()
            if topology_diagram is not None
            else None
        ),
        "topology_diagram_failure": audit_diagram_failure,
        "judge_observation_diagram": (
            judge_observation_diagram.as_posix()
            if judge_observation_diagram is not None
            else None
        ),
        "judge_observation_contract": {
            "mode": "neutral_occupancy_and_declared_spawns",
            "deterministic_scores_disclosed": False,
            "deterministic_verdicts_disclosed": False,
            "inferred_roles_disclosed": False,
            "case_identity_disclosed_in_pixels": False,
        },
        "benchmark_config_sha256": benchmark_config.sha256,
        "case_contract_sha256": case_contract.sha256,
    }


def merge_counter_strike_evaluation(
    canonical_report: dict[str, Any],
    l4_report: dict[str, Any],
    *,
    benchmark_config: CounterStrikeBenchmarkConfig,
) -> dict[str, Any]:
    """Merge canonical L1/L3 results with the five CS L4 metrics.

    The canonical report is not recomputed or reinterpreted.  This adapter
    selects the frozen Game-track metrics, preserves their original records,
    and applies the CS benchmark's own complete-only layer weights.
    """

    if not isinstance(benchmark_config, CounterStrikeBenchmarkConfig):
        raise CounterStrikeEvaluationError(
            "benchmark_config_unvalidated",
            "benchmark_config must be prevalidated",
        )
    if not isinstance(canonical_report, dict):
        raise CounterStrikeEvaluationError(
            "canonical_report_invalid",
            "canonical report must be a JSON object",
        )
    if canonical_report.get("workflow") != "canonical_l0_l4":
        raise CounterStrikeEvaluationError(
            "canonical_report_invalid",
            "canonical report workflow must be 'canonical_l0_l4'",
        )
    layer_reports = canonical_report.get("layer_reports")
    if not isinstance(layer_reports, dict):
        raise CounterStrikeEvaluationError(
            "canonical_report_invalid",
            "canonical report has no layer_reports object",
        )

    l1_metrics = _canonical_metrics(
        layer_reports,
        layer=L1,
        expected=CANONICAL_L1_METRICS,
    )
    l3_metrics = _canonical_metrics(
        layer_reports,
        layer=L3,
        expected=CANONICAL_L3_METRICS,
    )
    l4_metrics = _validated_l4_metrics(l4_report)

    canonical_weights = benchmark_config.raw["composite"][
        "canonical_metric_weights"
    ]
    l1 = _metric_layer_report(
        layer=L1,
        category="physical_plausibility",
        metrics=l1_metrics,
        weights={
            name: float(canonical_weights[name])
            for name in CANONICAL_L1_METRICS
        },
        expected=CANONICAL_L1_METRICS,
    )
    l3 = _metric_layer_report(
        layer=L3,
        category="scene_quality",
        metrics=l3_metrics,
        weights={"style_consistency": float(canonical_weights["style_consistency"])},
        expected=CANONICAL_L3_METRICS,
    )
    l4 = deepcopy(l4_report)
    l4["metrics"] = l4_metrics
    l4_score = _complete_weighted_score(
        l4_metrics,
        {
            name: float(benchmark_config.raw["l4_metrics"][name]["weight"])
            for name in COUNTER_STRIKE_L4_METRICS
        },
        expected_metrics=COUNTER_STRIKE_L4_METRICS,
    )
    l4_resolved = [
        name
        for name in COUNTER_STRIKE_L4_METRICS
        if _metric_resolved(l4_metrics[name])
    ]
    l4["score"] = l4_score
    l4["status"] = "evaluated" if l4_score is not None else "incomplete"
    l4["active_metrics"] = list(COUNTER_STRIKE_L4_METRICS)
    l4["resolved_metrics"] = l4_resolved
    l4["active_metric_signature"] = "+".join(COUNTER_STRIKE_L4_METRICS)
    l4["coverage"] = {
        "complete": l4_score is not None,
        "resolved_metric_count": len(l4_resolved),
        "required_metric_count": len(COUNTER_STRIKE_L4_METRICS),
        "unresolved_metrics": [
            name
            for name in COUNTER_STRIKE_L4_METRICS
            if name not in l4_resolved
        ],
        "aggregation": "complete_only_no_missing_metric_reweight",
    }

    scoring_layers = {L1: l1, L3: l3, L4: l4}
    raw_layer_weights = benchmark_config.raw["composite"]["layer_weights"]
    layer_weights = {
        L1: float(raw_layer_weights["l1_physical_plausibility"]),
        L3: float(raw_layer_weights["l3_scene_quality"]),
        L4: float(raw_layer_weights["l4_static_spatial_design"]),
    }
    benchmark_score = _complete_weighted_score(
        scoring_layers,
        layer_weights,
        expected_metrics=(L1, L3, L4),
    )
    resolved_layers = [
        name
        for name in (L1, L3, L4)
        if _metric_resolved(scoring_layers[name])
    ]

    l0 = deepcopy(
        layer_reports.get(
            L0,
            {
                "layer": L0,
                "status": "unresolved",
                "score": None,
                "affects_score": False,
                "reason": "canonical_l0_report_missing",
            },
        )
    )
    l2 = deepcopy(
        layer_reports.get(
            L2,
            {
                "layer": L2,
                "status": "not_applicable",
                "score": None,
                "affects_score": False,
            },
        )
    )
    result_layers = {
        L0: l0,
        L1: l1,
        L2: l2,
        L3: l3,
        L4: l4,
    }
    metric_vector = {
        **{name: deepcopy(l1_metrics[name]) for name in CANONICAL_L1_METRICS},
        **{name: deepcopy(l3_metrics[name]) for name in CANONICAL_L3_METRICS},
        **{name: deepcopy(l4_metrics[name]) for name in COUNTER_STRIKE_L4_METRICS},
    }
    return {
        "report_schema_version": COUNTER_STRIKE_INTEGRATED_REPORT_VERSION,
        "workflow": "counter_strike_canonical_adapter",
        "profile_version": benchmark_config.raw["profile_version"],
        "scene_id": canonical_report.get("scene_id"),
        "request_id": canonical_report.get("request_id"),
        "evaluation_status": (
            "complete" if benchmark_score is not None else "incomplete"
        ),
        "benchmark_score": benchmark_score,
        "benchmark_score_status": (
            "complete"
            if benchmark_score is not None
            else "insufficient_metric_coverage"
        ),
        "layer_reports": result_layers,
        "metric_vector": metric_vector,
        "coverage": {
            "complete": benchmark_score is not None,
            "required_layers": [L1, L3, L4],
            "resolved_layers": resolved_layers,
            "unresolved_layers": [
                name for name in (L1, L3, L4) if name not in resolved_layers
            ],
            "aggregation": "complete_only_no_missing_metric_reweight",
        },
        "evaluation_config": {
            "benchmark_config_sha256": benchmark_config.sha256,
            "layer_weights": layer_weights,
            "canonical_metric_weights": {
                name: float(canonical_weights[name])
                for name in (*CANONICAL_L1_METRICS, *CANONICAL_L3_METRICS)
            },
            "l4_metric_weights": {
                name: float(benchmark_config.raw["l4_metrics"][name]["weight"])
                for name in COUNTER_STRIKE_L4_METRICS
            },
            "canonical_source_profile_version": canonical_report.get(
                "profile_version"
            ),
            "canonical_source_workflow": canonical_report.get("workflow"),
            "game_profile_modified": False,
        },
    }


def _canonical_metrics(
    layer_reports: dict[str, Any],
    *,
    layer: str,
    expected: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    layer_report = layer_reports.get(layer)
    raw_metrics = (
        layer_report.get("metrics")
        if isinstance(layer_report, dict)
        else None
    )
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}
    results: dict[str, dict[str, Any]] = {}
    for metric in expected:
        raw = raw_metrics.get(metric)
        if not isinstance(raw, dict):
            results[metric] = {
                "metric": metric,
                "status": "metric_failed",
                "score": None,
                "verdict": "ambiguous",
                "reason": "canonical_metric_record_missing",
                "failure": {
                    "stage": "canonical_report_merge",
                    "error_type": "MissingMetricRecord",
                },
            }
            continue
        record = deepcopy(raw)
        record.setdefault("metric", metric)
        if record.get("metric") != metric:
            results[metric] = {
                "metric": metric,
                "status": "metric_failed",
                "score": None,
                "verdict": "ambiguous",
                "reason": "canonical_metric_identity_mismatch",
                "failure": {
                    "stage": "canonical_report_merge",
                    "error_type": "MetricIdentityMismatch",
                },
                "source_record": record,
            }
            continue
        if not _is_score(record.get("score")):
            record["score"] = None
            if record.get("status") not in {
                "unresolved",
                "metric_failed",
                "incomplete",
            }:
                record["source_status"] = record.get("status")
                record["status"] = "unresolved"
            record.setdefault("reason", "canonical_metric_not_resolved")
        results[metric] = record
    return results


def _validated_l4_metrics(
    l4_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(l4_report, dict):
        raise CounterStrikeEvaluationError(
            "l4_report_invalid",
            "L4 report must be a JSON object",
        )
    raw = l4_report.get("metrics")
    if not isinstance(raw, dict) or set(raw) != set(COUNTER_STRIKE_L4_METRICS):
        raise CounterStrikeEvaluationError(
            "l4_metric_set_invalid",
            "L4 report must contain exactly "
            f"{list(COUNTER_STRIKE_L4_METRICS)}",
        )
    results: dict[str, dict[str, Any]] = {}
    for metric in COUNTER_STRIKE_L4_METRICS:
        value = raw[metric]
        if not isinstance(value, dict) or value.get("metric") != metric:
            raise CounterStrikeEvaluationError(
                "l4_metric_record_invalid",
                f"L4 metric {metric!r} has an invalid identity record",
            )
        results[metric] = deepcopy(value)
    return results


def _metric_layer_report(
    *,
    layer: str,
    category: str,
    metrics: dict[str, dict[str, Any]],
    weights: dict[str, float],
    expected: tuple[str, ...],
) -> dict[str, Any]:
    score = _complete_weighted_score(
        metrics,
        weights,
        expected_metrics=expected,
    )
    resolved = [name for name in expected if _metric_resolved(metrics[name])]
    return {
        "layer": layer,
        "category": category,
        "status": "evaluated" if score is not None else "incomplete",
        "score": score,
        "metrics": deepcopy(metrics),
        "active_metrics": list(expected),
        "resolved_metrics": resolved,
        "active_metric_signature": "+".join(expected),
        "coverage": {
            "complete": score is not None,
            "resolved_metric_count": len(resolved),
            "required_metric_count": len(expected),
            "unresolved_metrics": [
                name for name in expected if name not in resolved
            ],
            "aggregation": "complete_only_no_missing_metric_reweight",
        },
        "metric_weights": deepcopy(weights),
    }


def _normalize_metric_record(
    metric: str,
    value: Any,
    *,
    expected_statuses: set[str],
    failure_stage: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("metric") != metric:
        return {
            "metric": metric,
            "status": "metric_failed",
            "score": None,
            "verdict": "ambiguous",
            "reason": "metric_record_invalid",
            "failure": {
                "stage": failure_stage,
                "error_type": "MetricRecordInvalid",
            },
        }
    result = deepcopy(value)
    if result.get("status") not in expected_statuses or not _is_score(
        result.get("score")
    ):
        result["status"] = "metric_failed"
        result["score"] = None
        result["verdict"] = "ambiguous"
        result["reason"] = "metric_record_invalid"
        result["failure"] = {
            "stage": failure_stage,
            "error_type": "MetricRecordInvalid",
        }
    return result


def _normalize_visual_result(metric: str, value: Any) -> dict[str, Any]:
    if isinstance(value, CounterStrikeVisualMetricResult):
        result = value.to_dict()
    elif callable(getattr(value, "to_dict", None)):
        result = value.to_dict()
    elif isinstance(value, dict):
        result = deepcopy(value)
    else:
        raise TypeError("visual judge returned an unsupported result")
    if not isinstance(result, dict) or result.get("metric") != metric:
        raise ValueError("visual judge returned the wrong metric identity")
    status = result.get("status")
    if status == "checked":
        if not _is_score(result.get("score")):
            raise ValueError("checked visual metric is missing a valid score")
    elif status == "unresolved":
        result["score"] = None
        result["verdict"] = "ambiguous"
    else:
        raise ValueError("visual metric status must be checked or unresolved")
    return result


def _merge_zone_clarity(
    *,
    deterministic: dict[str, Any],
    perceptual: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return _merge_deterministic_perceptual_metric(
        metric="zone_clarity",
        backend="deterministic_topology_plus_vlm",
        deterministic=deterministic,
        perceptual=perceptual,
        config=config,
    )


def _merge_cover_diversity(
    *,
    deterministic: dict[str, Any],
    perceptual: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return _merge_deterministic_perceptual_metric(
        metric="cover_diversity",
        backend="deterministic_cover_assemblies_plus_vlm",
        deterministic=deterministic,
        perceptual=perceptual,
        config=config,
    )


def _merge_deterministic_perceptual_metric(
    *,
    metric: str,
    backend: str,
    deterministic: dict[str, Any],
    perceptual: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Merge independent components without letting a proxy erase a defect.

    The VLM receives no deterministic score or verdict.  A resolved visual
    ``invalid`` is an explicit significant defect and therefore caps the final
    score at the perceptual score; a deterministic proxy may otherwise be
    rescued by independent visual evidence through the frozen weighted score.
    """

    result = {
        "metric": metric,
        "backend": backend,
        "deterministic_component": deepcopy(deterministic),
        "perceptual_component": deepcopy(perceptual),
    }
    if deterministic.get("status") == "metric_failed":
        return {
            **result,
            "status": "metric_failed",
            "score": None,
            "verdict": "ambiguous",
            "reason": f"deterministic_{metric}_component_failed",
            "failure": deepcopy(deterministic.get("failure")),
        }
    if perceptual.get("status") == "metric_failed":
        return {
            **result,
            "status": "metric_failed",
            "score": None,
            "verdict": "ambiguous",
            "reason": f"perceptual_{metric}_component_failed",
            "failure": deepcopy(perceptual.get("failure")),
        }
    if not _metric_resolved(deterministic) or not _metric_resolved(perceptual):
        return {
            **result,
            "status": "unresolved",
            "score": None,
            "verdict": "ambiguous",
            "reason": f"{metric}_component_unresolved",
        }
    weights = config["score_components"]
    weighted_score = (
        float(deterministic["score"]) * float(weights["deterministic"])
        + float(perceptual["score"]) * float(weights["perceptual"])
    )
    threshold = float(config["valid_threshold"])
    perceptual_invalid = perceptual.get("verdict") == "invalid"
    score = (
        min(weighted_score, float(perceptual["score"]))
        if perceptual_invalid
        else weighted_score
    )
    return {
        **result,
        "status": "checked",
        "score": score,
        "verdict": "valid" if score >= threshold else "invalid",
        "reason": (
            "significant_perceptual_defect_veto"
            if perceptual_invalid
            else "frozen_weighted_independent_components"
        ),
        "score_components": {
            "deterministic": float(weights["deterministic"]),
            "perceptual": float(weights["perceptual"]),
        },
        "weighted_score_before_perceptual_veto": weighted_score,
        "perceptual_invalid_veto": perceptual_invalid,
        "valid_threshold": threshold,
    }


def _neutral_visual_context(metric: str) -> dict[str, Any]:
    """Return the only non-pixel context disclosed to the CS visual judge."""

    if metric not in _VISUAL_L4_METRICS:
        raise ValueError(f"unsupported neutral visual metric {metric!r}")
    return {
        "schema_version": "counter_strike_neutral_visual_context_v1",
        "metric": metric,
        "scope": "static_3d_environment_only",
        "observation_aid": {
            "shows": [
                "walkable_free_space",
                "blocking_footprints",
                "declared_team_a_spawn_points",
                "declared_team_b_spawn_points",
            ],
            "omits": [
                "deterministic_scores",
                "deterministic_verdicts",
                "inferred_zone_roles",
                "inferred_routes",
                "cover_proposals",
                "engagement_anchor",
                "case_identity",
            ],
            "ground_truth": False,
        },
    }


def _finalize_visual_metric(
    metric: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    normalized = deepcopy(result)
    normalized["metric"] = metric
    if normalized.get("status") in {"metric_failed", "unresolved"}:
        normalized["score"] = None
        normalized["verdict"] = "ambiguous"
    return normalized


def _safe_metric_failure(*, stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "metric_failed",
        "score": None,
        "verdict": "ambiguous",
        "reason": "metric_execution_failed",
        "failure": {
            "stage": stage,
            "error_type": type(exc).__name__,
        },
    }


def _complete_weighted_score(
    metrics: dict[str, dict[str, Any]],
    weights: dict[str, float],
    *,
    expected_metrics: tuple[str, ...],
) -> float | None:
    if set(metrics) != set(expected_metrics) or set(weights) != set(
        expected_metrics
    ):
        return None
    if any(not _metric_resolved(metrics[name]) for name in expected_metrics):
        return None
    return float(
        sum(
            float(metrics[name]["score"]) * float(weights[name])
            for name in expected_metrics
        )
    )


def _metric_resolved(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("status")
        not in {
            "unresolved",
            "metric_failed",
            "incomplete",
            "not_applicable",
            "disabled",
        }
        and _is_score(value.get("score"))
    )


def _is_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )
