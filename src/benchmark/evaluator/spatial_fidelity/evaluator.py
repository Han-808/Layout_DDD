from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from benchmark.evaluator.spatial_fidelity.common import is_score
from benchmark.evaluator.spatial_fidelity.cooccurrence import (
    DEFAULT_COOCCURRENCE_CONFIG,
    evaluate_cooccurrence,
)
from benchmark.evaluator.spatial_fidelity.functional_grouping import (
    DEFAULT_FUNCTIONAL_GROUPING_CONFIG,
    evaluate_functional_grouping,
)
from benchmark.evaluator.spatial_fidelity.ontology import OntologyIndex, load_ontology
from benchmark.evaluator.spatial_fidelity.scale import DEFAULT_SCALE_CONFIG, evaluate_scale


SPATIAL_FIDELITY_EVALUATOR_VERSION = "spatial_fidelity_v1"
DEFAULT_SPATIAL_FIDELITY_CONFIG: dict[str, Any] = {
    "enabled": True,
    "vlm_policy": "fallback",
    "backend": "sceneonto_statistics_plus_conditional_vlm",
    "modules": ["scale", "cooccurrence_plausibility", "functional_grouping"],
    "ontology": {
        "path": None,
        "storage_semantics": "sparse_top_k",
        "category_aliases": {},
    },
    "metric_weights": {
        "scale": 0.5,
        "cooccurrence_plausibility": 0.5,
        "functional_grouping": 0.0,
    },
    "scale": deepcopy(DEFAULT_SCALE_CONFIG),
    "cooccurrence_plausibility": deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
    "functional_grouping": deepcopy(DEFAULT_FUNCTIONAL_GROUPING_CONFIG),
}


def evaluate_spatial_fidelity(
    scene: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    ontology: Mapping[str, Any] | str | Path | OntologyIndex | None = None,
    prompt: str = "",
    render_evidence: list[str] | None = None,
    vlm_judge: Any = None,
) -> dict[str, Any]:
    """Evaluate coarse-grained Spatial Fidelity on a canonical scene.

    ``ontology`` is deliberately injectable as an in-memory mapping for tests
    and as a path for diagnostic runs. Official callers should pass a
    hash-verified benchmark artifact rather than trusting the config path.
    """

    if not isinstance(scene, dict):
        raise TypeError("spatial fidelity scene must be a JSON object")
    resolved = _deep_merge(deepcopy(DEFAULT_SPATIAL_FIDELITY_CONFIG), config or {})
    _validate_config(resolved)
    if not bool(resolved["enabled"]):
        return _disabled_report(resolved)
    ontology_config = resolved["ontology"]
    ontology_value: Mapping[str, Any] | str | Path | None
    if isinstance(ontology, OntologyIndex):
        ontology_index = ontology
    else:
        ontology_value = ontology if ontology is not None else ontology_config.get("path")
        ontology_index = load_ontology(
            ontology_value,
            aliases=ontology_config.get("category_aliases"),
            storage_semantics=str(ontology_config["storage_semantics"]),
        )

    metrics = {
        "scale": evaluate_scale(
            scene,
            ontology_index,
            resolved["scale"],
            prompt=prompt,
            render_evidence=render_evidence,
            vlm_judge=vlm_judge,
        ),
        "cooccurrence_plausibility": evaluate_cooccurrence(
            scene,
            ontology_index,
            resolved["cooccurrence_plausibility"],
            prompt=prompt,
            render_evidence=render_evidence,
            vlm_judge=vlm_judge,
        ),
        "functional_grouping": evaluate_functional_grouping(
            resolved["functional_grouping"]
        ),
    }
    aggregate = _aggregate_metrics(metrics, resolved["metric_weights"])
    return {
        "category": "spatial_fidelity",
        "evaluator_version": SPATIAL_FIDELITY_EVALUATOR_VERSION,
        "vlm_policy": resolved["vlm_policy"],
        "backend": resolved["backend"],
        "modules": list(resolved["modules"]),
        "status": aggregate["status"],
        "reason": aggregate["reason"],
        "score": aggregate["score"],
        "partial_score": aggregate["partial_score"],
        "metric_scores": {
            name: report.get("score") for name, report in metrics.items()
        },
        "metric_partial_scores": {
            name: report.get("partial_score") for name, report in metrics.items()
        },
        "metric_weights": dict(resolved["metric_weights"]),
        "active_metrics": [
            name
            for name, weight in resolved["metric_weights"].items()
            if float(weight) > 0.0 and metrics[name].get("status") != "not_applicable"
        ],
        "placeholder_metrics": ["functional_grouping"],
        "coverage": aggregate["coverage"],
        "ontology_identity": {
            **dict(ontology_index.identity),
            "available": ontology_index.available,
        },
        "canonical_ownership": _canonical_ownership(),
        "metrics": metrics,
        "notes": [
            "Spatial Fidelity is active only for the coarse-grained evaluation plan.",
            "Statistical priors certify ordinary cases and route suspicious cases; rarity or an outlier alone is not a semantic invalidity verdict.",
            "Unknown ontology data remains visible as incomplete coverage and cannot silently increase a score.",
            "Functional Grouping is an explicit zero-weight placeholder and Functionality is out of scope.",
            "LEGACY OWNERSHIP: generic Scale and category/role Co-occurrence coherence are canonically owned by L3 Scene Quality (scale_consistency / object_pairing_consistency). Pairing excludes position, angle, orientation, and functional arrangement. These L2 implementations remain compatibility-only scored versions while L3 is a placeholder.",
        ],
    }


def _canonical_ownership() -> dict[str, Any]:
    """Legacy-ownership provenance for the coarse Spatial Fidelity metrics.

    This is documentation-only metadata; it does not change any score, weight, or
    routing. It records that generic scale/pairing coherence has moved to L3
    Scene Quality while the legacy L2 implementations are retained for
    backward compatibility.
    """

    return {
        "status": "legacy_compatibility",
        "canonical_layer": "L2_specification_fidelity_coarse",
        "scale": {
            "legacy": True,
            "still_scored": True,
            "canonical_owner": "scene_quality.scale_consistency (L3 semantic_coherence)",
        },
        "cooccurrence_plausibility": {
            "legacy": True,
            "still_scored": True,
            "canonical_owner": "scene_quality.object_pairing_consistency (L3 semantic_coherence)",
            "canonical_scope": "group_member_category_and_role_compatibility_only",
            "excluded_from_canonical_scope": [
                "position",
                "distance",
                "angle",
                "orientation",
                "functional_arrangement",
            ],
        },
        "note": (
            "Prompt-specific scale/pairing requirements remain an L2 'did the scene follow the prompt?' "
            "judgment; generic scale/pairing coherence is the L3 'is the scene coherent?' judgment. "
            "L3 Scene Quality is placeholder-only, so no metric is double counted."
        ),
    }


def _aggregate_metrics(
    metrics: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    applicable: list[str] = []
    covered: list[str] = []
    for name, raw_weight in weights.items():
        weight = float(raw_weight)
        if weight <= 0.0:
            continue
        report = metrics[name]
        if report.get("status") == "not_applicable":
            continue
        applicable.append(name)
        if is_score(report.get("score")):
            covered.append(name)
    required_weight = sum(float(weights[name]) for name in applicable)
    covered_weight = sum(float(weights[name]) for name in covered)
    partial_score = (
        sum(float(weights[name]) * float(metrics[name]["score"]) for name in covered)
        / covered_weight
        if covered_weight > 0.0
        else None
    )
    complete = bool(required_weight > 0.0 and math.isclose(covered_weight, required_weight))
    score = partial_score if complete else None
    if complete:
        status = "checked"
        reason = None
    elif required_weight <= 0.0:
        status = "not_evaluable"
        reason = "no_applicable_implemented_spatial_metrics"
    else:
        status = "incomplete"
        reason = "spatial_metric_coverage_incomplete"
    return {
        "status": status,
        "reason": reason,
        "score": None if score is None else float(score),
        "partial_score": None if partial_score is None else float(partial_score),
        "coverage": {
            "covered_metric_weight": float(covered_weight),
            "required_metric_weight": float(required_weight),
            "complete": complete,
            "covered_metrics": covered,
            "uncovered_metrics": [name for name in applicable if name not in covered],
            "not_applicable_metrics": [
                name
                for name, raw_weight in weights.items()
                if float(raw_weight) > 0.0 and metrics[name].get("status") == "not_applicable"
            ],
            "zero_weight_metrics": [
                name for name, raw_weight in weights.items() if float(raw_weight) == 0.0
            ],
        },
    }


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("enabled"), bool):
        raise ValueError("spatial_fidelity.enabled must be boolean")
    if config.get("vlm_policy") != "fallback":
        raise ValueError("spatial_fidelity.vlm_policy must be fallback")
    if config.get("backend") != "sceneonto_statistics_plus_conditional_vlm":
        raise ValueError(
            "spatial_fidelity.backend must be sceneonto_statistics_plus_conditional_vlm"
        )
    expected_modules = ["scale", "cooccurrence_plausibility", "functional_grouping"]
    if config.get("modules") != expected_modules:
        raise ValueError(f"spatial_fidelity.modules must equal {expected_modules}")
    ontology = config.get("ontology")
    if not isinstance(ontology, dict):
        raise ValueError("spatial_fidelity.ontology must be a JSON object")
    if not isinstance(ontology.get("storage_semantics"), str) or not ontology[
        "storage_semantics"
    ].strip():
        raise ValueError("spatial_fidelity.ontology.storage_semantics must be non-empty")
    if not isinstance(ontology.get("category_aliases"), dict):
        raise ValueError("spatial_fidelity.ontology.category_aliases must be a JSON object")
    weights = config.get("metric_weights")
    expected = {"scale", "cooccurrence_plausibility", "functional_grouping"}
    if not isinstance(weights, dict) or set(weights) != expected:
        raise ValueError(
            f"spatial_fidelity.metric_weights must contain exactly {sorted(expected)}"
        )
    total = 0.0
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"spatial_fidelity.metric_weights.{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(
                f"spatial_fidelity.metric_weights.{name} must be finite and non-negative"
            )
        total += number
    if not math.isclose(total, 1.0):
        raise ValueError(f"spatial_fidelity.metric_weights must sum to 1.0, got {total}")
    if float(weights["functional_grouping"]) != 0.0:
        raise ValueError(
            "functional_grouping is a placeholder and must keep metric weight 0"
        )
    for section in ("scale", "cooccurrence_plausibility", "functional_grouping"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"spatial_fidelity.{section} must be a JSON object")


def _disabled_report(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": "spatial_fidelity",
        "evaluator_version": SPATIAL_FIDELITY_EVALUATOR_VERSION,
        "vlm_policy": config["vlm_policy"],
        "backend": config["backend"],
        "modules": list(config["modules"]),
        "status": "not_applicable",
        "reason": "disabled_by_configuration",
        "score": None,
        "partial_score": None,
        "metric_scores": {
            "scale": None,
            "cooccurrence_plausibility": None,
            "functional_grouping": None,
        },
        "metric_partial_scores": {
            "scale": None,
            "cooccurrence_plausibility": None,
            "functional_grouping": None,
        },
        "metric_weights": dict(config["metric_weights"]),
        "active_metrics": [],
        "placeholder_metrics": ["functional_grouping"],
        "coverage": {
            "covered_metric_weight": 0.0,
            "required_metric_weight": 0.0,
            "complete": False,
            "covered_metrics": [],
            "uncovered_metrics": [],
            "not_applicable_metrics": ["scale", "cooccurrence_plausibility"],
            "zero_weight_metrics": ["functional_grouping"],
        },
        "ontology_identity": {
            "source": None,
            "sha256": None,
            "schema_version": None,
            "storage_semantics": config["ontology"]["storage_semantics"],
            "category_count": 0,
            "available": False,
        },
        "canonical_ownership": _canonical_ownership(),
        "metrics": {},
        "notes": [],
    }


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("spatial_fidelity config patch must be a JSON object")
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base
