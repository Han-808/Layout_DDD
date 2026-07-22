from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from benchmark.nl_scene.converter import COARSE_GRAINED, FINE_GRAINED, PROMPT_GRANULARITIES


VLM_POLICIES = {"never", "fallback", "primary"}
FINE_GRAINED_MODE = "fine_grained_mode"
COARSE_GRAINED_MODE = "coarse_grained_mode"
EVALUATION_MODES = {FINE_GRAINED_MODE, COARSE_GRAINED_MODE}
EVALUATION_MODE_BY_GRANULARITY = {
    FINE_GRAINED: FINE_GRAINED_MODE,
    COARSE_GRAINED: COARSE_GRAINED_MODE,
}
FIDELITY_CATEGORY_BY_GRANULARITY = {
    FINE_GRAINED: "prompt_fidelity",
    COARSE_GRAINED: "spatial_fidelity",
}

DEFAULT_EVALUATION_PROFILE = {
    "profile_version": "scene_evaluation_draft_v1",
    "status": "initial_not_frozen",
    "weights": {
        "prompt_fidelity": 0.25,
        "spatial_fidelity": 0.25,
        "structural_validity": 0.35,
        "visual_quality": 0.40,
    },
    "prompt_fidelity": {
        FINE_GRAINED: {
            "vlm_policy": "fallback",
            "backend": "structured_claims",
            "modules": ["object_presence", "object_count", "oor", "oar"],
            "alignment_modules": ["object_mapping"],
        },
    },
    "spatial_fidelity": {
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
            "scale": 0.50,
            "cooccurrence_plausibility": 0.50,
            "functional_grouping": 0.0,
        },
        "scale": {
            "enabled": True,
            "abs_tolerance_m": 0.05,
            "hard_low_factor": 0.5,
            "hard_high_factor": 2.0,
            "horizontal_axis_policy": "best_permutation",
            "required_axes": ["width", "depth", "height"],
            "skip_categories": ["door", "window", "wall", "floor", "ceiling"],
        },
        "cooccurrence_plausibility": {
            "enabled": True,
            "rare_threshold": 0.01,
            "absent_threshold": 0.001,
            "strong_threshold": 0.20,
            "functional_hint_threshold": 0.70,
            "min_anchor_observation_count_for_rarity": 100,
            "pair_unit": "unique_unordered_category_pair",
            "prefer_room_conditioned": True,
            "sparse_missing_means_unknown": True,
            "skip_categories": [
                "door",
                "window",
                "wall",
                "floor",
                "ceiling",
                "pendant_lamp",
                "curtain",
            ],
        },
        "functional_grouping": {
            "enabled": False,
            "implemented": False,
        },
    },
    "structural_validity": {
        "vlm_policy": "fallback",
        "backend": "deterministic_evidence_plus_conditional_vlm",
        "modules": ["collision", "oob", "navigability", "accessibility", "support"],
        "never_vlm_modules": ["navigability", "accessibility"],
        # Applicability is benchmark-case/profile owned. Accessibility remains
        # disabled until a frozen target/approach annotation exists; generated
        # ``metadata.interactive`` must not activate an official metric.
        "applicability": {
            "collision": True,
            "oob": True,
            "navigability": True,
            "accessibility": False,
            "support": True,
        },
    },
    "visual_quality": {
        "vlm_policy": "primary",
        "backend": "standardized_renders",
        "modules": ["visual_coherence", "commonsense_plausibility", "style_and_appearance"],
    },
}


def resolve_evaluation_profile(value: dict[str, Any] | None = None) -> dict[str, Any]:
    patch = deepcopy(value or {})
    # Compatibility for fully specified v0 fine-grained profiles.  The old
    # ``prompt_fidelity`` weight occupied the shared category-2 slot, so it is
    # the only safe value from which to initialize the new coarse track.
    patch_weights = patch.get("weights") if isinstance(patch, dict) else None
    if (
        isinstance(patch_weights, dict)
        and "prompt_fidelity" in patch_weights
        and "spatial_fidelity" not in patch_weights
    ):
        patch_weights["spatial_fidelity"] = patch_weights["prompt_fidelity"]
    profile = _deep_merge(deepcopy(DEFAULT_EVALUATION_PROFILE), patch)
    weights = profile.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("evaluation profile weights must be a JSON object")
    expected = {
        "prompt_fidelity",
        "spatial_fidelity",
        "structural_validity",
        "visual_quality",
    }
    if set(weights) != expected:
        raise ValueError(f"evaluation profile weights must contain exactly {sorted(expected)}")
    for name, raw_weight in weights.items():
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(f"evaluation profile weight {name} must be numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"evaluation profile weight {name} must be finite and non-negative")
        weights[name] = weight
    for granularity, fidelity_category in FIDELITY_CATEGORY_BY_GRANULARITY.items():
        total = sum(
            weights[name]
            for name in (fidelity_category, "structural_validity", "visual_quality")
        )
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"evaluation profile weights for {granularity} must sum to 1.0, got {total}"
            )
    for category in ["structural_validity", "visual_quality"]:
        _validate_policy(profile[category].get("vlm_policy"), category)
    _validate_policy(profile["spatial_fidelity"].get("vlm_policy"), "spatial_fidelity")
    if not isinstance(profile["spatial_fidelity"].get("enabled"), bool):
        raise ValueError("spatial_fidelity.enabled must be boolean")
    expected_spatial_modules = [
        "scale",
        "cooccurrence_plausibility",
        "functional_grouping",
    ]
    if profile["spatial_fidelity"].get("modules") != expected_spatial_modules:
        raise ValueError(
            f"spatial_fidelity.modules must equal {expected_spatial_modules}"
        )
    if (
        profile["spatial_fidelity"].get("backend")
        != "sceneonto_statistics_plus_conditional_vlm"
    ):
        raise ValueError(
            "spatial_fidelity.backend must be "
            "sceneonto_statistics_plus_conditional_vlm"
        )
    spatial_weights = profile["spatial_fidelity"].get("metric_weights")
    expected_spatial_metrics = {"scale", "cooccurrence_plausibility", "functional_grouping"}
    if not isinstance(spatial_weights, dict) or set(spatial_weights) != expected_spatial_metrics:
        raise ValueError(
            "spatial_fidelity.metric_weights must contain exactly "
            f"{sorted(expected_spatial_metrics)}"
        )
    for name, raw_weight in spatial_weights.items():
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(f"spatial_fidelity.metric_weights.{name} must be numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"spatial_fidelity.metric_weights.{name} must be finite and non-negative"
            )
        spatial_weights[name] = weight
    if not math.isclose(sum(spatial_weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("spatial_fidelity.metric_weights must sum to 1.0")
    if float(spatial_weights["functional_grouping"]) != 0.0:
        raise ValueError("unimplemented functional_grouping must have zero metric weight")
    functional_config = profile["spatial_fidelity"].get("functional_grouping")
    if not isinstance(functional_config, dict):
        raise ValueError("spatial_fidelity.functional_grouping must be a JSON object")
    if functional_config.get("enabled") is not False or functional_config.get("implemented") is not False:
        raise ValueError("functional_grouping is not implemented and must remain disabled")
    ontology_config = profile["spatial_fidelity"].get("ontology")
    if not isinstance(ontology_config, dict):
        raise ValueError("spatial_fidelity.ontology must be a JSON object")
    if ontology_config.get("path") is not None:
        raise ValueError(
            "spatial_fidelity.ontology.path must be null; pass a trusted ontology artifact at runtime"
        )
    scale_config = profile["spatial_fidelity"].get("scale")
    if not isinstance(scale_config, dict) or not isinstance(scale_config.get("enabled"), bool):
        raise ValueError("spatial_fidelity.scale.enabled must be boolean")
    cooccurrence_config = profile["spatial_fidelity"].get("cooccurrence_plausibility")
    if not isinstance(cooccurrence_config, dict) or not isinstance(
        cooccurrence_config.get("enabled"), bool
    ):
        raise ValueError("spatial_fidelity.cooccurrence_plausibility.enabled must be boolean")
    if cooccurrence_config.get("sparse_missing_means_unknown") is not True:
        raise ValueError(
            "spatial_fidelity.cooccurrence_plausibility.sparse_missing_means_unknown "
            "must remain true"
        )
    applicability = profile["structural_validity"].get("applicability")
    expected_metrics = {"collision", "oob", "navigability", "accessibility", "support"}
    if not isinstance(applicability, dict) or set(applicability) != expected_metrics:
        raise ValueError(
            "structural_validity.applicability must contain exactly "
            f"{sorted(expected_metrics)}"
        )
    for metric_name, value in applicability.items():
        if not isinstance(value, bool):
            raise ValueError(f"structural_validity.applicability.{metric_name} must be boolean")
    prompt_fidelity = profile.get("prompt_fidelity")
    if not isinstance(prompt_fidelity, dict) or set(prompt_fidelity) != {FINE_GRAINED}:
        raise ValueError(
            "prompt_fidelity must contain only the fine_grained configuration; "
            "coarse_grained uses spatial_fidelity"
        )
    _validate_policy(
        prompt_fidelity[FINE_GRAINED].get("vlm_policy"),
        f"prompt_fidelity.{FINE_GRAINED}",
    )
    return profile


def build_evaluation_plan(
    *,
    prompt_granularity: str,
    has_object_plan: bool,
    render_evidence_count: int,
    has_spatial_fidelity_ontology: bool = False,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve_evaluation_profile(profile)
    if prompt_granularity not in PROMPT_GRANULARITIES:
        raise ValueError(f"Unknown prompt granularity {prompt_granularity!r}")
    evaluation_mode = evaluation_mode_for_prompt_granularity(prompt_granularity)
    fidelity_category = FIDELITY_CATEGORY_BY_GRANULARITY[prompt_granularity]
    fidelity = (
        resolved["prompt_fidelity"][FINE_GRAINED]
        if prompt_granularity == FINE_GRAINED
        else resolved["spatial_fidelity"]
    )
    fidelity_missing = []
    if prompt_granularity == FINE_GRAINED and not has_object_plan:
        fidelity_missing.append("confirmed_reference_annotation")
    if prompt_granularity == COARSE_GRAINED and not has_spatial_fidelity_ontology:
        fidelity_missing.append("spatial_fidelity_ontology")
    visual_missing = [] if render_evidence_count > 0 else ["standardized_renders"]
    active_weights = {
        fidelity_category: resolved["weights"][fidelity_category],
        "structural_validity": resolved["weights"]["structural_validity"],
        "visual_quality": resolved["weights"]["visual_quality"],
    }
    return {
        "profile_version": resolved["profile_version"],
        "profile_status": resolved["status"],
        "prompt_granularity": prompt_granularity,
        "evaluation_mode": evaluation_mode,
        "gate": {
            "source": "scene_request.prompt_granularity",
            "prompt_granularity": prompt_granularity,
            "evaluation_mode": evaluation_mode,
            "category_2": fidelity_category,
        },
        "weights": active_weights,
        "categories": {
            fidelity_category: {
                **deepcopy(fidelity),
                "weight": active_weights[fidelity_category],
                "required_evidence_available": not fidelity_missing,
                "missing_evidence": fidelity_missing,
            },
            "structural_validity": {
                **deepcopy(resolved["structural_validity"]),
                "weight": resolved["weights"]["structural_validity"],
                "required_evidence_available": True,
                "missing_evidence": [],
            },
            "visual_quality": {
                **deepcopy(resolved["visual_quality"]),
                "weight": resolved["weights"]["visual_quality"],
                "required_evidence_available": not visual_missing,
                "missing_evidence": visual_missing,
            },
        },
    }


def evaluation_mode_for_prompt_granularity(prompt_granularity: str) -> str:
    try:
        return EVALUATION_MODE_BY_GRANULARITY[prompt_granularity]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt granularity {prompt_granularity!r}") from exc


def weighted_benchmark_score(category_reports: dict[str, dict], weights: dict[str, float]) -> float | None:
    if set(category_reports) != set(weights):
        return None
    scores = {}
    for name in weights:
        # A frozen zero weight means the category is not part of this track.
        # It must not require artificial evidence merely to complete a score.
        if float(weights[name]) == 0.0:
            continue
        report = category_reports.get(name)
        score = report.get("score") if isinstance(report, dict) else None
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        scores[name] = float(score)
    return sum(scores[name] * float(weights[name]) for name in scores)


def _validate_policy(value: Any, path: str) -> None:
    if value not in VLM_POLICIES:
        raise ValueError(f"{path}.vlm_policy must be one of {sorted(VLM_POLICIES)}")


def _deep_merge(base: dict, patch: dict) -> dict:
    if not isinstance(patch, dict):
        raise ValueError("evaluation profile patch must be a JSON object")
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base
