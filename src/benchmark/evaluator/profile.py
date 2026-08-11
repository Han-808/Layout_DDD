from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from benchmark.evaluator.evidence_contract import canonical_hierarchy
from benchmark.nl_scene.converter import COARSE_GRAINED, FINE_GRAINED, PROMPT_GRANULARITIES


CANONICAL_PROFILE_VERSION = "canonical_scene_evaluation_v2"
PREVIOUS_CANONICAL_PROFILE_VERSION = "canonical_scene_evaluation_v1"

# Kept only so the checked-in Game profile can remain byte-for-byte unchanged.
# Non-game legacy profiles are rejected and never enter the active scene
# evaluator.
LEGACY_PROFILE_VERSION = "scene_evaluation_draft_v1"
CLAIM_DRIVEN_PROFILE_VERSION = CANONICAL_PROFILE_VERSION

FINE_GRAINED_MODE = "fine_grained_mode"
COARSE_GRAINED_MODE = "coarse_grained_mode"
EVALUATION_MODES = {FINE_GRAINED_MODE, COARSE_GRAINED_MODE}
EVALUATION_MODE_BY_GRANULARITY = {
    FINE_GRAINED: FINE_GRAINED_MODE,
    COARSE_GRAINED: COARSE_GRAINED_MODE,
}

L0 = "l0_structural_validity"
L1 = "l1_physical_plausibility"
L2 = "l2_specification_fidelity"
L3 = "l3_scene_quality"
L4 = "l4_downstream_task_functionality"
CANONICAL_LAYERS = (L0, L1, L2, L3, L4)
SCORING_LAYERS = (L1, L2, L3, L4)

L1_METRICS = ("collision", "oob", "support", "navigability", "accessibility")
L2_METRICS = ("oor", "oar", "functional_semantic_fidelity")
PREVIOUS_L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
)
L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)

PROTOCOL_OWNED_METRIC_KEYS = {"enabled", "official_mode", "detector_only"}


DEFAULT_EVALUATION_PROFILE: dict[str, Any] = {
    "profile_version": CANONICAL_PROFILE_VERSION,
    "status": "frozen",
    "layer_weights": {
        L1: 0.35,
        L2: 0.25,
        L3: 0.40,
        L4: 0.0,
    },
    L0: {
        "enabled": True,
        "scoring": False,
        "checks": [
            "schema",
            "normalization",
            "coordinate_and_unit_consistency",
            "required_input_coverage",
        ],
    },
    L1: {
        "enabled": True,
        "backend": "deterministic_evidence_plus_conditional_vlm",
        "metrics": {
            "collision": {"enabled": True, "weight": 1.0 / 3.0},
            "oob": {"enabled": True, "weight": 1.0 / 3.0},
            "support": {"enabled": True, "weight": 1.0 / 3.0},
            "navigability": {"enabled": False, "weight": 0.0},
            "accessibility": {"enabled": False, "weight": 0.0},
        },
        "never_vlm_metrics": ["navigability", "accessibility"],
        "metric_config": {},
    },
    L2: {
        "enabled": True,
        "activation": "specification_contract",
        "metrics": {
            "oor": {"enabled": True, "weight": 1.0 / 3.0},
            "oar": {"enabled": True, "weight": 1.0 / 3.0},
            "functional_semantic_fidelity": {
                "enabled": True,
                "weight": 1.0 / 3.0,
                "global_components": [
                    "room_scene_type",
                    "visual_functional_intent",
                    "required_functional_areas",
                ],
                "local_component": "local_functionality",
                "local_activation": "prompt_specified_only",
                "required_area_local_fallback": "global_suspicious_or_insufficient",
            },
        },
    },
    L3: {
        "enabled": True,
        "metrics": {
            "scale_consistency": {"enabled": True, "weight": 1.0 / 5.0},
            "object_pairing_consistency": {
                "enabled": True,
                "weight": 1.0 / 5.0,
                "requires": ["object_grouping_report"],
                "scope": "group_member_category_and_role_compatibility_only",
            },
            "style_consistency": {"enabled": True, "weight": 1.0 / 5.0},
            "functional_consistency": {
                "enabled": True,
                "weight": 1.0 / 5.0,
                "requires": ["object_grouping_report"],
                "scope": "ordinary_static_visual_usability",
            },
            "semantic_placement_consistency": {
                "enabled": True,
                "weight": 1.0 / 5.0,
                "requires": ["object_grouping_report"],
                "scope": "semantic_location_plausibility_only",
            },
        },
    },
    L4: {
        "enabled": False,
        "implemented": False,
        "metrics": {},
    },
}


def is_legacy_game_profile(value: Any) -> bool:
    """Recognize the one intentionally preserved legacy-shaped Game profile.

    The adapter is deliberately narrow: Category 2 is fully disabled, Collision
    is the only active physical metric, and the broad visual category is active.
    A normal scene profile using the old schema is rejected.
    """

    if not isinstance(value, dict) or value.get("profile_version") != LEGACY_PROFILE_VERSION:
        return False
    weights = value.get("weights")
    structural = value.get("structural_validity")
    applicability = structural.get("applicability") if isinstance(structural, dict) else None
    prompt_weight = weights.get("prompt_fidelity") if isinstance(weights, dict) else None
    spatial_weight = weights.get("spatial_fidelity") if isinstance(weights, dict) else None
    return bool(
        isinstance(weights, dict)
        and _is_number(prompt_weight)
        and float(prompt_weight) == 0.0
        and _is_number(spatial_weight)
        and float(spatial_weight) == 0.0
        and isinstance(applicability, dict)
        and applicability.get("collision") is True
        and all(
            applicability.get(name) is False
            for name in ("oob", "support", "navigability", "accessibility")
        )
    )


def resolve_evaluation_profile(value: dict[str, Any] | None = None) -> dict[str, Any]:
    if is_legacy_game_profile(value):
        return _validate_legacy_game_profile(deepcopy(value))
    if (
        isinstance(value, dict)
        and value.get("profile_version")
        == PREVIOUS_CANONICAL_PROFILE_VERSION
    ):
        return _validate_previous_canonical_profile(deepcopy(value))
    if isinstance(value, dict) and (
        value.get("profile_version") == LEGACY_PROFILE_VERSION
        or any(
            key in value
            for key in ("prompt_fidelity", "spatial_fidelity", "structural_validity", "visual_quality")
        )
    ):
        raise ValueError(
            "legacy non-game evaluation profiles were removed; use "
            f"{CANONICAL_PROFILE_VERSION!r}"
        )

    profile = _deep_merge(deepcopy(DEFAULT_EVALUATION_PROFILE), deepcopy(value or {}))
    if profile.get("profile_version") != CANONICAL_PROFILE_VERSION:
        raise ValueError(
            f"evaluation profile_version must be {CANONICAL_PROFILE_VERSION!r}"
        )
    if profile.get("status") != "frozen":
        raise ValueError("canonical evaluation profile status must be 'frozen'")

    layer_weights = profile.get("layer_weights")
    if not isinstance(layer_weights, dict) or set(layer_weights) != set(SCORING_LAYERS):
        raise ValueError(
            f"layer_weights must contain exactly {list(SCORING_LAYERS)}"
        )
    _validate_weight_mapping(layer_weights, "layer_weights", require_sum=1.0)

    if set(profile) != {
        "profile_version",
        "status",
        "layer_weights",
        *CANONICAL_LAYERS,
    }:
        unknown = sorted(
            set(profile)
            - {"profile_version", "status", "layer_weights", *CANONICAL_LAYERS}
        )
        missing = sorted(
            {"profile_version", "status", "layer_weights", *CANONICAL_LAYERS}
            - set(profile)
        )
        raise ValueError(
            f"canonical evaluation profile has unknown keys {unknown} or missing keys {missing}"
        )

    _validate_l0(profile[L0])
    _validate_metric_layer(profile[L1], L1, L1_METRICS)
    _validate_structural_metric_config(
        profile[L1].get("metric_config"),
        set(L1_METRICS),
        prefix=f"{L1}.metric_config",
    )
    _validate_metric_layer(profile[L2], L2, L2_METRICS)
    if profile[L2].get("activation") != "specification_contract":
        raise ValueError(f"{L2}.activation must be 'specification_contract'")
    _validate_metric_layer(profile[L3], L3, L3_METRICS)
    _validate_l4(profile[L4])
    return profile


def _validate_previous_canonical_profile(
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Read the frozen three-metric v1 profile without changing its meaning."""

    if profile.get("status") != "frozen":
        raise ValueError("canonical evaluation profile status must be 'frozen'")
    layer_weights = profile.get("layer_weights")
    if not isinstance(layer_weights, dict) or set(layer_weights) != set(
        SCORING_LAYERS
    ):
        raise ValueError(
            f"layer_weights must contain exactly {list(SCORING_LAYERS)}"
        )
    _validate_weight_mapping(
        layer_weights,
        "layer_weights",
        require_sum=1.0,
    )
    expected = {
        "profile_version",
        "status",
        "layer_weights",
        *CANONICAL_LAYERS,
    }
    if set(profile) != expected:
        raise ValueError(
            "previous canonical evaluation profile must retain its exact "
            "top-level v1 shape"
        )
    _validate_l0(profile[L0])
    _validate_metric_layer(profile[L1], L1, L1_METRICS)
    _validate_structural_metric_config(
        profile[L1].get("metric_config"),
        set(L1_METRICS),
        prefix=f"{L1}.metric_config",
    )
    _validate_metric_layer(profile[L2], L2, L2_METRICS)
    if profile[L2].get("activation") != "specification_contract":
        raise ValueError(
            f"{L2}.activation must be 'specification_contract'"
        )
    _validate_metric_layer(
        profile[L3],
        L3,
        PREVIOUS_L3_METRICS,
    )
    _validate_l4(profile[L4])
    return profile


def build_evaluation_plan(
    *,
    prompt_granularity: str,
    render_evidence_count: int = 0,
    profile: dict[str, Any] | None = None,
    active_l2_metrics: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the canonical plan or the explicitly selected frozen Game plan."""

    resolved = resolve_evaluation_profile(profile)
    if is_legacy_game_profile(resolved):
        return _build_legacy_game_plan(
            prompt_granularity=prompt_granularity,
            render_evidence_count=render_evidence_count,
            profile=resolved,
        )
    if prompt_granularity not in PROMPT_GRANULARITIES:
        raise ValueError(f"Unknown prompt granularity {prompt_granularity!r}")

    active_claim_metrics = set(active_l2_metrics or ())
    unknown_active = active_claim_metrics - set(L2_METRICS)
    if unknown_active:
        raise ValueError(f"active_l2_metrics contains unknown metrics {sorted(unknown_active)}")

    layers = deepcopy({name: resolved[name] for name in CANONICAL_LAYERS})
    for metric_name, metric in layers[L2]["metrics"].items():
        metric["applicable"] = bool(metric["enabled"] and metric_name in active_claim_metrics)
        metric["activation_reason"] = (
            "frozen_specification_claims"
            if metric["applicable"]
            else "no_frozen_claims"
        )
    for layer_name in SCORING_LAYERS:
        layers[layer_name]["weight"] = float(resolved["layer_weights"][layer_name])

    return {
        "profile_version": resolved["profile_version"],
        "profile_status": resolved["status"],
        "workflow": "canonical_l0_l4",
        "prompt_granularity": prompt_granularity,
        "prompt_granularity_role": "metadata_only",
        "activation_source": "canonical_profile_plus_specification_contract",
        "hierarchy": canonical_hierarchy(),
        "layer_weights": deepcopy(resolved["layer_weights"]),
        "layers": layers,
    }


def specification_activation_mode(profile_version: str | None) -> str:
    return (
        "prompt_granularity_gate"
        if profile_version == LEGACY_PROFILE_VERSION
        else "specification_contract"
    )


def evaluation_mode_for_prompt_granularity(prompt_granularity: str) -> str:
    try:
        return EVALUATION_MODE_BY_GRANULARITY[prompt_granularity]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt granularity {prompt_granularity!r}") from exc


def weighted_benchmark_score(
    category_reports: dict[str, dict],
    weights: dict[str, float],
) -> float | None:
    """Aggregate either the canonical layers or the preserved Game profile.

    Canonical not-applicable layers are explicitly excluded and the active
    denominator is returned separately by :func:`canonical_score_coverage`.
    Any applicable positive-weight layer without a score keeps the result
    unresolved.
    """

    if set(weights) == set(SCORING_LAYERS):
        applicable = {
            name: report
            for name, report in category_reports.items()
            if name in weights
            and float(weights[name]) > 0.0
            and isinstance(report, dict)
            and report.get("status") != "not_applicable"
        }
        if not applicable:
            return None
        if any(not _is_score(report.get("score")) for report in applicable.values()):
            return None
        denominator = sum(float(weights[name]) for name in applicable)
        if denominator <= 0.0:
            return None
        return sum(
            float(report["score"]) * float(weights[name])
            for name, report in applicable.items()
        ) / denominator

    # Preserved Game profile aggregation.
    if set(category_reports) != set(weights):
        return None
    scores: dict[str, float] = {}
    for name in weights:
        if float(weights[name]) == 0.0:
            continue
        score = category_reports[name].get("score")
        if not _is_score(score):
            return None
        scores[name] = float(score)
    return sum(scores[name] * float(weights[name]) for name in scores)


def canonical_score_coverage(
    category_reports: dict[str, dict],
    weights: dict[str, float],
    *,
    profile_version: str,
    scoring_profile_id: str | None = None,
    scoring_spec_version: str | None = None,
) -> dict[str, Any]:
    active_layers = [
        name
        for name in SCORING_LAYERS
        if float(weights.get(name, 0.0)) > 0.0
        and isinstance(category_reports.get(name), dict)
        and category_reports[name].get("status") != "not_applicable"
    ]
    required_weight = sum(float(weights[name]) for name in active_layers)
    covered_layers = [
        name
        for name in active_layers
        if _is_score(category_reports[name].get("score"))
    ]
    covered_weight = sum(float(weights[name]) for name in covered_layers)
    active_metrics_by_layer: dict[str, list[str]] = {}
    resolved_metrics_by_layer: dict[str, list[str]] = {}
    active_metric_signatures: dict[str, str] = {}
    for layer_name in SCORING_LAYERS:
        layer_report = category_reports.get(layer_name)
        if not isinstance(layer_report, dict):
            active_metrics: list[str] = []
            resolved_metrics: list[str] = []
        else:
            active_metrics = [
                str(name) for name in (layer_report.get("active_metrics") or [])
            ]
            resolved_metrics = [
                str(name) for name in (layer_report.get("resolved_metrics") or [])
            ]
        active_metrics_by_layer[layer_name] = active_metrics
        resolved_metrics_by_layer[layer_name] = resolved_metrics
        active_metric_signatures[layer_name] = (
            "+".join(active_metrics) if active_metrics else "none"
        )
    per_layer_active_metric_signature = "|".join(
        f"{name}:{active_metric_signatures[name]}" for name in SCORING_LAYERS
    )
    layer_weight_signature = "|".join(
        f"{name}:{float(weights.get(name, 0.0)):.12g}" for name in SCORING_LAYERS
    )
    scoring_signature = (
        f"|scoring_profile:{scoring_profile_id}"
        f"|scoring_spec:{scoring_spec_version}"
        if scoring_profile_id and scoring_spec_version
        else ""
    )
    return {
        "active_layers": active_layers,
        "covered_layers": covered_layers,
        "active_layer_signature": "+".join(active_layers) if active_layers else "none",
        "active_metrics_by_layer": active_metrics_by_layer,
        "resolved_metrics_by_layer": resolved_metrics_by_layer,
        "active_metric_signatures": active_metric_signatures,
        "per_layer_active_metric_signature": per_layer_active_metric_signature,
        "layer_weight_signature": layer_weight_signature,
        "comparability_signature": (
            f"{profile_version}|{layer_weight_signature}|"
            f"{per_layer_active_metric_signature}{scoring_signature}"
        ),
        "covered_weight": covered_weight,
        "required_weight": required_weight,
        "complete": bool(active_layers and math.isclose(covered_weight, required_weight, abs_tol=1e-9)),
        "aggregation_denominator": required_weight,
        "case_comparability": (
            "compare_only_with_same_profile_version_layer_weight_signature_"
            "and_per_layer_active_metric_signatures"
        ),
    }


def _validate_l0(config: Any) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{L0} must be a JSON object")
    if config.get("enabled") is not True or config.get("scoring") is not False:
        raise ValueError(f"{L0} must remain enabled and non-scoring")
    checks = config.get("checks")
    if not isinstance(checks, list) or not checks or any(not isinstance(v, str) for v in checks):
        raise ValueError(f"{L0}.checks must be a non-empty string list")


def _validate_l4(config: Any) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{L4} must be a JSON object")
    if config.get("enabled") is not False or config.get("implemented") is not False:
        raise ValueError(f"{L4} is TBD and must remain disabled/unimplemented")
    if config.get("metrics") != {}:
        raise ValueError(f"{L4}.metrics must remain empty")


def _validate_metric_layer(
    config: Any,
    layer_name: str,
    expected_metrics: tuple[str, ...],
) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{layer_name} must be a JSON object")
    if not isinstance(config.get("enabled"), bool):
        raise ValueError(f"{layer_name}.enabled must be boolean")
    metrics = config.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(expected_metrics):
        raise ValueError(
            f"{layer_name}.metrics must contain exactly {list(expected_metrics)}"
        )
    weights: dict[str, float] = {}
    for name in expected_metrics:
        metric = metrics[name]
        if not isinstance(metric, dict) or not isinstance(metric.get("enabled"), bool):
            raise ValueError(f"{layer_name}.metrics.{name}.enabled must be boolean")
        raw_weight = metric.get("weight")
        if not _is_number(raw_weight):
            raise ValueError(f"{layer_name}.metrics.{name}.weight must be numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"{layer_name}.metrics.{name}.weight must be finite and non-negative"
            )
        if metric["enabled"] is False and weight != 0.0:
            raise ValueError(
                f"{layer_name}.metrics.{name} is disabled and must have zero weight"
            )
        weights[name] = weight
    if config.get("enabled") and not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"{layer_name} enabled metric weights must sum to 1.0")


def _validate_structural_metric_config(
    value: Any,
    known_metrics: set[str],
    *,
    prefix: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} must be a JSON object")
    unknown = sorted(set(value) - known_metrics)
    if unknown:
        raise ValueError(f"{prefix} contains unknown metrics: {unknown}")
    for metric_name, overrides in value.items():
        path = f"{prefix}.{metric_name}"
        if not isinstance(overrides, dict):
            raise ValueError(f"{path} must be a JSON object")
        reserved = sorted(set(overrides) & PROTOCOL_OWNED_METRIC_KEYS)
        if reserved:
            raise ValueError(
                f"{path} must not override protocol-owned keys {reserved}; "
                f"use {L1}.metrics.{metric_name}.enabled instead"
            )
        for key, override in overrides.items():
            if _is_number(override) and not math.isfinite(float(override)):
                raise ValueError(f"{path}.{key} must be finite")


def _validate_legacy_game_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not is_legacy_game_profile(profile):
        raise ValueError("only the checked-in legacy-shaped Game profile is supported")
    weights = profile.get("weights")
    if not isinstance(weights, dict) or set(weights) != {
        "prompt_fidelity",
        "spatial_fidelity",
        "structural_validity",
        "visual_quality",
    }:
        raise ValueError("Game profile weights are malformed")
    for name, value in weights.items():
        if not _is_number(value) or float(value) < 0.0:
            raise ValueError(f"Game profile weight {name} must be non-negative")
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Game profile weights must sum to 1.0")
    return profile


def _build_legacy_game_plan(
    *,
    prompt_granularity: str,
    render_evidence_count: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    if prompt_granularity not in PROMPT_GRANULARITIES:
        raise ValueError(f"Unknown prompt granularity {prompt_granularity!r}")
    category_2 = (
        "prompt_fidelity" if prompt_granularity == FINE_GRAINED else "spatial_fidelity"
    )
    active_weights = {
        category_2: float(profile["weights"][category_2]),
        "structural_validity": float(profile["weights"]["structural_validity"]),
        "visual_quality": float(profile["weights"]["visual_quality"]),
    }
    category_2_config = (
        profile["prompt_fidelity"][FINE_GRAINED]
        if category_2 == "prompt_fidelity"
        else profile["spatial_fidelity"]
    )
    return {
        "profile_version": profile["profile_version"],
        "profile_status": profile["status"],
        "prompt_granularity": prompt_granularity,
        "evaluation_mode": evaluation_mode_for_prompt_granularity(prompt_granularity),
        "gate": {
            "source": "scene_request.prompt_granularity",
            "prompt_granularity": prompt_granularity,
            "evaluation_mode": evaluation_mode_for_prompt_granularity(prompt_granularity),
            "category_2": category_2,
        },
        "weights": active_weights,
        "categories": {
            category_2: {
                **deepcopy(category_2_config),
                "weight": active_weights[category_2],
                "required_evidence_available": True,
                "missing_evidence": [],
            },
            "structural_validity": {
                **deepcopy(profile["structural_validity"]),
                "weight": active_weights["structural_validity"],
                "required_evidence_available": True,
                "missing_evidence": [],
            },
            "visual_quality": {
                **deepcopy(profile["visual_quality"]),
                "weight": active_weights["visual_quality"],
                "required_evidence_available": render_evidence_count > 0,
                "missing_evidence": (
                    [] if render_evidence_count > 0 else ["standardized_renders"]
                ),
            },
        },
    }


def _validate_weight_mapping(
    values: dict[str, Any],
    path: str,
    *,
    require_sum: float | None,
) -> None:
    for name, raw in values.items():
        if not _is_number(raw):
            raise ValueError(f"{path}.{name} must be numeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{path}.{name} must be finite and non-negative")
        values[name] = value
    if require_sum is not None and not math.isclose(
        sum(float(v) for v in values.values()),
        require_sum,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{path} must sum to {require_sum}")


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _is_score(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("evaluation profile patch must be a JSON object")
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base
