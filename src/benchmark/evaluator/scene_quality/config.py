"""Configuration resolution and validation for L3 Scene Quality.

The evaluator runtime deliberately imports this module as a boundary: metric
definitions and built-in defaults live in :mod:`definitions`, while historical
profile exceptions remain isolated in :mod:`compat`.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from benchmark.evaluator.evidence_contract import validate_evidence_plan
from benchmark.evaluator.scene_quality.compat import (
    historical_scene_quality_profile_override,
)
from benchmark.evaluator.scene_quality.definitions import (
    CAMERA_MODES,
    CAMERA_SCOPES,
    DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG,
    EVIDENCE_SELECTORS,
    IMAGE_ORDER_TOKENS,
    PRESENTATIONS,
    SCENE_QUALITY_INTERFACE_NAMESPACE,
    SUPPORTED_SCENE_QUALITY_METRICS,
    _RETIRED_CONFIG_NAMESPACES,
    _RETIRED_METRIC_NAMES,
)
from benchmark.evaluator.scene_quality.functional_prejudgement import (
    validate_functional_prejudgement_evidence_config,
)
from benchmark.evaluator.scene_quality.placement_severity import (
    ATYPICAL,
    IMPLAUSIBLE,
    PLACEMENT_SEVERITY_LEVELS,
)
from benchmark.rendering.camera_pose import CAMERA_POSE_MODES


class SceneQualityInterfaceConfigError(ValueError):
    """Raised when a Scene Quality interface configuration is malformed."""


def resolve_scene_quality_config(
    config: dict[str, Any] | None = None,
    *,
    profile: dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the layered Scene Quality interface configuration.

    Precedence, lowest to highest: built-in defaults, historical downstream
    profile exception, canonical profile override, explicit canonical config,
    and per-run override. Retired namespace and metric aliases are rejected
    rather than silently normalized. Unknown future-compatible fields are
    preserved while every known field is validated.
    """

    layers: list[dict[str, Any]] = []
    if isinstance(profile, dict):
        historical_override = historical_scene_quality_profile_override(profile)
        if historical_override:
            layers.append(historical_override)
        retired_namespaces = sorted(set(profile) & set(_RETIRED_CONFIG_NAMESPACES))
        if retired_namespaces:
            raise SceneQualityInterfaceConfigError(
                "retired L3 profile namespaces are not accepted by the canonical "
                f"resolver: {retired_namespaces}; use "
                f"{SCENE_QUALITY_INTERFACE_NAMESPACE!r}"
            )
        profile_section = profile.get(SCENE_QUALITY_INTERFACE_NAMESPACE)
        if profile_section is not None:
            layers.append(_normalize_layer(profile_section, "profile override"))
    if config is not None:
        layers.append(_normalize_layer(config, "config override"))
    if run_overrides is not None:
        layers.append(_normalize_layer(run_overrides, "run override"))

    resolved = deepcopy(DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG)
    for layer in layers:
        resolved = _deep_merge(resolved, layer)

    # A global evidence-policy override is lower precedence than an explicit
    # per-metric policy, but higher precedence than the built-in metric policy.
    global_defaults: dict[str, Any] = {}
    explicit_metric_policies: dict[str, dict[str, Any]] = {
        name: {} for name in SUPPORTED_SCENE_QUALITY_METRICS
    }
    for layer in layers:
        layer_defaults = layer.get("evidence_policy_defaults")
        if layer_defaults is not None:
            global_defaults = _deep_merge(
                global_defaults,
                _as_object(layer_defaults, "evidence_policy_defaults"),
            )
        layer_metrics = layer.get("metrics")
        if layer_metrics is not None:
            layer_metrics = _as_object(layer_metrics, "metrics")
            for name in SUPPORTED_SCENE_QUALITY_METRICS:
                metric_patch = layer_metrics.get(name)
                if (
                    isinstance(metric_patch, dict)
                    and metric_patch.get("evidence_policy") is not None
                ):
                    explicit_metric_policies[name] = _deep_merge(
                        explicit_metric_policies[name],
                        _as_object(
                            metric_patch["evidence_policy"],
                            f"metrics.{name}.evidence_policy",
                        ),
                    )

    _validate_top_level(resolved)
    try:
        resolved["functional_prejudgement_evidence"] = (
            validate_functional_prejudgement_evidence_config(
                resolved.get("functional_prejudgement_evidence")
            )
        )
    except (TypeError, ValueError) as exc:
        raise SceneQualityInterfaceConfigError(str(exc)) from exc
    if not isinstance(resolved.get("evidence_policy_defaults"), dict):
        raise SceneQualityInterfaceConfigError(
            "l3_scene_quality.evidence_policy_defaults must be a JSON object"
        )
    metrics = resolved.get("metrics")
    if not isinstance(metrics, dict):
        raise SceneQualityInterfaceConfigError(
            "l3_scene_quality.metrics must be a JSON object"
        )
    for metric_name in SUPPORTED_SCENE_QUALITY_METRICS:
        metric_config = metrics.get(metric_name)
        if not isinstance(metric_config, dict):
            raise SceneQualityInterfaceConfigError(
                f"l3_scene_quality.metrics.{metric_name} must be a JSON object"
            )
        _validate_metric_flags(metric_name, metric_config)
        built_in = DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG["metrics"][metric_name][
            "evidence_policy"
        ]
        policy = _deep_merge(deepcopy(built_in), global_defaults)
        policy = _deep_merge(policy, explicit_metric_policies[metric_name])
        metric_config["evidence_policy"] = _validate_evidence_policy(
            metric_name, policy
        )
        plan = metric_config.get("evidence_plan")
        if plan is not None:
            try:
                validate_evidence_plan(
                    plan,
                    where=f"l3_scene_quality.metrics.{metric_name}.evidence_plan",
                )
            except Exception as exc:
                raise SceneQualityInterfaceConfigError(str(exc)) from exc
    return resolved


def _validate_top_level(config: dict[str, Any]) -> None:
    for flag in ("enabled", "implemented"):
        if not isinstance(config.get(flag), bool):
            raise SceneQualityInterfaceConfigError(
                f"l3_scene_quality.{flag} must be boolean"
            )
    if config.get("implemented") is not True:
        raise SceneQualityInterfaceConfigError(
            "canonical l3_scene_quality implemented must remain true"
        )
    version = config.get("version")
    if not isinstance(version, str) or not version.strip():
        raise SceneQualityInterfaceConfigError(
            "l3_scene_quality.version must be a non-empty string"
        )


def _validate_metric_flags(
    metric_name: str,
    metric_config: dict[str, Any],
) -> None:
    for flag in ("enabled", "implemented"):
        if not isinstance(metric_config.get(flag), bool):
            raise SceneQualityInterfaceConfigError(
                f"l3_scene_quality.metrics.{metric_name}.{flag} must be boolean"
            )
    if metric_config.get("implemented") is not True:
        raise SceneQualityInterfaceConfigError(
            f"canonical l3_scene_quality metric {metric_name} "
            "implemented must remain true"
        )
    weight = metric_config.get("weight", 1.0)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) < 0.0
    ):
        raise SceneQualityInterfaceConfigError(
            f"l3_scene_quality.metrics.{metric_name}.weight must be non-negative"
        )
    if metric_name == "semantic_placement_consistency":
        _validate_placement_severity_policy(metric_config)
        _validate_residual_global_placement_policy(metric_config)
    if metric_name == "functional_consistency":
        granularity = metric_config.get(
            "group_local_check_granularity"
        )
        if granularity not in {"batched", "per_check"}:
            raise SceneQualityInterfaceConfigError(
                "functional_consistency.group_local_check_granularity "
                "must be exactly 'batched' or 'per_check'"
            )
        evidence_policy = metric_config.get(
            "group_local_evidence_policy"
        )
        if evidence_policy not in {
            "isolated_episode",
            "shared_group_bank",
        }:
            raise SceneQualityInterfaceConfigError(
                "functional_consistency.group_local_evidence_policy must "
                "be exactly 'isolated_episode' or 'shared_group_bank'"
            )
        if (
            evidence_policy == "shared_group_bank"
            and granularity != "per_check"
        ):
            raise SceneQualityInterfaceConfigError(
                "functional_consistency shared_group_bank requires "
                "group_local_check_granularity='per_check'"
            )
        window_max = metric_config.get(
            "group_local_active_window_max_images"
        )
        if (
            isinstance(window_max, bool)
            or not isinstance(window_max, int)
            or window_max < 2
        ):
            raise SceneQualityInterfaceConfigError(
                "functional_consistency."
                "group_local_active_window_max_images must be an integer "
                ">= 2"
            )


def _validate_placement_severity_policy(
    metric_config: dict[str, Any],
) -> None:
    policy = metric_config.get("severity_policy")
    if not isinstance(policy, dict):
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.severity_policy must be a JSON object"
        )
    if policy.get("schema_version") != "object_equivalent_burden_v1":
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.severity_policy.schema_version "
            "must remain object_equivalent_burden_v1"
        )
    if policy.get("levels") != list(PLACEMENT_SEVERITY_LEVELS):
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.severity_policy.levels must remain "
            f"{list(PLACEMENT_SEVERITY_LEVELS)}"
        )
    if policy.get("strict_level") != IMPLAUSIBLE:
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.severity_policy.strict_level "
            f"must remain {IMPLAUSIBLE}"
        )
    if policy.get("extended_level") != ATYPICAL:
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.severity_policy.extended_level "
            f"must remain {ATYPICAL}"
        )
    if policy.get("affects_existing_metric_score") is not True:
        raise SceneQualityInterfaceConfigError(
            "semantic-placement severity must control post-hoc burden scoring"
        )


def _validate_residual_global_placement_policy(
    metric_config: dict[str, Any],
) -> None:
    policy = metric_config.get("residual_global_review")
    if not isinstance(policy, dict):
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.residual_global_review must be "
            "a JSON object"
        )
    if not isinstance(policy.get("enabled"), bool):
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.residual_global_review.enabled "
            "must be boolean"
        )
    weight = policy.get("placement_weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or not 0.0 < float(weight) < 1.0
    ):
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.residual_global_review."
            "placement_weight must be a finite number strictly between 0 "
            "and 1"
        )
    image_budget = policy.get("image_budget")
    if (
        isinstance(image_budget, bool)
        or not isinstance(image_budget, int)
        or image_budget < 1
    ):
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.residual_global_review."
            "image_budget must be a positive integer"
        )
    if policy.get("allowed_check_types") != [
        "scene_zone",
        "contextual_anchor",
    ]:
        raise SceneQualityInterfaceConfigError(
            "semantic_placement_consistency.residual_global_review."
            "allowed_check_types must remain ['scene_zone', "
            "'contextual_anchor']"
        )


def _validate_evidence_policy(
    metric_name: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise SceneQualityInterfaceConfigError(
            f"l3_scene_quality.metrics.{metric_name}.evidence_policy "
            "must be a JSON object"
        )
    scope = policy.get("camera_scope")
    if scope not in CAMERA_SCOPES:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.camera_scope must be one of "
            f"{list(CAMERA_SCOPES)}, got {scope!r}"
        )
    mode = policy.get("camera_mode")
    if mode not in CAMERA_MODES:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.camera_mode must be one of "
            f"{list(CAMERA_MODES)}, got {mode!r}"
        )
    selector = policy.get("selector")
    if selector not in EVIDENCE_SELECTORS:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.selector must be one of "
            f"{list(EVIDENCE_SELECTORS)}, got {selector!r}"
        )
    presentation = policy.get("presentation")
    if presentation not in PRESENTATIONS:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.presentation must be one of "
            f"{list(PRESENTATIONS)}, got {presentation!r}"
        )
    budget = policy.get("image_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.image_budget must be a positive "
            f"integer, got {budget!r}"
        )
    for quota_name in ("global_image_budget", "scoped_image_budget"):
        quota = policy.get(quota_name)
        if quota is None:
            continue
        if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
            raise SceneQualityInterfaceConfigError(
                f"{metric_name}.evidence_policy.{quota_name} "
                "must be a non-negative integer"
            )
        if quota > budget:
            raise SceneQualityInterfaceConfigError(
                f"{metric_name}.evidence_policy.{quota_name} "
                "cannot exceed image_budget"
            )
    if not isinstance(policy.get("include_global_context"), bool):
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.include_global_context must be boolean"
        )
    image_order = policy.get("image_order")
    if image_order is not None:
        if not isinstance(image_order, list) or not image_order:
            raise SceneQualityInterfaceConfigError(
                f"{metric_name}.evidence_policy.image_order must be null or "
                "a non-empty list"
            )
        for token in image_order:
            if token not in IMAGE_ORDER_TOKENS:
                raise SceneQualityInterfaceConfigError(
                    f"{metric_name}.evidence_policy.image_order token "
                    f"{token!r} must be one of {list(IMAGE_ORDER_TOKENS)}"
                )
    camera_pose_mode = policy.get("camera_pose_mode")
    if (
        camera_pose_mode is not None
        and camera_pose_mode not in CAMERA_POSE_MODES
    ):
        raise SceneQualityInterfaceConfigError(
            f"{metric_name}.evidence_policy.camera_pose_mode must be null or "
            f"one of {list(CAMERA_POSE_MODES)}, got {camera_pose_mode!r}"
        )
    return policy


def _normalize_layer(layer: Any, label: str) -> dict[str, Any]:
    """Copy one canonical config layer and reject retired surface names."""

    obj = _as_object(layer, label)
    retired_namespaces = sorted(set(obj) & set(_RETIRED_CONFIG_NAMESPACES))
    if retired_namespaces:
        raise SceneQualityInterfaceConfigError(
            "retired L3 config namespaces are not accepted: "
            f"{retired_namespaces}; pass the canonical config directly"
        )
    result = deepcopy(obj)
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        retired_metrics = sorted(set(metrics) & set(_RETIRED_METRIC_NAMES))
        if retired_metrics:
            raise SceneQualityInterfaceConfigError(
                "retired L3 metric names are not accepted: "
                f"{retired_metrics}; use 'object_pairing_consistency'"
            )
    return result


def _as_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SceneQualityInterfaceConfigError(
            f"l3_scene_quality {label} must be a JSON object"
        )
    return value


def _deep_merge(
    base: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise SceneQualityInterfaceConfigError(
            "l3_scene_quality config patch must be a JSON object"
        )
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


__all__ = [
    "SceneQualityInterfaceConfigError",
    "resolve_scene_quality_config",
]
