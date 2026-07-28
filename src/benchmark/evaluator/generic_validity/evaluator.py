from __future__ import annotations

from copy import deepcopy

from benchmark.evaluator.generic_validity.accessibility import check_accessibility
from benchmark.evaluator.generic_validity.collision import check_collision
from benchmark.evaluator.generic_validity.navigability import check_navigability, compute_navigability_grid
from benchmark.evaluator.generic_validity.oob import check_oob
from benchmark.evaluator.generic_validity.support import check_support, disabled_support_report


DEFAULT_GENERIC_VALIDITY_CONFIG = {
    "collision": {
        "enabled": True,
        "official_mode": False,
        "detector_only": False,
        "obb_sat_eps": 1.0e-6,
        "mesh_enclosure_eps_m": 1.0e-4,
        "separation_threshold_m": 0.02,
        "score_mode": "invalid_pair_count_over_objects",
    },
    "oob": {
        "enabled": True,
        "official_mode": False,
        "detector_only": False,
        # numerical_eps is floating-point robustness for the wall and ceiling
        # planes only. floor_contact_tolerance_m is the separate semantic contact
        # tolerance for the floor plane so ordinary shallow floor sink stays valid
        # instead of being routed as a candidate OOB violation.
        "numerical_eps": 1.0e-6,
        "floor_contact_tolerance_m": 0.005,
        "score_mode": "invalid_object_count_over_objects",
    },
    "navigability": {
        "enabled": True,
        "grid_resolution": 0.08,
        "agent_radius": 0.25,
        "clearance_height": 1.70,
        "step_over_height": 0.15,
        "connectivity": 4,
    },
    "accessibility": {
        "enabled": True,
        "access_radius": 0.45,
        "require_largest_component": True,
    },
    "support": {
        "enabled": True,
        "official_mode": False,
        "detector_only": False,
        # support_p0b_v7 uses the scale-aware 2-3.5 cm contact tolerance. A
        # legacy contact_tolerance_m override may narrow the band, while the
        # deterministic certificate remains capped at 3.5 cm.
        "base_band_tolerance_m": 0.02,
        "minimum_contact_count": 1,
        "bottom_sample_grid": [4, 4],
        "max_representative_samples": 8,
        "mesh_bounds_tolerance_m": 0.03,
        "mesh_center_tolerance_m": 0.05,
    },
}

GENERIC_VALIDITY_NOTES = [
    "generic_validity_v0 uses canonical OBB proxies for navigability and accessibility.",
    "Collision uses collision_p0b_v2 with OBB SAT broad phase, guarded mesh/OBB frame contracts, and optional mesh narrow phase.",
    "OOB uses oob_p0b_v2 with exact OBB-vs-six-room-plane evidence and conservative VLM adjudication; the floor "
    "plane uses a separate semantic floor_contact_tolerance_m so ordinary shallow floor sink is not routed, and "
    "raw per-plane penetration is preserved in plane_penetration_m.",
    "Support uses support_p0b_v7: scale-aware 2-3.5 cm tolerance contact with a fixed-point path to the floor bypasses VLM; larger gaps, ungrounded stacks/cycles, attachment, and degraded geometry route to conservative VLM adjudication.",
    "Collision, OOB, and support are independent multi-label metrics; a Collision/OOB failure never implies a Support failure.",
    "Collision, OOB, and support never use physics simulation; VLM adjudicates ambiguous geometric events only.",
]


def evaluate_generic_validity(
    scene: dict,
    config: dict | None = None,
    *,
    collision_geometry: dict | None = None,
    prompt: str | None = None,
    relationships: list[dict] | dict | None = None,
    render_evidence: list[str] | None = None,
    vlm_judge: object | None = None,
    local_view_provider: object | None = None,
    support_enabled: bool | None = None,
    p0b_official_mode: bool | None = None,
    metric_applicability: dict[str, bool] | None = None,
) -> dict:
    resolved_config = _deep_merge(deepcopy(DEFAULT_GENERIC_VALIDITY_CONFIG), config or {})
    frozen_applicability = _resolve_metric_applicability(metric_applicability)
    for metric_name, applicable in frozen_applicability.items():
        resolved_config.setdefault(metric_name, {})["enabled"] = bool(applicable)
    # Explicit runtime overrides exist only for diagnostic/experiment callers.
    # The official submission API never exposes one and passes ``None``.
    if support_enabled is not None:
        resolved_config.setdefault("support", {})["enabled"] = bool(support_enabled)
    if p0b_official_mode is not None:
        for metric_name in ("collision", "oob", "support"):
            resolved_config.setdefault(metric_name, {})["official_mode"] = bool(p0b_official_mode)
    metrics: dict[str, dict] = {}
    navigability_cache = None

    if _enabled(resolved_config, "collision"):
        metrics["collision"] = check_collision(
            scene,
            resolved_config["collision"],
            collision_geometry=collision_geometry,
            prompt=prompt,
            relationships=relationships,
            render_evidence=render_evidence,
            vlm_judge=vlm_judge,
            local_view_provider=local_view_provider,
        )
    else:
        metrics["collision"] = _disabled_metric_report("collision")
    if _enabled(resolved_config, "oob"):
        metrics["oob"] = check_oob(
            scene,
            resolved_config["oob"],
            prompt=prompt,
            relationships=relationships,
            render_evidence=render_evidence,
            vlm_judge=vlm_judge,
            local_view_provider=local_view_provider,
        )
    else:
        metrics["oob"] = _disabled_metric_report("oob")
    if _enabled(resolved_config, "navigability") or _enabled(resolved_config, "accessibility"):
        navigability_cache = compute_navigability_grid(scene, resolved_config["navigability"])
    if _enabled(resolved_config, "navigability"):
        metrics["navigability"] = check_navigability(scene, resolved_config["navigability"], navigability_cache=navigability_cache)
    else:
        metrics["navigability"] = _disabled_metric_report("navigability")
    if _enabled(resolved_config, "accessibility"):
        metrics["accessibility"] = check_accessibility(scene, resolved_config["accessibility"], navigability_cache=navigability_cache)
    else:
        metrics["accessibility"] = _disabled_metric_report("accessibility")
    if _enabled(resolved_config, "support"):
        metrics["support"] = check_support(
            scene,
            resolved_config["support"],
            collision_geometry=collision_geometry,
            prompt=prompt,
            relationships=relationships,
            render_evidence=render_evidence,
            vlm_judge=vlm_judge,
            local_view_provider=local_view_provider,
        )
    else:
        metrics["support"] = disabled_support_report()

    active_metrics = {
        name: result
        for name, result in metrics.items()
        if result.get("status") in {"checked", "invalid_input"}
        and _is_number(result.get("score"))
    }
    unresolved_metrics = sorted(
        name
        for name, result in metrics.items()
        if result.get("status") not in {"checked", "invalid_input", "not_applicable"}
        or (
            result.get("status") in {"checked", "invalid_input"}
            and not _is_number(result.get("score"))
        )
    )
    excluded_metrics = sorted(
        name for name, result in metrics.items() if result.get("status") == "not_applicable"
    )
    active_metric_count = len(active_metrics)
    partial_score = (
        None
        if not active_metrics
        else sum(_numeric_score(result) for result in active_metrics.values()) / float(active_metric_count)
    )
    score = None if unresolved_metrics else partial_score
    metric_scores = {name: _optional_score(result) for name, result in metrics.items()}
    disabled_metrics = sorted(
        name
        for name, result in metrics.items()
        if isinstance(result, dict)
        and result.get("status") == "not_applicable"
        and result.get("reason") == "disabled_by_configuration"
    )
    return {
        "evaluator_version": "generic_validity_v0",
        "status": "incomplete" if unresolved_metrics else "ok" if active_metric_count else "no_checks_called",
        "score": None if score is None else float(score),
        "partial_score": None if partial_score is None else float(partial_score),
        "metrics": metrics,
        "metric_scores": metric_scores,
        "active_metric_count": active_metric_count,
        "unresolved_metrics": unresolved_metrics,
        "excluded_metrics": excluded_metrics,
        "disabled_metrics": disabled_metrics,
        "metric_applicability": frozen_applicability or None,
        "metric_applicability_source": "frozen_evaluation_profile" if frozen_applicability else "runtime_metric_defaults",
        "notes": list(GENERIC_VALIDITY_NOTES),
    }


def evaluate_scene_validity(scene: dict, config: dict | None = None) -> dict:
    return evaluate_generic_validity(scene, config=config)


def _enabled(config: dict, metric: str) -> bool:
    metric_config = config.get(metric)
    return isinstance(metric_config, dict) and bool(metric_config.get("enabled", True))


def _resolve_metric_applicability(value: dict[str, bool] | None) -> dict[str, bool]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metric_applicability must be a JSON object")
    allowed = {"collision", "oob", "navigability", "accessibility", "support"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"metric_applicability contains unknown metrics: {unknown}")
    resolved: dict[str, bool] = {}
    for metric_name, applicable in value.items():
        if not isinstance(applicable, bool):
            raise ValueError(f"metric_applicability.{metric_name} must be boolean")
        resolved[str(metric_name)] = applicable
    return resolved


def _disabled_metric_report(metric: str) -> dict:
    return {
        "metric": metric,
        "status": "not_applicable",
        "score": None,
        "enabled": False,
        "reason": "disabled_by_configuration",
    }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_score(result: dict) -> float:
    """Score for the averaged aggregate. Active metrics always carry a number;
    an unresolved metric (``score=None``) contributes 0.0 without ``float(None)``."""

    value = result.get("score")
    return float(value) if _is_number(value) else 0.0


def _optional_score(result: dict) -> float | None:
    """Per-metric score that preserves ``None`` instead of coercing it to 0.0."""

    value = result.get("score")
    if value is None:
        return None
    return float(value) if _is_number(value) else None


def _deep_merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base
