from __future__ import annotations

from copy import deepcopy

from benchmark.evaluator.generic_validity.accessibility import check_accessibility
from benchmark.evaluator.generic_validity.collision import check_collision
from benchmark.evaluator.generic_validity.navigability import check_navigability, compute_navigability_grid
from benchmark.evaluator.generic_validity.oob import check_oob
from benchmark.evaluator.generic_validity.support import check_support


DEFAULT_GENERIC_VALIDITY_CONFIG = {
    "collision": {
        "enabled": True,
        "z_overlap_eps": 0.03,
        "xy_overlap_area_eps": 0.005,
        "ignore_supported_or_contained_pairs": True,
        "score_mode": "collision_count_over_objects",
    },
    "oob": {
        "enabled": True,
        "inside_ratio_threshold": 0.98,
        "floor_eps": 0.05,
        "check_height": True,
        "height_eps": 0.05,
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
        "floor_eps": 0.05,
        "support_gap": 0.06,
        "sink_tolerance": 0.05,
        "bottom_sample_grid": [3, 3],
        "support_ratio_threshold": 0.30,
        "allow_wall_support_proxy": True,
        "wall_support_distance": 0.08,
    },
}

GENERIC_VALIDITY_NOTES = [
    "generic_validity_v0 uses bbox/OBB proxy geometry only.",
    "No mesh, physics, VLM, or semantic reasoning is used.",
]


def evaluate_generic_validity(scene: dict, config: dict | None = None) -> dict:
    resolved_config = _deep_merge(deepcopy(DEFAULT_GENERIC_VALIDITY_CONFIG), config or {})
    metrics: dict[str, dict] = {}
    navigability_cache = None

    if _enabled(resolved_config, "collision"):
        metrics["collision"] = check_collision(scene, resolved_config["collision"])
    if _enabled(resolved_config, "oob"):
        metrics["oob"] = check_oob(scene, resolved_config["oob"])
    if _enabled(resolved_config, "navigability") or _enabled(resolved_config, "accessibility"):
        navigability_cache = compute_navigability_grid(scene, resolved_config["navigability"])
    if _enabled(resolved_config, "navigability"):
        metrics["navigability"] = check_navigability(scene, resolved_config["navigability"], navigability_cache=navigability_cache)
    if _enabled(resolved_config, "accessibility"):
        metrics["accessibility"] = check_accessibility(scene, resolved_config["accessibility"], navigability_cache=navigability_cache)
    if _enabled(resolved_config, "support"):
        metrics["support"] = check_support(scene, resolved_config["support"])

    active_metrics = {name: result for name, result in metrics.items() if result.get("status") in {"checked", "invalid_input"}}
    active_metric_count = len(active_metrics)
    overall_score = 0.0 if not active_metrics else sum(float(result.get("score", 0.0)) for result in active_metrics.values()) / float(active_metric_count)
    metric_scores = {name: float(result.get("score", 0.0)) for name, result in metrics.items()}
    return {
        "evaluator_version": "generic_validity_v0",
        "status": "ok" if active_metric_count else "no_checks_called",
        "overall_score": float(overall_score),
        "metrics": metrics,
        "metric_scores": metric_scores,
        "active_metric_count": active_metric_count,
        "notes": list(GENERIC_VALIDITY_NOTES),
    }


def evaluate_scene_validity(scene: dict, config: dict | None = None) -> dict:
    return evaluate_generic_validity(scene, config=config)


def _enabled(config: dict, metric: str) -> bool:
    metric_config = config.get(metric)
    return isinstance(metric_config, dict) and bool(metric_config.get("enabled", True))


def _deep_merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base
