from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from benchmark.evaluator.spatial_fidelity.common import (
    adjudicate_candidate,
    compact_object,
    finalize_metric_report,
)
from benchmark.evaluator.spatial_fidelity.ontology import OntologyIndex, normalize_category_label


SCALE_EVALUATOR_VERSION = "spatial_scale_sceneonto_v1"
DEFAULT_SCALE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "abs_tolerance_m": 0.05,
    "hard_low_factor": 0.5,
    "hard_high_factor": 2.0,
    "horizontal_axis_policy": "best_permutation",
    "required_axes": ["width", "depth", "height"],
    "skip_categories": ["door", "window", "wall", "floor", "ceiling"],
}

_AXIS_INDEX = {"width": 0, "depth": 1, "height": 2}
_SEVERITY_RANK = {"typical": 0, "unusual": 1, "extreme": 2}


def evaluate_scale(
    scene: dict[str, Any],
    ontology: OntologyIndex,
    config: dict[str, Any],
    *,
    prompt: str = "",
    render_evidence: list[str] | None = None,
    vlm_judge: Any = None,
) -> dict[str, Any]:
    _validate_config(config)
    if not config["enabled"]:
        return _disabled_report()
    objects = scene.get("objects") if isinstance(scene.get("objects"), list) else []
    skip = {normalize_category_label(value) for value in config.get("skip_categories", [])}
    checks: list[dict[str, Any]] = []
    eligible_count = 0
    for raw_obj in objects:
        if not isinstance(raw_obj, dict):
            continue
        raw_category = str(raw_obj.get("category") or "")
        resolution = ontology.resolve(raw_category)
        if resolution.normalized_category in skip or normalize_category_label(
            resolution.ontology_category
        ) in skip:
            checks.append(
                {
                    "object_id": str(raw_obj.get("id") or ""),
                    "raw_category": raw_category,
                    "route": "not_applicable",
                    "status": "not_applicable",
                    "reason": "configured_scale_skip_category",
                    "score": None,
                }
            )
            continue
        eligible_count += 1
        checks.append(
            _evaluate_object(
                raw_obj,
                scene=scene,
                ontology=ontology,
                config=config,
                resolution=resolution,
                prompt=prompt,
                render_evidence=render_evidence,
                vlm_judge=vlm_judge,
            )
        )
    report = finalize_metric_report(
        metric="scale",
        evaluator_version=SCALE_EVALUATOR_VERSION,
        checks=checks,
        eligible_count=eligible_count,
        not_applicable_reason="no_scale_eligible_objects",
        notes=[
            (
                "Typical scale is certified from SceneOnto p5/p95 with the configured "
                f"{float(config['abs_tolerance_m']):g} m absolute tolerance."
            ),
            "Unusual and extreme statistical outliers are candidates, not verdicts; both require a metric-specific binary VLM judgement.",
            "Unknown categories or missing axes are coverage gaps and never passes.",
            "Width/depth use the least-violating permutation to avoid dataset axis-convention false positives.",
        ],
    )
    report["eligible_object_count"] = eligible_count
    report["configured_thresholds"] = {
        "abs_tolerance_m": float(config["abs_tolerance_m"]),
        "hard_low_factor": float(config["hard_low_factor"]),
        "hard_high_factor": float(config["hard_high_factor"]),
        "horizontal_axis_policy": str(config["horizontal_axis_policy"]),
    }
    return report


def _evaluate_object(
    obj: dict[str, Any],
    *,
    scene: dict[str, Any],
    ontology: OntologyIndex,
    config: dict[str, Any],
    resolution,
    prompt: str,
    render_evidence: list[str] | None,
    vlm_judge: Any,
) -> dict[str, Any]:
    object_id = str(obj.get("id") or "")
    base = {
        "object_id": object_id,
        "raw_category": str(obj.get("category") or ""),
        "category_resolution": resolution.as_dict(),
        "object": compact_object(obj),
        "score": None,
        "final_verdict": None,
    }
    if not resolution.known:
        return {
            **base,
            "route": "unknown",
            "status": "not_evaluable",
            "reason": "unknown_ontology_category",
            "axis_checks": [],
        }
    size = _valid_size(obj.get("size"), object_id=object_id)
    required_axes = [str(axis) for axis in config["required_axes"]]
    stats = {
        axis: ontology.dimension_stats(str(resolution.ontology_category), axis)
        for axis in required_axes
    }
    missing_axes = [axis for axis in required_axes if stats.get(axis) is None]
    if missing_axes:
        available_axis_checks = [
            _axis_check(
                canonical_axis=axis,
                ontology_axis=axis,
                actual=size[_AXIS_INDEX[axis]],
                stats=stats[axis],
                config=config,
            )
            for axis in required_axes
            if stats.get(axis) is not None
        ]
        return {
            **base,
            "route": "unknown",
            "status": "not_evaluable",
            "reason": "missing_ontology_dimension_axes",
            "missing_axes": missing_axes,
            "horizontal_assignment": "not_resolved_due_to_missing_axes",
            "axis_checks": available_axis_checks,
        }

    if str(config["horizontal_axis_policy"]) == "best_permutation":
        horizontal_checks, horizontal_assignment = _best_horizontal_checks(
            size,
            stats,
            config=config,
        )
    else:
        horizontal_assignment = "direct"
        horizontal_checks = [
            _axis_check(
                canonical_axis=axis,
                ontology_axis=axis,
                actual=size[_AXIS_INDEX[axis]],
                stats=stats[axis],
                config=config,
            )
            for axis in ("width", "depth")
        ]
    axis_checks = horizontal_checks + [
        _axis_check(
            canonical_axis="height",
            ontology_axis="height",
            actual=size[2],
            stats=stats["height"],
            config=config,
        )
    ]
    worst_rank = max(_SEVERITY_RANK[check["classification"]] for check in axis_checks)
    worst_classification = next(
        name for name, rank in _SEVERITY_RANK.items() if rank == worst_rank
    )
    evidence = {
        "object_id": object_id,
        "category": resolution.ontology_category,
        "raw_category": base["raw_category"],
        "size_m": list(size),
        "horizontal_assignment": horizontal_assignment,
        "axis_checks": deepcopy(axis_checks),
        "statistical_classification": worst_classification,
        "policy": (
            "SceneOnto p5/p95; unusual if outside the tolerance band; "
            "extreme if below p5*hard_low_factor or above p95*hard_high_factor"
        ),
    }
    if worst_classification == "typical":
        return {
            **base,
            **evidence,
            "route": "direct_valid",
            "candidate_route": None,
            "status": "checked",
            "reason": "all_scale_axes_within_typical_band",
            "final_verdict": "valid",
            "score": 1.0,
        }
    candidate = {
        "metric": "scale",
        "event": {
            "type": "scale_outlier",
            "object_ids": [object_id],
            "statistical_classification": worst_classification,
        },
        "rubric": (
            "Decide whether this object's scale is semantically appropriate for its category. "
            "The statistical outlier is evidence only; unusual designs and valid edge cases may be valid."
        ),
        **evidence,
    }
    if vlm_judge is None:
        return {
            **base,
            **evidence,
            "route": "requires_vlm",
            "candidate_route": "requires_vlm",
            "status": "requires_vlm",
            "reason": f"{worst_classification}_scale_requires_semantic_adjudication",
            "vlm_candidate": candidate,
        }
    try:
        judgement = adjudicate_candidate(
            metric="scale",
            candidate=candidate,
            scene=scene,
            prompt=prompt,
            render_evidence=render_evidence,
            judge=vlm_judge,
        )
    except Exception as exc:
        return {
            **base,
            **evidence,
            "route": "vlm_adjudication_failed",
            "candidate_route": "requires_vlm",
            "status": "vlm_adjudication_failed",
            "reason": "scale_vlm_adjudication_failed",
            "adjudication_error": str(exc),
            "vlm_candidate": candidate,
        }
    return {
        **base,
        **evidence,
        "route": "vlm_adjudicated",
        "candidate_route": "requires_vlm",
        "status": "checked",
        "reason": f"{worst_classification}_scale_vlm_adjudicated",
        "final_verdict": judgement["verdict"],
        "score": judgement["score"],
        "judge_result": judgement,
    }


def _best_horizontal_checks(
    size: tuple[float, float, float],
    stats: dict[str, dict[str, float]],
    *,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    direct = [
        _axis_check(
            canonical_axis="width",
            ontology_axis="width",
            actual=size[0],
            stats=stats["width"],
            config=config,
        ),
        _axis_check(
            canonical_axis="depth",
            ontology_axis="depth",
            actual=size[1],
            stats=stats["depth"],
            config=config,
        ),
    ]
    swapped = [
        _axis_check(
            canonical_axis="width",
            ontology_axis="depth",
            actual=size[0],
            stats=stats["depth"],
            config=config,
        ),
        _axis_check(
            canonical_axis="depth",
            ontology_axis="width",
            actual=size[1],
            stats=stats["width"],
            config=config,
        ),
    ]
    direct_key = _assignment_key(direct, prefer=0)
    swapped_key = _assignment_key(swapped, prefer=1)
    return (direct, "direct") if direct_key <= swapped_key else (swapped, "swapped")


def _assignment_key(checks: list[dict[str, Any]], *, prefer: int) -> tuple[float, ...]:
    ranks = [_SEVERITY_RANK[check["classification"]] for check in checks]
    return (
        float(max(ranks)),
        float(sum(ranks)),
        float(sum(check["normalized_distance_from_typical"] for check in checks)),
        float(prefer),
    )


def _axis_check(
    *,
    canonical_axis: str,
    ontology_axis: str,
    actual: float,
    stats: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    p5 = float(stats["p5"])
    p95 = float(stats["p95"])
    tolerance = float(config["abs_tolerance_m"])
    typical_low = max(0.0, p5 - tolerance)
    typical_high = p95 + tolerance
    extreme_low = p5 * float(config["hard_low_factor"])
    extreme_high = p95 * float(config["hard_high_factor"])
    if actual < extreme_low or actual > extreme_high:
        classification = "extreme"
    elif actual < typical_low or actual > typical_high:
        classification = "unusual"
    else:
        classification = "typical"
    if actual < typical_low:
        distance = (typical_low - actual) / max(p5, 1.0e-9)
    elif actual > typical_high:
        distance = (actual - typical_high) / max(p95, 1.0e-9)
    else:
        distance = 0.0
    return {
        "canonical_axis": canonical_axis,
        "ontology_axis": ontology_axis,
        "actual_m": float(actual),
        "expected_p5_m": p5,
        "expected_p95_m": p95,
        "median_m": stats.get("median"),
        "ontology_sample_count": stats.get("n_samples"),
        "typical_band_m": [typical_low, typical_high],
        "extreme_band_m": [extreme_low, extreme_high],
        "classification": classification,
        "normalized_distance_from_typical": float(distance),
    }


def _valid_size(value: Any, *, object_id: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"spatial scale object {object_id!r} requires size=[width,depth,height]")
    numbers: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"spatial scale object {object_id!r} size must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"spatial scale object {object_id!r} size must be positive and finite")
        numbers.append(number)
    return numbers[0], numbers[1], numbers[2]


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("enabled"), bool):
        raise ValueError("scale.enabled must be boolean")
    for key in ("abs_tolerance_m", "hard_low_factor", "hard_high_factor"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"scale.{key} must be a finite number")
    if float(config["abs_tolerance_m"]) < 0.0:
        raise ValueError("scale.abs_tolerance_m must be non-negative")
    if not 0.0 < float(config["hard_low_factor"]) <= 1.0:
        raise ValueError("scale.hard_low_factor must be in (0, 1]")
    if float(config["hard_high_factor"]) < 1.0:
        raise ValueError("scale.hard_high_factor must be at least 1")
    if config.get("horizontal_axis_policy") not in {"best_permutation", "direct"}:
        raise ValueError("scale.horizontal_axis_policy must be best_permutation or direct")
    required_axes = config.get("required_axes")
    if (
        not isinstance(required_axes, list)
        or len(required_axes) != 3
        or set(required_axes) != set(_AXIS_INDEX)
    ):
        raise ValueError("scale.required_axes must contain width, depth, and height exactly")
    if not isinstance(config.get("skip_categories"), list):
        raise ValueError("scale.skip_categories must be a list")


def _disabled_report() -> dict[str, Any]:
    return {
        "metric": "scale",
        "evaluator_version": SCALE_EVALUATOR_VERSION,
        "status": "not_applicable",
        "reason": "disabled_by_configuration",
        "score": None,
        "partial_score": None,
        "coverage": {
            "eligible_count": 0,
            "resolved_count": 0,
            "unknown_count": 0,
            "vlm_pending_count": 0,
            "fraction": None,
            "complete": False,
        },
        "routing": {
            "direct_valid": 0,
            "requires_vlm": 0,
            "vlm_adjudicated": 0,
            "vlm_adjudication_failed": 0,
            "unknown": 0,
        },
        "checks": [],
        "notes": [],
        "eligible_object_count": 0,
        "configured_thresholds": None,
    }
