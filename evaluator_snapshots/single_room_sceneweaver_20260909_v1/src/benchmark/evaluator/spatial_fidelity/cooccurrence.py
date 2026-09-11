from __future__ import annotations

import math
from copy import deepcopy
from itertools import combinations
from typing import Any

from benchmark.evaluator.spatial_fidelity.common import adjudicate_candidate, finalize_metric_report
from benchmark.evaluator.spatial_fidelity.ontology import (
    CooccurrenceRecord,
    OntologyIndex,
    normalize_category_label,
)


COOCCURRENCE_EVALUATOR_VERSION = "spatial_cooccurrence_sceneonto_v1"
DEFAULT_COOCCURRENCE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "rare_threshold": 0.01,
    "absent_threshold": 0.001,
    "strong_threshold": 0.20,
    "functional_hint_threshold": 0.70,
    # SceneOnto's per-category count is an object-observation count, not a
    # distinct-scene count. It is used only as a conservative support floor;
    # p_b_given_a from the pair record remains authoritative.
    "min_anchor_observation_count_for_rarity": 100,
    "pair_unit": "unique_unordered_category_pair",
    "prefer_room_conditioned": True,
    # SceneOnto retains only top-K neighbours. An absent entry therefore does
    # not establish zero co-occurrence and must remain an explicit coverage gap.
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
}


def evaluate_cooccurrence(
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
    buckets, mapping_records = _category_buckets(scene, ontology, config)
    category_keys = sorted(buckets)
    eligible_count = len(category_keys) * (len(category_keys) - 1) // 2
    checks = [
        _evaluate_category_pair(
            buckets[left],
            buckets[right],
            scene=scene,
            ontology=ontology,
            config=config,
            prompt=prompt,
            render_evidence=render_evidence,
            vlm_judge=vlm_judge,
        )
        for left, right in combinations(category_keys, 2)
    ]
    report = finalize_metric_report(
        metric="cooccurrence_plausibility",
        evaluator_version=COOCCURRENCE_EVALUATOR_VERSION,
        checks=checks,
        eligible_count=eligible_count,
        not_applicable_reason="fewer_than_two_distinct_semantic_categories",
        notes=[
            "Each unordered distinct category pair is checked once; repeated object instances do not multiply its weight.",
            "max(P(B|A), P(A|B)) preserves the SceneCritic compatibility rule.",
            "A recorded probability at or above the rare threshold certifies plausibility; a sufficiently supported rare pair routes to VLM and is never directly invalidated by frequency alone.",
            "Missing sparse ontology entries and insufficient two-direction support are coverage gaps, never zero-probability evidence.",
            "Proximity and functional placement are deliberately excluded and remain deferred to Functional Grouping.",
        ],
    )
    report.update(
        {
            "pair_unit": str(config["pair_unit"]),
            "distinct_semantic_category_count": len(category_keys),
            "distinct_semantic_categories": category_keys,
            "eligible_category_pair_count": eligible_count,
            "category_mapping": mapping_records,
            "configured_thresholds": {
                "absent": float(config["absent_threshold"]),
                "rare": float(config["rare_threshold"]),
                "strong": float(config["strong_threshold"]),
                "functional_hint": float(config["functional_hint_threshold"]),
                "min_anchor_observation_count_for_rarity": int(
                    config["min_anchor_observation_count_for_rarity"]
                ),
            },
        }
    )
    return report


def _category_buckets(
    scene: dict[str, Any],
    ontology: OntologyIndex,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    objects = scene.get("objects") if isinstance(scene.get("objects"), list) else []
    skip = {normalize_category_label(value) for value in config.get("skip_categories", [])}
    buckets: dict[str, dict[str, Any]] = {}
    mapping_records: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        resolution = ontology.resolve(obj.get("category"))
        resolved_normalized = normalize_category_label(resolution.ontology_category)
        skipped = resolution.normalized_category in skip or resolved_normalized in skip
        mapping_records.append(
            {
                "object_id": str(obj.get("id") or ""),
                **resolution.as_dict(),
                "excluded_from_cooccurrence": skipped,
                "exclusion_reason": "configured_skip_category" if skipped else None,
            }
        )
        if skipped:
            continue
        semantic_key = (
            str(resolution.ontology_category)
            if resolution.known
            else resolution.normalized_category or f"unknown_{str(obj.get('id') or '')}"
        )
        bucket = buckets.setdefault(
            semantic_key,
            {
                "semantic_category": semantic_key,
                "ontology_category": resolution.ontology_category,
                "known": resolution.known,
                "raw_categories": [],
                "object_ids": [],
            },
        )
        raw_category = str(obj.get("category") or "")
        if raw_category not in bucket["raw_categories"]:
            bucket["raw_categories"].append(raw_category)
        bucket["object_ids"].append(str(obj.get("id") or ""))
    return buckets, mapping_records


def _evaluate_category_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    scene: dict[str, Any],
    ontology: OntologyIndex,
    config: dict[str, Any],
    prompt: str,
    render_evidence: list[str] | None,
    vlm_judge: Any,
) -> dict[str, Any]:
    category_a = str(left["semantic_category"])
    category_b = str(right["semantic_category"])
    object_ids = list(left["object_ids"]) + list(right["object_ids"])
    base = {
        "category_a": category_a,
        "category_b": category_b,
        "object_ids_a": list(left["object_ids"]),
        "object_ids_b": list(right["object_ids"]),
        "raw_categories_a": list(left["raw_categories"]),
        "raw_categories_b": list(right["raw_categories"]),
        "score": None,
        "final_verdict": None,
    }
    if not left["known"] or not right["known"]:
        return {
            **base,
            "route": "unknown",
            "status": "not_evaluable",
            "reason": "unknown_ontology_category_in_pair",
            "directional_evidence": {},
        }

    selected = _select_directional_evidence(
        ontology,
        category_a,
        category_b,
        room_type=str(scene.get("scene_type") or ""),
        config=config,
    )
    if selected is None:
        return {
            **base,
            "route": "unknown",
            "status": "not_evaluable",
            "reason": "missing_sparse_cooccurrence_entry",
            "directional_evidence": {},
            "ontology_storage_semantics": ontology.storage_semantics,
        }
    record_a_b = selected["a_to_b"]
    record_b_a = selected["b_to_a"]
    probabilities = [
        record.probability
        for record in (record_a_b, record_b_a)
        if isinstance(record, CooccurrenceRecord)
    ]
    best_probability = max(probabilities)
    evidence = {
        "evidence_context": selected["context"],
        "fallback_reason": selected.get("fallback_reason"),
        "directional_evidence": {
            "a_to_b": record_a_b.as_dict() if record_a_b is not None else None,
            "b_to_a": record_b_a.as_dict() if record_b_a is not None else None,
        },
        "best_directional_probability": float(best_probability),
        "association_strength": _association_strength(best_probability, config),
        "npmi_values": [
            record.npmi
            for record in (record_a_b, record_b_a)
            if record is not None and record.npmi is not None
        ],
        "functional_grouping_hint": bool(
            best_probability >= float(config["functional_hint_threshold"])
        ),
    }
    if best_probability >= float(config["rare_threshold"]):
        return {
            **base,
            **evidence,
            "route": "direct_valid",
            "candidate_route": None,
            "status": "checked",
            "reason": "recorded_cooccurrence_supports_plausibility",
            "final_verdict": "valid",
            "score": 1.0,
        }
    if not bool(selected["rarity_evidence_sufficient"]):
        return {
            **base,
            **evidence,
            "route": "unknown",
            "status": "not_evaluable",
            "reason": "insufficient_two_direction_support_for_rarity",
            "rarity_support": selected["rarity_support"],
        }

    candidate = {
        "metric": "cooccurrence_plausibility",
        "event": {
            "type": "rare_category_cooccurrence",
            "object_ids": object_ids,
            "category_a": category_a,
            "category_b": category_b,
        },
        "rubric": (
            "Decide whether these two object categories can plausibly coexist in this scene. "
            "Low corpus frequency is evidence only; unusual but coherent combinations are valid. "
            "Do not judge their proximity or functional arrangement."
        ),
        "category_a": category_a,
        "category_b": category_b,
        **deepcopy(evidence),
        "rarity_support": deepcopy(selected["rarity_support"]),
    }
    if vlm_judge is None:
        return {
            **base,
            **evidence,
            "route": "requires_vlm",
            "candidate_route": "requires_vlm",
            "status": "requires_vlm",
            "reason": "rare_cooccurrence_requires_semantic_adjudication",
            "rarity_support": selected["rarity_support"],
            "vlm_candidate": candidate,
        }
    try:
        judgement = adjudicate_candidate(
            metric="cooccurrence_plausibility",
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
            "reason": "cooccurrence_vlm_adjudication_failed",
            "rarity_support": selected["rarity_support"],
            "adjudication_error": str(exc),
            "vlm_candidate": candidate,
        }
    return {
        **base,
        **evidence,
        "route": "vlm_adjudicated",
        "candidate_route": "requires_vlm",
        "status": "checked",
        "reason": "rare_cooccurrence_vlm_adjudicated",
        "rarity_support": selected["rarity_support"],
        "final_verdict": judgement["verdict"],
        "score": judgement["score"],
        "judge_result": judgement,
    }


def _select_directional_evidence(
    ontology: OntologyIndex,
    category_a: str,
    category_b: str,
    *,
    room_type: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    contexts: list[tuple[str, str | None]] = []
    if bool(config.get("prefer_room_conditioned", True)) and room_type.strip():
        contexts.append(("room_conditioned", room_type))
    contexts.append(("global", None))
    insufficient: dict[str, Any] | None = None
    for context_name, context_room in contexts:
        record_a_b = ontology.cooccurrence_record(
            category_a,
            category_b,
            room_type=context_room,
        )
        record_b_a = ontology.cooccurrence_record(
            category_b,
            category_a,
            room_type=context_room,
        )
        if context_name == "room_conditioned":
            # The ontology accessor intentionally falls back to global data for
            # ordinary callers. This selector must keep provenance exact so it
            # can perform an explicit global fallback below.
            if record_a_b is not None and record_a_b.context != "room_conditioned":
                record_a_b = None
            if record_b_a is not None and record_b_a.context != "room_conditioned":
                record_b_a = None
        records = [record for record in (record_a_b, record_b_a) if record is not None]
        if not records:
            continue
        best = max(record.probability for record in records)
        support = _rarity_support(record_a_b, record_b_a, config=config)
        candidate = {
            "context": context_name,
            "a_to_b": record_a_b,
            "b_to_a": record_b_a,
            "rarity_evidence_sufficient": support["sufficient"],
            "rarity_support": support,
            "fallback_reason": None,
        }
        # One recorded high-probability direction can safely certify that the
        # pair is observed. Low-frequency routing is stricter: both directions
        # and their support metadata must be adequate.
        if best >= float(config["rare_threshold"]) or support["sufficient"]:
            if context_name == "global" and insufficient is not None:
                candidate["fallback_reason"] = "room_conditioned_evidence_insufficient"
            return candidate
        insufficient = candidate
    return insufficient


def _rarity_support(
    record_a_b: CooccurrenceRecord | None,
    record_b_a: CooccurrenceRecord | None,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    minimum = int(config["min_anchor_observation_count_for_rarity"])
    both_directions = record_a_b is not None and record_b_a is not None
    authoritative_probabilities = bool(
        both_directions
        and record_a_b.probability_source == "recorded_conditional_probability"
        and record_b_a.probability_source == "recorded_conditional_probability"
    )
    observation_counts = [
        record.anchor_observation_count
        for record in (record_a_b, record_b_a)
        if record is not None
    ]
    adequate_anchor_support = bool(
        len(observation_counts) == 2
        and all(count is not None and count >= minimum for count in observation_counts)
    )
    pair_counts = [
        record.pair_count
        for record in (record_a_b, record_b_a)
        if record is not None and record.pair_count is not None
    ]
    joint_count_recorded = bool(pair_counts)
    sufficient = bool(
        both_directions
        and authoritative_probabilities
        and adequate_anchor_support
        and joint_count_recorded
    )
    return {
        "sufficient": sufficient,
        "both_directions_recorded": both_directions,
        "authoritative_directional_probabilities": authoritative_probabilities,
        "anchor_observation_counts": observation_counts,
        "minimum_anchor_observation_count": minimum,
        "adequate_anchor_observation_support": adequate_anchor_support,
        "joint_pair_count_recorded": joint_count_recorded,
        "pair_counts": pair_counts,
    }


def _association_strength(probability: float, config: dict[str, Any]) -> str:
    if probability >= float(config["functional_hint_threshold"]):
        return "functional_hint"
    if probability >= float(config["strong_threshold"]):
        return "strong"
    if probability >= float(config["rare_threshold"]):
        return "observed"
    if probability >= float(config["absent_threshold"]):
        return "rare"
    return "near_absent"


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("enabled"), bool):
        raise ValueError("cooccurrence.enabled must be boolean")
    threshold_keys = (
        "absent_threshold",
        "rare_threshold",
        "strong_threshold",
        "functional_hint_threshold",
    )
    values: list[float] = []
    for key in threshold_keys:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"cooccurrence.{key} must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"cooccurrence.{key} must be between 0 and 1")
        values.append(number)
    absent, rare, strong, functional = values
    if not absent <= rare <= strong <= functional:
        raise ValueError(
            "cooccurrence thresholds must satisfy absent <= rare <= strong <= functional_hint"
        )
    minimum = config.get("min_anchor_observation_count_for_rarity")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError(
            "cooccurrence.min_anchor_observation_count_for_rarity must be a positive integer"
        )
    if config.get("pair_unit") != "unique_unordered_category_pair":
        raise ValueError("cooccurrence.pair_unit must be unique_unordered_category_pair")
    if not isinstance(config.get("prefer_room_conditioned"), bool):
        raise ValueError("cooccurrence.prefer_room_conditioned must be boolean")
    if config.get("sparse_missing_means_unknown") is not True:
        raise ValueError(
            "cooccurrence.sparse_missing_means_unknown must remain true for a sparse top-K ontology"
        )
    if not isinstance(config.get("skip_categories"), list):
        raise ValueError("cooccurrence.skip_categories must be a list")


def _disabled_report() -> dict[str, Any]:
    return {
        "metric": "cooccurrence_plausibility",
        "evaluator_version": COOCCURRENCE_EVALUATOR_VERSION,
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
        "pair_unit": "unique_unordered_category_pair",
        "distinct_semantic_category_count": 0,
        "distinct_semantic_categories": [],
        "eligible_category_pair_count": 0,
        "category_mapping": [],
        "configured_thresholds": None,
    }
