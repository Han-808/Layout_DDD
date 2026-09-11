from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import networkx as nx

from benchmark.relation_identity import normalize_relation_id, provisional_relation_id


DEFAULT_OBJECT_MAPPING_CONFIG = {
    "minimum_score": 0.5,
    "ambiguity_margin": 0.08,
    "low_confidence_floor": 0.3,
}

_GENERIC_CATEGORIES = {"", "asset", "decor", "decoration", "furniture", "item", "object"}
_QUANTIFIER_TOKENS = {
    "a",
    "an",
    "the",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "single",
    "double",
    "triple",
    "pair",
    "couple",
    "both",
    "each",
    "every",
    "few",
    "many",
    "multiple",
    "several",
    "some",
}
_CATEGORY_ALIASES = {
    "alarm clock": "clock",
    "artbook": "book",
    "art book": "book",
    "bedside cabinet": "nightstand",
    "bedside table": "nightstand",
    "beverage can": "can",
    "bookcase": "shelf",
    "couch": "sofa",
    "display": "monitor",
    "drawer cabinet": "drawer",
    "floor lamp": "lamp",
    "glass bottle": "bottle",
    "mug": "cup",
    "night stand": "nightstand",
    "screen": "monitor",
    "soda can": "can",
    "spray bottle": "spray can",
    "table lamp": "lamp",
}


def evaluate_object_mapping(
    object_plan: dict,
    scene: dict,
    *,
    config: dict | None = None,
) -> dict[str, Any]:
    """Build an auditable one-to-one deterministic mapping without placement evidence.

    Mapping is non-scoring alignment infrastructure. No model is ever called:
    ambiguous and low-confidence identities remain unresolved and are excluded
    from downstream metric numerators and denominators.
    """

    resolved_config = {**DEFAULT_OBJECT_MAPPING_CONFIG, **(config or {})}
    minimum_score = float(resolved_config["minimum_score"])
    ambiguity_margin = float(resolved_config["ambiguity_margin"])
    low_confidence_floor = float(resolved_config["low_confidence_floor"])
    reference_slots = _reference_slots(object_plan)
    generated_objects = _generated_objects(scene)

    if not reference_slots:
        return {
            **_non_scoring_contract(),
            "evaluator_version": "object_mapping_v1",
            "status": "not_applicable",
            "mapping_policy": "deterministic_only",
            "placement_evidence_used": False,
            "summary": {
                "reference_instance_count": 0,
                "generated_object_count": len(generated_objects),
                "count_delta": len(generated_objects),
                "object_count_exact": len(generated_objects) == 0,
                "resolved_match_count": 0,
                "ambiguous_match_count": 0,
                "low_confidence_count": 0,
                "missing_reference_count": 0,
                "extra_generated_count": len(generated_objects),
                "eligible_count": 0,
                "resolved_count": 0,
                "unresolved_count": 0,
                "mapping_coverage": None,
                "resolved_coverage": None,
                "candidate_assignment_coverage": None,
            },
            "matches": [],
            "resolved_mapping": {},
            "unmatched_reference": [],
            "missing_mappings": [],
            "unmatched_generated": [_generated_identity(item) for item in generated_objects],
            "candidate_evidence": [],
            "ambiguous_mappings": [],
            "low_confidence_mappings": [],
            "unresolved_mappings": [],
            "notes": ["No explicit reference objects are available for deterministic object mapping."],
        }

    candidates = _candidate_matrix(reference_slots, generated_objects)
    selected_pairs = _maximum_weight_pairs(candidates, minimum_score=minimum_score)
    selected_by_reference = {reference_index: generated_index for reference_index, generated_index in selected_pairs}
    selected_generated = {generated_index for _, generated_index in selected_pairs}

    matches: list[dict[str, Any]] = []
    ambiguous_mappings: list[dict[str, Any]] = []
    for reference_index, generated_index in sorted(selected_pairs):
        slot = reference_slots[reference_index]
        generated = generated_objects[generated_index]
        candidate = candidates[reference_index][generated_index]
        ambiguity = _ambiguity_evidence(
            reference_index,
            generated_index,
            reference_slots,
            generated_objects,
            candidates,
            minimum_score=minimum_score,
            ambiguity_margin=ambiguity_margin,
        )
        status = "ambiguous" if ambiguity else "matched"
        confidence = _confidence(candidate["score"], status=status)
        match = {
            "plan_slot_id": slot["slot_id"],
            "plan_object_id": slot["plan_object_id"],
            "plan_instance_index": slot["instance_index"],
            "generated_object_id": str(generated.get("id")),
            "status": status,
            "confidence": confidence,
            "method": candidate["method"],
            "score": candidate["score"],
            "score_role": "alignment_similarity_only",
            "score_effect": "none",
            "score_components": candidate["components"],
            "ambiguity": ambiguity or None,
        }
        matches.append(match)
        if ambiguity:
            ambiguous_mappings.append(
                {
                    "plan_slot_id": slot["slot_id"],
                    "generated_object_id": str(generated.get("id")),
                    "candidate_generated_object_ids": list(
                        dict.fromkeys(
                            [
                                str(generated.get("id")),
                                *(ambiguity or {}).get("alternative_generated_object_ids", []),
                            ]
                        )
                    ),
                    "reason": "multiple semantic assignments are within the frozen ambiguity margin",
                    "status": "insufficient_alignment",
                    "score_effect": "none",
                }
            )

    unmatched_generated = [
        _generated_identity(obj)
        for index, obj in enumerate(generated_objects)
        if index not in selected_generated
    ]
    low_confidence_mappings: list[dict[str, Any]] = []
    for reference_index, slot in enumerate(reference_slots):
        if reference_index in selected_by_reference:
            continue
        possible = [
            candidate
            for candidate in candidates[reference_index]
            if low_confidence_floor <= candidate["score"] < minimum_score
        ]
        if possible:
            low_confidence_mappings.append(
                {
                    "plan_slot_id": slot["slot_id"],
                    "candidate_generated_object_ids": [
                        str(generated_objects[item["generated_index"]].get("id"))
                        for item in sorted(possible, key=lambda value: -value["score"])[:3]
                    ],
                    "reason": "semantic candidates exist but none reaches the deterministic acceptance threshold",
                    "score_range": {
                        "minimum_inclusive": low_confidence_floor,
                        "maximum_exclusive": minimum_score,
                    },
                    "status": "insufficient_alignment",
                    "score_effect": "none",
                }
            )

    resolved_matches = [item for item in matches if item["status"] == "matched"]
    ambiguous_matches = [item for item in matches if item["status"] == "ambiguous"]
    resolved_slot_ids = {str(item["plan_slot_id"]) for item in resolved_matches}
    ambiguous_slot_ids = {str(item["plan_slot_id"]) for item in ambiguous_matches}
    low_confidence_slot_ids = {str(item["plan_slot_id"]) for item in low_confidence_mappings}
    missing_mappings = [
        {
            **_reference_identity(slot),
            "status": "missing",
            "reason": "no_candidate_reaches_the_low_confidence_floor",
        }
        for index, slot in enumerate(reference_slots)
        if index not in selected_by_reference and str(slot["slot_id"]) not in low_confidence_slot_ids
    ]
    unmatched_reference = [deepcopy(item) for item in missing_mappings]
    mapped_count = len(matches)
    reference_count = len(reference_slots)

    unresolved_mappings: list[dict[str, Any]] = []
    for slot in reference_slots:
        slot_id = str(slot["slot_id"])
        if slot_id in resolved_slot_ids:
            continue
        if slot_id in ambiguous_slot_ids:
            reason = "ambiguous"
        elif slot_id in low_confidence_slot_ids:
            reason = "low_confidence"
        else:
            reason = "no_confident_candidate"
        unresolved_mappings.append(
            {
                "plan_slot_id": slot["slot_id"],
                "plan_object_id": slot["plan_object_id"],
                "reason": reason,
                "status": "insufficient_alignment",
                "score_effect": "none",
            }
        )

    status = "complete"
    if missing_mappings or low_confidence_mappings or unmatched_generated or ambiguous_matches:
        status = "partial"
    return {
        **_non_scoring_contract(),
        "evaluator_version": "object_mapping_v1",
        "status": status,
        "mapping_policy": "deterministic_only",
        "placement_evidence_used": False,
        "config": {
            "minimum_score": minimum_score,
            "ambiguity_margin": ambiguity_margin,
            "low_confidence_floor": low_confidence_floor,
        },
        "summary": {
            "reference_instance_count": reference_count,
            "generated_object_count": len(generated_objects),
            "count_delta": len(generated_objects) - reference_count,
            "object_count_exact": len(generated_objects) == reference_count,
            "resolved_match_count": len(resolved_matches),
            "ambiguous_match_count": len(ambiguous_matches),
            "low_confidence_count": len(low_confidence_mappings),
            "missing_reference_count": len(missing_mappings),
            "extra_generated_count": len(unmatched_generated),
            "eligible_count": reference_count,
            "resolved_count": len(resolved_matches),
            "unresolved_count": len(unresolved_mappings),
            "mapping_coverage": len(resolved_matches) / float(reference_count),
            "resolved_coverage": len(resolved_matches) / float(reference_count),
            "candidate_assignment_coverage": mapped_count / float(reference_count),
        },
        "matches": matches,
        "resolved_mapping": _resolved_mapping(resolved_matches),
        "unmatched_reference": unmatched_reference,
        "missing_mappings": missing_mappings,
        "unmatched_generated": unmatched_generated,
        "candidate_evidence": _candidate_evidence(reference_slots, generated_objects, candidates),
        "ambiguous_mappings": ambiguous_mappings,
        "low_confidence_mappings": low_confidence_mappings,
        "unresolved_mappings": unresolved_mappings,
        "notes": [
            "Object mapping is alignment infrastructure and never contributes a score or penalty.",
            "P0a records missing, ambiguous, and low-confidence identities separately; P0c decides any fidelity effect.",
            "Mapping is global, one-to-one, and deterministic; no model is ever called to resolve identity.",
            "Object IDs are not treated as semantic identity in the natural-language-only track.",
            "Generated placement and relation outcomes are intentionally excluded to avoid circular fidelity scoring.",
            "Ambiguous and low-confidence mappings remain unresolved for downstream relation metrics.",
        ],
    }


def route_relationship_intents(relationship_intents: dict | None, mapping_report: dict | None) -> dict[str, Any] | None:
    """Route plan IDs to generated IDs without changing relation semantics.

    Binary predicates require unique mappings. Group predicates deliberately
    support count expansion (for example, one ``chairs`` plan object with
    count=6 can become six members of an ``around`` relation).
    """

    if not isinstance(relationship_intents, dict):
        return None
    resolved = mapping_report.get("resolved_mapping") if isinstance(mapping_report, dict) else {}
    resolved = resolved if isinstance(resolved, dict) else {}
    routed_oor: list[dict[str, Any]] = []
    routed_oar: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for family, output in [("oor", routed_oor), ("oar", routed_oar)]:
        key = f"{family}_relations"
        relations = relationship_intents.get(key)
        for index, relation in enumerate(relations if isinstance(relations, list) else []):
            default_relation_id = provisional_relation_id(family, index)
            if not isinstance(relation, dict):
                unresolved.append(
                    {
                        "family": family,
                        "relation_id": default_relation_id,
                        "relation_id_generated": True,
                        "relation_id_provenance": "legacy_family_index",
                        "relation_index": index,
                        "status": "insufficient_alignment",
                        "reason": "invalid_relation_spec",
                        "score_effect": "none",
                    }
                )
                continue
            relation_with_id = deepcopy(relation)
            relation_id = normalize_relation_id(relation_with_id.get("relation_id"))
            if relation_id is None:
                relation_id = default_relation_id
                relation_with_id["relation_id_generated"] = True
                relation_with_id["relation_id_provenance"] = "legacy_family_index"
            relation_with_id["relation_id"] = relation_id
            routed, failure = _route_one_relation(
                relation_with_id,
                family=family,
                resolved=resolved,
            )
            if routed is None:
                unresolved.append(
                    {
                        "family": family,
                        "relation_id": relation_id,
                        "relation_id_generated": relation_with_id.get("relation_id_generated") is True,
                        "relation_id_provenance": relation_with_id.get("relation_id_provenance"),
                        "relation_index": index,
                        "status": "insufficient_alignment",
                        "reason": failure or "identity_mapping_not_uniquely_resolved",
                        "score_effect": "none",
                        "relation": deepcopy(relation_with_id),
                    }
                )
                continue
            output.append(routed)

    total = len(routed_oor) + len(routed_oar) + len(unresolved)
    routed_count = len(routed_oor) + len(routed_oar)
    return {
        **deepcopy(relationship_intents),
        "oor_relations": routed_oor,
        "oar_relations": routed_oar,
        "unresolved_relations": unresolved,
        "alignment": {
            "role": "routing_only",
            "score_effect": "none",
            "total_relation_count": total,
            "routed_relation_count": routed_count,
            "unresolved_relation_count": len(unresolved),
            "eligible_count": total,
            "resolved_count": routed_count,
            "unresolved_count": len(unresolved),
            "coverage": routed_count / float(total) if total else None,
        },
    }


def _route_one_relation(
    relation: dict[str, Any],
    *,
    family: str,
    resolved: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    routed = deepcopy(relation)
    relation_type = _compact_relation_type(_first_present(relation, ["type", "predicate", "relation"]))
    routed["type"] = relation_type
    routed["family"] = family

    if family == "oar":
        subject_plan_id = _first_present(relation, ["subject_id", "subject"])
        subject_id = _unique_resolved_id(resolved, subject_plan_id)
        if subject_id is None:
            return None, "oar_subject_mapping_not_uniquely_resolved"
        routed["subject_id"] = subject_id
        return routed, None

    if relation_type == "between":
        subject_id = _unique_resolved_id(resolved, _first_present(relation, ["subject_id", "subject"]))
        anchor_plan_ids = _relation_id_list(_first_present(relation, ["object_ids", "anchor_ids", "target_ids"]))
        anchor_ids = [_unique_resolved_id(resolved, value) for value in anchor_plan_ids]
        if subject_id is None or len(anchor_ids) != 2 or any(value is None for value in anchor_ids):
            return None, "between_requires_one_subject_and_two_uniquely_resolved_anchors"
        routed["subject_id"] = subject_id
        routed["object_ids"] = [str(value) for value in anchor_ids]
        routed.pop("object_id", None)
        return routed, None

    if relation_type == "ordered":
        member_plan_ids = _relation_id_list(_first_present(relation, ["object_ids", "member_ids", "subject_ids"]))
        member_ids = _expand_resolved_ids(resolved, member_plan_ids)
        if member_ids is None or len(member_ids) < 2:
            return None, "ordered_member_mapping_not_resolved"
        routed["object_ids"] = member_ids
        return routed, None

    if relation_type == "around":
        member_plan_ids = _relation_id_list(_first_present(relation, ["subject_ids", "member_ids", "object_ids"]))
        if not member_plan_ids:
            singular_members = _first_present(relation, ["subject_id", "subject"])
            member_plan_ids = [str(singular_members)] if singular_members is not None else []
        member_ids = _expand_resolved_ids(resolved, member_plan_ids)
        anchor_plan_id = _first_present(relation, ["object_id", "object", "anchor_id", "target_id"])
        anchor_id = _unique_resolved_id(resolved, anchor_plan_id)
        if member_ids is None or len(member_ids) < 2 or anchor_id is None:
            return None, "around_requires_resolved_members_and_one_center_anchor"
        routed["subject_ids"] = member_ids
        routed["object_id"] = anchor_id
        return routed, None

    # Unknown group predicates retain their lists for the VLM, but all plan IDs
    # still need to be resolved before the claim can enter scoring.
    subject_plan_ids = _relation_id_list(_first_present(relation, ["subject_ids", "member_ids"]))
    object_plan_ids = _relation_id_list(_first_present(relation, ["object_ids", "anchor_ids", "target_ids"]))
    if subject_plan_ids or object_plan_ids:
        subject_ids = _expand_resolved_ids(resolved, subject_plan_ids) if subject_plan_ids else []
        object_ids = _expand_resolved_ids(resolved, object_plan_ids) if object_plan_ids else []
        if subject_ids is None or object_ids is None or len(subject_ids) + len(object_ids) < 2:
            return None, "group_relation_identity_mapping_not_resolved"
        if subject_ids:
            routed["subject_ids"] = subject_ids
        if object_ids:
            routed["object_ids"] = object_ids
        return routed, None

    subject_plan_id = _first_present(relation, ["subject_id", "subject"])
    object_plan_id = _first_present(relation, ["object_id", "object", "anchor_id", "target_id"])
    subject_id = _unique_resolved_id(resolved, subject_plan_id)
    object_id = _unique_resolved_id(resolved, object_plan_id)
    if subject_id is None or object_id is None:
        return None, "binary_relation_identity_mapping_not_uniquely_resolved"
    routed["subject_id"] = subject_id
    routed["object_id"] = object_id
    return routed, None


def _expand_resolved_ids(resolved: dict[str, Any], plan_ids: list[str]) -> list[str] | None:
    expanded: list[str] = []
    for plan_id in plan_ids:
        values = resolved.get(str(plan_id))
        if not isinstance(values, list) or not values:
            return None
        expanded.extend(str(value) for value in values if str(value).strip())
    return list(dict.fromkeys(expanded))


def _relation_id_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _compact_relation_type(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _non_scoring_contract() -> dict[str, Any]:
    return {
        "metric_role": "alignment_only",
        "scorable": False,
        "affects_benchmark_score": False,
        "score": None,
        "penalty": None,
        "candidate_scores_are_metric_scores": False,
    }


def _unique_resolved_id(resolved_mapping: dict, plan_object_id: object) -> str | None:
    if plan_object_id is None:
        return None
    values = resolved_mapping.get(str(plan_object_id))
    if not isinstance(values, list) or len(values) != 1:
        return None
    value = str(values[0]).strip()
    return value or None


def _first_present(mapping: dict, keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _reference_slots(object_plan: dict) -> list[dict[str, Any]]:
    objects = object_plan.get("objects") if isinstance(object_plan, dict) else None
    slots: list[dict[str, Any]] = []
    for object_index, obj in enumerate(objects if isinstance(objects, list) else []):
        if not isinstance(obj, dict):
            continue
        plan_object_id = str(obj.get("id") or f"obj_{object_index:03d}")
        count = _positive_count(obj.get("count"))
        for instance_index in range(count):
            slots.append(
                {
                    "slot_id": plan_object_id if count == 1 else f"{plan_object_id}#{instance_index + 1}",
                    "plan_object_id": plan_object_id,
                    "instance_index": instance_index,
                    "category": str(obj.get("category") or ""),
                    "description": str(obj.get("description") or ""),
                    "role": str(obj.get("role") or ""),
                    "metadata": deepcopy(obj.get("metadata")) if isinstance(obj.get("metadata"), dict) else {},
                }
            )
    return slots


def _generated_objects(scene: dict) -> list[dict[str, Any]]:
    objects = scene.get("objects") if isinstance(scene, dict) else None
    return [deepcopy(obj) for obj in objects if isinstance(obj, dict)] if isinstance(objects, list) else []


def _candidate_matrix(reference_slots: list[dict], generated_objects: list[dict]) -> list[list[dict[str, Any]]]:
    return [
        [
            _mapping_candidate(reference_index, generated_index, slot, generated)
            for generated_index, generated in enumerate(generated_objects)
        ]
        for reference_index, slot in enumerate(reference_slots)
    ]


def _mapping_candidate(
    reference_index: int,
    generated_index: int,
    slot: dict,
    generated: dict,
) -> dict[str, Any]:
    category_score = _category_similarity(slot.get("category"), _generated_categories(generated))
    description_score = _description_similarity(
        [slot.get("description")],
        _generated_descriptions(generated),
    )
    if category_score > 0.0:
        score = 0.75 * category_score + 0.25 * description_score
        method = "category_description"
    else:
        # A unique near-exact description may resolve category-label drift, while
        # partial description overlap remains below the 0.5 acceptance threshold.
        score = 0.60 * description_score
        method = "description_only"
    return {
        "reference_index": reference_index,
        "generated_index": generated_index,
        "score": round(float(score), 6),
        "method": method,
        "components": {
            "category": round(float(category_score), 6),
            "description": round(float(description_score), 6),
        },
    }


def _maximum_weight_pairs(candidates: list[list[dict]], *, minimum_score: float) -> list[tuple[int, int]]:
    graph = nx.Graph()
    for reference_index, row in enumerate(candidates):
        graph.add_node(("reference", reference_index), bipartite=0)
        for candidate in row:
            generated_index = int(candidate["generated_index"])
            graph.add_node(("generated", generated_index), bipartite=1)
            if float(candidate["score"]) < minimum_score:
                continue
            tie_break = max(0, 999 - reference_index * 31 - generated_index)
            weight = int(round(float(candidate["score"]) * 1_000_000)) * 1_000 + tie_break
            graph.add_edge(
                ("reference", reference_index),
                ("generated", generated_index),
                weight=weight,
            )
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    pairs: list[tuple[int, int]] = []
    for left, right in matching:
        if left[0] == "generated":
            left, right = right, left
        if left[0] == "reference" and right[0] == "generated":
            pairs.append((int(left[1]), int(right[1])))
    return sorted(pairs)


def _ambiguity_evidence(
    reference_index: int,
    generated_index: int,
    reference_slots: list[dict],
    generated_objects: list[dict],
    candidates: list[list[dict]],
    *,
    minimum_score: float,
    ambiguity_margin: float,
) -> dict[str, Any] | None:
    selected_score = float(candidates[reference_index][generated_index]["score"])
    interchangeable_instances = sum(
        1
        for slot in reference_slots
        if slot["plan_object_id"] == reference_slots[reference_index]["plan_object_id"]
    ) > 1
    generated_alternatives = [
        item
        for index, item in enumerate(candidates[reference_index])
        if not interchangeable_instances
        and index != generated_index
        and float(item["score"]) >= minimum_score
        and selected_score - float(item["score"]) <= ambiguity_margin
    ]
    reference_alternatives = [
        row[generated_index]
        for index, row in enumerate(candidates)
        if index != reference_index
        and reference_slots[index]["plan_object_id"] != reference_slots[reference_index]["plan_object_id"]
        and float(row[generated_index]["score"]) >= minimum_score
        and selected_score - float(row[generated_index]["score"]) <= ambiguity_margin
    ]
    if not generated_alternatives and not reference_alternatives:
        return None
    return {
        "margin": ambiguity_margin,
        "alternative_generated_object_ids": [
            str(generated_objects[item["generated_index"]].get("id")) for item in generated_alternatives
        ],
        "alternative_plan_slot_ids": [
            reference_slots[item["reference_index"]]["slot_id"] for item in reference_alternatives
        ],
    }


def _category_similarity(reference: object, generated_values: list[object]) -> float:
    reference_key = _category_key(reference)
    if not reference_key or reference_key in _GENERIC_CATEGORIES:
        return 0.0
    best = 0.0
    reference_tokens = set(reference_key.split())
    for value in generated_values:
        generated_key = _category_key(value)
        if not generated_key or generated_key in _GENERIC_CATEGORIES:
            continue
        if reference_key == generated_key:
            return 1.0
        generated_tokens = set(generated_key.split())
        if reference_tokens <= generated_tokens or generated_tokens <= reference_tokens:
            best = max(best, 0.8)
        else:
            best = max(best, _token_f1(reference_key, generated_key) * 0.7)
    return best


def _description_similarity(reference_values: list[object], generated_values: list[object]) -> float:
    best = 0.0
    for reference in reference_values:
        for generated in generated_values:
            best = max(best, _token_f1(reference, generated))
    return best


def _generated_categories(generated: dict) -> list[object]:
    return [generated.get("category")]


def _generated_descriptions(generated: dict) -> list[object]:
    return [generated.get("description")]


def _candidate_evidence(reference_slots: list[dict], generated_objects: list[dict], candidates: list[list[dict]]) -> list[dict]:
    evidence = []
    for reference_index, row in enumerate(candidates):
        top = sorted(row, key=lambda item: (-float(item["score"]), int(item["generated_index"])))[:3]
        evidence.append(
            {
                "plan_slot_id": reference_slots[reference_index]["slot_id"],
                "top_candidates": [
                    {
                        "generated_object_id": str(generated_objects[item["generated_index"]].get("id")),
                        "score": item["score"],
                        "score_role": "alignment_similarity_only",
                        "method": item["method"],
                        "score_components": item["components"],
                    }
                    for item in top
                ],
            }
        )
    return evidence


def _resolved_mapping(matches: list[dict]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for match in matches:
        mapping.setdefault(str(match["plan_object_id"]), []).append(str(match["generated_object_id"]))
    return mapping


def _reference_identity(slot: dict) -> dict[str, Any]:
    return {
        "plan_slot_id": slot["slot_id"],
        "plan_object_id": slot["plan_object_id"],
        "plan_instance_index": slot["instance_index"],
        "category": slot["category"],
        "description": slot["description"],
        "score_effect": "none",
    }


def _generated_identity(obj: dict) -> dict[str, Any]:
    return {
        "generated_object_id": str(obj.get("id") or ""),
        "category": str(obj.get("category") or ""),
        "description": str(obj.get("description") or ""),
        "score_effect": "none",
    }


def _confidence(score: float, *, status: str) -> str:
    if status == "ambiguous":
        return "low"
    if score >= 0.85:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


def _category_key(value: object) -> str:
    normalized = _normal_text(value)
    return _CATEGORY_ALIASES.get(normalized, normalized)


def _token_f1(left: object, right: object) -> float:
    left_tokens = set(_normal_text(left).split())
    right_tokens = set(_normal_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / float(len(left_tokens))
    recall = overlap / float(len(right_tokens))
    return 2.0 * precision * recall / (precision + recall)


def _normal_text(value: object) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower().replace("_", " "))
    return " ".join(
        _singular_token(token)
        for token in tokens
        if token not in _QUANTIFIER_TOKENS and not token.isdigit()
    )


def _singular_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("es") and token[-3:-2] in {"s", "x", "z"}:
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _positive_count(value: object) -> int:
    if isinstance(value, bool):
        return 1
    try:
        count = int(value or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, count)
