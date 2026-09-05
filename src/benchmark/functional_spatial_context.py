"""Compact, non-decisional Function prerequisites for Placement discovery.

The bridge shares structured prerequisites rather than Functional images or
verdicts: usable-side clearance measurements, admitted operational relations,
and context-only relation proposals that Function intentionally did not own.
Placement remains responsible for proposing an independent typed location
question; the Functional ownership ledger prevents duplicate scoring.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable

FUNCTIONAL_SPATIAL_CONTEXT_VERSION = "functional_spatial_context_v3"
FUNCTIONAL_SPATIAL_CONTEXT_MAX_CLEARANCE_OBJECTS = 32
FUNCTIONAL_SPATIAL_CONTEXT_MAX_RELATED_PAIRS = 32
FUNCTIONAL_SPATIAL_CONTEXT_MAX_CHARS = 22_000
FUNCTIONAL_SPATIAL_CONTEXT_MAX_INTERSECTIONS = 4
_FUNCTIONAL_RELATION_PREDICATES = frozenset(
    {"directional_correspondence", "relative_use_geometry"}
)


def project_functional_spatial_context(
    functional_report: Any,
    *,
    known_object_ids: Iterable[str],
    max_clearance_objects: int = (
        FUNCTIONAL_SPATIAL_CONTEXT_MAX_CLEARANCE_OBJECTS
    ),
    max_related_pairs: int = FUNCTIONAL_SPATIAL_CONTEXT_MAX_RELATED_PAIRS,
    max_context_chars: int = FUNCTIONAL_SPATIAL_CONTEXT_MAX_CHARS,
) -> dict[str, Any] | None:
    """Project validated Function discovery into optional Placement context.

    Invalid, absent, or empty upstream context degrades to ``None``.  This is
    deliberate: the bridge is recall-oriented context and cannot become a
    prerequisite for Placement evaluation.
    """

    if not isinstance(functional_report, dict):
        return None
    discovery = functional_report.get("functional_discovery")
    if (
        not isinstance(discovery, dict)
        or discovery.get("decision_authority") != "none"
    ):
        return None

    known = tuple(str(item) for item in known_object_ids)
    if not known or len(known) != len(set(known)):
        return None
    known_set = set(known)
    clearance_limit = min(
        FUNCTIONAL_SPATIAL_CONTEXT_MAX_CLEARANCE_OBJECTS,
        max(0, int(max_clearance_objects)),
    )
    relation_limit = min(
        FUNCTIONAL_SPATIAL_CONTEXT_MAX_RELATED_PAIRS,
        max(0, int(max_related_pairs)),
    )

    clearance_measurements = _directional_clearance_by_target(
        functional_report.get("functional_measurement_bank")
    )
    clearance_requirements: list[dict[str, Any]] = []
    for item in discovery.get("approach_clearance_targets") or []:
        if not isinstance(item, dict) or item.get("need_clearance") is not True:
            continue
        object_id = str(item.get("target_id") or "").strip()
        observation_goal = str(
            item.get("observation_goal") or ""
        ).strip()[:240]
        source_check_id = str(
            item.get("discovery_id") or ""
        ).strip()
        if (
            object_id in known_set
            and observation_goal
            and source_check_id
            and object_id not in {
                row["object_id"] for row in clearance_requirements
            }
        ):
            clearance_requirements.append(
                {
                    "object_id": object_id,
                    "observation_goal": observation_goal,
                    "source_check_id": source_check_id,
                    "ownership": "neutral_prerequisite",
                    "measurement": deepcopy(
                        clearance_measurements.get(object_id)
                    ),
                }
            )

    related_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    relation_sources: list[tuple[str, dict[str, Any], str, str]] = []
    for field in ("within_group_correspondences", "cross_group_correspondences"):
        for item in discovery.get(field) or []:
            if isinstance(item, dict):
                relation_sources.append(
                    (
                        field,
                        item,
                        "function_owned",
                        "background_only",
                    )
                )
    admission_audit = discovery.get("relation_admission_audit")
    if isinstance(admission_audit, dict):
        for item in admission_audit.get("context_only_relations") or []:
            if isinstance(item, dict):
                scope = str(item.get("scope") or "").strip()
                relation_sources.append(
                    (
                        (
                            "cross_group_correspondences"
                            if scope == "cross_group"
                            else "within_group_correspondences"
                        ),
                        item,
                        "unowned_context",
                        "candidate_attention",
                    )
                )
    for field, item, ownership, placement_role in relation_sources:
        target_ids = item.get("target_ids")
        if not isinstance(target_ids, (list, tuple)):
            continue
        pair = [str(value).strip() for value in target_ids]
        if (
            len(pair) != 2
            or len(set(pair)) != 2
            or any(value not in known_set for value in pair)
        ):
            continue
        predicate = str(item.get("predicate") or "").strip()
        observation_goal = str(item.get("observation_goal") or "").strip()
        source_relation_id = str(
            item.get("discovery_id")
            or item.get("source_relation_id")
            or item.get("proposal_ref")
            or ""
        ).strip()
        dependency = str(item.get("dependency") or "").strip()
        counterpart_mode = str(item.get("counterpart_mode") or "").strip()
        ordinary_mobility = str(
            item.get("ordinary_mobility") or ""
        ).strip()
        if (
            predicate not in _FUNCTIONAL_RELATION_PREDICATES
            or not observation_goal
            or not source_relation_id
            or dependency not in {"required", "contextual"}
            or counterpart_mode not in {"dedicated", "shared", "alternative"}
            or ordinary_mobility
            not in {"fixed", "movable_companion", "portable_unrelated"}
        ):
            continue
        sorted_pair = tuple(sorted(pair))
        # Relation records are directed focal→counterpart assignments.  The
        # sorted pair is only a stable set identity for consumers; reciprocal
        # assignments may carry different mobility/role semantics and must
        # not be collapsed.
        identity = (pair[0], pair[1], predicate)
        if identity in seen_pairs:
            continue
        seen_pairs.add(identity)
        related_pairs.append(
            {
                "object_ids": list(sorted_pair),
                # ``ordinary_mobility`` and ``counterpart_mode`` are ordered
                # role attributes.  Keep the stable sorted pair for identity,
                # but never discard which endpoint is focal/counterpart.
                "focal_id": pair[0],
                "counterpart_id": pair[1],
                "predicate": predicate,
                "observation_goal": observation_goal[:240],
                "source_relation_id": source_relation_id,
                "scope": _relation_scope(item, fallback_field=field),
                "dependency": dependency,
                "counterpart_mode": counterpart_mode,
                "ordinary_mobility": ordinary_mobility,
                "ownership": ownership,
                "placement_role": placement_role,
            }
        )

    context = {
        "schema_version": FUNCTIONAL_SPATIAL_CONTEXT_VERSION,
        "context_role": "attention_only",
        "decision_authority": "none",
        "clearance_requirements": clearance_requirements[:clearance_limit],
        "related_pairs": related_pairs[:relation_limit],
    }
    if not context["clearance_requirements"] and not context["related_pairs"]:
        return None
    context = _compact_functional_spatial_context(
        context,
        max_context_chars=max(4_000, int(max_context_chars)),
        eligible_clearance_count=len(clearance_requirements),
        eligible_relation_count=len(related_pairs),
        pre_omitted_clearance_ids=[
            str(item["object_id"])
            for item in clearance_requirements[clearance_limit:]
        ],
        pre_omitted_relation_refs=[
            str(item["source_relation_id"])
            for item in related_pairs[relation_limit:]
        ],
    )
    return validate_functional_spatial_context(
        context,
        known_object_ids=known,
    )


def _directional_clearance_by_target(
    bank: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(bank, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in bank.get("check_measurements") or []:
        if not isinstance(row, dict) or row.get("check_type") != "clearance":
            continue
        target_ids = [
            str(item)
            for item in row.get("target_ids") or []
            if str(item).strip()
        ]
        extensions = row.get("measurement_extensions")
        profile = (
            extensions.get("directional_clearance")
            if isinstance(extensions, dict)
            else None
        )
        if isinstance(profile, dict) and isinstance(
            profile.get("measurement"), dict
        ):
            profile = profile["measurement"]
        if len(target_ids) != 1 or not isinstance(profile, dict):
            continue
        result[target_ids[0]] = {
            "status": str(profile.get("status") or "unavailable"),
            "usable_side_id": profile.get("usable_side_id"),
            "world_outward_direction_xy": deepcopy(
                profile.get("world_outward_direction_xy")
            ),
            "frontage_origin_xy": deepcopy(profile.get("frontage_origin_xy")),
            "corridor_depth_m": profile.get("corridor_depth_m"),
            "corridor_half_width_m": profile.get("corridor_half_width_m"),
            "nearest_forward_obstacle_distance_m": profile.get(
                "nearest_forward_obstacle_distance_m"
            ),
            "forward_intersections": [
                {
                    key: deepcopy(item.get(key))
                    for key in (
                        "object_id",
                        "forward_near_distance_m",
                        "forward_far_distance_m",
                        "lateral_clearance_m",
                        "corridor_overlap_depth_m",
                        "corridor_overlap_width_m",
                        "corridor_width_overlap_fraction",
                        "corridor_overlap_area_proxy_m2",
                        "vertical_overlap_with_approach_m",
                        "vertical_relevant",
                        "support_relation",
                        "thin_floor_layer",
                        "ordinary_mobility",
                        "excluded_from_obstacle",
                    )
                }
                for item in profile.get("forward_intersections") or []
                if isinstance(item, dict)
            ][:FUNCTIONAL_SPATIAL_CONTEXT_MAX_INTERSECTIONS],
            "unavailable_reason": profile.get("unavailable_reason"),
            "decision_authority": "none",
        }
    return result


def _relation_scope(item: dict[str, Any], *, fallback_field: str) -> str:
    declared = str(item.get("scope") or "").strip()
    if declared in {"within_group", "cross_group"}:
        return declared
    focal_group = str(item.get("focal_group_id") or "").strip()
    counterpart_group = str(item.get("counterpart_group_id") or "").strip()
    if focal_group and counterpart_group:
        return (
            "within_group"
            if focal_group == counterpart_group
            else "cross_group"
        )
    return fallback_field.removesuffix("_correspondences")


def validate_functional_spatial_context(
    value: Any,
    *,
    known_object_ids: Iterable[str],
) -> dict[str, Any]:
    """Validate the compact model-facing context without adding semantics."""

    if not isinstance(value, dict):
        raise TypeError("functional spatial context must be a JSON object")
    allowed = {
        "schema_version",
        "context_role",
        "decision_authority",
        "clearance_requirements",
        "related_pairs",
        "projection_audit",
    }
    extra = set(value) - allowed
    if extra:
        raise ValueError(
            "functional spatial context contains unsupported fields: "
            f"{sorted(extra)}"
        )
    if value.get("schema_version") != FUNCTIONAL_SPATIAL_CONTEXT_VERSION:
        raise ValueError("unsupported functional spatial context version")
    if value.get("context_role") != "attention_only":
        raise ValueError("functional spatial context must be attention-only")
    if value.get("decision_authority") != "none":
        raise ValueError(
            "functional spatial context cannot have decision authority"
        )

    known = {str(item) for item in known_object_ids}
    clearance = value.get("clearance_requirements")
    if not isinstance(clearance, list):
        raise ValueError("clearance_requirements must be a JSON list")
    if len(clearance) > FUNCTIONAL_SPATIAL_CONTEXT_MAX_CLEARANCE_OBJECTS:
        raise ValueError("clearance_requirements exceeds the compact limit")
    normalized_clearance: list[dict[str, Any]] = []
    seen_clearance: set[str] = set()
    for item in clearance:
        if not isinstance(item, dict) or set(item) != {
            "object_id",
            "observation_goal",
            "source_check_id",
            "ownership",
            "measurement",
        }:
            raise ValueError("clearance requirement has unsupported fields")
        object_id = str(item.get("object_id") or "").strip()
        goal = str(item.get("observation_goal") or "").strip()
        source = str(item.get("source_check_id") or "").strip()
        ownership = str(item.get("ownership") or "").strip()
        if object_id not in known:
            raise ValueError(
                "clearance requirement contains invalid object IDs"
            )
        if (
            object_id in seen_clearance
            or not goal
            or not source
            or len(goal) > 240
            or ownership != "neutral_prerequisite"
        ):
            raise ValueError("clearance requirement is invalid")
        seen_clearance.add(object_id)
        normalized_clearance.append(
            {
                "object_id": object_id,
                "observation_goal": goal,
                "source_check_id": source,
                "ownership": ownership,
                "measurement": _validate_clearance_measurement(
                    item.get("measurement"),
                    known_object_ids=known,
                ),
            }
        )

    relations = value.get("related_pairs")
    if not isinstance(relations, list):
        raise ValueError("related_pairs must be a JSON list")
    if len(relations) > FUNCTIONAL_SPATIAL_CONTEXT_MAX_RELATED_PAIRS:
        raise ValueError("related_pairs exceeds the compact context limit")
    normalized_relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in relations:
        if not isinstance(item, dict) or set(item) != {
            "object_ids",
            "focal_id",
            "counterpart_id",
            "predicate",
            "observation_goal",
            "source_relation_id",
            "scope",
            "dependency",
            "counterpart_mode",
            "ordinary_mobility",
            "ownership",
            "placement_role",
        }:
            raise ValueError(
                "each related pair requires the v3 neutral relation fields"
            )
        object_ids = item.get("object_ids")
        if not isinstance(object_ids, list):
            raise ValueError("related pair object_ids must be a JSON list")
        pair = [str(object_id).strip() for object_id in object_ids]
        if (
            len(pair) != 2
            or pair != sorted(pair)
            or len(set(pair)) != 2
            or any(object_id not in known for object_id in pair)
        ):
            raise ValueError("related pair contains invalid object IDs")
        focal_id = str(item.get("focal_id") or "").strip()
        counterpart_id = str(item.get("counterpart_id") or "").strip()
        if (
            focal_id not in known
            or counterpart_id not in known
            or focal_id == counterpart_id
            or {focal_id, counterpart_id} != set(pair)
        ):
            raise ValueError(
                "related pair requires trusted ordered focal/counterpart IDs"
            )
        predicate = str(item.get("predicate") or "").strip()
        goal = str(item.get("observation_goal") or "").strip()
        source = str(item.get("source_relation_id") or "").strip()
        scope = str(item.get("scope") or "").strip()
        dependency = str(item.get("dependency") or "").strip()
        counterpart_mode = str(item.get("counterpart_mode") or "").strip()
        ordinary_mobility = str(
            item.get("ordinary_mobility") or ""
        ).strip()
        ownership = str(item.get("ownership") or "").strip()
        placement_role = str(item.get("placement_role") or "").strip()
        identity = (focal_id, counterpart_id, predicate)
        if identity in seen:
            raise ValueError(
                "functional spatial context repeats a pair predicate"
            )
        seen.add(identity)
        if (
            predicate not in _FUNCTIONAL_RELATION_PREDICATES
            or not goal
            or len(goal) > 240
            or not source
            or scope not in {"within_group", "cross_group"}
            or dependency not in {"required", "contextual"}
            or counterpart_mode not in {"dedicated", "shared", "alternative"}
            or ordinary_mobility
            not in {"fixed", "movable_companion", "portable_unrelated"}
            or ownership not in {"function_owned", "unowned_context"}
            or placement_role not in {"background_only", "candidate_attention"}
        ):
            raise ValueError("related pair neutral context is invalid")
        normalized_relations.append(
            {
                "object_ids": pair,
                "focal_id": focal_id,
                "counterpart_id": counterpart_id,
                "predicate": predicate,
                "observation_goal": goal,
                "source_relation_id": source,
                "scope": scope,
                "dependency": dependency,
                "counterpart_mode": counterpart_mode,
                "ordinary_mobility": ordinary_mobility,
                "ownership": ownership,
                "placement_role": placement_role,
            }
        )

    projection_audit = _validate_projection_audit(
        value.get("projection_audit"),
        known_object_ids=known,
    )
    return {
        "schema_version": FUNCTIONAL_SPATIAL_CONTEXT_VERSION,
        "context_role": "attention_only",
        "decision_authority": "none",
        "clearance_requirements": normalized_clearance,
        "related_pairs": normalized_relations,
        "projection_audit": projection_audit,
    }


def _compact_functional_spatial_context(
    value: dict[str, Any],
    *,
    max_context_chars: int,
    eligible_clearance_count: int,
    eligible_relation_count: int,
    pre_omitted_clearance_ids: list[str],
    pre_omitted_relation_refs: list[str],
) -> dict[str, Any]:
    """Bound optional Placement guidance without corrupting its schema.

    Measurement Bank remains complete in the Functional report.  This bridge
    is a compact attention projection, so it first sheds per-obstacle detail,
    then full measurement payloads, and only as a last resort whole optional
    rows.  Every omission is explicit in ``projection_audit``.
    """

    result = deepcopy(value)
    omitted_measurement_ids: list[str] = []
    reduced_measurement_ids: list[str] = []
    omitted_relation_refs: list[str] = list(
        reversed(pre_omitted_relation_refs)
    )
    omitted_clearance_ids: list[str] = list(
        reversed(pre_omitted_clearance_ids)
    )

    def attach_audit() -> None:
        result["projection_audit"] = {
            "policy": "bounded_attention_projection_v1",
            "max_context_chars": max_context_chars,
            "eligible_clearance_count": eligible_clearance_count,
            "emitted_clearance_count": len(
                result["clearance_requirements"]
            ),
            "eligible_relation_count": eligible_relation_count,
            "emitted_relation_count": len(result["related_pairs"]),
            "reduced_measurement_object_ids": sorted(
                set(reduced_measurement_ids)
            ),
            "omitted_measurement_object_ids": sorted(
                set(omitted_measurement_ids)
            ),
            "omitted_clearance_object_ids": list(
                reversed(omitted_clearance_ids)
            ),
            "omitted_relation_source_ids": list(
                reversed(omitted_relation_refs)
            ),
            "decision_authority": "none",
        }

    def encoded_length() -> int:
        attach_audit()
        return len(
            json.dumps(
                result,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    # Preserve every target summary where possible.  Remove low-priority tail
    # intersections before removing a complete measurement payload.
    while encoded_length() > max_context_chars:
        changed = False
        for row in reversed(result["clearance_requirements"]):
            measurement = row.get("measurement")
            intersections = (
                measurement.get("forward_intersections")
                if isinstance(measurement, dict)
                else None
            )
            if isinstance(intersections, list) and intersections:
                intersections.pop()
                reduced_measurement_ids.append(str(row["object_id"]))
                changed = True
                break
        if not changed:
            break

    while encoded_length() > max_context_chars:
        changed = False
        for row in reversed(result["clearance_requirements"]):
            if row.get("measurement") is not None:
                row["measurement"] = None
                omitted_measurement_ids.append(str(row["object_id"]))
                changed = True
                break
        if not changed:
            break

    while encoded_length() > max_context_chars and result["related_pairs"]:
        omitted = result["related_pairs"].pop()
        omitted_relation_refs.append(str(omitted["source_relation_id"]))

    while (
        encoded_length() > max_context_chars
        and result["clearance_requirements"]
    ):
        omitted = result["clearance_requirements"].pop()
        omitted_clearance_ids.append(str(omitted["object_id"]))

    attach_audit()
    if encoded_length() > max_context_chars:
        raise ValueError(
            "functional spatial context cannot fit its structural budget"
        )
    return result


def _validate_projection_audit(
    value: Any,
    *,
    known_object_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("functional spatial context requires projection_audit")
    expected = {
        "policy",
        "max_context_chars",
        "eligible_clearance_count",
        "emitted_clearance_count",
        "eligible_relation_count",
        "emitted_relation_count",
        "reduced_measurement_object_ids",
        "omitted_measurement_object_ids",
        "omitted_clearance_object_ids",
        "omitted_relation_source_ids",
        "decision_authority",
    }
    if set(value) != expected:
        raise ValueError("functional spatial projection audit is invalid")
    if (
        value.get("policy") != "bounded_attention_projection_v1"
        or value.get("decision_authority") != "none"
    ):
        raise ValueError("functional spatial projection policy is invalid")
    for field in (
        "max_context_chars",
        "eligible_clearance_count",
        "emitted_clearance_count",
        "eligible_relation_count",
        "emitted_relation_count",
    ):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError("functional spatial projection counts are invalid")
    normalized = deepcopy(value)
    for field in (
        "reduced_measurement_object_ids",
        "omitted_measurement_object_ids",
        "omitted_clearance_object_ids",
    ):
        ids = value.get(field)
        if (
            not isinstance(ids, list)
            or len(ids) != len(set(ids))
            or any(str(item) not in known_object_ids for item in ids)
        ):
            raise ValueError("functional spatial projection object IDs are invalid")
        normalized[field] = [str(item) for item in ids]
    refs = value.get("omitted_relation_source_ids")
    if (
        not isinstance(refs, list)
        or len(refs) != len(set(refs))
        or any(not isinstance(item, str) or not item.strip() for item in refs)
    ):
        raise ValueError("functional spatial omitted relation refs are invalid")
    return normalized


def _validate_clearance_measurement(
    value: Any,
    *,
    known_object_ids: set[str],
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("clearance measurement must be an object or null")
    allowed = {
        "status",
        "usable_side_id",
        "world_outward_direction_xy",
        "frontage_origin_xy",
        "corridor_depth_m",
        "corridor_half_width_m",
        "nearest_forward_obstacle_distance_m",
        "forward_intersections",
        "unavailable_reason",
        "decision_authority",
    }
    if set(value) != allowed:
        raise ValueError("clearance measurement has unsupported fields")
    status = str(value.get("status") or "")
    if status not in {"available", "unavailable"}:
        raise ValueError("clearance measurement status is invalid")
    if value.get("decision_authority") != "none":
        raise ValueError("clearance measurement has no decision authority")
    intersections = value.get("forward_intersections")
    if not isinstance(intersections, list) or len(intersections) > 8:
        raise ValueError("clearance measurement intersections are invalid")
    normalized_intersections: list[dict[str, Any]] = []
    required = {
        "object_id",
        "forward_near_distance_m",
        "forward_far_distance_m",
        "lateral_clearance_m",
        "corridor_overlap_depth_m",
        "corridor_overlap_width_m",
        "corridor_width_overlap_fraction",
        "corridor_overlap_area_proxy_m2",
        "vertical_overlap_with_approach_m",
        "vertical_relevant",
        "support_relation",
        "thin_floor_layer",
        "ordinary_mobility",
        "excluded_from_obstacle",
    }
    seen: set[str] = set()
    for item in intersections:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("clearance intersection has unsupported fields")
        object_id = str(item.get("object_id") or "").strip()
        if object_id not in known_object_ids or object_id in seen:
            raise ValueError("clearance intersection object ID is invalid")
        seen.add(object_id)
        if item.get("support_relation") not in {
            "none",
            "supported_by_target",
            "supports_target",
        }:
            raise ValueError("clearance support relation is invalid")
        if item.get("ordinary_mobility") not in {
            "fixed",
            "movable_companion",
            "portable_unrelated",
            "unspecified",
        }:
            raise ValueError("clearance mobility role is invalid")
        if (
            not isinstance(item.get("thin_floor_layer"), bool)
            or not isinstance(item.get("vertical_relevant"), bool)
            or not isinstance(item.get("excluded_from_obstacle"), bool)
        ):
            raise ValueError("clearance intersection flags are invalid")
        for key in (
            "forward_near_distance_m",
            "forward_far_distance_m",
            "lateral_clearance_m",
            "corridor_overlap_depth_m",
            "corridor_overlap_width_m",
            "corridor_width_overlap_fraction",
            "corridor_overlap_area_proxy_m2",
            "vertical_overlap_with_approach_m",
        ):
            if not isinstance(item.get(key), (int, float)):
                raise ValueError("clearance intersection distance is invalid")
        normalized_intersections.append(deepcopy(item))
    return {
        key: deepcopy(value.get(key))
        for key in (
            "status",
            "usable_side_id",
            "world_outward_direction_xy",
            "frontage_origin_xy",
            "corridor_depth_m",
            "corridor_half_width_m",
            "nearest_forward_obstacle_distance_m",
        )
    } | {
        "forward_intersections": normalized_intersections,
        "unavailable_reason": value.get("unavailable_reason"),
        "decision_authority": "none",
    }


__all__ = [
    "FUNCTIONAL_SPATIAL_CONTEXT_MAX_CLEARANCE_OBJECTS",
    "FUNCTIONAL_SPATIAL_CONTEXT_MAX_CHARS",
    "FUNCTIONAL_SPATIAL_CONTEXT_MAX_INTERSECTIONS",
    "FUNCTIONAL_SPATIAL_CONTEXT_MAX_RELATED_PAIRS",
    "FUNCTIONAL_SPATIAL_CONTEXT_VERSION",
    "project_functional_spatial_context",
    "validate_functional_spatial_context",
]
