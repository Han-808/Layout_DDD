from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re
from statistics import median
from typing import Any

from benchmark.grouping.interfaces import (
    GroupingRequest,
    GroupingResult,
)
from benchmark.grouping.scene import (
    NormalizedGroupingScene,
    floor_box,
    floor_gap,
    footprint_area,
    normalize_grouping_scene,
    scene_diagonal,
)


ANCHOR_GROUPING_POLICY_ID = "anchor_object_partition_v1"

DEFAULT_ANCHOR_GROUPING_CONFIG: dict[str, Any] = {
    "target_objects_per_group": 6,
    "max_anchors": 12,
    "max_objects_per_group": 12,
    "min_anchor_score": 0.70,
    "strong_semantic_anchor_score": 1.05,
    "min_anchor_separation_m": 0.75,
    "max_assignment_gap_m": 3.0,
    "max_assignment_gap_ratio": 0.30,
    "semantic_gap_multiplier": 1.20,
    "semantic_match_bonus": 0.12,
    "same_region_bonus": 0.15,
    "region_mismatch_penalty": 0.35,
    "support_link_bonus": 1.00,
    "support_vertical_tolerance_m": 0.10,
    "support_min_xy_overlap_ratio": 0.15,
}

_ANCHOR_CUES: tuple[tuple[str, str, float], ...] = (
    ("daybed", "seating", 0.95),
    ("dining table", "dining", 1.00),
    ("kitchen island", "kitchen", 1.00),
    ("workstation", "work", 0.95),
    ("office desk", "work", 0.95),
    ("writing desk", "work", 0.90),
    ("coffee table", "seating", 0.45),
    ("side table", "seating", 0.30),
    ("nightstand", "sleep", 0.25),
    ("bedside table", "sleep", 0.25),
    ("media console", "media", 0.80),
    ("tv stand", "media", 0.80),
    ("television stand", "media", 0.80),
    ("tv cabinet", "media", 0.80),
    ("television cabinet", "media", 0.80),
    ("refrigerator", "kitchen", 0.90),
    ("fridge", "kitchen", 0.90),
    ("kitchen counter", "kitchen", 0.85),
    ("stove", "kitchen", 0.80),
    ("bed", "sleep", 1.00),
    ("sofa", "seating", 1.00),
    ("couch", "seating", 1.00),
    ("sectional", "seating", 1.00),
    ("desk", "work", 0.90),
    ("table", "table", 0.70),
    ("island", "kitchen", 0.90),
    ("bathtub", "bath", 0.95),
    ("vanity", "bath", 0.80),
    ("toilet", "bath", 0.70),
    ("bookshelf", "storage", 0.70),
    ("bookcase", "storage", 0.70),
    ("wardrobe", "storage", 0.70),
    ("dresser", "storage", 0.65),
    ("hutch", "storage", 0.65),
    ("chest", "storage", 0.55),
    ("cabinet", "storage", 0.55),
    ("armchair", "seating", 0.65),
    ("bench", "seating", 0.55),
)

_ACCESSORY_CUES = (
    "rug",
    "carpet",
    "lamp",
    "light",
    "pillow",
    "cushion",
    "curtain",
    "picture",
    "painting",
    "mirror",
    "plant",
    "vase",
    "decor",
    "ornament",
    "clock",
    "chair",
    "stool",
    "nightstand",
    "side table",
    "suitcase",
)

_PREFERRED_FAMILY_CUES: tuple[
    tuple[str, tuple[str, ...]], ...
] = (
    ("nightstand", ("sleep",)),
    ("bedside", ("sleep",)),
    ("pillow", ("sleep", "seating")),
    ("headboard", ("sleep",)),
    ("office chair", ("work",)),
    ("desk chair", ("work",)),
    ("monitor", ("work",)),
    ("computer", ("work",)),
    ("printer", ("work",)),
    ("dining chair", ("dining",)),
    ("bar stool", ("kitchen", "dining")),
    ("stool", ("kitchen", "dining", "work")),
    ("chair", ("seating", "dining", "work")),
    ("coffee table", ("seating",)),
    ("side table", ("seating", "sleep")),
    ("ottoman", ("seating",)),
    ("rug", ("seating", "dining")),
    ("carpet", ("seating", "dining")),
    ("floor lamp", ("seating", "work", "sleep")),
    ("television", ("media", "seating")),
    (" tv ", ("media", "seating")),
    ("speaker", ("media", "seating")),
    ("wardrobe", ("sleep", "storage")),
    ("dresser", ("sleep", "storage")),
    ("suitcase", ("sleep", "storage")),
    ("bookshelf", ("storage", "work")),
    ("bookcase", ("storage", "work")),
    ("cabinet", ("storage", "kitchen", "bath")),
    ("refrigerator", ("kitchen",)),
    ("fridge", ("kitchen",)),
    ("oven", ("kitchen",)),
    ("stove", ("kitchen",)),
    ("sink", ("kitchen", "bath")),
    ("toilet", ("bath",)),
    ("shower", ("bath",)),
    ("towel", ("bath",)),
)


@dataclass(frozen=True)
class _AnchorCandidate:
    object_id: str
    source_index: int
    family: str | None
    score: float
    semantic_weight: float
    area_ratio: float
    explicit: bool
    accessory: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "source_index": self.source_index,
            "family": self.family,
            "ranking_score": _round(self.score),
            "score_role": "backend_anchor_ranking_only",
            "semantic_weight": _round(self.semantic_weight),
            "area_ratio": _round(self.area_ratio),
            "explicit": self.explicit,
            "accessory": self.accessory,
            "reason_codes": list(self.reason_codes),
        }


class AnchorGroupingAlgorithm:
    """Deterministic anchor-centered evidence partition.

    The algorithm identifies stable local anchors, then assigns other objects
    using explicit support, semantic family affinity, region consistency, and
    scale-normalized floor distance. An object with no defensible assignment is
    kept as a singleton instead of being forced into a functional claim.
    """

    backend = "anchor"
    policy_id = ANCHOR_GROUPING_POLICY_ID

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is not None and not isinstance(config, dict):
            raise TypeError("anchor grouping config must be a JSON object")
        self.config = deepcopy(config or {})

    def group(self, request: GroupingRequest) -> GroupingResult:
        request = GroupingRequest.from_value(request)
        scene = normalize_grouping_scene(request.scene)
        config = _effective_anchor_config(self.config, request.config)
        if not scene.objects:
            return GroupingResult.create(
                groups=[],
                expected_object_ids=(),
                backend=self.backend,
                policy_id=self.policy_id,
                reason="The scene has no renderable objects to partition.",
                provenance={
                    **scene.provenance(),
                    "deterministic": True,
                    "model_calls": 0,
                    "anchor_candidates": [],
                    "assignments": [],
                },
                resolved_grouping_config=config,
                object_catalog=[],
            )

        by_id = {
            str(item["object_id"]): item for item in scene.objects
        }
        children = _support_child_counts(scene)
        areas = [footprint_area(item) for item in scene.objects]
        median_area = max(1.0e-9, float(median(areas)))
        candidates = [
            _candidate(item, median_area=median_area, support_children=children)
            for item in scene.objects
        ]
        accepted, rejected = _select_anchors(
            candidates,
            by_id=by_id,
            object_count=len(scene.objects),
            config=config,
        )
        groups, assignments = _assign_to_anchors(
            scene,
            accepted=accepted,
            by_id=by_id,
            config=config,
        )
        return GroupingResult.create(
            groups=groups,
            expected_object_ids=scene.object_ids,
            backend=self.backend,
            policy_id=self.policy_id,
            reason=(
                "Deterministic anchor selection followed by support-, "
                "semantic-family-, region-, and distance-aware assignment; "
                "unmatched objects remain singleton evidence scopes."
            ),
            provenance={
                **scene.provenance(),
                "deterministic": True,
                "model_calls": 0,
                "anchor_candidates": [
                    item.to_dict() for item in candidates
                ],
                "selected_anchor_ids": [
                    item.object_id for item in accepted
                ],
                "rejected_anchor_candidates": rejected,
                "assignments": assignments,
            },
            resolved_grouping_config=config,
            object_catalog=scene.object_catalog(),
        )


def _candidate(
    item: dict[str, Any],
    *,
    median_area: float,
    support_children: dict[str, int],
) -> _AnchorCandidate:
    description = _description(item)
    family, semantic_weight = _anchor_semantics(description)
    accessory = _matches_any(description, _ACCESSORY_CUES)
    area_ratio = footprint_area(item) / median_area
    area_component = 0.45 * min(
        1.5,
        math.log1p(max(0.0, area_ratio)) / math.log(4.0),
    )
    support_component = min(
        0.40,
        0.12 * support_children.get(str(item["object_id"]), 0),
    )
    explicit = item.get("is_anchor") is True
    score = (
        area_component
        + 0.85 * semantic_weight
        + support_component
        + (2.0 if explicit else 0.0)
        - (0.85 if accessory and not explicit else 0.0)
    )
    reasons = []
    if explicit:
        reasons.append("explicit_anchor_metadata")
    if family is not None:
        reasons.append(f"semantic_anchor_family:{family}")
    if area_ratio >= 1.5:
        reasons.append("large_local_footprint")
    if support_component > 0.0:
        reasons.append("explicit_support_children")
    if accessory:
        reasons.append("accessory_penalty")
    return _AnchorCandidate(
        object_id=str(item["object_id"]),
        source_index=int(item["source_index"]),
        family=family,
        score=score,
        semantic_weight=semantic_weight,
        area_ratio=area_ratio,
        explicit=explicit,
        accessory=accessory,
        reason_codes=tuple(reasons),
    )


def _select_anchors(
    candidates: list[_AnchorCandidate],
    *,
    by_id: dict[str, dict[str, Any]],
    object_count: int,
    config: dict[str, Any],
) -> tuple[list[_AnchorCandidate], list[dict[str, Any]]]:
    target = max(
        1,
        math.ceil(object_count / config["target_objects_per_group"]),
    )
    target = min(target, config["max_anchors"])
    ordered = sorted(
        candidates,
        key=lambda item: (
            not item.explicit,
            -item.score,
            -item.area_ratio,
            item.source_index,
        ),
    )
    accepted: list[_AnchorCandidate] = []
    rejected: list[dict[str, Any]] = []

    primary = [
        item
        for item in ordered
        if item.explicit or item.score >= config["min_anchor_score"]
    ]
    for item in primary:
        reason = _anchor_rejection_reason(
            item,
            accepted=accepted,
            by_id=by_id,
            config=config,
            target=target,
        )
        if reason is not None:
            rejected.append(
                {**item.to_dict(), "rejected_reason": reason}
            )
            continue
        accepted.append(item)

    if len(accepted) < target:
        for item in ordered:
            if item in accepted or item.accessory:
                continue
            if any(
                record["object_id"] == item.object_id
                for record in rejected
            ):
                continue
            reason = _near_duplicate_anchor(
                item,
                accepted=accepted,
                by_id=by_id,
                config=config,
            )
            if reason is not None:
                rejected.append(
                    {**item.to_dict(), "rejected_reason": reason}
                )
                continue
            accepted.append(item)
            if len(accepted) >= target:
                break

    if not accepted:
        fallback = max(
            candidates,
            key=lambda item: (item.area_ratio, -item.source_index),
        )
        accepted.append(fallback)
    accepted.sort(key=lambda item: item.source_index)
    return accepted, rejected


def _anchor_rejection_reason(
    item: _AnchorCandidate,
    *,
    accepted: list[_AnchorCandidate],
    by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    target: int,
) -> str | None:
    if len(accepted) >= config["max_anchors"]:
        return "max_anchors"
    duplicate = _near_duplicate_anchor(
        item,
        accepted=accepted,
        by_id=by_id,
        config=config,
    )
    if duplicate is not None:
        return duplicate
    if (
        len(accepted) >= target
        and not item.explicit
        and item.score < config["strong_semantic_anchor_score"]
    ):
        return "target_anchor_count_reached"
    return None


def _near_duplicate_anchor(
    item: _AnchorCandidate,
    *,
    accepted: list[_AnchorCandidate],
    by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> str | None:
    for current in accepted:
        if item.family != current.family:
            continue
        if floor_gap(
            by_id[item.object_id],
            by_id[current.object_id],
        ) < config["min_anchor_separation_m"]:
            return f"near_same_family_anchor:{current.object_id}"
    return None


def _assign_to_anchors(
    scene: NormalizedGroupingScene,
    *,
    accepted: list[_AnchorCandidate],
    by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_by_id = {item.object_id: item for item in accepted}
    members: dict[str, list[str]] = {
        item.object_id: [item.object_id] for item in accepted
    }
    object_to_group = {
        item.object_id: item.object_id for item in accepted
    }
    assignments: list[dict[str, Any]] = [
        {
            "object_id": item.object_id,
            "anchor_object_id": item.object_id,
            "reason_codes": ["selected_anchor"],
            "floor_gap_m": 0.0,
            "assignment_cost": 0.0,
        }
        for item in accepted
    ]
    diagonal = scene_diagonal(scene)
    max_gap = max(
        config["max_assignment_gap_m"],
        diagonal * config["max_assignment_gap_ratio"],
    )
    nonanchors = [
        item
        for item in scene.objects
        if str(item["object_id"]) not in anchor_by_id
    ]
    nonanchors.sort(
        key=lambda item: (
            _support_depth(item, by_id),
            int(item["source_index"]),
        )
    )
    for item in nonanchors:
        object_id = str(item["object_id"])
        parent_id = str(item.get("support_parent") or "")
        if parent_id in object_to_group:
            group_key = object_to_group[parent_id]
            members[group_key].append(object_id)
            object_to_group[object_id] = group_key
            assignments.append(
                {
                    "object_id": object_id,
                    "anchor_object_id": (
                        group_key if group_key in anchor_by_id else None
                    ),
                    "reason_codes": ["support_parent"],
                    "support_parent": parent_id,
                    "floor_gap_m": _round(
                        floor_gap(item, by_id[parent_id])
                    ),
                    "assignment_cost": -config["support_link_bonus"],
                }
            )
            continue

        best: tuple[
            float,
            int,
            _AnchorCandidate,
            float,
            list[str],
        ] | None = None
        preferred = _preferred_families(_description(item))
        for anchor in accepted:
            anchor_item = by_id[anchor.object_id]
            gap = floor_gap(item, anchor_item)
            semantic_match = anchor.family in preferred
            support_contact = _supported_by(item, anchor_item, config)
            allowed_gap = max_gap * (
                config["semantic_gap_multiplier"]
                if semantic_match
                else 1.0
            )
            if gap > allowed_gap and not support_contact:
                continue
            if len(members[anchor.object_id]) >= config[
                "max_objects_per_group"
            ]:
                continue
            cost = gap / max(1.0e-9, max_gap)
            reason_codes = ["nearest_anchor"]
            if semantic_match:
                cost -= config["semantic_match_bonus"]
                reason_codes.append(
                    f"semantic_family_match:{anchor.family}"
                )
            item_region = item.get("region_id")
            anchor_region = anchor_item.get("region_id")
            if item_region and anchor_region:
                if item_region == anchor_region:
                    cost -= config["same_region_bonus"]
                    reason_codes.append("same_region")
                else:
                    cost += config["region_mismatch_penalty"]
                    reason_codes.append("different_region_penalty")
            if support_contact:
                cost -= config["support_link_bonus"]
                reason_codes.append("derived_support_contact")
            key = (
                cost,
                anchor.source_index,
                anchor,
                gap,
                reason_codes,
            )
            if best is None or key[:2] < best[:2]:
                best = key

        if best is None:
            group_key = f"singleton:{object_id}"
            members[group_key] = [object_id]
            object_to_group[object_id] = group_key
            assignments.append(
                {
                    "object_id": object_id,
                    "anchor_object_id": None,
                    "reason_codes": [
                        "no_feasible_anchor",
                        "singleton_fail_closed",
                    ],
                    "floor_gap_m": None,
                    "assignment_cost": None,
                }
            )
            continue

        cost, _, anchor, gap, reason_codes = best
        members[anchor.object_id].append(object_id)
        object_to_group[object_id] = anchor.object_id
        assignments.append(
            {
                "object_id": object_id,
                "anchor_object_id": anchor.object_id,
                "reason_codes": reason_codes,
                "floor_gap_m": _round(gap),
                "assignment_cost": _round(cost),
            }
        )

    order = {
        str(item["object_id"]): int(item["source_index"])
        for item in scene.objects
    }
    assignment_by_object = {
        item["object_id"]: item for item in assignments
    }
    group_records: list[dict[str, Any]] = []
    for group_key, object_ids in members.items():
        object_ids.sort(key=order.__getitem__)
        anchor = anchor_by_id.get(group_key)
        group_objects = [by_id[object_id] for object_id in object_ids]
        if anchor is None:
            only = group_objects[0]
            group_records.append(
                {
                    "group_source": "anchor_singleton",
                    "object_ids": object_ids,
                    "label": (
                        "standalone context: "
                        + str(only["description"])[:80]
                    ),
                    "anchor_object_id": None,
                    "reason": (
                        "No configured anchor was sufficiently close or "
                        "compatible; singleton scope preserves the partition "
                        "without guessing."
                    ),
                    "group_footprint_diameter_m": _round(
                        _group_diameter(group_objects)
                    ),
                    "assignments": [
                        deepcopy(assignment_by_object[object_ids[0]])
                    ],
                }
            )
            continue
        anchor_item = by_id[anchor.object_id]
        family = anchor.family or "local"
        group_records.append(
            {
                "group_source": "anchor_object",
                "object_ids": object_ids,
                "label": (
                    f"{family} context around "
                    f"{str(anchor_item['description'])[:80]}"
                ),
                "anchor_object_id": anchor.object_id,
                "reason": (
                    "Objects share a deterministic anchor-centered evidence "
                    "scope; this is not a functional-validity judgement."
                ),
                "anchor_family": anchor.family,
                "anchor_ranking_score": _round(anchor.score),
                "anchor_ranking_score_role": (
                    "backend_selection_feature_not_metric_score"
                ),
                "group_footprint_diameter_m": _round(
                    _group_diameter(group_objects)
                ),
                "assignments": [
                    deepcopy(assignment_by_object[object_id])
                    for object_id in object_ids
                ],
            }
        )
    return group_records, assignments


def _effective_anchor_config(
    configured: dict[str, Any],
    request_config: dict[str, Any],
) -> dict[str, Any]:
    combined = _deep_merge(configured, request_config)
    section = combined.get("grouping")
    if isinstance(section, dict):
        combined = deepcopy(section)
    anchor = combined.get("anchor")
    if isinstance(anchor, dict):
        patch = deepcopy(anchor)
    else:
        patch = {
            key: deepcopy(value)
            for key, value in combined.items()
            if key not in {"backend", "topology", "vlm"}
        }
    unknown = sorted(set(patch) - set(DEFAULT_ANCHOR_GROUPING_CONFIG))
    if unknown:
        raise ValueError(
            f"unknown anchor grouping config fields {unknown}"
        )
    result = _deep_merge(DEFAULT_ANCHOR_GROUPING_CONFIG, patch)
    for key in (
        "target_objects_per_group",
        "max_anchors",
        "max_objects_per_group",
    ):
        result[key] = _positive_int(result[key], key)
    for key in (
        "min_anchor_score",
        "strong_semantic_anchor_score",
        "min_anchor_separation_m",
        "max_assignment_gap_m",
        "max_assignment_gap_ratio",
        "semantic_match_bonus",
        "same_region_bonus",
        "region_mismatch_penalty",
        "support_link_bonus",
        "support_vertical_tolerance_m",
        "support_min_xy_overlap_ratio",
    ):
        result[key] = _nonnegative_float(result[key], key)
    result["semantic_gap_multiplier"] = _positive_float(
        result["semantic_gap_multiplier"],
        "semantic_gap_multiplier",
    )
    if result["support_min_xy_overlap_ratio"] > 1.0:
        raise ValueError(
            "support_min_xy_overlap_ratio must be between 0 and 1"
        )
    return result


def _support_child_counts(
    scene: NormalizedGroupingScene,
) -> dict[str, int]:
    ids = set(scene.object_ids)
    counts = {object_id: 0 for object_id in scene.object_ids}
    for item in scene.objects:
        parent = str(item.get("support_parent") or "")
        if parent in ids:
            counts[parent] += 1
    return counts


def _support_depth(
    item: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> int:
    depth = 0
    seen = {str(item["object_id"])}
    parent = str(item.get("support_parent") or "")
    while parent in by_id and parent not in seen:
        seen.add(parent)
        depth += 1
        parent = str(by_id[parent].get("support_parent") or "")
    return depth


def _supported_by(
    child: dict[str, Any],
    parent: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    child_bottom = float(child["center"][2]) - float(child["size"][2]) / 2.0
    parent_top = float(parent["center"][2]) + float(parent["size"][2]) / 2.0
    if (
        abs(child_bottom - parent_top)
        > config["support_vertical_tolerance_m"]
    ):
        return False
    overlap = _overlap_area(floor_box(child), floor_box(parent))
    minimum_area = max(
        1.0e-9,
        min(footprint_area(child), footprint_area(parent)),
    )
    return (
        overlap / minimum_area
        >= config["support_min_xy_overlap_ratio"]
    )


def _overlap_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    overlap_x = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    overlap_y = max(0.0, min(left[3], right[3]) - max(left[2], right[2]))
    return overlap_x * overlap_y


def _group_diameter(objects: list[dict[str, Any]]) -> float:
    boxes = [floor_box(item) for item in objects]
    return math.hypot(
        max(box[1] for box in boxes) - min(box[0] for box in boxes),
        max(box[3] for box in boxes) - min(box[2] for box in boxes),
    )


def _description(item: dict[str, Any]) -> str:
    return (
        f"{item.get('category') or ''} "
        f"{item.get('description') or ''}"
    ).strip().lower()


def _anchor_semantics(description: str) -> tuple[str | None, float]:
    for phrase, family, weight in _ANCHOR_CUES:
        if _phrase_match(description, phrase):
            return family, weight
    return None, 0.0


def _preferred_families(description: str) -> set[str]:
    result: set[str] = set()
    family, _ = _anchor_semantics(description)
    if family is not None:
        result.add(family)
    for phrase, families in _PREFERRED_FAMILY_CUES:
        if phrase == " tv ":
            if " tv " in f" {description} ":
                result.update(families)
        elif _phrase_match(description, phrase):
            result.update(families)
    return result


def _matches_any(description: str, phrases: tuple[str, ...]) -> bool:
    return any(_phrase_match(description, phrase) for phrase in phrases)


def _phrase_match(description: str, phrase: str) -> bool:
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
            description,
        )
    )


def _deep_merge(
    base: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _nonnegative_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _round(value: float) -> float:
    return round(float(value), 6)
