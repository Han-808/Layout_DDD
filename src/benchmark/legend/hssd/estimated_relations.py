"""Legend HSSD relation-cue heuristics.

These helpers only support the legacy HSSD input adapter. They estimate soft
spatial cues from imported bbox metadata and do not define the current natural
language input contract.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


DEFAULT_MAX_NEAR_CUES_TOTAL = 80
DEFAULT_MAX_NEAR_CUES_PER_OBJECT = 3
DEFAULT_MAX_SUPPORT_CANDIDATES_PER_CHILD = 2
DEFAULT_BASE_NEAR_THRESHOLD_M = 2.5
DEFAULT_NEAR_SCALE_FACTOR = 1.5
DEFAULT_SUPPORT_VERTICAL_TOLERANCE_M = 0.08
DEFAULT_SUPPORT_MIN_OVERLAP_RATIO = 0.15


def build_estimated_spatial_cues(
    objects: list[dict],
    *,
    max_near_cues_total: int = DEFAULT_MAX_NEAR_CUES_TOTAL,
    max_near_cues_per_object: int = DEFAULT_MAX_NEAR_CUES_PER_OBJECT,
    max_support_candidates_per_child: int = DEFAULT_MAX_SUPPORT_CANDIDATES_PER_CHILD,
    base_near_threshold_m: float = DEFAULT_BASE_NEAR_THRESHOLD_M,
    near_scale_factor: float = DEFAULT_NEAR_SCALE_FACTOR,
    support_vertical_tolerance_m: float = DEFAULT_SUPPORT_VERTICAL_TOLERANCE_M,
    support_min_overlap_ratio: float = DEFAULT_SUPPORT_MIN_OVERLAP_RATIO,
) -> list[dict]:
    """Generate deterministic soft spatial cues from HSSD-derived bbox metadata.

    These records are heuristic evidence, not HSSD ground-truth relations.
    """

    positioned = [obj for obj in objects if isinstance(obj, dict) and _floor_center(obj)]
    positioned.sort(key=lambda item: str(item.get("id") or ""))
    support_cues = _support_candidate_cues(
        positioned,
        max_per_child=max_support_candidates_per_child,
        vertical_tolerance=support_vertical_tolerance_m,
        min_overlap_ratio=support_min_overlap_ratio,
    )
    candidates = []
    for index, first in enumerate(positioned):
        for second in positioned[index + 1 :]:
            first_center = _floor_center(first)
            second_center = _floor_center(second)
            if first_center is None or second_center is None:
                continue
            distance = math.dist(first_center, second_center)
            first_diag = _footprint_diag(first)
            second_diag = _footprint_diag(second)
            scale = max(first_diag, second_diag, 1.0e-6)
            threshold = max(base_near_threshold_m, near_scale_factor * scale)
            if distance > threshold:
                continue
            normalized = distance / scale
            candidates.append((distance, normalized, str(first["id"]), str(second["id"]), threshold))

    candidates.sort(key=lambda item: (item[0], item[2], item[3]))
    counts: dict[str, int] = defaultdict(int)
    cues = list(support_cues)
    for distance, normalized, subject, target, threshold in candidates:
        if len(cues) >= max_near_cues_total:
            break
        if counts[subject] >= max_near_cues_per_object or counts[target] >= max_near_cues_per_object:
            continue
        counts[subject] += 1
        counts[target] += 1
        confidence = max(0.05, min(1.0, 1.0 - (distance / max(threshold, 1.0e-6)) * 0.5))
        cues.append(
            {
                "id": f"near__{subject}__{target}",
                "relation_id": f"near__{subject}__{target}",
                "type": "near",
                "subject": subject,
                "object": target,
                "target": target,
                "source": "bbox_geometry_heuristic",
                "provenance": "bbox_geometry_heuristic",
                "confidence": round(confidence, 4),
                "hard": False,
                "visible_to_model": True,
                "evidence": {
                    "horizontal_distance": round(distance, 4),
                    "normalized_distance": round(normalized, 4),
                    "threshold_m": round(threshold, 4),
                },
            }
        )
    return cues


def cue_counts_by_type(cues: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cue in cues:
        cue_type = str(cue.get("type") or "unknown") if isinstance(cue, dict) else "unknown"
        counts[cue_type] = counts.get(cue_type, 0) + 1
    return dict(sorted(counts.items()))


def relation_policy_metadata(cues: list[dict]) -> dict:
    return {
        "relation_policy": "deterministic_estimated_spatial_cues_v1",
        "relation_generation_version": "deterministic_spatial_cues_v1",
        "relation_counts_by_type": cue_counts_by_type(cues),
        "relations_are_ground_truth": False,
        "relations_source_note": "Generated from bbox/source geometry heuristics; HSSD-HAB does not provide benchmark-ready relation ground truth.",
    }


def compatibility_relations(cues: list[dict]) -> list[dict]:
    """Return legacy relation records for relation-aware evaluators/prompts."""

    return [cue for cue in cues if isinstance(cue, dict) and cue.get("type") in {"near", "left_of", "right_of", "in_front_of", "behind", "facing", "against_wall"}]


def _support_candidate_cues(
    objects: list[dict],
    *,
    max_per_child: int,
    vertical_tolerance: float,
    min_overlap_ratio: float,
) -> list[dict]:
    candidates_by_child: dict[str, list[tuple[float, float, str, dict]]] = defaultdict(list)
    for child in objects:
        child_id = str(child.get("id") or "")
        child_size = _bbox_size(child)
        child_center = _layout_center(child)
        child_box = _footprint_box(child)
        if not child_id or child_size is None or child_center is None or child_box is None:
            continue
        child_bottom = child_center[2] - child_size[2] / 2.0
        child_area = max(1.0e-6, child_size[0] * child_size[1])
        for parent in objects:
            parent_id = str(parent.get("id") or "")
            if not parent_id or parent_id == child_id:
                continue
            parent_size = _bbox_size(parent)
            parent_center = _layout_center(parent)
            parent_box = _footprint_box(parent)
            if parent_size is None or parent_center is None or parent_box is None:
                continue
            parent_top = parent_center[2] + parent_size[2] / 2.0
            vertical_gap = abs(child_bottom - parent_top)
            if vertical_gap > vertical_tolerance:
                continue
            overlap = _overlap_area(child_box, parent_box)
            overlap_ratio = overlap / child_area
            if overlap_ratio < min_overlap_ratio:
                continue
            parent_area = parent_size[0] * parent_size[1]
            if parent_area + 1.0e-9 < child_area:
                continue
            confidence = max(0.05, min(1.0, 0.55 + 0.35 * min(1.0, overlap_ratio) + 0.10 * (1.0 - vertical_gap / max(vertical_tolerance, 1.0e-6))))
            cue = {
                "id": f"support_candidate__{child_id}__{parent_id}",
                "relation_id": f"support_candidate__{child_id}__{parent_id}",
                "type": "support_candidate",
                "subject": child_id,
                "object": parent_id,
                "target": parent_id,
                "source": "bbox_geometry_heuristic",
                "provenance": "bbox_geometry_heuristic",
                "confidence": round(confidence, 4),
                "hard": False,
                "visible_to_model": True,
                "evidence": {
                    "vertical_gap": round(vertical_gap, 4),
                    "footprint_overlap_ratio": round(overlap_ratio, 4),
                },
            }
            candidates_by_child[child_id].append((vertical_gap, -overlap_ratio, parent_id, cue))
    cues = []
    for child_id in sorted(candidates_by_child):
        ranked = sorted(candidates_by_child[child_id], key=lambda item: (item[0], item[1], item[2]))
        cues.extend(item[3] for item in ranked[:max_per_child])
    return cues


def _floor_center(obj: dict) -> tuple[float, float] | None:
    floor = obj.get("source_floor_position")
    if isinstance(floor, list) and len(floor) >= 2:
        try:
            return float(floor[0]), float(floor[1])
        except (TypeError, ValueError):
            return None
    source = obj.get("source_position")
    if isinstance(source, list) and len(source) >= 3:
        try:
            return float(source[0]), float(source[2])
        except (TypeError, ValueError):
            return None
    return None


def _layout_center(obj: dict) -> tuple[float, float, float] | None:
    hint = obj.get("layout_center_hint")
    if isinstance(hint, list) and len(hint) >= 3:
        try:
            return float(hint[0]), float(hint[1]), float(hint[2])
        except (TypeError, ValueError):
            return None
    floor = _floor_center(obj)
    size = _bbox_size(obj)
    if floor is None or size is None:
        return None
    return floor[0], floor[1], size[2] / 2.0


def _bbox_size(obj: dict) -> tuple[float, float, float] | None:
    size = obj.get("bbox_size")
    if not isinstance(size, list) or len(size) < 3:
        return None
    try:
        width, depth, height = float(size[0]), float(size[1]), float(size[2])
    except (TypeError, ValueError):
        return None
    if width <= 0 or depth <= 0 or height <= 0:
        return None
    return width, depth, height


def _footprint_box(obj: dict) -> tuple[float, float, float, float] | None:
    center = _layout_center(obj)
    size = _bbox_size(obj)
    if center is None or size is None:
        return None
    return center[0] - size[0] / 2.0, center[0] + size[0] / 2.0, center[1] - size[1] / 2.0, center[1] + size[1] / 2.0


def _overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    overlap_x = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    overlap_y = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return overlap_x * overlap_y


def _footprint_diag(obj: dict) -> float:
    size = obj.get("bbox_size")
    if not isinstance(size, list) or len(size) < 2:
        return 1.0
    try:
        return max(1.0e-6, math.hypot(float(size[0]), float(size[1])))
    except (TypeError, ValueError):
        return 1.0
