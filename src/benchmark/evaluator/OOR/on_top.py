"""Target-specific ``on_top_of`` evidence with one-sided conservative routing.

Deterministic geometry is only allowed to certify an obvious valid relation.
Everything else, including clear-looking negative proxy evidence, is routed to
the semantic judge because OBB/support proxies cannot establish the absence of
an attachment, assembly convention, or representation mismatch.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np
from shapely.geometry import MultiPoint

from benchmark.evaluator.generic_validity.geometry import get_obb_corners
from benchmark.evaluator.generic_validity.support import (
    HARD_CONTACT_TOLERANCE_BASE_M,
    HARD_CONTACT_TOLERANCE_MAX_M,
    HARD_CONTACT_TOLERANCE_MIN_M,
    HARD_CONTACT_TOLERANCE_PER_SIZE_Z,
    NEAR_SUPPORT_TOLERANCE_BASE_M,
    NEAR_SUPPORT_TOLERANCE_MAX_M,
    NEAR_SUPPORT_TOLERANCE_MIN_M,
    NEAR_SUPPORT_TOLERANCE_PER_SIZE_Z,
)
from benchmark.scene_io.object_normalization import NormalizedObject


DEFAULT_ON_TOP_CONFIG = {
    "valid_subject_overlap_fraction": 0.50,
    "invalid_subject_overlap_fraction": 0.02,
    "maximum_direct_tilt_degrees": 20.0,
    "clear_vertical_reversal_margin_m": 0.02,
}

ON_TOP_DETECTOR_VERSION = "oor_on_top_of_v2"

ON_TOP_VLM_INSTRUCTION = (
    "Judge only whether the subject is on top of the claimed anchor. The detector is a "
    "high-recall evidence provider and routing carries no invalid prior. Use the signed "
    "vertical gap, subject-footprint overlap, downward first-support hits, geometry quality, "
    "prompt, object descriptions, and renders. A different supporting object does not satisfy "
    "the claim. Do not infer centimeter-scale geometry from pixels when detector measurements "
    "are available. Return exactly one binary valid/invalid verdict."
)


def check_on_top_of(
    subject: NormalizedObject,
    anchor: NormalizedObject,
    *,
    support_record: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_ON_TOP_CONFIG, **(config or {})}
    overlap_fraction = _subject_overlap_fraction(subject, anchor)
    signed_gap = float(subject.bottom_z - anchor.top_z)
    hard_tolerance = _hard_contact_tolerance(float(subject.size[2]))
    near_tolerance = _near_support_tolerance(float(subject.size[2]))
    subject_tilt = _tilt_degrees(subject)
    anchor_tilt = _tilt_degrees(anchor)

    support = support_record if isinstance(support_record, dict) else {}
    per_target = support.get("per_target_hit_counts")
    per_target = per_target if isinstance(per_target, dict) else {}
    anchor_hit_count = max(0, int(per_target.get(anchor.id) or 0))
    support_targets = (
        [str(value) for value in support.get("support_targets", []) if str(value).strip()]
        if isinstance(support.get("support_targets"), list)
        else []
    )
    other_support_targets = sorted(value for value in support_targets if value != anchor.id)
    geometry_degraded = bool(support.get("geometry_evidence_degraded"))
    evidence_level = str(support.get("evidence_level") or "obb")

    invalid_reasons: list[str] = []
    if overlap_fraction <= float(cfg["invalid_subject_overlap_fraction"]):
        invalid_reasons.append("projected_footprints_are_clearly_disjoint")
    if subject.top_z < anchor.bottom_z - float(cfg["clear_vertical_reversal_margin_m"]):
        invalid_reasons.append("subject_is_entirely_below_anchor")
    if signed_gap > near_tolerance:
        invalid_reasons.append("clear_positive_gap_above_claimed_anchor")
    if anchor_hit_count == 0 and other_support_targets and (
        overlap_fraction < float(cfg["valid_subject_overlap_fraction"])
        or abs(signed_gap) > hard_tolerance
    ):
        invalid_reasons.append("a_different_target_is_the_measured_support")

    direct_valid = (
        not invalid_reasons
        and not geometry_degraded
        and anchor_hit_count > 0
        and overlap_fraction >= float(cfg["valid_subject_overlap_fraction"])
        and -hard_tolerance <= signed_gap <= hard_tolerance
        and subject_tilt <= float(cfg["maximum_direct_tilt_degrees"])
        and anchor_tilt <= float(cfg["maximum_direct_tilt_degrees"])
    )

    if direct_valid:
        route = "direct_valid"
        status = "checked"
        passed: bool | None = True
        score: float | None = 1.0
    else:
        route = "requires_vlm"
        status = "requires_vlm"
        passed = None
        score = None

    detector_evidence = {
        "detector": ON_TOP_DETECTOR_VERSION,
        "instruction": ON_TOP_VLM_INSTRUCTION,
        "routing_has_invalid_prior": False,
        "route": route,
        "signed_vertical_gap_m": signed_gap,
        "hard_contact_tolerance_m": hard_tolerance,
        "near_support_tolerance_m": near_tolerance,
        "subject_footprint_overlap_fraction": overlap_fraction,
        "valid_overlap_threshold": float(cfg["valid_subject_overlap_fraction"]),
        "invalid_overlap_threshold": float(cfg["invalid_subject_overlap_fraction"]),
        "subject_tilt_degrees": subject_tilt,
        "anchor_tilt_degrees": anchor_tilt,
        "maximum_direct_tilt_degrees": float(cfg["maximum_direct_tilt_degrees"]),
        "claimed_anchor_first_support_hit_count": anchor_hit_count,
        "measured_support_targets": support_targets,
        "other_measured_support_targets": other_support_targets,
        "support_evidence_level": evidence_level,
        "support_geometry_degraded": geometry_degraded,
        "support_geometry_degraded_reasons": deepcopy(
            support.get("geometry_degraded_reasons") or []
        ),
        "representative_support_samples": deepcopy(
            support.get("representative_samples") or []
        ),
        # Keep the legacy key for report readers while making its role explicit:
        # these are candidate reasons, never a deterministic invalid verdict.
        "direct_invalid_reasons": invalid_reasons,
        "candidate_invalid_reasons": invalid_reasons,
        "subject": _compact_object(subject),
        "claimed_anchor": _compact_object(anchor),
    }
    return {
        "relation": "on_top_of",
        "category": "target_support",
        "subject_id": subject.id,
        "object_id": anchor.id,
        "passed": passed,
        "score": score,
        "status": status,
        "backend": "deterministic" if status == "checked" else "deterministic_router",
        "route": route,
        "evidence": detector_evidence,
    }


def _subject_overlap_fraction(subject: NormalizedObject, anchor: NormalizedObject) -> float:
    subject_polygon = MultiPoint(np.asarray(get_obb_corners(subject))[:, :2]).convex_hull
    anchor_polygon = MultiPoint(np.asarray(get_obb_corners(anchor))[:, :2]).convex_hull
    if subject_polygon.is_empty or anchor_polygon.is_empty or subject_polygon.area <= 1.0e-12:
        return 0.0
    return float(subject_polygon.intersection(anchor_polygon).area / subject_polygon.area)


def _tilt_degrees(obj: NormalizedObject) -> float:
    cosine = float(np.clip(np.dot(obj.up, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    return float(math.degrees(math.acos(abs(cosine))))


def _hard_contact_tolerance(size_z_m: float) -> float:
    value = HARD_CONTACT_TOLERANCE_BASE_M + HARD_CONTACT_TOLERANCE_PER_SIZE_Z * size_z_m
    return max(HARD_CONTACT_TOLERANCE_MIN_M, min(HARD_CONTACT_TOLERANCE_MAX_M, value))


def _near_support_tolerance(size_z_m: float) -> float:
    value = NEAR_SUPPORT_TOLERANCE_BASE_M + NEAR_SUPPORT_TOLERANCE_PER_SIZE_Z * size_z_m
    return max(NEAR_SUPPORT_TOLERANCE_MIN_M, min(NEAR_SUPPORT_TOLERANCE_MAX_M, value))


def _compact_object(obj: NormalizedObject) -> dict[str, Any]:
    return {
        "id": obj.id,
        "category": obj.category,
        "description": obj.desc,
        "center": [float(value) for value in obj.center],
        "size": [float(value) for value in obj.size],
        "rotation_degrees": [float(value) for value in obj.rotation],
    }
