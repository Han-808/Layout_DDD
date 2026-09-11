"""Deterministic usable-side-forward clearance measurements.

The measurements in this module are deliberately non-decisional.  They
describe which oriented object footprints intersect a bounded approach
corridor in front of a trusted usable-side hypothesis.  A Judge still decides
whether the measured free space is adequate for the ordinary use in question.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Iterable


FUNCTIONAL_DIRECTIONAL_CLEARANCE_VERSION = (
    "functional_directional_clearance_v1"
)
_LOCAL_SIDE_AXES = {
    "local_pos_x": (1.0, 0.0),
    "local_neg_x": (-1.0, 0.0),
    "local_pos_y": (0.0, 1.0),
    "local_neg_y": (0.0, -1.0),
}
_DEFAULT_CORRIDOR_DEPTH_M = 1.2
_MIN_CORRIDOR_HALF_WIDTH_M = 0.3
_MAX_CORRIDOR_HALF_WIDTH_M = 1.2
_LATERAL_MARGIN_M = 0.15
_THIN_FLOOR_LAYER_HEIGHT_M = 0.05
_SUPPORT_CONTACT_TOLERANCE_M = 0.04
_ORDINARY_APPROACH_HEIGHT_M = 1.8


def build_directional_clearance_extensions(
    *,
    scene: dict[str, Any] | None,
    discovery: dict[str, Any],
    functional_check_ledger: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build one bank extension for every accepted clearance check.

    Missing or ambiguous usable-side hypotheses produce an explicit
    unavailable extension rather than removing the check or failing the
    remaining bank.
    """

    hypotheses = {
        str(item.get("target_id") or ""): item.get(
            "precomputed_usable_surface_hypothesis"
        )
        for item in discovery.get("directed_surface_targets") or []
        if isinstance(item, dict) and item.get("target_id")
    }
    relations = [
        deepcopy(item)
        for field in ("within_group_correspondences", "cross_group_correspondences")
        for item in discovery.get(field) or []
        if isinstance(item, dict)
    ]
    admission_audit = discovery.get("relation_admission_audit")
    if isinstance(admission_audit, dict):
        relations.extend(
            deepcopy(item)
            for item in admission_audit.get("context_only_relations") or []
            if isinstance(item, dict)
        )
    result: dict[str, dict[str, Any]] = {}
    for check in functional_check_ledger.get("checks") or []:
        if (
            not isinstance(check, dict)
            or check.get("check_type") != "clearance"
        ):
            continue
        check_id = str(check.get("check_id") or "").strip()
        target_ids = [
            str(item)
            for item in check.get("target_ids") or []
            if str(item).strip()
        ]
        if not check_id or len(target_ids) != 1:
            continue
        result[check_id] = build_directional_clearance_profile(
            scene=scene,
            target_id=target_ids[0],
            usable_surface_hypothesis=hypotheses.get(target_ids[0]),
            relation_records=relations,
        )
    return result


def apply_directional_clearance_profiles_to_ledger(
    functional_check_ledger: dict[str, Any],
    *,
    by_check_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Replace directed AABB routing priors with forward-corridor facts."""

    result = deepcopy(functional_check_ledger)
    for check in result.get("checks") or []:
        if (
            not isinstance(check, dict)
            or check.get("check_type") != "clearance"
        ):
            continue
        check_id = str(check.get("check_id") or "")
        profile = by_check_id.get(check_id)
        if not isinstance(profile, dict):
            continue
        check["directional_clearance_profile"] = deepcopy(profile)
        if profile.get("status") != "available":
            check["causal_candidate_policy"] = (
                "directional_profile_unavailable_judge_not_restricted"
            )
            continue
        candidates = [
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
                    "vertical_overlap_with_target_m",
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
            and item.get("excluded_from_obstacle") is not True
        ]
        check["causal_candidates"] = candidates
        check["causal_candidate_ids"] = [
            str(item["object_id"])
            for item in candidates
            if str(item.get("object_id") or "").strip()
        ]
        check["causal_candidate_policy"] = (
            "usable_side_forward_corridor_v1"
        )
        check["causal_candidates_are_routing_prior"] = True
    return result


def build_directional_clearance_profile(
    *,
    scene: dict[str, Any] | None,
    target_id: str,
    usable_surface_hypothesis: dict[str, Any] | None,
    relation_records: Iterable[dict[str, Any]] = (),
    corridor_depth_m: float = _DEFAULT_CORRIDOR_DEPTH_M,
) -> dict[str, Any]:
    """Measure the oriented free-space corridor in front of ``target_id``.

    The result is suitable for a Functional Measurement Bank extension.  It
    never contains a validity verdict or a category-specific clearance
    threshold.
    """

    objects = {
        str(item.get("id") or ""): item
        for item in (scene or {}).get("objects") or []
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    target = objects.get(str(target_id))
    base = {
        "schema_version": FUNCTIONAL_DIRECTIONAL_CLEARANCE_VERSION,
        "decision_authority": "none",
        "target_id": str(target_id),
        "status": "unavailable",
        "corridor_depth_m": _positive_finite(
            corridor_depth_m,
            default=_DEFAULT_CORRIDOR_DEPTH_M,
        ),
        "corridor_half_width_m": None,
        "usable_side_id": None,
        "world_outward_direction_xy": None,
        "frontage_origin_xy": None,
        "nearest_forward_obstacle_distance_m": None,
        "forward_intersections": [],
        "depth_samples": [],
        "unavailable_reason": None,
    }
    if target is None:
        base["unavailable_reason"] = "target_not_in_scene"
        return base
    side = _trusted_side(usable_surface_hypothesis)
    if side is None:
        base["unavailable_reason"] = "usable_side_unavailable_or_ambiguous"
        return base
    target_geometry = _object_geometry(target)
    if target_geometry is None:
        base["unavailable_reason"] = "target_geometry_unavailable"
        return base

    side_id, surface_role, confidence = side
    local_axis = _LOCAL_SIDE_AXES[side_id]
    yaw = math.radians(target_geometry["yaw_degrees"])
    direction = (
        local_axis[0] * math.cos(yaw) - local_axis[1] * math.sin(yaw),
        local_axis[0] * math.sin(yaw) + local_axis[1] * math.cos(yaw),
    )
    direction = _normalize(direction)
    lateral = (-direction[1], direction[0])
    target_forward_extent = _support_extent(
        target_geometry,
        direction,
    )
    target_lateral_extent = _support_extent(
        target_geometry,
        lateral,
    )
    frontage_origin = (
        target_geometry["center"][0] + direction[0] * target_forward_extent,
        target_geometry["center"][1] + direction[1] * target_forward_extent,
    )
    corridor_half_width = max(
        _MIN_CORRIDOR_HALF_WIDTH_M,
        min(
            _MAX_CORRIDOR_HALF_WIDTH_M,
            target_lateral_extent + _LATERAL_MARGIN_M,
        ),
    )
    corridor_center = (
        frontage_origin[0]
        + direction[0] * base["corridor_depth_m"] / 2.0,
        frontage_origin[1]
        + direction[1] * base["corridor_depth_m"] / 2.0,
    )
    mobility = _relation_mobility_map(
        target_id=str(target_id),
        relation_records=relation_records,
    )
    approach_z_interval = (
        min(0.0, target_geometry["z_interval"][0]),
        max(_ORDINARY_APPROACH_HEIGHT_M, target_geometry["z_interval"][1]),
    )

    intersections: list[dict[str, Any]] = []
    for object_id, candidate in sorted(objects.items()):
        if object_id == str(target_id):
            continue
        geometry = _object_geometry(candidate)
        if geometry is None:
            continue
        delta = (
            geometry["center"][0] - frontage_origin[0],
            geometry["center"][1] - frontage_origin[1],
        )
        forward_center = _dot(delta, direction)
        lateral_center = _dot(delta, lateral)
        forward_extent = _support_extent(geometry, direction)
        lateral_extent = _support_extent(geometry, lateral)
        forward_near = forward_center - forward_extent
        forward_far = forward_center + forward_extent
        lateral_clearance = (
            abs(lateral_center)
            - lateral_extent
            - corridor_half_width
        )
        corridor_overlap = max(
            0.0,
            min(base["corridor_depth_m"], forward_far)
            - max(0.0, forward_near),
        )
        lateral_overlap = max(
            0.0,
            min(corridor_half_width, lateral_center + lateral_extent)
            - max(-corridor_half_width, lateral_center - lateral_extent),
        )
        support_relation = _support_relation(
            target_geometry,
            geometry,
        )
        vertical_overlap = _interval_overlap(
            target_geometry["z_interval"],
            geometry["z_interval"],
        )
        approach_vertical_overlap = _interval_overlap(
            approach_z_interval,
            geometry["z_interval"],
        )
        vertical_relevant = approach_vertical_overlap > 1.0e-9
        thin_floor_layer = bool(
            geometry["z_interval"][1]
            - geometry["z_interval"][0]
            <= _THIN_FLOOR_LAYER_HEIGHT_M
            and geometry["z_interval"][0]
            <= _THIN_FLOOR_LAYER_HEIGHT_M
        )
        intersects = bool(
            corridor_overlap > 0.0
            and lateral_overlap > 0.0
            and _oriented_rectangles_overlap(
                left_center=corridor_center,
                left_axes=(direction, lateral),
                left_half_extents=(
                    base["corridor_depth_m"] / 2.0,
                    corridor_half_width,
                ),
                right_geometry=geometry,
            )
        )
        excluded_from_obstacle = bool(
            thin_floor_layer
            or not vertical_relevant
            or support_relation in {"supported_by_target", "supports_target"}
        )
        if not intersects:
            continue
        intersections.append(
            {
                "object_id": object_id,
                "forward_near_distance_m": round(max(0.0, forward_near), 6),
                "forward_far_distance_m": round(forward_far, 6),
                "lateral_clearance_m": round(lateral_clearance, 6),
                "corridor_overlap_depth_m": round(corridor_overlap, 6),
                "corridor_overlap_width_m": round(lateral_overlap, 6),
                "corridor_width_overlap_fraction": round(
                    lateral_overlap / (2.0 * corridor_half_width),
                    6,
                ),
                "corridor_overlap_area_proxy_m2": round(
                    corridor_overlap * lateral_overlap,
                    6,
                ),
                "vertical_overlap_with_target_m": round(
                    vertical_overlap,
                    6,
                ),
                "vertical_overlap_with_approach_m": round(
                    approach_vertical_overlap,
                    6,
                ),
                "vertical_relevant": vertical_relevant,
                "support_relation": support_relation,
                "thin_floor_layer": thin_floor_layer,
                "ordinary_mobility": mobility.get(object_id, "unspecified"),
                "excluded_from_obstacle": excluded_from_obstacle,
            }
        )

    intersections.sort(
        key=lambda item: (
            bool(item["excluded_from_obstacle"]),
            float(item["forward_near_distance_m"]),
            str(item["object_id"]),
        )
    )
    effective = [
        item for item in intersections if not item["excluded_from_obstacle"]
    ]
    samples = []
    for distance in (0.25, 0.5, 0.75, 1.0):
        if distance > base["corridor_depth_m"] + 1.0e-9:
            continue
        samples.append(
            {
                "distance_m": distance,
                "intersecting_object_ids": [
                    str(item["object_id"])
                    for item in effective
                    if float(item["forward_near_distance_m"])
                    <= distance
                    <= float(item["forward_far_distance_m"])
                ],
            }
        )
    base.update(
        status="available",
        usable_side_id=side_id,
        surface_role=surface_role,
        surface_confidence=confidence,
        world_outward_direction_xy=[direction[0], direction[1]],
        frontage_origin_xy=[frontage_origin[0], frontage_origin[1]],
        corridor_half_width_m=round(corridor_half_width, 6),
        nearest_forward_obstacle_distance_m=(
            float(effective[0]["forward_near_distance_m"])
            if effective
            else None
        ),
        forward_intersections=deepcopy(intersections),
        depth_samples=samples,
        unavailable_reason=None,
    )
    return base


def _trusted_side(
    hypothesis: dict[str, Any] | None,
) -> tuple[str, str | None, Any] | None:
    if not isinstance(hypothesis, dict):
        return None
    if str(hypothesis.get("status") or "") != "identified":
        return None
    surfaces = [
        item
        for item in hypothesis.get("surfaces") or []
        if isinstance(item, dict)
        and str(item.get("side_id") or "") in _LOCAL_SIDE_AXES
    ]
    if len(surfaces) != 1:
        return None
    surface = surfaces[0]
    return (
        str(surface["side_id"]),
        str(surface.get("surface_role") or "").strip() or None,
        surface.get("confidence"),
    )


def _object_geometry(item: dict[str, Any]) -> dict[str, Any] | None:
    center = _vector3(item.get("center"))
    size = _vector3(item.get("size"))
    rotation = _vector3(item.get("rotation") or [0.0, 0.0, 0.0])
    if center is None or size is None or rotation is None:
        return None
    if any(value <= 0.0 for value in size):
        return None
    yaw = math.radians(rotation[2])
    axes = (
        (math.cos(yaw), math.sin(yaw)),
        (-math.sin(yaw), math.cos(yaw)),
    )
    return {
        "center": center,
        "size": size,
        "yaw_degrees": rotation[2],
        "axes": axes,
        "z_interval": (
            center[2] - size[2] / 2.0,
            center[2] + size[2] / 2.0,
        ),
    }


def _support_extent(
    geometry: dict[str, Any],
    direction: tuple[float, float],
) -> float:
    axes = geometry["axes"]
    size = geometry["size"]
    return float(
        size[0] / 2.0 * abs(_dot(axes[0], direction))
        + size[1] / 2.0 * abs(_dot(axes[1], direction))
    )


def _support_relation(
    target: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    target_top = target["z_interval"][1]
    target_bottom = target["z_interval"][0]
    candidate_top = candidate["z_interval"][1]
    candidate_bottom = candidate["z_interval"][0]
    xy_overlap = _obb_xy_overlap_proxy(target, candidate)
    if not xy_overlap:
        return "none"
    if abs(candidate_bottom - target_top) <= _SUPPORT_CONTACT_TOLERANCE_M:
        return "supported_by_target"
    if abs(target_bottom - candidate_top) <= _SUPPORT_CONTACT_TOLERANCE_M:
        return "supports_target"
    return "none"


def _obb_xy_overlap_proxy(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    # Separating-axis test over both OBB local bases.
    delta = (
        right["center"][0] - left["center"][0],
        right["center"][1] - left["center"][1],
    )
    for axis in (*left["axes"], *right["axes"]):
        separation = abs(_dot(delta, axis))
        if separation > (
            _support_extent(left, axis)
            + _support_extent(right, axis)
            + 1.0e-9
        ):
            return False
    return True


def _oriented_rectangles_overlap(
    *,
    left_center: tuple[float, float],
    left_axes: tuple[tuple[float, float], tuple[float, float]],
    left_half_extents: tuple[float, float],
    right_geometry: dict[str, Any],
) -> bool:
    """Full four-axis SAT for a corridor rectangle and candidate OBB."""

    delta = (
        right_geometry["center"][0] - left_center[0],
        right_geometry["center"][1] - left_center[1],
    )
    right_axes = right_geometry["axes"]
    right_half_extents = (
        right_geometry["size"][0] / 2.0,
        right_geometry["size"][1] / 2.0,
    )
    for axis in (*left_axes, *right_axes):
        separation = abs(_dot(delta, axis))
        left_radius = sum(
            left_half_extents[index]
            * abs(_dot(left_axes[index], axis))
            for index in range(2)
        )
        right_radius = sum(
            right_half_extents[index]
            * abs(_dot(right_axes[index], axis))
            for index in range(2)
        )
        if separation > left_radius + right_radius + 1.0e-9:
            return False
    return True


def _relation_mobility_map(
    *,
    target_id: str,
    relation_records: Iterable[dict[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in relation_records:
        if not isinstance(item, dict):
            continue
        target_ids = [
            str(value)
            for value in item.get("target_ids") or []
            if str(value).strip()
        ]
        if len(target_ids) != 2:
            continue
        focal_id = str(item.get("focal_id") or target_ids[0]).strip()
        counterpart = str(
            item.get("counterpart_id") or target_ids[1]
        ).strip()
        # ``ordinary_mobility`` describes the ordered counterpart.  It is a
        # useful obstacle role only when the clearance target is the focal;
        # applying it symmetrically reverses table/chair and similar roles.
        if (
            target_id != focal_id
            or not counterpart
            or counterpart == focal_id
            or {focal_id, counterpart} != set(target_ids)
        ):
            continue
        role = str(item.get("ordinary_mobility") or "").strip()
        if role in {"fixed", "movable_companion", "portable_unrelated"}:
            previous = result.get(counterpart)
            result[counterpart] = (
                role
                if previous in {None, role}
                else "unspecified"
            )
    return result


def _interval_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _normalize(value: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(value[0], value[1])
    if norm <= 1.0e-12:
        return 0.0, 1.0
    return value[0] / norm, value[1] / norm


def _dot(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result  # type: ignore[return-value]


def _positive_finite(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) and result > 0.0 else default


__all__ = [
    "FUNCTIONAL_DIRECTIONAL_CLEARANCE_VERSION",
    "apply_directional_clearance_profiles_to_ledger",
    "build_directional_clearance_extensions",
    "build_directional_clearance_profile",
]
