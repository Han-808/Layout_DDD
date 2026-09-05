"""Polygon-aware camera feasibility layered over existing metric pose banks."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable

import numpy as np

from benchmark.evaluator.generic_validity.geometry import get_obb_corners
from benchmark.non_rectangular.geometry import (
    point_inside_object_obb,
    polygon_geometry_from_scene,
)
from benchmark.scene_io.object_normalization import normalize_object


POLYGON_CAMERA_POLICY_VERSION = "non_rectangular_polygon_camera_gate_v2"
NONRECT_CAMERA_EXHAUSTION_AUDIT_KEY = (
    "_nonrect_camera_candidate_audit"
)
OOB_MAX_FOCUS_INSET_M = 0.10


class NonRectangularCameraEvidenceExhausted(RuntimeError):
    """Typed normal exhaustion of a bounded nonrect camera search."""


def generate_polygon_camera_pose_candidates(
    request: dict[str, Any],
    *,
    max_candidates: int,
    policy: str,
    base_generator: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    scene = request.get("scene")
    geometry = polygon_geometry_from_scene(scene)
    if geometry is None:
        raise ValueError("polygon camera generator requires non-rectangular room metadata")
    metric = str(request.get("metric") or "event").strip().lower()
    edge_records = _oob_edge_records(request) if metric == "oob" else []
    if edge_records:
        return _oob_edge_candidates(
            request,
            geometry=geometry,
            edge_records=edge_records,
            max_candidates=max_candidates,
        )
    raw = base_generator(
        request,
        max_candidates=max_candidates,
    )
    return _gate_candidates(
        raw,
        scene=scene,
        geometry=geometry,
        max_candidates=max_candidates,
        policy=policy,
    )


def generate_polygon_usable_surface_side_bank(
    scene: dict[str, Any],
    *,
    target_id: str,
    repair: bool,
) -> list[dict[str, Any]]:
    from benchmark.rendering import camera_pose as canonical_camera

    geometry = polygon_geometry_from_scene(scene)
    if geometry is None:
        raise ValueError("polygon usable-side bank requires room geometry")
    raw = canonical_camera._generate_usable_surface_side_bank(
        scene,
        target_id=target_id,
        elevation_degrees=24.0 if repair else 10.0,
        distance_scale=1.0,
        context_margin_m=0.30 if repair else 0.45,
        preferred_lens_mm=52.0 if repair else 45.0,
        policy_source=(
            "usable_surface_elevated_detail_repair_v1"
            if repair
            else "usable_surface_local_side_bank_v1"
        ),
    )
    return _gate_candidates(
        raw,
        scene=scene,
        geometry=geometry,
        max_candidates=len(raw) or 4,
        policy="local",
    )


def generate_polygon_global_context_poses(
    scene: dict[str, Any],
) -> list[dict[str, Any]]:
    geometry = polygon_geometry_from_scene(scene)
    if geometry is None:
        raise ValueError("polygon global context requires room geometry")
    min_x, max_x, min_y, max_y = geometry.bounds_xy
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    span = max(geometry.span_m, 0.5)
    target_z = min(
        geometry.floor_z_m
        + (geometry.ceiling_z_m - geometry.floor_z_m) * 0.4,
        geometry.floor_z_m + 1.2,
    )
    top = {
        "id": "global_top",
        "name": "global_top",
        "camera_type": "ORTHO",
        "location": [
            center_x,
            center_y,
            geometry.ceiling_z_m + max(5.0, span * 1.15),
        ],
        "target": [center_x, center_y, target_z],
        "ortho_scale": span * 1.15,
        "lens_mm": 48.0,
        "sensor_width_mm": 36.0,
        "sensor_fit": "HORIZONTAL",
        "clip_start_m": 0.02,
        "clip_end_m": max(100.0, geometry.ceiling_z_m * 10.0),
        "target_object_ids": [],
        "room_bounds": [
            min_x,
            max_x,
            min_y,
            max_y,
            geometry.floor_z_m,
            geometry.ceiling_z_m,
        ],
        "policy_source": POLYGON_CAMERA_POLICY_VERSION,
        "polygon_context": {
            "framing": "complete_polygon_envelope",
            "camera_xy_may_be_outside_polygon": True,
        },
    }
    anchor_x, anchor_y = geometry.representative_interior_xy()
    target = np.asarray([anchor_x, anchor_y, target_z], dtype=float)
    perspective_options: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for azimuth in (225.0, 315.0, 135.0, 45.0, 270.0, 180.0, 0.0, 90.0):
        elevation = 28.0
        direction = _direction(azimuth, elevation)
        placement = geometry.place_on_feasible_ray(
            target=target,
            direction=direction,
            desired_distance_m=max(1.5, span * 0.85),
        )
        if placement is None:
            continue
        location, distance, audit = placement
        perspective_options.append((distance, location, audit))
    if not perspective_options:
        top["polygon_context"]["perspective_status"] = (
            "unavailable_after_bounded_polygon_search"
        )
        return [top]
    distance, location, audit = max(
        perspective_options,
        key=lambda item: item[0],
    )
    perspective = {
        "id": "global_perspective",
        "name": "global_perspective",
        "camera_type": "PERSP",
        "location": _vector(location),
        "target": _vector(target),
        "lens_mm": 36.0,
        "sensor_width_mm": 36.0,
        "sensor_fit": "HORIZONTAL",
        "clip_start_m": 0.02,
        "clip_end_m": max(100.0, geometry.ceiling_z_m * 10.0),
        "target_object_ids": [],
        "room_bounds": [
            min_x,
            max_x,
            min_y,
            max_y,
            geometry.floor_z_m,
            geometry.ceiling_z_m,
        ],
        "policy_source": POLYGON_CAMERA_POLICY_VERSION,
        "candidate_policy": "local",
        "distance_m": float(distance),
        "technical_feasibility": True,
        "feasibility": audit,
        "polygon_context": {
            "anchor_source": "polylabel_representative_interior",
            "single_perspective_budget": True,
        },
    }
    return [top, perspective]


def revalidate_polygon_camera_pose(
    pose: dict[str, Any],
    *,
    scene: dict[str, Any],
) -> dict[str, Any]:
    geometry = polygon_geometry_from_scene(scene)
    if geometry is None or str(pose.get("camera_type")) == "ORTHO":
        return pose
    return _gate_candidate(
        pose,
        scene=scene,
        geometry=geometry,
        policy=str(pose.get("candidate_policy") or "local"),
    ) or _raise_no_action_pose()


def _gate_candidates(
    candidates: list[dict[str, Any]],
    *,
    scene: dict[str, Any],
    geometry: Any,
    max_candidates: int,
    policy: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        resolved = _gate_candidate(
            candidate,
            scene=scene,
            geometry=geometry,
            policy=policy,
        )
        if resolved is None:
            continue
        if _duplicates(resolved, result):
            continue
        result.append(resolved)
        if len(result) >= max(1, int(max_candidates)):
            break
    return result


def _gate_candidate(
    candidate: dict[str, Any],
    *,
    scene: dict[str, Any],
    geometry: Any,
    policy: str,
) -> dict[str, Any] | None:
    from benchmark.rendering import camera_pose as canonical_camera

    result = deepcopy(candidate)
    target = np.asarray(result.get("target"), dtype=float)
    location = np.asarray(result.get("location"), dtype=float)
    if target.shape != (3,) or location.shape != (3,):
        return None
    coerced_xy = geometry.coerce_xy_inside(target[:2])
    target[:2] = np.asarray(coerced_xy, dtype=float)
    direction = _candidate_direction(result, location=location, target=target)
    desired = float(
        result.get("intended_distance_m")
        or result.get("distance_m")
        or np.linalg.norm(location - target)
    )
    raw_bounds = result.get("proxy_framing_bounds") or result.get("target_bounds")
    bounds = None
    framing = None
    aspect = 16.0 / 9.0
    if isinstance(raw_bounds, list) and len(raw_bounds) == 2:
        try:
            bounds = (
                np.asarray(raw_bounds[0], dtype=float),
                np.asarray(raw_bounds[1], dtype=float),
            )
            prior = result.get("proxy_framing")
            aspect = (
                float(prior.get("aspect_ratio") or 16.0 / 9.0)
                if isinstance(prior, dict)
                else 16.0 / 9.0
            )
        except (TypeError, ValueError):
            return None
    base_azimuth = math.degrees(
        math.atan2(float(direction[1]), float(direction[0]))
    )
    direction_options: list[tuple[np.ndarray, str]] = [(direction, "none")]
    if str(result.get("camera_type") or "PERSP") == "PERSP":
        for elevation in (45.0, 60.0, 72.0):
            repaired = _direction(base_azimuth, elevation)
            if float(np.dot(repaired, direction)) < 0.9999:
                direction_options.append(
                    (repaired, f"elevation_{int(elevation)}")
                )
    resolved: tuple[np.ndarray, dict[str, Any], float | None, dict[str, Any] | None, str] | None = None
    for candidate_direction, repair in direction_options:
        placement = geometry.place_on_feasible_ray(
            target=target,
            direction=candidate_direction,
            desired_distance_m=max(0.25, desired),
        )
        if placement is None:
            continue
        candidate_location, _, candidate_audit = placement
        if any(
            point_inside_object_obb(
                candidate_location,
                raw,
                clearance_m=0.03,
            )
            for raw in scene.get("objects", [])
            if isinstance(raw, dict)
        ):
            continue
        lens: float | None = None
        candidate_framing: dict[str, Any] | None = None
        if bounds is not None:
            lens, candidate_framing = canonical_camera._fit_proxy_framing_lens(
                location=candidate_location,
                target=target,
                bounds=bounds,
                preferred_lens_mm=float(result.get("lens_mm") or 45.0),
                aspect_ratio=aspect,
            )
            if (
                candidate_framing.get("all_corners_in_front") is not True
                or candidate_framing.get("proxy_bounds_fit") is not True
            ):
                continue
        resolved = (
            candidate_location,
            candidate_audit,
            lens,
            candidate_framing,
            repair,
        )
        break
    if resolved is None:
        return None
    location, audit, lens, framing, analytic_repair = resolved
    audit = {**audit, "analytic_repair": analytic_repair}
    if framing is not None:
        framing["validation_status"] = "fits_proxy_bounds_after_polygon_gate"
        result["lens_mm"] = float(lens)
        result["proxy_framing"] = framing
    azimuth, elevation, measured = canonical_camera._pose_angles(
        location,
        target,
    )
    result.update(
        location=_vector(location),
        target=_vector(target),
        azimuth_degrees=float(azimuth),
        elevation_degrees=float(elevation),
        distance_m=float(measured),
        technical_feasibility=True,
        candidate_policy=policy,
        feasibility={
            **deepcopy(result.get("feasibility") or {}),
            **audit,
        },
        polygon_geometry_gate={
            "schema_version": POLYGON_CAMERA_POLICY_VERSION,
            "point_inside_polygon": True,
            "wall_clearance_checked": True,
            "room_wall_los_checked": True,
            "proxy_framing_checked": framing is not None,
            "target_xy_coerced_inside": bool(
                np.linalg.norm(target[:2] - np.asarray(candidate["target"][:2], dtype=float))
                > geometry.tolerance_m
            ),
        },
    )
    return result


def _oob_edge_candidates(
    request: dict[str, Any],
    *,
    geometry: Any,
    edge_records: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    from benchmark.rendering import camera_pose as canonical_camera

    scene = request["scene"]
    object_ids = [str(item) for item in request.get("object_ids") or []]
    raw_object = next(
        (
            item
            for item in scene.get("objects", [])
            if isinstance(item, dict)
            and str(item.get("id")) in object_ids
        ),
        None,
    )
    if raw_object is None:
        return []
    normalized = normalize_object(raw_object)
    corners = get_obb_corners(normalized)
    bounds = (np.min(corners, axis=0), np.max(corners, axis=0))
    extent = np.maximum(bounds[1] - bounds[0], 0.1)
    desired = max(1.2, float(max(extent)) * 2.2)
    count = max(1, int(max_candidates))
    nominal_variants = (
        (0.0, 18.0, "normal"),
        (35.0, 24.0, "oblique_left"),
        (-35.0, 24.0, "oblique_right"),
        (70.0, 20.0, "tangent_left"),
        (-70.0, 20.0, "tangent_right"),
        (0.0, 34.0, "normal_high"),
    )
    repair_tiers = (
        {
            "tier": "r0_nominal",
            "focus_inset_m": 0.0,
            "wall_clearance_m": 0.08,
            "object_padding_m": 0.03,
            "variants": nominal_variants,
            "orthographic_top": False,
        },
        {
            "tier": "r1_near_wall",
            "focus_inset_m": 0.02,
            "wall_clearance_m": 0.04,
            "object_padding_m": 0.01,
            "variants": nominal_variants,
            "orthographic_top": False,
        },
        {
            "tier": "r2_high_angle",
            "focus_inset_m": 0.05,
            "wall_clearance_m": 0.02,
            "object_padding_m": 0.0,
            "variants": (
                (0.0, 45.0, "normal_45"),
                (35.0, 45.0, "oblique_left_45"),
                (-35.0, 45.0, "oblique_right_45"),
                (0.0, 60.0, "normal_60"),
                (35.0, 60.0, "oblique_left_60"),
                (-35.0, 60.0, "oblique_right_60"),
                (0.0, 72.0, "normal_72"),
                (35.0, 72.0, "oblique_left_72"),
                (-35.0, 72.0, "oblique_right_72"),
            ),
            "orthographic_top": False,
        },
        {
            "tier": "r3_last_local",
            "focus_inset_m": OOB_MAX_FOCUS_INSET_M,
            "wall_clearance_m": 0.0,
            "object_padding_m": 0.0,
            "variants": (
                (0.0, 72.0, "normal_72"),
                (35.0, 72.0, "oblique_left_72"),
                (-35.0, 72.0, "oblique_right_72"),
            ),
            "orthographic_top": True,
        },
    )
    generation_audit: dict[str, Any] = {
        "schema_version": "non_rectangular_oob_camera_repair_audit_v1",
        "policy": "nominal_then_bounded_repair_v1",
        "maximum_focus_inset_m": OOB_MAX_FOCUS_INSET_M,
        "normal_path_unchanged_until_empty": True,
        "tiers": [],
        "selected_tier": None,
        "terminal_outcome": None,
    }
    for tier in repair_tiers:
        result, tier_audit = _oob_edge_candidates_for_tier(
            scene=scene,
            raw_object=raw_object,
            normalized=normalized,
            geometry=geometry,
            edge_records=edge_records,
            bounds=bounds,
            desired=desired,
            count=count,
            tier=tier,
        )
        generation_audit["tiers"].append(tier_audit)
        if result:
            generation_audit["selected_tier"] = tier["tier"]
            generation_audit["terminal_outcome"] = "candidates_available"
            request[NONRECT_CAMERA_EXHAUSTION_AUDIT_KEY] = generation_audit
            return result
    generation_audit["terminal_outcome"] = "bounded_local_bank_exhausted"
    request[NONRECT_CAMERA_EXHAUSTION_AUDIT_KEY] = generation_audit
    return []


def _oob_edge_candidates_for_tier(
    *,
    scene: dict[str, Any],
    raw_object: dict[str, Any],
    normalized: Any,
    geometry: Any,
    edge_records: list[dict[str, Any]],
    bounds: tuple[np.ndarray, np.ndarray],
    desired: float,
    count: int,
    tier: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from benchmark.rendering import camera_pose as canonical_camera

    result: list[dict[str, Any]] = []
    rejected = {
        "invalid_edge_frame": 0,
        "no_feasible_ray": 0,
        "camera_inside_object_obb": 0,
        "proxy_framing_failed": 0,
        "duplicate": 0,
    }
    attempted = 0
    focus_inset = min(
        max(0.0, float(tier["focus_inset_m"])),
        OOB_MAX_FOCUS_INSET_M,
    )
    object_ids = [str(normalized.id)]
    for edge in edge_records:
        normal = edge.get("inward_normal_xy")
        focus = edge.get("focus_xy") or edge.get("wall_focus_xy")
        if (
            not isinstance(normal, list)
            or len(normal) < 2
            or not isinstance(focus, list)
            or len(focus) < 2
        ):
            rejected["invalid_edge_frame"] += 1
            continue
        normal_xy = np.asarray(normal[:2], dtype=float)
        normal_norm = float(np.linalg.norm(normal_xy))
        if normal_norm <= 1.0e-9:
            rejected["invalid_edge_frame"] += 1
            continue
        normal_xy /= normal_norm
        crossing_focus = np.asarray(focus[:2], dtype=float)
        camera_focus = crossing_focus + normal_xy * focus_inset
        if not geometry.contains_xy(camera_focus):
            camera_focus = np.asarray(
                geometry.coerce_xy_inside(camera_focus),
                dtype=float,
            )
        base_azimuth = math.degrees(
            math.atan2(float(normal_xy[1]), float(normal_xy[0]))
        )
        target = np.asarray(
            [
                float(camera_focus[0]),
                float(camera_focus[1]),
                float(
                    np.clip(
                        normalized.center[2],
                        geometry.floor_z_m + 0.20,
                        geometry.ceiling_z_m - 0.20,
                    )
                ),
            ],
            dtype=float,
        )
        wall_id = str(edge.get("wall_id") or "wall")
        if tier.get("orthographic_top"):
            attempted += 1
            ortho = _oob_local_orthographic_candidate(
                geometry=geometry,
                bounds=bounds,
                target=target,
                object_ids=object_ids,
                wall_id=wall_id,
                normal=normal_xy,
                tangent=edge.get("tangent_xy") or [],
                crossing_focus=crossing_focus,
                focus_inset_m=focus_inset,
                tier_name=str(tier["tier"]),
            )
            if not _duplicates(ortho, result):
                result.append(ortho)
            else:
                rejected["duplicate"] += 1
            if len(result) >= count:
                break
        for delta, elevation, label in tier["variants"]:
            attempted += 1
            direction = _direction(base_azimuth + delta, elevation)
            placement = geometry.place_on_feasible_ray(
                target=target,
                direction=direction,
                desired_distance_m=desired,
                wall_clearance_m=float(tier["wall_clearance_m"]),
                relax_wall_clearance=False,
            )
            if placement is None:
                rejected["no_feasible_ray"] += 1
                continue
            location, _, audit = placement
            if any(
                point_inside_object_obb(
                    location,
                    raw,
                    clearance_m=float(tier["object_padding_m"]),
                )
                for raw in scene.get("objects", [])
                if isinstance(raw, dict)
            ):
                rejected["camera_inside_object_obb"] += 1
                continue
            lens, framing = canonical_camera._fit_proxy_framing_lens(
                location=location,
                target=target,
                bounds=bounds,
                preferred_lens_mm=52.0,
                aspect_ratio=16.0 / 9.0,
            )
            if framing.get("proxy_bounds_fit") is not True:
                rejected["proxy_framing_failed"] += 1
                continue
            azimuth, actual_elevation, distance = (
                canonical_camera._pose_angles(location, target)
            )
            candidate = {
                "id": (
                    f"oob_{len(result):02d}_{wall_id}_"
                    f"{tier['tier']}_{label}"
                ),
                "name": f"oob_{wall_id}_{tier['tier']}_{label}",
                "camera_type": "PERSP",
                "location": _vector(location),
                "target": _vector(target),
                "lens_mm": float(lens),
                "sensor_width_mm": 36.0,
                "sensor_fit": "HORIZONTAL",
                "clip_start_m": 0.02,
                "clip_end_m": max(100.0, geometry.ceiling_z_m * 10.0),
                "azimuth_degrees": float(azimuth),
                "elevation_degrees": float(actual_elevation),
                "distance_m": float(distance),
                "intended_azimuth_degrees": (
                    base_azimuth + delta
                ) % 360.0,
                "intended_elevation_degrees": elevation,
                "intended_distance_m": desired,
                "target_object_ids": object_ids,
                "target_bounds": [_vector(bounds[0]), _vector(bounds[1])],
                "proxy_framing_bounds": [
                    _vector(bounds[0]),
                    _vector(bounds[1]),
                ],
                "room_bounds": [
                    *geometry.bounds_xy,
                    geometry.floor_z_m,
                    geometry.ceiling_z_m,
                ],
                "policy_source": (
                    "non_rectangular_oob_edge_local_bank_v2"
                ),
                "candidate_policy": "local",
                "event_focus_source": "violated_polygon_wall_edge",
                "focus_kind": "oob_wall_edge",
                "focus_wall_id": wall_id,
                "crossing_focus_xy": _vector(crossing_focus),
                "camera_focus_inset_m": focus_inset,
                "repair_tier": str(tier["tier"]),
                "edge_local_frame": {
                    "inward_normal_xy": _vector(normal_xy),
                    "tangent_xy": list(edge.get("tangent_xy") or []),
                },
                "technical_feasibility": True,
                "feasibility": {
                    **audit,
                    "repair_tier": str(tier["tier"]),
                    "object_obb_padding_m": float(
                        tier["object_padding_m"]
                    ),
                    "camera_focus_inset_m": focus_inset,
                },
                "proxy_framing": framing,
                "polygon_geometry_gate": {
                    "schema_version": POLYGON_CAMERA_POLICY_VERSION,
                    "point_inside_polygon": True,
                    "wall_clearance_checked": True,
                    "room_wall_los_checked": True,
                    "proxy_framing_checked": True,
                    "crossing_geometry_unchanged": True,
                },
            }
            if _duplicates(candidate, result):
                rejected["duplicate"] += 1
                continue
            result.append(candidate)
            if len(result) >= count:
                break
        if len(result) >= count:
            break
    return result, {
        "tier": str(tier["tier"]),
        "focus_inset_m": focus_inset,
        "wall_clearance_m": float(tier["wall_clearance_m"]),
        "object_padding_m": float(tier["object_padding_m"]),
        "attempted_candidate_count": attempted,
        "accepted_candidate_count": len(result),
        "rejections": rejected,
    }


def _oob_local_orthographic_candidate(
    *,
    geometry: Any,
    bounds: tuple[np.ndarray, np.ndarray],
    target: np.ndarray,
    object_ids: list[str],
    wall_id: str,
    normal: np.ndarray,
    tangent: list[Any],
    crossing_focus: np.ndarray,
    focus_inset_m: float,
    tier_name: str,
) -> dict[str, Any]:
    edge_inset_focus = np.asarray(target[:2], dtype=float)
    framing_center = np.asarray(
        [
            float((bounds[0][0] + bounds[1][0]) / 2.0),
            float((bounds[0][1] + bounds[1][1]) / 2.0),
        ],
        dtype=float,
    )
    if not geometry.contains_xy(framing_center):
        framing_center = np.asarray(
            geometry.coerce_xy_inside(framing_center),
            dtype=float,
        )
    framing_min = np.minimum(bounds[0][:2], crossing_focus)
    framing_max = np.maximum(bounds[1][:2], crossing_focus)
    horizontal_span = max(
        float(framing_max[0] - framing_min[0]),
        float(framing_max[1] - framing_min[1]),
        0.10,
    )
    orthographic_target = np.asarray(
        [framing_center[0], framing_center[1], float(target[2])],
        dtype=float,
    )
    location = np.asarray(
        [
            float(framing_center[0]),
            float(framing_center[1]),
            min(
                geometry.camera_max_z_m,
                max(
                    geometry.ceiling_z_m + 2.0,
                    float(bounds[1][2]) + 2.0,
                ),
            ),
        ],
        dtype=float,
    )
    return {
        "id": f"oob_ortho_{wall_id}_{tier_name}",
        "name": f"oob_ortho_{wall_id}_{tier_name}",
        "camera_type": "ORTHO",
        "location": _vector(location),
        "target": _vector(orthographic_target),
        "ortho_scale": horizontal_span + 0.20,
        "lens_mm": 48.0,
        "sensor_width_mm": 36.0,
        "sensor_fit": "HORIZONTAL",
        "clip_start_m": 0.02,
        "clip_end_m": max(100.0, geometry.camera_max_z_m * 10.0),
        "azimuth_degrees": 0.0,
        "elevation_degrees": 90.0,
        "distance_m": float(location[2] - orthographic_target[2]),
        "target_object_ids": list(object_ids),
        "target_bounds": [_vector(bounds[0]), _vector(bounds[1])],
        "proxy_framing_bounds": [
            _vector(bounds[0]),
            _vector(bounds[1]),
        ],
        "room_bounds": [
            *geometry.bounds_xy,
            geometry.floor_z_m,
            geometry.ceiling_z_m,
        ],
        "policy_source": "non_rectangular_oob_edge_local_bank_v2",
        "candidate_policy": "local",
        "event_focus_source": "violated_polygon_wall_edge",
        "focus_kind": "oob_wall_edge",
        "focus_wall_id": wall_id,
        "crossing_focus_xy": _vector(crossing_focus),
        "edge_inset_focus_xy": _vector(edge_inset_focus),
        "orthographic_framing_center_xy": _vector(framing_center),
        "camera_focus_inset_m": float(focus_inset_m),
        "repair_tier": tier_name,
        "edge_local_frame": {
            "inward_normal_xy": _vector(normal),
            "tangent_xy": list(tangent),
        },
        "technical_feasibility": True,
        "feasibility": {
            "method": "polygon_local_orthographic_top_v1",
            "room_id": geometry.room_id,
            "wall_clearance_requested_m": 0.0,
            "wall_clearance_applied_m": 0.0,
            "wall_clearance_relaxation_allowed": False,
            "camera_focus_inset_m": float(focus_inset_m),
            "repair_tier": tier_name,
            "point_inside_polygon": True,
            "room_wall_los_valid": True,
        },
        "proxy_framing": {
            "validation_status": "orthographic_target_span_plus_context",
            "proxy_bounds_fit": True,
            "context_padding_m": 0.20,
            "framing_center_source": "target_bounds_center",
            "wall_crossing_included_in_span": True,
        },
        "polygon_geometry_gate": {
            "schema_version": POLYGON_CAMERA_POLICY_VERSION,
            "point_inside_polygon": True,
            "wall_clearance_checked": True,
            "room_wall_los_checked": True,
            "proxy_framing_checked": True,
            "crossing_geometry_unchanged": True,
        },
    }


def _oob_edge_records(request: dict[str, Any]) -> list[dict[str, Any]]:
    event = request.get("event")
    evidence = request.get("detector_evidence")
    for source in (event, evidence):
        if not isinstance(source, dict):
            continue
        values = source.get("violated_edges")
        if not isinstance(values, list):
            continue
        result = []
        for item in values:
            if not isinstance(item, dict):
                continue
            local = item.get("edge_local_frame")
            result.append(
                {
                    **deepcopy(item),
                    "inward_normal_xy": (
                        list(local.get("inward_normal_xy") or [])
                        if isinstance(local, dict)
                        else list(item.get("inward_normal_xy") or [])
                    ),
                    "tangent_xy": (
                        list(local.get("tangent_xy") or [])
                        if isinstance(local, dict)
                        else list(item.get("tangent_xy") or [])
                    ),
                }
            )
        return result
    return []


def _candidate_direction(
    candidate: dict[str, Any],
    *,
    location: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    azimuth = candidate.get("intended_azimuth_degrees")
    elevation = candidate.get("intended_elevation_degrees")
    if isinstance(azimuth, (int, float)) and isinstance(elevation, (int, float)):
        return _direction(float(azimuth), float(elevation))
    value = location - target
    return value / max(float(np.linalg.norm(value)), 1.0e-12)


def _direction(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    azimuth = math.radians(float(azimuth_degrees))
    elevation = math.radians(float(elevation_degrees))
    horizontal = math.cos(elevation)
    return np.asarray(
        [
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=float,
    )


def _duplicates(
    candidate: dict[str, Any],
    existing: list[dict[str, Any]],
) -> bool:
    location = np.asarray(candidate["location"], dtype=float)
    target = np.asarray(candidate["target"], dtype=float)
    direction = target - location
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    for other in existing:
        other_location = np.asarray(other["location"], dtype=float)
        if float(np.linalg.norm(location - other_location)) > 0.02:
            continue
        other_target = np.asarray(other["target"], dtype=float)
        other_direction = other_target - other_location
        other_direction /= max(
            float(np.linalg.norm(other_direction)),
            1.0e-12,
        )
        angle = math.degrees(
            math.acos(float(np.clip(np.dot(direction, other_direction), -1.0, 1.0)))
        )
        if angle <= 2.0:
            return True
    return False


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float)]


def _raise_no_action_pose() -> dict[str, Any]:
    raise ValueError("camera action has no polygon-feasible pose")


__all__ = [
    "NONRECT_CAMERA_EXHAUSTION_AUDIT_KEY",
    "OOB_MAX_FOCUS_INSET_M",
    "POLYGON_CAMERA_POLICY_VERSION",
    "NonRectangularCameraEvidenceExhausted",
    "generate_polygon_camera_pose_candidates",
    "generate_polygon_global_context_poses",
    "generate_polygon_usable_surface_side_bank",
    "revalidate_polygon_camera_pose",
]
