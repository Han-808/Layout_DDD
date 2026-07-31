from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


CAMERA_POSE_MODES = (
    "global_only",
    "bbox_track",
    "visibility_ranked",
    "support_contact_plane",
    "query_cov",
    "auto",
)
P0B_CAMERA_METRICS = (
    "collision",
    "object_architecture_penetration",
    "oob",
    "support",
    "functional_semantic_fidelity",
)
L3_CAMERA_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
    "functional_consistency",
)
CAMERA_EVIDENCE_METRICS = (*P0B_CAMERA_METRICS, *L3_CAMERA_METRICS)
DEFAULT_CAMERA_MODE_BY_METRIC = {
    "collision": "visibility_ranked",
    "object_architecture_penetration": "visibility_ranked",
    "oob": "visibility_ranked",
    "support": "support_contact_plane",
    # Canonical L2 local requests are already scoped by a frozen prompt claim;
    # visibility ranking chooses evidence views but never judges the claim.
    "functional_semantic_fidelity": "visibility_ranked",
    # Canonical L3 local policies use the same geometry-only candidate
    # generation. These mappings do not alter any frozen L1 VisualConfig.
    "scale_consistency": "visibility_ranked",
    "object_pairing_consistency": "visibility_ranked",
    "functional_consistency": "visibility_ranked",
    # Style consumes the trusted overview packet by default.
    "style_consistency": "global_only",
}
CAMERA_ACTIONS = (
    "orbit_left",
    "orbit_right",
    "elevate",
    "lower",
    "dolly_in",
    "dolly_out",
)
CAMERA_ACTION_PROTOCOL_VERSION = "bounded_camera_action_v3"
CAMERA_ACTION_PARAMETERS: dict[str, dict[str, float | str]] = {
    "orbit_left": {"axis": "target_z", "delta_degrees": 20.0},
    "orbit_right": {"axis": "target_z", "delta_degrees": -20.0},
    "elevate": {"delta_degrees": 12.0, "maximum_degrees": 75.0},
    "lower": {
        "delta_degrees": -10.0,
        "minimum_feasible_degrees": -75.0,
        "minimum_legacy_degrees": 5.0,
    },
    "dolly_in": {"radius_scale": 0.75, "minimum_distance_m": 0.5},
    "dolly_out": {"radius_scale": 1.25},
}
CAMERA_CANDIDATE_POLICIES = ("local", "legacy")
CAMERA_CANDIDATE_POLICY_ALIASES = {
    "feasible_v2": "local",
    "legacy_v1": "legacy",
}
DEFAULT_CAMERA_CANDIDATE_POLICY = "local"
CAMERA_SENSOR_WIDTH_MM = 36.0
CAMERA_SENSOR_FIT = "HORIZONTAL"
CAMERA_MIN_LENS_MM = 24.0
CAMERA_FRAME_MARGIN_NDC = 0.85


@dataclass(frozen=True)
class _CameraObject:
    id: str
    center: np.ndarray
    half: np.ndarray
    R: np.ndarray
    bottom_z: float
    top_z: float


def validate_camera_pose_mode(value: str | None) -> str | None:
    if value is None:
        return None
    resolved = str(value).strip().lower()
    if resolved not in CAMERA_POSE_MODES:
        raise ValueError(f"camera_pose_mode must be one of {list(CAMERA_POSE_MODES)}, got {value!r}")
    return resolved


def normalize_camera_candidate_policy(value: str | None) -> str:
    """Resolve persisted policy aliases to the current public names."""

    resolved = str(
        DEFAULT_CAMERA_CANDIDATE_POLICY if value is None else value
    ).strip().lower()
    resolved = CAMERA_CANDIDATE_POLICY_ALIASES.get(resolved, resolved)
    if resolved not in CAMERA_CANDIDATE_POLICIES:
        accepted = [
            *CAMERA_CANDIDATE_POLICIES,
            *CAMERA_CANDIDATE_POLICY_ALIASES,
        ]
        raise ValueError(
            f"camera candidate policy must be one of {accepted}"
        )
    return resolved


def validate_metric_camera_modes(value: dict[str, str] | None) -> dict[str, str]:
    """Validate per-metric camera-policy overrides.

    ``auto`` is deliberately rejected as an override value: an override must
    resolve to one concrete evidence policy. The global provider mode may be
    ``auto`` and then falls back to :data:`DEFAULT_CAMERA_MODE_BY_METRIC`.
    """

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metric camera modes must be a mapping")
    resolved: dict[str, str] = {}
    for raw_metric, raw_mode in value.items():
        metric = str(raw_metric).strip().lower()
        if metric not in CAMERA_EVIDENCE_METRICS:
            raise ValueError(
                "camera metric must be one of "
                f"{list(CAMERA_EVIDENCE_METRICS)}, got {raw_metric!r}"
            )
        mode = validate_camera_pose_mode(raw_mode)
        if mode in {None, "auto"}:
            raise ValueError("per-metric camera mode must resolve to a concrete mode, not auto")
        if mode == "support_contact_plane" and metric != "support":
            raise ValueError("support_contact_plane is only valid for the support metric")
        resolved[metric] = mode
    return resolved


def resolve_camera_pose_mode(
    mode: str,
    metric: str,
    *,
    metric_modes: dict[str, str] | None = None,
) -> str:
    """Resolve one request to a concrete camera evidence policy."""

    base = validate_camera_pose_mode(mode)
    if base is None:
        raise ValueError("camera evidence provider requires an active mode")
    metric_name = str(metric).strip().lower()
    overrides = validate_metric_camera_modes(metric_modes)
    if metric_name in overrides:
        return overrides[metric_name]
    if base == "auto":
        return DEFAULT_CAMERA_MODE_BY_METRIC.get(metric_name, "global_only")
    if base == "support_contact_plane" and metric_name != "support":
        raise ValueError("support_contact_plane camera mode is only valid for support events")
    return base


def parse_metric_camera_modes(values: list[str] | tuple[str, ...] | None) -> dict[str, str]:
    """Parse repeatable CLI values formatted as ``METRIC=MODE``."""

    result: dict[str, str] = {}
    for raw in values or []:
        text = str(raw).strip()
        if "=" not in text:
            raise ValueError(f"camera metric override must be METRIC=MODE, got {raw!r}")
        metric, mode = (part.strip() for part in text.split("=", 1))
        if not metric or not mode:
            raise ValueError(f"camera metric override must be METRIC=MODE, got {raw!r}")
        result[metric] = mode
    return validate_metric_camera_modes(result)


def generate_camera_pose_candidates(
    request: dict[str, Any],
    *,
    max_candidates: int = 6,
    policy: str = DEFAULT_CAMERA_CANDIDATE_POLICY,
) -> list[dict[str, Any]]:
    """Create a frozen metric-aware pose bank from canonical scene geometry.

    ``local`` is the production policy. It preserves intended camera
    rays, uses detector-localized event focus when available, and validates
    proxy framing in screen space. ``legacy`` is retained only so frozen
    camera-policy experiments can be reproduced without changing arm meaning.
    """

    policy_name = normalize_camera_candidate_policy(policy)
    if policy_name == "legacy":
        return _generate_legacy_camera_pose_candidates(
            request,
            max_candidates=max_candidates,
        )
    return _generate_feasible_camera_pose_candidates(
        request,
        max_candidates=max_candidates,
    )


def _generate_legacy_camera_pose_candidates(
    request: dict[str, Any],
    *,
    max_candidates: int = 6,
) -> list[dict[str, Any]]:
    """Create a frozen, metric-aware pose bank from canonical scene geometry.

    The function uses only the event targets and their OBB-derived world bounds.
    It does not inspect a rendered result or make a metric judgement.
    """

    if not isinstance(request, dict):
        raise TypeError("camera evidence request must be a JSON object")
    scene = request.get("scene")
    if not isinstance(scene, dict):
        raise ValueError("camera evidence request requires a canonical scene")
    count = max(1, min(12, int(max_candidates)))
    room = _room_bounds(scene)
    target_object_ids = _object_id_list(request.get("object_ids"))
    objects = _legacy_target_objects(scene, target_object_ids)
    metric = str(request.get("metric") or "event")
    resolved_mode = str(request.get("_resolved_camera_pose_mode") or "").strip().lower()
    support_contact_policy = metric == "support" and resolved_mode in {
        "support_contact_plane",
        "query_cov",
    }
    framing_objects = objects[:1] if support_contact_policy and objects else objects
    target_bounds = (
        _union_bounds(framing_objects)
        if framing_objects
        else _fallback_target_bounds(scene, room)
    )
    support_focus: dict[str, Any] | None = None
    if support_contact_policy:
        support_focus = _support_contact_focus(request, framing_objects, target_bounds, room)
        target = np.asarray(support_focus["target"], dtype=float)
        directions = _support_contact_plane_directions(count)
    else:
        target = _event_target(request, objects, target_bounds, room)
        directions = _metric_directions(metric, request, objects)
    extent = np.maximum(target_bounds[1] - target_bounds[0], 0.1)
    target_span = float(max(extent))
    if support_contact_policy:
        horizontal_span = float(max(extent[0], extent[1], 0.1))
        distance = max(0.8, min(3.5, horizontal_span * 2.5))
        lens = 58.0 if horizontal_span < 1.5 else 52.0
        policy_source = "support_contact_plane_candidate_bank_v1"
    else:
        distance = max(1.2, target_span * 2.2)
        lens = 52.0 if target_span < 2.5 else 45.0
        policy_source = "metric_aware_obb_candidate_bank_v1"

    candidates: list[dict[str, Any]] = []
    for index, (azimuth, elevation, label) in enumerate(directions[:count]):
        if support_contact_policy:
            location, elevation = _support_contact_camera_location(
                target=target,
                horizontal_distance=distance,
                azimuth_degrees=azimuth,
                object_height=float(extent[2]),
                room=room,
            )
        else:
            location = _camera_location(
                target=target,
                distance=distance,
                azimuth_degrees=azimuth,
                elevation_degrees=elevation,
                room=room,
            )
        if float(np.linalg.norm(location - target)) < 0.25:
            continue
        candidates.append(
            {
                "id": f"{metric}_{index:02d}_{label}",
                "name": f"{metric}_{label}",
                "camera_type": "PERSP",
                "location": _vector_list(location),
                "target": _vector_list(target),
                "lens_mm": lens,
                "clip_start_m": 0.02,
                "clip_end_m": max(100.0, room[5] * 10.0),
                "azimuth_degrees": float(azimuth),
                "elevation_degrees": float(elevation),
                "target_object_ids": target_object_ids,
                "target_bounds": [_vector_list(target_bounds[0]), _vector_list(target_bounds[1])],
                "room_bounds": [float(value) for value in room],
                "policy_source": policy_source,
                **({"support_contact_focus": deepcopy(support_focus)} if support_focus else {}),
            }
        )
    if not candidates:
        raise ValueError("camera candidate generation produced no valid poses")
    return candidates


def _generate_feasible_camera_pose_candidates(
    request: dict[str, Any],
    *,
    max_candidates: int = 6,
) -> list[dict[str, Any]]:
    """Generate an exact-size bank of feasible, truthfully described poses.

    This policy deliberately keeps proposal generation deterministic and
    renderer-independent.  In contrast to ``legacy``, room feasibility is
    solved along the intended camera ray rather than by independently clamping
    XYZ, so azimuth/elevation labels continue to describe the returned pose.
    """

    if not isinstance(request, dict):
        raise TypeError("camera evidence request must be a JSON object")
    scene = request.get("scene")
    if not isinstance(scene, dict):
        raise ValueError("camera evidence request requires a canonical scene")

    count = max(1, min(12, int(max_candidates)))
    room = _room_bounds(scene)
    target_object_ids = _object_id_list(request.get("object_ids"))
    objects = _target_objects(scene, target_object_ids)
    scene_objects = _target_objects(scene, [])
    metric = str(request.get("metric") or "event").strip().lower()
    resolved_mode = str(request.get("_resolved_camera_pose_mode") or "").strip().lower()
    support_contact_policy = metric == "support" and resolved_mode in {
        "support_contact_plane",
        "query_cov",
    }

    # _target_objects preserves request order.  Support requests put the
    # subject first and candidate supporting objects afterwards, so this is now
    # the intended subject rather than whichever object happened to be first in
    # scene serialization order.
    framing_objects = objects[:1] if support_contact_policy and objects else objects
    target_bounds = (
        _union_bounds(framing_objects)
        if framing_objects
        else _fallback_target_bounds(scene, room)
    )
    extent = np.maximum(target_bounds[1] - target_bounds[0], 0.1)

    support_focus: dict[str, Any] | None = None
    event_focus_source = "bounds_center"
    event_focus_radius_m: float | None = None
    axis_source: str | None = None
    render_aspect_ratio = _render_aspect_ratio(request)
    if support_contact_policy:
        support_focus = _support_contact_focus(request, framing_objects, target_bounds, room)
        target = np.asarray(support_focus["target"], dtype=float)
        support_proxy_bounds = _support_focus_bounds(
            support_focus,
            target_bounds,
        )
        horizontal_span = float(max(extent[0], extent[1], 0.1))
        desired_distance = max(0.8, min(3.5, horizontal_span * 2.5))
        base_lens = 58.0 if horizontal_span < 1.5 else 52.0
        specifications = _support_feasible_specifications(
            target=target,
            desired_horizontal_distance=desired_distance,
            object_height=float(extent[2]),
            room=room,
            count=count,
            framing_bounds=support_proxy_bounds,
        )
        policy_source = "support_contact_plane_candidate_bank_v2"
        event_focus_source = str(support_focus.get("source") or "support_contact_focus")
    elif metric in {"oob", "object_architecture_penetration"}:
        desired_distance = max(1.2, float(max(extent)) * 2.2)
        base_lens = 52.0 if float(max(extent)) < 2.5 else 45.0
        specifications = _oob_feasible_specifications(
            request=request,
            bounds=target_bounds,
            room=room,
            count=count,
        )
        policy_source = "metric_aware_feasible_candidate_bank_v2"
        event_focus_source = "active_room_plane_proxy"
    elif metric == "collision" and len(objects) >= 2:
        target, event_focus_source, event_focus_radius_m = _collision_event_focus(
            request,
            objects,
            target_bounds,
        )
        axis, axis_source = _collision_axis_degrees(request, objects)
        desired_distance = max(1.2, float(max(extent)) * 2.2)
        base_lens = 52.0 if float(max(extent)) < 2.5 else 45.0
        specifications = _collision_feasible_specifications(
            focus_target=target,
            context_target=(target_bounds[0] + target_bounds[1]) / 2.0,
            focus_bounds=_focus_region_bounds(target, event_focus_radius_m),
            context_bounds=target_bounds,
            axis_degrees=axis,
            count=count,
        )
        policy_source = "metric_aware_feasible_candidate_bank_v2"
    else:
        target = _event_target(request, objects, target_bounds, room)
        desired_distance = max(1.2, float(max(extent)) * 2.2)
        base_lens = 52.0 if float(max(extent)) < 2.5 else 45.0
        specifications = _expanded_angle_specifications(
            target=target,
            templates=_metric_directions(metric, request, objects),
            requested_count=count,
        )
        policy_source = "metric_aware_feasible_candidate_bank_v2"

    candidates: list[dict[str, Any]] = []
    for specification in specifications:
        candidate_target = np.asarray(specification["target"], dtype=float)
        raw_framing_bounds = specification.get("framing_bounds")
        candidate_framing_bounds = (
            (
                np.asarray(raw_framing_bounds[0], dtype=float),
                np.asarray(raw_framing_bounds[1], dtype=float),
            )
            if isinstance(raw_framing_bounds, list)
            and len(raw_framing_bounds) == 2
            else target_bounds
        )
        direction = _direction_from_angles(
            float(specification["azimuth_degrees"]),
            float(specification["elevation_degrees"]),
        )
        requested_distance = float(specification.get("distance_m") or desired_distance)
        placement = _place_on_feasible_ray(
            target=candidate_target,
            direction=direction,
            desired_distance=requested_distance,
            room=room,
        )
        if placement is None:
            continue
        location, actual_distance, feasibility = placement
        if any(_point_inside_object_obb(location, obj, clearance_m=0.03) for obj in scene_objects):
            continue
        actual_azimuth, actual_elevation, measured_distance = _pose_angles(location, candidate_target)
        lens, framing = _fit_proxy_framing_lens(
            location=location,
            target=candidate_target,
            bounds=candidate_framing_bounds,
            preferred_lens_mm=base_lens,
            aspect_ratio=render_aspect_ratio,
        )
        if not framing.get("proxy_bounds_fit"):
            maximum_distance = float(feasibility["feasible_distance_interval_m"][1]) - 1.0e-4
            if maximum_distance > measured_distance + 1.0e-4:
                location = candidate_target + direction * maximum_distance
                if any(
                    _point_inside_object_obb(location, obj, clearance_m=0.03)
                    for obj in scene_objects
                ):
                    continue
                actual_azimuth, actual_elevation, measured_distance = _pose_angles(
                    location,
                    candidate_target,
                )
                feasibility = {
                    **feasibility,
                    "distance_truncated": True,
                    "actual_distance_m": measured_distance,
                    "framing_distance_extended": True,
                }
                lens, framing = _fit_proxy_framing_lens(
                    location=location,
                    target=candidate_target,
                    bounds=candidate_framing_bounds,
                    preferred_lens_mm=base_lens,
                    aspect_ratio=render_aspect_ratio,
                )
        framing["validation_status"] = (
            "fits_proxy_bounds"
            if framing.get("proxy_bounds_fit")
            else "proxy_fit_unresolved_at_room_limit"
        )
        candidate_index = len(candidates)
        label = str(specification["label"])
        candidate = {
            "id": f"{metric}_{candidate_index:02d}_{label}",
            "name": f"{metric}_{label}",
            "camera_type": "PERSP",
            "location": _vector_list(location),
            "target": _vector_list(candidate_target),
            "lens_mm": float(lens),
            "sensor_width_mm": CAMERA_SENSOR_WIDTH_MM,
            "sensor_fit": CAMERA_SENSOR_FIT,
            "clip_start_m": 0.02,
            "clip_end_m": max(100.0, room[5] * 10.0),
            "azimuth_degrees": actual_azimuth,
            "elevation_degrees": actual_elevation,
            "distance_m": measured_distance,
            "intended_azimuth_degrees": float(specification["azimuth_degrees"]) % 360.0,
            "intended_elevation_degrees": float(specification["elevation_degrees"]),
            "intended_distance_m": requested_distance,
            "target_object_ids": target_object_ids,
            "target_bounds": [_vector_list(target_bounds[0]), _vector_list(target_bounds[1])],
            "proxy_framing_bounds": [
                _vector_list(candidate_framing_bounds[0]),
                _vector_list(candidate_framing_bounds[1]),
            ],
            "room_bounds": [float(value) for value in room],
            "policy_source": policy_source,
            "candidate_policy": "local",
            "event_focus_source": event_focus_source,
            "focus_kind": str(specification.get("focus_kind") or "event"),
            **(
                {"event_focus_radius_m": event_focus_radius_m}
                if event_focus_radius_m is not None
                else {}
            ),
            "feasibility": feasibility,
            "proxy_framing": framing,
            "camera_free_space_proxy": {
                "canonical_obb_clearance_checked": True,
                "clearance_m": 0.03,
            },
            **({"collision_axis_source": axis_source} if axis_source else {}),
            **({"support_contact_focus": deepcopy(support_focus)} if support_focus else {}),
            **(
                {"focus_plane_flag": specification["focus_plane_flag"]}
                if specification.get("focus_plane_flag")
                else {}
            ),
        }
        if _duplicates_existing_pose(candidate, candidates):
            continue
        candidates.append(candidate)
        if len(candidates) == count:
            break

    if len(candidates) != count:
        raise ValueError(
            "feasible camera candidate generation could not satisfy the exact "
            f"bank size: requested={count}, generated={len(candidates)}, metric={metric}"
        )
    return candidates


def _collision_event_focus(
    request: dict[str, Any],
    objects: list[_CameraObject],
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, str, float]:
    detector = request.get("detector_evidence")
    detector = detector if isinstance(detector, dict) else {}
    candidates = [detector.get("focus_region")]
    mesh = detector.get("mesh")
    if isinstance(mesh, dict):
        candidates.append(mesh.get("focus_region"))
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        try:
            center = _vector3(raw.get("center"), "collision focus center")
        except ValueError:
            continue
        radius = _finite_float(raw.get("radius_m"), fallback=0.35)
        return center, str(raw.get("source") or "detector_focus_region"), max(0.05, radius)
    if len(objects) >= 2:
        first = _object_bounds(objects[0])
        second = _object_bounds(objects[1])
        overlap_min = np.maximum(first[0], second[0])
        overlap_max = np.minimum(first[1], second[1])
        if np.all(overlap_max >= overlap_min):
            radius = max(0.05, min(0.5, float(np.linalg.norm(overlap_max - overlap_min)) * 0.5))
            return (overlap_min + overlap_max) / 2.0, "world_aabb_overlap_proxy", radius
    return (bounds[0] + bounds[1]) / 2.0, "bounds_center_fallback", 0.5


def _focus_region_bounds(
    center: np.ndarray,
    radius_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    radius = max(0.10, float(radius_m or 0.35))
    extent = np.array([radius, radius, radius], dtype=float)
    return np.asarray(center, dtype=float) - extent, np.asarray(center, dtype=float) + extent


def _support_focus_bounds(
    focus: dict[str, Any],
    subject_bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    support_point = _vector3(focus.get("support_point"), "support point")
    base_point = _vector3(focus.get("base_point"), "support base point")
    object_height = max(0.1, float(subject_bounds[1][2] - subject_bounds[0][2]))
    z_margin = min(0.35, max(0.12, object_height * 0.20))
    minimum = np.asarray(subject_bounds[0], dtype=float).copy()
    maximum = np.asarray(subject_bounds[1], dtype=float).copy()
    minimum[2] = min(float(support_point[2]), float(base_point[2])) - 0.05
    maximum[2] = max(float(support_point[2]), float(base_point[2])) + z_margin
    return minimum, maximum


def _collision_axis_degrees(
    request: dict[str, Any],
    objects: list[_CameraObject],
) -> tuple[float, str]:
    detector = request.get("detector_evidence")
    detector = detector if isinstance(detector, dict) else {}
    closest = detector.get("closest_points")
    if closest is None and isinstance(detector.get("mesh"), dict):
        closest = detector["mesh"].get("closest_points")
    if isinstance(closest, dict):
        try:
            point_a = _vector3(closest.get("object_a"), "closest point A")
            point_b = _vector3(closest.get("object_b"), "closest point B")
            delta = point_b[:2] - point_a[:2]
            if float(np.linalg.norm(delta)) > 1.0e-6:
                return _undirected_axis_degrees(delta), "detector_closest_points"
        except ValueError:
            pass

    center_delta = objects[1].center[:2] - objects[0].center[:2]
    if float(np.linalg.norm(center_delta)) > 1.0e-6:
        return _undirected_axis_degrees(center_delta), "object_center_axis"

    # A vertical/coincident pair has no meaningful XY center axis.  Use the
    # longest available horizontal OBB principal axis, treating every axis as
    # undirected so object order cannot alter the bank.
    principal_axes: list[tuple[float, float]] = []
    for obj in objects[:2]:
        for local_index in (0, 1):
            vector = np.asarray(obj.R[:2, local_index], dtype=float)
            strength = float(obj.half[local_index] * np.linalg.norm(vector))
            if strength > 1.0e-8:
                principal_axes.append((strength, _undirected_axis_degrees(vector)))
    if principal_axes:
        _, axis = max(principal_axes, key=lambda item: (item[0], -item[1]))
        return axis, "longest_horizontal_obb_axis"
    return 0.0, "world_x_last_resort"


def _undirected_axis_degrees(delta_xy: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(delta_xy[1]), float(delta_xy[0]))) % 180.0)


def _collision_feasible_specifications(
    *,
    focus_target: np.ndarray,
    context_target: np.ndarray,
    focus_bounds: tuple[np.ndarray, np.ndarray],
    context_bounds: tuple[np.ndarray, np.ndarray],
    axis_degrees: float,
    count: int,
) -> list[dict[str, Any]]:
    primary = (axis_degrees + 90.0) % 360.0
    base = [
        (primary, 20.0, "separation_side"),
        ((primary + 180.0) % 360.0, 24.0, "separation_reverse"),
        (axis_degrees % 360.0, 38.0, "pair_axis_oblique"),
        ((axis_degrees + 180.0) % 360.0, 38.0, "pair_axis_reverse"),
        ((primary + 45.0) % 360.0, 55.0, "high_oblique"),
        ((primary - 45.0) % 360.0, 12.0, "low_oblique"),
    ]
    # Extra deterministic orbit positions are only reached for larger banks or
    # when a base pose is infeasible/duplicated.
    for index in range(24):
        base.append(
            (
                (axis_degrees + index * 137.507764) % 360.0,
                (18.0, 32.0, 48.0)[index % 3],
                f"refill_{index:02d}",
            )
        )
    result = []
    for index, (azimuth, elevation, label) in enumerate(base[: max(count * 4, count)]):
        context_view = index % 6 in {4, 5}
        target = context_target if context_view else focus_target
        framing_bounds = context_bounds if context_view else focus_bounds
        result.append(
            {
                "target": _vector_list(target),
                "framing_bounds": [
                    _vector_list(framing_bounds[0]),
                    _vector_list(framing_bounds[1]),
                ],
                "focus_kind": "pair_context" if context_view else "collision_focus",
                "azimuth_degrees": azimuth,
                "elevation_degrees": elevation,
                "label": label,
            }
        )
    return result


_OOB_PLANE_ORDER = (
    "west_oob",
    "east_oob",
    "south_oob",
    "north_oob",
    "floor_oob",
    "ceiling_oob",
)


def _oob_feasible_specifications(
    *,
    request: dict[str, Any],
    bounds: tuple[np.ndarray, np.ndarray],
    room: tuple[float, float, float, float, float, float],
    count: int,
) -> list[dict[str, Any]]:
    flags = _plane_flags(request)
    active = [flag for flag in _OOB_PLANE_ORDER if flags.get(flag)]
    if not active:
        return _expanded_angle_specifications(
            target=(bounds[0] + bounds[1]) / 2.0,
            templates=_metric_directions("event", request, []),
            requested_count=count,
        )

    specifications: list[dict[str, Any]] = []
    # Round-robin plane allocation guarantees every active plane receives a
    # candidate before any plane receives a second one. Opposing flags therefore
    # remain separately represented rather than one silently winning an elif.
    variants_per_plane = max(6, math.ceil((count * 5) / len(active)))
    for variant_index in range(variants_per_plane):
        for flag in active:
            target = _oob_plane_target(flag, bounds, room)
            azimuth, elevation, variant = _oob_inward_angles(flag, variant_index)
            specifications.append(
                {
                    "target": _vector_list(target),
                    "azimuth_degrees": azimuth,
                    "elevation_degrees": elevation,
                    "label": f"{flag.removesuffix('_oob')}_{variant}",
                    "focus_plane_flag": flag,
                }
            )
    return specifications


def _oob_plane_target(
    flag: str,
    bounds: tuple[np.ndarray, np.ndarray],
    room: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    target = (bounds[0] + bounds[1]) / 2.0
    index_and_value = {
        "west_oob": (0, room[0]),
        "east_oob": (0, room[1]),
        "south_oob": (1, room[2]),
        "north_oob": (1, room[3]),
        "floor_oob": (2, room[4]),
        "ceiling_oob": (2, room[5]),
    }
    axis, value = index_and_value[flag]
    target[axis] = float(value)
    return target


def _oob_inward_angles(flag: str, variant_index: int) -> tuple[float, float, str]:
    variant = variant_index % 12
    if flag in {"floor_oob", "ceiling_oob"}:
        azimuths = (45.0, 135.0, 225.0, 315.0, 0.0, 180.0, 90.0, 270.0, 30.0, 150.0, 210.0, 330.0)
        magnitudes = (55.0, 45.0, 55.0, 45.0, 70.0, 30.0, 62.0, 38.0, 50.0, 58.0, 42.0, 66.0)
        sign = 1.0 if flag == "floor_oob" else -1.0
        return azimuths[variant], sign * magnitudes[variant], f"inward_{variant:02d}"

    primary = {
        "west_oob": 0.0,
        "east_oob": 180.0,
        "south_oob": 90.0,
        "north_oob": 270.0,
    }[flag]
    offsets = (0.0, 30.0, -30.0, 55.0, -55.0, 0.0, 15.0, -15.0, 45.0, -45.0, 70.0, -70.0)
    elevations = (20.0, 38.0, 12.0, 45.0, 30.0, 55.0, 25.0, 48.0, 18.0, 35.0, 42.0, 28.0)
    return (primary + offsets[variant]) % 360.0, elevations[variant], f"inward_{variant:02d}"


def _support_feasible_specifications(
    *,
    target: np.ndarray,
    desired_horizontal_distance: float,
    object_height: float,
    room: tuple[float, float, float, float, float, float],
    count: int,
    framing_bounds: tuple[np.ndarray, np.ndarray],
) -> list[dict[str, Any]]:
    lower, _ = _room_interior_bounds(room)
    base_height = float(np.clip(0.10 * max(object_height, 0.1), 0.05, 0.25))
    # A focus target may be one centimetre above the floor while valid cameras
    # must be at least 12 cm above it. Make that necessary lift explicit in the
    # intended ray instead of relying on a later Z clamp.
    height = max(base_height, float(lower[2] - target[2] + 0.01))
    elevation = math.degrees(math.atan2(height, desired_horizontal_distance))
    distance = math.hypot(desired_horizontal_distance, height)
    specifications = []
    for index in range(max(count * 4, count)):
        azimuth = (360.0 * index / count) % 360.0 if index < count else (index * 137.507764) % 360.0
        specifications.append(
            {
                "target": _vector_list(target),
                "framing_bounds": [
                    _vector_list(framing_bounds[0]),
                    _vector_list(framing_bounds[1]),
                ],
                "focus_kind": "support_contact_band",
                "azimuth_degrees": azimuth,
                "elevation_degrees": elevation,
                "distance_m": distance,
                "label": f"contact_{int(round(azimuth)) % 360:03d}",
            }
        )
    return specifications


def _expanded_angle_specifications(
    *,
    target: np.ndarray,
    templates: list[tuple[float, float, str]],
    requested_count: int,
) -> list[dict[str, Any]]:
    result = [
        {
            "target": _vector_list(target),
            "azimuth_degrees": azimuth,
            "elevation_degrees": elevation,
            "label": label,
        }
        for azimuth, elevation, label in templates
    ]
    for index in range(max(requested_count * 4 - len(result), 0)):
        result.append(
            {
                "target": _vector_list(target),
                "azimuth_degrees": (index * 137.507764) % 360.0,
                "elevation_degrees": (20.0, 35.0, 50.0)[index % 3],
                "label": f"refill_{index:02d}",
            }
        )
    return result


def _direction_from_angles(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    return np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=float,
    )


def _room_interior_bounds(
    room: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    min_x, max_x, min_y, max_y, floor_z, ceiling_z = room
    margin_x = min(0.2, max(0.02, (max_x - min_x) * 0.03))
    margin_y = min(0.2, max(0.02, (max_y - min_y) * 0.03))
    return (
        np.array([min_x + margin_x, min_y + margin_y, floor_z + 0.12], dtype=float),
        np.array([max_x - margin_x, max_y - margin_y, max(floor_z + 0.2, ceiling_z - 0.12)], dtype=float),
    )


def _place_on_feasible_ray(
    *,
    target: np.ndarray,
    direction: np.ndarray,
    desired_distance: float,
    room: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, float, dict[str, Any]] | None:
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        return None
    ray = np.asarray(direction, dtype=float) / norm
    lower, upper = _room_interior_bounds(room)
    t_enter = -math.inf
    t_exit = math.inf
    for axis in range(3):
        component = float(ray[axis])
        if abs(component) <= 1.0e-10:
            if float(target[axis]) < float(lower[axis]) or float(target[axis]) > float(upper[axis]):
                return None
            continue
        t0 = (float(lower[axis]) - float(target[axis])) / component
        t1 = (float(upper[axis]) - float(target[axis])) / component
        t_enter = max(t_enter, min(t0, t1))
        t_exit = min(t_exit, max(t0, t1))
    minimum = max(0.25, t_enter, 0.0)
    maximum = t_exit
    epsilon = 1.0e-4
    if not math.isfinite(maximum) or maximum < minimum + epsilon:
        return None
    distance = min(max(float(desired_distance), minimum + epsilon), maximum - epsilon)
    if distance < 0.25 or distance > maximum:
        return None
    location = target + ray * distance
    return location, distance, {
        "method": "ray_box_interval_v2",
        "ray_preserved": True,
        "distance_truncated": bool(abs(distance - float(desired_distance)) > 1.0e-6),
        "feasible_distance_interval_m": [float(minimum), float(maximum)],
        "actual_distance_m": float(distance),
    }


def _pose_angles(location: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    delta = np.asarray(location, dtype=float) - np.asarray(target, dtype=float)
    distance = float(np.linalg.norm(delta))
    horizontal = float(np.linalg.norm(delta[:2]))
    azimuth = float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0)
    elevation = float(math.degrees(math.atan2(float(delta[2]), horizontal)))
    return azimuth, elevation, distance


def _fit_proxy_framing_lens(
    *,
    location: np.ndarray,
    target: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    preferred_lens_mm: float,
    aspect_ratio: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    projection = _proxy_projection(
        location,
        target,
        bounds,
        preferred_lens_mm,
        aspect_ratio=aspect_ratio,
    )
    allowed_lens = projection.get("maximum_fitting_lens_mm")
    lens = float(preferred_lens_mm)
    if isinstance(allowed_lens, (int, float)) and math.isfinite(float(allowed_lens)):
        lens = min(lens, max(CAMERA_MIN_LENS_MM, float(allowed_lens)))
    diagnostics = _proxy_projection(
        location,
        target,
        bounds,
        lens,
        aspect_ratio=aspect_ratio,
    )
    diagnostics.update(
        {
            "preferred_lens_mm": float(preferred_lens_mm),
            "selected_lens_mm": lens,
            "lens_adjusted": bool(abs(lens - float(preferred_lens_mm)) > 1.0e-6),
            "sensor_width_mm": CAMERA_SENSOR_WIDTH_MM,
            "sensor_fit": CAMERA_SENSOR_FIT,
            "aspect_ratio": float(aspect_ratio),
            "safe_ndc_limit": CAMERA_FRAME_MARGIN_NDC,
        }
    )
    return lens, diagnostics


def _proxy_projection(
    location: np.ndarray,
    target: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    lens_mm: float,
    *,
    aspect_ratio: float = 1.0,
) -> dict[str, Any]:
    forward = np.asarray(target, dtype=float) - np.asarray(location, dtype=float)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1.0e-9:
        return {"all_corners_in_front": False, "proxy_bounds_fit": False}
    forward /= forward_norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) <= 1.0e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        right /= float(np.linalg.norm(right))
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1.0e-9)
    corners = _bounds_corners(bounds)
    max_abs_x = 0.0
    max_abs_y = 0.0
    maximum_fitting_lens = math.inf
    all_in_front = True
    half_sensor = CAMERA_SENSOR_WIDTH_MM / 2.0
    vertical_half_sensor = half_sensor / max(float(aspect_ratio), 1.0e-6)
    for corner in corners:
        relative = corner - location
        depth = float(np.dot(relative, forward))
        if depth <= 1.0e-6:
            all_in_front = False
            continue
        horizontal = abs(float(np.dot(relative, right)))
        vertical = abs(float(np.dot(relative, up)))
        x_ndc = horizontal * float(lens_mm) / (depth * half_sensor)
        y_ndc = vertical * float(lens_mm) / (depth * vertical_half_sensor)
        max_abs_x = max(max_abs_x, x_ndc)
        max_abs_y = max(max_abs_y, y_ndc)
        if horizontal > 1.0e-9:
            maximum_fitting_lens = min(
                maximum_fitting_lens,
                CAMERA_FRAME_MARGIN_NDC * depth * half_sensor / horizontal,
            )
        if vertical > 1.0e-9:
            maximum_fitting_lens = min(
                maximum_fitting_lens,
                CAMERA_FRAME_MARGIN_NDC * depth * vertical_half_sensor / vertical,
            )
    fit = all_in_front and max(max_abs_x, max_abs_y) <= CAMERA_FRAME_MARGIN_NDC + 1.0e-9
    return {
        "all_corners_in_front": all_in_front,
        "max_abs_ndc_x": float(max_abs_x),
        "max_abs_ndc_y": float(max_abs_y),
        "proxy_bounds_fit": bool(fit),
        "maximum_fitting_lens_mm": (
            float(maximum_fitting_lens) if math.isfinite(maximum_fitting_lens) else None
        ),
    }


def _bounds_corners(bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    minimum, maximum = bounds
    return np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=float,
    )


def _duplicates_existing_pose(
    candidate: dict[str, Any],
    existing: list[dict[str, Any]],
    *,
    position_tolerance_m: float = 0.02,
    angular_tolerance_degrees: float = 2.0,
) -> bool:
    location = np.asarray(candidate["location"], dtype=float)
    target = np.asarray(candidate["target"], dtype=float)
    view = target - location
    view /= max(float(np.linalg.norm(view)), 1.0e-9)
    for other in existing:
        other_location = np.asarray(other["location"], dtype=float)
        if float(np.linalg.norm(location - other_location)) > position_tolerance_m:
            continue
        other_target = np.asarray(other["target"], dtype=float)
        other_view = other_target - other_location
        other_view /= max(float(np.linalg.norm(other_view)), 1.0e-9)
        angle = math.degrees(math.acos(float(np.clip(np.dot(view, other_view), -1.0, 1.0))))
        if angle <= angular_tolerance_degrees:
            return True
    return False


def select_bbox_track_views(candidates: list[dict[str, Any]], *, max_views: int = 2) -> list[dict[str, Any]]:
    """Select the frozen leading complementary poses for deterministic mode."""

    if not candidates:
        raise ValueError("bbox_track requires at least one camera candidate")
    count = max(1, int(max_views))
    return [deepcopy(item) for item in candidates[:count]]


def generate_global_context_poses(scene: dict[str, Any]) -> list[dict[str, Any]]:
    """Reproduce the benchmark's frozen top and perspective overview poses.

    These poses are used only for highlighted global correspondence. They match
    the camera definitions in ``blender_worker.py`` and never depend on the
    metric, target, or backing VLM.
    """

    if not isinstance(scene, dict):
        raise TypeError("global camera poses require a canonical scene")
    min_x, max_x, min_y, max_y, floor_z, ceiling_z = _room_bounds(scene)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 0.5)
    target = [center_x, center_y, min(floor_z + (ceiling_z - floor_z) * 0.4, floor_z + 1.2)]
    common = {
        "target": target,
        "lens_mm": 48.0,
        "sensor_width_mm": CAMERA_SENSOR_WIDTH_MM,
        "sensor_fit": CAMERA_SENSOR_FIT,
        "clip_start_m": 0.02,
        "clip_end_m": max(100.0, ceiling_z * 10.0),
        "target_object_ids": [],
        "room_bounds": [min_x, max_x, min_y, max_y, floor_z, ceiling_z],
        "policy_source": "frozen_global_context_v1",
    }
    return [
        {
            **common,
            "id": "global_top",
            "name": "global_top",
            "camera_type": "ORTHO",
            "location": [center_x, center_y, ceiling_z + max(5.0, span * 1.15)],
            "ortho_scale": span * 1.15,
        },
        {
            **common,
            "id": "global_perspective",
            "name": "global_perspective",
            "camera_type": "PERSP",
            "location": [
                center_x + span * 0.85,
                center_y - span * 0.95,
                ceiling_z + span * 0.70,
            ],
        },
    ]


def apply_camera_action(
    pose: dict[str, Any],
    action: str,
    *,
    scene: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one bounded CoV-style action and revalidate its proxy geometry.

    ``local`` actions preserve the original look-at target, stay on a
    feasible room-interior ray, and, when the canonical scene is available,
    reject camera locations inside any object OBB.  Proxy framing is recomputed
    after motion so an action cannot silently turn a fitted proposal into a
    clipped one.
    """

    action_name = str(action)
    if action_name not in CAMERA_ACTIONS:
        raise ValueError(f"camera action must be one of {list(CAMERA_ACTIONS)}, got {action!r}")
    result = deepcopy(pose)
    location = _vector3(result.get("location"), "camera location")
    target = _vector3(result.get("target"), "camera target")
    delta = location - target
    radius = max(0.25, float(np.linalg.norm(delta)))
    azimuth = math.atan2(float(delta[1]), float(delta[0]))
    elevation = math.asin(float(np.clip(delta[2] / radius, -1.0, 1.0)))
    raw_policy = result.get("candidate_policy")
    # Frozen legacy candidates did not record this field.  Keep that exact
    # behavior, while rejecting rather than silently treating unknown values
    # as legacy.
    effective_policy = (
        "legacy"
        if raw_policy is None
        else normalize_camera_candidate_policy(str(raw_policy))
    )
    feasible_policy = effective_policy == "local"
    if action_name == "orbit_left":
        azimuth += math.radians(
            float(CAMERA_ACTION_PARAMETERS[action_name]["delta_degrees"])
        )
    elif action_name == "orbit_right":
        azimuth += math.radians(
            float(CAMERA_ACTION_PARAMETERS[action_name]["delta_degrees"])
        )
    elif action_name == "elevate":
        elevation = min(
            math.radians(
                float(CAMERA_ACTION_PARAMETERS[action_name]["maximum_degrees"])
            ),
            elevation
            + math.radians(
                float(CAMERA_ACTION_PARAMETERS[action_name]["delta_degrees"])
            ),
        )
    elif action_name == "lower":
        minimum_key = (
            "minimum_feasible_degrees"
            if feasible_policy
            else "minimum_legacy_degrees"
        )
        lower_bound = math.radians(
            float(CAMERA_ACTION_PARAMETERS[action_name][minimum_key])
        )
        elevation = max(
            lower_bound,
            elevation
            + math.radians(
                float(CAMERA_ACTION_PARAMETERS[action_name]["delta_degrees"])
            ),
        )
    elif action_name == "dolly_in":
        radius = max(
            float(CAMERA_ACTION_PARAMETERS[action_name]["minimum_distance_m"]),
            radius
            * float(CAMERA_ACTION_PARAMETERS[action_name]["radius_scale"]),
        )
    elif action_name == "dolly_out":
        radius *= float(CAMERA_ACTION_PARAMETERS[action_name]["radius_scale"])
    horizontal = radius * math.cos(elevation)
    moved = target + np.array(
        [
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype=float,
    )
    room = result.get("room_bounds")
    if isinstance(room, list) and len(room) == 6:
        resolved_room = tuple(float(value) for value in room)
        if feasible_policy:
            placement = _place_on_feasible_ray(
                target=target,
                direction=moved - target,
                desired_distance=radius,
                room=resolved_room,
            )
            if placement is None:
                raise ValueError(f"camera action {action_name!r} has no feasible room-interior pose")
            moved, _, feasibility = placement
            result["feasibility"] = feasibility
        else:
            moved = _clamp_to_room(moved, resolved_room)
    validation = {
        "room_interior_checked": bool(
            isinstance(room, list) and len(room) == 6
        ),
        "canonical_obb_clearance_checked": False,
        "proxy_framing_checked": False,
    }
    if isinstance(scene, dict):
        scene_objects = _target_objects(scene, [])
        validation["canonical_obb_clearance_checked"] = True
        if any(
            _point_inside_object_obb(moved, obj, clearance_m=0.03)
            for obj in scene_objects
        ):
            raise ValueError(
                f"camera action {action_name!r} places the camera inside an "
                "object clearance proxy"
            )
    raw_framing_bounds = result.get("proxy_framing_bounds")
    if (
        isinstance(raw_framing_bounds, list)
        and len(raw_framing_bounds) == 2
    ):
        try:
            framing_bounds = (
                _vector3(raw_framing_bounds[0], "proxy framing minimum"),
                _vector3(raw_framing_bounds[1], "proxy framing maximum"),
            )
            prior_framing = (
                result.get("proxy_framing")
                if isinstance(result.get("proxy_framing"), dict)
                else {}
            )
            aspect_ratio = float(prior_framing.get("aspect_ratio") or 1.0)
            lens, framing = _fit_proxy_framing_lens(
                location=moved,
                target=target,
                bounds=framing_bounds,
                preferred_lens_mm=float(result.get("lens_mm") or 50.0),
                aspect_ratio=aspect_ratio,
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"camera action {action_name!r} has invalid proxy framing metadata"
            ) from None
        validation["proxy_framing_checked"] = True
        if not framing.get("proxy_bounds_fit"):
            raise ValueError(
                f"camera action {action_name!r} clips the proxy framing bounds"
            )
        framing["validation_status"] = "fits_proxy_bounds"
        result["lens_mm"] = float(lens)
        result["proxy_framing"] = framing
    actual_azimuth, actual_elevation, actual_distance = _pose_angles(moved, target)
    result["location"] = _vector_list(moved)
    result["intended_azimuth_degrees"] = float(math.degrees(azimuth) % 360.0)
    result["intended_elevation_degrees"] = float(math.degrees(elevation))
    result["intended_distance_m"] = float(radius)
    result["azimuth_degrees"] = actual_azimuth
    result["elevation_degrees"] = actual_elevation
    result["distance_m"] = actual_distance
    result["id"] = f"{result.get('id') or 'view'}__{action_name}"
    result["name"] = f"{result.get('name') or 'view'} {action_name}"
    result["parent_view_id"] = pose.get("id")
    result["camera_action"] = action_name
    result["camera_action_protocol"] = CAMERA_ACTION_PROTOCOL_VERSION
    result["camera_action_parameters"] = deepcopy(
        CAMERA_ACTION_PARAMETERS[action_name]
    )
    result["active_action_validation"] = validation
    result["policy_source"] = CAMERA_ACTION_PROTOCOL_VERSION
    return result


def _target_objects(scene: dict[str, Any], object_ids: Any) -> list[Any]:
    """Resolve objects in request order, falling back to scene order.

    Request order carries metric roles (notably Support subject first).  A set
    membership filter used to discard those roles and made camera geometry
    depend on arbitrary ``scene.objects`` serialization order.
    """

    requested = _object_id_list(object_ids)
    raw_objects = [raw for raw in scene.get("objects", []) if isinstance(raw, dict)]
    by_id = {str(raw.get("id")): raw for raw in raw_objects if raw.get("id") is not None}
    ordered_raw = [by_id[object_id] for object_id in requested if object_id in by_id]
    if not requested:
        ordered_raw = raw_objects

    objects = []
    for raw in ordered_raw:
        try:
            center = _vector3(raw.get("center"), "object center")
            asset_proxy = raw.get("asset_proxy") if isinstance(raw.get("asset_proxy"), dict) else {}
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            size_value = raw.get("size") or asset_proxy.get("bbox_size") or metadata.get("transformed_size")
            size = _vector3(size_value, "object size")
            if np.any(size <= 0.0):
                raise ValueError("object size must be positive")
            rotation_value = raw.get("rotation")
            if rotation_value is None:
                rotation_value = [0.0, 0.0, float(raw.get("yaw_degrees") or 0.0)]
            rotation = _vector3(rotation_value, "object rotation")
            matrix = _rotation_matrix_from_euler(rotation)
            half = size / 2.0
            provisional = _CameraObject(
                id=str(raw.get("id")),
                center=center,
                half=half,
                R=matrix,
                bottom_z=0.0,
                top_z=0.0,
            )
            minimum, maximum = _object_bounds(provisional)
            objects.append(
                _CameraObject(
                    id=provisional.id,
                    center=center,
                    half=half,
                    R=matrix,
                    bottom_z=float(minimum[2]),
                    top_z=float(maximum[2]),
                )
            )
        except (TypeError, ValueError):
            continue
    return objects


def _legacy_target_objects(scene: dict[str, Any], object_ids: Any) -> list[Any]:
    """Retain the v1 scene-serialization ordering for frozen experiments."""

    requested = set(_object_id_list(object_ids))
    scene_order = [
        str(raw.get("id"))
        for raw in scene.get("objects", [])
        if isinstance(raw, dict)
        and raw.get("id") is not None
        and (not requested or str(raw.get("id")) in requested)
    ]
    return _target_objects(scene, scene_order)


def _point_inside_object_obb(
    point: np.ndarray,
    obj: _CameraObject,
    *,
    clearance_m: float = 0.0,
) -> bool:
    local = obj.R.T @ (np.asarray(point, dtype=float) - obj.center)
    return bool(np.all(np.abs(local) <= obj.half + max(0.0, float(clearance_m))))


def _object_bounds(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    local = np.array(
        [
            [sx * obj.half[0], sy * obj.half[1], sz * obj.half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )
    corners = obj.center + local @ obj.R.T
    return np.min(corners, axis=0), np.max(corners, axis=0)


def _union_bounds(objects: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    bounds = [_object_bounds(obj) for obj in objects]
    return (
        np.min(np.stack([item[0] for item in bounds]), axis=0),
        np.max(np.stack([item[1] for item in bounds]), axis=0),
    )


def _fallback_target_bounds(scene: dict[str, Any], room: tuple[float, float, float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    all_objects = _target_objects(scene, [])
    if all_objects:
        return _union_bounds(all_objects)
    min_x, max_x, min_y, max_y, floor_z, ceiling_z = room
    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (floor_z + ceiling_z) / 2.0])
    return center - 0.25, center + 0.25


def _event_target(
    request: dict[str, Any],
    objects: list[Any],
    bounds: tuple[np.ndarray, np.ndarray],
    room: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    target = (bounds[0] + bounds[1]) / 2.0
    metric = str(request.get("metric") or "")
    if metric == "support" and objects:
        source = objects[0]
        target[2] = max(room[4] + 0.03, float(source.bottom_z) + 0.03)
    elif metric in {"oob", "object_architecture_penetration"} and objects:
        flags = _plane_flags(request)
        if flags.get("west_oob"):
            target[0] = room[0]
        elif flags.get("east_oob"):
            target[0] = room[1]
        if flags.get("south_oob"):
            target[1] = room[2]
        elif flags.get("north_oob"):
            target[1] = room[3]
        if flags.get("floor_oob"):
            target[2] = room[4]
        elif flags.get("ceiling_oob"):
            target[2] = room[5]
    return target


def _support_contact_focus(
    request: dict[str, Any],
    objects: list[Any],
    bounds: tuple[np.ndarray, np.ndarray],
    room: tuple[float, float, float, float, float, float],
) -> dict[str, Any]:
    """Locate the measured gap between a Support subject and its nearest surface.

    Support detector ray hits record a point on the candidate support surface and
    the positive vertical clearance to the subject base. The midpoint of that
    segment is the camera target. This is deterministic evidence routing only;
    it does not infer whether the gap is plausible or issue a metric verdict.
    """

    detector = request.get("detector_evidence")
    detector = detector if isinstance(detector, dict) else {}
    raw_hits = detector.get("representative_ray_hits")
    hits: list[tuple[bool, float, np.ndarray, dict[str, Any]]] = []
    if isinstance(raw_hits, list):
        for raw in raw_hits:
            if not isinstance(raw, dict):
                continue
            try:
                position = _vector3(raw.get("position"), "support ray-hit position")
                gap = float(raw.get("gap_m"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(gap) or gap < 0.0:
                continue
            hits.append((bool(raw.get("is_center")), gap, position, raw))

    if hits:
        # Prefer the semantically stable center probe; otherwise use the smallest
        # measured positive clearance as the conservative suspicious location.
        _, gap, support_point, raw_hit = min(
            hits,
            key=lambda item: (not item[0], item[1]),
        )
        focus_source = "representative_center_ray" if raw_hit.get("is_center") else "minimum_representative_gap"
        support_target = raw_hit.get("target")
    else:
        primary = objects[0] if objects else None
        center = primary.center if primary is not None else (bounds[0] + bounds[1]) / 2.0
        base_z = _finite_float(detector.get("base_min_z_m"), fallback=float(bounds[0][2]))
        gap = _finite_float(detector.get("minimum_positive_clearance_m"), fallback=0.0)
        support_z = base_z - max(0.0, gap)
        support_point = np.array([float(center[0]), float(center[1]), support_z], dtype=float)
        focus_source = "detector_base_gap_fallback"
        support_target = None

    base_point = support_point + np.array([0.0, 0.0, max(0.0, gap)], dtype=float)
    target = (support_point + base_point) / 2.0
    target[2] = float(np.clip(target[2], room[4] + 0.01, room[5] - 0.01))
    return {
        "target": _vector_list(target),
        "support_point": _vector_list(support_point),
        "base_point": _vector_list(base_point),
        "gap_m": float(max(0.0, gap)),
        "support_target": support_target,
        "source": focus_source,
    }


def _support_contact_plane_directions(count: int) -> list[tuple[float, float, str]]:
    """Generate a full-circle low-angle bank without privileging one room axis."""

    resolved_count = max(1, int(count))
    return [
        (
            360.0 * index / resolved_count,
            0.0,  # Replaced by the adaptive contact-plane camera height.
            f"contact_{int(round(360.0 * index / resolved_count)) % 360:03d}",
        )
        for index in range(resolved_count)
    ]


def _support_contact_camera_location(
    *,
    target: np.ndarray,
    horizontal_distance: float,
    azimuth_degrees: float,
    object_height: float,
    room: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, float]:
    """Place a low camera near the contact plane and return its true elevation."""

    azimuth = math.radians(azimuth_degrees)
    height_offset = float(np.clip(0.10 * max(object_height, 0.1), 0.05, 0.25))
    location = target + np.array(
        [
            horizontal_distance * math.cos(azimuth),
            horizontal_distance * math.sin(azimuth),
            height_offset,
        ],
        dtype=float,
    )
    location = _clamp_to_room(location, room)
    delta = location - target
    horizontal = max(1.0e-6, float(np.linalg.norm(delta[:2])))
    elevation = math.degrees(math.atan2(float(delta[2]), horizontal))
    return location, elevation


def _metric_directions(metric: str, request: dict[str, Any], objects: list[Any]) -> list[tuple[float, float, str]]:
    if metric == "support":
        return [
            (45.0, 10.0, "low_ne"),
            (225.0, 12.0, "low_sw"),
            (135.0, 18.0, "low_nw"),
            (315.0, 18.0, "low_se"),
            (45.0, 35.0, "oblique_ne"),
            (225.0, 35.0, "oblique_sw"),
        ]
    if metric in {"oob", "object_architecture_penetration"}:
        flags = _plane_flags(request)
        if flags.get("west_oob"):
            primary = 0.0
        elif flags.get("east_oob"):
            primary = 180.0
        elif flags.get("south_oob"):
            primary = 90.0
        elif flags.get("north_oob"):
            primary = 270.0
        else:
            primary = 45.0
        elevation = 55.0 if flags.get("floor_oob") else 20.0
        if flags.get("ceiling_oob"):
            elevation = 65.0
        return [
            (primary, elevation, "plane_normal"),
            ((primary + 35.0) % 360.0, min(70.0, elevation + 18.0), "plane_oblique"),
            ((primary - 35.0) % 360.0, max(8.0, elevation - 8.0), "plane_complement"),
            ((primary + 180.0) % 360.0, 35.0, "reverse_context"),
            ((primary + 90.0) % 360.0, 40.0, "tangent_left"),
            ((primary - 90.0) % 360.0, 40.0, "tangent_right"),
        ]
    if metric == "collision" and len(objects) >= 2:
        delta = objects[1].center[:2] - objects[0].center[:2]
        axis = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) if np.linalg.norm(delta) > 1.0e-6 else 0.0
        primary = (axis + 90.0) % 360.0
        return [
            (primary, 20.0, "separation_side"),
            ((primary + 180.0) % 360.0, 24.0, "separation_reverse"),
            (axis % 360.0, 38.0, "pair_axis_oblique"),
            ((axis + 180.0) % 360.0, 38.0, "pair_axis_reverse"),
            ((primary + 45.0) % 360.0, 55.0, "high_oblique"),
            ((primary - 45.0) % 360.0, 12.0, "low_oblique"),
        ]
    return [
        (45.0, 30.0, "ne"),
        (225.0, 30.0, "sw"),
        (135.0, 45.0, "nw_high"),
        (315.0, 20.0, "se_low"),
        (0.0, 55.0, "east_high"),
        (180.0, 15.0, "west_low"),
    ]


def _finite_float(value: Any, *, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return result if math.isfinite(result) else float(fallback)


def _render_aspect_ratio(request: dict[str, Any]) -> float:
    render = request.get("_camera_render")
    render = render if isinstance(render, dict) else {}
    width = _finite_float(render.get("width"), fallback=1.0)
    height = _finite_float(render.get("height"), fallback=1.0)
    if width <= 0.0 or height <= 0.0:
        return 1.0
    return float(width / height)


def _plane_flags(request: dict[str, Any]) -> dict[str, bool]:
    event = request.get("event") if isinstance(request.get("event"), dict) else {}
    evidence = request.get("detector_evidence") if isinstance(request.get("detector_evidence"), dict) else {}
    raw = event.get("plane_flags") if isinstance(event.get("plane_flags"), dict) else evidence.get("plane_flags")
    return {str(key): bool(value) for key, value in raw.items()} if isinstance(raw, dict) else {}


def _room_bounds(scene: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    boundary = scene.get("boundary")
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    if not isinstance(boundary, list):
        boundary = room.get("boundary")
    points = [item for item in boundary or [] if isinstance(item, list) and len(item) >= 2]
    if points:
        xs = [float(item[0]) for item in points]
        ys = [float(item[1]) for item in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x, max_x, min_y, max_y = 0.0, 7.0, 0.0, 5.0
    height = scene.get("scene_height", room.get("height", 3.0))
    try:
        ceiling_z = max(0.5, float(height))
    except (TypeError, ValueError):
        ceiling_z = 3.0
    return min_x, max_x, min_y, max_y, 0.0, ceiling_z


def _camera_location(
    *,
    target: np.ndarray,
    distance: float,
    azimuth_degrees: float,
    elevation_degrees: float,
    room: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    horizontal = distance * math.cos(elevation)
    location = target + np.array(
        [
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            distance * math.sin(elevation),
        ],
        dtype=float,
    )
    return _clamp_to_room(location, room)


def _clamp_to_room(location: np.ndarray, room: tuple[float, float, float, float, float, float]) -> np.ndarray:
    min_x, max_x, min_y, max_y, floor_z, ceiling_z = room
    margin_x = min(0.2, max(0.02, (max_x - min_x) * 0.03))
    margin_y = min(0.2, max(0.02, (max_y - min_y) * 0.03))
    return np.array(
        [
            float(np.clip(location[0], min_x + margin_x, max_x - margin_x)),
            float(np.clip(location[1], min_y + margin_y, max_y - margin_y)),
            float(np.clip(location[2], floor_z + 0.12, max(floor_z + 0.2, ceiling_z - 0.12))),
        ],
        dtype=float,
    )


def _vector3(value: Any, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric vector") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite three-vector")
    return result


def _vector_list(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float)]


def _object_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return list(dict.fromkeys(str(item) for item in values if str(item)))


def _rotation_matrix_from_euler(rotation_degrees: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.radians(rotation_degrees)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx
