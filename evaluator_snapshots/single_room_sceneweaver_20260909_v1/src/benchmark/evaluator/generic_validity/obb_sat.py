"""Full 3D oriented-bounding-box broad-phase collision via Separating Axis Theorem.

Tests 15 candidate axes: 3 local axes from each OBB plus 9 pairwise cross
products. Uses a frozen numerical epsilon for near-parallel cross-product axes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from benchmark.evaluator.generic_validity.geometry import NormalizedObject


OBB_SAT_BACKEND_VERSION = "obb_sat_v1"
DEFAULT_OBB_SAT_EPS = 1.0e-6
CANONICAL_UNITS = "meter"
CANONICAL_UP_AXIS = "z"
CANONICAL_COORDINATE_FRAME = {
    "origin": "room_min_corner_floor",
    "axes": "x_width_y_depth_z_up",
    "unit": CANONICAL_UNITS,
    "rotation_unit": "degree",
}


def obb_sat_test(
    obj_a: NormalizedObject,
    obj_b: NormalizedObject,
    *,
    eps: float = DEFAULT_OBB_SAT_EPS,
) -> dict[str, Any]:
    """Run a full 3D OBB-vs-OBB SAT test between two normalized objects."""

    return obb_sat_test_parts(
        obj_a.center,
        obj_a.half,
        obj_a.R,
        obj_b.center,
        obj_b.half,
        obj_b.R,
        eps=eps,
    )


def obb_sat_test_parts(
    center_a: np.ndarray,
    half_a: np.ndarray,
    rotation_a: np.ndarray,
    center_b: np.ndarray,
    half_b: np.ndarray,
    rotation_b: np.ndarray,
    *,
    eps: float = DEFAULT_OBB_SAT_EPS,
) -> dict[str, Any]:
    axes = _candidate_axes(rotation_a, rotation_b, eps=eps)
    tested_axis_count = len(axes)
    best_separation = -math.inf
    separating_axis: list[float] | None = None
    min_overlap_depth = math.inf
    overlap_axis: list[float] | None = None

    for axis in axes:
        min_a, max_a = _project_obb_interval(center_a, half_a, rotation_a, axis)
        min_b, max_b = _project_obb_interval(center_b, half_b, rotation_b, axis)
        if max_a < min_b - eps:
            separation = float(min_b - max_a)
            if separation > best_separation:
                best_separation = separation
                separating_axis = axis.tolist()
        elif max_b < min_a - eps:
            separation = float(min_a - max_b)
            if separation > best_separation:
                best_separation = separation
                separating_axis = axis.tolist()
        else:
            overlap_depth = float(min(max_a, max_b) - max(min_a, min_b))
            if overlap_depth < min_overlap_depth:
                min_overlap_depth = overlap_depth
                overlap_axis = axis.tolist()

    intersects = best_separation <= 0.0
    if intersects:
        return {
            "backend": OBB_SAT_BACKEND_VERSION,
            "units": CANONICAL_UNITS,
            "coordinate_frame": dict(CANONICAL_COORDINATE_FRAME),
            "intersects": True,
            "obb_certifiably_separated": False,
            "separating_axis": None,
            "minimum_separation_margin_m": None,
            "minimum_overlap_axis": overlap_axis,
            "minimum_overlap_depth_proxy_m": None if min_overlap_depth == math.inf else float(min_overlap_depth),
            "tested_axis_count": tested_axis_count,
        }
    return {
        "backend": OBB_SAT_BACKEND_VERSION,
        "units": CANONICAL_UNITS,
        "coordinate_frame": dict(CANONICAL_COORDINATE_FRAME),
        "intersects": False,
        "obb_certifiably_separated": True,
        "separating_axis": separating_axis,
        "minimum_separation_margin_m": float(best_separation),
        "minimum_overlap_axis": None,
        "minimum_overlap_depth_proxy_m": None,
        "tested_axis_count": tested_axis_count,
    }


def obb_encloses_points(
    center: np.ndarray,
    half: np.ndarray,
    rotation: np.ndarray,
    points: np.ndarray,
    *,
    eps: float = DEFAULT_OBB_SAT_EPS,
) -> bool:
    """Return whether every world-space point lies inside the OBB."""

    if points.size == 0:
        return True
    local = (np.asarray(points, dtype=float) - center) @ rotation
    return bool(np.all(np.abs(local) <= half + eps))


def _candidate_axes(rotation_a: np.ndarray, rotation_b: np.ndarray, *, eps: float) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for rotation in (rotation_a, rotation_b):
        for index in range(3):
            axis = _unit(rotation[:, index])
            key = _axis_key(axis)
            if key not in seen:
                seen.add(key)
                axes.append(axis)
    for index_a in range(3):
        for index_b in range(3):
            cross = np.cross(rotation_a[:, index_a], rotation_b[:, index_b])
            norm = float(np.linalg.norm(cross))
            if norm <= eps:
                continue
            axis = cross / norm
            key = _axis_key(axis)
            if key not in seen:
                seen.add(key)
                axes.append(axis)
    return axes


def _project_obb_interval(
    center: np.ndarray,
    half: np.ndarray,
    rotation: np.ndarray,
    axis: np.ndarray,
) -> tuple[float, float]:
    center_projection = float(np.dot(center, axis))
    radius = sum(abs(float(np.dot(rotation[:, index], axis))) * float(half[index]) for index in range(3))
    return center_projection - radius, center_projection + radius


def _axis_key(axis: np.ndarray) -> tuple[int, ...]:
    rounded = np.round(axis, decimals=6)
    return tuple(int(value * 1_000_000) for value in rounded)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.zeros(3, dtype=float)
    return vector.astype(float) / norm
