from __future__ import annotations

import math
from itertools import product
from typing import Any, Iterable

from benchmark.materialization.contracts import MaterializationError


TRANSFORM_TOLERANCE_M = 1.0e-6


def finite_vec3(value: Any, path: str, *, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise MaterializationError(f"{path} must be a three-component vector")
    result: list[float] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool):
            raise MaterializationError(f"{path}[{index}] must be numeric, not boolean")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise MaterializationError(f"{path}[{index}] must be numeric") from exc
        if not math.isfinite(number):
            raise MaterializationError(f"{path}[{index}] must be finite")
        if positive and number <= 0.0:
            raise MaterializationError(f"{path}[{index}] must be greater than zero")
        result.append(number)
    return result


def exact_uniform_scale(
    catalog_bbox_size_m: Iterable[float],
    requested_uniform_scale: Any,
) -> dict[str, Any]:
    """Apply a generator-owned uniform scale without contain-fit reinterpretation."""

    source = finite_vec3(
        list(catalog_bbox_size_m), "catalog_bbox_size_m", positive=True
    )
    if isinstance(requested_uniform_scale, bool):
        raise MaterializationError(
            "requested_uniform_scale must be numeric, not boolean"
        )
    try:
        scale = float(requested_uniform_scale)
    except (TypeError, ValueError) as exc:
        raise MaterializationError(
            "requested_uniform_scale must be numeric"
        ) from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise MaterializationError(
            "requested_uniform_scale must be finite and greater than zero"
        )
    actual = [component * scale for component in source]
    return {
        "requested_uniform_scale": scale,
        "effective_uniform_scale": scale,
        "catalog_bbox_size_m": source,
        "actual_local_bbox_size_m": actual,
    }


def rotation_matrix_xyz_degrees(rotation: Iterable[float]) -> list[list[float]]:
    roll, pitch, yaw = finite_vec3(list(rotation), "rotation_euler_xyz_deg")
    rx, ry, rz = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # Intrinsic XYZ for column vectors: Rz @ Ry @ Rx.
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


def world_bounds(
    center_m: Iterable[float],
    local_bbox_size_m: Iterable[float],
    rotation_euler_xyz_deg: Iterable[float],
) -> dict[str, Any]:
    center = finite_vec3(list(center_m), "center_m")
    size = finite_vec3(list(local_bbox_size_m), "local_bbox_size_m", positive=True)
    rotation = finite_vec3(
        list(rotation_euler_xyz_deg),
        "rotation_euler_xyz_deg",
    )
    matrix = rotation_matrix_xyz_degrees(rotation)
    half = [component * 0.5 for component in size]
    corners: list[list[float]] = []
    for signs in product((-1.0, 1.0), repeat=3):
        local = [signs[axis] * half[axis] for axis in range(3)]
        rotated = [
            sum(matrix[row][column] * local[column] for column in range(3))
            for row in range(3)
        ]
        corners.append([center[axis] + rotated[axis] for axis in range(3)])
    minimum = [min(corner[axis] for corner in corners) for axis in range(3)]
    maximum = [max(corner[axis] for corner in corners) for axis in range(3)]
    return {
        "obb": {
            "center_m": center,
            "local_size_m": size,
            "rotation_euler_xyz_deg": rotation,
            "rotation_matrix": matrix,
            "corners_m": corners,
        },
        "aabb": {
            "min_m": minimum,
            "max_m": maximum,
            "size_m": [maximum[axis] - minimum[axis] for axis in range(3)],
        },
    }


def nearly_equal(left: Any, right: Any, *, tolerance: float = TRANSFORM_TOLERANCE_M) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(
            float(left) - float(right)
        ) <= tolerance
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            nearly_equal(a, b, tolerance=tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            nearly_equal(left[key], right[key], tolerance=tolerance) for key in left
        )
    return left == right
