"""Small, dependency-free geometry helpers for harness conversion."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from benchmark.scene_io.validate import (
    CANONICAL_SCENE_SCHEMA_VERSION,
    ArtifactValidationError,
)


def finite_float(value: Any, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{path} must be numeric") from exc
    if not math.isfinite(number):
        raise ArtifactValidationError(f"{path} must be finite")
    return number


def vector3(value: Any, path: str, *, positive: bool = False) -> list[float]:
    if isinstance(value, Mapping):
        if all(axis in value for axis in ("x", "y", "z")):
            value = [value["x"], value["y"], value["z"]]
        elif all(axis in value for axis in ("length", "width", "height")):
            value = [value["length"], value["width"], value["height"]]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise ArtifactValidationError(f"{path} must be a 3-vector")
    result = [finite_float(value[index], f"{path}[{index}]") for index in range(3)]
    if positive and any(item <= 0.0 for item in result):
        raise ArtifactValidationError(f"{path} must contain positive values")
    return result


def canonical_room(generation_input: dict) -> tuple[list[list[float]], float, str]:
    request = generation_input.get("scene_request")
    if not isinstance(request, dict):
        raise ArtifactValidationError("generation_input.scene_request must be a JSON object")
    room = request.get("room")
    if not isinstance(room, dict):
        raise ArtifactValidationError("external harness conversion requires scene_request.room")
    boundary = boundary2(room.get("boundary"), "generation_input.scene_request.room.boundary")
    height = finite_float(room.get("height"), "generation_input.scene_request.room.height")
    if height <= 0.0:
        raise ArtifactValidationError("generation_input.scene_request.room.height must be positive")
    return boundary, height, str(request.get("scene_type") or "room")


def boundary2(value: Any, path: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise ArtifactValidationError(f"{path} must contain at least three points")
    result: list[list[float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            raise ArtifactValidationError(f"{path}[{index}] must contain two coordinates")
        result.append(
            [
                finite_float(point[0], f"{path}[{index}][0]"),
                finite_float(point[1], f"{path}[{index}][1]"),
            ]
        )
    return result


def shift_boundary_to_origin(
    boundary: Iterable[Sequence[float]],
) -> tuple[list[list[float]], list[float]]:
    points = [[float(point[0]), float(point[1])] for point in boundary]
    if len(points) > 3 and points[0] == points[-1]:
        points.pop()
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    return (
        [[point[0] - min_x, point[1] - min_y] for point in points],
        [-min_x, -min_y, 0.0],
    )


def shift_center(center: Sequence[float], offset: Sequence[float]) -> list[float]:
    return [
        float(center[0]) + float(offset[0]),
        float(center[1]) + float(offset[1]),
        float(center[2]) + float(offset[2]),
    ]


def category_from_identifier(value: Any, fallback: str = "object") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = text.split("(", 1)[0]
    text = re.sub(r"\[[0-9]+\]$", "", text)
    text = re.sub(r"[-_][0-9]+$", "", text)
    text = text.rsplit(".", 1)[-1]
    text = re.sub(r"Factory$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[_-]+", " ", text).strip().lower()
    return text or fallback


def quaternion_xyzw_to_matrix(value: Any, path: str) -> list[list[float]]:
    quat = vector4(value, path)
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ArtifactValidationError(f"{path} must not be the zero quaternion")
    x, y, z, w = (component / norm for component in quat)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def vector4(value: Any, path: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        raise ArtifactValidationError(f"{path} must be a 4-vector")
    return [finite_float(value[index], f"{path}[{index}]") for index in range(4)]


def matrix_multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def matrix_transpose(value: list[list[float]]) -> list[list[float]]:
    return [[value[column][row] for column in range(3)] for row in range(3)]


def matrix_vector(value: list[list[float]], vector: Sequence[float]) -> list[float]:
    return [sum(value[row][column] * float(vector[column]) for column in range(3)) for row in range(3)]


def rotation_matrix_to_euler_xyz_degrees(matrix: list[list[float]]) -> list[float]:
    """Return XYZ Euler angles for a matrix composed as Rz @ Ry @ Rx."""

    sy = math.sqrt(matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0])
    singular = sy < 1.0e-9
    if not singular:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = 0.0
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def scene_state_matrix(value: Any, path: str) -> tuple[list[list[float]], list[float], list[float]]:
    """Decode a SceneState column-major 4x4 transform.

    Returns a normalized 3x3 rotation, per-axis scale, and translation.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 16:
        raise ArtifactValidationError(f"{path} must contain 16 column-major values")
    data = [finite_float(item, f"{path}[{index}]") for index, item in enumerate(value)]
    matrix = [[data[column * 4 + row] for column in range(4)] for row in range(4)]
    linear = [[matrix[row][column] for column in range(3)] for row in range(3)]
    scales = [
        math.sqrt(sum(linear[row][column] ** 2 for row in range(3)))
        for column in range(3)
    ]
    if any(scale <= 1.0e-12 for scale in scales):
        raise ArtifactValidationError(f"{path} contains a singular scale")
    rotation = [
        [linear[row][column] / scales[column] for column in range(3)]
        for row in range(3)
    ]
    translation = [matrix[0][3], matrix[1][3], matrix[2][3]]
    return rotation, scales, translation


def build_scene(
    generation_input: dict,
    *,
    adapter_name: str,
    native_schema: str,
    boundary: list[list[float]],
    scene_height: float,
    objects: list[dict],
    coordinate_conversion: dict,
    extra_metadata: dict | None = None,
) -> dict:
    request_id = str(generation_input.get("request_id") or "").strip()
    if not request_id:
        raise ArtifactValidationError("generation_input.request_id must be non-empty")
    request = generation_input.get("scene_request")
    request = request if isinstance(request, dict) else {}
    metadata = {
        "coordinate_frame": {
            "origin": "room_min_corner_floor",
            "axes": "x_width_y_depth_z_up",
            "unit": "meter",
            "rotation_unit": "degree",
        },
        "harness_compatibility": {
            "adapter": adapter_name,
            "native_schema": native_schema,
            "coordinate_conversion": coordinate_conversion,
        },
    }
    if extra_metadata:
        metadata["harness_compatibility"].update(extra_metadata)
    return {
        "schema_version": CANONICAL_SCENE_SCHEMA_VERSION,
        "scene_id": f"{adapter_name}_{request_id}",
        "request_id": request_id,
        "scene_type": str(request.get("scene_type") or "room"),
        "boundary": boundary,
        "scene_height": float(scene_height),
        "objects": objects,
        "metadata": metadata,
    }


__all__ = [
    "boundary2",
    "build_scene",
    "canonical_room",
    "category_from_identifier",
    "finite_float",
    "matrix_multiply",
    "matrix_transpose",
    "matrix_vector",
    "quaternion_xyzw_to_matrix",
    "rotation_matrix_to_euler_xyz_degrees",
    "scene_state_matrix",
    "shift_boundary_to_origin",
    "shift_center",
    "vector3",
    "vector4",
]
