from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = 1.0e-9


@dataclass(frozen=True)
class NormalizedObject:
    id: str
    jid: str | None
    category: str | None
    retrieval_category: str | None
    desc: str | None
    short_desc: str | None
    center: np.ndarray
    size: np.ndarray
    half: np.ndarray
    rotation: np.ndarray
    yaw_degrees: float
    R: np.ndarray
    right: np.ndarray
    front: np.ndarray
    up: np.ndarray
    bottom: float
    top: float
    bottom_z: float
    top_z: float
    asset_ref: dict
    asset_proxy: dict
    metadata: dict
    interactive: bool


def load_objects(scene: dict) -> list[dict]:
    if not isinstance(scene, dict):
        return []
    value = scene.get("objects")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def get_room_boundary(scene: dict) -> list[list[float]]:
    if not isinstance(scene, dict):
        return []
    boundary = scene.get("boundary")
    if boundary is None and isinstance(scene.get("room"), dict):
        boundary = scene["room"].get("boundary")
    return _boundary_list(boundary)


def get_scene_height(scene: dict) -> float | None:
    if not isinstance(scene, dict):
        return None
    value = scene.get("scene_height")
    if value is None and isinstance(scene.get("room"), dict):
        value = scene["room"].get("height")
    try:
        height = float(value)
    except (TypeError, ValueError):
        return None
    return height if height > 0.0 else None


def normalize_object(obj: dict, *, asset_csv_path: str | None = None, asset_root: str | None = None) -> NormalizedObject:
    if not isinstance(obj, dict):
        raise ValueError("object must be a mapping")
    asset_ref_input = obj.get("asset_ref") if isinstance(obj.get("asset_ref"), dict) else {}
    needs_asset_resolution = bool(asset_csv_path or asset_root or asset_ref_input.get("metadata_uri"))
    if needs_asset_resolution:
        from benchmark.evaluator.generic_validity.asset_resolver import resolve_asset_metadata

        obj = resolve_asset_metadata(obj, asset_csv_path=asset_csv_path, asset_root=asset_root)
    object_id = _first_present(obj, ["id", "object_id"])
    if object_id is None:
        raise ValueError("object id is missing")
    center = _vector3(obj.get("center"))
    if center is None:
        raise ValueError(f"object {object_id!r} center is missing or invalid")
    asset_ref = obj.get("asset_ref") if isinstance(obj.get("asset_ref"), dict) else {}
    asset_proxy = obj.get("asset_proxy") if isinstance(obj.get("asset_proxy"), dict) else {}
    metadata = dict(obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {})
    metadata.setdefault("interactive", False)
    size = _vector3(obj.get("size"))
    if size is None:
        size = _vector3(asset_proxy.get("bbox_size"))
    if size is None:
        size = _vector3(metadata.get("transformed_size"))
    if size is None or np.any(size <= 0):
        raise ValueError(
            f"object {object_id!r} size is missing or invalid; provide size, asset_proxy.bbox_size, metadata transformed_size, or asset_info.csv bbx"
        )
    rotation, yaw_degrees = _rotation_degrees(obj)
    R = rotation_matrix_from_euler(rotation)
    half = size / 2.0
    corners = _obb_corners_from_parts(center, half, R)
    jid = obj.get("jid") or asset_ref.get("asset_key")
    category = obj.get("category")
    retrieval_category = obj.get("retrieval_category") or category
    desc = obj.get("desc")
    short_desc = obj.get("short_desc") or desc
    return NormalizedObject(
        id=str(object_id),
        jid=str(jid) if jid is not None else None,
        category=str(category) if category is not None else None,
        retrieval_category=str(retrieval_category) if retrieval_category is not None else None,
        desc=str(desc) if desc is not None else None,
        short_desc=str(short_desc) if short_desc is not None else None,
        center=center,
        size=size,
        half=half,
        rotation=rotation,
        yaw_degrees=float(yaw_degrees),
        R=R,
        right=_unit(R @ np.array([1.0, 0.0, 0.0])),
        front=_unit(R @ np.array([0.0, -1.0, 0.0])),
        up=_unit(R @ np.array([0.0, 0.0, 1.0])),
        bottom=float(np.min(corners[:, 2])),
        top=float(np.max(corners[:, 2])),
        bottom_z=float(np.min(corners[:, 2])),
        top_z=float(np.max(corners[:, 2])),
        asset_ref=dict(asset_ref),
        asset_proxy=dict(asset_proxy),
        metadata=metadata,
        interactive=is_interactive_object(obj),
    )


def normalize_objects(scene: dict, *, asset_csv_path: str | None = None, asset_root: str | None = None) -> tuple[list[NormalizedObject], dict[str, str]]:
    objects = []
    errors = {}
    for raw_obj in load_objects(scene):
        object_id = _first_present(raw_obj, ["id", "object_id"])
        try:
            objects.append(normalize_object(raw_obj, asset_csv_path=asset_csv_path, asset_root=asset_root))
        except ValueError as exc:
            if object_id is not None:
                errors[str(object_id)] = str(exc)
    return objects, errors


def is_interactive_object(obj: dict) -> bool:
    if bool(obj.get("interactive")):
        return True
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    return bool(metadata.get("interactive"))


def rotation_matrix_from_yaw(yaw_degrees: float) -> np.ndarray:
    yaw = math.radians(float(yaw_degrees))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotation_matrix_from_euler(rotation_degrees: Iterable[float] | np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rotation_degrees]
    rr, pr, yr = np.radians([roll, pitch, yaw])
    cr, sr = math.cos(rr), math.sin(rr)
    cp, sp = math.cos(pr), math.sin(pr)
    cy, sy = math.cos(yr), math.sin(yr)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _rotation_degrees(obj: dict) -> tuple[np.ndarray, float]:
    rotation_value = obj.get("rotation")
    if rotation_value is None:
        yaw_degrees = obj.get("yaw_degrees")
        if yaw_degrees is not None:
            yaw = _safe_float(yaw_degrees, 0.0)
            return np.array([0.0, 0.0, yaw], dtype=float), yaw
        yaw = obj.get("yaw")
        yaw_deg = _angle_to_degrees(_safe_float(yaw, 0.0))
        return np.array([0.0, 0.0, yaw_deg], dtype=float), yaw_deg
    vector = _vector3(rotation_value)
    if vector is None:
        vector = np.zeros(3, dtype=float)
    max_abs = float(np.max(np.abs(vector))) if vector.size else 0.0
    if max_abs <= (2.0 * math.pi + 1.0e-6):
        vector = np.degrees(vector)
    return vector.astype(float), float(vector[2])


def _angle_to_degrees(value: float) -> float:
    return math.degrees(value) if abs(value) <= (2.0 * math.pi + 1.0e-6) else value


def _obb_corners_from_parts(center: np.ndarray, half: np.ndarray, R: np.ndarray) -> np.ndarray:
    local = np.array(
        [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=float,
    )
    return center + local @ R.T


def _boundary_list(value: object) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            continue
    return points


def _vector3(value: object | None) -> np.ndarray | None:
    if isinstance(value, np.ndarray) and value.shape == (3,):
        return value.astype(float)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)
        except (TypeError, ValueError):
            return None
    return None


def _first_present(obj: dict, keys: list[str]) -> object | None:
    for key in keys:
        if isinstance(obj, dict) and key in obj and obj[key] is not None:
            return obj[key]
    return None


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector, dtype=float)
    return vector.astype(float) / norm
