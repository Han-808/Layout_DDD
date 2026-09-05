"""Canonical identities used by controlled generation comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from benchmark.scene_io.validate import ArtifactValidationError


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically, rejecting non-JSON and non-finite values."""

    normalized = _canonical_json_value(value, path="value")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalize_rectangular_architecture(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the v1 comparison identity for one rectangular room.

    Translation, cyclic vertex order, and winding are representation details and
    therefore do not affect this identity. Width/depth orientation and metric
    dimensions do affect it.
    """

    if not isinstance(value, Mapping):
        raise ArtifactValidationError("comparison architecture must be an object")
    room_model = str(value.get("room_model") or "single_room")
    boundary_model = str(value.get("boundary_model") or "axis_aligned_rectangle")
    if room_model != "single_room":
        raise ArtifactValidationError(
            "generation comparison v1 supports only room_model=single_room"
        )
    if boundary_model != "axis_aligned_rectangle":
        raise ArtifactValidationError(
            "generation comparison v1 supports only "
            "boundary_model=axis_aligned_rectangle"
        )
    features = value.get("architecture_features") or []
    if not (
        isinstance(features, Sequence)
        and not isinstance(features, (str, bytes))
        and not features
    ):
        raise ArtifactValidationError(
            "generation comparison v1 architecture_features must be empty"
        )
    room_value = value.get("room") if isinstance(value.get("room"), Mapping) else value
    room = dict(room_value)
    unit = str(room.get("unit") or "meter").strip().lower()
    if unit not in {"m", "meter", "meters", "metre", "metres"}:
        raise ArtifactValidationError(
            "generation comparison v1 room dimensions must use meters"
        )
    boundary = room.get("boundary")
    if not (
        isinstance(boundary, Sequence)
        and not isinstance(boundary, (str, bytes))
        and len(boundary) == 4
    ):
        raise ArtifactValidationError(
            "generation comparison v1 requires a four-corner room boundary"
        )
    points: list[tuple[float, float]] = []
    for index, point in enumerate(boundary):
        if not (
            isinstance(point, Sequence)
            and not isinstance(point, (str, bytes))
            and len(point) >= 2
        ):
            raise ArtifactValidationError(
                f"comparison room boundary[{index}] must be an [x, y] point"
            )
        points.append(
            (
                _finite_number(point[0], f"comparison room boundary[{index}][0]"),
                _finite_number(point[1], f"comparison room boundary[{index}][1]"),
            )
        )
    xs = sorted({_clean_number(point[0]) for point in points})
    ys = sorted({_clean_number(point[1]) for point in points})
    if len(xs) != 2 or len(ys) != 2:
        raise ArtifactValidationError(
            "generation comparison v1 boundary must be an axis-aligned rectangle"
        )
    expected = {(x, y) for x in xs for y in ys}
    actual = {(_clean_number(x), _clean_number(y)) for x, y in points}
    if actual != expected:
        raise ArtifactValidationError(
            "generation comparison v1 boundary must contain all rectangle corners"
        )
    width = _clean_number(xs[-1] - xs[0])
    depth = _clean_number(ys[-1] - ys[0])
    height = _finite_number(room.get("height"), "comparison room height")
    if width <= 0.0 or depth <= 0.0 or height <= 0.0:
        raise ArtifactValidationError(
            "comparison room width, depth, and height must be positive"
        )
    return {
        "room_model": "single_room",
        "boundary_model": "axis_aligned_rectangle",
        "boundary": [
            [0.0, 0.0],
            [width, 0.0],
            [width, depth],
            [0.0, depth],
        ],
        "height": _clean_number(height),
        "unit": "meter",
    }


def architecture_sha256(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256(normalize_rectangular_architecture(value))


def architecture_from_generation_input(generation_input: Mapping[str, Any]) -> dict[str, Any]:
    contract = generation_input.get("generation_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    architecture = contract.get("architecture")
    if isinstance(architecture, Mapping):
        return normalize_rectangular_architecture(architecture)
    request = generation_input.get("scene_request")
    request = request if isinstance(request, Mapping) else {}
    room = request.get("room")
    if not isinstance(room, Mapping):
        raise ArtifactValidationError(
            "generation input lacks a public room for controlled comparison"
        )
    return normalize_rectangular_architecture(room)


def architecture_from_canonical_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_rectangular_architecture(
        {
            "room_model": "single_room",
            "boundary_model": "axis_aligned_rectangle",
            "room": {
                "boundary": scene.get("boundary"),
                "height": scene.get("scene_height"),
                "unit": "meter",
            },
        }
    )


def _canonical_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactValidationError(f"{path} contains a non-finite number")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactValidationError(f"{path} contains a non-string key")
            normalized[key] = _canonical_json_value(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ArtifactValidationError(
        f"{path} contains unsupported JSON value {type(value).__name__}"
    )


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ArtifactValidationError(f"{path} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{path} must be a finite number") from exc
    if not math.isfinite(number):
        raise ArtifactValidationError(f"{path} must be a finite number")
    return number


def _clean_number(value: float) -> float:
    cleaned = float(round(float(value), 9))
    return 0.0 if cleaned == 0.0 else cleaned


__all__ = [
    "architecture_from_canonical_scene",
    "architecture_from_generation_input",
    "architecture_sha256",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "normalize_rectangular_architecture",
]
