"""Validated accessors for the public generator-visible projection."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import math
from typing import Any

from benchmark.scene_io.validate import ArtifactValidationError


def public_generator_input(method_input: Mapping[str, Any]) -> dict[str, Any]:
    value = method_input.get("generator_input")
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("method_input.generator_input must be an object")
    return deepcopy(dict(value))


def public_room(method_input: Mapping[str, Any]) -> dict[str, Any]:
    visible = public_generator_input(method_input)
    environment = visible.get("benchmark_environment")
    environment = environment if isinstance(environment, Mapping) else {}
    architecture = environment.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ArtifactValidationError(
            "public generator input lacks benchmark_environment.architecture"
        )
    room = architecture.get("room")
    if not isinstance(room, Mapping):
        raise ArtifactValidationError(
            "public generator architecture lacks a room contract"
        )
    return deepcopy(dict(room))


def public_room_dimensions(
    method_input: Mapping[str, Any],
) -> tuple[float, float, float]:
    room = public_room(method_input)
    dimensions = room.get("dimensions")
    dimensions = dimensions if isinstance(dimensions, Mapping) else {}
    boundary = room.get("boundary")
    try:
        if isinstance(boundary, list) and boundary:
            xs = [float(point[0]) for point in boundary]
            ys = [float(point[1]) for point in boundary]
            width = max(xs) - min(xs)
            depth = max(ys) - min(ys)
        else:
            width = float(dimensions["width"])
            depth = float(dimensions["depth"])
        height_value = room.get("height")
        if height_value is None:
            height_value = dimensions["height"]
        height = float(height_value)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "public room requires finite width, depth, and height"
        ) from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in (width, depth, height)):
        raise ArtifactValidationError(
            "public room width, depth, and height must be positive and finite"
        )
    return width, depth, height


def public_instruction(method_input: Mapping[str, Any]) -> str:
    visible = public_generator_input(method_input)
    value = visible.get("natural_language")
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(
            "public generator input requires non-empty natural_language"
        )
    return value


def public_scene_type(method_input: Mapping[str, Any]) -> str:
    request = method_input.get("public_request")
    request = request if isinstance(request, Mapping) else {}
    value = request.get("scene_type")
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError("method_input.public_request.scene_type is required")
    return value


def public_request_id(method_input: Mapping[str, Any]) -> str:
    value = method_input.get("request_id")
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError("method_input.request_id is required")
    return value


def public_asset_selection(method_input: Mapping[str, Any]) -> dict[str, Any] | None:
    visible = public_generator_input(method_input)
    assistance = visible.get("assistance")
    assistance = assistance if isinstance(assistance, Mapping) else {}
    value = assistance.get("asset_selection")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(
            "public generator assistance.asset_selection must be an object"
        )
    return deepcopy(dict(value))


__all__ = [
    "public_asset_selection",
    "public_generator_input",
    "public_instruction",
    "public_request_id",
    "public_room",
    "public_room_dimensions",
    "public_scene_type",
]
