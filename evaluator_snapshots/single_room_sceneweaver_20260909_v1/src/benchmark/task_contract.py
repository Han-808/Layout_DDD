from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

from benchmark.architecture_policy import (
    ARCHITECTURE_CONTRACT_ID,
    DEFAULT_PHYSICAL_WALL_POLICY,
    build_architecture_contract,
)


ARCHITECTURE_ID = ARCHITECTURE_CONTRACT_ID
ROOM_DIMENSION_POLICY = "room_dimension_policy_v1"
ROOM_AXES = ("width", "depth", "height")
DEFAULT_ROOM_DIMENSIONS: dict[str, float] = {
    "width": 7.0,
    "depth": 5.0,
    "height": 3.0,
}
SINGLE_AXIS_RATIO: dict[str, float] = {
    "width": 1.0,
    "depth": 0.75,
    "height": 0.4,
}


def default_room() -> dict[str, Any]:
    """Return the benchmark fallback room used when no dimensions are claimed."""

    return _build_room(
        DEFAULT_ROOM_DIMENSIONS,
        {axis: "benchmark_fallback" for axis in ROOM_AXES},
        explicit_dimensions={},
    )


def resolve_room_contract(
    room: dict | None,
    *,
    prompt_dimensions: Mapping[str, Any] | None = None,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Resolve explicit room dimensions and deterministically fill missing axes.

    The room remains a bounded rectangular coordinate domain. Physical wall
    activation is resolved separately from this logical boundary.
    """

    structured = _structured_room_dimensions(room, tolerance=tolerance)
    prompted = _dimension_mapping(prompt_dimensions, "prompt_dimensions")
    explicit: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for source, dimensions in (
        ("structured_input", structured),
        ("natural_language", prompted),
    ):
        for axis, value in dimensions.items():
            if axis in explicit and abs(explicit[axis] - value) > tolerance:
                raise ValueError(
                    f"conflicting explicit room {axis}: {explicit[axis]:g} from "
                    f"{', '.join(sources[axis])} versus {value:g} from {source}"
                )
            explicit.setdefault(axis, value)
            sources.setdefault(axis, []).append(source)

    resolved = dict(explicit)
    provenance = {
        axis: _explicit_source_label(axis_sources)
        for axis, axis_sources in sources.items()
    }
    specified = set(explicit)
    if not specified:
        resolved.update(DEFAULT_ROOM_DIMENSIONS)
        provenance.update({axis: "benchmark_fallback" for axis in ROOM_AXES})
    elif len(specified) == 1:
        known_axis = next(iter(specified))
        scale = explicit[known_axis] / SINGLE_AXIS_RATIO[known_axis]
        for axis in ROOM_AXES:
            if axis not in resolved:
                resolved[axis] = scale * SINGLE_AXIS_RATIO[axis]
                provenance[axis] = f"derived_single_axis_ratio_from_{known_axis}"
    elif specified == {"width", "depth"}:
        resolved["height"] = 0.25 * (resolved["width"] + resolved["depth"])
        provenance["height"] = "derived_from_two_horizontal_dimensions"
    elif len(specified) == 2:
        missing_axis = next(axis for axis in ROOM_AXES if axis not in specified)
        resolved[missing_axis] = DEFAULT_ROOM_DIMENSIONS[missing_axis]
        provenance[missing_axis] = "benchmark_fallback_for_partial_combination"

    clean_dimensions = {axis: _clean_number(resolved[axis]) for axis in ROOM_AXES}
    return _build_room(
        clean_dimensions,
        provenance,
        explicit_dimensions={axis: _clean_number(value) for axis, value in explicit.items()},
    )


def architecture_contract_for_room(
    room: dict | None,
    *,
    physical_wall_policy: str = DEFAULT_PHYSICAL_WALL_POLICY,
    active_wall_ids: tuple[str, ...] | list[str] = (),
    policy_source: str = "canonical_default",
) -> dict[str, Any]:
    """Describe the resolved generator/evaluator architecture contract."""

    resolved = deepcopy(room) if _is_resolved_room(room) else resolve_room_contract(room)
    resolved_walls = (
        ("north_wall", "south_wall", "east_wall", "west_wall")
        if physical_wall_policy == "always_enclosed" and not active_wall_ids
        else active_wall_ids
    )
    return build_architecture_contract(
        resolved,
        physical_wall_policy=physical_wall_policy,
        requested_policy=physical_wall_policy,
        policy_source=policy_source,
        active_wall_ids=resolved_walls,
        activation_sources=(
            ("compatibility_policy",)
            if physical_wall_policy == "always_enclosed"
            else ()
        ),
        activation_claims=(
            (
                {
                    "source": "compatibility_policy",
                    "claim": "always_enclosed",
                    "active_wall_ids": list(resolved_walls),
                },
            )
            if physical_wall_policy == "always_enclosed"
            else ()
        ),
    )


def room_matches_contract(
    room: dict,
    expected_room: dict,
    *,
    tolerance: float = 1.0e-6,
) -> bool:
    if not isinstance(room, dict) or not isinstance(expected_room, dict):
        return False
    try:
        actual = _structured_room_dimensions(room, tolerance=tolerance)
        expected = _structured_room_dimensions(expected_room, tolerance=tolerance)
    except ValueError:
        return False
    return all(
        axis in actual
        and axis in expected
        and abs(actual[axis] - expected[axis]) <= tolerance
        for axis in ROOM_AXES
    )


def require_scene_matches_architecture(scene: dict, expected_room: dict) -> None:
    room = {
        "boundary": scene.get("boundary"),
        "height": scene.get("scene_height"),
    }
    if not room_matches_contract(room, expected_room):
        expected = _structured_room_dimensions(expected_room)
        raise ValueError(
            "generated canonical scene conflicts with the resolved benchmark room "
            f"{expected['width']:g}m x {expected['depth']:g}m x {expected['height']:g}m"
        )


def _structured_room_dimensions(
    room: dict | None,
    *,
    tolerance: float = 1.0e-6,
) -> dict[str, float]:
    if room is None:
        return {}
    if not isinstance(room, dict):
        raise ValueError("room must be a JSON object")
    unit = room.get("unit")
    if unit is not None and str(unit).strip().lower() not in {"m", "meter", "meters", "metre", "metres"}:
        raise ValueError("room dimensions must use meters")

    candidates: list[tuple[str, dict[str, float]]] = []
    if room.get("dimensions") is not None:
        candidates.append(("room.dimensions", _dimension_mapping(room["dimensions"], "room.dimensions")))
    flat = {axis: room.get(axis) for axis in ROOM_AXES if room.get(axis) is not None}
    if flat:
        candidates.append(("room", _dimension_mapping(flat, "room")))
    if room.get("size") is not None:
        size = room["size"]
        if not isinstance(size, (list, tuple)) or len(size) not in {2, 3}:
            raise ValueError("room.size must contain [width, depth] or [width, depth, height]")
        size_dimensions = {"width": size[0], "depth": size[1]}
        if len(size) == 3:
            size_dimensions["height"] = size[2]
        candidates.append(("room.size", _dimension_mapping(size_dimensions, "room.size")))
    if room.get("boundary") is not None:
        boundary_dimensions = _rectangular_boundary_dimensions(room["boundary"], tolerance=tolerance)
        if room.get("height") is not None:
            boundary_dimensions["height"] = _positive_number(room["height"], "room.height")
        candidates.append(("room.boundary", boundary_dimensions))
    elif room.get("height") is not None and "height" not in flat:
        candidates.append(("room.height", {"height": _positive_number(room["height"], "room.height")}))

    merged: dict[str, float] = {}
    origins: dict[str, str] = {}
    for origin, dimensions in candidates:
        for axis, value in dimensions.items():
            if axis in merged and abs(merged[axis] - value) > tolerance:
                raise ValueError(
                    f"conflicting structured room {axis}: {merged[axis]:g} from "
                    f"{origins[axis]} versus {value:g} from {origin}"
                )
            merged.setdefault(axis, value)
            origins.setdefault(axis, origin)
    return merged


def _rectangular_boundary_dimensions(boundary: Any, *, tolerance: float) -> dict[str, float]:
    if not isinstance(boundary, list) or len(boundary) != 4:
        raise ValueError("room.boundary must contain four corners")
    try:
        points = [(float(point[0]), float(point[1])) for point in boundary]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("room.boundary corners must be finite [x, y] numbers") from exc
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("room.boundary corners must be finite [x, y] numbers")
    xs = sorted({point[0] for point in points})
    ys = sorted({point[1] for point in points})
    if len(xs) != 2 or len(ys) != 2:
        raise ValueError("room.boundary must be an axis-aligned rectangle")
    expected = {(x, y) for x in xs for y in ys}
    if any(not any(abs(x - ex) <= tolerance and abs(y - ey) <= tolerance for ex, ey in expected) for x, y in points):
        raise ValueError("room.boundary must contain the four corners of an axis-aligned rectangle")
    width = xs[-1] - xs[0]
    depth = ys[-1] - ys[0]
    return {
        "width": _positive_number(width, "room.boundary width"),
        "depth": _positive_number(depth, "room.boundary depth"),
    }


def _dimension_mapping(value: Mapping[str, Any] | None, path: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    unknown = sorted(str(key) for key in value if key not in ROOM_AXES)
    if unknown:
        raise ValueError(f"{path} contains unsupported dimension axes: {unknown}")
    return {
        axis: _positive_number(value[axis], f"{path}.{axis}")
        for axis in ROOM_AXES
        if value.get(axis) is not None
    }


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{path} must be a positive finite number")
    return number


def _explicit_source_label(sources: list[str]) -> str:
    unique = list(dict.fromkeys(sources))
    return "explicit_" + "_and_".join(unique)


def _clean_number(value: float) -> float:
    return float(round(float(value), 9))


def _build_room(
    dimensions: Mapping[str, float],
    provenance: Mapping[str, str],
    *,
    explicit_dimensions: Mapping[str, float],
) -> dict[str, Any]:
    width = float(dimensions["width"])
    depth = float(dimensions["depth"])
    height = float(dimensions["height"])
    return {
        "boundary": [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]],
        "height": height,
        "unit": "meter",
        "dimensions": {axis: float(dimensions[axis]) for axis in ROOM_AXES},
        "explicit_dimensions": {
            axis: float(explicit_dimensions[axis])
            for axis in ROOM_AXES
            if axis in explicit_dimensions
        },
        "dimension_provenance": {axis: str(provenance[axis]) for axis in ROOM_AXES},
        "resolution_policy": ROOM_DIMENSION_POLICY,
        "topology": "rectangular_logical_boundary",
        "floor_z": 0.0,
    }


def _is_resolved_room(room: dict | None) -> bool:
    return bool(
        isinstance(room, dict)
        and room.get("resolution_policy") == ROOM_DIMENSION_POLICY
        and isinstance(room.get("dimensions"), dict)
        and isinstance(room.get("dimension_provenance"), dict)
    )
