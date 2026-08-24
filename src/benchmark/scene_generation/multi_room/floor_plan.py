"""Strict generic floor-plan loading for the additive multi-room mode."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from benchmark.resources import runtime_resource_path


FLOOR_PLAN_SCHEMA_VERSION = "multi_room_floor_plan_v1"
GENERATION_MODE = "multi_room_with_architecture_v1"
ROOM_PROMPT_VERSION = "room_generation_prompt_v1"
FLOOR_PLAN_SCHEMA_PATH = runtime_resource_path(
    "schemas/multi_room/floor_plan_v1.schema.json"
)
_ROOM_ID = re.compile(r"^room_[0-9]+$")
_SHARED_WALL_ID = re.compile(r"^shared_wall_[0-9]+$")
_WALL_IDS = frozenset(
    {"north_wall", "south_wall", "east_wall", "west_wall"}
)
_OPPOSITE_WALL = {
    "north_wall": "south_wall",
    "south_wall": "north_wall",
    "east_wall": "west_wall",
    "west_wall": "east_wall",
}
_TIER_BY_NAME = {
    "compact_7_10": (7, 10),
    "standard_11_14": (11, 14),
    "large_15_19": (15, 19),
}


class FloorPlanValidationError(ValueError):
    """Raised when a multi-room floor plan fails a closed contract gate."""


@dataclass(frozen=True, slots=True)
class LoadedFloorPlan:
    """One validated caller-supplied floor plan and its stable identity."""

    path: Path
    value: Mapping[str, Any]
    canonical_sha256: str
    source_sha256: str
    validation_report: Mapping[str, Any]
    source_bytes: bytes

    @property
    def layout_id(self) -> str:
        return str(self.value["layout_id"])

    @property
    def generation_order(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.value["generation_order"])

    @property
    def room_count(self) -> int:
        return int(self.value["room_count"])

    def room(self, room_id: str) -> dict[str, Any]:
        for room in self.value["rooms"]:
            if room["room_id"] == room_id:
                return deepcopy(room)
        raise KeyError(room_id)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FLOOR_PLAN_SCHEMA_VERSION,
            "generation_mode": GENERATION_MODE,
            "layout_id": self.layout_id,
            "room_count": self.room_count,
            "generation_order": list(self.generation_order),
            "source_sha256": self.source_sha256,
            "canonical_sha256": self.canonical_sha256,
            "valid": True,
        }


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise FloorPlanValidationError(f"duplicate floor-plan key: {key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise FloorPlanValidationError(f"non-finite floor-plan number: {value}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _audit_numeric_source_identity(
    source_value: Any,
    runtime_value: Any,
    *,
    path: str = "<root>",
) -> None:
    """Reject JSON numeric literals lost by the runtime parser."""

    if isinstance(source_value, Decimal):
        if isinstance(runtime_value, bool) or not isinstance(
            runtime_value, (int, float)
        ):
            raise FloorPlanValidationError(f"{path} changed numeric type")
        if source_value != Decimal(str(runtime_value)):
            raise FloorPlanValidationError(
                f"{path} numeric literal loses precision in the runtime format"
            )
        return
    if isinstance(source_value, dict):
        if not isinstance(runtime_value, dict) or source_value.keys() != runtime_value.keys():
            raise FloorPlanValidationError(f"{path} source/runtime structure differs")
        for key in source_value:
            _audit_numeric_source_identity(
                source_value[key], runtime_value[key], path=f"{path}.{key}"
            )
        return
    if isinstance(source_value, list):
        if not isinstance(runtime_value, list) or len(source_value) != len(runtime_value):
            raise FloorPlanValidationError(f"{path} source/runtime structure differs")
        for index, child in enumerate(source_value):
            _audit_numeric_source_identity(
                child, runtime_value[index], path=f"{path}[{index}]"
            )


def _decimal(value: Any, path: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FloorPlanValidationError(f"{path} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise FloorPlanValidationError(f"{path} must be finite")
    number = Decimal(str(value))
    as_float = float(number)
    if not math.isfinite(as_float) or Decimal(str(as_float)) != number:
        raise FloorPlanValidationError(
            f"{path} cannot round-trip through the runtime numeric format"
        )
    return number


def _same(left: Any, right: Any, path: str) -> None:
    if _decimal(left, path) != _decimal(right, path):
        raise FloorPlanValidationError(
            f"{path} differs: expected={right!r}, actual={left!r}"
        )


def _rect(boundary: Any, path: str) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    if not isinstance(boundary, list) or len(boundary) != 4:
        raise FloorPlanValidationError(f"{path} must contain four points")
    points: list[tuple[Decimal, Decimal]] = []
    for index, point in enumerate(boundary):
        if not isinstance(point, list) or len(point) != 2:
            raise FloorPlanValidationError(f"{path}[{index}] must be [x, y]")
        x = _decimal(point[0], f"{path}[{index}][0]")
        y = _decimal(point[1], f"{path}[{index}][1]")
        if x < 0 or y < 0:
            raise FloorPlanValidationError(f"{path} coordinates must be non-negative")
        points.append((x, y))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0 or y1 <= y0:
        raise FloorPlanValidationError(f"{path} must have positive dimensions")
    expected = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if points != expected:
        raise FloorPlanValidationError(
            f"{path} must use canonical counter-clockwise rectangle ordering"
        )
    return x0, y0, x1, y1


def _edge(
    rect: tuple[Decimal, Decimal, Decimal, Decimal], wall_id: str
) -> tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]]:
    x0, y0, x1, y1 = rect
    if wall_id == "south_wall":
        return (x0, y0), (x1, y0)
    if wall_id == "north_wall":
        return (x0, y1), (x1, y1)
    if wall_id == "west_wall":
        return (x0, y0), (x0, y1)
    if wall_id == "east_wall":
        return (x1, y0), (x1, y1)
    raise FloorPlanValidationError(f"unknown wall ID: {wall_id!r}")


def _canonical_segment(
    value: Any, path: str
) -> tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]]:
    if not isinstance(value, list) or len(value) != 2:
        raise FloorPlanValidationError(f"{path} must contain two endpoints")
    points = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise FloorPlanValidationError(f"{path}[{index}] must be [x, y]")
        x = _decimal(point[0], f"{path}[{index}][0]")
        y = _decimal(point[1], f"{path}[{index}][1]")
        if x < 0 or y < 0:
            raise FloorPlanValidationError(f"{path} coordinates must be non-negative")
        points.append((x, y))
    first, second = sorted(points)
    if first == second or (first[0] != second[0] and first[1] != second[1]):
        raise FloorPlanValidationError(f"{path} must be a positive axis-aligned segment")
    return first, second


def _shared_edge(
    left_rect: tuple[Decimal, Decimal, Decimal, Decimal],
    left_wall: str,
    right_rect: tuple[Decimal, Decimal, Decimal, Decimal],
    right_wall: str,
) -> tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]] | None:
    if _OPPOSITE_WALL[left_wall] != right_wall:
        return None
    left = _edge(left_rect, left_wall)
    right = _edge(right_rect, right_wall)
    if left_wall in {"north_wall", "south_wall"}:
        if left[0][1] != right[0][1]:
            return None
        start = max(left[0][0], right[0][0])
        end = min(left[1][0], right[1][0])
        if end <= start:
            return None
        return (start, left[0][1]), (end, left[0][1])
    if left[0][0] != right[0][0]:
        return None
    start = max(left[0][1], right[0][1])
    end = min(left[1][1], right[1][1])
    if end <= start:
        return None
    return (left[0][0], start), (left[0][0], end)


def _geometric_adjacencies(
    room_ids: Iterable[str],
    rects: Mapping[str, tuple[Decimal, Decimal, Decimal, Decimal]],
) -> dict[frozenset[str], tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]]]:
    ids = tuple(room_ids)
    result = {}
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index + 1 :]:
            found = None
            for left_wall in _WALL_IDS:
                expected = _shared_edge(
                    rects[left_id],
                    left_wall,
                    rects[right_id],
                    _OPPOSITE_WALL[left_wall],
                )
                if expected is not None:
                    found = expected
                    break
            if found is not None:
                result[frozenset({left_id, right_id})] = found
    return result


def _tier_for_area(area: Decimal) -> tuple[str, tuple[int, int]]:
    if area <= Decimal("15"):
        return "compact_7_10", _TIER_BY_NAME["compact_7_10"]
    if area < Decimal("28"):
        return "standard_11_14", _TIER_BY_NAME["standard_11_14"]
    return "large_15_19", _TIER_BY_NAME["large_15_19"]


def _validate_schema(value: Mapping[str, Any]) -> None:
    try:
        schema = json.loads(FLOOR_PLAN_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FloorPlanValidationError(
            f"cannot load packaged floor-plan schema: {type(exc).__name__}"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise FloorPlanValidationError(
            f"floor-plan schema validation failed at {path}: {error.message}"
        )


def validate_floor_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, geometry, ordering, counts, walls, and connectivity."""

    if not isinstance(value, Mapping):
        raise FloorPlanValidationError("floor plan must be a JSON object")
    _validate_schema(value)
    if value.get("schema_version") != FLOOR_PLAN_SCHEMA_VERSION:
        raise FloorPlanValidationError("unsupported floor-plan schema version")
    if value.get("generation_mode") != GENERATION_MODE:
        raise FloorPlanValidationError("unsupported generation mode")
    if value.get("room_prompt_version") != ROOM_PROMPT_VERSION:
        raise FloorPlanValidationError("unsupported room prompt version")

    rooms = value["rooms"]
    order = value["generation_order"]
    room_count = value["room_count"]
    if room_count != len(rooms) or room_count != len(order):
        raise FloorPlanValidationError(
            "room_count must exactly match rooms and generation_order"
        )
    room_by_id: dict[str, Mapping[str, Any]] = {}
    room_types: set[str] = set()
    rects: dict[str, tuple[Decimal, Decimal, Decimal, Decimal]] = {}
    local_rects: dict[str, tuple[Decimal, Decimal, Decimal, Decimal]] = {}
    z_offsets: dict[str, Decimal] = {}
    heights: dict[str, Decimal] = {}
    active_walls: dict[str, frozenset[str]] = {}
    for index, raw_room in enumerate(rooms):
        room = raw_room
        room_id = room["room_id"]
        if not _ROOM_ID.fullmatch(room_id):
            raise FloorPlanValidationError(f"invalid room ID: {room_id!r}")
        if room_id in room_by_id:
            raise FloorPlanValidationError(f"duplicate room ID: {room_id}")
        room_by_id[room_id] = room
        room_type = room["room_type"].strip()
        if not room_type or room_type in room_types:
            raise FloorPlanValidationError(
                f"room_type must be non-empty and unique: {room_type!r}"
            )
        room_types.add(room_type)
        if not room["theme"].strip() or not room["instruction"].strip():
            raise FloorPlanValidationError(
                f"room {room_id} theme/instruction must not be blank"
            )
        global_rect = _rect(room["room"]["boundary"], f"rooms[{index}].room.boundary")
        local_rect = _rect(
            room["runner_projection"]["local_room"]["boundary"],
            f"rooms[{index}].runner_projection.local_room.boundary",
        )
        if local_rect[0] != 0 or local_rect[1] != 0:
            raise FloorPlanValidationError(
                f"room {room_id} local boundary must start at [0, 0]"
            )
        width = global_rect[2] - global_rect[0]
        depth = global_rect[3] - global_rect[1]
        local_width = local_rect[2] - local_rect[0]
        local_depth = local_rect[3] - local_rect[1]
        dimensions = room["room_dimensions_m"]
        for actual, expected, label in (
            (dimensions[0], width, "width"),
            (dimensions[1], depth, "depth"),
            (dimensions[2], room["room"]["height"], "height"),
            (local_width, width, "local width"),
            (local_depth, depth, "local depth"),
            (
                room["runner_projection"]["local_room"]["height"],
                room["room"]["height"],
                "local height",
            ),
        ):
            _same(actual, expected, f"room {room_id} {label}")
        offset = room["runner_projection"]["local_to_global_offset_m"]
        _same(offset[0], global_rect[0], f"room {room_id} offset x")
        _same(offset[1], global_rect[1], f"room {room_id} offset y")
        if _decimal(offset[2], f"room {room_id} offset z") < 0:
            raise FloorPlanValidationError(f"room {room_id} offset z must be non-negative")
        tier, expected_range = _tier_for_area(width * depth)
        if room["object_count_tier"] != tier:
            raise FloorPlanValidationError(
                f"room {room_id} object_count_tier must be {tier!r}"
            )
        target = room["target_instances"]
        if (target["min"], target["max"]) != expected_range:
            raise FloorPlanValidationError(
                f"room {room_id} target range must be {expected_range}"
            )
        attachment = room["wall_attachment_requirement"]
        if not (
            0
            <= attachment["minimum_count"]
            <= attachment["maximum_count"]
            <= target["max"]
        ):
            raise FloorPlanValidationError(
                f"room {room_id} wall attachment range is invalid"
            )
        walls = tuple(room["architecture"]["active_wall_ids"])
        if len(walls) != len(set(walls)) or not set(walls) <= _WALL_IDS:
            raise FloorPlanValidationError(f"room {room_id} has invalid active walls")
        _same(
            room["architecture"]["wall_thickness_m"],
            value["wall_thickness_m"],
            f"room {room_id} wall thickness",
        )
        rects[room_id] = global_rect
        local_rects[room_id] = local_rect
        z_offsets[room_id] = _decimal(offset[2], f"room {room_id} offset z")
        heights[room_id] = _decimal(room["room"]["height"], f"room {room_id} height")
        active_walls[room_id] = frozenset(walls)

    if len(order) != len(set(order)) or set(order) != set(room_by_id):
        raise FloorPlanValidationError(
            "generation_order must contain every unique declared room exactly once"
        )
    indexed = sorted(room_by_id, key=lambda room_id: room_by_id[room_id]["generation_index"])
    if [room_by_id[room_id]["generation_index"] for room_id in indexed] != list(
        range(room_count)
    ):
        raise FloorPlanValidationError(
            "generation_index values must be exactly zero through room_count-1"
        )
    if list(order) != indexed:
        raise FloorPlanValidationError(
            "generation_order must exactly follow generation_index"
        )

    room_ids = tuple(order)
    for left_index, left_id in enumerate(room_ids):
        left = rects[left_id]
        for right_id in room_ids[left_index + 1 :]:
            right = rects[right_id]
            overlap_x = min(left[2], right[2]) - max(left[0], right[0])
            overlap_y = min(left[3], right[3]) - max(left[1], right[1])
            if overlap_x > 0 and overlap_y > 0:
                raise FloorPlanValidationError(
                    f"rooms {left_id} and {right_id} overlap with positive area"
                )

    declared_pairs: dict[
        frozenset[str], tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]]
    ] = {}
    seen_shared_ids: set[str] = set()
    graph: dict[str, set[str]] = {room_id: set() for room_id in room_ids}
    for index, shared in enumerate(value["shared_walls"]):
        shared_id = shared["shared_wall_id"]
        if not _SHARED_WALL_ID.fullmatch(shared_id) or shared_id in seen_shared_ids:
            raise FloorPlanValidationError(
                f"shared wall ID must be unique and canonical: {shared_id!r}"
            )
        seen_shared_ids.add(shared_id)
        endpoints = shared["rooms"]
        left_endpoint, right_endpoint = endpoints
        left_id = left_endpoint["room_id"]
        right_id = right_endpoint["room_id"]
        if left_id == right_id or left_id not in room_by_id or right_id not in room_by_id:
            raise FloorPlanValidationError(
                f"shared wall {shared_id} must reference two declared rooms"
            )
        left_wall = left_endpoint["wall_id"]
        right_wall = right_endpoint["wall_id"]
        if left_wall not in active_walls[left_id] or right_wall not in active_walls[right_id]:
            raise FloorPlanValidationError(
                f"shared wall {shared_id} references an inactive local wall"
            )
        expected = _shared_edge(
            rects[left_id], left_wall, rects[right_id], right_wall
        )
        actual = _canonical_segment(
            shared["segment_global_m"], f"shared_walls[{index}].segment_global_m"
        )
        if expected is None or actual != expected:
            raise FloorPlanValidationError(
                f"shared wall {shared_id} is not the exact opposing room-edge overlap"
            )
        if z_offsets[left_id] != z_offsets[right_id] or heights[left_id] != heights[right_id]:
            raise FloorPlanValidationError(
                f"shared wall {shared_id} cannot be full-height for mismatched rooms"
            )
        pair = frozenset({left_id, right_id})
        if pair in declared_pairs:
            raise FloorPlanValidationError(
                f"rooms {left_id} and {right_id} declare duplicate shared walls"
            )
        declared_pairs[pair] = actual
        graph[left_id].add(right_id)
        graph[right_id].add(left_id)

    geometric = _geometric_adjacencies(room_ids, rects)
    if declared_pairs != geometric:
        missing = sorted(
            sorted(pair) for pair in geometric.keys() - declared_pairs.keys()
        )
        extra = sorted(
            sorted(pair) for pair in declared_pairs.keys() - geometric.keys()
        )
        raise FloorPlanValidationError(
            f"shared_walls must exactly cover room edge adjacencies: missing={missing} extra={extra}"
        )
    visited: set[str] = set()
    pending = [room_ids[0]]
    while pending:
        room_id = pending.pop()
        if room_id in visited:
            continue
        visited.add(room_id)
        pending.extend(sorted(graph[room_id] - visited))
    if visited != set(room_ids):
        raise FloorPlanValidationError("room adjacency graph must be connected")

    envelope = _rect(value["global_envelope"]["boundary"], "global_envelope.boundary")
    expected_envelope = (
        min(rect[0] for rect in rects.values()),
        min(rect[1] for rect in rects.values()),
        max(rect[2] for rect in rects.values()),
        max(rect[3] for rect in rects.values()),
    )
    if envelope != expected_envelope or envelope[0] != 0 or envelope[1] != 0:
        raise FloorPlanValidationError(
            "global envelope must be the exact shared-frame room envelope with min [0, 0]"
        )
    dimensions = value["global_envelope"]["dimensions_m"]
    _same(dimensions[0], envelope[2] - envelope[0], "global envelope width")
    _same(dimensions[1], envelope[3] - envelope[1], "global envelope depth")
    if min(z_offsets.values()) != 0:
        raise FloorPlanValidationError("shared global floor-plan z origin must be zero")
    expected_height = max(z_offsets[room_id] + heights[room_id] for room_id in room_ids)
    _same(dimensions[2], expected_height, "global envelope height")

    canonical_sha = _sha256_bytes(_canonical_bytes(value))
    return {
        "schema_version": "multi_room_floor_plan_validation_v1",
        "valid": True,
        "generation_mode": GENERATION_MODE,
        "layout_id": value["layout_id"],
        "room_count": room_count,
        "generation_order": list(order),
        "shared_wall_count": len(value["shared_walls"]),
        "canonical_floor_plan_sha256": canonical_sha,
        "semantic_checks": {
            "ordered_room_identity": "passed",
            "rectangular_nonoverlap": "passed",
            "local_global_projection": "passed",
            "count_tiers": "passed",
            "wall_contract": "passed",
            "connected_adjacency": "passed",
            "exact_global_envelope": "passed",
            "float_safe_geometry": "passed",
        },
    }


def load_floor_plan(path: str | Path) -> LoadedFloorPlan:
    """Strictly load a regular floor-plan artifact and validate it completely."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FloorPlanValidationError(
            f"floor plan must be a regular non-symlink file: {source}"
        )
    data = source.read_bytes()
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
        source_numeric_value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_float=Decimal,
            parse_int=Decimal,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FloorPlanValidationError(
            f"invalid floor-plan JSON: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise FloorPlanValidationError("floor plan root must be a JSON object")
    _audit_numeric_source_identity(source_numeric_value, value)
    report = validate_floor_plan(value)
    return LoadedFloorPlan(
        path=source,
        value=deepcopy(value),
        canonical_sha256=report["canonical_floor_plan_sha256"],
        source_sha256=_sha256_bytes(data),
        validation_report=report,
        source_bytes=bytes(data),
    )


def compile_room_brief(plan: LoadedFloorPlan, room_id: str) -> dict[str, Any]:
    """Compile one room-only brief without importing any peer-room content."""

    room = plan.room(room_id)
    return {
        "schema_version": "multi_room_room_brief_v1",
        "generation_mode": GENERATION_MODE,
        "layout_id": plan.layout_id,
        "floor_plan_sha256": plan.canonical_sha256,
        "room_id": room["room_id"],
        "generation_index": room["generation_index"],
        "room_type": room["room_type"],
        "theme": room["theme"],
        "instruction": room["instruction"],
        "object_count_tier": room["object_count_tier"],
        "target_instances": deepcopy(room["target_instances"]),
        "room_dimensions_m": deepcopy(room["room_dimensions_m"]),
        "local_room": deepcopy(room["runner_projection"]["local_room"]),
        "local_to_global_offset_m": deepcopy(
            room["runner_projection"]["local_to_global_offset_m"]
        ),
        "architecture": deepcopy(room["architecture"]),
        "wall_attachment_requirement": deepcopy(
            room["wall_attachment_requirement"]
        ),
    }
