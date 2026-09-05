"""Strict additive contracts for non-rectangular multi-room artifacts."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from benchmark.resources import runtime_resource_path


ROOM_PROGRAM_SCHEMA_VERSION = "non_rectangular_room_program_v1"
OBJECT_PLAN_SCHEMA_VERSION = "non_rectangular_multi_room_object_plan_v1"
OBJECT_PLAN_V2_SCHEMA_VERSION = "non_rectangular_multi_room_object_plan_v2"
MULTI_ROOM_SCENE_SCHEMA_VERSION = "non_rectangular_multi_room_scene_v1"
NON_RECTANGULAR_EVALUATION_MODE = "non_rectangular_multi_room"

ROOM_PROGRAM_SCHEMA_PATH = runtime_resource_path(
    "schemas/non_rectangular/room_program_v1.schema.json"
)
OBJECT_PLAN_SCHEMA_PATH = runtime_resource_path(
    "schemas/non_rectangular/object_plan_v1.schema.json"
)
OBJECT_PLAN_V2_SCHEMA_PATH = runtime_resource_path(
    "schemas/non_rectangular/object_plan_v2.schema.json"
)
MULTI_ROOM_SCENE_SCHEMA_PATH = runtime_resource_path(
    "schemas/non_rectangular/scene_v1.schema.json"
)


class NonRectangularContractError(ValueError):
    """Raised when an additive non-rectangular artifact is invalid."""


def validate_room_program(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one benchmark-owned layout-level room-purpose multiset."""

    _require_mapping(value, "room program")
    _validate_schema(
        value,
        path=ROOM_PROGRAM_SCHEMA_PATH,
        label="room-program",
    )
    programs = list(value["programs"])
    program_ids = [str(item["program_id"]) for item in programs]
    _require_exact_order(
        value["program_order"],
        program_ids,
        path="program_order",
    )
    target = value["target_total_instances"]
    minimum = int(target["min"])
    maximum = int(target["max"])
    if minimum > maximum:
        raise NonRectangularContractError(
            "target_total_instances.min must be <= max"
        )
    room_types = [str(item["room_type"]) for item in programs]
    if any(not room_type.strip() for room_type in room_types):
        raise NonRectangularContractError(
            "programs[].room_type must contain non-whitespace text"
        )
    return {
        "schema_version": ROOM_PROGRAM_SCHEMA_VERSION,
        "layout_id": str(value["layout_id"]),
        "valid": True,
        "program_count": len(programs),
        "program_ids": program_ids,
        "room_types": room_types,
        "target_total_instances": {"min": minimum, "max": maximum},
    }


def validate_multi_room_object_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate either preserved v1 or simplified v2 Stage-A artifacts."""

    _require_mapping(value, "multi-room object plan")
    schema_version = value.get("schema_version")
    if schema_version == OBJECT_PLAN_SCHEMA_VERSION:
        schema_path = OBJECT_PLAN_SCHEMA_PATH
    elif schema_version == OBJECT_PLAN_V2_SCHEMA_VERSION:
        schema_path = OBJECT_PLAN_V2_SCHEMA_PATH
    else:
        raise NonRectangularContractError(
            f"unsupported object-plan schema_version: {schema_version!r}"
        )
    _validate_schema(
        value,
        path=schema_path,
        label="object-plan",
    )
    rooms = list(value["rooms"])
    room_ids = [str(room["room_id"]) for room in rooms]
    _require_exact_order(value["room_order"], room_ids, path="room_order")

    room_instance_counts: dict[str, int] = {}
    room_slot_ids: dict[str, list[str]] = {}
    room_slot_counts: dict[str, dict[str, int]] = {}
    mapping_complete = True
    for room_index, room in enumerate(rooms):
        room_id = str(room["room_id"])
        room_path = f"rooms[{room_index}]"
        if not _present_string(room.get("program_id")) or not _present_string(
            room.get("room_type")
        ):
            mapping_complete = False

        objects = list(room["objects"])
        slot_ids = [str(item["id"]) for item in objects]
        _require_unique(slot_ids, path=f"{room_path}.objects[].id")
        instance_count = 0
        slot_counts: dict[str, int] = {}
        if schema_version == OBJECT_PLAN_SCHEMA_VERSION:
            zones = list(room["zones"])
            zone_ids = [str(item["id"]) for item in zones]
            _require_unique(zone_ids, path=f"{room_path}.zones[].id")
            zone_id_set = set(zone_ids)
        else:
            zone_id_set = set()
        for object_index, item in enumerate(objects):
            object_path = f"{room_path}.objects[{object_index}]"
            count = int(item["count"])
            if schema_version == OBJECT_PLAN_SCHEMA_VERSION:
                requested_count = int(item["metadata"]["requested_count"])
                if requested_count != count:
                    raise NonRectangularContractError(
                        f"{object_path}.metadata.requested_count must equal count"
                    )
                zone_id = str(item["metadata"]["zone"])
                if zone_id not in zone_id_set:
                    raise NonRectangularContractError(
                        f"{object_path}.metadata.zone references unknown zone "
                        f"{zone_id!r}"
                    )
            _finite_vector(
                item["estimated_size"],
                path=f"{object_path}.estimated_size",
                positive=True,
            )
            instance_count += count
            slot_counts[str(item["id"])] = count

        if schema_version == OBJECT_PLAN_SCHEMA_VERSION:
            slot_id_set = set(slot_ids)
            for relation_index, relation in enumerate(room["relations"]):
                relation_path = f"{room_path}.relations[{relation_index}]"
                for field in ("subject_id", "object_id"):
                    reference = str(relation[field])
                    if reference not in slot_id_set:
                        raise NonRectangularContractError(
                            f"{relation_path}.{field} references unknown object "
                            f"slot {reference!r}"
                        )
        room_instance_counts[room_id] = instance_count
        room_slot_ids[room_id] = slot_ids
        room_slot_counts[room_id] = slot_counts

    return {
        "schema_version": str(schema_version),
        "plan_contract_version": (
            "v1"
            if schema_version == OBJECT_PLAN_SCHEMA_VERSION
            else "v2"
        ),
        "layout_id": str(value["layout_id"]),
        "valid": True,
        "room_count": len(rooms),
        "room_ids": room_ids,
        "room_instance_counts": room_instance_counts,
        "room_slot_ids": room_slot_ids,
        "room_slot_counts": room_slot_counts,
        "planned_instance_count": sum(room_instance_counts.values()),
        "mapping_complete": mapping_complete,
    }


def validate_multi_room_scene(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one Stage-C scene with canonical objects nested under rooms."""

    _require_mapping(value, "multi-room scene")
    _validate_schema(
        value,
        path=MULTI_ROOM_SCENE_SCHEMA_PATH,
        label="scene",
    )
    rooms = list(value["rooms"])
    room_ids = [str(room["room_id"]) for room in rooms]
    _require_exact_order(value["room_order"], room_ids, path="room_order")

    global_object_ids: set[str] = set()
    room_object_counts: dict[str, int] = {}
    room_slot_counts: dict[str, dict[str, int]] = {}
    mapping_complete = True
    for room_index, room in enumerate(rooms):
        room_id = str(room["room_id"])
        room_path = f"rooms[{room_index}]"
        if not _present_string(room.get("program_id")) or not _present_string(
            room.get("room_type")
        ):
            mapping_complete = False
        slot_counts: dict[str, int] = {}
        for object_index, item in enumerate(room["objects"]):
            object_path = f"{room_path}.objects[{object_index}]"
            object_id = str(item["id"])
            if object_id in global_object_ids:
                raise NonRectangularContractError(
                    f"duplicate scene-global object ID: {object_id!r}"
                )
            global_object_ids.add(object_id)
            slot_id = str(item["slot_id"])
            slot_counts[slot_id] = slot_counts.get(slot_id, 0) + 1
            _finite_vector(item["size"], path=f"{object_path}.size", positive=True)
            _finite_vector(item["center"], path=f"{object_path}.center")
            _finite_vector(item["rotation"], path=f"{object_path}.rotation")
        room_object_counts[room_id] = len(room["objects"])
        room_slot_counts[room_id] = dict(sorted(slot_counts.items()))

    return {
        "schema_version": MULTI_ROOM_SCENE_SCHEMA_VERSION,
        "layout_id": str(value["layout_id"]),
        "valid": True,
        "room_count": len(rooms),
        "room_ids": room_ids,
        "room_object_counts": room_object_counts,
        "room_slot_counts": room_slot_counts,
        "generated_object_count": len(global_object_ids),
        "object_ids": sorted(global_object_ids),
        "mapping_complete": mapping_complete,
        "coordinate_frame_shared": True,
    }


def _validate_schema(
    value: Mapping[str, Any],
    *,
    path: Any,
    label: str,
) -> None:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NonRectangularContractError(
            f"cannot load packaged {label} schema"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    error_path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise NonRectangularContractError(
        f"{label} schema validation failed at {error_path}: {error.message}"
    )


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise NonRectangularContractError(f"{label} must be a JSON object")


def _require_exact_order(actual: Any, expected: list[str], *, path: str) -> None:
    values = [str(item) for item in actual]
    _require_unique(values, path=path)
    if values != expected:
        raise NonRectangularContractError(
            f"{path} must exactly match rooms/programs serialization order"
        )


def _require_unique(values: list[str], *, path: str) -> None:
    if len(values) != len(set(values)):
        raise NonRectangularContractError(f"{path} must be unique")


def _finite_vector(
    value: Any,
    *,
    path: str,
    positive: bool = False,
) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise NonRectangularContractError(f"{path} must be a 3-vector")
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise NonRectangularContractError(f"{path}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise NonRectangularContractError(f"{path}[{index}] must be finite")
        if positive and number <= 0.0:
            raise NonRectangularContractError(f"{path}[{index}] must be positive")


def _present_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "MULTI_ROOM_SCENE_SCHEMA_PATH",
    "MULTI_ROOM_SCENE_SCHEMA_VERSION",
    "NON_RECTANGULAR_EVALUATION_MODE",
    "NonRectangularContractError",
    "OBJECT_PLAN_SCHEMA_PATH",
    "OBJECT_PLAN_SCHEMA_VERSION",
    "OBJECT_PLAN_V2_SCHEMA_PATH",
    "OBJECT_PLAN_V2_SCHEMA_VERSION",
    "ROOM_PROGRAM_SCHEMA_PATH",
    "ROOM_PROGRAM_SCHEMA_VERSION",
    "validate_multi_room_object_plan",
    "validate_multi_room_scene",
    "validate_room_program",
]
