"""Deterministic structural validation for raw abstract layouts."""

from __future__ import annotations

import math
from typing import Any


TOP_LEVEL_FIELDS = frozenset({"rooms", "objects"})
ROOM_FIELDS = frozenset({"id", "category", "origin_m", "size_m"})
OBJECT_FIELDS = frozenset(
    {"id", "room_id", "category", "appearance", "position_m", "size_m", "yaw_deg"}
)


def _error(path: str, code: str, message: str) -> dict[str, str]:
    return {"path": path, "code": code, "message": message}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_exact_fields(
    value: dict[str, Any], expected: frozenset[str], path: str
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    for name in missing:
        errors.append(_error(path, "missing_field", f"missing required field {name!r}"))
    for name in extra:
        errors.append(_error(path, "extra_field", f"unexpected field {name!r}"))
    return errors


def _check_vector(
    value: Any, path: str, *, minimum: float | None, exclusive: bool = False
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return [_error(path, "type", "must be an array")]
    if len(value) != 3:
        return [_error(path, "length", "must contain exactly 3 numbers")]

    errors: list[dict[str, str]] = []
    for index, number in enumerate(value):
        item_path = f"{path}/{index}"
        if not _is_number(number):
            errors.append(_error(item_path, "type", "must be a JSON number"))
        elif not math.isfinite(number):
            errors.append(_error(item_path, "finite", "must be finite"))
        elif minimum is not None and exclusive and number <= minimum:
            errors.append(_error(item_path, "exclusive_minimum", "must be > 0"))
        elif minimum is not None and not exclusive and number < minimum:
            errors.append(_error(item_path, "minimum", f"must be >= {minimum:g}"))
    return errors


def validate_layout(value: Any) -> list[dict[str, str]]:
    """Return structural errors only; never judge or alter layout semantics."""
    if not isinstance(value, dict):
        return [_error("/", "type", "top-level JSON value must be an object")]

    errors = _check_exact_fields(value, TOP_LEVEL_FIELDS, "/")
    rooms = value.get("rooms")
    objects = value.get("objects")
    if not isinstance(rooms, list):
        if "rooms" in value:
            errors.append(_error("/rooms", "type", "must be an array"))
        rooms = []
    elif not rooms:
        errors.append(_error("/rooms", "min_items", "must contain at least one room"))
    if not isinstance(objects, list):
        if "objects" in value:
            errors.append(_error("/objects", "type", "must be an array"))
        return errors

    room_ids: set[str] = set()
    for index, item in enumerate(rooms):
        path = f"/rooms/{index}"
        if not isinstance(item, dict):
            errors.append(_error(path, "type", "must be an object"))
            continue
        errors.extend(_check_exact_fields(item, ROOM_FIELDS, path))
        for field in ("id", "category"):
            if field not in item:
                continue
            field_value = item[field]
            field_path = f"{path}/{field}"
            if not isinstance(field_value, str):
                errors.append(_error(field_path, "type", "must be a string"))
            elif not field_value.strip():
                errors.append(_error(field_path, "min_length", "must be non-empty"))
        room_id = item.get("id")
        if isinstance(room_id, str) and room_id.strip():
            if room_id in room_ids:
                errors.append(
                    _error(f"{path}/id", "unique", f"duplicate room id {room_id!r}")
                )
            room_ids.add(room_id)
        if "origin_m" in item:
            errors.extend(
                _check_vector(
                    item["origin_m"],
                    f"{path}/origin_m",
                    minimum=0,
                )
            )
        if "size_m" in item:
            errors.extend(
                _check_vector(
                    item["size_m"],
                    f"{path}/size_m",
                    minimum=0,
                    exclusive=True,
                )
            )

    seen_ids: set[str] = set()
    for index, item in enumerate(objects):
        path = f"/objects/{index}"
        if not isinstance(item, dict):
            errors.append(_error(path, "type", "must be an object"))
            continue

        errors.extend(_check_exact_fields(item, OBJECT_FIELDS, path))

        for field in ("id", "room_id", "category", "appearance"):
            if field not in item:
                continue
            field_value = item[field]
            field_path = f"{path}/{field}"
            if not isinstance(field_value, str):
                errors.append(_error(field_path, "type", "must be a string"))
            elif not field_value.strip():
                errors.append(_error(field_path, "min_length", "must be non-empty"))

        object_id = item.get("id")
        if isinstance(object_id, str) and object_id.strip():
            if object_id in seen_ids:
                errors.append(
                    _error(f"{path}/id", "unique", f"duplicate object id {object_id!r}")
                )
            seen_ids.add(object_id)

        room_id = item.get("room_id")
        if isinstance(room_id, str) and room_id.strip() and room_id not in room_ids:
            errors.append(
                _error(
                    f"{path}/room_id",
                    "reference",
                    f"unknown room id {room_id!r}",
                )
            )

        if "position_m" in item:
            errors.extend(
                _check_vector(
                    item["position_m"],
                    f"{path}/position_m",
                    minimum=0,
                )
            )
        if "size_m" in item:
            errors.extend(
                _check_vector(
                    item["size_m"],
                    f"{path}/size_m",
                    minimum=0,
                    exclusive=True,
                )
            )
        if "yaw_deg" in item:
            yaw = item["yaw_deg"]
            yaw_path = f"{path}/yaw_deg"
            if not _is_number(yaw):
                errors.append(_error(yaw_path, "type", "must be a JSON number"))
            elif not math.isfinite(yaw):
                errors.append(_error(yaw_path, "finite", "must be finite"))

    return errors
