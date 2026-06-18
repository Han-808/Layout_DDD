from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class SchemaCheckResult:
    valid: bool
    layout: dict | None
    failures: list[dict]


def check_layout_schema(layout_input: dict | str, layout_schema: dict | None = None) -> SchemaCheckResult:
    failures: list[dict] = []
    layout = _parse_layout(layout_input, failures)
    if layout is None:
        return SchemaCheckResult(valid=False, layout=None, failures=failures)

    if layout_schema:
        failures.extend(_jsonschema_failures(layout, layout_schema))

    failures.extend(_manual_layout_failures(layout))
    return SchemaCheckResult(valid=not failures, layout=layout, failures=failures)


def _parse_layout(layout_input: dict | str, failures: list[dict]) -> dict | None:
    if isinstance(layout_input, dict):
        return layout_input
    if not isinstance(layout_input, str):
        failures.append(
            {
                "type": "json_parse_error",
                "message": f"Layout must be a JSON object or JSON string, got {type(layout_input).__name__}.",
            }
        )
        return None
    try:
        parsed = json.loads(layout_input)
    except json.JSONDecodeError as exc:
        failures.append(
            {
                "type": "json_parse_error",
                "message": f"Layout JSON parse failed: {exc.msg}.",
            }
        )
        return None
    if not isinstance(parsed, dict):
        failures.append({"type": "schema_validation", "message": "Layout JSON root must be an object."})
        return None
    return parsed


def _jsonschema_failures(layout: dict, layout_schema: dict) -> list[dict]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("jsonschema is required for schema validation.") from exc

    validator = Draft202012Validator(layout_schema)
    failures = []
    for error in sorted(validator.iter_errors(layout), key=lambda e: list(e.path)):
        path = "$" + "".join(f"[{p!r}]" if isinstance(p, int) else f".{p}" for p in error.path)
        failures.append(
            {
                "type": "schema_validation",
                "path": path,
                "message": error.message,
            }
        )
    return failures


def _manual_layout_failures(layout: dict) -> list[dict]:
    failures: list[dict] = []
    required_top_level = ["scene_id", "unit", "coordinate_system", "objects"]
    for field in required_top_level:
        if field not in layout:
            failures.append({"type": "missing_required_field", "field": field, "message": f"Missing field '{field}'."})

    if layout.get("unit") != "meter":
        failures.append({"type": "invalid_unit", "message": "Layout unit must be 'meter'."})

    coordinate_system = layout.get("coordinate_system")
    if not isinstance(coordinate_system, dict):
        failures.append({"type": "missing_coordinate_system", "message": "coordinate_system must exist."})
    elif coordinate_system.get("rotation_unit") != "degree":
        failures.append(
            {
                "type": "invalid_rotation_unit",
                "message": "coordinate_system.rotation_unit must be 'degree'.",
            }
        )

    objects = layout.get("objects")
    if not isinstance(objects, list):
        failures.append({"type": "invalid_objects", "message": "objects must be a list."})
        return failures

    seen: dict[str, int] = {}
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            failures.append({"type": "invalid_object", "index": idx, "message": "Each object must be a JSON object."})
            continue

        object_id = obj.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            failures.append({"type": "invalid_object_id", "index": idx, "message": "object_id must be a non-empty string."})
        else:
            seen[object_id] = seen.get(object_id, 0) + 1

        if not isinstance(obj.get("category"), str) or not obj.get("category"):
            failures.append(
                {
                    "type": "invalid_category",
                    "object_id": object_id,
                    "message": "category must be a non-empty string.",
                }
            )
        if not _is_numeric_list(obj.get("center"), length=3, positive=False):
            failures.append(
                {
                    "type": "invalid_center",
                    "object_id": object_id,
                    "message": "center must be a length-3 numeric list.",
                }
            )
        if not _is_numeric_list(obj.get("size"), length=3, positive=True):
            failures.append(
                {
                    "type": "invalid_size",
                    "object_id": object_id,
                    "message": "size must be a length-3 positive numeric list.",
                }
            )
        if not _is_number(obj.get("yaw")):
            failures.append(
                {
                    "type": "invalid_yaw",
                    "object_id": object_id,
                    "message": "yaw must be numeric.",
                }
            )

    for object_id, count in sorted(seen.items()):
        if count > 1:
            failures.append(
                {
                    "type": "duplicate_object_id",
                    "objects": [object_id],
                    "message": f"object_id '{object_id}' appears {count} times.",
                }
            )
    return failures


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_list(value: Any, *, length: int, positive: bool) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    if not all(_is_number(item) for item in value):
        return False
    if positive and not all(item > 0 for item in value):
        return False
    return True
