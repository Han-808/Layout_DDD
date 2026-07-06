from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.object_aliasing import remap_layout_aliases_to_canonical


OPTIONAL_NULL_FIELDS = {
    "region_id",
    "room_id",
    "support_parent",
    "support_surface",
    "parent_id",
}


def sanitize_layout_optional_nulls(layout: dict, optional_fields: set[str] | None = None) -> tuple[dict, dict]:
    """Remove known optional null fields from a layout copy and report counts."""

    if not isinstance(layout, dict):
        return layout, {"removed_optional_null_fields": [], "sanitized_layout_used": False}

    fields = optional_fields or OPTIONAL_NULL_FIELDS
    sanitized = deepcopy(layout)
    counts: dict[str, int] = {}
    _remove_optional_nulls(sanitized, fields, counts)
    return sanitized, {
        "removed_optional_null_fields": [
            {"field": field, "count": count}
            for field, count in sorted(counts.items())
        ],
        "sanitized_layout_used": bool(counts),
    }


def enforce_layout_object_set(
    layout: dict,
    bm_instance: dict,
    *,
    previous_layout: dict | None = None,
    stage: str = "generation",
) -> tuple[dict, dict]:
    """Preserve the benchmark object-id set after model generation/repair."""

    if not isinstance(layout, dict):
        return layout, {"object_set_normalization_used": False, "reason": "layout_not_object"}

    required_specs = [obj for obj in bm_instance.get("objects", []) if isinstance(obj, dict) and isinstance(obj.get("id"), str)]
    if not required_specs:
        return layout, {"object_set_normalization_used": False, "reason": "no_required_object_specs"}

    normalized, alias_report = remap_layout_aliases_to_canonical(layout, bm_instance, stage=stage)
    current_by_id, duplicate_flags = _objects_by_id(normalized.get("objects"))
    previous_by_id, _ = _objects_by_id(previous_layout.get("objects") if isinstance(previous_layout, dict) else [])
    required_ids = [str(obj["id"]) for obj in required_specs]
    required_set = set(required_ids)
    flags = list(duplicate_flags)
    objects = []

    for spec in required_specs:
        object_id = str(spec["id"])
        source = "model_output"
        if object_id in current_by_id:
            obj = deepcopy(current_by_id[object_id])
        elif object_id in previous_by_id:
            obj = deepcopy(previous_by_id[object_id])
            source = "previous_layout"
            flags.append(_normalization_flag("missing_object_restored", object_id, f"{object_id} was missing from {stage} output and restored from previous layout."))
        else:
            obj = _object_from_spec(spec)
            source = "input_spec"
            flags.append(_normalization_flag("missing_object_synthesized", object_id, f"{object_id} was missing from {stage} output and synthesized from input bbox metadata."))
        _restore_required_fields(obj, spec, source)
        objects.append(obj)

    extra_ids = sorted(object_id for object_id in current_by_id if object_id not in required_set)
    for object_id in extra_ids:
        flags.append(_normalization_flag("extra_object_dropped", object_id, f"{object_id} is not in the input object set and was dropped."))

    normalized["objects"] = objects
    if isinstance(normalized.get("hierarchy"), dict):
        normalized["hierarchy"] = _normalize_hierarchy(normalized["hierarchy"], required_set)

    report = {
        "object_set_normalization_used": bool(flags),
        "stage": stage,
        "required_object_count": len(required_ids),
        "final_object_count": len(objects),
        "required_object_ids": required_ids,
        "missing_restored_or_synthesized": [flag["object_id"] for flag in flags if flag["type"] in {"missing_object_restored", "missing_object_synthesized"}],
        "extra_dropped": extra_ids,
        "duplicate_dropped": [flag["object_id"] for flag in flags if flag["type"] == "duplicate_object_dropped"],
        "flags": flags,
    }
    if alias_report.get("alias_remap_used"):
        report["alias_remap"] = alias_report
        report["object_set_normalization_used"] = bool(flags or alias_report.get("flags"))
    normalized["_layout_object_set_normalization"] = report
    return normalized, report


def _remove_optional_nulls(value: Any, fields: set[str], counts: dict[str, int]) -> None:
    if isinstance(value, dict):
        for key in list(value.keys()):
            if key in fields and value.get(key) is None:
                value.pop(key)
                counts[key] = counts.get(key, 0) + 1
            else:
                _remove_optional_nulls(value[key], fields, counts)
    elif isinstance(value, list):
        for item in value:
            _remove_optional_nulls(item, fields, counts)


def _objects_by_id(objects: object) -> tuple[dict[str, dict], list[dict]]:
    by_id: dict[str, dict] = {}
    flags = []
    if not isinstance(objects, list):
        return by_id, flags
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("object_id") or obj.get("id")
        if not isinstance(object_id, str) or not object_id:
            continue
        if object_id in by_id:
            flags.append(_normalization_flag("duplicate_object_dropped", object_id, f"Duplicate {object_id} at index {index} was dropped."))
            continue
        by_id[object_id] = obj
    return by_id, flags


def _restore_required_fields(obj: dict, spec: dict, source: str) -> None:
    object_id = str(spec["id"])
    obj["object_id"] = object_id
    obj.setdefault("canonical_object_id", object_id)
    if obj.get("model_object_id"):
        obj["model_object_id"] = str(obj["model_object_id"])
    obj["category"] = str(spec.get("category") or obj.get("category") or "object")
    if obj.get("model_category"):
        obj["model_category"] = str(obj["model_category"])
    if not _valid_vector(obj.get("size"), positive=True):
        obj["size"] = _size_from_spec(spec)
    if not _valid_vector(obj.get("center"), positive=False):
        obj["center"] = _center_from_spec(spec)
    obj.setdefault("yaw", 0)
    obj.setdefault("support_parent", "floor")
    obj.setdefault("normalization_source", source)


def _object_from_spec(spec: dict) -> dict:
    return {
        "object_id": str(spec["id"]),
        "category": str(spec.get("category") or "object"),
        "center": _center_from_spec(spec),
        "size": _size_from_spec(spec),
        "yaw": 0,
        "support_parent": "floor",
        "normalization_source": "input_spec",
    }


def _center_from_spec(spec: dict) -> list[float]:
    hint = spec.get("layout_center_hint")
    if _valid_vector(hint, positive=False):
        return [float(hint[0]), float(hint[1]), float(hint[2])]
    floor_position = spec.get("source_floor_position")
    size = _size_from_spec(spec)
    if isinstance(floor_position, list) and len(floor_position) >= 2:
        return [_safe_float(floor_position[0], 0.0), _safe_float(floor_position[1], 0.0), size[2] / 2.0]
    return [0.0, 0.0, size[2] / 2.0]


def _size_from_spec(spec: dict) -> list[float]:
    size = spec.get("bbox_size")
    if _valid_vector(size, positive=True):
        return [float(size[0]), float(size[1]), float(size[2])]
    return [0.8, 0.8, 0.8]


def _normalize_hierarchy(hierarchy: dict, required_set: set[str]) -> dict:
    normalized = deepcopy(hierarchy)
    for key in ["floor_objects", "supported_objects"]:
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = [item for item in value if isinstance(item, str) and item in required_set]
    return normalized


def _normalization_flag(flag_type: str, object_id: str, message: str) -> dict:
    return {
        "type": flag_type,
        "object_id": object_id,
        "objects": [object_id],
        "severity": "medium",
        "message": message,
    }


def _valid_vector(value: object, *, positive: bool) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    for item in value:
        if not isinstance(item, (int, float)):
            return False
        if positive and float(item) <= 0:
            return False
    return True


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
