from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_UNIT = "meter"
DEFAULT_COORDINATE_SYSTEM = {
    "origin": "case floor-plan coordinate frame; HSSD cases may use negative x/y values",
    "x_axis": "floor-plan x coordinate",
    "y_axis": "floor-plan y/depth coordinate",
    "z_axis": "height",
    "rotation_unit": "degree",
}

SUPPORT_FIELDS = ("support_parent", "support_surface", "parent_id", "region_id")
LEGACY_LAYOUT_BBOX_FIELDS = {"center", "size", "yaw"}
ASSET_CORE_FIELDS = {
    "asset_id",
    "object_id",
    "category",
    "bbox",
    "asset_ref",
    "metadata",
}


def normalize_scene(scene: dict) -> dict:
    """Return a scene-shaped copy with stable defaults for evaluation.

    The canonical evaluation input is a scene with assets. If callers pass a
    legacy bbox layout by mistake, keep the API forgiving and adapt it.
    """

    if not isinstance(scene, dict):
        return {
            "scene_id": "scene",
            "unit": DEFAULT_UNIT,
            "assets": [],
            "metadata": {"normalization_warning": f"scene must be a JSON object, got {type(scene).__name__}"},
        }

    if "assets" not in scene and isinstance(scene.get("objects"), list):
        return layout_to_scene(scene)

    normalized = deepcopy(scene)
    scene_id = _first_nonempty_str(
        normalized.get("scene_id"),
        normalized.get("case_id"),
        normalized.get("task_id"),
        default="scene",
    )
    normalized["scene_id"] = scene_id
    normalized.setdefault("unit", DEFAULT_UNIT)
    if "assets" not in normalized or not isinstance(normalized.get("assets"), list):
        normalized["assets"] = []

    assets = []
    for index, asset in enumerate(normalized["assets"]):
        if not isinstance(asset, dict):
            assets.append(
                {
                    "asset_id": f"asset_{index + 1:03d}",
                    "category": "object",
                    "metadata": {"normalization_warning": "asset entry was not a JSON object", "raw_asset": asset},
                }
            )
            continue
        item = deepcopy(asset)
        item["asset_id"] = _first_nonempty_str(item.get("asset_id"), item.get("object_id"), default=f"asset_{index + 1:03d}")
        item["category"] = _first_nonempty_str(item.get("category"), default="object")
        assets.append(item)
    normalized["assets"] = assets
    return normalized


def layout_to_scene(layout: dict, case: dict | None = None) -> dict:
    """Adapt a legacy bbox layout into an asset-aware evaluation scene."""

    legacy_layout = deepcopy(layout) if isinstance(layout, dict) else {}
    case_copy = deepcopy(case) if isinstance(case, dict) else None
    scene_id = _first_nonempty_str(
        legacy_layout.get("scene_id"),
        case_copy.get("scene_id") if case_copy else None,
        case_copy.get("case_id") if case_copy else None,
        case_copy.get("task_id") if case_copy else None,
        default="scene",
    )
    scene: dict[str, Any] = {
        "scene_id": scene_id,
        "unit": _first_nonempty_str(legacy_layout.get("unit"), case_copy.get("unit") if case_copy else None, default=DEFAULT_UNIT),
        "assets": [],
    }

    _copy_first_mapping(scene, "coordinate_system", legacy_layout, case_copy)
    _copy_first_mapping(scene, "room", case_copy, legacy_layout)

    objects = legacy_layout.get("objects") if isinstance(legacy_layout.get("objects"), list) else []
    case_object_refs = _case_asset_ref_lookup(case_copy)
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        asset_id = _first_nonempty_str(obj.get("object_id"), obj.get("id"), obj.get("asset_id"), default=f"asset_{index + 1:03d}")
        asset: dict[str, Any] = {
            "asset_id": asset_id,
            "object_id": _first_nonempty_str(obj.get("object_id"), obj.get("id"), default=asset_id),
            "category": _first_nonempty_str(obj.get("category"), default="object"),
        }
        bbox = _bbox_from_legacy_object(obj)
        if bbox is not None:
            asset["bbox"] = bbox
        if isinstance(obj.get("asset_ref"), dict):
            asset["asset_ref"] = deepcopy(obj["asset_ref"])
        elif asset["object_id"] in case_object_refs:
            asset["asset_ref"] = deepcopy(case_object_refs[asset["object_id"]])
        elif asset["category"] in case_object_refs:
            asset["asset_ref"] = deepcopy(case_object_refs[asset["category"]])
        for field in SUPPORT_FIELDS:
            if field in obj and obj.get(field) is not None:
                asset[field] = deepcopy(obj[field])
        for key, value in obj.items():
            if (
                key not in ASSET_CORE_FIELDS
                and key not in SUPPORT_FIELDS
                and key not in LEGACY_LAYOUT_BBOX_FIELDS
                and key not in asset
            ):
                asset[key] = deepcopy(value)
        metadata = deepcopy(obj.get("metadata")) if isinstance(obj.get("metadata"), dict) else {}
        metadata.setdefault("source_layout_object", deepcopy(obj))
        asset["metadata"] = metadata
        scene["assets"].append(asset)

    _copy_first_list(scene, "relations", case_copy, legacy_layout)
    _copy_first_list(scene, "attachments", case_copy, legacy_layout)
    _copy_first_mapping(scene, "hierarchy", legacy_layout, case_copy)

    source = deepcopy(case_copy.get("source")) if case_copy and isinstance(case_copy.get("source"), dict) else {}
    if isinstance(legacy_layout.get("source"), dict):
        source = {**deepcopy(legacy_layout["source"]), **source}
    source.setdefault("evaluation_scene_adapter", "layout_to_scene")
    source.setdefault("legacy_layout_representation", "bbox_objects")
    if source:
        scene["source"] = source

    eval_context: dict[str, Any] = {}
    if case_copy is not None:
        eval_context["legacy_case"] = case_copy
    layout_metadata = {
        key: deepcopy(value)
        for key, value in legacy_layout.items()
        if isinstance(key, str) and key.startswith("_")
    }
    if layout_metadata:
        eval_context["legacy_layout_metadata"] = layout_metadata
    if eval_context:
        scene["eval_context"] = eval_context
    return scene


def scene_to_layout(scene: dict) -> dict:
    """Adapt bbox-bearing scene assets back to the legacy bbox layout shape."""

    normalized = normalize_scene(scene)
    layout: dict[str, Any] = {
        "scene_id": _first_nonempty_str(normalized.get("scene_id"), default="scene"),
        "unit": _first_nonempty_str(normalized.get("unit"), default=DEFAULT_UNIT),
        "coordinate_system": deepcopy(normalized.get("coordinate_system"))
        if isinstance(normalized.get("coordinate_system"), dict)
        else deepcopy(DEFAULT_COORDINATE_SYSTEM),
        "objects": [],
    }

    non_bbox_assets: list[dict] = []
    for index, asset in enumerate(normalized.get("assets", [])):
        if not isinstance(asset, dict):
            continue
        bbox = _bbox_from_asset(asset)
        if bbox is None:
            non_bbox_assets.append(_non_bbox_asset_record(asset, index))
            continue
        asset_id = _first_nonempty_str(asset.get("asset_id"), default=f"asset_{index + 1:03d}")
        obj: dict[str, Any] = {
            "object_id": _first_nonempty_str(asset.get("object_id"), asset_id, default=asset_id),
            "category": _first_nonempty_str(asset.get("category"), default="object"),
            "center": deepcopy(bbox.get("center")),
            "size": deepcopy(bbox.get("size")),
            "yaw": deepcopy(bbox.get("yaw")),
            "asset_id": asset_id,
        }
        if isinstance(asset.get("asset_ref"), dict):
            obj["asset_ref"] = deepcopy(asset["asset_ref"])
        if isinstance(asset.get("metadata"), dict):
            obj["metadata"] = deepcopy(asset["metadata"])
        _copy_support_fields(obj, asset, bbox)
        for key, value in asset.items():
            if key not in ASSET_CORE_FIELDS and key not in SUPPORT_FIELDS and key not in obj:
                obj[key] = deepcopy(value)
        layout["objects"].append(obj)

    relations = _legacy_relations_from_scene(normalized)
    if relations:
        layout["relations"] = relations
    _copy_if_present(layout, "hierarchy", normalized)
    if isinstance(normalized.get("attachments"), list):
        layout["_scene_attachments"] = deepcopy(normalized["attachments"])
    if non_bbox_assets:
        layout["_non_bbox_assets"] = non_bbox_assets
    if isinstance(normalized.get("source"), dict):
        layout["_scene_source"] = deepcopy(normalized["source"])
    return layout


def scene_to_case(scene: dict) -> dict:
    """Build a benchmark-case view from a scene for existing judge/metric code."""

    normalized = normalize_scene(scene)
    eval_context = normalized.get("eval_context") if isinstance(normalized.get("eval_context"), dict) else {}
    legacy_case = eval_context.get("legacy_case") if isinstance(eval_context, dict) else None
    if isinstance(legacy_case, dict):
        case = deepcopy(legacy_case)
    else:
        case = {
            "case_id": normalized["scene_id"],
            "task_id": normalized["scene_id"],
            "input_level": "structured_basic",
            "scene_representation_mode": "evaluation_scene",
            "description": {"text": str(normalized.get("description") or "")},
            "objects": [],
        }

    case.setdefault("case_id", normalized["scene_id"])
    case.setdefault("task_id", normalized["scene_id"])
    case.setdefault("input_level", "structured_basic")
    case.setdefault("scene_representation_mode", "evaluation_scene")
    if isinstance(normalized.get("room"), dict):
        case["room"] = deepcopy(normalized["room"])
    case["objects"] = _case_objects_from_scene(normalized, fallback=case.get("objects"))
    if isinstance(normalized.get("relations"), list):
        case["relations"] = deepcopy(normalized["relations"])
    if isinstance(normalized.get("attachments"), list):
        case["attachments"] = deepcopy(normalized["attachments"])
    source = case.get("source") if isinstance(case.get("source"), dict) else {}
    if isinstance(normalized.get("source"), dict):
        source = {**deepcopy(source), **deepcopy(normalized["source"])}
    source.setdefault("scene_input_type", "evaluation_scene")
    case["source"] = source
    return case


def scene_adapter_summary(scene: dict, layout: dict | None = None) -> dict:
    normalized = normalize_scene(scene)
    bbox_asset_ids = []
    asset_ref_ids = []
    non_bbox_assets = []
    for index, asset in enumerate(normalized.get("assets", [])):
        if not isinstance(asset, dict):
            continue
        asset_id = _first_nonempty_str(asset.get("asset_id"), default=f"asset_{index + 1:03d}")
        if isinstance(asset.get("asset_ref"), dict):
            asset_ref_ids.append(asset_id)
        if _bbox_from_asset(asset) is None:
            non_bbox_assets.append(_non_bbox_asset_record(asset, index))
        else:
            bbox_asset_ids.append(asset_id)
    asset_count = len(normalized.get("assets", []))
    return {
        "input_type": "scene",
        "scene_id": normalized.get("scene_id"),
        "asset_count": asset_count,
        "bbox_asset_count": len(bbox_asset_ids),
        "non_bbox_asset_count": len(non_bbox_assets),
        "asset_ref_asset_count": len(asset_ref_ids),
        "asset_ref_available_rate": (float(len(asset_ref_ids)) / float(asset_count)) if asset_count else None,
        "bbox_asset_ids": bbox_asset_ids,
        "asset_ref_asset_ids": asset_ref_ids,
        "non_bbox_assets": non_bbox_assets,
        "legacy_layout_object_count": len(layout.get("objects", [])) if isinstance(layout, dict) and isinstance(layout.get("objects"), list) else 0,
    }


def _bbox_from_legacy_object(obj: dict) -> dict | None:
    if not any(key in obj for key in ("center", "size", "yaw")):
        return None
    return {
        "center": deepcopy(obj.get("center")),
        "size": deepcopy(obj.get("size")),
        "yaw": deepcopy(obj.get("yaw", 0)),
    }


def _bbox_from_asset(asset: dict) -> dict | None:
    bbox = asset.get("bbox")
    if isinstance(bbox, dict) and all(key in bbox for key in ("center", "size", "yaw")):
        return bbox
    if all(key in asset for key in ("center", "size", "yaw")):
        return {"center": asset.get("center"), "size": asset.get("size"), "yaw": asset.get("yaw")}
    return None


def _case_objects_from_scene(scene: dict, fallback: object = None) -> list[dict]:
    assets = [asset for asset in scene.get("assets", []) if isinstance(asset, dict)]
    if not assets and isinstance(fallback, list):
        return deepcopy(fallback)
    objects = []
    for index, asset in enumerate(assets):
        asset_id = _first_nonempty_str(asset.get("asset_id"), asset.get("object_id"), default=f"asset_{index + 1:03d}")
        obj = {
            "id": _first_nonempty_str(asset.get("object_id"), asset_id, default=asset_id),
            "category": _first_nonempty_str(asset.get("category"), default="object"),
            "required": asset.get("required", True),
        }
        bbox = _bbox_from_asset(asset)
        if bbox is not None and isinstance(bbox.get("size"), list):
            obj["bbox_size"] = deepcopy(bbox["size"])
        objects.append(obj)
    return objects


def _case_asset_ref_lookup(case: dict | None) -> dict[str, dict]:
    if not isinstance(case, dict):
        return {}
    refs: dict[str, dict] = {}
    objects = [item for item in case.get("objects", []) if isinstance(item, dict)]
    category_counts: dict[str, int] = {}
    for item in objects:
        category = _first_nonempty_str(item.get("category"))
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
    for item in objects:
        asset_ref = item.get("asset_ref")
        if not isinstance(asset_ref, dict):
            source = item.get("source")
            asset_ref = source.get("asset_ref") if isinstance(source, dict) else None
        if not isinstance(asset_ref, dict):
            continue
        for key in [
            _first_nonempty_str(item.get("object_id")),
            _first_nonempty_str(item.get("id")),
            _first_nonempty_str(item.get("asset_id")),
        ]:
            if key:
                refs.setdefault(key, deepcopy(asset_ref))
        category = _first_nonempty_str(item.get("category"))
        if category and category_counts.get(category) == 1:
            refs.setdefault(category, deepcopy(asset_ref))
    return refs


def _copy_support_fields(obj: dict, asset: dict, bbox: dict) -> None:
    bbox_metadata = bbox.get("metadata") if isinstance(bbox.get("metadata"), dict) else {}
    for field in SUPPORT_FIELDS:
        value = asset.get(field)
        if value is None:
            value = bbox.get(field)
        if value is None:
            value = bbox_metadata.get(field)
        if value is not None:
            obj[field] = deepcopy(value)


def _legacy_relations_from_scene(scene: dict) -> list[dict]:
    relations = []
    for item in scene.get("relations", []) if isinstance(scene.get("relations"), list) else []:
        if not isinstance(item, dict):
            continue
        relation = deepcopy(item)
        if "source" not in relation and relation.get("subject") is not None:
            relation["source"] = relation.get("subject")
        if "target" not in relation and relation.get("object") is not None:
            relation["target"] = relation.get("object")
        if isinstance(relation.get("type"), str) and isinstance(relation.get("source"), str) and relation.get("source"):
            relations.append(relation)
    return relations


def _non_bbox_asset_record(asset: dict, index: int) -> dict:
    return {
        "asset_index": index,
        "asset_id": _first_nonempty_str(asset.get("asset_id"), asset.get("object_id"), default=f"asset_{index + 1:03d}"),
        "object_id": asset.get("object_id"),
        "category": asset.get("category"),
        "reason": "asset has no complete bbox",
    }


def _copy_first_mapping(target: dict, key: str, *sources: dict | None) -> None:
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get(key), dict):
            target[key] = deepcopy(source[key])
            return


def _copy_first_list(target: dict, key: str, *sources: dict | None) -> None:
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get(key), list):
            target[key] = deepcopy(source[key])
            return


def _copy_if_present(target: dict, key: str, source: dict) -> None:
    if key in source:
        target[key] = deepcopy(source[key])


def _first_nonempty_str(*values: object, default: str = "") -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
        if value is not None and value != "" and not isinstance(value, (dict, list, tuple, set)):
            return str(value)
    return default
