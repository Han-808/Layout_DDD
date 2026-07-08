from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from benchmark.data.local_assets import LOCAL_ASSET_SOURCE, resolve_local_asset_ref
from benchmark.data.local_scenes import LOCAL_SCENE_SOURCE, load_local_scene, resolve_local_scene_ref


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
    "placement",
    "dimensions",
    "bbox",
    "asset_ref",
    "metadata",
}


def normalize_scene(scene: dict) -> dict:
    """Return a scene-shaped copy with stable defaults for evaluation.

    The canonical evaluation input is a scene with assets. If callers pass an
    old legend layout by mistake, keep the API forgiving and adapt it.
    """

    if not isinstance(scene, dict):
        return {
            "scene_id": "scene",
            "unit": DEFAULT_UNIT,
            "assets": [],
            "metadata": {"normalization_warning": f"scene must be a JSON object, got {type(scene).__name__}"},
        }

    scene = _load_or_enrich_local_scene(scene)

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
        _enrich_local_asset(item)
        item["category"] = _first_nonempty_str(
            item.get("category"),
            item.get("asset_ref", {}).get("category") if isinstance(item.get("asset_ref"), dict) else None,
            default="object",
        )
        assets.append(item)
    normalized["assets"] = assets
    return normalized


def layout_to_scene(layout: dict, case: dict | None = None) -> dict:
    """Adapt an old legend layout into an asset-aware evaluation scene."""

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
    _copy_scene_import_fields(scene, legacy_layout)
    _enrich_local_scene(scene)

    objects = legacy_layout.get("objects") if isinstance(legacy_layout.get("objects"), list) else []
    case_object_refs = _case_asset_ref_lookup(case_copy)
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        asset_id = _first_nonempty_str(
            obj.get("object_id"),
            obj.get("id"),
            obj.get("asset_id"),
            obj.get("jid"),
            obj.get("name"),
            default=f"asset_{index + 1:03d}",
        )
        asset: dict[str, Any] = {
            "asset_id": asset_id,
            "object_id": _first_nonempty_str(obj.get("object_id"), obj.get("id"), obj.get("jid"), default=asset_id),
            "category": _first_nonempty_str(obj.get("category"), obj.get("class"), obj.get("type"), default="object"),
        }
        geometry = _geometry_from_legacy_object(obj)
        if geometry is not None:
            asset["placement"] = {
                "position": deepcopy(geometry.get("center")),
                "yaw_degrees": deepcopy(geometry.get("yaw", 0)),
            }
            asset["dimensions"] = deepcopy(geometry.get("size"))
        if isinstance(obj.get("asset_ref"), dict):
            asset["asset_ref"] = deepcopy(obj["asset_ref"])
        elif asset["object_id"] in case_object_refs:
            asset["asset_ref"] = deepcopy(case_object_refs[asset["object_id"]])
        elif asset["category"] in case_object_refs:
            asset["asset_ref"] = deepcopy(case_object_refs[asset["category"]])
        _enrich_local_asset(asset)
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
        if geometry is not None:
            metadata.setdefault("legend_geometry_source", "layout_center_size_yaw")
        asset["metadata"] = metadata
        scene["assets"].append(asset)

    _copy_first_list(scene, "relations", case_copy, legacy_layout)
    _copy_first_list(scene, "attachments", case_copy, legacy_layout)
    _copy_first_mapping(scene, "hierarchy", legacy_layout, case_copy)

    source = deepcopy(case_copy.get("source")) if case_copy and isinstance(case_copy.get("source"), dict) else {}
    if isinstance(legacy_layout.get("source"), dict):
        source = {**deepcopy(legacy_layout["source"]), **source}
    source.setdefault("evaluation_scene_adapter", "layout_to_scene")
    source.setdefault("legend_layout_representation", "geometry_proxy_objects")
    if source:
        scene["source"] = source
    _enrich_local_scene(scene)

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
    """Adapt scene assets back to the legacy geometry-proxy layout shape."""

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
        geometry = _geometry_from_asset(asset)
        if geometry is None:
            non_bbox_assets.append(_non_geometric_asset_record(asset, index))
            continue
        asset_id = _first_nonempty_str(asset.get("asset_id"), default=f"asset_{index + 1:03d}")
        obj: dict[str, Any] = {
            "object_id": _first_nonempty_str(asset.get("object_id"), asset_id, default=asset_id),
            "category": _first_nonempty_str(asset.get("category"), default="object"),
            "center": deepcopy(geometry.get("center")),
            "size": deepcopy(geometry.get("size")),
            "yaw": deepcopy(geometry.get("yaw")),
            "asset_id": asset_id,
        }
        if isinstance(asset.get("asset_ref"), dict):
            obj["asset_ref"] = deepcopy(asset["asset_ref"])
        if isinstance(asset.get("metadata"), dict):
            obj["metadata"] = deepcopy(asset["metadata"])
        _copy_support_fields(obj, asset, geometry)
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
        layout["_non_geometric_assets"] = non_bbox_assets
        layout["_non_bbox_assets"] = non_bbox_assets
    if isinstance(normalized.get("source"), dict):
        layout["_scene_source"] = deepcopy(normalized["source"])
    return layout


def legend_layout_to_scene(layout: dict, case: dict | None = None) -> dict:
    return layout_to_scene(layout, case)


def scene_to_legend_layout(scene: dict) -> dict:
    return scene_to_layout(scene)


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
    geometry_asset_ids = []
    asset_ref_ids = []
    non_geometric_assets = []
    for index, asset in enumerate(normalized.get("assets", [])):
        if not isinstance(asset, dict):
            continue
        asset_id = _first_nonempty_str(asset.get("asset_id"), default=f"asset_{index + 1:03d}")
        if isinstance(asset.get("asset_ref"), dict):
            asset_ref_ids.append(asset_id)
        if _geometry_from_asset(asset) is None:
            non_geometric_assets.append(_non_geometric_asset_record(asset, index))
        else:
            geometry_asset_ids.append(asset_id)
    asset_count = len(normalized.get("assets", []))
    local_asset_ref_count = sum(1 for asset in normalized.get("assets", []) if _has_local_asset_ref(asset))
    local_scene_ref = _local_scene_ref(normalized)
    return {
        "input_type": "scene",
        "scene_id": normalized.get("scene_id"),
        "asset_count": asset_count,
        "geometry_asset_count": len(geometry_asset_ids),
        "non_geometric_asset_count": len(non_geometric_assets),
        "geometry_available_rate": (float(len(geometry_asset_ids)) / float(asset_count)) if asset_count else None,
        "asset_ref_asset_count": len(asset_ref_ids),
        "asset_ref_available_rate": (float(len(asset_ref_ids)) / float(asset_count)) if asset_count else None,
        "local_asset_ref_count": local_asset_ref_count,
        "local_asset_available_rate": (
            float(local_asset_ref_count) / float(asset_count)
        )
        if asset_count
        else None,
        "local_scene_ref_available": isinstance(local_scene_ref, dict) and local_scene_ref.get("source") == LOCAL_SCENE_SOURCE,
        "local_scene_id": local_scene_ref.get("scene_id") if isinstance(local_scene_ref, dict) else None,
        "local_scene_json_path": local_scene_ref.get("scene_json_path") if isinstance(local_scene_ref, dict) else None,
        "geometry_asset_ids": geometry_asset_ids,
        "asset_ref_asset_ids": asset_ref_ids,
        "non_geometric_assets": non_geometric_assets,
        "legacy_layout_object_count": len(layout.get("objects", [])) if isinstance(layout, dict) and isinstance(layout.get("objects"), list) else 0,
        "legend_compat": {
            "bbox_asset_count": len(geometry_asset_ids),
            "non_bbox_asset_count": len(non_geometric_assets),
            "bbox_asset_ids": geometry_asset_ids,
            "non_bbox_assets": non_geometric_assets,
        },
    }



def _load_or_enrich_local_scene(scene: dict) -> dict:
    existing_ref = _local_scene_ref(scene)
    if "assets" not in scene and "objects" not in scene:
        loaded = load_local_scene(existing_ref, scene_id=scene.get("scene_id"))
        if isinstance(loaded, dict):
            merged = deepcopy(loaded)
            for key, value in scene.items():
                if value is None:
                    continue
                if key in {"scene_ref", "source", "metadata"} and isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**deepcopy(merged[key]), **deepcopy(value)}
                elif key not in {"assets", "objects"}:
                    merged[key] = deepcopy(value)
            _enrich_local_scene(merged)
            return merged
    normalized = deepcopy(scene)
    _enrich_local_scene(normalized)
    return normalized


def _copy_scene_import_fields(scene: dict, source: dict) -> None:
    if isinstance(source.get("scene_ref"), dict):
        scene["scene_ref"] = deepcopy(source["scene_ref"])
    if source.get("scene_type") is not None:
        scene["scene_type"] = deepcopy(source["scene_type"])
    room = deepcopy(scene.get("room")) if isinstance(scene.get("room"), dict) else {}
    if isinstance(source.get("boundary"), list) and "boundary" not in room:
        room["boundary"] = deepcopy(source["boundary"])
    if source.get("scene_height") is not None and "wall_height" not in room:
        room["wall_height"] = deepcopy(source["scene_height"])
    if source.get("scene_type") is not None and "scene_type" not in room:
        room["scene_type"] = deepcopy(source["scene_type"])
    if room:
        scene["room"] = room


def _enrich_local_scene(scene: dict) -> None:
    existing_ref = _local_scene_ref(scene)
    local_ref = resolve_local_scene_ref(existing_ref, scene_id=scene.get("scene_id"))
    if not local_ref:
        return
    merged_ref = _merge_scene_ref(local_ref, existing_ref if isinstance(existing_ref, dict) else {})
    scene["scene_ref"] = merged_ref
    source = deepcopy(scene.get("source")) if isinstance(scene.get("source"), dict) else {}
    source["scene_ref"] = merged_ref
    scene["source"] = source


def _merge_scene_ref(local_ref: dict, existing_ref: dict) -> dict:
    merged = deepcopy(local_ref)
    for key, value in existing_ref.items():
        if value is None:
            continue
        if key == "metadata" and isinstance(value, dict) and isinstance(merged.get("metadata"), dict):
            merged["metadata"] = {**merged["metadata"], **deepcopy(value)}
        else:
            merged[key] = deepcopy(value)
    return merged


def _local_scene_ref(scene: dict) -> dict | None:
    if not isinstance(scene, dict):
        return None
    if isinstance(scene.get("scene_ref"), dict):
        return scene["scene_ref"]
    source = scene.get("source") if isinstance(scene.get("source"), dict) else None
    if isinstance(source, dict) and isinstance(source.get("scene_ref"), dict):
        return source["scene_ref"]
    return None


def _has_local_scene_ref(scene: dict) -> bool:
    scene_ref = _local_scene_ref(scene)
    return isinstance(scene_ref, dict) and scene_ref.get("source") == LOCAL_SCENE_SOURCE


def _enrich_local_asset(asset: dict) -> None:
    local_ref = resolve_local_asset_ref(asset.get("asset_ref"), asset_id=asset.get("asset_id"))
    if not local_ref:
        return
    existing_ref = asset.get("asset_ref") if isinstance(asset.get("asset_ref"), dict) else {}
    asset["asset_ref"] = _merge_asset_ref(local_ref, existing_ref)
    if not isinstance(asset.get("dimensions"), list) and isinstance(local_ref.get("dimensions"), list):
        asset["dimensions"] = deepcopy(local_ref["dimensions"])


def _merge_asset_ref(local_ref: dict, existing_ref: dict) -> dict:
    merged = deepcopy(local_ref)
    for key, value in existing_ref.items():
        if value is None:
            continue
        if key == "metadata" and isinstance(value, dict) and isinstance(merged.get("metadata"), dict):
            merged["metadata"] = {**merged["metadata"], **deepcopy(value)}
        else:
            merged[key] = deepcopy(value)
    return merged


def _has_local_asset_ref(asset: dict) -> bool:
    asset_ref = asset.get("asset_ref") if isinstance(asset, dict) else None
    return isinstance(asset_ref, dict) and asset_ref.get("source") == LOCAL_ASSET_SOURCE


def _geometry_from_legacy_object(obj: dict) -> dict | None:
    if not any(key in obj for key in ("center", "size", "yaw", "rotation", "quaternion")):
        return None
    yaw = obj.get("yaw")
    if yaw is None:
        yaw = _yaw_from_rotation(obj.get("rotation"))
    if yaw is None:
        yaw = _yaw_from_quaternion(obj.get("quaternion"))
    return {
        "center": deepcopy(obj.get("center")),
        "size": deepcopy(obj.get("size")),
        "yaw": deepcopy(0 if yaw is None else yaw),
    }


def _yaw_from_rotation(rotation: object) -> float | None:
    if not isinstance(rotation, list) or len(rotation) < 3:
        return None
    try:
        return float(rotation[2])
    except (TypeError, ValueError):
        return None


def _yaw_from_quaternion(quaternion: object) -> float | None:
    if not isinstance(quaternion, list) or len(quaternion) < 4:
        return None
    try:
        w, _x, _y, z = [float(value) for value in quaternion[:4]]
    except (TypeError, ValueError):
        return None
    return math.degrees(2.0 * math.atan2(z, w))


def _geometry_from_asset(asset: dict) -> dict | None:
    placement = asset.get("placement")
    dimensions = asset.get("dimensions")
    if isinstance(placement, dict) and isinstance(dimensions, list):
        position = placement.get("position") or placement.get("center")
        yaw = placement.get("yaw_degrees", placement.get("yaw", 0))
        if isinstance(position, list) and len(position) >= 3 and len(dimensions) >= 3:
            geometry = {"center": position[:3], "size": dimensions[:3], "yaw": yaw}
            for field in SUPPORT_FIELDS:
                if placement.get(field) is not None:
                    geometry[field] = deepcopy(placement[field])
            return geometry
    geometry = asset.get("geometry")
    if isinstance(geometry, dict):
        position = geometry.get("position") or geometry.get("center")
        size = geometry.get("dimensions") or geometry.get("size")
        yaw = geometry.get("yaw_degrees", geometry.get("yaw", 0))
        if isinstance(position, list) and isinstance(size, list) and len(position) >= 3 and len(size) >= 3:
            return {"center": position[:3], "size": size[:3], "yaw": yaw}
    bbox = asset.get("bbox")
    if isinstance(bbox, dict) and all(key in bbox for key in ("center", "size", "yaw")):
        return bbox
    if all(key in asset for key in ("center", "size", "yaw")):
        return {"center": asset.get("center"), "size": asset.get("size"), "yaw": asset.get("yaw")}
    return None


def _bbox_from_asset(asset: dict) -> dict | None:
    return _geometry_from_asset(asset)


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
        geometry = _geometry_from_asset(asset)
        if geometry is not None and isinstance(geometry.get("size"), list):
            obj["dimensions"] = deepcopy(geometry["size"])
            obj["bbox_size"] = deepcopy(geometry["size"])
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


def _non_geometric_asset_record(asset: dict, index: int) -> dict:
    return {
        "asset_index": index,
        "asset_id": _first_nonempty_str(asset.get("asset_id"), asset.get("object_id"), default=f"asset_{index + 1:03d}"),
        "object_id": asset.get("object_id"),
        "category": asset.get("category"),
        "reason": "asset has no complete placement/dimensions geometry",
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
