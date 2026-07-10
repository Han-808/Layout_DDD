from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmark.models.base_model import parse_json_object
from benchmark.scene_io.validate import ArtifactValidationError, validate_generated_scene
from benchmark.utils.io import load_json_schema


SCHEMA_PATH = Path(__file__).resolve().parents[4] / "schemas" / "generator_layout_v1.schema.json"


def validate_layout_json(layout: dict) -> dict:
    if not isinstance(layout, dict):
        raise ArtifactValidationError("layout_json_v1 output must be a JSON object")
    errors = sorted(
        Draft202012Validator(load_json_schema(SCHEMA_PATH)).iter_errors(layout),
        key=lambda item: tuple(str(part) for part in item.path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ArtifactValidationError(f"layout_json_v1 validation failed at {path}: {error.message}")
    return layout


def extract_layout_json(payload: Any) -> dict:
    """Extract a layout object from raw JSON, text, or an OpenAI chat envelope."""

    if isinstance(payload, dict) and isinstance(payload.get("choices"), list):
        try:
            payload = payload["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ArtifactValidationError("OpenAI-compatible response does not contain choices[0].message.content") from exc
    layout = parse_json_object(payload)
    if isinstance(layout.get("layout"), dict):
        layout = layout["layout"]
    return validate_layout_json(layout)


def convert_layout_json_to_scene(layout: dict, generation_input: dict) -> dict:
    """Convert one generator-native layout schema into evaluator canonical JSON."""

    validate_layout_json(layout)
    request = generation_input.get("scene_request") if isinstance(generation_input.get("scene_request"), dict) else {}
    request_id = str(generation_input.get("request_id") or request.get("request_id") or "request_001")
    room = layout["room"]
    boundary = _room_boundary(room)
    scene_height = _room_height(room, request)
    selected_assets = _selected_assets_by_object_id(generation_input)
    objects = [
        _convert_object(raw_object, selected_assets.get(str(raw_object.get("id"))))
        for raw_object in layout["objects"]
    ]
    relations = [_convert_relationship(item) for item in layout.get("relationships", [])]
    scene = {
        "scene_id": str(layout.get("scene_id") or f"generated_{request_id}"),
        "request_id": request_id,
        "scene_type": str(layout.get("scene_type") or request.get("scene_type") or "room"),
        "boundary": boundary,
        "scene_height": scene_height,
        "objects": objects,
        "relations": relations,
        "metadata": {
            "generator_output_schema": "layout_json_v1",
            "output_adapter": "layout_json",
            "asset_grounding": "selected_or_unresolved_proxy",
        },
    }
    validate_generated_scene(scene)
    return scene


def _convert_object(raw: dict, selected_asset: dict | None) -> dict:
    object_id = str(raw["id"])
    description = str(raw.get("description") or raw.get("category") or "object")
    category = str(raw.get("category") or "object")
    selected = deepcopy(selected_asset) if isinstance(selected_asset, dict) else {}
    requested_asset_id = raw.get("asset_id")
    selected_jid = selected.get("jid")
    if requested_asset_id and selected_jid and str(requested_asset_id) != str(selected_jid):
        selected = {}
        selected_jid = None
    jid = str(selected_jid or requested_asset_id or f"layout_json_proxy:{object_id}")
    source_db = "layout_json_proxy" if not selected else str((selected.get("asset_ref") or {}).get("source_db") or "imaginarium")
    asset_ref = deepcopy(selected.get("asset_ref")) if isinstance(selected.get("asset_ref"), dict) else {}
    asset_ref.setdefault("source_db", source_db)
    asset_ref.setdefault("asset_key", jid)
    asset_ref.setdefault("mesh_uri", None)
    asset_ref.setdefault("pointcloud_uri", None)
    asset_ref.setdefault("metadata_uri", None)
    size = _vec3(raw["size"])
    metadata = deepcopy(selected.get("metadata")) if isinstance(selected.get("metadata"), dict) else {}
    metadata.setdefault("interactive", False)
    metadata["asset_resolution"] = "selected" if selected else "unresolved"
    metadata["generator_description"] = description
    if raw.get("support_parent") is not None:
        metadata["support_parent"] = str(raw["support_parent"])
    return {
        "id": object_id,
        "jid": jid,
        "category": category,
        "retrieval_category": str(selected.get("retrieval_category") or category),
        "desc": description,
        "short_desc": str(selected.get("short_desc") or description),
        "size": size,
        "center": _vec3(raw["center"]),
        "rotation": _vec3(raw.get("rotation") or [0, 0, 0]),
        "asset_ref": asset_ref,
        "asset_proxy": {
            "type": "obb_from_generator_layout",
            "bbox_center_local": [0.0, 0.0, 0.0],
            "bbox_size": size,
        },
        "metadata": metadata,
    }


def _convert_relationship(raw: dict) -> dict:
    return {
        "subject_id": str(raw["subject"]),
        "type": str(raw["predicate"]),
        "object_id": str(raw["object"]),
    }


def _room_boundary(room: dict) -> list[list[float]]:
    boundary = room.get("boundary")
    if isinstance(boundary, list):
        return [[float(point[0]), float(point[1])] for point in boundary]
    size = room.get("size")
    width = float(size[0])
    depth = float(size[1])
    return [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]]


def _room_height(room: dict, request: dict) -> float:
    if room.get("height") is not None:
        return float(room["height"])
    size = room.get("size")
    if isinstance(size, list) and len(size) >= 3:
        return float(size[2])
    request_room = request.get("room") if isinstance(request.get("room"), dict) else {}
    return float(request_room.get("height") or 2.8)


def _selected_assets_by_object_id(generation_input: dict) -> dict[str, dict]:
    selection = generation_input.get("asset_selection") if isinstance(generation_input.get("asset_selection"), dict) else {}
    result: dict[str, dict] = {}
    for item in selection.get("objects", []) if isinstance(selection.get("objects"), list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("selected_asset"), dict):
            continue
        result[str(item.get("object_id") or "")] = item["selected_asset"]
    return result


def _vec3(value: Any) -> list[float]:
    return [float(value[0]), float(value[1]), float(value[2])]
