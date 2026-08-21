from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmark.architecture_policy import (
    require_generated_architecture_targets_active,
    validate_architecture_contract,
)
from benchmark.adapters.layout_json.prompt import (
    AXIS_CONVENTION,
    CENTER_ORIGIN,
    COORDINATE_UNIT,
    MIN_CORNER_ORIGIN,
    ROTATION_UNIT,
)
from benchmark.models.json_response import parse_json_object
from benchmark.resources import runtime_resource_path
from benchmark.scene_io.validate import ArtifactValidationError, validate_generated_scene
from benchmark.task_contract import architecture_contract_for_room
from benchmark.utils.io import load_json_schema


SCHEMA_PATH = runtime_resource_path("schemas/generator_layout_v1.schema.json")


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
    _validate_layout_semantics(layout)
    _validate_coordinate_frame_consistency(layout)
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
    request_id = str(generation_input.get("request_id") or request.get("request_id") or "").strip()
    if not request_id:
        raise ArtifactValidationError("generation_input must provide a non-empty request_id")
    room = request.get("room")
    if not isinstance(room, dict):
        raise ArtifactValidationError(
            "generation_input.scene_request.room must provide the benchmark architecture contract"
        )
    source_coordinate_frame = deepcopy(layout["coordinate_frame"])
    _validate_declared_room_matches_contract(layout.get("room"), room, source_coordinate_frame)
    xy_offset = _coordinate_offset(room, source_coordinate_frame)
    boundary = _room_boundary(room)
    scene_height = _room_height(room)
    selected_assets = _selected_asset_catalog(generation_input)
    selection_enforced = isinstance(generation_input.get("asset_selection"), dict)
    objects = []
    for index, raw_object in enumerate(layout["objects"]):
        selected_asset, binding = _resolve_explicit_asset(
            raw_object,
            index=index,
            selected_assets=selected_assets,
            selection_enforced=selection_enforced,
        )
        objects.append(_convert_object(raw_object, selected_asset, binding, xy_offset))
    bound_count = sum(1 for obj in objects if isinstance(obj.get("asset_ref"), dict))
    asset_binding_summary = {
        "selection_provided": selection_enforced,
        "strategy": "explicit_asset_id_exact_lookup" if selection_enforced else "none",
        "selected_asset_count": len(selected_assets),
        "generated_object_count": len(objects),
        "bound_object_count": bound_count,
        "unresolved_object_count": 0,
    }
    relations: list[dict] = []
    oar_relations: list[dict] = []
    for item in layout.get("relationships", []):
        converted = _convert_relationship(item)
        if converted.get("family") == "oar":
            oar_relations.append(converted)
        else:
            relations.append(converted)
    raw_architecture = (
        (generation_input.get("generation_contract") or {}).get(
            "architecture"
        )
    )
    architecture_contract = validate_architecture_contract(
        raw_architecture
        if raw_architecture is not None
        else architecture_contract_for_room(room)
    )
    try:
        require_generated_architecture_targets_active(
            oar_relations,
            architecture_contract,
        )
    except ValueError as exc:
        raise ArtifactValidationError(
            f"layout_json_v1 architecture-contract mismatch: {exc}"
        ) from exc
    scene = {
        "schema_version": "canonical_scene_v1",
        "scene_id": str(layout.get("scene_id") or f"generated_{request_id}"),
        "request_id": request_id,
        "scene_type": str(layout["scene_type"]),
        "boundary": boundary,
        "scene_height": scene_height,
        "objects": objects,
        "relations": relations,
        "oar_relations": oar_relations,
        "metadata": {
            "generator_output_schema": "layout_json_v1",
            "output_adapter": "layout_json",
            "asset_grounding": "explicit_asset_id" if selection_enforced else "none",
            "asset_binding": asset_binding_summary,
            "architecture_contract": deepcopy(architecture_contract),
            "source_coordinate_frame": source_coordinate_frame,
            "coordinate_frame": {
                "origin": MIN_CORNER_ORIGIN,
                "axes": AXIS_CONVENTION,
                "unit": COORDINATE_UNIT,
                "rotation_unit": ROTATION_UNIT,
            },
        },
    }
    validate_generated_scene(scene)
    return scene


def _convert_object(
    raw: dict,
    selected_asset: dict | None,
    binding: dict | None,
    xy_offset: tuple[float, float],
) -> dict:
    object_id = str(raw["id"])
    category = str(raw["category"])
    selected = deepcopy(selected_asset) if isinstance(selected_asset, dict) else None
    size = _vec3(raw["size"])
    center = _vec3(raw["center"])
    center[0] += xy_offset[0]
    center[1] += xy_offset[1]
    metadata: dict[str, Any] = {}
    if binding is not None:
        metadata["asset_binding"] = deepcopy(binding)
    converted = {
        "id": object_id,
        "category": category,
        "size": size,
        "center": center,
        "rotation": _vec3(raw["rotation"]),
        "geometry_provenance": _selected_geometry_provenance(selected),
        "metadata": metadata,
    }
    metadata["layout_transform_provenance"] = "explicit_center_size_rotation"
    if raw.get("description") is not None:
        converted["description"] = str(raw["description"])
    if raw.get("support_parent") is not None:
        converted["support_parent"] = str(raw["support_parent"])
    if selected is not None:
        converted["jid"] = str(selected["jid"])
        if isinstance(selected.get("asset_ref"), dict):
            converted["asset_ref"] = deepcopy(selected["asset_ref"])
        if isinstance(selected.get("asset_proxy"), dict):
            converted["asset_proxy"] = deepcopy(selected["asset_proxy"])
    elif raw.get("asset_id") is not None:
        converted["jid"] = str(raw["asset_id"])
        metadata["native_asset_id_unresolved"] = True
    return converted


def _selected_geometry_provenance(selected: dict | None) -> str:
    if not isinstance(selected, dict):
        return "bbox_proxy"
    asset_ref = selected.get("asset_ref") if isinstance(selected.get("asset_ref"), dict) else {}
    return "generated_mesh" if asset_ref.get("source_db") == "generated" else "asset_mesh"


def _convert_relationship(raw: dict) -> dict:
    target = str(raw["object"])
    converted = {
        "family": str(raw["family"]),
        "subject_id": str(raw["subject"]),
        "type": str(raw["predicate"]),
    }
    if converted["family"] == "oar":
        converted["architectural_element"] = target
    else:
        converted["object_id"] = target
    return converted


def _selected_asset_catalog(generation_input: dict) -> dict[str, dict[str, Any]]:
    selection = generation_input.get("asset_selection")
    if not isinstance(selection, dict):
        return {}
    catalog: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(selection.get("objects", [])):
        if not isinstance(item, dict) or not isinstance(item.get("selected_asset"), dict):
            continue
        selected = deepcopy(item["selected_asset"])
        jid = str(selected.get("jid") or "").strip()
        if not jid:
            raise ArtifactValidationError(
                f"generation_input.asset_selection.objects.{index}.selected_asset.jid must be non-empty"
            )
        existing = catalog.get(jid)
        if existing is not None and existing["selected_asset"] != selected:
            raise ArtifactValidationError(
                f"asset_selection contains conflicting metadata for selected asset_id {jid!r}"
            )
        entry = catalog.setdefault(
            jid,
            {
                "selected_asset": selected,
                "selection_object_ids": [],
            },
        )
        object_id = str(item.get("object_id") or "").strip()
        if object_id and object_id not in entry["selection_object_ids"]:
            entry["selection_object_ids"].append(object_id)
    return catalog


def _resolve_explicit_asset(
    raw: dict,
    *,
    index: int,
    selected_assets: dict[str, dict[str, Any]],
    selection_enforced: bool,
) -> tuple[dict | None, dict | None]:
    requested_asset_id = str(raw.get("asset_id") or "").strip()
    if not selection_enforced:
        return None, None
    if not requested_asset_id:
        raise ArtifactValidationError(
            f"layout_json_v1 validation failed at objects.{index}.asset_id: "
            "structured_assets mode requires an explicit selected asset_id"
        )
    entry = selected_assets.get(requested_asset_id)
    if entry is None:
        raise ArtifactValidationError(
            f"layout_json_v1 validation failed at objects.{index}.asset_id: "
            f"{requested_asset_id!r} is not in the request's selected asset catalog"
        )
    return deepcopy(entry["selected_asset"]), {
        "method": "explicit_asset_id_exact_lookup",
        "requested_asset_id": requested_asset_id,
        "selection_object_ids": deepcopy(entry["selection_object_ids"]),
    }


def _room_boundary(room: dict) -> list[list[float]]:
    boundary = room.get("boundary")
    if isinstance(boundary, list):
        return [[float(point[0]), float(point[1])] for point in boundary]
    size = room.get("size")
    width = float(size[0])
    depth = float(size[1])
    return [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]]


def _coordinate_offset(room: dict, coordinate_frame: dict) -> tuple[float, float]:
    if coordinate_frame["origin"] == MIN_CORNER_ORIGIN:
        return (0.0, 0.0)
    boundary = room.get("boundary")
    if isinstance(boundary, list):
        xs = [float(point[0]) for point in boundary]
        ys = [float(point[1]) for point in boundary]
        return ((max(xs) - min(xs)) / 2.0, (max(ys) - min(ys)) / 2.0)
    size = room["size"]
    return (float(size[0]) / 2.0, float(size[1]) / 2.0)


def _validate_declared_room_matches_contract(
    declared_room: Any,
    contract_room: dict,
    coordinate_frame: dict,
    *,
    tolerance: float = 1.0e-6,
) -> None:
    """Treat an echoed generator room as an assertion, never as scene geometry."""

    if declared_room is None:
        return
    if not isinstance(declared_room, dict):
        raise ArtifactValidationError("layout_json_v1 room must be a JSON object when provided")
    declared_width, declared_depth = _room_xy_size(declared_room)
    contract_width, contract_depth = _room_xy_size(contract_room)
    declared_height = _room_height(declared_room)
    contract_height = _room_height(contract_room)
    if any(
        abs(actual - expected) > tolerance
        for actual, expected in [
            (declared_width, contract_width),
            (declared_depth, contract_depth),
            (declared_height, contract_height),
        ]
    ):
        raise ArtifactValidationError(
            "layout_json_v1 room conflicts with the benchmark architecture contract; "
            "the generator may omit room but may not resize it"
        )
    if coordinate_frame.get("origin") not in {MIN_CORNER_ORIGIN, CENTER_ORIGIN}:
        raise ArtifactValidationError("layout_json_v1 coordinate_frame.origin is unsupported")


def _room_xy_size(room: dict) -> tuple[float, float]:
    boundary = room.get("boundary")
    if isinstance(boundary, list):
        xs = [float(point[0]) for point in boundary]
        ys = [float(point[1]) for point in boundary]
        return max(xs) - min(xs), max(ys) - min(ys)
    size = room.get("size")
    if isinstance(size, list) and len(size) >= 2:
        return float(size[0]), float(size[1])
    raise ArtifactValidationError("room must provide boundary or size")


def _validate_coordinate_frame_consistency(layout: dict) -> None:
    room = layout.get("room")
    if not isinstance(room, dict):
        return
    boundary = room.get("boundary")
    if not isinstance(boundary, list):
        return
    xs = [float(point[0]) for point in boundary]
    ys = [float(point[1]) for point in boundary]
    origin = layout["coordinate_frame"]["origin"]
    tolerance = 1.0e-6
    if origin == MIN_CORNER_ORIGIN and (abs(min(xs)) > tolerance or abs(min(ys)) > tolerance):
        raise ArtifactValidationError(
            "layout_json_v1 coordinate_frame is room_min_corner_floor, but room.boundary minima are not [0, 0]"
        )
    if origin == CENTER_ORIGIN and (
        abs(min(xs) + max(xs)) > tolerance or abs(min(ys) + max(ys)) > tolerance
    ):
        raise ArtifactValidationError(
            "layout_json_v1 coordinate_frame is room_center_floor, but room.boundary is not centered on [0, 0]"
        )


def _validate_layout_semantics(layout: dict) -> None:
    object_ids: set[str] = set()
    for index, obj in enumerate(layout["objects"]):
        object_id = str(obj["id"]).strip()
        if object_id in object_ids:
            raise ArtifactValidationError(
                f"layout_json_v1 validation failed at objects.{index}.id: duplicate object id {object_id!r}"
            )
        object_ids.add(object_id)
        for key in ["center", "size", "rotation"]:
            if key not in obj:
                continue
            for component_index, value in enumerate(obj[key]):
                _require_finite_layout_number(value, f"objects.{index}.{key}.{component_index}")
    room = layout.get("room")
    if isinstance(room, dict):
        for key in ["size", "boundary"]:
            value = room.get(key)
            if not isinstance(value, list):
                continue
            rows = value if key == "boundary" else [value]
            for row_index, row in enumerate(rows):
                for component_index, number in enumerate(row):
                    _require_finite_layout_number(number, f"room.{key}.{row_index}.{component_index}")
        if room.get("height") is not None:
            _require_finite_layout_number(room["height"], "room.height")
    for index, relation in enumerate(layout.get("relationships", [])):
        subject_id = str(relation["subject"])
        object_id = str(relation["object"])
        if subject_id not in object_ids:
            raise ArtifactValidationError(
                f"layout_json_v1 validation failed at relationships.{index}.subject: "
                f"unknown object id {subject_id!r}"
            )
        family = str(relation["family"])
        if family == "oor" and object_id not in object_ids:
            raise ArtifactValidationError(
                f"layout_json_v1 validation failed at relationships.{index}.object: "
                f"unknown object id {object_id!r} for an oor relation"
            )


def _require_finite_layout_number(value: Any, path: str) -> None:
    if isinstance(value, bool):
        raise ArtifactValidationError(f"layout_json_v1 validation failed at {path}: boolean is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"layout_json_v1 validation failed at {path}: value is not numeric") from exc
    if not math.isfinite(number):
        raise ArtifactValidationError(f"layout_json_v1 validation failed at {path}: value must be finite")


def _room_height(room: dict) -> float:
    if room.get("height") is not None:
        return float(room["height"])
    size = room.get("size")
    if isinstance(size, list) and len(size) >= 3:
        return float(size[2])
    raise ArtifactValidationError(
        "layout_json_v1 room must provide height or a three-component size"
    )


def _vec3(value: Any) -> list[float]:
    return [float(value[0]), float(value[1]), float(value[2])]
