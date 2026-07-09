from __future__ import annotations

from typing import Any


class ArtifactValidationError(ValueError):
    """Raised when a canonical harness artifact is malformed."""


def validate_scene_request(scene_request: dict) -> dict:
    _require_mapping(scene_request, "scene_request")
    _require_string(scene_request, "request_id", "scene_request")
    _require_string(scene_request, "instruction", "scene_request")
    room = _require_mapping(scene_request.get("room"), "scene_request.room")
    _require_boundary(room.get("boundary"), "scene_request.room.boundary")
    if "height" in room:
        _require_positive_number(room.get("height"), "scene_request.room.height")
    return scene_request


def validate_object_plan(object_plan: dict) -> dict:
    _require_mapping(object_plan, "object_plan")
    _require_string(object_plan, "request_id", "object_plan")
    objects = _require_list(object_plan.get("objects"), "object_plan.objects")
    for index, obj in enumerate(objects):
        path = f"object_plan.objects[{index}]"
        _require_mapping(obj, path)
        _require_string(obj, "id", path)
        _require_string(obj, "category", path)
        _require_string(obj, "description", path)
        _forbid_keys(obj, {"center", "position", "rotation", "target_pose", "pose", "jid", "asset_jid"}, path)
        if "estimated_size" in obj:
            _require_vector3(obj["estimated_size"], f"{path}.estimated_size", positive=True)
        placement = obj.get("placement_intent")
        if placement is not None:
            _require_mapping(placement, f"{path}.placement_intent")
            for key in ["absolute_relations", "relative_relations"]:
                if key in placement:
                    _require_list(placement[key], f"{path}.placement_intent.{key}")
    return object_plan


def validate_asset_selection(asset_selection: dict) -> dict:
    _require_mapping(asset_selection, "asset_selection")
    _require_string(asset_selection, "request_id", "asset_selection")
    objects = _require_list(asset_selection.get("objects"), "asset_selection.objects")
    for index, item in enumerate(objects):
        path = f"asset_selection.objects[{index}]"
        _require_mapping(item, path)
        _require_string(item, "object_id", path)
        _require_mapping(item.get("object_spec"), f"{path}.object_spec")
        selected = _require_mapping(item.get("selected_asset"), f"{path}.selected_asset")
        _require_string(selected, "jid", f"{path}.selected_asset")
        if "size" in selected:
            _require_vector3(selected["size"], f"{path}.selected_asset.size", positive=True)
        if "asset_proxy" in selected and isinstance(selected["asset_proxy"], dict) and "bbox_size" in selected["asset_proxy"]:
            _require_vector3(selected["asset_proxy"]["bbox_size"], f"{path}.selected_asset.asset_proxy.bbox_size", positive=True)
    return asset_selection


def validate_generation_input(generation_input: dict) -> dict:
    _require_mapping(generation_input, "generation_input")
    _require_string(generation_input, "request_id", "generation_input")
    validate_scene_request(_require_mapping(generation_input.get("scene_request"), "generation_input.scene_request"))
    validate_object_plan(_require_mapping(generation_input.get("object_plan"), "generation_input.object_plan"))
    validate_asset_selection(_require_mapping(generation_input.get("asset_selection"), "generation_input.asset_selection"))
    contract = _require_mapping(generation_input.get("generation_contract"), "generation_input.generation_contract")
    if contract.get("output_format") != "canonical_generated_scene_v1":
        raise ArtifactValidationError("generation_input.generation_contract.output_format must be canonical_generated_scene_v1")
    return generation_input


def validate_generated_scene(scene: dict) -> dict:
    _require_mapping(scene, "generated_scene")
    _require_string(scene, "scene_id", "generated_scene")
    _require_string(scene, "request_id", "generated_scene")
    boundary = scene.get("boundary")
    if boundary is None and isinstance(scene.get("room"), dict):
        boundary = scene["room"].get("boundary")
    _require_boundary(boundary, "generated_scene.boundary")
    if "scene_height" in scene:
        _require_positive_number(scene["scene_height"], "generated_scene.scene_height")
    objects = _require_list(scene.get("objects"), "generated_scene.objects")
    for index, obj in enumerate(objects):
        path = f"generated_scene.objects[{index}]"
        _require_mapping(obj, path)
        _require_string(obj, "id", path)
        _require_string(obj, "jid", path)
        _require_vector3(obj.get("center"), f"{path}.center")
        _require_vector3(obj.get("rotation"), f"{path}.rotation")
        size = obj.get("size")
        asset_proxy = obj.get("asset_proxy") if isinstance(obj.get("asset_proxy"), dict) else {}
        if size is None:
            size = asset_proxy.get("bbox_size")
        _require_vector3(size, f"{path}.size_or_asset_proxy.bbox_size", positive=True)
        _require_mapping(obj.get("asset_ref"), f"{path}.asset_ref")
    return scene


def _require_mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{path} must be a JSON object")
    return value


def _require_list(value: Any, path: str) -> list:
    if not isinstance(value, list):
        raise ArtifactValidationError(f"{path} must be a JSON list")
    return value


def _require_string(mapping: dict, key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{path}.{key} must be a non-empty string")
    return value


def _require_boundary(value: Any, path: str) -> None:
    points = _require_list(value, path)
    if len(points) < 3:
        raise ArtifactValidationError(f"{path} must contain at least three points")
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) < 2:
            raise ArtifactValidationError(f"{path}[{index}] must be [x, y]")
        _number(point[0], f"{path}[{index}][0]")
        _number(point[1], f"{path}[{index}][1]")


def _require_vector3(value: Any, path: str, *, positive: bool = False) -> None:
    if not isinstance(value, list) or len(value) < 3:
        raise ArtifactValidationError(f"{path} must be a 3-vector")
    for index in range(3):
        number = _number(value[index], f"{path}[{index}]")
        if positive and number <= 0:
            raise ArtifactValidationError(f"{path}[{index}] must be positive")


def _require_positive_number(value: Any, path: str) -> None:
    number = _number(value, path)
    if number <= 0:
        raise ArtifactValidationError(f"{path} must be positive")


def _number(value: Any, path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{path} must be numeric") from exc


def _forbid_keys(mapping: dict, keys: set[str], path: str) -> None:
    present = sorted(key for key in keys if key in mapping)
    if present:
        raise ArtifactValidationError(f"{path} must not contain pose/asset keys: {present}")
