from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

from jsonschema import Draft202012Validator

from benchmark.adapters.catalog_placement.prompt import (
    CATALOG_PLACEMENT_VERSION,
    public_slot_ids_from_generation_input as _prompt_public_slot_ids,
)
from benchmark.assets.facing import benchmark_catalog_facing_contract
from benchmark.nl_scene.generation_input import STRUCTURED_ASSETS_INPUT_MODE
from benchmark.resources import runtime_resource_path
from benchmark.scene_io.validate import (
    ArtifactValidationError,
    validate_generated_scene,
    validate_generation_input,
)
from benchmark.utils.io import load_json_schema


SCHEMA_PATH = runtime_resource_path(
    "schemas/generator_catalog_placement_v1.schema.json"
)
COORDINATE_FRAME = {
    "origin": "room_min_corner_floor",
    "axes": "x_width_y_depth_z_up",
    "unit": "meter",
    "rotation_unit": "degree",
}


def public_slot_ids_from_generation_input(generation_input: dict) -> set[str]:
    """Return the exact slot vocabulary exposed by the structured generator input."""

    return set(_prompt_public_slot_ids(generation_input))


def selected_asset_ids_from_generation_input(
    generation_input: dict,
) -> set[str]:
    """Return the exact frozen asset IDs exposed to the generator."""

    return set(_selected_asset_catalog(generation_input))


def public_task_slots_from_generation_input(
    generation_input: dict,
) -> dict[str, dict[str, Any]]:
    """Return benchmark-owned intended semantics keyed by public slot ID."""

    object_plan = generation_input.get("object_plan")
    if not isinstance(object_plan, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(object_plan.get("objects", [])):
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("slot_id") or item.get("id") or "").strip()
        if not slot_id:
            continue
        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        raw_role = (
            item["role"]
            if "role" in item
            else metadata.get("intended_role")
        )
        intended_role = (
            str(raw_role).strip()
            if raw_role not in (None, "")
            else None
        )
        semantics: dict[str, Any] = {
            "slot_id": slot_id,
            "intended_category": str(item.get("category") or "").strip(),
            "intended_role": intended_role,
            "description": str(item.get("description") or "").strip(),
            "source": "public_object_plan",
        }
        existing = result.get(slot_id)
        if existing is not None and existing != semantics:
            raise ArtifactValidationError(
                "generation_input.object_plan contains conflicting intended "
                f"semantics for public slot {slot_id!r} at objects.{index}"
            )
        result[slot_id] = semantics
    return result


def validate_catalog_placement(
    placement: dict,
    *,
    public_slot_ids: Iterable[str] | None = None,
    require_slot_binding: bool = False,
) -> dict:
    """Validate the strict generator-owned placement contract."""

    if not isinstance(placement, dict):
        raise ArtifactValidationError(
            "catalog_placement_v1 output must be a JSON object"
        )
    errors = sorted(
        Draft202012Validator(load_json_schema(SCHEMA_PATH)).iter_errors(placement),
        key=lambda item: tuple(str(part) for part in item.path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ArtifactValidationError(
            f"catalog_placement_v1 validation failed at {path}: {error.message}"
        )

    allowed_slots = (
        {str(slot_id) for slot_id in public_slot_ids}
        if public_slot_ids is not None
        else None
    )
    instance_ids: set[str] = set()
    for index, instance in enumerate(placement["instances"]):
        instance_id = str(instance["instance_id"])
        if instance_id in instance_ids:
            raise ArtifactValidationError(
                "catalog_placement_v1 validation failed at "
                f"instances.{index}.instance_id: duplicate instance id {instance_id!r}"
            )
        instance_ids.add(instance_id)
        asset_id = str(instance["asset_id"])
        if not asset_id.strip():
            raise ArtifactValidationError(
                "catalog_placement_v1 validation failed at "
                f"instances.{index}.asset_id: value must not be blank"
            )
        if require_slot_binding and instance.get("slot_id") is None:
            raise ArtifactValidationError(
                "catalog_placement_v1 validation failed at "
                f"instances.{index}.slot_id: structured_assets requires a "
                "generator-visible public slot binding"
            )
        if "slot_id" in instance and not str(instance["slot_id"]).strip():
            raise ArtifactValidationError(
                "catalog_placement_v1 validation failed at "
                f"instances.{index}.slot_id: value must not be blank"
            )
        if (
            allowed_slots is not None
            and instance.get("slot_id") is not None
            and str(instance["slot_id"]) not in allowed_slots
        ):
            raise ArtifactValidationError(
                "catalog_placement_v1 validation failed at "
                f"instances.{index}.slot_id: {instance['slot_id']!r} is not a "
                "generator-visible public slot"
            )
        for field in ("center_m", "rotation_euler_xyz_deg"):
            _finite_vec3(instance[field], f"instances.{index}.{field}")
        uniform_scale = _finite_number(
            instance["uniform_scale"], f"instances.{index}.uniform_scale"
        )
        if uniform_scale <= 0.0:
            raise ArtifactValidationError(
                "catalog_placement_v1 validation failed at "
                f"instances.{index}.uniform_scale: value must be positive"
            )
    return placement


def extract_catalog_placement(
    payload: Any,
    *,
    public_slot_ids: Iterable[str] | None = None,
    require_slot_binding: bool = False,
) -> dict:
    """Extract a placement from raw JSON text or an OpenAI chat envelope."""

    if isinstance(payload, dict) and isinstance(payload.get("choices"), list):
        try:
            payload = payload["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ArtifactValidationError(
                "OpenAI-compatible response does not contain choices[0].message.content"
            ) from exc
    if isinstance(payload, dict):
        placement = payload
    elif isinstance(payload, str):
        try:
            placement = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(
                "catalog_placement_v1 generator output must be exactly one JSON "
                f"object: {exc.msg} at char {exc.pos}"
            ) from exc
    else:
        raise ArtifactValidationError(
            "catalog_placement_v1 generator output must be a JSON object"
        )
    return validate_catalog_placement(
        placement,
        public_slot_ids=public_slot_ids,
        require_slot_binding=require_slot_binding,
    )


def convert_catalog_placement_to_scene(
    placement: dict,
    generation_input: dict,
) -> dict:
    """Convert fixed-catalog identities and rigid poses into canonical_scene_v1."""

    public_slots = public_slot_ids_from_generation_input(generation_input)
    validate_generation_input(generation_input)
    contract = generation_input.get("generation_contract")
    structured_assets = (
        isinstance(contract, dict)
        and contract.get("input_mode") == STRUCTURED_ASSETS_INPUT_MODE
    )
    validate_catalog_placement(
        placement,
        public_slot_ids=public_slots,
        require_slot_binding=structured_assets,
    )
    task_slots = public_task_slots_from_generation_input(generation_input)
    request = generation_input["scene_request"]
    request_id = str(
        generation_input.get("request_id") or request.get("request_id") or ""
    ).strip()
    if not request_id:
        raise ArtifactValidationError(
            "generation_input must provide a non-empty request_id"
        )
    room = request.get("room")
    if not isinstance(room, dict):
        raise ArtifactValidationError(
            "generation_input.scene_request.room must provide the benchmark room"
        )
    selected_assets = _selected_asset_catalog(generation_input)
    objects: list[dict] = []
    registry_entries: list[dict] = []
    for index, instance in enumerate(
        sorted(placement["instances"], key=lambda item: str(item["instance_id"]))
    ):
        slot_id = instance.get("slot_id")
        if slot_id is not None and str(slot_id) not in public_slots:
            raise ArtifactValidationError(
                "catalog_placement_v1 validation failed at "
                f"instances.{index}.slot_id: {slot_id!r} is not a generator-visible public slot"
            )
        asset_id = str(instance["asset_id"])
        catalog_entry = selected_assets.get(asset_id)
        if catalog_entry is None:
            raise ArtifactValidationError(
                "catalog_placement_v1 validation failed at "
                f"instances.{index}.asset_id: {asset_id!r} is not in the request's "
                "selected frozen-asset catalog"
            )
        evaluator_object_id = _evaluator_object_id(str(instance["instance_id"]))
        converted, registry_entry = _convert_instance(
            instance,
            selected_asset=catalog_entry["selected_asset"],
            selection_object_ids=catalog_entry["selection_object_ids"],
            evaluator_object_id=evaluator_object_id,
            task_slot=task_slots.get(str(slot_id)) if slot_id is not None else None,
        )
        objects.append(converted)
        registry_entries.append(registry_entry)

    registry = {
        "schema_version": "catalog_instance_registry_v1",
        "request_id": request_id,
        "generator_output_schema": CATALOG_PLACEMENT_VERSION,
        "catalog_facing_contract": benchmark_catalog_facing_contract(),
        "instances": registry_entries,
    }
    scene = {
        "schema_version": "canonical_scene_v1",
        "scene_id": f"generated_{request_id}",
        "request_id": request_id,
        "scene_type": str(request.get("scene_type") or "room"),
        "boundary": _room_boundary(room),
        "scene_height": _room_height(room),
        "objects": objects,
        "relations": [],
        "oar_relations": [],
        "metadata": {
            "generator_output_schema": CATALOG_PLACEMENT_VERSION,
            "output_adapter": "catalog_placement",
            "asset_grounding": "selected_frozen_catalog_exact_asset_id",
            "catalog_facing_contract": benchmark_catalog_facing_contract(),
            "coordinate_frame": deepcopy(COORDINATE_FRAME),
            "instance_registry": deepcopy(registry),
            "transform_semantics": {
                "center": "scaled_catalog_canonical_local_bbox_center_world_m",
                "requested_scale": "generator_uniform_scale",
                "effective_scale": "exactly_equal_to_requested_uniform_scale",
                "rotation": "intrinsic_xyz_degrees_column_vector_rz_ry_rx",
            },
        },
    }
    validate_generated_scene(scene)
    return scene


def build_catalog_instance_registry(
    placement: dict,
    generation_input: dict,
) -> dict:
    """Return the same stable identity registry embedded in the canonical scene."""

    scene = convert_catalog_placement_to_scene(placement, generation_input)
    return deepcopy(scene["metadata"]["instance_registry"])


def rotation_matrix_rz_ry_rx(
    rotation_euler_xyz_deg: Sequence[float],
) -> list[list[float]]:
    """Intrinsic XYZ Euler matrix for column vectors: Rz @ Ry @ Rx."""

    rx, ry, rz = (
        math.radians(float(rotation_euler_xyz_deg[index])) for index in range(3)
    )
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    matrix = [
        [
            cz * cy,
            cz * sy * sx - sz * cx,
            cz * sy * cx + sz * sx,
        ],
        [
            sz * cy,
            sz * sy * sx + cz * cx,
            sz * sy * cx - cz * sx,
        ],
        [-sy, cy * sx, cy * cx],
    ]
    return [[_clean_float(value) for value in row] for row in matrix]


def world_bounds_for_local_bbox(
    center_m: Sequence[float],
    local_bbox_size_m: Sequence[float],
    rotation_euler_xyz_deg: Sequence[float],
) -> dict:
    """Compute exact OBB corners and its enclosing AABB."""

    center = _finite_vec3(center_m, "center_m")
    size = _positive_vec3(local_bbox_size_m, "local_bbox_size_m")
    rotation = _finite_vec3(rotation_euler_xyz_deg, "rotation_euler_xyz_deg")
    matrix = rotation_matrix_rz_ry_rx(rotation)
    half = [value * 0.5 for value in size]
    corners: list[list[float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = [sx * half[0], sy * half[1], sz * half[2]]
                rotated = _matvec3(matrix, local)
                corners.append(
                    [
                        _clean_float(center[axis] + rotated[axis])
                        for axis in range(3)
                    ]
                )
    minimum = [
        _clean_float(min(corner[axis] for corner in corners)) for axis in range(3)
    ]
    maximum = [
        _clean_float(max(corner[axis] for corner in corners)) for axis in range(3)
    ]
    aabb_size = [
        _clean_float(maximum[axis] - minimum[axis]) for axis in range(3)
    ]
    axes = [
        [matrix[row][column] for row in range(3)] for column in range(3)
    ]
    return {
        "world_obb": {
            "center_m": center,
            "size_m": size,
            "half_size_m": half,
            "axes": axes,
            "rotation_matrix_rz_ry_rx": matrix,
            "corners_m": corners,
        },
        "world_aabb": {
            "min_m": minimum,
            "max_m": maximum,
            "size_m": aabb_size,
        },
    }


def _convert_instance(
    instance: dict,
    *,
    selected_asset: dict,
    selection_object_ids: list[str],
    evaluator_object_id: str,
    task_slot: dict[str, Any] | None,
) -> tuple[dict, dict]:
    asset_id = str(instance["asset_id"])
    category = _selected_asset_string(selected_asset, "category", asset_id)
    description = _selected_asset_description(selected_asset, asset_id)
    asset_ref = _selected_asset_mapping(selected_asset, "asset_ref", asset_id)
    asset_proxy = _selected_asset_mapping(selected_asset, "asset_proxy", asset_id)
    source_bbox_center = _finite_vec3(
        asset_proxy.get("bbox_center_local"),
        f"selected_asset[{asset_id!r}].asset_proxy.bbox_center_local",
    )
    source_bbox_size = _positive_vec3(
        asset_proxy.get("bbox_size"),
        f"selected_asset[{asset_id!r}].asset_proxy.bbox_size",
    )
    if not str(asset_proxy.get("type") or "").strip():
        raise ArtifactValidationError(
            f"selected_asset[{asset_id!r}].asset_proxy.type must be non-empty"
        )
    if not str(asset_ref.get("source_db") or "").strip():
        raise ArtifactValidationError(
            f"selected_asset[{asset_id!r}].asset_ref.source_db must be non-empty"
        )
    if not str(asset_ref.get("asset_key") or "").strip():
        raise ArtifactValidationError(
            f"selected_asset[{asset_id!r}].asset_ref.asset_key must be non-empty"
        )

    center = _finite_vec3(instance["center_m"], "center_m")
    requested_uniform_scale = _finite_number(
        instance["uniform_scale"], "uniform_scale"
    )
    if requested_uniform_scale <= 0.0:
        raise ArtifactValidationError("uniform_scale must be finite and positive")
    rotation = _finite_vec3(
        instance["rotation_euler_xyz_deg"], "rotation_euler_xyz_deg"
    )
    effective_uniform_scale = requested_uniform_scale
    actual_local_size = [
        _clean_float(source_bbox_size[axis] * effective_uniform_scale)
        for axis in range(3)
    ]
    scaled_source_center = [
        _clean_float(source_bbox_center[axis] * effective_uniform_scale)
        for axis in range(3)
    ]
    rotation_matrix = rotation_matrix_rz_ry_rx(rotation)
    rotated_source_center = _matvec3(rotation_matrix, scaled_source_center)
    root_translation = [
        _clean_float(center[axis] - rotated_source_center[axis])
        for axis in range(3)
    ]
    bounds = world_bounds_for_local_bbox(center, actual_local_size, rotation)
    instance_id = str(instance["instance_id"])
    slot_id = str(instance["slot_id"]) if instance.get("slot_id") is not None else None
    placement_metadata = {
        "contract": CATALOG_PLACEMENT_VERSION,
        "instance_id": instance_id,
        "evaluator_object_id": evaluator_object_id,
        "asset_id": asset_id,
        "slot_id": slot_id,
        "selection_object_ids": sorted(selection_object_ids),
        "center_m": center,
        "rotation_euler_xyz_deg": rotation,
        "catalog_bbox_center_local_m": source_bbox_center,
        "catalog_bbox_size_m": source_bbox_size,
        "requested_uniform_scale": float(requested_uniform_scale),
        "effective_uniform_scale": float(effective_uniform_scale),
        "actual_local_bbox_size_m": actual_local_size,
        "scaled_catalog_bbox_center_local_m": scaled_source_center,
        "asset_root_translation_m": root_translation,
        **bounds,
    }
    selected_metadata = (
        deepcopy(selected_asset.get("metadata"))
        if isinstance(selected_asset.get("metadata"), dict)
        else {}
    )
    obj = {
        "id": evaluator_object_id,
        "jid": asset_id,
        "category": category,
        "description": description,
        "size": actual_local_size,
        "center": center,
        "rotation": rotation,
        "geometry_provenance": "asset_mesh",
        "asset_ref": deepcopy(asset_ref),
        "asset_proxy": deepcopy(asset_proxy),
        "metadata": {
            "catalog_placement": placement_metadata,
            "asset_metadata": selected_metadata,
        },
    }
    if task_slot is not None:
        obj["metadata"]["task_slot"] = deepcopy(task_slot)
    for key in ("retrieval_category", "desc", "short_desc"):
        value = selected_asset.get(key)
        if value is not None:
            obj[key] = str(value)
    registry_entry = {
        "instance_id": instance_id,
        "evaluator_object_id": evaluator_object_id,
        "asset_id": asset_id,
        "slot_id": slot_id,
        "task_slot": deepcopy(task_slot),
        "center_m": center,
        "rotation_euler_xyz_deg": rotation,
        "requested_uniform_scale": float(requested_uniform_scale),
        "effective_uniform_scale": float(effective_uniform_scale),
        "actual_local_bbox_size_m": actual_local_size,
        "world_obb": deepcopy(bounds["world_obb"]),
        "world_aabb": deepcopy(bounds["world_aabb"]),
    }
    return obj, registry_entry


def _selected_asset_catalog(generation_input: dict) -> dict[str, dict[str, Any]]:
    selection = generation_input.get("asset_selection")
    if not isinstance(selection, dict):
        raise ArtifactValidationError(
            "catalog_placement requires generation_input.asset_selection"
        )
    catalog: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(selection.get("objects", [])):
        if not isinstance(item, dict) or not isinstance(
            item.get("selected_asset"), dict
        ):
            raise ArtifactValidationError(
                f"generation_input.asset_selection.objects.{index}.selected_asset "
                "must be a JSON object"
            )
        selected = deepcopy(item["selected_asset"])
        asset_id = str(selected.get("jid") or "").strip()
        if not asset_id:
            raise ArtifactValidationError(
                f"generation_input.asset_selection.objects.{index}.selected_asset.jid "
                "must be non-empty"
            )
        existing = catalog.get(asset_id)
        if existing is not None and existing["selected_asset"] != selected:
            raise ArtifactValidationError(
                "asset_selection contains conflicting frozen metadata for selected "
                f"asset_id {asset_id!r}"
            )
        entry = catalog.setdefault(
            asset_id,
            {"selected_asset": selected, "selection_object_ids": []},
        )
        object_id = str(item.get("object_id") or "").strip()
        if object_id and object_id not in entry["selection_object_ids"]:
            entry["selection_object_ids"].append(object_id)
    return catalog


def _selected_asset_string(
    selected_asset: dict, key: str, asset_id: str
) -> str:
    value = str(selected_asset.get(key) or "").strip()
    if not value:
        raise ArtifactValidationError(
            f"selected_asset[{asset_id!r}].{key} must be non-empty"
        )
    return value


def _selected_asset_description(selected_asset: dict, asset_id: str) -> str:
    for key in ("desc", "short_desc", "description"):
        value = str(selected_asset.get(key) or "").strip()
        if value:
            return value
    raise ArtifactValidationError(
        f"selected_asset[{asset_id!r}] must provide desc, short_desc, or description"
    )


def _selected_asset_mapping(
    selected_asset: dict, key: str, asset_id: str
) -> dict:
    value = selected_asset.get(key)
    if not isinstance(value, dict):
        raise ArtifactValidationError(
            f"selected_asset[{asset_id!r}].{key} must be a JSON object"
        )
    return value


def _evaluator_object_id(instance_id: str) -> str:
    # Identity intentionally depends on no array index, slot, asset, task category,
    # or evaluator mapping heuristic.
    return instance_id


def _room_boundary(room: dict) -> list[list[float]]:
    boundary = room.get("boundary")
    if not isinstance(boundary, list):
        raise ArtifactValidationError(
            "generation_input.scene_request.room.boundary must be provided"
        )
    return [
        [_finite_number(point[0], "room.boundary.x"), _finite_number(point[1], "room.boundary.y")]
        for point in boundary
    ]


def _room_height(room: dict) -> float:
    if room.get("height") is not None:
        value = _finite_number(room["height"], "room.height")
    else:
        size = room.get("size")
        if not isinstance(size, list) or len(size) < 3:
            raise ArtifactValidationError(
                "generation_input.scene_request.room must provide height"
            )
        value = _finite_number(size[2], "room.size.2")
    if value <= 0.0:
        raise ArtifactValidationError("room height must be positive")
    return value


def _finite_vec3(value: Any, path: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ArtifactValidationError(f"{path} must contain exactly three numbers")
    return [
        _finite_number(component, f"{path}.{index}")
        for index, component in enumerate(value)
    ]


def _positive_vec3(value: Any, path: str) -> list[float]:
    result = _finite_vec3(value, path)
    for index, number in enumerate(result):
        if number <= 0.0:
            raise ArtifactValidationError(f"{path}.{index} must be positive")
    return result


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ArtifactValidationError(f"{path} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{path} must be numeric") from exc
    if not math.isfinite(number):
        raise ArtifactValidationError(f"{path} must be finite")
    return number


def _matvec3(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    return [
        _clean_float(
            sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
        )
        for row in range(3)
    ]


def _clean_float(value: float, *, tolerance: float = 1.0e-15) -> float:
    number = float(value)
    return 0.0 if abs(number) < tolerance else number
