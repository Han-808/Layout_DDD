"""SceneWeaver final-iteration layout to canonical scene conversion."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.adapters.common.assets import (
    AssetProvider,
    asset_fields,
    record_asset_local_bbox_size,
    record_bbox_center,
    resolve_asset_record,
)
from benchmark.adapters.common.geometry import (
    build_scene,
    canonical_room,
    category_from_identifier,
    euler_xyz_to_matrix,
    finite_float,
    matrix_vector,
    reject_unsupported_architecture,
    require_room_geometry_match,
    shift_boundary_to_origin,
    shift_center,
    vector3,
)
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


LAYOUT_PATTERN = re.compile(r"^layout_([0-9]+)\.json$")
ITERATION_SELECTION_POLICIES = {"explicit", "latest"}


def convert_scene_weaver(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
) -> dict:
    layout_path, iteration_selection = _select_layout(source_path, config)
    payload = read_json(layout_path)
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("SceneWeaver layout must be a JSON object")
    if payload.get("rooms") or payload.get("room_layouts"):
        raise ArtifactValidationError(
            "SceneWeaver adapter is single-room only and will not collapse multi-room output"
        )
    reject_unsupported_architecture(payload, path="SceneWeaver layout")
    unsupported_boundary_fields = sorted(
        field for field in ("boundary", "floor_polygon") if payload.get(field)
    )
    if unsupported_boundary_fields:
        raise ArtifactValidationError(
            "SceneWeaver layout contains unsupported native boundary fields "
            f"{unsupported_boundary_fields}; only rectangular roomsize is supported"
        )
    native_objects = payload.get("objects")
    if not isinstance(native_objects, Mapping):
        raise ArtifactValidationError("SceneWeaver layout.objects must be a JSON object")

    fallback_boundary, fallback_height, _ = canonical_room(generation_input)
    roomsize = payload.get("roomsize")
    if (
        isinstance(roomsize, Sequence)
        and not isinstance(roomsize, (str, bytes))
        and len(roomsize) == 2
    ):
        width = finite_float(roomsize[0], "SceneWeaver roomsize[0]")
        depth = finite_float(roomsize[1], "SceneWeaver roomsize[1]")
        if width <= 0.0 or depth <= 0.0:
            raise ArtifactValidationError("SceneWeaver roomsize values must be positive")
        boundary = [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]]
        boundary_source = "sceneweaver_roomsize"
    else:
        raise ArtifactValidationError(
            "SceneWeaver layout requires native roomsize [length, width]"
        )
    scene_height = finite_float(
        config.get("scene_height", fallback_height),
        "SceneWeaver scene_height",
    )
    if scene_height <= 0.0:
        raise ArtifactValidationError("SceneWeaver scene_height must be positive")
    require_room_geometry_match(
        boundary,
        scene_height,
        fallback_boundary,
        fallback_height,
        path="SceneWeaver native room",
    )
    boundary, origin_shift = shift_boundary_to_origin(boundary)
    rotation_unit = str(config.get("rotation_unit") or "radian").strip().lower()
    rotation_unit = {
        "radians": "radian",
        "degrees": "degree",
    }.get(rotation_unit, rotation_unit)
    if rotation_unit not in {"radian", "degree"}:
        raise ArtifactValidationError("SceneWeaver rotation_unit must be radian or degree")

    asset_bindings = config.get("asset_bindings")
    asset_bindings = asset_bindings if isinstance(asset_bindings, Mapping) else {}
    default_source = str(config.get("asset_source_db") or "sceneweaver")
    objects: list[dict[str, Any]] = []

    for index, (object_id_value, native_value) in enumerate(native_objects.items()):
        object_id = str(object_id_value)
        if not isinstance(native_value, Mapping):
            raise ArtifactValidationError(f"SceneWeaver objects.{object_id} must be an object")
        native = dict(native_value)
        size = vector3(native.get("size"), f"SceneWeaver objects.{object_id}.size", positive=True)
        bottom_center = vector3(
            native.get("location") or native.get("position"),
            f"SceneWeaver objects.{object_id}.location",
        )
        native_rotation = native.get("rotation")
        if not (
            isinstance(native_rotation, Sequence)
            and not isinstance(native_rotation, (str, bytes))
            and len(native_rotation) == 3
        ):
            raise ArtifactValidationError(
                f"SceneWeaver objects.{object_id}.rotation must be a 3-vector"
            )
        source_rotation = vector3(
            native_rotation,
            f"SceneWeaver objects.{object_id}.rotation",
        )
        rotation_matrix = euler_xyz_to_matrix(
            source_rotation,
            f"SceneWeaver objects.{object_id}.rotation",
            unit=rotation_unit,
        )
        center_offset = matrix_vector(
            rotation_matrix,
            [0.0, 0.0, size[2] / 2.0],
        )
        center = [
            bottom_center[axis] + center_offset[axis]
            for axis in range(3)
        ]
        rotation = list(source_rotation)
        if rotation_unit == "radian":
            rotation = [math.degrees(value) for value in rotation]
        binding = asset_bindings.get(object_id)
        binding = dict(binding) if isinstance(binding, Mapping) else {}
        asset_key_value = (
            native.get("asset_id")
            or native.get("jid")
            or binding.get("asset_key")
            or binding.get("asset_id")
        )
        asset_key = str(asset_key_value) if asset_key_value is not None else None
        category = str(
            native.get("category")
            or binding.get("category")
            or category_from_identifier(object_id)
        )
        description = str(
            native.get("description")
            or binding.get("description")
            or category
        )
        native_record = {**binding, **native}
        record = resolve_asset_record(
            provider,
            asset_key=asset_key,
            source_db=default_source,
            category=category,
            description=description,
            size=size,
            hint=native,
            native_record=native_record,
            resolution_policy=str(
                config.get("asset_resolution_policy") or "exact_only"
            ),
        )
        asset_local_size = record_asset_local_bbox_size(record)
        asset_local_center = record_bbox_center(record)
        geometry_audit: dict[str, Any] = {
            "evaluated_size_source": "native.size",
            "native_size": size,
            "native_size_axes": "x_width_y_depth_z_height",
            "native_size_semantics": "scaled_object_local_bbox_dimensions",
            "native_location_semantics": "world_bbox_bottom_center",
            "bottom_center_to_center_offset": center_offset,
        }
        if asset_local_size is not None:
            geometry_audit["asset_local_bbox_size"] = asset_local_size
            geometry_audit["asset_local_bbox_source"] = "asset_metadata_or_binding"
        if asset_local_center is not None:
            geometry_audit["asset_local_bbox_center"] = asset_local_center
        fields = asset_fields(
            object_id=object_id,
            target_size=size,
            record=record,
            fallback_category=category,
            fallback_description=description,
            config=config,
            geometry_provenance="bbox_proxy",
            evaluated_bbox_center_local=[0.0, 0.0, 0.0],
            geometry_audit=geometry_audit,
        )
        metadata = dict(fields["metadata"])
        metadata.setdefault("canonical_front", [1.0, 0.0, 0.0])
        metadata.setdefault(
            "canonical_front_source",
            "sceneweaver_local_x_front_contract",
        )
        metadata.update(
            {
                "native_object_index": index,
                "native_asset_id": asset_key,
                "native_position_anchor": "bottom_center",
                "native_parent_relations": native.get("parent") or [],
                "native_rotation": native_rotation,
            }
        )
        objects.append(
            {
                "id": object_id,
                **{key: value for key, value in fields.items() if key != "metadata"},
                "size": size,
                "center": shift_center(center, origin_shift),
                "rotation": rotation,
                "metadata": metadata,
            }
        )

    return build_scene(
        generation_input,
        adapter_name="scene_weaver",
        native_schema="sceneweaver_layout_v1",
        boundary=boundary,
        scene_height=scene_height,
        objects=objects,
        coordinate_conversion={
            "source": "sceneweaver_layout",
            "source_axes": "x_width_y_depth_z_up",
            "source_unit": "meter",
            "source_rotation_unit": rotation_unit,
            "source_rotation_encoding": "blender_euler_xyz",
            "source_rotation_composition": "Rz@Ry@Rx_active",
            "position_semantics": "bbox_bottom_center",
            "origin_shift": origin_shift,
        },
        extra_metadata={
            "source_artifact": layout_path.as_posix(),
            "source_boundary": boundary_source,
            **iteration_selection,
            "room_contract_match": "exact_modulo_origin_cycle_winding",
            "native_structure": payload.get("structure") or {},
        },
    )


def _select_layout(
    source_path: Path,
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source = Path(source_path)
    policy = str(
        config.get("iteration_selection_policy") or "explicit"
    ).strip()
    if policy not in ITERATION_SELECTION_POLICIES:
        raise ArtifactValidationError(
            "SceneWeaver iteration_selection_policy must be explicit or latest"
        )
    explicit_path = config.get("layout_path")
    selected_value = config.get("selected_iteration")
    if explicit_path is not None and selected_value is not None:
        raise ArtifactValidationError(
            "SceneWeaver layout_path and selected_iteration are mutually exclusive"
        )
    candidates = discover_layout_iterations(source)
    available_iterations = sorted(candidates)

    if source.is_file():
        if explicit_path is not None or selected_value is not None:
            raise ArtifactValidationError(
                "SceneWeaver source file cannot be combined with another layout selector"
            )
        path = source
        selection_policy = "source_file"
    elif explicit_path is not None:
        if policy != "explicit":
            raise ArtifactValidationError(
                "SceneWeaver layout_path requires explicit selection policy"
            )
        path = Path(str(explicit_path)).expanduser()
        if not path.is_absolute():
            path = source / path
        if not path.is_file():
            raise FileNotFoundError(f"SceneWeaver layout_path not found: {path}")
        selection_policy = "explicit_layout_path"
    elif selected_value is not None:
        if policy != "explicit":
            raise ArtifactValidationError(
                "SceneWeaver selected_iteration requires explicit selection policy"
            )
        selected_iteration = _iteration_number(selected_value)
        path = candidates.get(selected_iteration)
        if path is None:
            raise ArtifactValidationError(
                "SceneWeaver selected_iteration "
                f"{selected_iteration} is unavailable; available={available_iterations}"
            )
        selection_policy = "explicit_iteration"
    elif policy == "latest":
        if not candidates:
            raise ArtifactValidationError(
                "SceneWeaver output contains no layout_<iteration>.json"
            )
        path = candidates[max(candidates)]
        selection_policy = "latest_non_strict"
    else:
        raise ArtifactValidationError(
            "strict SceneWeaver conversion requires selected_iteration, "
            "layout_path, or a direct layout JSON file"
        )

    match = LAYOUT_PATTERN.match(path.name)
    selected_iteration = int(match.group(1)) if match else None
    return path, {
        "selected_iteration": selected_iteration,
        "available_iterations": available_iterations,
        "iteration_selection_policy": selection_policy,
    }


def discover_layout_iterations(source: Path) -> dict[int, Path]:
    """Return every native layout iteration without selecting or converting it."""

    if source.is_file():
        search_roots = [source.parent]
    elif source.is_dir():
        nested = source / "record_scene"
        search_roots = [nested] if nested.is_dir() else [source]
    else:
        raise FileNotFoundError(f"harness output does not exist: {source}")
    candidates: dict[int, Path] = {}
    for root in search_roots:
        for path in sorted(root.glob("layout_*.json")):
            match = LAYOUT_PATTERN.match(path.name)
            if match:
                candidates[int(match.group(1))] = path
    return candidates


def _iteration_number(value: Any) -> int:
    if isinstance(value, bool):
        raise ArtifactValidationError(
            "SceneWeaver selected_iteration must be a non-negative integer"
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "SceneWeaver selected_iteration must be a non-negative integer"
        ) from exc
    if number < 0 or str(number) != str(value).strip():
        raise ArtifactValidationError(
            "SceneWeaver selected_iteration must be a non-negative integer"
        )
    return number


__all__ = ["convert_scene_weaver", "discover_layout_iterations"]
