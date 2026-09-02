"""SceneWeaver final-iteration layout to canonical scene conversion."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.adapters.common.artifacts import latest_numbered_json
from benchmark.adapters.common.assets import (
    AssetProvider,
    asset_fields,
    resolve_asset_record,
)
from benchmark.adapters.common.geometry import (
    build_scene,
    canonical_room,
    category_from_identifier,
    finite_float,
    shift_boundary_to_origin,
    shift_center,
    vector3,
)
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


def convert_scene_weaver(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
) -> dict:
    layout_path = _layout_path(source_path, config)
    payload = read_json(layout_path)
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("SceneWeaver layout must be a JSON object")
    native_objects = payload.get("objects")
    if not isinstance(native_objects, Mapping):
        raise ArtifactValidationError("SceneWeaver layout.objects must be a JSON object")

    fallback_boundary, scene_height, _ = canonical_room(generation_input)
    roomsize = payload.get("roomsize")
    if isinstance(roomsize, Sequence) and not isinstance(roomsize, (str, bytes)) and len(roomsize) >= 2:
        width = finite_float(roomsize[0], "SceneWeaver roomsize[0]")
        depth = finite_float(roomsize[1], "SceneWeaver roomsize[1]")
        if width <= 0.0 or depth <= 0.0:
            raise ArtifactValidationError("SceneWeaver roomsize values must be positive")
        boundary = [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]]
        boundary_source = "sceneweaver_roomsize"
    else:
        boundary = fallback_boundary
        boundary_source = "generation_input"
    boundary, origin_shift = shift_boundary_to_origin(boundary)
    scene_height = finite_float(config.get("scene_height", scene_height), "SceneWeaver scene_height")
    rotation_unit = str(config.get("rotation_unit") or "radian").strip().lower()
    if rotation_unit not in {"radian", "radians", "degree", "degrees"}:
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
        center = [bottom_center[0], bottom_center[1], bottom_center[2] + size[2] / 2.0]
        rotation = vector3(
            native.get("rotation", [0.0, 0.0, 0.0]),
            f"SceneWeaver objects.{object_id}.rotation",
        )
        if rotation_unit.startswith("radian"):
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
        )
        if not record.get("asset_key"):
            record["asset_key"] = f"sceneweaver_proxy:{object_id}"
            record["source_db"] = "sceneweaver_layout"
        fields = asset_fields(
            object_id=object_id,
            target_size=size,
            record=record,
            fallback_category=category,
            fallback_description=description,
            config=config,
        )
        metadata = dict(fields["metadata"])
        metadata.update(
            {
                "native_object_index": index,
                "native_position_anchor": "bottom_center",
                "native_parent_relations": native.get("parent") or [],
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

    iteration_match = re.search(r"_([0-9]+)\.json$", layout_path.name)
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
            "position_semantics": "bbox_bottom_center",
            "origin_shift": origin_shift,
        },
        extra_metadata={
            "source_artifact": layout_path.as_posix(),
            "source_boundary": boundary_source,
            "selected_iteration": int(iteration_match.group(1)) if iteration_match else None,
            "native_structure": payload.get("structure") or {},
        },
    )


def _layout_path(source_path: Path, config: Mapping[str, Any]) -> Path:
    explicit = config.get("layout_path")
    if explicit:
        path = Path(str(explicit)).expanduser()
        if not path.is_absolute():
            base = source_path if source_path.is_dir() else source_path.parent
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(f"SceneWeaver layout_path not found: {path}")
        return path
    return latest_numbered_json(source_path, prefix="layout")


__all__ = ["convert_scene_weaver"]
