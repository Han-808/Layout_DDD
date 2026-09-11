"""LayoutVLM optimized asset poses to canonical scene conversion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.adapters.common.artifacts import read_json_source
from benchmark.adapters.common.assets import (
    AssetProvider,
    asset_fields,
    record_bbox_size,
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


def convert_layout_vlm(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
) -> dict:
    payload, resolved_path = read_json_source(source_path, candidates=("layout.json",))
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("LayoutVLM layout output must be a JSON object")
    layout = payload.get("layout") if isinstance(payload.get("layout"), Mapping) else payload
    scene_config = _scene_config(payload, config, resolved_path)
    assets = scene_config.get("assets") if isinstance(scene_config.get("assets"), Mapping) else {}

    source_boundary, fallback_height, _ = canonical_room(generation_input)
    native_boundary = scene_config.get("boundary")
    native_boundary = native_boundary if isinstance(native_boundary, Mapping) else {}
    floor_vertices = native_boundary.get("floor_vertices")
    if isinstance(floor_vertices, Sequence) and not isinstance(floor_vertices, (str, bytes)):
        boundary = []
        for index, point in enumerate(floor_vertices):
            coords = vector3(point, f"LayoutVLM boundary.floor_vertices[{index}]")
            boundary.append([coords[0], coords[1]])
        source_boundary_kind = "layoutvlm_scene_config"
    else:
        boundary = source_boundary
        source_boundary_kind = "generation_input"
    boundary, origin_shift = shift_boundary_to_origin(boundary)
    scene_height = finite_float(
        native_boundary.get("wall_height", fallback_height),
        "LayoutVLM boundary.wall_height",
    )
    if scene_height <= 0.0:
        raise ArtifactValidationError("LayoutVLM wall height must be positive")

    objects: list[dict[str, Any]] = []
    default_source = str(config.get("asset_source_db") or "layoutvlm_objaverse")
    for index, (instance_id_value, placement_value) in enumerate(layout.items()):
        instance_id = str(instance_id_value)
        if not isinstance(placement_value, Mapping):
            raise ArtifactValidationError(f"LayoutVLM layout.{instance_id} must be an object")
        placement = dict(placement_value)
        native_asset = assets.get(instance_id)
        native_asset = dict(native_asset) if isinstance(native_asset, Mapping) else {}
        asset_key = str(
            native_asset.get("uid")
            or native_asset.get("asset_id")
            or _strip_instance_suffix(instance_id)
        )
        category = str(
            native_asset.get("category")
            or _nested(native_asset, "annotations", "category")
            or category_from_identifier(native_asset.get("asset_var_name") or instance_id)
        )
        description = str(
            native_asset.get("description")
            or _nested(native_asset, "annotations", "description")
            or category
        )
        record = resolve_asset_record(
            provider,
            asset_key=asset_key,
            source_db=default_source,
            category=category,
            description=description,
            size=None,
            hint=placement,
            native_record=native_asset,
        )
        size = record_bbox_size(record)
        if size is None:
            raise ArtifactValidationError(
                f"LayoutVLM asset {instance_id!r} requires assetMetadata.boundingBox or provider bbox_size"
            )
        center = vector3(placement.get("position"), f"LayoutVLM layout.{instance_id}.position")
        rotation = _rotation(placement.get("rotation"), f"LayoutVLM layout.{instance_id}.rotation")
        fields = asset_fields(
            object_id=instance_id,
            target_size=size,
            record=record,
            fallback_category=category,
            fallback_description=description,
            config=config,
        )
        metadata = dict(fields["metadata"])
        metadata.update(
            {
                "native_instance_id": instance_id,
                "native_asset_id": asset_key,
                "native_placement_index": index,
            }
        )
        objects.append(
            {
                "id": instance_id,
                **{key: value for key, value in fields.items() if key != "metadata"},
                "size": size,
                "center": shift_center(center, origin_shift),
                "rotation": rotation,
                "metadata": metadata,
            }
        )

    return build_scene(
        generation_input,
        adapter_name="layout_vlm",
        native_schema="layoutvlm_layout_v1",
        boundary=boundary,
        scene_height=scene_height,
        objects=objects,
        coordinate_conversion={
            "source": "layoutvlm",
            "source_axes": "x_width_y_depth_z_up",
            "source_unit": "meter",
            "rotation_unit": "degree",
            "position_semantics": "asset_instance_center",
            "origin_shift": origin_shift,
        },
        extra_metadata={
            "source_artifact": resolved_path.as_posix(),
            "source_boundary": source_boundary_kind,
        },
    )


def _scene_config(payload: Mapping[str, Any], config: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    value = payload.get("scene_config") or config.get("scene_config")
    if isinstance(value, Mapping):
        return dict(value)
    path_value = config.get("scene_config_path")
    if not path_value:
        return {}
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    loaded = read_json(path)
    if not isinstance(loaded, Mapping):
        raise ArtifactValidationError("LayoutVLM scene_config_path must contain a JSON object")
    return dict(loaded)


def _rotation(value: Any, path: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [0.0, 0.0, finite_float(value, path)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 1:
            return [0.0, 0.0, finite_float(value[0], f"{path}[0]")]
        return vector3(value, path)
    raise ArtifactValidationError(f"{path} must be an angle or Euler vector")


def _strip_instance_suffix(value: str) -> str:
    return re.sub(r"-[0-9]+$", "", value)


def _nested(mapping: Mapping[str, Any], outer: str, inner: str) -> Any:
    value = mapping.get(outer)
    return value.get(inner) if isinstance(value, Mapping) else None


__all__ = ["convert_layout_vlm"]
