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
    record_asset_local_bbox_size,
    record_bbox_center,
    record_bbox_size,
    resolve_asset_record,
)
from benchmark.adapters.common.geometry import (
    build_scene,
    canonical_room,
    category_from_identifier,
    compose_front_basis_rotation,
    finite_float,
    reject_unsupported_architecture,
    require_boundary_model,
    require_room_geometry_match,
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
    wrapped_layout = isinstance(payload.get("layout"), Mapping)
    layout = payload["layout"] if wrapped_layout else payload
    scene_config = _scene_config(payload, config, resolved_path)
    if (
        (
            wrapped_layout
            and (payload.get("rooms") or payload.get("room_layouts"))
        )
        or scene_config.get("rooms")
        or scene_config.get("room_layouts")
    ):
        raise ArtifactValidationError(
            "LayoutVLM adapter is single-room only and will not collapse multi-room output"
        )
    if wrapped_layout:
        reject_unsupported_architecture(payload, path="LayoutVLM output")
    reject_unsupported_architecture(scene_config, path="LayoutVLM scene_config")
    assets = scene_config.get("assets") if isinstance(scene_config.get("assets"), Mapping) else {}

    source_boundary, fallback_height, _ = canonical_room(generation_input)
    native_boundary = scene_config.get("boundary")
    native_boundary = native_boundary if isinstance(native_boundary, Mapping) else {}
    reject_unsupported_architecture(
        native_boundary,
        path="LayoutVLM scene_config.boundary",
    )
    floor_vertices = native_boundary.get("floor_vertices")
    if isinstance(floor_vertices, Sequence) and not isinstance(floor_vertices, (str, bytes)):
        boundary = []
        floor_z_values = []
        for index, point in enumerate(floor_vertices):
            coords = vector3(point, f"LayoutVLM boundary.floor_vertices[{index}]")
            boundary.append([coords[0], coords[1]])
            floor_z_values.append(coords[2])
        floor_z = floor_z_values[0]
        if any(abs(value - floor_z) > 1.0e-6 for value in floor_z_values[1:]):
            raise ArtifactValidationError(
                "LayoutVLM boundary.floor_vertices must be planar"
            )
        source_boundary_kind = "layoutvlm_scene_config"
    else:
        boundary = source_boundary
        floor_z = 0.0
        source_boundary_kind = "generation_input"
    require_boundary_model(
        boundary,
        supported=("axis_aligned_rectangle",),
        path="LayoutVLM boundary.floor_vertices",
    )
    scene_height = finite_float(
        native_boundary.get("wall_height", fallback_height),
        "LayoutVLM boundary.wall_height",
    )
    if scene_height <= 0.0:
        raise ArtifactValidationError("LayoutVLM wall height must be positive")
    require_room_geometry_match(
        boundary,
        scene_height,
        source_boundary,
        fallback_height,
        path="LayoutVLM native room",
    )
    boundary, origin_shift = shift_boundary_to_origin(boundary)
    origin_shift[2] = -floor_z

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
            resolution_policy=str(
                config.get("asset_resolution_policy") or "exact_only"
            ),
        )
        native_asset_local_size, native_bbox_axis_transform = (
            _native_asset_local_bbox_size(native_asset)
        )
        asset_local_size = (
            native_asset_local_size
            or record_asset_local_bbox_size(record)
            or record_bbox_size(record)
        )
        if asset_local_size is None:
            raise ArtifactValidationError(
                f"LayoutVLM asset {instance_id!r} requires "
                "assetMetadata.boundingBox or provider bbox_size"
            )
        native_scale_value = placement.get("scale")
        applied_scale = _scale3(
            native_scale_value,
            f"LayoutVLM layout.{instance_id}.scale",
        )
        size = [
            asset_local_size[axis] * applied_scale[axis]
            for axis in range(3)
        ]
        center = vector3(placement.get("position"), f"LayoutVLM layout.{instance_id}.position")
        source_rotation = _rotation(
            placement.get("rotation"),
            f"LayoutVLM layout.{instance_id}.rotation",
        )
        rotation = list(source_rotation)
        front_basis_yaw = None
        canonical_front = record.get("canonical_front")
        if canonical_front is not None:
            rotation, front_basis_yaw = compose_front_basis_rotation(
                source_rotation,
                canonical_front=canonical_front,
                native_zero_front=[1.0, 0.0, 0.0],
                path=f"LayoutVLM layout.{instance_id}.front_basis",
            )
        asset_local_center = record_bbox_center(record)
        geometry_audit: dict[str, Any] = {
            "evaluated_size_source": (
                "asset_local_bbox_times_native_scale"
                if native_scale_value is not None
                else "asset_local_bbox"
            ),
            "native_size_semantics": "asset_local_bbox",
            "asset_local_bbox_size": asset_local_size,
            "asset_local_bbox_source": (
                "layoutvlm_scene_config"
                if native_asset_local_size is not None
                else "exact_asset_metadata"
            ),
            "asset_local_bbox_axes": "processed_asset_xyz",
            "applied_scale": applied_scale,
            "scale_source": (
                "native.scale"
                if native_scale_value is not None
                else "implicit_identity"
            ),
        }
        if native_bbox_axis_transform is not None:
            geometry_audit["native_bbox_axis_transform"] = native_bbox_axis_transform
            geometry_audit["asset_local_bbox_axes"] = "canonical_mesh_xyz"
        if native_scale_value is not None:
            geometry_audit["native_scale"] = applied_scale
        if asset_local_center is not None:
            geometry_audit["asset_local_bbox_center"] = asset_local_center
        if front_basis_yaw is not None:
            geometry_audit.update(
                {
                    "native_zero_front": [1.0, 0.0, 0.0],
                    "canonical_front_basis_yaw_degrees": front_basis_yaw,
                    "orientation_basis_transform": (
                        "canonical_asset_front_to_layoutvlm_positive_x"
                    ),
                }
            )
        fields = asset_fields(
            object_id=instance_id,
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
            "layoutvlm_processed_asset_contract",
        )
        metadata.update(
            {
                "native_instance_id": instance_id,
                "native_asset_id": asset_key,
                "native_placement_index": index,
                "native_rotation_degrees": source_rotation,
                "native_zero_front": [1.0, 0.0, 0.0],
            }
        )
        if native_asset.get("frontView") is not None:
            metadata["native_front_view"] = native_asset["frontView"]
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
            "source_rotation_encoding": "euler_xyz_or_yaw",
            "rotation_unit": "degree",
            "rotation_action": "active",
            "source_zero_rotation_front": "positive_x",
            "asset_front_basis_transform": "matrix_composition_when_available",
            "position_semantics": "asset_instance_center",
            "origin_shift": origin_shift,
        },
        extra_metadata={
            "source_artifact": resolved_path.as_posix(),
            "source_boundary": source_boundary_kind,
            "room_contract_match": "exact_modulo_origin_cycle_winding",
        },
    )


def _scene_config(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
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


def _native_asset_local_bbox_size(
    native_asset: Mapping[str, Any],
) -> tuple[list[float] | None, str | None]:
    """Decode the released LayoutVLM preprocessing bbox contract.

    The official entrypoint swaps source X/Y before the solver. Frozen inputs
    preserve the original canonical-mesh bbox alongside that solver bbox so
    canonical output can retain the mesh-local dimensions and express the
    solver frame as a rotation-basis transform instead of relabeling axes.
    """

    metadata = native_asset.get("assetMetadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    axis_transform = metadata.get("axisTransform")
    canonical = metadata.get("canonicalBoundingBoxBeforeLayoutVLMSwap")
    if axis_transform is not None or canonical is not None:
        if axis_transform != "swap_xy_for_layoutvlm_processed_positive_x_frame":
            raise ArtifactValidationError(
                "LayoutVLM assetMetadata declares an unsupported bbox axis transform"
            )
        return (
            vector3(
                canonical,
                "LayoutVLM assetMetadata.canonicalBoundingBoxBeforeLayoutVLMSwap",
                positive=True,
            ),
            str(axis_transform),
        )
    return record_asset_local_bbox_size(native_asset), None


def _rotation(value: Any, path: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [0.0, 0.0, finite_float(value, path)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 1:
            return [0.0, 0.0, finite_float(value[0], f"{path}[0]")]
        if len(value) == 3:
            return vector3(value, path)
    raise ArtifactValidationError(f"{path} must be an angle or Euler vector")


def _scale3(value: Any, path: str) -> list[float]:
    if value is None:
        return [1.0, 1.0, 1.0]
    if isinstance(value, (int, float)):
        scale = finite_float(value, path)
        if scale <= 0.0:
            raise ArtifactValidationError(f"{path} must be positive")
        return [scale, scale, scale]
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        raise ArtifactValidationError(f"{path} must be a scalar or 3-vector")
    return vector3(value, path, positive=True)


def _strip_instance_suffix(value: str) -> str:
    return re.sub(r"-[0-9]+$", "", value)


def _nested(mapping: Mapping[str, Any], outer: str, inner: str) -> Any:
    value = mapping.get(outer)
    return value.get(inner) if isinstance(value, Mapping) else None


__all__ = ["convert_layout_vlm"]
