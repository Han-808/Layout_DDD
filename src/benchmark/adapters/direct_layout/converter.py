"""DirectLayout output-array to canonical scene conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.adapters.common.artifacts import read_json_source
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
    compose_front_basis_rotation,
    finite_float,
    reject_unsupported_architecture,
    shift_boundary_to_origin,
    shift_center,
    vector3,
)
from benchmark.scene_io.validate import ArtifactValidationError


def convert_direct_layout(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
) -> dict:
    payload, resolved_path = read_json_source(
        source_path,
        candidates=("layout.json", "output_layout.json"),
    )
    if isinstance(payload, Mapping):
        if payload.get("rooms") or payload.get("room_layouts"):
            raise ArtifactValidationError(
                "DirectLayout adapter is single-room only and will not collapse multi-room output"
            )
        reject_unsupported_architecture(payload, path="DirectLayout output")
        native_room_fields = sorted(
            field
            for field in ("boundary", "floor_polygon", "roomsize")
            if payload.get(field)
        )
        if native_room_fields:
            raise ArtifactValidationError(
                "DirectLayout object output contains unsupported native room geometry "
                f"{native_room_fields}; the converter uses the declared generation-input room"
            )
        payload = payload.get("objects") or payload.get("layout") or payload.get("placements")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ArtifactValidationError("DirectLayout output must be an object placement array")

    source_boundary, scene_height, _ = canonical_room(generation_input)
    boundary, origin_shift = shift_boundary_to_origin(source_boundary)
    objects: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    default_source = str(config.get("asset_source_db") or "directlayout")
    asset_bindings = config.get("asset_bindings")
    asset_bindings = asset_bindings if isinstance(asset_bindings, Mapping) else {}

    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            raise ArtifactValidationError(f"DirectLayout output[{index}] must be an object")
        object_id = str(
            entry.get("new_object_id")
            or entry.get("object_id")
            or entry.get("id")
            or f"object_{index}"
        ).strip()
        if not object_id:
            raise ArtifactValidationError(
                f"DirectLayout output[{index}] has an empty object id"
            )
        if object_id in seen_ids:
            raise ArtifactValidationError(
                f"DirectLayout output contains duplicate id {object_id!r}"
            )
        seen_ids.add(object_id)
        size_field = (
            "size_in_meters"
            if entry.get("size_in_meters") is not None
            else "size"
        )
        size_mapping = entry.get(size_field)
        size = vector3(
            size_mapping,
            f"DirectLayout output[{index}].size",
            positive=True,
        )
        position = vector3(
            entry.get("position") or entry.get("center"),
            f"DirectLayout output[{index}].position",
        )
        rotation = entry.get("rotation")
        has_scalar_rotation = "orientation" in entry or "yaw" in entry
        if rotation is None and not has_scalar_rotation:
            raise ArtifactValidationError(
                f"DirectLayout output[{index}] requires a native rotation"
            )
        if isinstance(rotation, Mapping):
            if "z_angle" not in rotation and "z" not in rotation:
                raise ArtifactValidationError(
                    f"DirectLayout output[{index}].rotation requires z_angle or z"
                )
            yaw = finite_float(
                rotation.get("z_angle", rotation.get("z")),
                f"DirectLayout output[{index}].rotation.z_angle",
            )
        elif isinstance(rotation, Sequence) and not isinstance(rotation, (str, bytes)):
            if len(rotation) != 3:
                raise ArtifactValidationError(
                    f"DirectLayout output[{index}].rotation must be a 3-vector"
                )
            yaw = vector3(rotation, f"DirectLayout output[{index}].rotation")[2]
        else:
            yaw = finite_float(
                entry.get("orientation", entry.get("yaw")),
                f"DirectLayout output[{index}].yaw",
            )
        binding_value = asset_bindings.get(object_id)
        binding = (
            dict(binding_value)
            if isinstance(binding_value, Mapping)
            else {"asset_key": binding_value}
            if binding_value is not None
            else {}
        )
        category = str(
            entry.get("category")
            or binding.get("category")
            or category_from_identifier(object_id)
        )
        description = str(
            entry.get("description")
            or entry.get("prompt")
            or binding.get("description")
            or category
        )
        asset_key_value = (
            entry.get("asset_id")
            or entry.get("jid")
            or binding.get("asset_key")
            or binding.get("asset_id")
            or object_id
        )
        asset_key = str(asset_key_value) if asset_key_value is not None else None
        record = resolve_asset_record(
            provider,
            asset_key=asset_key,
            source_db=default_source,
            category=category,
            description=description,
            size=size,
            hint=entry,
            native_record={**binding, **entry},
            resolution_policy=str(
                config.get("asset_resolution_policy") or "exact_only"
            ),
        )
        asset_local_size = record_asset_local_bbox_size(record)
        asset_local_center = record_bbox_center(record)
        canonical_rotation = [0.0, 0.0, yaw]
        front_basis_yaw = None
        canonical_front = record.get("canonical_front")
        if canonical_front is not None:
            canonical_rotation, front_basis_yaw = compose_front_basis_rotation(
                canonical_rotation,
                canonical_front=canonical_front,
                native_zero_front=[0.0, 1.0, 0.0],
                path=f"DirectLayout output[{index}].front_basis",
            )
        geometry_audit: dict[str, Any] = {
            "evaluated_size_source": f"native.{size_field}",
            "native_size": size,
            "native_size_axes": "x_length_y_width_z_height",
            "native_size_semantics": "placed_bbox_after_directlayout_rescale",
        }
        if asset_local_size is not None:
            geometry_audit["asset_local_bbox_size"] = asset_local_size
            geometry_audit["asset_local_bbox_source"] = "asset_metadata"
        if asset_local_center is not None:
            geometry_audit["asset_local_bbox_center"] = asset_local_center
        if front_basis_yaw is not None:
            geometry_audit.update(
                {
                    "native_zero_front": [0.0, 1.0, 0.0],
                    "canonical_front_basis_yaw_degrees": front_basis_yaw,
                    "orientation_basis_transform": (
                        "canonical_asset_front_to_directlayout_positive_y"
                    ),
                }
            )
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
        metadata.update(
            {
                "native_object_id": object_id,
                "native_asset_id": asset_key,
                "native_asset_binding_source": (
                    "native_object"
                    if entry.get("asset_id") or entry.get("jid")
                    else "preserved_asset_bindings_sidecar"
                    if binding
                    else "native_object_id"
                ),
                "native_rotation_degrees": yaw,
                "native_zero_front": [0.0, 1.0, 0.0],
            }
        )
        objects.append(
            {
                "id": object_id,
                **{key: value for key, value in fields.items() if key != "metadata"},
                "size": size,
                "center": shift_center(position, origin_shift),
                "rotation": canonical_rotation,
                "metadata": metadata,
            }
        )

    return build_scene(
        generation_input,
        adapter_name="direct_layout",
        native_schema="directlayout_output_v1",
        boundary=boundary,
        scene_height=scene_height,
        objects=objects,
        coordinate_conversion={
            "source": "directlayout",
            "source_axes": "x_width_y_depth_z_up",
            "source_unit": "meter",
            "source_rotation_encoding": "yaw_z",
            "source_rotation_unit": "degree",
            "source_orientation_reference": "positive_y_at_zero_degrees",
            "asset_front_basis_transform": "matrix_composition_when_available",
            "position_semantics": "bbox_center",
            "origin_shift": origin_shift,
        },
        extra_metadata={"source_artifact": resolved_path.as_posix()},
    )


__all__ = ["convert_direct_layout"]
