"""DirectLayout output-array to canonical scene conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.adapters.common.artifacts import read_json_source
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
            raise ArtifactValidationError(f"DirectLayout output[{index}] has an empty object id")
        if object_id in seen_ids:
            raise ArtifactValidationError(f"DirectLayout output contains duplicate id {object_id!r}")
        seen_ids.add(object_id)
        size_mapping = entry.get("size_in_meters") or entry.get("size")
        size = vector3(size_mapping, f"DirectLayout output[{index}].size", positive=True)
        position = vector3(entry.get("position") or entry.get("center"), f"DirectLayout output[{index}].position")
        rotation = entry.get("rotation")
        if isinstance(rotation, Mapping):
            yaw = finite_float(
                rotation.get("z_angle", rotation.get("z", 0.0)),
                f"DirectLayout output[{index}].rotation.z_angle",
            )
        elif isinstance(rotation, Sequence) and not isinstance(rotation, (str, bytes)):
            yaw = vector3(rotation, f"DirectLayout output[{index}].rotation")[2]
        else:
            yaw = finite_float(entry.get("orientation", entry.get("yaw", 0.0)), f"DirectLayout output[{index}].yaw")
        category = str(entry.get("category") or category_from_identifier(object_id))
        description = str(entry.get("description") or entry.get("prompt") or category)
        asset_key = str(entry.get("asset_id") or entry.get("jid") or object_id)
        record = resolve_asset_record(
            provider,
            asset_key=asset_key,
            source_db=default_source,
            category=category,
            description=description,
            size=size,
            hint=entry,
            native_record=entry,
            resolution_policy=str(
                config.get("asset_resolution_policy") or "exact_only"
            ),
        )
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
                "native_object_id": object_id,
                "native_asset_id": asset_key,
            }
        )
        objects.append(
            {
                "id": object_id,
                **{key: value for key, value in fields.items() if key != "metadata"},
                "size": size,
                "center": shift_center(position, origin_shift),
                "rotation": [0.0, 0.0, yaw],
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
            "position_semantics": "bbox_center",
            "origin_shift": origin_shift,
        },
        extra_metadata={"source_artifact": resolved_path.as_posix()},
    )


__all__ = ["convert_direct_layout"]
