"""ReSpace Y-up SSR to canonical Z-up object state."""

from __future__ import annotations

import math
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
    matrix_multiply,
    matrix_transpose,
    quaternion_xyzw_to_matrix,
    reject_unsupported_architecture,
    rotation_matrix_to_euler_xyz_degrees,
    require_boundary_model,
    shift_boundary_to_origin,
    shift_center,
    vector3,
)
from benchmark.scene_io.validate import ArtifactValidationError


SOURCE_TO_CANONICAL = [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
]


def convert_respace(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
) -> dict:
    payload, resolved_path = read_json_source(
        source_path,
        candidates=("scene.json", "ssr.json"),
    )
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("ReSpace SSR output must be a JSON object")
    scene = payload.get("scene") if isinstance(payload.get("scene"), Mapping) else payload
    if not isinstance(scene, Mapping):
        raise ArtifactValidationError("ReSpace SSR scene must be a JSON object")
    if (
        payload.get("rooms")
        or payload.get("room_layouts")
        or scene.get("rooms")
        or scene.get("room_layouts")
    ):
        raise ArtifactValidationError(
            "ReSpace adapter is single-room only and will not collapse multi-room output"
        )
    reject_unsupported_architecture(payload, path="ReSpace SSR output")
    if scene is not payload:
        reject_unsupported_architecture(scene, path="ReSpace SSR scene")

    bottom = scene.get("bounds_bottom")
    top = scene.get("bounds_top")
    if not isinstance(bottom, Sequence) or isinstance(bottom, (str, bytes)) or len(bottom) < 3:
        fallback_boundary, fallback_height, _ = canonical_room(generation_input)
        boundary = fallback_boundary
        scene_height = fallback_height
        source_boundary = "generation_input"
    else:
        boundary = []
        bottom_y_values = []
        for index, point in enumerate(bottom):
            value = vector3(point, f"ReSpace bounds_bottom[{index}]")
            boundary.append([value[0], -value[2]])
            bottom_y_values.append(value[1])
        bottom_y = min(bottom_y_values)
        if isinstance(top, Sequence) and not isinstance(top, (str, bytes)) and top:
            top_y = max(vector3(point, "ReSpace bounds_top point")[1] for point in top)
            scene_height = top_y - bottom_y
        else:
            _, scene_height, _ = canonical_room(generation_input)
        if scene_height <= 0.0:
            raise ArtifactValidationError("ReSpace bounds_top must be above bounds_bottom")
        source_boundary = "respace_bounds"
    require_boundary_model(
        boundary,
        supported=("axis_aligned_rectangle",),
        path="ReSpace bounds_bottom",
    )
    boundary, origin_shift = shift_boundary_to_origin(boundary)

    native_objects = scene.get("objects")
    if not isinstance(native_objects, Sequence) or isinstance(native_objects, (str, bytes)):
        raise ArtifactValidationError("ReSpace SSR objects must be a list")
    objects: list[dict[str, Any]] = []
    default_source = str(config.get("asset_source_db") or "3d_future")
    anchor = str(config.get("position_anchor") or "bottom_center")
    if anchor not in {"bottom_center", "center"}:
        raise ArtifactValidationError("ReSpace position_anchor must be bottom_center or center")

    for index, native in enumerate(native_objects):
        if not isinstance(native, Mapping):
            raise ArtifactValidationError(f"ReSpace objects[{index}] must be an object")
        object_id = str(native.get("id") or native.get("instance_id") or f"object_{index}")
        asset_key = str(native.get("sampled_jid") or native.get("jid") or object_id)
        description = str(native.get("desc") or native.get("description") or "object")
        category_hint = native.get("category") or native.get("class_label") or native.get("label")
        category = str(category_hint or category_from_identifier(category_hint or "object"))
        source_size = vector3(native.get("size"), f"ReSpace objects[{index}].size", positive=True)
        size = [source_size[0], source_size[2], source_size[1]]
        source_position = vector3(
            native.get("pos") or native.get("position"),
            f"ReSpace objects[{index}].pos",
        )
        center = [source_position[0], -source_position[2], source_position[1]]
        if anchor == "bottom_center":
            center[2] += size[2] / 2.0
        rotation = _rotation(native.get("rot") or native.get("rotation"), index)
        record = resolve_asset_record(
            provider,
            asset_key=asset_key,
            source_db=default_source,
            category=category,
            description=description,
            size=size,
            hint=native,
            native_record=native,
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
                "native_asset_id": asset_key,
                "native_position_anchor": anchor,
                "native_description": description,
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
        adapter_name="respace",
        native_schema="respace_ssr_v1",
        boundary=boundary,
        scene_height=scene_height,
        objects=objects,
        coordinate_conversion={
            "source": "respace_ssr",
            "source_axes": "x_width_y_up_z_depth",
            "canonical_mapping": "[x, -z, y]",
            "source_unit": "meter",
            "position_anchor": anchor,
            "origin_shift": origin_shift,
        },
        extra_metadata={
            "source_artifact": resolved_path.as_posix(),
            "source_boundary": source_boundary,
            "native_room_id": scene.get("room_id"),
            "native_room_type": scene.get("room_type") or payload.get("room_type"),
        },
    )


def _rotation(value: Any, index: int) -> list[float]:
    path = f"ReSpace objects[{index}].rot"
    if value is None:
        return [0.0, 0.0, 0.0]
    if isinstance(value, (int, float)):
        angle = finite_float(value, path)
        degrees = math.degrees(angle) if abs(angle) <= 2.0 * math.pi + 1.0e-6 else angle
        return [0.0, 0.0, -degrees]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        source_rotation = quaternion_xyzw_to_matrix(value, path)
        canonical_rotation = matrix_multiply(
            matrix_multiply(SOURCE_TO_CANONICAL, source_rotation),
            matrix_transpose(SOURCE_TO_CANONICAL),
        )
        return rotation_matrix_to_euler_xyz_degrees(canonical_rotation)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        source_euler = vector3(value, path)
        return [source_euler[0], -source_euler[2], -source_euler[1]]
    raise ArtifactValidationError(f"{path} must be a quaternion, Euler vector, or yaw")


__all__ = ["SOURCE_TO_CANONICAL", "convert_respace"]
