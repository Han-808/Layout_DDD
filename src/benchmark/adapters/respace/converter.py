"""ReSpace Y-up SSR to canonical Z-up object state."""

from __future__ import annotations

import math
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
    resolve_asset_record,
)
from benchmark.adapters.common.geometry import (
    build_scene,
    canonical_room,
    category_from_identifier,
    euler_xyz_to_matrix,
    finite_float,
    matrix_multiply,
    matrix_transpose,
    matrix_vector,
    quaternion_xyzw_to_matrix,
    reject_unsupported_architecture,
    require_boundary_match,
    rotation_matrix_to_euler_xyz_degrees,
    require_boundary_model,
    require_room_geometry_match,
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
ROTATION_ENCODINGS = {"quaternion_xyzw", "euler_xyz", "yaw"}
ROTATION_UNITS = {"degree", "radian"}
# Released SSR validates four components and uses SciPy's scalar-last convention.
RELEASED_ROTATION_ENCODING = "quaternion_xyzw"
UUID_IDENTIFIER = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


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

    benchmark_boundary, benchmark_height, _ = canonical_room(generation_input)
    bottom = scene.get("bounds_bottom")
    top = scene.get("bounds_top")
    if (
        not isinstance(bottom, Sequence)
        or isinstance(bottom, (str, bytes))
        or len(bottom) < 3
    ):
        raise ArtifactValidationError(
            "ReSpace SSR requires bounds_bottom from the native scene"
        )
    boundary, bottom_y = _horizontal_boundary(bottom, "bounds_bottom")
    require_boundary_model(
        boundary,
        supported=("axis_aligned_rectangle",),
        path="ReSpace bounds_bottom",
    )
    if (
        not isinstance(top, Sequence)
        or isinstance(top, (str, bytes))
        or len(top) < 3
    ):
        raise ArtifactValidationError(
            "ReSpace SSR requires bounds_top from the native scene"
        )
    top_boundary, top_y = _horizontal_boundary(top, "bounds_top")
    require_boundary_match(
        top_boundary,
        boundary,
        path="ReSpace bounds_top footprint",
        allow_translation=False,
    )
    scene_height = top_y - bottom_y
    if scene_height <= 0.0:
        raise ArtifactValidationError("ReSpace bounds_top must be above bounds_bottom")
    require_room_geometry_match(
        boundary,
        scene_height,
        benchmark_boundary,
        benchmark_height,
        path="ReSpace native room",
    )
    boundary, origin_shift = shift_boundary_to_origin(boundary)
    origin_shift[2] = -bottom_y

    native_objects = scene.get("objects")
    if not isinstance(native_objects, Sequence) or isinstance(native_objects, (str, bytes)):
        raise ArtifactValidationError("ReSpace SSR objects must be a list")
    objects: list[dict[str, Any]] = []
    default_source = str(config.get("asset_source_db") or "3d_future")
    anchor = str(config.get("position_anchor") or "bottom_center")
    if anchor not in {"bottom_center", "center"}:
        raise ArtifactValidationError("ReSpace position_anchor must be bottom_center or center")
    rotation_convention = _rotation_convention(config)

    for index, native in enumerate(native_objects):
        if not isinstance(native, Mapping):
            raise ArtifactValidationError(f"ReSpace objects[{index}] must be an object")
        object_id, object_id_source = _object_identity(native, index)
        asset_key_value = (
            native.get("sampled_asset_jid")
            or native.get("sampled_jid")
            or native.get("jid")
        )
        asset_key = str(asset_key_value) if asset_key_value is not None else None
        category, category_source = _native_category(
            native,
            object_id=object_id,
            object_id_source=object_id_source,
        )
        description = str(
            native.get("desc") or native.get("description") or category
        )
        source_size = vector3(native.get("size"), f"ReSpace objects[{index}].size", positive=True)
        size = [source_size[0], source_size[2], source_size[1]]
        source_position = vector3(
            native.get("pos") or native.get("position"),
            f"ReSpace objects[{index}].pos",
        )
        center = [source_position[0], -source_position[2], source_position[1]]
        native_rotation = (
            native["rot"] if "rot" in native else native.get("rotation")
        )
        rotation, canonical_rotation_matrix = _rotation(
            native_rotation,
            index,
            convention=rotation_convention,
        )
        center_offset = [0.0, 0.0, 0.0]
        if anchor == "bottom_center":
            center_offset = matrix_vector(
                canonical_rotation_matrix,
                [0.0, 0.0, size[2] / 2.0],
            )
            center = [
                center[axis] + center_offset[axis]
                for axis in range(3)
            ]
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
        asset_local_size = _native_asset_size(native, index)
        if asset_local_size is None:
            asset_local_size = record_asset_local_bbox_size(record)
        asset_local_center = record_bbox_center(record)
        native_scale = _native_scale(native.get("scale"), index)
        geometry_audit: dict[str, Any] = {
            "evaluated_size_source": "native.size",
            "native_size": source_size,
            "native_size_axes": "x_width_y_height_z_depth",
            "native_size_semantics": "ssr_placed_bbox",
            "bottom_center_to_center_offset": center_offset,
        }
        if asset_local_size is not None:
            geometry_audit["asset_local_bbox_size"] = asset_local_size
            geometry_audit["asset_local_bbox_source"] = (
                "native.sampled_asset_size"
                if native.get("sampled_asset_size") is not None
                else "asset_metadata"
            )
        if asset_local_center is not None:
            geometry_audit["asset_local_bbox_center"] = asset_local_center
        if native_scale is not None:
            geometry_audit["native_scale"] = native_scale
            geometry_audit["native_scale_source"] = "native.scale"
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
        record_category = _record_category(record)
        if category_source == "unavailable" and record_category:
            fields["category"] = record_category
            category_source = "asset_metadata"
        else:
            fields["category"] = category
            if not record.get("retrieval_category"):
                fields["retrieval_category"] = category
        metadata = dict(fields["metadata"])
        metadata.update(
            {
                "native_asset_id": asset_key,
                "native_unsampled_asset_id": native.get("jid"),
                "native_object_id_source": object_id_source,
                "native_category_source": category_source,
                "native_position_anchor": anchor,
                "native_description": description,
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
            "rotation_convention": rotation_convention,
        },
        extra_metadata={
            "source_artifact": resolved_path.as_posix(),
            "source_boundary": "respace_bounds",
            "room_contract_match": "exact_modulo_origin_cycle_winding",
            "native_room_id": scene.get("room_id"),
            "native_room_type": scene.get("room_type") or payload.get("room_type"),
        },
)


def _rotation_convention(config: Mapping[str, Any]) -> dict[str, Any]:
    encoding = str(
        config.get("rotation_encoding") or RELEASED_ROTATION_ENCODING
    ).strip()
    if encoding not in ROTATION_ENCODINGS:
        raise ArtifactValidationError(
            "ReSpace rotation_encoding must be quaternion_xyzw, euler_xyz, or yaw"
        )
    raw_unit = config.get("rotation_unit")
    if encoding == "quaternion_xyzw":
        if raw_unit is not None:
            raise ArtifactValidationError(
                "ReSpace rotation_unit is invalid for quaternion_xyzw encoding"
            )
        return {
            "encoding": encoding,
            "unit": "unitless",
            "quaternion_order": "xyzw",
            "rotation_action": "active",
            "basis_transform_source_to_canonical": SOURCE_TO_CANONICAL,
        }
    if raw_unit is None:
        raise ArtifactValidationError(
            f"ReSpace rotation_unit is required for {encoding} encoding"
        )
    unit = str(raw_unit).strip().lower()
    aliases = {
        "degrees": "degree",
        "radians": "radian",
    }
    unit = aliases.get(unit, unit)
    if unit not in ROTATION_UNITS:
        raise ArtifactValidationError(
            "ReSpace rotation_unit must be degree or radian"
        )
    convention = {
        "encoding": encoding,
        "unit": unit,
        "rotation_action": "active",
        "basis_transform_source_to_canonical": SOURCE_TO_CANONICAL,
    }
    if encoding == "euler_xyz":
        convention["euler_order"] = "xyz"
        convention["composition"] = "Rz@Ry@Rx_active"
    else:
        convention["rotation_axis"] = "source_y_up"
    return convention


def _rotation(
    value: Any,
    index: int,
    *,
    convention: Mapping[str, Any],
) -> tuple[list[float], list[list[float]]]:
    path = f"ReSpace objects[{index}].rot"
    if value is None:
        raise ArtifactValidationError(f"{path} is required by the native SSR contract")
    encoding = str(convention["encoding"])
    if encoding == "quaternion_xyzw":
        if not (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 4
        ):
            raise ArtifactValidationError(
                f"{path} must be an xyzw quaternion; use an explicit "
                "rotation_encoding for non-released representations"
            )
        source_rotation = quaternion_xyzw_to_matrix(value, path)
    elif encoding == "euler_xyz":
        if not (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 3
        ):
            raise ArtifactValidationError(f"{path} must be a 3-vector XYZ Euler rotation")
        source_rotation = euler_xyz_to_matrix(
            value,
            path,
            unit=str(convention["unit"]),
        )
    else:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 1:
                raise ArtifactValidationError(f"{path} must be a scalar yaw")
            value = value[0]
        angle = finite_float(value, path)
        if convention["unit"] == "degree":
            angle = math.radians(angle)
        source_rotation = euler_xyz_to_matrix(
            [0.0, angle, 0.0],
            path,
            unit="radian",
        )
    canonical_rotation = matrix_multiply(
        matrix_multiply(SOURCE_TO_CANONICAL, source_rotation),
        matrix_transpose(SOURCE_TO_CANONICAL),
    )
    return (
        rotation_matrix_to_euler_xyz_degrees(canonical_rotation),
        canonical_rotation,
    )


def _horizontal_boundary(
    points: Sequence[Any],
    name: str,
) -> tuple[list[list[float]], float]:
    boundary: list[list[float]] = []
    vertical: list[float] = []
    for index, point in enumerate(points):
        value = vector3(point, f"ReSpace {name}[{index}]")
        boundary.append([value[0], -value[2]])
        vertical.append(value[1])
    floor = vertical[0]
    if any(abs(value - floor) > 1.0e-6 for value in vertical[1:]):
        raise ArtifactValidationError(f"ReSpace {name} must be planar")
    return boundary, floor


def _object_identity(
    native: Mapping[str, Any],
    index: int,
) -> tuple[str, str]:
    for field in ("id", "instance_id", "uuid"):
        value = native.get(field)
        if value is not None and str(value).strip():
            return str(value).strip(), field
    return f"object_{index}", "generated_transport_id"


def _native_category(
    native: Mapping[str, Any],
    *,
    object_id: str,
    object_id_source: str,
) -> tuple[str, str]:
    for field in ("category", "class_label", "label"):
        value = native.get(field)
        if value is not None and str(value).strip():
            return str(value).strip(), field
    if (
        object_id_source in {"id", "instance_id"}
        and not UUID_IDENTIFIER.fullmatch(object_id)
    ):
        parsed = category_from_identifier(object_id, fallback="")
        if parsed and parsed != "object":
            return parsed, object_id_source
    return "unknown", "unavailable"


def _record_category(record: Mapping[str, Any]) -> str:
    for field in ("category", "class", "object_type"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _native_asset_size(
    native: Mapping[str, Any],
    index: int,
) -> list[float] | None:
    value = native.get("sampled_asset_size")
    if value is None:
        return None
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        raise ArtifactValidationError(
            f"ReSpace objects[{index}].scale must be a scalar or 3-vector"
        )
    source = vector3(
        value,
        f"ReSpace objects[{index}].sampled_asset_size",
        positive=True,
    )
    return [source[0], source[2], source[1]]


def _native_scale(value: Any, index: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        scale = finite_float(value, f"ReSpace objects[{index}].scale")
        if scale <= 0.0:
            raise ArtifactValidationError(
                f"ReSpace objects[{index}].scale must be positive"
            )
        return [scale, scale, scale]
    source = vector3(
        value,
        f"ReSpace objects[{index}].scale",
        positive=True,
    )
    return [source[0], source[2], source[1]]


__all__ = ["SOURCE_TO_CANONICAL", "convert_respace"]
