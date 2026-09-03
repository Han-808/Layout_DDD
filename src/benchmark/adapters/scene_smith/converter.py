"""SceneSmith checkpoint/SceneState output to canonical scene conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
    matrix_vector,
    quaternion_xyzw_to_matrix,
    require_boundary_model,
    require_room_geometry_match,
    rotation_matrix_to_euler_xyz_degrees,
    shift_boundary_to_origin,
    shift_center,
    vector3,
    vector4,
)
from benchmark.adapters.common.scene_state import convert_scene_state
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


def convert_scene_smith(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
) -> dict:
    resolved_path = _state_path(source_path, config)
    payload = read_json(resolved_path)
    if isinstance(payload, Mapping) and (
        payload.get("format") == "sceneState"
        or isinstance(payload.get("scene"), Mapping)
        and isinstance(payload["scene"].get("arch"), Mapping)
    ):
        return convert_scene_state(
            resolved_path,
            generation_input,
            config,
            provider,
            adapter_name="scene_smith",
            native_schema="scenesmith_scene_state_v1",
        )
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("SceneSmith state must be a JSON object")

    room_id, room_state, layout = _select_room(payload, config)
    geometry = room_state.get("room_geometry")
    if geometry is not None and not isinstance(geometry, Mapping):
        raise ArtifactValidationError("SceneSmith room_geometry must be an object")
    geometry = geometry if isinstance(geometry, Mapping) else {}
    fallback_boundary, fallback_height, _ = canonical_room(generation_input)
    length = geometry.get("length")
    width = geometry.get("width")
    if "length" in geometry or "width" in geometry:
        if length is None or width is None:
            raise ArtifactValidationError(
                "SceneSmith room_geometry requires both length and width"
            )
        length = finite_float(length, "SceneSmith room_geometry.length")
        width = finite_float(width, "SceneSmith room_geometry.width")
        if length <= 0.0 or width <= 0.0:
            raise ArtifactValidationError("SceneSmith room_geometry dimensions must be positive")
        boundary = [
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
        ]
        boundary_source = "room_geometry"
    else:
        boundary = fallback_boundary
        boundary_source = "generation_input"
    require_boundary_model(
        boundary,
        supported=("axis_aligned_rectangle",),
        path="SceneSmith room_geometry boundary",
    )
    height_source = (
        "room_geometry" if "wall_height" in geometry
        else "layout" if "wall_height" in layout else "generation_input"
    )
    scene_height = finite_float(
        geometry.get("wall_height", layout.get("wall_height", fallback_height)),
        "SceneSmith wall height",
    )
    if scene_height <= 0.0:
        raise ArtifactValidationError("SceneSmith wall height must be positive")
    require_room_geometry_match(
        boundary, scene_height, fallback_boundary, fallback_height,
        path="SceneSmith native room",
    )
    boundary, origin_shift = shift_boundary_to_origin(boundary)

    native_objects = room_state.get("objects")
    if not isinstance(native_objects, Mapping):
        raise ArtifactValidationError("SceneSmith room state.objects must be a JSON object")
    objects: list[dict[str, Any]] = []
    default_source = str(config.get("asset_source_db") or "scenesmith")
    asset_base = _asset_base(resolved_path, payload, room_id)

    for index, (key, native_value) in enumerate(native_objects.items()):
        if not isinstance(native_value, Mapping):
            raise ArtifactValidationError(f"SceneSmith objects.{key} must be an object")
        native = dict(native_value)
        object_type = str(native.get("object_type") or "object")
        if object_type in {"wall", "floor"}:
            continue
        object_id = str(native.get("object_id") or key)
        transform = native.get("transform")
        if not isinstance(transform, Mapping):
            raise ArtifactValidationError(f"SceneSmith object {object_id!r} requires transform")
        translation = vector3(
            transform.get("translation"), f"SceneSmith object {object_id}.transform.translation"
        )
        wxyz = vector4(
            transform.get("rotation_wxyz"), f"SceneSmith object {object_id}.transform.rotation_wxyz"
        )
        rotation_matrix = quaternion_xyzw_to_matrix(
            [wxyz[1], wxyz[2], wxyz[3], wxyz[0]],
            f"SceneSmith object {object_id}.transform.rotation_wxyz",
        )
        bbox_min = vector3(native.get("bbox_min"), f"SceneSmith object {object_id}.bbox_min")
        bbox_max = vector3(native.get("bbox_max"), f"SceneSmith object {object_id}.bbox_max")
        size = [bbox_max[axis] - bbox_min[axis] for axis in range(3)]
        if any(value <= 0.0 for value in size):
            raise ArtifactValidationError(f"SceneSmith object {object_id!r} has invalid bbox bounds")
        bbox_center = [(bbox_min[axis] + bbox_max[axis]) / 2.0 for axis in range(3)]
        rotated_center = matrix_vector(rotation_matrix, bbox_center)
        center = [translation[axis] + rotated_center[axis] for axis in range(3)]
        metadata_native = native.get("metadata")
        metadata_native = metadata_native if isinstance(metadata_native, Mapping) else {}
        asset_key = str(
            native.get("asset_id")
            or metadata_native.get("asset_id")
            or metadata_native.get("source_asset_id")
            or object_id
        )
        category = str(
            native.get("name")
            or metadata_native.get("category")
            or category_from_identifier(object_id)
        )
        description = str(native.get("description") or category)
        native_record = dict(native)
        native_record["asset_key"] = asset_key
        native_record["bbox_size"] = size
        native_record["bbox_center_local"] = bbox_center
        geometry_path = native.get("geometry_path")
        if geometry_path:
            native_record["mesh_uri"] = _resolve_native_path(asset_base, geometry_path)
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
                "interactive": object_type in {"manipuland", "furniture"},
                "native_object_type": object_type,
                "native_asset_id": asset_key,
                "native_room_id": room_id,
                "native_sdf_uri": (
                    _resolve_native_path(asset_base, native["sdf_path"])
                    if native.get("sdf_path")
                    else None
                ),
                "placement_info": native.get("placement_info"),
                "immutable": bool(native.get("immutable", False)),
                "native_object_index": index,
            }
        )
        objects.append(
            {
                "id": object_id,
                **{key: value for key, value in fields.items() if key != "metadata"},
                "size": size,
                "center": shift_center(center, origin_shift),
                "rotation": rotation_matrix_to_euler_xyz_degrees(rotation_matrix),
                "metadata": metadata,
            }
        )

    return build_scene(
        generation_input,
        adapter_name="scene_smith",
        native_schema="scenesmith_state_v1",
        boundary=boundary,
        scene_height=scene_height,
        objects=objects,
        coordinate_conversion={
            "source": "scenesmith_state",
            "source_axes": "x_width_y_depth_z_up",
            "source_unit": "meter",
            "position_semantics": "object_frame_origin_plus_local_bbox_center",
            "origin_shift": origin_shift,
        },
        extra_metadata={
            "source_artifact": resolved_path.as_posix(),
            "source_boundary": boundary_source,
            "source_height": height_source,
            "room_geometry_match": "validated_against_benchmark",
            "room_id": room_id,
        },
    )


def _state_path(source_path: Path, config: Mapping[str, Any]) -> Path:
    explicit = config.get("state_path") or config.get("scene_state_path")
    if explicit:
        path = Path(str(explicit)).expanduser()
        if not path.is_absolute():
            base = source_path if source_path.is_dir() else source_path.parent
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(f"SceneSmith configured state not found: {path}")
        return path
    if source_path.is_file():
        return source_path
    if not source_path.is_dir():
        raise FileNotFoundError(f"SceneSmith output does not exist: {source_path}")
    candidates = (
        "combined_house/house_state.json",
        "house_state.json",
        "scene_states/final_scene/scene_state.json",
        "scene_state.json",
        "combined_house/sceneeval_state.json",
        "sceneeval_state.json",
    )
    for relative in candidates:
        path = source_path / relative
        if path.is_file():
            return path
    raise ArtifactValidationError(
        f"SceneSmith output directory contains none of the expected state artifacts: {list(candidates)}"
    )


def _select_room(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(payload.get("objects"), Mapping):
        return str(config.get("room_id") or "room"), payload, {}
    rooms = payload.get("rooms")
    if not isinstance(rooms, Mapping) or not rooms:
        raise ArtifactValidationError("SceneSmith state requires top-level objects or rooms")
    room_id = str(config.get("room_id") or "").strip()
    if not room_id:
        if len(rooms) != 1:
            raise ArtifactValidationError(
                "SceneSmith house_state contains multiple rooms; set adapter_config.room_id"
            )
        room_id = str(next(iter(rooms)))
    room_state = rooms.get(room_id)
    if not isinstance(room_state, Mapping):
        raise ArtifactValidationError(f"SceneSmith room_id {room_id!r} matched no room")
    layout = payload.get("layout")
    return room_id, room_state, layout if isinstance(layout, Mapping) else {}


def _asset_base(state_path: Path, payload: Mapping[str, Any], room_id: str) -> Path:
    if isinstance(payload.get("rooms"), Mapping):
        scene_root = (
            state_path.parent.parent
            if state_path.parent.name.startswith("combined_house")
            else state_path.parent
        )
        return scene_root / f"room_{room_id}"
    return (
        state_path.parent.parent.parent
        if state_path.parent.name == "final_scene"
        else state_path.parent
    )


def _resolve_native_path(base: Path, value: Any) -> str:
    text = str(value)
    if text.startswith(("package://", "file://")):
        return text
    path = Path(text).expanduser()
    return (base / path).resolve().as_posix() if not path.is_absolute() else path.as_posix()


__all__ = ["convert_scene_smith"]
