"""Holodeck ProcTHOR/SceneState output to canonical scene conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
from benchmark.adapters.common.scene_state import convert_scene_state
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


def convert_holodeck(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
) -> dict:
    resolved_path = _source_path(source_path, config)
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
            adapter_name="holodeck",
            native_schema="holodeck_scene_state_v1",
        )
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("Holodeck output must be a ProcTHOR JSON object")

    rooms = payload.get("rooms")
    if not isinstance(rooms, Sequence) or isinstance(rooms, (str, bytes)) or not rooms:
        raise ArtifactValidationError("Holodeck output.rooms must be a non-empty list")
    rooms = [room for room in rooms if isinstance(room, Mapping)]
    room_id = str(config.get("room_id") or "").strip() or None
    if room_id is not None:
        rooms = [room for room in rooms if str(room.get("id") or "") == room_id]
    if not rooms:
        raise ArtifactValidationError("Holodeck room_id matched no room")
    if len(rooms) > 1:
        raise ArtifactValidationError(
            "Holodeck output contains multiple rooms; set adapter_config.room_id or convert a per-room SceneState"
        )
    room = rooms[0]
    selected_room_id = str(room.get("id") or room_id or "room")
    floor_polygon = room.get("floorPolygon")
    if not isinstance(floor_polygon, Sequence) or isinstance(floor_polygon, (str, bytes)) or len(floor_polygon) < 3:
        raise ArtifactValidationError("Holodeck room.floorPolygon must contain at least three points")
    boundary = []
    for index, point in enumerate(floor_polygon):
        if not isinstance(point, Mapping):
            raise ArtifactValidationError(f"Holodeck floorPolygon[{index}] must be an object")
        boundary.append(
            [
                finite_float(point.get("x"), f"Holodeck floorPolygon[{index}].x"),
                finite_float(point.get("z"), f"Holodeck floorPolygon[{index}].z"),
            ]
        )
    boundary, origin_shift = shift_boundary_to_origin(boundary)
    _, fallback_height, _ = canonical_room(generation_input)
    wall_heights = [
        finite_float(wall.get("height"), "Holodeck wall.height")
        for wall in payload.get("walls", [])
        if isinstance(wall, Mapping)
        and wall.get("height") is not None
        and str(wall.get("roomId") or selected_room_id) == selected_room_id
    ]
    scene_height = finite_float(
        payload.get("wall_height", max(wall_heights) if wall_heights else fallback_height),
        "Holodeck wall height",
    )

    native_objects = payload.get("objects")
    if not isinstance(native_objects, Sequence) or isinstance(native_objects, (str, bytes)):
        raise ArtifactValidationError("Holodeck output.objects must be a list")
    default_source = str(config.get("asset_source_db") or "objathor")
    bbox_axes = str(config.get("asset_bbox_axes") or "unity_xyz")
    if bbox_axes not in {"unity_xyz", "canonical_xyz"}:
        raise ArtifactValidationError("Holodeck asset_bbox_axes must be unity_xyz or canonical_xyz")
    objects: list[dict[str, Any]] = []

    for index, native in enumerate(native_objects):
        if not isinstance(native, Mapping):
            raise ArtifactValidationError(f"Holodeck objects[{index}] must be an object")
        native_room = str(native.get("roomId") or selected_room_id)
        if native_room != selected_room_id:
            continue
        object_id = str(native.get("id") or f"object_{index}")
        asset_key = str(native.get("assetId") or native.get("asset_id") or object_id)
        category = str(
            native.get("object_name")
            or native.get("category")
            or category_from_identifier(object_id)
        )
        description = str(native.get("description") or category)
        record = resolve_asset_record(
            provider,
            asset_key=asset_key,
            source_db=default_source,
            category=category,
            description=description,
            size=None,
            hint=native,
            native_record=native,
            resolution_policy=str(
                config.get("asset_resolution_policy") or "exact_only"
            ),
        )
        source_size = record_bbox_size(record) or _native_bbox_size(native)
        if source_size is None:
            raise ArtifactValidationError(
                f"Holodeck object {object_id!r} requires asset bbox_size from its Objathor provider"
            )
        size = (
            [source_size[0], source_size[2], source_size[1]]
            if bbox_axes == "unity_xyz"
            else list(source_size)
        )
        source_position = vector3(native.get("position"), f"Holodeck objects[{index}].position")
        center = [source_position[0], source_position[2], source_position[1]]
        rotation = native.get("rotation")
        rotation = rotation if isinstance(rotation, Mapping) else {}
        source_rotation = [
            finite_float(rotation.get("x", 0.0), f"Holodeck objects[{index}].rotation.x"),
            finite_float(rotation.get("y", 0.0), f"Holodeck objects[{index}].rotation.y"),
            finite_float(rotation.get("z", 0.0), f"Holodeck objects[{index}].rotation.z"),
        ]
        yaw_offset = finite_float(
            record.get("rotation_offset_degrees", config.get("yaw_offset_degrees", 0.0)),
            f"Holodeck object {object_id}.rotation_offset_degrees",
        )
        canonical_rotation = [source_rotation[2], source_rotation[0], -source_rotation[1] + yaw_offset]
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
                "native_room_id": native_room,
                "native_kinematic": native.get("kinematic"),
                "native_unity_rotation": source_rotation,
                "raw_pose_conversion": True,
            }
        )
        objects.append(
            {
                "id": object_id,
                **{key: value for key, value in fields.items() if key != "metadata"},
                "size": size,
                "center": shift_center(center, origin_shift),
                "rotation": canonical_rotation,
                "metadata": metadata,
            }
        )

    return build_scene(
        generation_input,
        adapter_name="holodeck",
        native_schema="holodeck_procthor_v1",
        boundary=boundary,
        scene_height=scene_height,
        objects=objects,
        coordinate_conversion={
            "source": "holodeck_procthor",
            "source_axes": "unity_x_y_up_z",
            "canonical_mapping": "[x, z, y]",
            "source_unit": "meter",
            "position_semantics": "bbox_center",
            "origin_shift": origin_shift,
            "note": "SceneState export is preferred when exact Unity-composed transforms are available",
        },
        extra_metadata={
            "source_artifact": resolved_path.as_posix(),
            "room_id": selected_room_id,
            "conversion_route": "raw_procthor",
        },
    )


def _source_path(source_path: Path, config: Mapping[str, Any]) -> Path:
    explicit = config.get("scene_state_path") or config.get("scene_json_path")
    if explicit:
        path = Path(str(explicit)).expanduser()
        if not path.is_absolute():
            base = source_path if source_path.is_dir() else source_path.parent
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(f"Holodeck configured scene artifact not found: {path}")
        return path
    if source_path.is_file():
        return source_path
    if source_path.is_dir():
        direct_candidates = [
            source_path / "sceneeval_state.json",
            source_path / "converted_scene.json",
            source_path / "scene.json",
        ]
        for candidate in direct_candidates:
            if candidate.is_file():
                return candidate
        json_files = sorted(source_path.glob("*.json"))
        if len(json_files) == 1:
            return json_files[0]
        raise ArtifactValidationError(
            "Holodeck output directory is ambiguous; set adapter_config.scene_json_path or scene_state_path"
        )
    raise FileNotFoundError(f"Holodeck output does not exist: {source_path}")


def _native_bbox_size(native: Mapping[str, Any]) -> list[float] | None:
    bbox = native.get("axisAlignedBoundingBox")
    if isinstance(bbox, Mapping):
        size = bbox.get("size")
        if isinstance(size, Mapping):
            try:
                return [float(size["x"]), float(size["y"]), float(size["z"])]
            except (KeyError, TypeError, ValueError):
                pass
    return None


__all__ = ["convert_holodeck"]
