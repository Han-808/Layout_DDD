"""Convert SmartScenes/SceneEval SceneState JSON to canonical object state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.adapters.common.artifacts import read_json_source
from benchmark.adapters.common.assets import (
    AssetProvider,
    asset_fields,
    record_bbox_center,
    record_bbox_size,
    resolve_asset_record,
)
from benchmark.adapters.common.geometry import (
    build_scene,
    canonical_room,
    category_from_identifier,
    finite_float,
    matrix_vector,
    require_boundary_model,
    require_room_geometry_match,
    rotation_matrix_to_euler_xyz_degrees,
    scene_state_matrix,
    shift_boundary_to_origin,
    shift_center,
)
from benchmark.scene_io.validate import ArtifactValidationError


SCENE_STATE_CANDIDATES = (
    "sceneeval_state.json",
    "stk_scene_state.json",
    "scene_states/final_scene/sceneeval_state.json",
    "combined_house/sceneeval_state.json",
)


def convert_scene_state(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
    *,
    adapter_name: str,
    native_schema: str,
) -> dict:
    payload, resolved_path = read_json_source(source_path, candidates=SCENE_STATE_CANDIDATES)
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("SceneState artifact must be a JSON object")
    scene = payload.get("scene") if isinstance(payload.get("scene"), Mapping) else payload
    if not isinstance(scene, Mapping):
        raise ArtifactValidationError("SceneState.scene must be a JSON object")
    if payload.get("format") not in (None, "sceneState"):
        raise ArtifactValidationError("SceneState.format must be 'sceneState'")

    arch = scene.get("arch")
    if not isinstance(arch, Mapping):
        raise ArtifactValidationError("SceneState.scene.arch must be a JSON object")
    elements = arch.get("elements")
    if not isinstance(elements, Sequence) or isinstance(elements, (str, bytes)):
        raise ArtifactValidationError("SceneState.scene.arch.elements must be a list")

    room_id = str(config.get("room_id") or "").strip() or None
    all_floors = [
        item
        for item in elements
        if isinstance(item, Mapping)
        and str(item.get("type") or "").casefold() == "floor"
    ]
    floors = [
        item
        for item in all_floors
        if room_id is None or str(item.get("roomId") or "") == room_id
    ]
    if not floors and all_floors and room_id is not None:
        raise ArtifactValidationError(f"SceneState room_id {room_id!r} matched no floor")
    if not floors:
        fallback_boundary, _, _ = canonical_room(generation_input)
        boundary = fallback_boundary
        selected_room_id = room_id
        source_boundary = "generation_input"
        scale_to_meters = 1.0
    else:
        if len(floors) > 1:
            raise ArtifactValidationError(
                "SceneState selection contains multiple floors; select exactly one room "
                "or provide a per-room export. The converter will not discard floors."
            )
        floor = floors[0]
        selected_room_id = str(floor.get("roomId") or "").strip() or room_id
        coords2d = arch.get("coords2d") or [0, 1]
        if not isinstance(coords2d, Sequence) or len(coords2d) < 2:
            raise ArtifactValidationError("SceneState.scene.arch.coords2d must contain two axis indices")
        axis0, axis1 = int(coords2d[0]), int(coords2d[1])
        unit = finite_float(scene.get("unit", 1.0), "SceneState.scene.unit")
        scale_to_meters = unit * finite_float(
            arch.get("scaleToMeters", 1.0), "SceneState.scene.arch.scaleToMeters"
        )
        if unit <= 0.0 or scale_to_meters <= 0.0:
            raise ArtifactValidationError("SceneState unit and scaleToMeters must be positive")
        points = floor.get("points")
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or len(points) < 3:
            raise ArtifactValidationError("SceneState floor.points must contain at least three points")
        boundary = []
        for index, point in enumerate(points):
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
                raise ArtifactValidationError(f"SceneState floor.points[{index}] must be a vector")
            try:
                boundary.append(
                    [
                        finite_float(point[axis0], f"SceneState floor.points[{index}][{axis0}]") * scale_to_meters,
                        finite_float(point[axis1], f"SceneState floor.points[{index}][{axis1}]") * scale_to_meters,
                    ]
                )
            except IndexError as exc:
                raise ArtifactValidationError(
                    f"SceneState floor.points[{index}] does not contain coords2d axes"
                ) from exc
        source_boundary = "scene_state_arch"

    require_boundary_model(
        boundary,
        supported=("axis_aligned_rectangle",),
        path="SceneState selected floor",
    )
    benchmark_boundary, fallback_height, _ = canonical_room(generation_input)
    wall_heights = [
        finite_float(item.get("height"), "SceneState wall.height") * scale_to_meters
        for item in elements
        if isinstance(item, Mapping)
        and str(item.get("type") or "").casefold() == "wall"
        and item.get("height") is not None
        and (selected_room_id is None or str(item.get("roomId") or "") == selected_room_id)
    ]
    scene_height = max(wall_heights) if wall_heights else fallback_height
    if scene_height <= 0.0 or any(height <= 0.0 for height in wall_heights):
        raise ArtifactValidationError("SceneState wall height must be positive")
    if any(abs(height - scene_height) > 1.0e-6 for height in wall_heights):
        raise ArtifactValidationError(
            "SceneState wall heights conflict; a single-height room cannot preserve them"
        )
    require_room_geometry_match(
        boundary, scene_height, benchmark_boundary, fallback_height,
        path="SceneState native room",
    )
    boundary, origin_shift = shift_boundary_to_origin(boundary)

    native_objects = scene.get("object")
    if not isinstance(native_objects, Sequence) or isinstance(native_objects, (str, bytes)):
        raise ArtifactValidationError("SceneState.scene.object must be a list")
    objects: list[dict[str, Any]] = []
    object_metadata = config.get("object_metadata")
    object_metadata = object_metadata if isinstance(object_metadata, Mapping) else {}
    default_source = str(config.get("asset_source_db") or adapter_name)

    for index, native in enumerate(native_objects):
        if not isinstance(native, Mapping):
            raise ArtifactValidationError(f"SceneState.scene.object[{index}] must be an object")
        native_room_id = str(native.get("roomId") or "").strip() or None
        if selected_room_id is not None and native_room_id is not None and native_room_id != selected_room_id:
            continue
        if len(all_floors) > 1 and selected_room_id is not None and native_room_id is None:
            raise ArtifactValidationError(
                "multi-room SceneState objects require roomId for projection; provide a per-room export otherwise"
            )
        object_id = str(native.get("id") or f"object_{index}")
        model_id = str(native.get("modelId") or native.get("model_id") or object_id)
        source_db, asset_key = _split_model_id(model_id, default_source)
        native_meta = object_metadata.get(object_id) or object_metadata.get(model_id) or {}
        native_meta = native_meta if isinstance(native_meta, Mapping) else {}
        category = str(
            native.get("category")
            or native_meta.get("category")
            or category_from_identifier(object_id)
        )
        description = str(
            native.get("description")
            or native_meta.get("description")
            or category
        )
        record = resolve_asset_record(
            provider,
            asset_key=asset_key,
            source_db=source_db,
            category=category,
            description=description,
            size=None,
            hint=native,
            native_record=native_meta,
            resolution_policy=str(
                config.get("asset_resolution_policy") or "exact_only"
            ),
        )
        intrinsic_size = record_bbox_size(record) or _optional_size(native)
        if intrinsic_size is None:
            raise ArtifactValidationError(
                f"SceneState object {object_id!r} requires bbox_size from the object or asset provider"
            )
        transform = native.get("transform")
        transform = transform if isinstance(transform, Mapping) else {}
        rotation, scale, translation = scene_state_matrix(
            transform.get("data"), f"SceneState.scene.object[{index}].transform.data"
        )
        scaled_size = [
            intrinsic_size[axis] * abs(scale[axis]) * scale_to_meters
            for axis in range(3)
        ]
        local_center = record_bbox_center(record) or [0.0, 0.0, 0.0]
        scaled_local_center = [
            local_center[axis] * scale[axis] * scale_to_meters for axis in range(3)
        ]
        rotated_local_center = matrix_vector(rotation, scaled_local_center)
        world_center = [
            translation[axis] * scale_to_meters + rotated_local_center[axis]
            for axis in range(3)
        ]
        fields = asset_fields(
            object_id=object_id,
            target_size=scaled_size,
            record=record,
            fallback_category=category,
            fallback_description=description,
            config=config,
        )
        metadata = dict(fields["metadata"])
        metadata.update(
            {
                "native_model_id": model_id,
                "native_object_index": native.get("index", index),
                "native_parent_id": native.get("parentId"),
            }
        )
        if native.get("sdfPath"):
            metadata["native_sdf_uri"] = str(native["sdfPath"])
        objects.append(
            {
                "id": object_id,
                **{key: value for key, value in fields.items() if key != "metadata"},
                "size": scaled_size,
                "center": shift_center(world_center, origin_shift),
                "rotation": rotation_matrix_to_euler_xyz_degrees(rotation),
                "metadata": metadata,
            }
        )

    return build_scene(
        generation_input,
        adapter_name=adapter_name,
        native_schema=native_schema,
        boundary=boundary,
        scene_height=scene_height,
        objects=objects,
        coordinate_conversion={
            "source": "scene_state",
            "source_up": scene.get("up", [0, 0, 1]),
            "source_front": scene.get("front", [0, 1, 0]),
            "scale_to_meters": scale_to_meters,
            "origin_shift": origin_shift,
        },
        extra_metadata={
            "source_artifact": resolved_path.as_posix(),
            "source_boundary": source_boundary,
            "source_height": "scene_state_arch" if wall_heights else "generation_input",
            "room_geometry_match": "validated_against_benchmark",
            "room_id": selected_room_id,
        },
    )


def _split_model_id(model_id: str, default_source: str) -> tuple[str, str]:
    if "." in model_id:
        source, key = model_id.split(".", 1)
        if source and key:
            return source, key
    return default_source, model_id


def _optional_size(value: Mapping[str, Any]) -> list[float] | None:
    raw = value.get("bbox_size") or value.get("size")
    if isinstance(raw, Mapping) and all(axis in raw for axis in ("x", "y", "z")):
        raw = [raw["x"], raw["y"], raw["z"]]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 3:
        return None
    try:
        result = [float(raw[index]) for index in range(3)]
    except (TypeError, ValueError):
        return None
    return result if all(item > 0.0 for item in result) else None


__all__ = ["SCENE_STATE_CANDIDATES", "convert_scene_state"]
