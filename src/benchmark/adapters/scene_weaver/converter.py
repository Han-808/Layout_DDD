"""SceneWeaver final-iteration layout to canonical scene conversion."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
    euler_xyz_to_matrix,
    finite_float,
    matrix_vector,
    reject_unsupported_architecture,
    require_room_geometry_match,
    shift_boundary_to_origin,
    shift_center,
    vector3,
)
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


LAYOUT_PATTERN = re.compile(r"^layout_([0-9]+)\.json$")
ITERATION_SELECTION_POLICIES = {"explicit", "latest"}


def convert_scene_weaver(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
) -> dict:
    layout_path, iteration_selection = _select_layout(source_path, config)
    payload = read_json(layout_path)
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("SceneWeaver layout must be a JSON object")
    if payload.get("rooms") or payload.get("room_layouts"):
        raise ArtifactValidationError(
            "SceneWeaver adapter is single-room only and will not collapse multi-room output"
        )
    reject_unsupported_architecture(payload, path="SceneWeaver layout")
    unsupported_boundary_fields = sorted(
        field for field in ("boundary", "floor_polygon") if payload.get(field)
    )
    if unsupported_boundary_fields:
        raise ArtifactValidationError(
            "SceneWeaver layout contains unsupported native boundary fields "
            f"{unsupported_boundary_fields}; only rectangular roomsize is supported"
        )
    native_objects = payload.get("objects")
    if not isinstance(native_objects, Mapping):
        raise ArtifactValidationError("SceneWeaver layout.objects must be a JSON object")

    fallback_boundary, fallback_height, _ = canonical_room(generation_input)
    roomsize = payload.get("roomsize")
    if (
        isinstance(roomsize, Sequence)
        and not isinstance(roomsize, (str, bytes))
        and len(roomsize) == 2
    ):
        width = finite_float(roomsize[0], "SceneWeaver roomsize[0]")
        depth = finite_float(roomsize[1], "SceneWeaver roomsize[1]")
        if width <= 0.0 or depth <= 0.0:
            raise ArtifactValidationError("SceneWeaver roomsize values must be positive")
        boundary = [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]]
        boundary_source = "sceneweaver_roomsize"
    else:
        raise ArtifactValidationError(
            "SceneWeaver layout requires native roomsize [length, width]"
        )
    scene_height = finite_float(
        config.get("scene_height", fallback_height),
        "SceneWeaver scene_height",
    )
    if scene_height <= 0.0:
        raise ArtifactValidationError("SceneWeaver scene_height must be positive")
    require_room_geometry_match(
        boundary,
        scene_height,
        fallback_boundary,
        fallback_height,
        path="SceneWeaver native room",
    )
    boundary, origin_shift = shift_boundary_to_origin(boundary)
    rotation_unit = str(config.get("rotation_unit") or "radian").strip().lower()
    rotation_unit = {
        "radians": "radian",
        "degrees": "degree",
    }.get(rotation_unit, rotation_unit)
    if rotation_unit not in {"radian", "degree"}:
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
        size_semantics = str(
            config.get("sceneweaver_native_size_semantics")
            or "scaled_object_local_bbox_dimensions"
        )
        native_size = vector3(
            native.get("size"),
            f"SceneWeaver objects.{object_id}.size",
        )
        if size_semantics == "released_world_aabb_rounded_2dp":
            if any(value < 0.0 for value in native_size):
                raise ArtifactValidationError(
                    f"SceneWeaver objects.{object_id}.size must be non-negative"
                )
        elif any(value <= 0.0 for value in native_size):
            raise ArtifactValidationError(
                f"SceneWeaver objects.{object_id}.size must be positive"
            )
        bottom_center = vector3(
            native.get("location") or native.get("position"),
            f"SceneWeaver objects.{object_id}.location",
        )
        native_rotation = native.get("rotation")
        if not (
            isinstance(native_rotation, Sequence)
            and not isinstance(native_rotation, (str, bytes))
            and len(native_rotation) == 3
        ):
            raise ArtifactValidationError(
                f"SceneWeaver objects.{object_id}.rotation must be a 3-vector"
            )
        source_rotation = vector3(
            native_rotation,
            f"SceneWeaver objects.{object_id}.rotation",
        )
        rotation = list(source_rotation)
        if rotation_unit == "radian":
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
            size=native_size,
            hint=native,
            native_record=native_record,
            resolution_policy=str(
                config.get("asset_resolution_policy") or "exact_only"
            ),
        )
        asset_local_size = record_asset_local_bbox_size(record)
        asset_local_center = record_bbox_center(record)
        front_basis_yaw = None
        if size_semantics == "released_world_aabb_rounded_2dp":
            if rotation_unit != "radian":
                raise ArtifactValidationError(
                    "released SceneWeaver FrozenAssets rotation is fixed to radians"
                )
            if config.get("sceneweaver_orientation_basis") != (
                "bake_catalog_front_to_sceneweaver_positive_x"
            ):
                raise ArtifactValidationError(
                    "released SceneWeaver FrozenAssets conversion requires the "
                    "declared canonical-front to native +X basis contract"
                )
            if config.get("sceneweaver_anchor_basis") != (
                "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"
            ):
                raise ArtifactValidationError(
                    "released SceneWeaver FrozenAssets conversion requires the "
                    "declared catalog-bbox bottom-center origin contract"
                )
            anchor_basis = binding.get("anchor_basis")
            if (
                not isinstance(anchor_basis, Mapping)
                or anchor_basis.get("policy")
                != "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"
                or anchor_basis.get("native_origin_semantics")
                != "bbox_bottom_center"
                or anchor_basis.get("applied") is not True
            ):
                raise ArtifactValidationError(
                    f"SceneWeaver object {object_id!r} lacks observed anchor-basis evidence"
                )
            size = vector3(
                record.get("physical_dimensions") or asset_local_size,
                f"SceneWeaver objects.{object_id}.asset_physical_dimensions",
                positive=True,
            )
            selected_iteration = iteration_selection.get("selected_iteration")
            precise_rotations = binding.get(
                "full_precision_native_euler_xyz_by_iteration"
            )
            precise_rotations = (
                precise_rotations if isinstance(precise_rotations, Mapping) else {}
            )
            precise_rotation_value = precise_rotations.get(str(selected_iteration))
            precise_source_rotation = vector3(
                precise_rotation_value,
                (
                    f"SceneWeaver objects.{object_id}."
                    "full_precision_native_euler_xyz"
                ),
            )
            serialized_expected = [
                round(value, 2) for value in precise_source_rotation
            ]
            tolerance = finite_float(
                config.get("sceneweaver_world_aabb_tolerance", 1.0e-6),
                "SceneWeaver world AABB tolerance",
            )
            if tolerance < 0.0:
                raise ArtifactValidationError(
                    "SceneWeaver world AABB tolerance must be non-negative"
                )
            if not _vectors_close(source_rotation, serialized_expected, tolerance):
                raise ArtifactValidationError(
                    f"SceneWeaver objects.{object_id}.rotation does not match the "
                    "released two-decimal serialization of the observed native pose"
                )
            precise_local_bboxes = binding.get(
                "full_precision_native_local_bbox_size_by_iteration"
            )
            precise_local_bboxes = (
                precise_local_bboxes
                if isinstance(precise_local_bboxes, Mapping)
                else {}
            )
            observed_local_size = vector3(
                precise_local_bboxes.get(str(selected_iteration)),
                (
                    f"SceneWeaver objects.{object_id}."
                    "full_precision_native_local_bbox_size"
                ),
                positive=True,
            )
            geometry_tolerance = finite_float(
                config.get("sceneweaver_asset_geometry_tolerance_m"),
                "SceneWeaver asset geometry tolerance",
            )
            if geometry_tolerance <= 0.0:
                raise ArtifactValidationError(
                    "SceneWeaver asset geometry tolerance must be positive"
                )
            if not _vectors_close(observed_local_size, size, geometry_tolerance):
                raise ArtifactValidationError(
                    f"SceneWeaver objects.{object_id} observed GLB bbox differs "
                    "from the frozen catalog physical dimensions"
                )
            rotation = list(precise_source_rotation)
            if rotation_unit == "radian":
                rotation = [math.degrees(value) for value in rotation]
            canonical_front = record.get("canonical_front")
            if canonical_front is not None:
                rotation, front_basis_yaw = compose_front_basis_rotation(
                    rotation,
                    canonical_front=canonical_front,
                    native_zero_front=[1.0, 0.0, 0.0],
                    path=f"SceneWeaver objects.{object_id}.front_basis",
                )
            rotation_matrix = euler_xyz_to_matrix(
                rotation,
                f"SceneWeaver objects.{object_id}.canonical_rotation",
                unit="degree",
            )
            expected_world_aabb = _world_aabb_size(
                rotation_matrix,
                observed_local_size,
            )
            expected_rounded = [round(value, 2) for value in expected_world_aabb]
            if not _vectors_close(native_size, expected_rounded, tolerance):
                raise ArtifactValidationError(
                    f"SceneWeaver objects.{object_id}.size is not the released "
                    "two-decimal world AABB implied by the frozen local bbox and pose"
                )
        elif size_semantics == "scaled_object_local_bbox_dimensions":
            size = native_size
            rotation_matrix = euler_xyz_to_matrix(
                source_rotation,
                f"SceneWeaver objects.{object_id}.rotation",
                unit=rotation_unit,
            )
            expected_world_aabb = None
            expected_rounded = None
        else:
            raise ArtifactValidationError(
                "unsupported SceneWeaver native size semantics "
                f"{size_semantics!r}"
            )
        center_offset = matrix_vector(
            rotation_matrix,
            [0.0, 0.0, size[2] / 2.0],
        )
        center = [
            bottom_center[axis] + center_offset[axis]
            for axis in range(3)
        ]
        geometry_audit: dict[str, Any] = {
            "evaluated_size_source": (
                "exact_asset_physical_dimensions"
                if size_semantics == "released_world_aabb_rounded_2dp"
                else "native.size"
            ),
            "native_size": native_size,
            "native_size_axes": "x_width_y_depth_z_height",
            "native_size_semantics": size_semantics,
            "native_location_semantics": "world_bbox_bottom_center",
            "bottom_center_to_center_offset": center_offset,
        }
        if expected_world_aabb is not None:
            geometry_audit.update(
                {
                    "expected_world_aabb_before_rounding": expected_world_aabb,
                    "expected_released_world_aabb": expected_rounded,
                    "released_world_aabb_verified": True,
                    "native_serialized_rotation": source_rotation,
                    "full_precision_native_rotation": precise_source_rotation,
                    "released_rotation_quantization_verified": True,
                    "observed_runtime_local_bbox_size": observed_local_size,
                    "asset_geometry_tolerance_m": geometry_tolerance,
                }
            )
            geometry_audit["anchor_basis"] = dict(anchor_basis)
        if front_basis_yaw is not None:
            geometry_audit.update(
                {
                    "native_zero_front": [1.0, 0.0, 0.0],
                    "canonical_front_basis_yaw_degrees": front_basis_yaw,
                    "orientation_basis_transform": (
                        "canonical_asset_front_to_sceneweaver_positive_x"
                    ),
                }
            )
        if asset_local_size is not None:
            geometry_audit["asset_local_bbox_size"] = asset_local_size
            geometry_audit["asset_local_bbox_source"] = "asset_metadata_or_binding"
        if asset_local_center is not None:
            geometry_audit["asset_local_bbox_center"] = asset_local_center
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
        if size_semantics != "released_world_aabb_rounded_2dp":
            metadata.setdefault("canonical_front", [1.0, 0.0, 0.0])
            metadata.setdefault(
                "canonical_front_source",
                "sceneweaver_local_x_front_contract",
            )
        metadata.update(
            {
                "native_object_index": index,
                "native_asset_id": asset_key,
                "native_position_anchor": "bottom_center",
                "native_parent_relations": native.get("parent") or [],
                "native_rotation": native_rotation,
                "native_zero_front": [1.0, 0.0, 0.0],
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
            "source_rotation_encoding": "blender_euler_xyz",
            "source_rotation_composition": "Rz@Ry@Rx_active",
            "source_size_semantics": str(
                config.get("sceneweaver_native_size_semantics")
                or "scaled_object_local_bbox_dimensions"
            ),
            "asset_front_basis_transform": "matrix_composition_when_available",
            "position_semantics": "bbox_bottom_center",
            "origin_shift": origin_shift,
        },
        extra_metadata={
            "source_artifact": layout_path.as_posix(),
            "source_boundary": boundary_source,
            **iteration_selection,
            "room_contract_match": "exact_modulo_origin_cycle_winding",
            "native_structure": payload.get("structure") or {},
        },
    )


def _world_aabb_size(
    rotation_matrix: list[list[float]],
    local_size: Sequence[float],
) -> list[float]:
    return [
        sum(
            abs(rotation_matrix[row][column]) * float(local_size[column])
            for column in range(3)
        )
        for row in range(3)
    ]


def _vectors_close(
    left: Sequence[float], right: Sequence[float], tolerance: float
) -> bool:
    return all(
        abs(float(left[index]) - float(right[index])) <= tolerance
        for index in range(3)
    )


def _select_layout(
    source_path: Path,
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source = Path(source_path)
    policy = str(
        config.get("iteration_selection_policy") or "explicit"
    ).strip()
    if policy not in ITERATION_SELECTION_POLICIES:
        raise ArtifactValidationError(
            "SceneWeaver iteration_selection_policy must be explicit or latest"
        )
    explicit_path = config.get("layout_path")
    selected_value = config.get("selected_iteration")
    if explicit_path is not None and selected_value is not None:
        raise ArtifactValidationError(
            "SceneWeaver layout_path and selected_iteration are mutually exclusive"
        )
    candidates = discover_layout_iterations(source)
    available_iterations = sorted(candidates)

    if source.is_file():
        if explicit_path is not None or selected_value is not None:
            raise ArtifactValidationError(
                "SceneWeaver source file cannot be combined with another layout selector"
            )
        path = source
        selection_policy = "source_file"
    elif explicit_path is not None:
        if policy != "explicit":
            raise ArtifactValidationError(
                "SceneWeaver layout_path requires explicit selection policy"
            )
        path = Path(str(explicit_path)).expanduser()
        if not path.is_absolute():
            path = source / path
        if not path.is_file():
            raise FileNotFoundError(f"SceneWeaver layout_path not found: {path}")
        selection_policy = "explicit_layout_path"
    elif selected_value is not None:
        if policy != "explicit":
            raise ArtifactValidationError(
                "SceneWeaver selected_iteration requires explicit selection policy"
            )
        selected_iteration = _iteration_number(selected_value)
        path = candidates.get(selected_iteration)
        if path is None:
            raise ArtifactValidationError(
                "SceneWeaver selected_iteration "
                f"{selected_iteration} is unavailable; available={available_iterations}"
            )
        selection_policy = "explicit_iteration"
    elif policy == "latest":
        if not candidates:
            raise ArtifactValidationError(
                "SceneWeaver output contains no layout_<iteration>.json"
            )
        path = candidates[max(candidates)]
        selection_policy = "latest_non_strict"
    else:
        raise ArtifactValidationError(
            "strict SceneWeaver conversion requires selected_iteration, "
            "layout_path, or a direct layout JSON file"
        )

    match = LAYOUT_PATTERN.match(path.name)
    selected_iteration = int(match.group(1)) if match else None
    return path, {
        "selected_iteration": selected_iteration,
        "available_iterations": available_iterations,
        "iteration_selection_policy": selection_policy,
    }


def discover_layout_iterations(source: Path) -> dict[int, Path]:
    """Return every native layout iteration without selecting or converting it."""

    if source.is_file():
        paths = sorted(source.parent.glob("layout_*.json"))
    elif source.is_dir():
        # Released SceneWeaver creates a scene-named directory below ``basedir``
        # and writes ``record_scene/layout_N.json`` inside it. Preserve the full
        # native root and discover that shape without flattening or relocating it.
        paths = sorted(source.rglob("layout_*.json"))
    else:
        raise FileNotFoundError(f"harness output does not exist: {source}")
    candidates: dict[int, Path] = {}
    for path in paths:
        match = LAYOUT_PATTERN.match(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        if iteration in candidates:
            raise ArtifactValidationError(
                "SceneWeaver output contains ambiguous duplicate iteration "
                f"{iteration}: {candidates[iteration]} and {path}"
            )
        candidates[iteration] = path
    return candidates


def _iteration_number(value: Any) -> int:
    if isinstance(value, bool):
        raise ArtifactValidationError(
            "SceneWeaver selected_iteration must be a non-negative integer"
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "SceneWeaver selected_iteration must be a non-negative integer"
        ) from exc
    if number < 0 or str(number) != str(value).strip():
        raise ArtifactValidationError(
            "SceneWeaver selected_iteration must be a non-negative integer"
        )
    return number


__all__ = ["convert_scene_weaver", "discover_layout_iterations"]
