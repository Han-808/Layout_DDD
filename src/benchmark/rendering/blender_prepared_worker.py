"""Render standardized evidence directly from a trusted prepared ``.blend``.

The source scene is loaded by Blender with auto-execution disabled.  This
worker validates its registered identities against the normalized scene, adds
only ephemeral benchmark cameras/lights, renders, and never saves the source.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import bpy
from mathutils import Euler, Vector


WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))
MATERIALIZATION_WORKER_DIR = WORKER_DIR.parent / "materialization"
if str(MATERIALIZATION_WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(MATERIALIZATION_WORKER_DIR))

from blender_worker import (  # noqa: E402
    DEFAULT_COLLISION_MAX_FACES_PER_OBJECT,
    DEFAULT_COLLISION_MAX_TOTAL_FACES,
    DEFAULT_COLLISION_MAX_TOTAL_VERTICES,
    DEFAULT_COLLISION_MAX_VERTICES_PER_OBJECT,
    _add_lighting,
    _collision_geometry_export_result,
    _collision_geometry_limits,
    _configure_render,
    _render_identity_map,
    _render_views,
    _source_architecture_contract,
    _write_collision_geometry_manifest,
)
from blend_inspector_worker import (  # noqa: E402
    _asset_assembly_fingerprint,
    _geometry_fingerprint,
    _is_render_capable_object,
    _material_fingerprint,
    _object_render_state_mismatches,
    _validate_sanitized_render_state,
    _validate_architecture_allowlist,
)


INSTANCE_ID_PROPERTY = "benchmark_instance_id"
EVALUATOR_ID_PROPERTY = "benchmark_evaluator_object_id"
CANONICAL_ID_PROPERTY = "benchmark_object_id"
ASSET_ID_PROPERTY = "benchmark_asset_id"
ROLE_PROPERTY = "benchmark_role"


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    normalized_path = Path(args.normalized_scene_json).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = _load_json(normalized_path)

    source_path = (
        Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
    )
    if source_path is None:
        raise RuntimeError("prepared render worker was not given a source blend")
    source_camera_names = sorted(
        obj.name for obj in bpy.data.objects if obj.type == "CAMERA"
    )
    source_light_names = sorted(
        obj.name for obj in bpy.data.objects if obj.type == "LIGHT"
    )
    if source_camera_names or source_light_names:
        raise RuntimeError(
            "trusted prepared blend contains non-ephemeral cameras or lights"
        )
    source_render_state = _validate_sanitized_render_state()
    if not source_render_state["passed"]:
        raise RuntimeError(
            "trusted prepared blend contains non-canonical scene render "
            "state: "
            + json.dumps(
                source_render_state["mismatches"],
                sort_keys=True,
                allow_nan=False,
            )
        )

    expected_objects = _expected_objects(normalized)
    roots = _registered_roots()
    if set(roots) != set(expected_objects):
        raise RuntimeError(
            "trusted blend identity set differs from normalized scene: "
            f"blend={sorted(roots)!r}, normalized={sorted(expected_objects)!r}"
        )
    objects = []
    registered_objects = set()
    for evaluator_id in sorted(expected_objects):
        expected = expected_objects[evaluator_id]
        root = roots[evaluator_id]
        expected_asset_id = str(
            ((expected.get("asset_ref") or {}).get("asset_key"))
            or expected.get("jid")
            or ""
        )
        observed_asset_id = str(root.get(ASSET_ID_PROPERTY) or "")
        if not expected_asset_id or observed_asset_id != expected_asset_id:
            raise RuntimeError(
                f"trusted blend asset identity mismatch for {evaluator_id!r}"
            )
        meshes = [
            obj for obj in {root, *_descendants(root)} if obj.type == "MESH"
        ]
        if not meshes:
            raise RuntimeError(
                f"trusted blend instance {evaluator_id!r} has no renderable mesh"
            )
        if any(_technically_hidden(obj) for obj in {root, *_descendants(root)}):
            raise RuntimeError(
                f"trusted blend instance {evaluator_id!r} is hidden or disabled"
            )
        descendants = {root, *_descendants(root)}
        if registered_objects & descendants:
            raise RuntimeError(
                f"trusted blend instance {evaluator_id!r} overlaps another "
                "registered instance hierarchy"
            )
        registered_objects.update(descendants)
        _validate_instance_geometry(
            evaluator_id=evaluator_id,
            root=root,
            meshes=meshes,
            expected=expected,
        )
        objects.append(
            {
                "id": evaluator_id,
                "instance_id": str(root.get(INSTANCE_ID_PROPERTY) or ""),
                "asset_id": observed_asset_id,
                "root_object_name": root.name,
                "root_object": root.name,
                "mesh_object_names": sorted(obj.name for obj in meshes),
                "representation": "asset_mesh",
                "mesh_path": None,
                "canonical_center": list(expected.get("center") or []),
                "canonical_size": list(expected.get("size") or []),
                "canonical_rotation_degrees": list(
                    expected.get("rotation") or []
                ),
                "rendered_bounds_center": list(expected.get("center") or []),
                "vertical_anchor": None,
                "vertical_anchor_source": "fixed_catalog_no_anchor",
            }
        )
    architecture = _source_architecture_contract()
    expected_architecture = (
        (normalized.get("metadata") or {}).get("architecture_contract")
        if isinstance(normalized.get("metadata"), dict)
        else None
    )
    if not isinstance(architecture, dict) or architecture != expected_architecture:
        raise RuntimeError(
            "trusted blend architecture differs from normalized scene"
        )
    boundary = _scene_boundary()
    scene_height = _scene_height()
    if boundary != normalized.get("boundary"):
        raise RuntimeError(
            "trusted blend room boundary differs from normalized scene"
        )
    if abs(scene_height - float(normalized.get("scene_height"))) > 1.0e-6:
        raise RuntimeError(
            "trusted blend scene height differs from normalized scene"
        )
    architecture_validation = _validate_architecture_allowlist(
        boundary=boundary,
        scene_height=scene_height,
        architecture=architecture,
        required=True,
    )
    if not architecture_validation["passed"]:
        raise RuntimeError(
            "trusted blend architecture differs from the materializer "
            "allowlist: "
            + json.dumps(
                architecture_validation["mismatches"],
                sort_keys=True,
                allow_nan=False,
            )
        )
    architecture_objects = {
        bpy.data.objects[name]
        for name in architecture_validation["allowed_object_names"]
        if bpy.data.objects.get(name) is not None
    }
    extras = sorted(
        f"{obj.name}:{obj.type}"
        for obj in bpy.data.objects
        if (
            (
                obj.type == "MESH"
                or (
                    _is_render_capable_object(obj)
                    and not _technically_hidden(obj)
                )
            )
            and obj not in registered_objects
            and obj not in architecture_objects
        )
    )
    if extras:
        raise RuntimeError(
            "trusted blend contains extra renderable objects outside the "
            f"instance and architecture allowlists: {extras}"
        )

    render_config = _configure_render(
        args.width,
        args.height,
        args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
    )
    _add_lighting(boundary, scene_height)
    views = _render_views(boundary, scene_height, out_dir)
    canonical_ids = sorted(expected_objects)
    identity_legend: dict[str, str] = {}
    identity_palette: dict[str, str] = {}
    identity_render = {
        "status": "not_applicable",
        "reason": "scene_has_no_renderable_objects",
        "canonical_object_count": 0,
        "scene_mutated": False,
    }
    if canonical_ids:
        identity_view, identity_legend, identity_palette = _render_identity_map(
            canonical_ids,
            out_dir,
        )
        views.append(identity_view)
        identity_render = {
            "status": "available",
            "camera_source": "standardized_perspective",
            "architecture_identity": "neutral_background",
            "canonical_object_count": len(canonical_ids),
            "scene_mutated": False,
            "color_encoding": "raw_linear_rgb_8bit",
        }
    geometry_manifest_path = _write_collision_geometry_manifest(
        out_dir,
        objects,
        max_vertices_per_object=args.collision_max_vertices_per_object,
        max_faces_per_object=args.collision_max_faces_per_object,
        max_total_vertices=args.collision_max_total_vertices,
        max_total_faces=args.collision_max_total_faces,
    )
    collision_geometry_export = _collision_geometry_export_result(
        geometry_manifest_path,
        limits=_collision_geometry_limits(args),
    )

    manifest = {
        "backend": "blender_prepared_scene_read_only_v1",
        "blender_version": bpy.app.version_string,
        "blend_file": source_path.as_posix(),
        "normalized_scene_path": normalized_path.as_posix(),
        "source_scene_saved": False,
        "scene_mutation_scope": "ephemeral_benchmark_camera_and_lighting_only",
        "render_engine": args.render_engine,
        "render_config": render_config,
        "views": views,
        "identity_legend": identity_legend,
        "identity_palette": identity_palette,
        "identity_render": identity_render,
        "objects": objects,
        "collision_geometry_manifest": (
            str(geometry_manifest_path)
            if geometry_manifest_path is not None
            else None
        ),
        "collision_geometry_export": collision_geometry_export,
        "architecture": architecture,
        "architecture_policy_version": architecture.get(
            "architecture_policy_version"
        ),
        "wall_policy": (
            (architecture.get("physical_walls") or {}).get("policy")
        ),
        "active_wall_ids": list(
            (architecture.get("physical_walls") or {}).get("active_wall_ids")
            or []
        ),
        "source_pre_render_state": {
            "camera_names": source_camera_names,
            "light_names": source_light_names,
        },
        "render_invocation_count": len(views),
    }
    (out_dir / "prepared_render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-scene-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument(
        "--render-engine",
        choices=["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"],
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument(
        "--cycles-device",
        choices=["CPU", "CUDA", "OPTIX", "AUTO"],
        default="CPU",
    )
    parser.add_argument("--cycles-samples", type=int, default=16)
    parser.add_argument("--cycles-denoising", action="store_true")
    parser.add_argument(
        "--collision-max-vertices-per-object",
        type=int,
        default=DEFAULT_COLLISION_MAX_VERTICES_PER_OBJECT,
    )
    parser.add_argument(
        "--collision-max-faces-per-object",
        type=int,
        default=DEFAULT_COLLISION_MAX_FACES_PER_OBJECT,
    )
    parser.add_argument(
        "--collision-max-total-vertices",
        type=int,
        default=DEFAULT_COLLISION_MAX_TOTAL_VERTICES,
    )
    parser.add_argument(
        "--collision-max-total-faces",
        type=int,
        default=DEFAULT_COLLISION_MAX_TOTAL_FACES,
    )
    return parser.parse_args(values)


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read normalized scene {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("normalized scene must contain a JSON object")
    return value


def _expected_objects(scene: dict) -> dict[str, dict]:
    values = scene.get("objects")
    if not isinstance(values, list):
        raise ValueError("normalized scene objects must be a JSON array")
    result = {}
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"normalized scene object {index} must be an object")
        object_id = str(item.get("id") or "").strip()
        if not object_id or object_id in result:
            raise ValueError(
                f"normalized scene contains invalid or duplicate object id {object_id!r}"
            )
        result[object_id] = item
    return result


def _registered_roots() -> dict[str, object]:
    result = {}
    for obj in bpy.data.objects:
        if obj.get(ROLE_PROPERTY) != "instance_root":
            continue
        evaluator_id = str(
            obj.get(EVALUATOR_ID_PROPERTY)
            or obj.get(CANONICAL_ID_PROPERTY)
            or ""
        ).strip()
        if not evaluator_id or evaluator_id in result:
            raise RuntimeError(
                f"trusted blend contains invalid or duplicate root id {evaluator_id!r}"
            )
        result[evaluator_id] = obj
    return result


def _scene_boundary() -> list[list[float]]:
    raw = bpy.context.scene.get("benchmark_request_boundary")
    try:
        value = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("trusted blend has invalid room boundary provenance") from exc
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(point, list) or len(point) != 2 for point in value)
    ):
        raise RuntimeError("trusted blend room boundary provenance is malformed")
    return [[float(component) for component in point] for point in value]


def _scene_height() -> float:
    try:
        value = float(bpy.context.scene.get("benchmark_scene_height"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("trusted blend has invalid scene height provenance") from exc
    if value <= 0.0:
        raise RuntimeError("trusted blend scene height must be positive")
    return value


def _descendants(root) -> list:
    result = []
    stack = list(root.children)
    while stack:
        obj = stack.pop()
        result.append(obj)
        stack.extend(obj.children)
    return result


def _technically_hidden(obj) -> bool:
    if obj.hide_render or obj.hide_viewport or obj.hide_get():
        return True
    try:
        if not obj.visible_get(view_layer=bpy.context.view_layer):
            return True
    except (RuntimeError, TypeError):
        return True
    return any(
        collection.hide_render or collection.hide_viewport
        for collection in obj.users_collection
    )


def _validate_instance_geometry(
    *,
    evaluator_id: str,
    root,
    meshes: list,
    expected: dict,
) -> None:
    descendants = {root, *_descendants(root)}
    disallowed_types = sorted(
        f"{obj.name}:{obj.type}"
        for obj in descendants
        if obj.type not in {"EMPTY", "MESH"}
    )
    if disallowed_types:
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} contains disallowed "
            f"object types: {disallowed_types}"
        )
    render_state_mismatches = _object_render_state_mismatches(descendants)
    if render_state_mismatches:
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} contains object render "
            "state overrides: "
            + json.dumps(
                render_state_mismatches,
                sort_keys=True,
                allow_nan=False,
            )
        )
    if root.parent is not None:
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} has a parented root"
        )
    materialization = (
        (expected.get("metadata") or {}).get("materialization")
        if isinstance(expected.get("metadata"), dict)
        else None
    )
    if not isinstance(materialization, dict):
        raise RuntimeError(
            f"normalized instance {evaluator_id!r} has no materialization record"
        )
    expected_geometry_sha256 = str(
        materialization.get("geometry_sha256") or ""
    ).lower()
    expected_material_sha256 = str(
        materialization.get("material_sha256") or ""
    ).lower()
    expected_asset_assembly_sha256 = str(
        materialization.get("asset_assembly_sha256") or ""
    ).lower()
    observed_geometry_sha256 = str(
        _geometry_fingerprint(root, meshes) or ""
    ).lower()
    observed_material_sha256 = str(
        _material_fingerprint(meshes) or ""
    ).lower()
    observed_asset_assembly_sha256 = str(
        _asset_assembly_fingerprint(root, meshes) or ""
    ).lower()
    if (
        not expected_geometry_sha256
        or observed_geometry_sha256 != expected_geometry_sha256
    ):
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} geometry fingerprint "
            "mismatch"
        )
    if (
        not expected_material_sha256
        or observed_material_sha256 != expected_material_sha256
    ):
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} material fingerprint "
            "mismatch"
        )
    if (
        not expected_asset_assembly_sha256
        or observed_asset_assembly_sha256
        != expected_asset_assembly_sha256
    ):
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} asset assembly "
            "fingerprint mismatch"
        )
    comparisons = {
        "instance_id": (
            root.get(INSTANCE_ID_PROPERTY),
            materialization.get("instance_id"),
        ),
        "center_m": (
            _id_property_value(root.get("benchmark_center_m")),
            expected.get("center"),
        ),
        "target_size_m": (
            _id_property_value(root.get("benchmark_target_size_m")),
            materialization.get("target_size_m"),
        ),
        "rotation_euler_xyz_deg": (
            _id_property_value(
                root.get("benchmark_rotation_euler_xyz_deg")
            ),
            expected.get("rotation"),
        ),
        "uniform_scale": (
            root.get("benchmark_uniform_scale"),
            materialization.get("uniform_scale"),
        ),
        "local_bbox_size_m": (
            _id_property_value(
                root.get("benchmark_local_bbox_size_m")
            ),
            expected.get("size"),
        ),
    }
    for field, (observed, wanted) in comparisons.items():
        if not _close_json(observed, wanted):
            raise RuntimeError(
                f"trusted blend {evaluator_id!r} {field} property mismatch"
            )

    matrix = root.matrix_world
    columns = [
        Vector((matrix[0][axis], matrix[1][axis], matrix[2][axis]))
        for axis in range(3)
    ]
    scales = [float(column.length) for column in columns]
    if (
        any(not math.isfinite(value) or value <= 1.0e-8 for value in scales)
        or max(scales) - min(scales) > max(1.0e-6, max(scales) * 1.0e-6)
        or float(matrix.to_3x3().determinant()) <= 0.0
    ):
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} is not rigid uniform-scale"
        )
    normalized_columns = [column / scales[index] for index, column in enumerate(columns)]
    if any(
        abs(normalized_columns[left].dot(normalized_columns[right])) > 1.0e-6
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} contains shear"
        )
    rotation = [
        math.radians(float(value))
        for value in expected.get("rotation", [])
    ]
    if len(rotation) != 3:
        raise RuntimeError(
            f"normalized instance {evaluator_id!r} has invalid rotation"
        )
    expected_rotation = Euler(rotation, "XYZ").to_matrix()
    observed_rotation = matrix.to_3x3().normalized()
    if any(
        abs(
            float(observed_rotation[row][column])
            - float(expected_rotation[row][column])
        )
        > 1.0e-5
        for row in range(3)
        for column in range(3)
    ):
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} rotation mismatch"
        )

    inverse = matrix.inverted()
    local_points = [
        inverse @ obj.matrix_world @ Vector(corner)
        for obj in meshes
        for corner in obj.bound_box
    ]
    minimum = Vector(
        [min(point[axis] for point in local_points) for axis in range(3)]
    )
    maximum = Vector(
        [max(point[axis] for point in local_points) for axis in range(3)]
    )
    local_center = (minimum + maximum) * 0.5
    local_size = (maximum - minimum) * (sum(scales) / 3.0)
    center_world = matrix @ local_center
    if not _close_json(
        [float(value) for value in center_world],
        expected.get("center"),
    ) or not _close_json(
        [float(value) for value in local_size],
        expected.get("size"),
    ):
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} actual bounds mismatch"
        )

    corners_world = []
    for signs in product((-1.0, 1.0), repeat=3):
        point = Vector(
            [
                local_center[axis]
                + signs[axis] * (maximum[axis] - minimum[axis]) * 0.5
                for axis in range(3)
            ]
        )
        corners_world.append([float(value) for value in matrix @ point])
    actual_aabb = {
        "min_m": [
            min(point[axis] for point in corners_world)
            for axis in range(3)
        ],
        "max_m": [
            max(point[axis] for point in corners_world)
            for axis in range(3)
        ],
    }
    expected_bounds = materialization.get("world_bounds")
    expected_aabb = (
        expected_bounds.get("aabb")
        if isinstance(expected_bounds, dict)
        else None
    )
    if not isinstance(expected_aabb, dict) or not (
        _close_json(actual_aabb["min_m"], expected_aabb.get("min_m"))
        and _close_json(actual_aabb["max_m"], expected_aabb.get("max_m"))
    ):
        raise RuntimeError(
            f"trusted blend instance {evaluator_id!r} world AABB mismatch"
        )
    for obj in descendants:
        if obj is not root and obj.get(ROLE_PROPERTY) != "asset_descendant":
            raise RuntimeError(
                f"trusted blend instance {evaluator_id!r} has an unregistered "
                f"descendant {obj.name!r}"
            )
        for property_name, expected_value in (
            (INSTANCE_ID_PROPERTY, materialization.get("instance_id")),
            (EVALUATOR_ID_PROPERTY, evaluator_id),
            (CANONICAL_ID_PROPERTY, evaluator_id),
            (
                ASSET_ID_PROPERTY,
                ((expected.get("asset_ref") or {}).get("asset_key")),
            ),
        ):
            if str(obj.get(property_name) or "") != str(
                expected_value or ""
            ):
                raise RuntimeError(
                    f"trusted blend instance {evaluator_id!r} descendant "
                    f"identity mismatch: {property_name}"
                )


def _id_property_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return [_id_property_value(item) for item in value]
    except TypeError:
        return str(value)


def _close_json(left, right, *, tolerance: float = 1.0e-5) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and abs(float(left) - float(right))
            <= max(
                tolerance,
                tolerance * max(abs(float(left)), abs(float(right))),
            )
        )
    if isinstance(left, (list, tuple)) and isinstance(
        right, (list, tuple)
    ):
        return len(left) == len(right) and all(
            _close_json(a, b, tolerance=tolerance)
            for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close_json(left[key], right[key], tolerance=tolerance)
            for key in left
        )
    return left == right


if __name__ == "__main__":
    main()
