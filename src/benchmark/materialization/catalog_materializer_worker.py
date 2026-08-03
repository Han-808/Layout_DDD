"""Blender-side build-only worker for fixed-catalog placement plans.

This worker is deliberately narrower than the legacy scene renderer:

* it starts from factory state and accepts benchmark-resolved mesh paths only;
* import failure is fatal and never falls back to a proxy;
* placement applies the generator-requested uniform scale exactly, with the
  scaled catalog bbox center mapped to ``center_m`` (there is no
  floor/ceiling anchoring);
* it writes a sanitized ``.blend`` and a construction report, but never calls
  Blender's render operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from itertools import product
from pathlib import Path

import bpy
from mathutils import Euler, Vector


RENDERING_DIR = Path(__file__).resolve().parents[1] / "rendering"
if str(RENDERING_DIR) not in sys.path:
    sys.path.insert(0, str(RENDERING_DIR))

from blender_worker import (  # noqa: E402
    ARCHITECTURE_CONTRACT_PROPERTY,
    _active_wall_ids,
    _build_room,
    _clear_scene,
    _root_location,
    _world_bounds,
)


PLAN_VERSION = "catalog_materialization_plan_v1"
INSTANCE_ID_PROPERTY = "benchmark_instance_id"
EVALUATOR_ID_PROPERTY = "benchmark_evaluator_object_id"
CANONICAL_ID_PROPERTY = "benchmark_object_id"
ASSET_ID_PROPERTY = "benchmark_asset_id"
ROLE_PROPERTY = "benchmark_role"
MESH_TOLERANCE_M = 1.0e-5
SHAPE_KEY_BAKE_TOLERANCE_M = 1.0e-8
SUPPORTED_MESH_SUFFIXES = {".fbx", ".glb", ".gltf", ".obj"}


def main() -> None:
    args = _parse_args()
    plan_path = Path(args.plan_json).expanduser().resolve()
    out_blend = Path(args.out_blend).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve()
    if not plan_path.is_file():
        raise ValueError(f"materialization plan does not exist: {plan_path}")
    if len({plan_path, out_blend, report_path}) != 3:
        raise ValueError("plan, blend output and report output paths must be distinct")
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    plan_hash_before = _sha256_file(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    request, instances = _validate_plan(plan)

    # Blender was already launched with --factory-startup.  Clearing the
    # startup cube/camera/light is nevertheless explicit and testable.
    _clear_scene()
    boundary = request["boundary"]
    scene_height = request["scene_height"]
    architecture = request["architecture"]
    before_architecture = set(bpy.data.objects)
    rendered_wall_ids = _build_room(
        boundary,
        scene_height,
        active_wall_ids=_active_wall_ids(architecture),
    )
    architecture_objects = [
        obj for obj in bpy.data.objects if obj not in before_architecture
    ]
    _move_to_collection(architecture_objects, "benchmark_architecture")
    for obj in architecture_objects:
        obj[ROLE_PROPERTY] = "architecture"
        if obj.get("benchmark_architecture_id") is None:
            obj["benchmark_architecture_id"] = "floor"

    scene = bpy.context.scene
    scene[ARCHITECTURE_CONTRACT_PROPERTY] = _canonical_json(architecture)
    scene["benchmark_materialization_plan_sha256"] = plan_hash_before
    scene["benchmark_materialization_plan_version"] = PLAN_VERSION
    scene["benchmark_materialization_revision"] = str(
        plan["materialization_revision"]
    )
    scene["benchmark_adapter_contract_revision"] = str(
        plan["adapter_contract_revision"]
    )
    scene["benchmark_catalog_snapshot_id"] = str(plan["catalog_snapshot_id"])
    scene["benchmark_request_id"] = str(request["request_id"])
    scene["benchmark_request_boundary"] = _canonical_json(boundary)
    scene["benchmark_scene_height"] = float(scene_height)
    factory_timing = {
        "fps": int(scene.render.fps),
        "fps_base": float(scene.render.fps_base),
    }

    built_instances = [
        _build_instance(item, plan=plan)
        for item in instances
    ]
    scene_normalizations = _restore_factory_scene_timing(
        scene,
        expected=factory_timing,
    )
    _pack_external_images()
    _assert_sanitized_build_state()
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))

    plan_hash_after = _sha256_file(plan_path)
    if plan_hash_after != plan_hash_before:
        raise RuntimeError("materialization plan changed while Blender was building")
    report = {
        "backend": "fixed_catalog_blender_materializer_v1",
        "status": "built",
        "blender_version": bpy.app.version_string,
        "plan_path": plan_path.as_posix(),
        "plan_sha256_before": plan_hash_before,
        "plan_sha256_after": plan_hash_after,
        "out_blend": out_blend.as_posix(),
        "out_blend_sha256": _sha256_file(out_blend),
        "render_invocation_count": 0,
        "placement": {
            "scale_mode": "exact_uniform_scale",
            "requested_scale_equals_effective_scale": True,
            "vertical_anchor": None,
            "rotation_semantics": "intrinsic_xyz_degrees_Rz_Ry_Rx",
            "center_semantics": "scaled_catalog_local_bbox_center_in_world",
        },
        "catalog_snapshot_id": str(plan["catalog_snapshot_id"]),
        "materialization_revision": str(plan["materialization_revision"]),
        "adapter_contract_revision": str(plan["adapter_contract_revision"]),
        "architecture": architecture,
        "rendered_wall_ids": rendered_wall_ids,
        "instances": built_instances,
        "scene_normalizations": scene_normalizations,
        "source_scene_saved": True,
        "source_scene_kind": "benchmark_owned_sanitized_output",
    }
    _write_json(report_path, report)


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--out-blend", required=True)
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(values)


def _validate_plan(plan: object) -> tuple[dict, list[dict]]:
    if not isinstance(plan, dict):
        raise ValueError("materialization plan must be a JSON object")
    if plan.get("schema_version") != PLAN_VERSION:
        raise ValueError(
            f"materialization plan schema_version must be {PLAN_VERSION!r}"
        )
    for key in (
        "materialization_revision",
        "adapter_contract_revision",
        "catalog_snapshot_id",
    ):
        if not isinstance(plan.get(key), str) or not str(plan[key]).strip():
            raise ValueError(f"materialization plan {key} must be a non-empty string")
    request = plan.get("request")
    if not isinstance(request, dict):
        raise ValueError("materialization plan request must be a JSON object")
    for key in ("request_id", "scene_type"):
        if not isinstance(request.get(key), str) or not str(request[key]).strip():
            raise ValueError(f"materialization plan request.{key} must be non-empty")
    boundary = request.get("boundary")
    if not isinstance(boundary, list) or len(boundary) != 4:
        raise ValueError("materialization plan request.boundary must have four points")
    request["boundary"] = [
        _finite_vector(point, 2, f"request.boundary[{index}]")
        for index, point in enumerate(boundary)
    ]
    scene_height = _finite_number(
        request.get("scene_height"),
        "request.scene_height",
        positive=True,
    )
    request["scene_height"] = scene_height
    architecture = request.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("materialization plan request.architecture must be an object")
    logical = architecture.get("logical_boundary")
    if not isinstance(logical, dict) or logical.get("boundary") != request["boundary"]:
        raise ValueError(
            "architecture logical boundary must exactly match request.boundary"
        )
    floor_z = _finite_number(
        architecture.get("floor_z", 0.0),
        "request.architecture.floor_z",
    )
    if abs(floor_z) > MESH_TOLERANCE_M:
        raise ValueError("catalog placement room-min origin requires floor_z=0")
    _active_wall_ids(architecture)

    instances = plan.get("instances")
    if not isinstance(instances, list):
        raise ValueError("materialization plan instances must be a JSON array")
    validated = [
        _validate_instance(item, index=index)
        for index, item in enumerate(instances)
    ]
    instance_ids = [item["instance_id"] for item in validated]
    evaluator_ids = [item["evaluator_object_id"] for item in validated]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("materialization plan contains duplicate instance_id values")
    if len(evaluator_ids) != len(set(evaluator_ids)):
        raise ValueError(
            "materialization plan contains duplicate evaluator_object_id values"
        )
    if instance_ids != sorted(instance_ids):
        raise ValueError("materialization plan instances must be sorted by instance_id")
    return request, validated


def _validate_instance(item: object, *, index: int) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"materialization plan instances[{index}] must be an object")
    result = dict(item)
    for key in ("instance_id", "evaluator_object_id", "asset_id", "mesh_path"):
        if not isinstance(result.get(key), str) or not str(result[key]).strip():
            raise ValueError(f"instances[{index}].{key} must be a non-empty string")
    slot_id = result.get("slot_id")
    if slot_id is not None and (
        not isinstance(slot_id, str) or not slot_id.strip()
    ):
        raise ValueError(f"instances[{index}].slot_id must be null or non-empty")
    result["center_m"] = _finite_vector(
        result.get("center_m"), 3, f"instances[{index}].center_m"
    )
    result["rotation_euler_xyz_deg"] = _finite_vector(
        result.get("rotation_euler_xyz_deg"),
        3,
        f"instances[{index}].rotation_euler_xyz_deg",
    )
    result["catalog_bbox_center_m"] = _finite_vector(
        result.get("catalog_bbox_center_m"),
        3,
        f"instances[{index}].catalog_bbox_center_m",
    )
    result["catalog_bbox_size_m"] = _finite_vector(
        result.get("catalog_bbox_size_m"),
        3,
        f"instances[{index}].catalog_bbox_size_m",
        positive=True,
    )
    result["actual_local_bbox_size_m"] = _finite_vector(
        result.get("actual_local_bbox_size_m"),
        3,
        f"instances[{index}].actual_local_bbox_size_m",
        positive=True,
    )
    result["local_bbox_size_m"] = _finite_vector(
        result.get("local_bbox_size_m"),
        3,
        f"instances[{index}].local_bbox_size_m",
        positive=True,
    )
    result["requested_uniform_scale"] = _finite_number(
        result.get("requested_uniform_scale"),
        f"instances[{index}].requested_uniform_scale",
        positive=True,
    )
    result["effective_uniform_scale"] = _finite_number(
        result.get("effective_uniform_scale"),
        f"instances[{index}].effective_uniform_scale",
        positive=True,
    )
    result["uniform_scale"] = _finite_number(
        result.get("uniform_scale"),
        f"instances[{index}].uniform_scale",
        positive=True,
    )
    if not _close(
        result["requested_uniform_scale"],
        result["effective_uniform_scale"],
    ):
        raise ValueError(
            f"instances[{index}].effective_uniform_scale must exactly follow "
            "requested_uniform_scale"
        )
    if not _close(
        result["uniform_scale"], result["effective_uniform_scale"]
    ):
        raise ValueError(
            f"instances[{index}].uniform_scale disagrees with effective scale"
        )
    expected_local_size = [
        component * result["effective_uniform_scale"]
        for component in result["catalog_bbox_size_m"]
    ]
    if not _close_vector(
        result["actual_local_bbox_size_m"], expected_local_size
    ):
        raise ValueError(
            f"instances[{index}].actual_local_bbox_size_m disagrees with exact "
            "uniform scale"
        )
    if not _close_vector(
        result["local_bbox_size_m"], result["actual_local_bbox_size_m"]
    ):
        raise ValueError(
            f"instances[{index}].local_bbox_size_m disagrees with actual local bbox"
        )
    expected_bounds = _expected_world_bounds(
        result["center_m"],
        result["local_bbox_size_m"],
        result["rotation_euler_xyz_deg"],
    )
    if not isinstance(result.get("world_bounds"), dict) or not _close_json(
        result["world_bounds"],
        expected_bounds,
    ):
        raise ValueError(
            f"instances[{index}].world_bounds disagrees with fixed transform semantics"
        )
    hashes = result.get("asset_hashes")
    if not isinstance(hashes, dict):
        raise ValueError(f"instances[{index}].asset_hashes must be an object")
    mesh_hash = hashes.get("mesh_sha256")
    if not isinstance(mesh_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", mesh_hash):
        raise ValueError(
            f"instances[{index}].asset_hashes.mesh_sha256 must be a SHA-256 digest"
        )
    asset_tree_hash = hashes.get("asset_tree_sha256")
    if not isinstance(asset_tree_hash, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", asset_tree_hash
    ):
        raise ValueError(
            f"instances[{index}].asset_hashes.asset_tree_sha256 must be a "
            "SHA-256 digest"
        )
    mesh_path = Path(result["mesh_path"]).expanduser().resolve()
    if not mesh_path.is_file():
        raise ValueError(f"instances[{index}].mesh_path does not exist: {mesh_path}")
    if mesh_path.suffix.lower() not in SUPPORTED_MESH_SUFFIXES:
        raise ValueError(
            f"instances[{index}].mesh_path uses an unsupported rigid mesh format"
        )
    result["mesh_path"] = mesh_path.as_posix()
    return result


def _build_instance(item: dict, *, plan: dict) -> dict:
    path = Path(item["mesh_path"])
    asset_dir = path.parent.resolve()
    source_hash_before = _sha256_file(path)
    expected_hash = str(item["asset_hashes"]["mesh_sha256"]).lower()
    if source_hash_before != expected_hash:
        raise RuntimeError(
            f"frozen mesh hash mismatch for asset {item['asset_id']!r}"
        )
    asset_tree_hash_before = _sha256_asset_tree(asset_dir)
    expected_tree_hash = str(
        item["asset_hashes"]["asset_tree_sha256"]
    ).lower()
    if asset_tree_hash_before != expected_tree_hash:
        raise RuntimeError(
            f"frozen asset dependency hash mismatch for asset "
            f"{item['asset_id']!r}"
        )

    try:
        imported, meshes, asset_normalizations = _strict_import(
            path,
            asset_dir=asset_dir,
        )
    finally:
        source_hash_after = _sha256_file(path)
        asset_tree_hash_after = _sha256_asset_tree(asset_dir)
        if source_hash_after != source_hash_before:
            raise RuntimeError(
                f"Blender import modified frozen mesh source {path}"
            )
        if asset_tree_hash_after != asset_tree_hash_before:
            raise RuntimeError(
                f"Blender import modified frozen asset dependencies under "
                f"{asset_dir}"
            )
    minimum, maximum = _world_bounds(meshes)
    source_center = [float(value) for value in (minimum + maximum) * 0.5]
    source_size = [float(value) for value in maximum - minimum]
    if not _close_vector(source_center, item["catalog_bbox_center_m"]):
        raise RuntimeError(
            f"imported bbox center disagrees with frozen catalog for "
            f"{item['asset_id']!r}: observed={source_center!r}, "
            f"expected={item['catalog_bbox_center_m']!r}"
        )
    if not _close_vector(source_size, item["catalog_bbox_size_m"]):
        raise RuntimeError(
            f"imported bbox size disagrees with frozen catalog for "
            f"{item['asset_id']!r}: observed={source_size!r}, "
            f"expected={item['catalog_bbox_size_m']!r}"
        )

    instance_collection = _collection("benchmark_instances")
    for obj in imported:
        _link_only(obj, instance_collection)
    root = bpy.data.objects.new(
        f"benchmark_instance_{_safe_name(item['instance_id'])}",
        None,
    )
    instance_collection.objects.link(root)
    imported_set = set(imported)
    for obj in imported:
        if obj.parent not in imported_set:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world

    rotation_radians = [
        math.radians(value)
        for value in item["rotation_euler_xyz_deg"]
    ]
    rotation_matrix = Euler(rotation_radians, "XYZ").to_matrix()
    scale = float(item["uniform_scale"])
    root.rotation_mode = "XYZ"
    root.rotation_euler = rotation_radians
    root.scale = (scale, scale, scale)
    # The frozen catalog center, not an observed/adapted center and not a
    # vertical anchor, defines placement.
    root.location = Vector(
        _root_location(
            item["catalog_bbox_center_m"],
            scale,
            rotation_matrix,
            item["center_m"],
        )
    )
    _stamp_root(root, item, plan=plan)
    for obj in imported:
        _stamp_descendant(obj, item)
    bpy.context.view_layer.update()

    return {
        "instance_id": item["instance_id"],
        "evaluator_object_id": item["evaluator_object_id"],
        "asset_id": item["asset_id"],
        "slot_id": item.get("slot_id"),
        "root_object_name": root.name,
        "mesh_object_names": sorted(obj.name for obj in meshes),
        "mesh_path": path.as_posix(),
        "mesh_sha256_before": source_hash_before,
        "mesh_sha256_after": source_hash_after,
        "asset_tree_sha256_before": asset_tree_hash_before,
        "asset_tree_sha256_after": asset_tree_hash_after,
        "asset_normalizations": asset_normalizations,
        "catalog_bbox_center_m_expected": item["catalog_bbox_center_m"],
        "catalog_bbox_center_m_imported": source_center,
        "catalog_bbox_size_m_expected": item["catalog_bbox_size_m"],
        "catalog_bbox_size_m_imported": source_size,
        "center_m": item["center_m"],
        "requested_uniform_scale": item["requested_uniform_scale"],
        "effective_uniform_scale": item["effective_uniform_scale"],
        "actual_local_bbox_size_m": item["actual_local_bbox_size_m"],
        "local_bbox_size_m": item["local_bbox_size_m"],
        "rotation_euler_xyz_deg": item["rotation_euler_xyz_deg"],
        "uniform_scale": scale,
        "root_location": [float(value) for value in root.location],
        "root_matrix_world": _matrix_rows(root.matrix_world),
        "vertical_anchor": None,
        "representation": "fixed_catalog_asset_mesh",
    }


def _strict_import(
    path: Path,
    *,
    asset_dir: Path,
) -> tuple[list, list, list[dict]]:
    before = set(bpy.data.objects)
    before_images = set(bpy.data.images)
    before_materials = set(bpy.data.materials)
    _validate_text_dependency_paths(path, asset_dir=asset_dir)
    suffix = path.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:  # guarded by plan validation, retained as a fail-closed boundary.
        raise RuntimeError(f"unsupported frozen mesh format: {suffix}")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"asset import produced no Blender objects: {path}")
    disallowed = [
        f"{obj.name}:{obj.type}"
        for obj in imported
        if obj.type not in {"MESH", "EMPTY"}
    ]
    if disallowed:
        raise RuntimeError(
            "frozen rigid asset import produced disallowed object types: "
            + ", ".join(sorted(disallowed))
        )
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"asset import produced no mesh objects: {path}")
    for obj in imported:
        if obj.library is not None or (obj.data is not None and obj.data.library is not None):
            raise RuntimeError("frozen catalog import produced linked library data")
        if obj.constraints:
            raise RuntimeError(
                f"frozen rigid asset {path.name} contains object constraints"
            )
        if obj.modifiers:
            raise RuntimeError(
                f"frozen rigid asset {path.name} contains object modifiers"
            )
        if obj.animation_data is not None:
            raise RuntimeError(
                f"frozen rigid asset {path.name} contains object animation"
            )
    normalizations = _bake_static_shape_key_mixes(
        meshes,
        asset_name=path.name,
    )
    for mesh in meshes:
        if len(mesh.data.vertices) == 0:
            raise RuntimeError(
                f"frozen rigid asset {path.name} contains an empty mesh"
            )
    normalizations.extend(
        _prune_zero_polygon_material_slots(
            meshes,
            imported_materials=set(bpy.data.materials) - before_materials,
            imported_images=set(bpy.data.images) - before_images,
        )
    )
    for image in list(set(bpy.data.images) - before_images):
        if (
            image.source != "FILE"
            or not image.filepath
            or bool(image.packed_files)
        ):
            continue
        image_path = Path(bpy.path.abspath(image.filepath)).expanduser().resolve()
        try:
            image_path.relative_to(asset_dir)
        except ValueError as exc:
            raise RuntimeError(
                "frozen catalog asset references an external image outside its "
                f"hashed dependency tree: {image_path}"
            ) from exc
        if not image_path.is_file():
            normalizations.append(
                _replace_missing_image_with_diagnostic_fallback(
                    image,
                    missing_path=image_path,
                )
            )
    return imported, meshes, normalizations


def _bake_static_shape_key_mixes(
    meshes: list,
    *,
    asset_name: str,
) -> list[dict]:
    """Bake a fixed current shape-key mix without changing visible geometry."""

    records: list[dict] = []
    for obj in meshes:
        shape_keys = getattr(obj.data, "shape_keys", None)
        if shape_keys is None:
            continue
        animation = getattr(shape_keys, "animation_data", None)
        if (
            animation is not None
            and (
                getattr(animation, "action", None) is not None
                or tuple(getattr(animation, "drivers", ()) or ())
                or tuple(getattr(animation, "nla_tracks", ()) or ())
            )
        ):
            raise RuntimeError(
                f"frozen rigid asset {asset_name} contains animated shape keys"
            )
        before = _evaluated_local_vertex_coordinates(obj)
        key_state = [
            {
                "name": str(key.name),
                "value": float(key.value),
                "mute": bool(key.mute),
            }
            for key in shape_keys.key_blocks
        ]
        with bpy.context.temp_override(
            object=obj,
            active_object=obj,
            selected_objects=[obj],
            selected_editable_objects=[obj],
        ):
            bpy.ops.object.shape_key_remove(
                all=True,
                apply_mix=True,
            )
        bpy.context.view_layer.update()
        after = _evaluated_local_vertex_coordinates(obj)
        if len(before) != len(after):
            raise RuntimeError(
                f"shape-key bake changed vertex count for {asset_name}"
            )
        max_delta = max(
            (
                (Vector(left) - Vector(right)).length
                for left, right in zip(before, after)
            ),
            default=0.0,
        )
        if not math.isfinite(max_delta) or (
            max_delta > SHAPE_KEY_BAKE_TOLERANCE_M
        ):
            raise RuntimeError(
                f"shape-key bake changed visible geometry for {asset_name}: "
                f"max_delta_m={max_delta!r}"
            )
        records.append(
            {
                "type": "static_shape_key_mix_baked",
                "object_name": obj.name,
                "shape_keys": key_state,
                "vertex_count": len(after),
                "max_vertex_delta_m": float(max_delta),
                "source_modified": False,
            }
        )
    return records


def _evaluated_local_vertex_coordinates(obj) -> list[tuple[float, float, float]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(
        preserve_all_data_layers=False,
        depsgraph=depsgraph,
    )
    try:
        return [
            (float(vertex.co.x), float(vertex.co.y), float(vertex.co.z))
            for vertex in mesh.vertices
        ]
    finally:
        evaluated.to_mesh_clear()


def _prune_zero_polygon_material_slots(
    meshes: list,
    *,
    imported_materials: set,
    imported_images: set,
) -> list[dict]:
    """Remove imported material slots that cannot affect any polygon."""

    records: list[dict] = []
    for obj in meshes:
        used_indices = {
            int(polygon.material_index)
            for polygon in obj.data.polygons
        }
        for index in range(len(obj.data.materials) - 1, -1, -1):
            if index in used_indices:
                continue
            material = obj.data.materials[index]
            records.append(
                {
                    "type": "zero_polygon_material_slot_pruned",
                    "object_name": obj.name,
                    "slot_index": index,
                    "material_name": (
                        str(material.name) if material is not None else None
                    ),
                    "source_modified": False,
                }
            )
            obj.data.materials.pop(index=index)
    for material in sorted(
        imported_materials,
        key=lambda value: value.name,
    ):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in sorted(
        imported_images,
        key=lambda value: value.name,
    ):
        if image.users == 0:
            bpy.data.images.remove(image)
    return records


def _replace_missing_image_with_diagnostic_fallback(
    image,
    *,
    missing_path: Path,
) -> dict:
    """Freeze Blender's missing-image magenta diagnostic into the blend."""

    original_name = str(image.name)
    original_colorspace = str(image.colorspace_settings.name)
    original_alpha_mode = str(image.alpha_mode)
    image.name = f"{original_name}__unresolved"
    replacement = bpy.data.images.new(
        name=original_name,
        width=1,
        height=1,
        alpha=True,
        float_buffer=False,
    )
    try:
        replacement.colorspace_settings.name = original_colorspace
        replacement.alpha_mode = original_alpha_mode
        replacement.pixels[:] = (1.0, 0.0, 1.0, 1.0)
        replacement.pack()
    except (TypeError, ValueError, RuntimeError) as exc:
        bpy.data.images.remove(replacement)
        raise RuntimeError(
            "cannot preserve missing-image state while freezing "
            f"diagnostic fallback for {missing_path}"
        ) from exc

    replaced_references = 0
    node_trees = []
    for owner in (
        *tuple(bpy.data.materials),
        *tuple(bpy.data.worlds),
        *tuple(bpy.data.node_groups),
    ):
        tree = getattr(owner, "node_tree", None)
        if tree is not None:
            node_trees.append(tree)
    seen_trees = set()
    for tree in node_trees:
        pointer = int(tree.as_pointer())
        if pointer in seen_trees:
            continue
        seen_trees.add(pointer)
        for node in tree.nodes:
            if getattr(node, "image", None) == image:
                node.image = replacement
                node.id_data.update_tag()
                replaced_references += 1
    if replaced_references == 0:
        bpy.data.images.remove(replacement)
        raise RuntimeError(
            "missing frozen catalog image has no verifiable shader reference: "
            f"{missing_path}"
        )
    if image.users != 0:
        bpy.data.images.remove(replacement)
        raise RuntimeError(
            "cannot fully replace missing frozen catalog image dependency: "
            f"{missing_path}; remaining_users={image.users}"
        )
    bpy.data.images.remove(image)
    bpy.context.view_layer.update()
    return {
        "type": "missing_image_diagnostic_fallback_packed",
        "image_name": original_name,
        "missing_path": missing_path.as_posix(),
        "replacement_rgba_8bit": [255, 0, 255, 255],
        "colorspace": original_colorspace,
        "alpha_mode": original_alpha_mode,
        "shader_reference_count": replaced_references,
        "packed": bool(replacement.packed_files),
        "semantic": "preserve_blender_missing_image_diagnostic",
        "source_modified": False,
    }


def _validate_text_dependency_paths(path: Path, *, asset_dir: Path) -> None:
    """Reject external OBJ/MTL/glTF dependencies before Blender reads them."""

    suffix = path.suffix.lower()
    referenced: list[Path] = []
    if suffix == ".obj":
        for raw_line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = raw_line.strip()
            if line.lower().startswith("mtllib "):
                referenced.append(
                    (path.parent / line.split(maxsplit=1)[1]).resolve()
                )
    elif suffix == ".gltf":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for collection in ("buffers", "images"):
            for item in payload.get(collection, []):
                uri = item.get("uri") if isinstance(item, dict) else None
                if not isinstance(uri, str) or uri.startswith("data:"):
                    continue
                referenced.append((path.parent / uri).resolve())
    for dependency in referenced:
        try:
            dependency.relative_to(asset_dir)
        except ValueError as exc:
            raise RuntimeError(
                "frozen catalog asset references a dependency outside its "
                f"hashed asset tree: {dependency}"
            ) from exc
        if not dependency.is_file():
            raise RuntimeError(
                f"frozen catalog dependency does not exist: {dependency}"
            )


def _stamp_root(root, item: dict, *, plan: dict) -> None:
    root[ROLE_PROPERTY] = "instance_root"
    root[INSTANCE_ID_PROPERTY] = item["instance_id"]
    root[EVALUATOR_ID_PROPERTY] = item["evaluator_object_id"]
    root[CANONICAL_ID_PROPERTY] = item["evaluator_object_id"]
    root[ASSET_ID_PROPERTY] = item["asset_id"]
    if item.get("slot_id") is not None:
        root["benchmark_slot_id"] = item["slot_id"]
    root["benchmark_catalog_snapshot_id"] = str(plan["catalog_snapshot_id"])
    root["benchmark_materialization_revision"] = str(
        plan["materialization_revision"]
    )
    root["benchmark_adapter_contract_revision"] = str(
        plan["adapter_contract_revision"]
    )
    root["benchmark_center_m"] = item["center_m"]
    root["benchmark_requested_uniform_scale"] = float(
        item["requested_uniform_scale"]
    )
    root["benchmark_effective_uniform_scale"] = float(
        item["effective_uniform_scale"]
    )
    root["benchmark_rotation_euler_xyz_deg"] = item["rotation_euler_xyz_deg"]
    root["benchmark_uniform_scale"] = float(item["uniform_scale"])
    root["benchmark_catalog_bbox_center_m"] = item["catalog_bbox_center_m"]
    root["benchmark_catalog_bbox_size_m"] = item["catalog_bbox_size_m"]
    root["benchmark_local_bbox_size_m"] = item["local_bbox_size_m"]
    root["benchmark_actual_local_bbox_size_m"] = item[
        "actual_local_bbox_size_m"
    ]
    root["benchmark_mesh_sha256"] = str(
        item["asset_hashes"]["mesh_sha256"]
    ).lower()
    root["benchmark_asset_tree_sha256"] = str(
        item["asset_hashes"]["asset_tree_sha256"]
    ).lower()


def _stamp_descendant(obj, item: dict) -> None:
    obj[ROLE_PROPERTY] = "asset_descendant"
    obj[INSTANCE_ID_PROPERTY] = item["instance_id"]
    obj[EVALUATOR_ID_PROPERTY] = item["evaluator_object_id"]
    obj[CANONICAL_ID_PROPERTY] = item["evaluator_object_id"]
    obj[ASSET_ID_PROPERTY] = item["asset_id"]


def _pack_external_images() -> None:
    external = [
        image
        for image in bpy.data.images
        if image.source == "FILE" and not image.packed_files
    ]
    if external:
        bpy.ops.file.pack_all()
    remaining = [
        image.name
        for image in bpy.data.images
        if image.source == "FILE" and not image.packed_files
    ]
    if remaining:
        raise RuntimeError(
            "sanitized blend contains unpacked external images: "
            + ", ".join(sorted(remaining))
        )


def _restore_factory_scene_timing(
    scene,
    *,
    expected: dict[str, float | int],
) -> list[dict]:
    """Undo importer-owned timing side effects before freezing the scene."""

    normalizations = []
    for field in ("fps", "fps_base"):
        observed = getattr(scene.render, field)
        wanted = expected[field]
        if observed == wanted:
            continue
        normalizations.append(
            {
                "type": "factory_render_timing_restored",
                "field": f"scene.render.{field}",
                "imported_value": observed,
                "restored_value": wanted,
                "source_modified": False,
            }
        )
        setattr(scene.render, field, wanted)
    return normalizations


def _assert_sanitized_build_state() -> None:
    if bpy.data.cameras or bpy.data.lights:
        raise RuntimeError("build-only sanitized scene must not contain cameras or lights")
    if bpy.data.libraries:
        raise RuntimeError("sanitized scene must not contain linked libraries")
    if bpy.data.actions:
        raise RuntimeError("sanitized rigid scene must not contain actions")
    for scene in bpy.data.scenes:
        sequence_editor = scene.sequence_editor
        sequence_strips = (
            getattr(sequence_editor, "sequences_all", None)
            if sequence_editor is not None
            else None
        )
        if sequence_editor is not None and sequence_strips is None:
            sequence_strips = getattr(sequence_editor, "strips", ())
        if sequence_editor is not None and sequence_strips:
            raise RuntimeError("sanitized scene must not contain sequencer strips")
        compositor = getattr(scene, "node_tree", None)
        if compositor is None:
            compositor = getattr(scene, "compositing_node_group", None)
        if bool(getattr(scene, "use_nodes", False)) and compositor and compositor.nodes:
            raise RuntimeError("sanitized scene must not contain compositor nodes")


def _move_to_collection(objects: list, name: str) -> None:
    collection = _collection(name)
    for obj in objects:
        _link_only(obj, collection)


def _collection(name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def _link_only(obj, collection) -> None:
    if collection not in obj.users_collection:
        collection.objects.link(obj)
    for existing in list(obj.users_collection):
        if existing != collection:
            existing.objects.unlink(obj)


def _expected_world_bounds(
    center: list[float],
    local_size: list[float],
    rotation_degrees: list[float],
) -> dict:
    rotation = _rotation_matrix(rotation_degrees)
    half = [value * 0.5 for value in local_size]
    corners = []
    for signs in product((-1.0, 1.0), repeat=3):
        local = [signs[index] * half[index] for index in range(3)]
        rotated = [
            sum(rotation[row][column] * local[column] for column in range(3))
            for row in range(3)
        ]
        corners.append(
            [center[index] + rotated[index] for index in range(3)]
        )
    minimum = [min(corner[index] for corner in corners) for index in range(3)]
    maximum = [max(corner[index] for corner in corners) for index in range(3)]
    return {
        "obb": {
            "center_m": center,
            "local_size_m": local_size,
            "rotation_euler_xyz_deg": rotation_degrees,
            "rotation_matrix": rotation,
            "corners_m": corners,
        },
        "aabb": {
            "min_m": minimum,
            "max_m": maximum,
            "size_m": [
                maximum[index] - minimum[index] for index in range(3)
            ],
        },
    }


def _rotation_matrix(rotation_degrees: list[float]) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in Euler(
            [math.radians(value) for value in rotation_degrees],
            "XYZ",
        ).to_matrix()
    ]


def _finite_vector(
    value: object,
    length: int,
    label: str,
    *,
    positive: bool = False,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must be a {length}-component vector")
    return [
        _finite_number(component, f"{label}[{index}]", positive=positive)
        for index, component in enumerate(value)
    ]


def _finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{label} must be greater than zero")
    return number


def _close(left: float, right: float) -> bool:
    tolerance = max(
        MESH_TOLERANCE_M,
        MESH_TOLERANCE_M * max(abs(float(left)), abs(float(right))),
    )
    return abs(float(left) - float(right)) <= tolerance


def _close_vector(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        _close(a, b) for a, b in zip(left, right)
    )


def _close_json(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and _close(
            float(left),
            float(right),
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close_json(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close_json(left[key], right[key]) for key in left
        )
    return left == right


def _matrix_rows(matrix) -> list[list[float]]:
    return [
        [float(matrix[row][column]) for column in range(4)]
        for row in range(4)
    ]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "instance"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_asset_tree(asset_dir: Path) -> str:
    root = asset_dir.expanduser().resolve()
    entries = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise RuntimeError(
                f"frozen asset dependency tree contains a symbolic link: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(
                f"frozen asset dependency tree contains a non-regular file: {path}"
            )
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not entries:
        raise RuntimeError(f"frozen asset dependency tree is empty: {root}")
    return hashlib.sha256(
        json.dumps(
            entries,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    main()
