"""Blender build-only worker for one non-rectangular authoritative room."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy


MATERIALIZATION_DIR = Path(__file__).resolve().parents[1] / "materialization"
RENDERING_DIR = Path(__file__).resolve().parents[1] / "rendering"
for directory in (MATERIALIZATION_DIR, RENDERING_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import catalog_materializer_worker as base  # noqa: E402
from blender_worker import (  # noqa: E402
    ARCHITECTURE_CONTRACT_PROPERTY,
    _clear_scene,
    _material,
)
from saved_blend_view import configure_textured_inspection_view  # noqa: E402


PLAN_VERSION = "non_rectangular_catalog_materialization_plan_v1"
ROLE_PROPERTY = "benchmark_role"
ARCHITECTURE_ID_PROPERTY = "benchmark_architecture_id"
TOLERANCE = 1.0e-6


def main() -> None:
    args = _parse_args()
    plan_path = Path(args.plan_json).expanduser().resolve()
    out_blend = Path(args.out_blend).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve()
    if not plan_path.is_file():
        raise ValueError(f"materialization plan does not exist: {plan_path}")
    plan_hash_before = base._sha256_file(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    request, instances = _validate_plan(plan)

    _clear_scene()
    architecture_objects, rendered_wall_ids = _build_architecture(
        request["architecture"]
    )
    base._move_to_collection(architecture_objects, "benchmark_architecture")
    for obj in architecture_objects:
        obj[ROLE_PROPERTY] = "architecture"

    scene = bpy.context.scene
    scene[ARCHITECTURE_CONTRACT_PROPERTY] = base._canonical_json(
        request["architecture"]
    )
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
    scene["benchmark_request_boundary"] = base._canonical_json(
        request["boundary"]
    )
    scene["benchmark_scene_height"] = float(request["scene_height"])
    scene["benchmark_nonrect_room_id"] = str(request["room_id"])
    scene["benchmark_nonrect_global_coordinates"] = True
    scene["benchmark_nonrect_adjacent_room_objects"] = False
    factory_timing = {
        "fps": int(scene.render.fps),
        "fps_base": float(scene.render.fps_base),
    }

    built_instances = [base._build_instance(item, plan=plan) for item in instances]
    scene_normalizations = base._restore_factory_scene_timing(
        scene,
        expected=factory_timing,
    )
    base._pack_external_images()
    base._assert_sanitized_build_state()
    inspection_view = configure_textured_inspection_view(bpy)
    scene["benchmark_saved_inspection_view"] = base._canonical_json(
        inspection_view
    )
    bpy.context.view_layer.update()
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))

    plan_hash_after = base._sha256_file(plan_path)
    if plan_hash_after != plan_hash_before:
        raise RuntimeError("materialization plan changed while Blender was building")
    report = {
        "backend": "non_rectangular_fixed_catalog_blender_materializer_v1",
        "status": "built",
        "blender_version": bpy.app.version_string,
        "plan_sha256_before": plan_hash_before,
        "plan_sha256_after": plan_hash_after,
        "out_blend_sha256": base._sha256_file(out_blend),
        "render_invocation_count": 0,
        "placement": {
            "scale_mode": "exact_uniform_scale",
            "coordinates_transformed": False,
            "global_centers_preserved": True,
            "adjacent_room_objects_included": False,
        },
        "catalog_snapshot_id": str(plan["catalog_snapshot_id"]),
        "materialization_revision": str(plan["materialization_revision"]),
        "adapter_contract_revision": str(plan["adapter_contract_revision"]),
        "architecture": request["architecture"],
        "rendered_wall_ids": rendered_wall_ids,
        "instances": built_instances,
        "scene_normalizations": scene_normalizations,
        "saved_inspection_view": inspection_view,
        "source_scene_saved": True,
        "source_scene_kind": "benchmark_owned_nonrect_room_sanitized_output",
    }
    base._write_json(report_path, report)


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--out-blend", required=True)
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(values)


def _validate_plan(value: object) -> tuple[dict, list[dict]]:
    if not isinstance(value, dict) or value.get("schema_version") != PLAN_VERSION:
        raise ValueError("unsupported non-rectangular materialization plan")
    for key in (
        "materialization_revision",
        "adapter_contract_revision",
        "catalog_snapshot_id",
    ):
        if not isinstance(value.get(key), str) or not str(value[key]).strip():
            raise ValueError(f"materialization plan {key} must be non-empty")
    request = value.get("request")
    if not isinstance(request, dict):
        raise ValueError("materialization plan request must be an object")
    for key in ("request_id", "scene_type", "layout_id", "room_id"):
        if not isinstance(request.get(key), str) or not str(request[key]).strip():
            raise ValueError(f"request.{key} must be non-empty")
    boundary = request.get("boundary")
    if not isinstance(boundary, list) or len(boundary) < 3:
        raise ValueError("request.boundary must be a polygon")
    request["boundary"] = [
        base._finite_vector(point, 2, f"request.boundary[{index}]")
        for index, point in enumerate(boundary)
    ]
    request["floor_z_m"] = base._finite_number(
        request.get("floor_z_m"),
        "request.floor_z_m",
    )
    request["scene_height"] = base._finite_number(
        request.get("scene_height"),
        "request.scene_height",
        positive=True,
    )
    architecture = request.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("request.architecture must be an object")
    _validate_architecture(request, architecture)
    raw_instances = value.get("instances")
    if not isinstance(raw_instances, list):
        raise ValueError("materialization plan instances must be an array")
    instances = [
        base._validate_instance(item, index=index)
        for index, item in enumerate(raw_instances)
    ]
    ids = [item["instance_id"] for item in instances]
    evaluator_ids = [item["evaluator_object_id"] for item in instances]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("instance IDs must be unique and sorted")
    if len(evaluator_ids) != len(set(evaluator_ids)):
        raise ValueError("evaluator object IDs must be unique")
    return request, instances


def _validate_architecture(request: dict, architecture: dict) -> None:
    if architecture.get("room_id") != request["room_id"]:
        raise ValueError("architecture room_id mismatch")
    floor = architecture.get("floor")
    walls = architecture.get("wall_segments")
    if not isinstance(floor, dict) or not isinstance(walls, list):
        raise ValueError("architecture floor/wall inventory is missing")
    if floor.get("polygon_xy") != request["boundary"]:
        raise ValueError("architecture floor polygon differs from request boundary")
    if not math.isclose(
        float(floor.get("floor_z_m")),
        float(request["floor_z_m"]),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    ):
        raise ValueError("architecture floor elevation mismatch")
    if len(walls) != len(request["boundary"]):
        raise ValueError("ordered wall count must equal polygon edge count")
    wall_ids: list[str] = []
    for index, wall in enumerate(walls):
        if not isinstance(wall, dict) or wall.get("wall_index") != index:
            raise ValueError("wall segments must preserve authoritative order")
        wall_id = str(wall.get("wall_id") or "")
        if not wall_id or wall_id in wall_ids:
            raise ValueError("wall IDs must be non-empty and unique")
        wall_ids.append(wall_id)
        start = request["boundary"][index]
        end = request["boundary"][(index + 1) % len(request["boundary"])]
        if not base._close_vector(wall.get("start_xy"), start) or not base._close_vector(
            wall.get("end_xy"), end
        ):
            raise ValueError("wall segment differs from polygon edge")
        base._finite_vector(
            wall.get("inward_normal_xy"),
            2,
            f"wall_segments[{index}].inward_normal_xy",
        )
        base._finite_vector(
            wall.get("tangent_xy"),
            2,
            f"wall_segments[{index}].tangent_xy",
        )
        base._finite_number(
            wall.get("height_m"),
            f"wall_segments[{index}].height_m",
            positive=True,
        )
        thickness = base._finite_number(
            wall.get("thickness_m"),
            f"wall_segments[{index}].thickness_m",
        )
        if thickness < 0.0:
            raise ValueError("wall thickness must be non-negative")
    if architecture.get("excluded_architecture") != [
        "ceiling",
        "doors",
        "windows",
    ]:
        raise ValueError("non-rectangular architecture exclusion contract drift")


def _build_architecture(architecture: dict) -> tuple[list, list[str]]:
    floor = architecture["floor"]
    polygon = floor["polygon_xy"]
    floor_z = float(floor["floor_z_m"])
    floor_mesh = bpy.data.meshes.new("benchmark_nonrect_floor_mesh")
    floor_mesh.from_pydata(
        [(float(x), float(y), floor_z) for x, y in polygon],
        [],
        [list(range(len(polygon)))],
    )
    floor_object = bpy.data.objects.new("benchmark_nonrect_floor", floor_mesh)
    bpy.context.collection.objects.link(floor_object)
    floor_id = str(floor["floor_id"])
    floor_object[ARCHITECTURE_ID_PROPERTY] = floor_id
    floor_object["benchmark_nonrect_room_id"] = str(architecture["room_id"])
    floor_object["benchmark_nonrect_polygon_xy"] = base._canonical_json(polygon)
    floor_object.data.materials.append(
        _material(floor_id, (0.34, 0.36, 0.39, 1.0))
    )
    objects = [floor_object]
    rendered_wall_ids: list[str] = []
    for wall in architecture["wall_segments"]:
        wall_object = _build_wall(wall, floor_z=floor_z)
        objects.append(wall_object)
        rendered_wall_ids.append(str(wall["wall_id"]))
    return objects, rendered_wall_ids


def _build_wall(wall: dict, *, floor_z: float):
    start = [float(value) for value in wall["start_xy"]]
    end = [float(value) for value in wall["end_xy"]]
    height = float(wall["height_m"])
    thickness = float(wall["thickness_m"])
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    wall_id = str(wall["wall_id"])
    if thickness > TOLERANCE:
        bpy.ops.mesh.primitive_cube_add(
            location=(
                (start[0] + end[0]) / 2.0,
                (start[1] + end[1]) / 2.0,
                floor_z + height / 2.0,
            )
        )
        obj = bpy.context.object
        obj.dimensions = (length, thickness, height)
        obj.rotation_euler[2] = math.atan2(dy, dx)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    else:
        mesh = bpy.data.meshes.new(f"benchmark_{wall_id}_mesh")
        mesh.from_pydata(
            [
                (start[0], start[1], floor_z),
                (end[0], end[1], floor_z),
                (end[0], end[1], floor_z + height),
                (start[0], start[1], floor_z + height),
            ],
            [],
            [[0, 1, 2, 3]],
        )
        obj = bpy.data.objects.new(f"benchmark_{wall_id}", mesh)
        bpy.context.collection.objects.link(obj)
    obj.name = f"benchmark_{wall_id}"
    obj[ARCHITECTURE_ID_PROPERTY] = wall_id
    obj["benchmark_nonrect_room_id"] = str(wall["room_id"])
    obj["benchmark_nonrect_wall_index"] = int(wall["wall_index"])
    obj["benchmark_nonrect_start_xy"] = base._canonical_json(start)
    obj["benchmark_nonrect_end_xy"] = base._canonical_json(end)
    obj["benchmark_nonrect_inward_normal_xy"] = base._canonical_json(
        wall["inward_normal_xy"]
    )
    obj["benchmark_nonrect_tangent_xy"] = base._canonical_json(
        wall["tangent_xy"]
    )
    obj["benchmark_nonrect_height_m"] = height
    obj["benchmark_nonrect_thickness_m"] = thickness
    obj.data.materials.append(_material(wall_id, (0.60, 0.62, 0.66, 1.0)))
    return obj


if __name__ == "__main__":
    main()
