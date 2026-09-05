"""Independent read-only inspector for a non-rectangular room blend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy


MATERIALIZATION_DIR = Path(__file__).resolve().parents[1] / "materialization"
if str(MATERIALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(MATERIALIZATION_DIR))

import blend_inspector_worker as base  # noqa: E402


PLAN_VERSION = "non_rectangular_catalog_materialization_plan_v1"
TOLERANCE = 1.0e-5


def main() -> None:
    args = _parse_args()
    plan_path = Path(args.plan_json).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    plan = base._load_json(plan_path)
    if plan.get("schema_version") != PLAN_VERSION:
        raise ValueError("unsupported non-rectangular materialization plan")
    records = base._records(plan, "non-rectangular materialization plan")
    expected = base._merge_expected_records(
        records,
        records,
        expected_data=plan,
        catalog_data=plan,
    )
    base._validate_architecture_allowlist = _validate_nonrect_architecture
    report = base._inspect(
        mode="sanitized",
        expected_records=expected,
        expected_data=plan,
        catalog_data=plan,
        expected_path=plan_path,
        catalog_path=plan_path,
    )
    report["backend"] = "non_rectangular_blender_read_only_inspector_v1"
    report["non_rectangular_room_scope"] = {
        "room_id": plan["request"]["room_id"],
        "global_coordinates_preserved": True,
        "adjacent_room_objects_included": False,
        "polygon_floor": True,
        "ordered_wall_segments": True,
        "ceiling_included": False,
    }
    base._write_json(out_path, report)


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args(values)


def _validate_nonrect_architecture(
    *,
    boundary,
    scene_height,
    architecture,
    required,
) -> dict:
    del scene_height
    mismatches: list[dict] = []
    architecture_objects = [
        obj
        for obj in bpy.data.objects
        if obj.get(base.ARCHITECTURE_ID_PROPERTY) is not None
        or obj.get(base.ROLE_PROPERTY) == "architecture"
    ]
    if not required or not isinstance(architecture, dict):
        mismatches.append({"code": "missing_nonrect_architecture_spec"})
        return _result(architecture_objects, [], mismatches)
    floor = architecture.get("floor")
    walls = architecture.get("wall_segments")
    if not isinstance(boundary, list) or not isinstance(floor, dict) or not isinstance(
        walls, list
    ):
        mismatches.append({"code": "invalid_nonrect_architecture_spec"})
        return _result(architecture_objects, [], mismatches)
    if floor.get("polygon_xy") != boundary:
        mismatches.append({"code": "floor_polygon_spec_mismatch"})
    expected_ids = [str(floor.get("floor_id") or "")]
    expected_ids.extend(str(item.get("wall_id") or "") for item in walls)
    if not all(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        mismatches.append({"code": "invalid_nonrect_architecture_ids"})
    by_id: dict[str, list] = {}
    for obj in architecture_objects:
        architecture_id = str(obj.get(base.ARCHITECTURE_ID_PROPERTY) or "")
        by_id.setdefault(architecture_id, []).append(obj)
    observed_ids = sorted(by_id)
    if set(observed_ids) != set(expected_ids):
        mismatches.append(
            {
                "code": "architecture_id_set_mismatch",
                "expected": sorted(expected_ids),
                "observed": observed_ids,
            }
        )
    for architecture_id, objects in by_id.items():
        if len(objects) != 1:
            mismatches.append(
                {
                    "code": "duplicate_architecture_id",
                    "architecture_id": architecture_id,
                    "count": len(objects),
                }
            )
    floor_objects = by_id.get(expected_ids[0], []) if expected_ids else []
    if len(floor_objects) == 1:
        _inspect_floor(floor_objects[0], floor, mismatches)
    for index, wall in enumerate(walls):
        wall_id = str(wall.get("wall_id") or "")
        candidates = by_id.get(wall_id, [])
        if len(candidates) == 1:
            _inspect_wall(candidates[0], wall, index=index, mismatches=mismatches)
    if any("ceiling" in item.lower() for item in observed_ids):
        mismatches.append({"code": "ceiling_architecture_forbidden"})
    return _result(architecture_objects, expected_ids, mismatches)


def _result(objects: list, expected_ids: list[str], mismatches: list[dict]) -> dict:
    return {
        "passed": not mismatches,
        "allowed_object_names": sorted(obj.name for obj in objects),
        "expected_architecture_ids": expected_ids,
        "observed_architecture_ids": sorted(
            str(obj.get(base.ARCHITECTURE_ID_PROPERTY) or "") for obj in objects
        ),
        "mismatches": mismatches,
        "architecture_mode": "exact_polygon_floor_and_ordered_wall_segments",
    }


def _inspect_floor(obj, floor: dict, mismatches: list[dict]) -> None:
    if obj.type != "MESH" or obj.get(base.ROLE_PROPERTY) != "architecture":
        mismatches.append({"code": "floor_role_or_type_mismatch"})
        return
    expected = [
        (float(point[0]), float(point[1]), float(floor["floor_z_m"]))
        for point in floor["polygon_xy"]
    ]
    observed = _world_vertices(obj)
    if not _ordered_close(observed, expected):
        mismatches.append(
            {
                "code": "floor_polygon_geometry_mismatch",
                "expected_vertex_count": len(expected),
                "observed_vertex_count": len(observed),
            }
        )
    if len(obj.data.polygons) != 1:
        mismatches.append({"code": "floor_face_count_mismatch"})
    _material_name(obj, str(floor["floor_id"]), mismatches)


def _inspect_wall(obj, wall: dict, *, index: int, mismatches: list[dict]) -> None:
    if obj.type != "MESH" or obj.get(base.ROLE_PROPERTY) != "architecture":
        mismatches.append(
            {"code": "wall_role_or_type_mismatch", "wall_id": wall["wall_id"]}
        )
        return
    exact = {
        "benchmark_nonrect_room_id": str(wall["room_id"]),
        "benchmark_nonrect_wall_index": int(index),
        "benchmark_nonrect_start_xy": wall["start_xy"],
        "benchmark_nonrect_end_xy": wall["end_xy"],
        "benchmark_nonrect_inward_normal_xy": wall["inward_normal_xy"],
        "benchmark_nonrect_tangent_xy": wall["tangent_xy"],
        "benchmark_nonrect_height_m": float(wall["height_m"]),
        "benchmark_nonrect_thickness_m": float(wall["thickness_m"]),
    }
    for key, expected in exact.items():
        observed = obj.get(key)
        if isinstance(expected, list):
            try:
                observed = json.loads(str(observed))
            except (TypeError, json.JSONDecodeError):
                observed = None
            passed = _close_json(observed, expected)
        elif isinstance(expected, float):
            try:
                passed = math.isclose(
                    float(observed), expected, rel_tol=0.0, abs_tol=TOLERANCE
                )
            except (TypeError, ValueError):
                passed = False
        else:
            passed = observed == expected
        if not passed:
            mismatches.append(
                {
                    "code": "wall_provenance_mismatch",
                    "wall_id": wall["wall_id"],
                    "field": key,
                }
            )
    expected_vertices = _expected_wall_vertices(wall)
    if not _point_multiset_close(_world_vertices(obj), expected_vertices):
        mismatches.append(
            {"code": "wall_geometry_mismatch", "wall_id": wall["wall_id"]}
        )
    _material_name(obj, str(wall["wall_id"]), mismatches)


def _expected_wall_vertices(wall: dict) -> list[tuple[float, float, float]]:
    start = [float(value) for value in wall["start_xy"]]
    end = [float(value) for value in wall["end_xy"]]
    height = float(wall["height_m"])
    thickness = float(wall["thickness_m"])
    floor_z = float(
        bpy.context.scene.get("benchmark_architecture_contract") is None and 0.0
        or json.loads(str(bpy.context.scene["benchmark_architecture_contract"]))[
            "floor"
        ]["floor_z_m"]
    )
    if thickness <= TOLERANCE:
        return [
            (start[0], start[1], floor_z),
            (end[0], end[1], floor_z),
            (end[0], end[1], floor_z + height),
            (start[0], start[1], floor_z + height),
        ]
    normal = [float(value) for value in wall["inward_normal_xy"]]
    half = thickness / 2.0
    result = []
    for point in (start, end):
        for side in (-1.0, 1.0):
            for z in (floor_z, floor_z + height):
                result.append(
                    (
                        point[0] + side * normal[0] * half,
                        point[1] + side * normal[1] * half,
                        z,
                    )
                )
    return result


def _world_vertices(obj) -> list[tuple[float, float, float]]:
    return [
        tuple(float(value) for value in (obj.matrix_world @ vertex.co))
        for vertex in obj.data.vertices
    ]


def _ordered_close(left: list[tuple], right: list[tuple]) -> bool:
    return len(left) == len(right) and all(
        all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=TOLERANCE)
            for a, b in zip(observed, expected)
        )
        for observed, expected in zip(left, right)
    )


def _point_multiset_close(left: list[tuple], right: list[tuple]) -> bool:
    quantize = lambda point: tuple(round(float(value) / TOLERANCE) for value in point)
    return sorted(quantize(item) for item in left) == sorted(
        quantize(item) for item in right
    )


def _close_json(left, right) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close_json(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=TOLERANCE)
    return left == right


def _material_name(obj, expected: str, mismatches: list[dict]) -> None:
    names = [material.name for material in obj.data.materials if material is not None]
    if names != [expected]:
        mismatches.append(
            {
                "code": "architecture_material_mismatch",
                "architecture_id": expected,
                "observed": names,
            }
        )


if __name__ == "__main__":
    main()
