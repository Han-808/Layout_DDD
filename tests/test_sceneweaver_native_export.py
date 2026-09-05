"""Captured real bpy exporter states; no upstream, Blender, model, or dataset in CI.

The capture is a native-export diagnostic, not certification of a frozen GLB
initializer or the complete SceneWeaver loop. Rotated box vertices independently
check the converter's pose, including tilted XYZ rotations and native rounding.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import itertools
import math
from pathlib import Path
import sys

import pytest

from benchmark.adapters.common.geometry import euler_xyz_to_matrix, matrix_vector
from benchmark.adapters.scene_weaver.converter import convert_scene_weaver
from benchmark.nl_scene.generation_input import build_generation_input, build_scene_request
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = read_json(ROOT / "tests/fixtures/external_harnesses/sceneweaver_native_export_v1.json")
SLOT = "diagnostic_slot_1"
CONTRACT = "released_object_dimensions_rounded_2dp"


def _input():
    return build_generation_input(scene_request=build_scene_request(
        request_id="native_export_diagnostic",
        instruction="Place the supplied frozen diagnostic box.",
        scene_type="diagnostic",
        room={"boundary": [[0, 0], [5, 0], [5, 4], [0, 4]],
              "height": 3, "unit": "meter"},
        structure=False,
    ))


def _config(row):
    iteration = str(row["iteration"])
    # Inverse of the fixed cardinal input basis. The observed runtime bbox is
    # kept separately from the exact frozen metadata (float32 vs float64).
    local = list(row["native_local_bbox_size"])
    if row["basis_yaw_degrees"] == 90.0:
        local[0], local[1] = local[1], local[0]
    binding = {
        "asset_key": "synthetic.exact_box", "source_db": "synthetic_fixture",
        "category": "box", "description": "frozen diagnostic box",
        "bbox_size_local": CAPTURE["canonical_size"],
        "bbox_center_local": [0, 0, CAPTURE["canonical_size"][2] / 2],
        "physical_dimensions": CAPTURE["canonical_size"],
        "canonical_front": ([0, -1, 0] if row["basis_yaw_degrees"] else [1, 0, 0]),
        "anchor_basis": {
            "policy": "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin",
            "native_origin_semantics": "bbox_bottom_center", "applied": True,
        },
        "full_precision_native_euler_xyz_by_iteration": {
            iteration: row["full_precision_native_euler_xyz"]},
        "full_precision_native_bottom_center_by_iteration": {
            iteration: row["full_precision_native_bottom_center"]},
        "full_precision_native_local_bbox_size_by_iteration": {iteration: local},
        "full_precision_native_object_dimensions_by_iteration": {
            iteration: row["object_dimensions"]},
    }
    return {
        "rotation_unit": "radian",
        "sceneweaver_native_size_semantics": CONTRACT,
        "sceneweaver_asset_geometry_tolerance_m": 1e-4,
        "sceneweaver_orientation_basis": "bake_catalog_front_to_sceneweaver_positive_x",
        "sceneweaver_anchor_basis": "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin",
        "asset_bindings": {SLOT: binding},
    }


def _native(tmp_path, row):
    return write_json(tmp_path / f"layout_{row['iteration']}.json", {
        "roomsize": [5, 4], "structure": {},
        "objects": {SLOT: deepcopy(row["native_serialized"])},
    })


@pytest.mark.parametrize("row", CAPTURE["cases"], ids=lambda row: f"iteration_{row['iteration']}")
def test_actual_native_export_roundtrips_pose_and_vertices(tmp_path, row):
    config = _config(row)
    path = _native(tmp_path, row)
    original = path.read_bytes()
    original_config = deepcopy(config)
    resolved = []

    class ExactProvider:
        def resolve(self, key, *, source_db=None, hint=None):
            assert key == "synthetic.exact_box"
            resolved.append(key)
            return dict(config["asset_bindings"][SLOT])

        def retrieve(self, *args, **kwargs):
            raise AssertionError("conversion must not retrieve assets")

    scene = convert_scene_weaver(path, _input(), config, ExactProvider())
    assert path.read_bytes() == original
    assert config == original_config
    assert resolved == ["synthetic.exact_box"]
    obj, = scene["objects"]
    assert obj["id"] == SLOT
    assert obj["asset_ref"]["asset_key"] == "synthetic.exact_box"
    assert obj["size"] == CAPTURE["canonical_size"]
    assert obj["metadata"]["canonical_front"] == config["asset_bindings"][SLOT]["canonical_front"]
    audit = obj["metadata"]["geometry_audit"]
    assert audit["released_location_quantization_verified"] is True
    assert audit["released_rotation_quantization_verified"] is True
    assert audit["released_object_dimensions_verified"] is True
    assert audit["observed_native_object_dimensions"] == row["object_dimensions"]
    assert audit["full_precision_native_bottom_center"] == row["full_precision_native_bottom_center"]
    assert row["native_serialized"]["size"] == [round(x, 2) for x in row["object_dimensions"]]
    matrix = euler_xyz_to_matrix(obj["rotation"], "test canonical rotation", unit="degree")
    vertices = []
    for signs in itertools.product((-1, 1), repeat=3):
        offset = matrix_vector(matrix, [signs[i] * obj["size"][i] / 2 for i in range(3)])
        vertices.append([obj["center"][i] + offset[i] for i in range(3)])
    actual = row["world_vertices"]
    for left, right in [(vertices, actual), (actual, vertices)]:
        assert max(min(math.dist(a, b) for b in right) for a in left) < 1e-6


@pytest.mark.parametrize("field, message", [
    ("full_precision_native_bottom_center_by_iteration", "full_precision_native_bottom_center"),
    ("full_precision_native_object_dimensions_by_iteration", "full_precision_native_object_dimensions"),
    ("full_precision_native_euler_xyz_by_iteration", "full_precision_native_euler_xyz"),
    ("full_precision_native_local_bbox_size_by_iteration", "full_precision_native_local_bbox_size"),
])
def test_missing_selected_iteration_observations_fail_closed(tmp_path, field, message):
    row = CAPTURE["cases"][1]
    config = _config(row)
    config["asset_bindings"][SLOT][field] = {"999": [1, 1, 1]}
    with pytest.raises(ArtifactValidationError, match=message):
        convert_scene_weaver(_native(tmp_path, row), _input(), config, None)


@pytest.mark.parametrize("change, message", [
    ("world_aabb", "two-decimal serialization of observed local object dimensions"),
    ("location", "serialization of the observed native bottom center"),
    ("rotation", "serialization of the observed native pose"),
    ("object_scale", "observed object dimensions differ"),
    ("local_bbox", "observed GLB bbox differs"),
    ("retired_contract", "unsupported SceneWeaver native size semantics"),
])
def test_export_contract_mismatches_are_rejected_not_repaired(tmp_path, change, message):
    row = deepcopy(CAPTURE["cases"][1])
    config = _config(row)
    binding = config["asset_bindings"][SLOT]
    if change == "world_aabb":
        row["native_serialized"]["size"] = [round(x, 2) for x in row["actual_world_aabb_size"]]
    elif change in {"location", "rotation"}:
        row["native_serialized"][change][0] += 0.1
    elif change == "object_scale":
        binding["full_precision_native_object_dimensions_by_iteration"]["1"] = [1.4, 2.6, 2.2]
    elif change == "local_bbox":
        binding["full_precision_native_local_bbox_size_by_iteration"]["1"] = [1.4, 2.6, 2.2]
    else:
        config["sceneweaver_native_size_semantics"] = "released_world_aabb_rounded_2dp"
    path = _native(tmp_path, row)
    original = path.read_bytes()
    with pytest.raises(ArtifactValidationError, match=message):
        convert_scene_weaver(path, _input(), config, None)
    assert path.read_bytes() == original


def test_missing_front_is_not_invented(tmp_path):
    row = CAPTURE["cases"][1]
    config = _config(row)
    del config["asset_bindings"][SLOT]["canonical_front"]
    obj, = convert_scene_weaver(_native(tmp_path, row), _input(), config, None)["objects"]
    assert "canonical_front" not in obj["metadata"]
    assert obj["rotation"][2] == pytest.approx(math.degrees(row["full_precision_native_euler_xyz"][2]))


def test_bridge_observes_actual_exporter_states_before_conversion(tmp_path):
    bridge_path = ROOT / "scripts/external_harness_bridges/scene_weaver_frozen.py"
    sys.path.insert(0, str(bridge_path.parent))
    try:
        spec = importlib.util.spec_from_file_location("native_export_test_bridge", bridge_path)
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
    finally:
        sys.path.pop(0)
    mesh = tmp_path / "fixture.glb"
    mesh.write_bytes(b"synthetic CI identity only; geometry is from the bpy capture")
    for basis in (0, 90):
        rows = [row for row in CAPTURE["cases"] if row["basis_yaw_degrees"] == basis]
        asset = dict(_config(rows[0])["asset_bindings"][SLOT],
                     mesh_uri=str(mesh), mesh_sha256=bridge.file_sha256(mesh), native_scale=[1, 1, 1])
        catalog = {"logical_to_native_slot": {SLOT: SLOT}, "frozen_asset_bindings": {SLOT: asset}}
        observations = []
        layouts = []
        for row in rows:
            layouts.append((row["iteration"], _native(tmp_path, row)))
            local = _config(row)["asset_bindings"][SLOT]["full_precision_native_local_bbox_size_by_iteration"][str(row["iteration"])]
            observations.append({"iteration": row["iteration"], "objects": {SLOT: {
                "asset_id": asset["asset_key"], "mesh_path": str(mesh), "mesh_sha256": asset["mesh_sha256"],
                "canonical_local_bbox_size": local,
                "orientation_basis": bridge._orientation_basis(asset),
                "anchor_basis": bridge._anchor_basis(asset),
                "full_precision_native_euler_xyz": row["full_precision_native_euler_xyz"],
                "full_precision_native_bottom_center": row["full_precision_native_bottom_center"],
                "full_precision_native_object_dimensions": row["object_dimensions"],
            }}})
        room = {"roomsize": [5, 4], "height": 3, "unit": "meter"}
        report = {"native_room_observation": room, "iteration_asset_observations": observations}
        kwargs = dict(layouts=layouts, control={"generation": {"asset_geometry_tolerance_m": 1e-4}},
                      catalog=catalog, request={"benchmark_room": room}, tolerance=1e-6)
        original = {path: path.read_bytes() for _, path in layouts}
        audit, bindings = bridge._observe_trajectory(plugin_report=report, **kwargs)
        assert audit["valid"] is True, audit["violations"]
        for row, (_, path) in zip(rows, layouts):
            config = _config(row)
            config["asset_bindings"] = bindings
            config["selected_iteration"] = row["iteration"]
            convert_scene_weaver(path.parent, _input(), config, None)
        for field, violation in [
            ("full_precision_native_bottom_center", "full_precision_position_missing"),
            ("full_precision_native_object_dimensions", "native_object_dimensions_missing"),
        ]:
            changed = deepcopy(report)
            del changed["iteration_asset_observations"][1]["objects"][SLOT][field]
            failed, _ = bridge._observe_trajectory(plugin_report=changed, **kwargs)
            assert failed["valid"] is False
            assert f"iteration_{rows[1]['iteration']}:{violation}:{SLOT}" in failed["violations"]
        assert {path: path.read_bytes() for path in original} == original
