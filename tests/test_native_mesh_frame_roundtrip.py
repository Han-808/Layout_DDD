"""Golden geometry captured from real pinned upstream bpy assembly, no API.

These tests compare vertex sets, not just an OBB: a 180-degree facing error is
invisible to a symmetric bounding box. CI consumes the small captured synthetic
fixture and does not need upstream repositories, Blender, or real assets.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from benchmark.adapters.direct_layout.converter import convert_direct_layout
from benchmark.adapters.layout_vlm.converter import convert_layout_vlm
from benchmark.nl_scene.generation_input import build_generation_input, build_scene_request
from benchmark.utils.io import read_json, write_json


CAPTURE = read_json(
    Path(__file__).parent / "fixtures/external_harnesses/native_mesh_frame_v1.json"
)
ORIGIN = [1.7, -2.3, 0.0]


def _input():
    return build_generation_input(scene_request=build_scene_request(
        request_id="mesh_frame_diagnostic",
        instruction="Place the supplied diagnostic object.",
        scene_type="diagnostic",
        room={"boundary": [[1.7, -2.3], [9.7, -2.3], [9.7, 5.7], [1.7, 5.7]],
              "height": 3, "unit": "meter"},
        structure=False,
    ))


@pytest.mark.parametrize("method", ["direct_layout", "layout_vlm"])
@pytest.mark.parametrize("yaw", [0, 90, 37])
@pytest.mark.parametrize("front", [None, [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
def test_converter_matches_actual_native_mesh_vertices(tmp_path, method, yaw, front):
    asset_id = "diagnostic.exact_asset"
    source_db = "synthetic_fixture"
    asset = {"asset_key": asset_id, "source_db": source_db,
             "category": "furniture", "description": "asymmetric diagnostic mesh",
             "bbox_size_local": CAPTURE["bbox_size_local"],
             "bbox_center_local": CAPTURE["bbox_center_local"]}
    if front is not None:
        asset["canonical_front"] = front
    resolved = []

    class ExactProvider:
        def resolve(self, key, *, source_db=None, hint=None):
            assert key == asset_id
            resolved.append(key)
            return dict(asset)

        def retrieve(self, *args, **kwargs):
            raise AssertionError("representation conversion must not retrieve")

    size = CAPTURE["bbox_size_local"]
    if method == "direct_layout":
        slot = "diagnostic_1"
        native = [{"new_object_id": slot,
                   "rotation": {"z_angle": float(yaw)},
                   "position": dict(zip("xyz", CAPTURE["native_position"])),
                   "size_in_meters": dict(zip(["length", "width", "height"], size))}]
        config = {"asset_bindings": {slot: dict(asset)}}
        converter = convert_direct_layout
    else:
        slot = "diagnostic_1-0"
        native = {slot: {"position": CAPTURE["native_position"],
                         "rotation": [0.0, 0.0, float(yaw)], "scale": [1.0, 1.0, 1.0]}}
        config = {"scene_config": {"assets": {slot: {
            "uid": asset_id, "category": "furniture",
            "assetMetadata": {
                "boundingBox": {"x": size[1], "y": size[0], "z": size[2]},
                "canonicalBoundingBoxBeforeLayoutVLMSwap": dict(zip("xyz", size)),
                "axisTransform": "swap_xy_for_layoutvlm_processed_positive_x_frame",
            },
        }}}}
        converter = convert_layout_vlm
    path = write_json(tmp_path / "native.json", native)
    original = path.read_bytes()
    scene = converter(path, _input(), config, ExactProvider())
    assert path.read_bytes() == original
    assert resolved == [asset_id]
    assert len(scene["objects"]) == 1
    obj = scene["objects"][0]
    assert obj["id"] == slot
    assert obj["asset_ref"]["asset_key"] == asset_id
    assert obj["asset_ref"]["source_db"] == source_db
    assert obj["size"] == pytest.approx(size)
    assert obj["center"] == pytest.approx([2.75, 3.43, 0.86])
    audit = obj["metadata"]["geometry_audit"]
    assert audit["asset_local_bbox_center"] == CAPTURE["bbox_center_local"]
    if front is None:
        assert "canonical_front" not in obj["metadata"]
    else:
        assert obj["metadata"]["canonical_front"] == front

    angle = math.radians(obj["rotation"][2])
    c, s = math.cos(angle), math.sin(angle)
    points = []
    for vertex in CAPTURE["source_vertices"]:
        x, y, z = [vertex[i] - CAPTURE["bbox_center_local"][i] for i in range(3)]
        points.append([c*x - s*y + obj["center"][0],
                       s*x + c*y + obj["center"][1], z + obj["center"][2]])
    actual_native = [[v[i] - ORIGIN[i] for i in range(3)]
                     for v in CAPTURE[method][str(yaw)]]
    for left, right in [(points, actual_native), (actual_native, points)]:
        assert max(min(math.dist(a, b) for b in right) for a in left) < 1e-6
    if method == "direct_layout":
        assert audit["native_rotation_direction"] == "clockwise_viewed_from_positive_z"
        assert audit["native_zero_mesh_basis_yaw_degrees"] == 180.0
        convention = scene["metadata"]["harness_compatibility"]["coordinate_conversion"]
        assert convention["canonical_rotation_formula"] == (
            "Rz(180 - native_yaw_degrees)"
        )


@pytest.mark.parametrize("yaw, expected", [(5.0, 175.0), (-37.0, -143.0),
                                          (360.0, -180.0), (450.0, 90.0)])
def test_direct_layout_released_clockwise_degrees_are_not_inferred(tmp_path, yaw, expected):
    native = write_json(tmp_path / "native.json", [{
        "new_object_id": "asset_1", "rotation": {"z_angle": yaw},
        "size_in_meters": {"length": 0.7, "width": 1.3, "height": 1.1},
        "position": {"x": 4.45, "y": 1.13, "z": 0.86},
    }])
    obj = convert_direct_layout(native, _input(), {}, None)["objects"][0]
    assert obj["rotation"][2] == pytest.approx(expected)
    assert obj["metadata"]["native_rotation_degrees"] == yaw
