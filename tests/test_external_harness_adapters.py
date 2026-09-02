from __future__ import annotations

import math
from pathlib import Path

import pytest

from benchmark.adapters import get_adapter, list_adapters
from benchmark.api.generation import run_generate
from benchmark.io_contracts import O1_OBJECT_STATE
from benchmark.nl_scene.generation_input import (
    build_direct_natural_language_generation_input,
)
from benchmark.utils.io import read_json, write_json


EXTERNAL_ADAPTERS = {
    "direct_layout",
    "holodeck",
    "layout_gpt",
    "layout_vlm",
    "respace",
    "scene_smith",
    "scene_weaver",
}


def test_selected_external_harnesses_are_registered_without_excluded_methods() -> None:
    names = set(list_adapters())

    assert EXTERNAL_ADAPTERS <= names
    assert "hsm" not in names
    assert "instruct_scene" not in names


@pytest.mark.parametrize("adapter_name", sorted(EXTERNAL_ADAPTERS))
def test_external_harnesses_share_one_generator_input_contract(
    adapter_name: str,
    tmp_path: Path,
) -> None:
    adapter = get_adapter(adapter_name)
    method_input = read_json(
        adapter.prepare_input(_generation_input(), tmp_path / adapter_name)
    )

    assert method_input["harness"] == adapter_name
    assert method_input["protocol"] == adapter.output_schema
    assert method_input["io_contract"]["evaluator_output_type"] == O1_OBJECT_STATE
    assert method_input["generator_input"]["natural_language"] == "Design a furnished room."


def test_direct_layout_official_object_array_converts_with_external_asset_manifest(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "direct.json",
        [
            {
                "new_object_id": "bed_1",
                "rotation": {"z_angle": 90.0},
                "size_in_meters": {"length": 2.0, "width": 1.6, "height": 0.8},
                "position": {"x": 2.0, "y": 2.5, "z": 0.4},
            }
        ],
    )
    write_json(
        tmp_path / "assets.json",
        {
            "bed_1": {
                "source_db": "custom_furniture",
                "category": "bed",
                "description": "upholstered bed",
                "mesh_uri": "/assets/bed.glb",
            }
        },
    )
    scene = _materialize(
        "direct_layout",
        native,
        tmp_path / "direct_out",
        {"asset_manifest_path": "assets.json"},
    )

    obj = scene["objects"][0]
    assert obj["id"] == "bed_1"
    assert obj["size"] == [2.0, 1.6, 0.8]
    assert obj["center"] == [2.0, 2.5, 0.4]
    assert obj["rotation"] == [0.0, 0.0, 90.0]
    assert obj["asset_ref"] == {
        "source_db": "custom_furniture",
        "asset_key": "bed_1",
        "mesh_uri": "/assets/bed.glb",
    }


def test_python_asset_provider_is_injected_without_evaluator_changes(tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []

    class Provider:
        def resolve(self, asset_key, *, source_db=None, hint=None):
            del hint
            calls.append((asset_key, source_db))
            return {
                "asset_key": "resolved-chair",
                "source_db": "remote_asset_service",
                "category": "chair",
                "mesh_uri": "/remote-cache/resolved-chair.glb",
            }

        def retrieve(self, query, *, category=None, size=None, hint=None):
            raise AssertionError("DirectLayout has a native key and must use resolve()")

    native = write_json(
        tmp_path / "provider_direct.json",
        [
            {
                "new_object_id": "chair_1",
                "rotation": {"z_angle": 0},
                "size_in_meters": {"length": 0.8, "width": 0.8, "height": 1.0},
                "position": {"x": 1.0, "y": 1.0, "z": 0.5},
            }
        ],
    )
    result = run_generate(
        generation_input=_generation_input(),
        adapter_name="direct_layout",
        out_dir=tmp_path / "provider_out",
        method_output=native,
        adapter_config={"asset_provider": Provider()},
    )
    scene = read_json(result["generated_scene"])

    assert calls == [("chair_1", "directlayout")]
    assert scene["objects"][0]["asset_ref"]["source_db"] == "remote_asset_service"
    assert result["status"]["status"] == "generated_scene_available"


def test_pointcloud_only_asset_metadata_remains_obb_proxy(tmp_path: Path) -> None:
    native = write_json(
        tmp_path / "pointcloud_direct.json",
        [
            {
                "new_object_id": "chair_1",
                "rotation": {"z_angle": 0},
                "size_in_meters": {"length": 0.8, "width": 0.8, "height": 1.0},
                "position": {"x": 1.0, "y": 1.0, "z": 0.5},
            }
        ],
    )
    scene = _materialize(
        "direct_layout",
        native,
        tmp_path / "pointcloud_out",
        {
            "asset_manifest": {
                "chair_1": {
                    "category": "chair",
                    "pointcloud_uri": "/assets/chair.ply",
                }
            }
        },
    )

    obj = scene["objects"][0]
    assert obj["geometry_provenance"] == "bbox_proxy"
    assert "mesh_uri" not in obj["asset_ref"]


def test_existing_dataset_retrieval_runtime_is_a_supported_asset_backend(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, list[float] | None]] = []

    class Dataset:
        asset_namespace = "portable_catalog"

    class Composed:
        dataset = Dataset()

    class RetrievalRuntime:
        composed = Composed()
        assets = {}

        def retrieve(self, description, *, size_constraint):
            calls.append((description, list(size_constraint) if size_constraint else None))
            return {
                "jid": "retrieved-bed",
                "category": "double_bed",
                "description": "retrieved double bed",
                "size": [2.0, 1.0, 0.5],
            }

        def retrieve_batch(self, request):
            raise AssertionError(request)

    native = write_json(
        tmp_path / "retrieval_layoutgpt.json",
        {
            "unit": "m",
            "object_list": [
                [
                    "double_bed",
                    {
                        "length": 2.0,
                        "width": 1.0,
                        "height": 0.5,
                        "left": 2.0,
                        "top": 2.0,
                        "depth": 0.25,
                        "orientation": 0,
                    },
                ]
            ],
        },
    )
    scene = _materialize(
        "layout_gpt",
        native,
        tmp_path / "retrieval_layoutgpt_out",
        {"retrieval_runtime": RetrievalRuntime()},
    )

    assert calls == [("double_bed", [2.0, 1.0, 0.5])]
    assert scene["objects"][0]["asset_ref"] == {
        "source_db": "portable_catalog",
        "asset_key": "retrieved-bed",
    }


def test_layout_gpt_parsed_output_scales_official_pixel_coordinates(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "layoutgpt.json",
        [
            {
                "iter": 0,
                "query_id": "bedroom-case",
                "prompt": "Room Size: max length 256px, max width 320px",
                "object_list": [
                    [
                        "double_bed",
                        {
                            "length": 128,
                            "width": 64,
                            "height": 32,
                            "left": 128,
                            "top": 160,
                            "depth": 16,
                            "orientation": -90,
                        },
                    ]
                ],
            }
        ],
    )
    scene = _materialize(
        "layout_gpt",
        native,
        tmp_path / "layoutgpt_out",
        {
            "query_id": "bedroom-case",
            "asset_manifest": {
                "bed_asset": {
                    "category": "double_bed",
                    "source_db": "another_catalog",
                    "bbox_size": [2.0, 1.0, 0.5],
                }
            },
        },
    )

    obj = scene["objects"][0]
    assert obj["size"] == pytest.approx([2.0, 1.0, 0.5])
    assert obj["center"] == pytest.approx([2.0, 2.5, 0.25])
    assert obj["rotation"][2] == -90.0
    assert obj["asset_ref"]["source_db"] == "another_catalog"
    assert obj["asset_ref"]["asset_key"] == "bed_asset"


def test_layout_vlm_joins_layout_with_input_asset_table(tmp_path: Path) -> None:
    native = write_json(
        tmp_path / "layoutvlm.json",
        {
            "layout": {
                "abc123-0": {
                    "position": [1.5, 2.0, 0.5],
                    "rotation": [0.0, 0.0, 45.0],
                }
            },
            "scene_config": {
                "boundary": {
                    "floor_vertices": [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]],
                    "wall_height": 3.0,
                },
                "assets": {
                    "abc123-0": {
                        "uid": "abc123",
                        "category": "chair",
                        "description": "rattan chair",
                        "path": "/assets/abc123.glb",
                        "assetMetadata": {
                            "boundingBox": {"x": 0.8, "y": 0.7, "z": 1.0}
                        },
                    }
                },
            },
        },
    )
    scene = _materialize("layout_vlm", native, tmp_path / "layoutvlm_out")

    obj = scene["objects"][0]
    assert obj["jid"] == "abc123"
    assert obj["category"] == "chair"
    assert obj["size"] == [0.8, 0.7, 1.0]
    assert obj["center"] == [1.5, 2.0, 0.5]
    assert obj["asset_ref"]["mesh_uri"] == "/assets/abc123.glb"


def test_respace_y_up_ssr_converts_axes_bottom_anchor_and_asset_id(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "respace.json",
        {
            "room_type": "bedroom",
            "bounds_bottom": [
                [-2.0, 0.0, 1.5],
                [2.0, 0.0, 1.5],
                [2.0, 0.0, -1.5],
                [-2.0, 0.0, -1.5],
            ],
            "bounds_top": [
                [-2.0, 3.0, 1.5],
                [2.0, 3.0, 1.5],
                [2.0, 3.0, -1.5],
                [-2.0, 3.0, -1.5],
            ],
            "objects": [
                {
                    "id": "nightstand_1",
                    "desc": "wood nightstand",
                    "sampled_jid": "future-1",
                    "pos": [-1.0, 0.0, 1.0],
                    "rot": [0.0, 0.0, 0.0, 1.0],
                    "size": [0.6, 0.8, 0.5],
                }
            ],
        },
    )
    scene = _materialize(
        "respace",
        native,
        tmp_path / "respace_out",
        {
            "asset_manifest": {
                "future-1": {
                    "category": "nightstand",
                    "source_db": "replacement_db",
                }
            }
        },
    )

    obj = scene["objects"][0]
    assert scene["boundary"] == [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    assert scene["scene_height"] == 3.0
    assert obj["size"] == [0.6, 0.5, 0.8]
    assert obj["center"] == pytest.approx([1.0, 0.5, 0.4])
    assert obj["rotation"] == pytest.approx([0.0, 0.0, 0.0])
    assert obj["asset_ref"]["source_db"] == "replacement_db"


def test_scene_weaver_selects_latest_iteration_and_converts_bottom_center(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sceneweaver"
    write_json(root / "record_scene" / "layout_0.json", {"roomsize": [4, 5], "objects": {}})
    write_json(
        root / "record_scene" / "layout_2.json",
        {
            "roomsize": [4, 5],
            "structure": {},
            "objects": {
                "sofa_0": {
                    "location": [2.0, 2.0, 0.],
                    "rotation": [0.0, 0.0, math.pi / 2.0],
                    "size": [2.0, 1.0, 0.8],
                    "parent": [],
                }
            },
        },
    )
    scene = _materialize(
        "scene_weaver",
        root,
        tmp_path / "sceneweaver_out",
        {
            "asset_manifest": {
                "sofa_asset": {
                    "category": "sofa",
                    "source_db": "custom_sceneweaver_assets",
                }
            }
        },
    )

    obj = scene["objects"][0]
    assert obj["center"] == pytest.approx([2.0, 2.0, 0.4])
    assert obj["rotation"] == pytest.approx([0.0, 0.0, 90.0])
    assert scene["metadata"]["harness_compatibility"]["selected_iteration"] == 2


def test_holodeck_raw_procthor_scene_converts_unity_axes(tmp_path: Path) -> None:
    native = write_json(
        tmp_path / "holodeck.json",
        {
            "wall_height": 3.0,
            "rooms": [
                {
                    "id": "living-room",
                    "roomType": "Living Room",
                    "floorPolygon": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 4, "y": 0, "z": 0},
                        {"x": 4, "y": 0, "z": 5},
                        {"x": 0, "y": 0, "z": 5},
                    ],
                }
            ],
            "walls": [],
            "objects": [
                {
                    "id": "armchair (living-room)",
                    "object_name": "armchair",
                    "assetId": "objathor-1",
                    "roomId": "living-room",
                    "position": {"x": 1.0, "y": 0.5, "z": 2.0},
                    "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                    "kinematic": True,
                }
            ],
        },
    )
    scene = _materialize(
        "holodeck",
        native,
        tmp_path / "holodeck_out",
        {
            "asset_manifest": {
                "objathor-1": {
                    "category": "armchair",
                    "source_db": "objathor_repacked",
                    "bbox_size": [1.0, 1.0, 2.0],
                }
            }
        },
    )

    obj = scene["objects"][0]
    assert obj["size"] == [1.0, 2.0, 1.0]
    assert obj["center"] == [1.0, 2.0, 0.5]
    assert obj["rotation"] == [0.0, 0.0, -90.0]
    assert obj["asset_ref"]["source_db"] == "objathor_repacked"


def test_holodeck_accepts_official_scene_state_export(tmp_path: Path) -> None:
    native = write_json(
        tmp_path / "holodeck_scene_state.json",
        _scene_state(
            model_id="objaverse.chair-1",
            translation=[1.0, 2.0, 0.5],
        ),
    )
    scene = _materialize(
        "holodeck",
        native,
        tmp_path / "holodeck_scene_state_out",
        {
            "asset_manifest": {
                "chair-1": {
                    "category": "chair",
                    "source_db": "objaverse",
                    "bbox_size": [0.8, 0.8, 1.0],
                }
            }
        },
    )

    obj = scene["objects"][0]
    assert obj["center"] == [1.0, 2.0, 0.5]
    assert obj["size"] == [0.8, 0.8, 1.0]
    assert scene["metadata"]["harness_compatibility"]["coordinate_conversion"]["source"] == "scene_state"


def test_scene_state_transform_applies_rotation_scale_and_local_bbox_center(
    tmp_path: Path,
) -> None:
    payload = _scene_state(model_id="objaverse.offset-chair", translation=[1.0, 1.0, 0.0])
    payload["scene"]["object"][0]["transform"]["data"] = [
        0, 2, 0, 0,
        -3, 0, 0, 0,
        0, 0, 4, 0,
        1, 1, 0, 1,
    ]
    native = write_json(tmp_path / "scaled_scene_state.json", payload)
    scene = _materialize(
        "holodeck",
        native,
        tmp_path / "scaled_scene_state_out",
        {
            "asset_manifest": {
                "offset-chair": {
                    "category": "chair",
                    "bbox_size": [1.0, 2.0, 3.0],
                    "bbox_center_local": [0.5, 0.0, 0.0],
                }
            }
        },
    )

    obj = scene["objects"][0]
    assert obj["size"] == pytest.approx([2.0, 6.0, 12.0])
    assert obj["center"] == pytest.approx([1.0, 2.0, 0.0])
    assert obj["rotation"] == pytest.approx([0.0, 0.0, 90.0])


def test_scene_smith_native_state_uses_local_bbox_and_geometry_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "scenesmith"
    mesh = root / "generated_assets" / "table.glb"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"glTF-test-placeholder")
    native = write_json(
        root / "scene_states" / "final_scene" / "scene_state.json",
        {
            "room_geometry": {"length": 4.0, "width": 3.0, "wall_height": 2.8},
            "objects": {
                "table_0": {
                    "object_id": "table_0",
                    "object_type": "furniture",
                    "name": "Dining Table",
                    "description": "wood dining table",
                    "transform": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    "geometry_path": "generated_assets/table.glb",
                    "sdf_path": "generated_assets/table.sdf",
                    "bbox_min": [-1.0, -0.5, 0.0],
                    "bbox_max": [1.0, 0.5, 1.0],
                    "metadata": {"asset_id": "generated-table"},
                    "placement_info": None,
                    "immutable": False,
                }
            },
        },
    )
    scene = _materialize("scene_smith", native, tmp_path / "scenesmith_out")

    obj = scene["objects"][0]
    assert scene["boundary"] == [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    assert obj["size"] == [2.0, 1.0, 1.0]
    assert obj["center"] == [2.0, 1.5, 0.5]
    assert obj["asset_ref"]["asset_key"] == "generated-table"
    assert obj["asset_ref"]["mesh_uri"] == mesh.resolve().as_posix()
    assert obj["metadata"]["interactive"] is True


def _materialize(
    adapter_name: str,
    native_path: Path,
    out_dir: Path,
    config: dict | None = None,
) -> dict:
    generated_path = get_adapter(adapter_name).materialize_output(
        native_path,
        _generation_input(),
        out_dir,
        config=config,
    )
    return read_json(generated_path)


def _generation_input() -> dict:
    return build_direct_natural_language_generation_input(
        request_id="external-harness-case",
        instruction="Design a furnished room.",
        scene_type="room",
        room={"boundary": [[0, 0], [4, 0], [4, 5], [0, 5]], "height": 3.0},
        evaluator_output_type=O1_OBJECT_STATE,
    )


def _scene_state(*, model_id: str, translation: list[float]) -> dict:
    tx, ty, tz = translation
    return {
        "format": "sceneState",
        "scene": {
            "version": "scene@1.0.2",
            "id": "scene-state-case",
            "unit": 1.0,
            "up": [0, 0, 1],
            "front": [0, 1, 0],
            "arch": {
                "version": "arch@1.0.2",
                "coords2d": [0, 1],
                "scaleToMeters": 1.0,
                "elements": [
                    {
                        "id": "floor|room",
                        "roomId": "room",
                        "type": "Floor",
                        "points": [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]],
                    },
                    {
                        "id": "wall|room|0",
                        "roomId": "room",
                        "type": "Wall",
                        "height": 3.0,
                        "points": [[0, 0, 0], [4, 0, 0]],
                    },
                ],
            },
            "object": [
                {
                    "index": 0,
                    "id": "chair_1",
                    "modelId": model_id,
                    "transform": {
                        "data": [
                            1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            tx, ty, tz, 1,
                        ]
                    },
                }
            ],
        },
    }
