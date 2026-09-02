from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from benchmark.adapters import get_adapter
from benchmark.api.evaluation import run_evaluate
from benchmark.io_contracts import O1_OBJECT_STATE
from benchmark.nl_scene.generation_input import (
    build_direct_natural_language_generation_input,
)
from benchmark.scene_io.validate import ArtifactValidationError, validate_generated_scene
from benchmark.utils.io import read_json, write_json


def test_respace_uses_native_identifier_when_explicit_category_is_missing(
    tmp_path: Path,
) -> None:
    native = _respace_artifact(
        tmp_path / "respace_category.json",
        objects=[
            {
                "id": "nightstand_7",
                "desc": "A description that must not be parsed into a category.",
                "sampled_asset_jid": "chosen-nightstand",
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0, 1.0],
                "size": [0.6, 0.8, 0.5],
            }
        ],
    )
    scene = _materialize(
        "respace",
        native,
        tmp_path / "respace_category_out",
        {
            "asset_manifest": {
                "chosen-nightstand": {
                    "category": "conflicting-provider-category",
                }
            }
        },
    )

    obj = scene["objects"][0]
    assert obj["category"] == "nightstand"
    assert obj["description"] == "A description that must not be parsed into a category."
    assert obj["metadata"]["native_category_source"] == "id"


def test_respace_does_not_invent_category_from_description_or_uuid(
    tmp_path: Path,
) -> None:
    native = _respace_artifact(
        tmp_path / "respace_unknown_category.json",
        objects=[
            {
                "uuid": "6f0e45be-67dd-4ab4-9f8f-563b7af84db0",
                "desc": "An unmistakable velvet armchair.",
                "sampled_asset_jid": "opaque-asset-id",
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0, 1.0],
                "size": [0.8, 1.0, 0.8],
            }
        ],
    )
    scene = _materialize(
        "respace",
        native,
        tmp_path / "respace_unknown_category_out",
    )

    obj = scene["objects"][0]
    assert obj["id"] == "6f0e45be-67dd-4ab4-9f8f-563b7af84db0"
    assert obj["category"] == "unknown"
    assert obj["metadata"]["native_category_source"] == "unavailable"
    assert "canonical_front" not in obj["metadata"]


def test_respace_strict_conversion_requires_persisted_asset_selection(
    tmp_path: Path,
) -> None:
    native = _respace_artifact(
        tmp_path / "respace_missing_asset.json",
        objects=[
            {
                "id": "chair_1",
                "category": "chair",
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0, 1.0],
                "size": [0.8, 1.0, 0.8],
            }
        ],
    )

    with pytest.raises(ArtifactValidationError, match="persisted native asset ID"):
        _materialize(
            "respace",
            native,
            tmp_path / "respace_missing_asset_out",
        )


def test_respace_prefers_released_sampled_asset_identity_and_separates_geometry(
    tmp_path: Path,
) -> None:
    native = _respace_artifact(
        tmp_path / "respace_geometry.json",
        floor_y=1.0,
        objects=[
            {
                "id": "chair_1",
                "category": "chair",
                "jid": "original-asset",
                "sampled_asset_jid": "selected-asset",
                "sampled_asset_size": [0.7, 0.9, 0.8],
                "scale": [2.0, 3.0, 4.0],
                "pos": [0.0, 1.0, 0.0],
                "rot": [0.0, 0.0, 0.0, 1.0],
                "size": [0.8, 1.0, 0.9],
            }
        ],
    )
    scene = _materialize(
        "respace",
        native,
        tmp_path / "respace_geometry_out",
    )

    obj = scene["objects"][0]
    assert obj["asset_ref"]["asset_key"] == "selected-asset"
    assert obj["metadata"]["native_unsampled_asset_id"] == "original-asset"
    assert obj["center"] == pytest.approx([2.0, 2.5, 0.5])
    assert obj["size"] == pytest.approx([0.8, 0.9, 1.0])
    assert obj["geometry_provenance"] == "bbox_proxy"
    assert obj["asset_proxy"]["bbox_center_local"] == [0.0, 0.0, 0.0]
    audit = obj["metadata"]["geometry_audit"]
    assert audit["asset_local_bbox_size"] == pytest.approx([0.7, 0.8, 0.9])
    assert audit["native_scale"] == pytest.approx([2.0, 4.0, 3.0])
    assert audit["evaluated_obb_size"] == pytest.approx([0.8, 0.9, 1.0])
    assert scene["metadata"]["harness_compatibility"]["coordinate_conversion"][
        "origin_shift"
    ] == pytest.approx([2.0, 2.5, -1.0])


@pytest.mark.parametrize(
    ("rotation", "config", "expected"),
    [
        (5.0, {"rotation_encoding": "yaw", "rotation_unit": "degree"}, [0, 0, 5]),
        (
            math.pi / 6,
            {"rotation_encoding": "yaw", "rotation_unit": "radian"},
            [0, 0, 30],
        ),
        (
            [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)],
            {},
            [0, 0, 90],
        ),
        (
            [10.0, 0.0, 0.0],
            {"rotation_encoding": "euler_xyz", "rotation_unit": "degree"},
            [10, 0, 0],
        ),
        (
            [10.0, 20.0, 30.0],
            {"rotation_encoding": "euler_xyz", "rotation_unit": "degree"},
            [-1.1702294331, -28.0243206736, 22.7958772589],
        ),
    ],
)
def test_respace_rotation_contract_is_explicit_and_matrix_based(
    tmp_path: Path,
    rotation: Any,
    config: dict,
    expected: list[float],
) -> None:
    native = _respace_artifact(
        tmp_path / "respace_rotation.json",
        objects=[
            {
                "id": "chair_1",
                "category": "chair",
                "sampled_asset_jid": "chair-asset",
                "pos": [0.0, 0.0, 0.0],
                "rot": rotation,
                "size": [0.8, 1.0, 0.8],
            }
        ],
    )
    scene = _materialize(
        "respace",
        native,
        tmp_path / "respace_rotation_out",
        config,
    )

    assert scene["objects"][0]["rotation"] == pytest.approx(expected)
    convention = scene["metadata"]["harness_compatibility"][
        "coordinate_conversion"
    ]["rotation_convention"]
    assert convention["encoding"] == config.get(
        "rotation_encoding", "quaternion_xyzw"
    )
    assert convention["basis_transform_source_to_canonical"] == [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]


@pytest.mark.parametrize(
    ("rotation", "config", "message"),
    [
        (None, {}, "required by the native SSR contract"),
        (5.0, {}, "xyzw quaternion"),
        (5.0, {"rotation_encoding": "yaw"}, "rotation_unit is required"),
        (
            [0, 0, 0, 1],
            {"rotation_unit": "degree"},
            "invalid for quaternion_xyzw",
        ),
        (
            5.0,
            {"rotation_encoding": "yaw", "rotation_unit": "turn"},
            "must be degree or radian",
        ),
    ],
)
def test_respace_rotation_rejects_ambiguous_or_invalid_contracts(
    tmp_path: Path,
    rotation: Any,
    config: dict,
    message: str,
) -> None:
    native = _respace_artifact(
        tmp_path / "respace_bad_rotation.json",
        objects=[
            {
                "id": "chair_1",
                "sampled_asset_jid": "chair-asset",
                "pos": [0.0, 0.0, 0.0],
                "rot": rotation,
                "size": [0.8, 1.0, 0.8],
            }
        ],
    )

    with pytest.raises(ArtifactValidationError, match=message):
        _materialize(
            "respace",
            native,
            tmp_path / "respace_bad_rotation_out",
            config,
        )


def test_respace_bottom_center_offset_follows_full_quaternion_rotation(
    tmp_path: Path,
) -> None:
    native = _respace_artifact(
        tmp_path / "respace_rotated_anchor.json",
        objects=[
            {
                "id": "bench_1",
                "category": "bench",
                "sampled_asset_jid": "bench-asset",
                "pos": [0.0, 0.0, 0.0],
                "rot": [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)],
                "size": [1.0, 2.0, 1.0],
            }
        ],
    )
    scene = _materialize(
        "respace",
        native,
        tmp_path / "respace_rotated_anchor_out",
    )

    obj = scene["objects"][0]
    assert obj["rotation"] == pytest.approx([90.0, 0.0, 0.0])
    assert obj["center"] == pytest.approx([2.0, 1.5, 0.0])
    assert obj["metadata"]["geometry_audit"][
        "bottom_center_to_center_offset"
    ] == pytest.approx([0.0, -1.0, 0.0])


def test_scene_weaver_requires_explicit_directory_iteration_selection(
    tmp_path: Path,
) -> None:
    root = _scene_weaver_iterations(tmp_path / "sceneweaver_strict")

    with pytest.raises(ArtifactValidationError, match="requires selected_iteration"):
        _materialize(
            "scene_weaver",
            root,
            tmp_path / "sceneweaver_strict_out",
            _scene_weaver_config(),
        )


def test_scene_weaver_each_iteration_uses_the_same_full_evaluator(
    tmp_path: Path,
) -> None:
    root = _scene_weaver_iterations(tmp_path / "sceneweaver_iterations")

    for iteration in range(3):
        scene = _materialize(
            "scene_weaver",
            root,
            tmp_path / f"sceneweaver_iteration_{iteration}",
            {
                **_scene_weaver_config(),
                "selected_iteration": iteration,
            },
        )
        compatibility = scene["metadata"]["harness_compatibility"]
        assert compatibility["selected_iteration"] == iteration
        assert compatibility["available_iterations"] == [0, 1, 2]
        assert compatibility["iteration_selection_policy"] == "explicit_iteration"
        obj = scene["objects"][0]
        assert obj["center"] == pytest.approx([1.0 + iteration, 2.0, 0.5])
        assert obj["geometry_provenance"] == "bbox_proxy"
        assert obj["metadata"]["geometry_audit"]["native_location_semantics"] == (
            "world_bbox_bottom_center"
        )
        assert obj["metadata"]["canonical_front"] == [1.0, 0.0, 0.0]
        assert obj["metadata"]["canonical_front_source"] == (
            "sceneweaver_local_x_front_contract"
        )
        assert validate_generated_scene(scene)

        report = run_evaluate(
            scene=scene,
            out=tmp_path / f"sceneweaver_iteration_{iteration}_report.json",
        )
        assert report["workflow"] == "canonical_l0_l4"
        assert report["request_id"] == "converter-correctness"


def test_scene_weaver_explicit_layout_path_is_auditable(tmp_path: Path) -> None:
    root = _scene_weaver_iterations(tmp_path / "sceneweaver_layout_path")
    scene = _materialize(
        "scene_weaver",
        root,
        tmp_path / "sceneweaver_layout_path_out",
        {
            **_scene_weaver_config(),
            "layout_path": "record_scene/layout_1.json",
        },
    )

    compatibility = scene["metadata"]["harness_compatibility"]
    assert compatibility["selected_iteration"] == 1
    assert compatibility["available_iterations"] == [0, 1, 2]
    assert compatibility["iteration_selection_policy"] == "explicit_layout_path"


def test_scene_weaver_bottom_center_offset_follows_full_euler_rotation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sceneweaver_rotated_anchor"
    write_json(
        root / "record_scene" / "layout_0.json",
        {
            "roomsize": [4, 5],
            "structure": {},
            "objects": {
                "bench_0": {
                    "location": [2.0, 3.0, 1.0],
                    "rotation": [math.pi / 2, 0.0, 0.0],
                    "size": [1.0, 1.0, 2.0],
                    "parent": [],
                }
            },
        },
    )
    scene = _materialize(
        "scene_weaver",
        root,
        tmp_path / "sceneweaver_rotated_anchor_out",
        {
            "selected_iteration": 0,
            "asset_bindings": {
                "bench_0": {"asset_key": "bench-asset", "category": "bench"}
            },
        },
    )

    obj = scene["objects"][0]
    assert obj["rotation"] == pytest.approx([90.0, 0.0, 0.0])
    assert obj["center"] == pytest.approx([2.0, 2.0, 1.0])
    assert obj["metadata"]["geometry_audit"][
        "bottom_center_to_center_offset"
    ] == pytest.approx([0.0, -1.0, 0.0])


def test_direct_layout_mesh_uri_does_not_override_bbox_geometry_provenance(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "direct_geometry.json",
        [
            {
                "new_object_id": "chair_1",
                "rotation": {"z_angle": 5.0},
                "size_in_meters": {"length": 0.8, "width": 0.7, "height": 1.0},
                "position": {"x": 2.0, "y": 2.0, "z": 0.5},
            }
        ],
    )
    scene = _materialize(
        "direct_layout",
        native,
        tmp_path / "direct_geometry_out",
        {
            "asset_manifest": {
                "chair_1": {
                    "mesh_uri": "/assets/chair.glb",
                    "bbox_size": [0.75, 0.65, 0.95],
                    "bbox_center_local": [0.1, -0.2, 0.3],
                    "canonical_front": [0.0, 1.0, 0.0],
                }
            }
        },
    )

    obj = scene["objects"][0]
    assert obj["geometry_provenance"] == "bbox_proxy"
    assert obj["asset_proxy"]["bbox_center_local"] == [0.0, 0.0, 0.0]
    audit = obj["metadata"]["geometry_audit"]
    assert audit["evaluated_obb_size"] == [0.8, 0.7, 1.0]
    assert audit["asset_local_bbox_size"] == [0.75, 0.65, 0.95]
    assert audit["asset_local_bbox_center"] == [0.1, -0.2, 0.3]
    assert audit["mesh_uri_available"] is True
    assert audit["mesh_used_for_evaluation"] is False
    assert obj["metadata"]["canonical_front"] == [0.0, 1.0, 0.0]
    assert obj["metadata"]["canonical_front_source"] == "asset_metadata"


def test_layout_vlm_separates_local_bbox_scale_and_evaluated_dimensions(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "layout_vlm_scaled.json",
        {
            "layout": {
                "chair-asset-0": {
                    "position": [12.0, 22.0, 2.5],
                    "rotation": [0.0, 0.0, 5.0],
                    "scale": [2.0, 0.5, 1.5],
                }
            },
            "scene_config": {
                "boundary": {
                    "floor_vertices": [
                        [14, 25, 2],
                        [14, 20, 2],
                        [10, 20, 2],
                        [10, 25, 2],
                    ],
                    "wall_height": 3.0,
                },
                "assets": {
                    "chair-asset-0": {
                        "uid": "chair-asset",
                        "category": "chair",
                        "path": "/assets/chair-asset.glb",
                        "bbox_center_local": [0.1, 0.2, 0.3],
                        "frontView": 2,
                        "assetMetadata": {
                            "boundingBox": {"x": 1.0, "y": 2.0, "z": 1.0}
                        },
                    }
                },
            },
        },
    )
    scene = _materialize(
        "layout_vlm",
        native,
        tmp_path / "layout_vlm_scaled_out",
        {
            "asset_manifest": {
                "chair-asset": {
                    "category": "conflicting-provider-category",
                    "bbox_size": [9.0, 9.0, 9.0],
                }
            }
        },
    )

    obj = scene["objects"][0]
    assert scene["boundary"] == [
        [4.0, 5.0],
        [4.0, 0.0],
        [0.0, 0.0],
        [0.0, 5.0],
    ]
    assert obj["category"] == "chair"
    assert obj["center"] == pytest.approx([2.0, 2.0, 0.5])
    assert obj["size"] == pytest.approx([2.0, 1.0, 1.5])
    assert obj["geometry_provenance"] == "bbox_proxy"
    assert obj["asset_proxy"]["bbox_center_local"] == [0.0, 0.0, 0.0]
    audit = obj["metadata"]["geometry_audit"]
    assert audit["asset_local_bbox_size"] == [1.0, 2.0, 1.0]
    assert audit["asset_local_bbox_center"] == [0.1, 0.2, 0.3]
    assert audit["native_scale"] == [2.0, 0.5, 1.5]
    assert audit["applied_scale"] == [2.0, 0.5, 1.5]
    assert audit["scale_source"] == "native.scale"
    assert audit["evaluated_obb_size"] == [2.0, 1.0, 1.5]
    assert audit["mesh_used_for_evaluation"] is False
    assert obj["metadata"]["canonical_front"] == [1.0, 0.0, 0.0]
    assert obj["metadata"]["canonical_front_source"] == (
        "layoutvlm_processed_asset_contract"
    )
    assert obj["metadata"]["native_front_view"] == 2


@pytest.mark.parametrize("adapter_name", ["layout_vlm", "respace", "scene_weaver"])
def test_native_room_geometry_conflicts_are_rejected(
    adapter_name: str,
    tmp_path: Path,
) -> None:
    native, config = _conflicting_room_fixture(adapter_name, tmp_path / adapter_name)

    with pytest.raises(ArtifactValidationError, match="conflicts"):
        _materialize(
            adapter_name,
            native,
            tmp_path / f"{adapter_name}_room_conflict_out",
            config,
        )


@pytest.mark.parametrize(
    "adapter_name",
    ["direct_layout", "layout_vlm", "respace", "scene_weaver"],
)
def test_native_multi_room_structures_are_rejected(
    adapter_name: str,
    tmp_path: Path,
) -> None:
    native, config = _multi_room_fixture(adapter_name, tmp_path / adapter_name)

    with pytest.raises(ArtifactValidationError, match="single-room only"):
        _materialize(
            adapter_name,
            native,
            tmp_path / f"{adapter_name}_multi_room_out",
            config,
        )


def _materialize(
    adapter_name: str,
    native_path: Path,
    out_dir: Path,
    config: dict | None = None,
) -> dict:
    result = get_adapter(adapter_name).materialize_output(
        native_path,
        _generation_input(),
        out_dir,
        config=config,
    )
    return read_json(result)


def _generation_input() -> dict:
    return build_direct_natural_language_generation_input(
        request_id="converter-correctness",
        instruction="Design a furnished room.",
        scene_type="room",
        room={"boundary": [[0, 0], [4, 0], [4, 5], [0, 5]], "height": 3.0},
        evaluator_output_type=O1_OBJECT_STATE,
    )


def _respace_artifact(
    path: Path,
    *,
    objects: list[dict],
    width: float = 4.0,
    depth: float = 5.0,
    floor_y: float = 0.0,
    height: float = 3.0,
) -> Path:
    half_width = width / 2.0
    half_depth = depth / 2.0
    bottom = [
        [-half_width, floor_y, half_depth],
        [half_width, floor_y, half_depth],
        [half_width, floor_y, -half_depth],
        [-half_width, floor_y, -half_depth],
    ]
    top = [[x, floor_y + height, z] for x, _, z in bottom]
    return write_json(
        path,
        {
            "room_type": "room",
            "bounds_bottom": bottom,
            "bounds_top": top,
            "objects": objects,
        },
    )


def _scene_weaver_iterations(root: Path) -> Path:
    for iteration in range(3):
        write_json(
            root / "record_scene" / f"layout_{iteration}.json",
            {
                "roomsize": [4, 5],
                "structure": {},
                "objects": {
                    "chair_0": {
                        "location": [1.0 + iteration, 2.0, 0.0],
                        "rotation": [0.0, 0.0, iteration * 0.1],
                        "size": [0.8, 0.8, 1.0],
                        "parent": [],
                    }
                },
            },
        )
    return root


def _scene_weaver_config() -> dict:
    return {
        "asset_bindings": {
            "chair_0": {
                "asset_key": "sceneweaver-chair",
                "category": "chair",
            }
        }
    }


def _conflicting_room_fixture(
    adapter_name: str,
    root: Path,
) -> tuple[Path, dict]:
    if adapter_name == "layout_vlm":
        return (
            write_json(
                root / "layout.json",
                {
                    "layout": {},
                    "scene_config": {
                        "boundary": {
                            "floor_vertices": [
                                [0, 0, 0],
                                [4, 0, 0],
                                [4, 4, 0],
                                [0, 4, 0],
                            ],
                            "wall_height": 3.0,
                        },
                        "assets": {},
                    },
                },
            ),
            {},
        )
    if adapter_name == "respace":
        return _respace_artifact(root / "scene.json", objects=[], depth=4.0), {}
    if adapter_name == "scene_weaver":
        write_json(
            root / "record_scene" / "layout_0.json",
            {"roomsize": [4, 4], "structure": {}, "objects": {}},
        )
        return root, {"selected_iteration": 0}
    raise AssertionError(adapter_name)


def _multi_room_fixture(
    adapter_name: str,
    root: Path,
) -> tuple[Path, dict]:
    rooms = [{"id": "room_a"}, {"id": "room_b"}]
    if adapter_name == "direct_layout":
        return write_json(root / "layout.json", {"rooms": rooms, "objects": []}), {}
    if adapter_name == "layout_vlm":
        return (
            write_json(
                root / "layout.json",
                {"layout": {}, "rooms": rooms, "scene_config": {}},
            ),
            {},
        )
    if adapter_name == "respace":
        return write_json(root / "scene.json", {"rooms": rooms, "objects": []}), {}
    if adapter_name == "scene_weaver":
        write_json(
            root / "record_scene" / "layout_0.json",
            {"rooms": rooms, "roomsize": [4, 5], "objects": {}},
        )
        return root, {"selected_iteration": 0}
    raise AssertionError(adapter_name)
