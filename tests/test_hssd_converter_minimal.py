from __future__ import annotations

from pathlib import Path

from jsonschema import Draft202012Validator

from benchmark.legend.hssd.hssd_hab_converter import convert_hssd_hab
from benchmark.utils.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def test_hssd_converter_writes_prompt_and_structured_basic_cases(tmp_path: Path) -> None:
    hssd_root = tmp_path / "hssd"
    scene_dir = hssd_root / "scenes"
    scene_dir.mkdir(parents=True)
    write_json(
        scene_dir / "tiny.scene_instance.json",
        {
            "scene_id": "tiny_scene",
            "object_instances": [
                {"id": "chair_001", "category": "chair", "translation": [0.0, 0.0, 0.0]},
                {"id": "desk_001", "category": "desk", "translation": [1.0, 0.0, 0.0]},
            ],
        },
    )

    out_dir = tmp_path / "cases"
    paths = convert_hssd_hab(hssd_root=hssd_root, out_dir=out_dir, limit=1, levels=["prompt_only", "structured_basic"])

    assert len(paths) == 2
    validator = Draft202012Validator(read_json(ROOT / "schemas" / "bm_instance.schema.json"))
    for path in paths:
        assert list(validator.iter_errors(read_json(path))) == []


def test_hssd_converter_can_write_small_compact_case(tmp_path: Path) -> None:
    hssd_root = tmp_path / "hssd"
    scene_dir = hssd_root / "scenes"
    scene_dir.mkdir(parents=True)
    write_json(
        scene_dir / "large.scene_instance.json",
        {
            "scene_id": "large_scene",
            "object_instances": [
                {"template_name": "hash_a", "translation": [0.0, 0.0, 0.0]},
                {"template_name": "hash_b", "translation": [1.0, 0.0, 2.0]},
                {"template_name": "hash_c", "translation": [2.0, 0.0, 3.0]},
            ],
        },
    )

    out_dir = tmp_path / "cases"
    paths = convert_hssd_hab(
        hssd_root=hssd_root,
        out_dir=out_dir,
        limit=1,
        levels=["structured_relation"],
        max_objects=2,
        compact_object_ids=True,
    )

    assert len(paths) == 1
    case = read_json(paths[0])
    assert [obj["id"] for obj in case["objects"]] == ["object_001", "object_002"]
    assert [obj["category"] for obj in case["objects"]] == ["hssd_object_001", "hssd_object_002"]
    assert case["objects"][0]["source_id"] == "hash_a_001"
    assert len(case["objects"]) == 2
    assert len(case["relations"]) == 1
    assert len(case["spatial_cues"]) == 1
    assert case["relations"][0]["subject"] == "object_001"
    assert case["spatial_cues"][0]["source"] == "bbox_geometry_heuristic"
    assert case["spatial_cues"][0]["hard"] is False
    assert case["spatial_cues"][0]["confidence"] > 0
    assert case["room"]["boundary"] == [[-1.5, -1.5], [2.5, -1.5], [2.5, 3.5], [-1.5, 3.5]]
    assert case["room"]["boundary_source_kind"] == "object_position_extent_fallback"
    assert case["room"]["geometry_fidelity"] == "proxy_rectangle"
    assert case["room"]["is_proxy_geometry"] is True
    assert case["scene_representation_mode"] == "compact_objects_with_estimated_relations"
    assert case["source"]["input_representation_mode"] == "compact_objects_with_estimated_relations"
    assert case["source"]["mesh_imported"] is False
    assert case["source"]["mesh_free_import"] is True
    assert case["source"]["room_boundary_source_kind"] == "object_position_extent_fallback"
    assert case["source"]["room_geometry_fidelity"] == "proxy_rectangle"
    assert case["source"]["relation_policy"] == "deterministic_estimated_spatial_cues_v1"
    assert case["source"]["relations_are_ground_truth"] is False

    validator = Draft202012Validator(read_json(ROOT / "schemas" / "bm_instance.schema.json"))
    assert list(validator.iter_errors(case)) == []


def test_hssd_converter_imports_available_mesh_free_metadata(tmp_path: Path) -> None:
    hssd_root = tmp_path / "hssd"
    scene_dir = hssd_root / "scenes-uncluttered"
    scene_dir.mkdir(parents=True)
    write_json(
        scene_dir / "semantic.scene_instance.json",
        {
            "scene_id": "semantic_scene",
            "stage_instance": {"template_name": "stages/semantic_stage"},
            "semantic_scene_instance": "semantic_scene",
            "user_defined": {"scene_filter_file": "scene_filter_files/semantic.rec_filter.json"},
            "object_instances": [
                {"template_name": "chair_hash", "translation": [0.0, 0.0, 0.0], "non_uniform_scale": [0.6, 0.9, 0.5]},
            ],
        },
    )
    write_json(
        hssd_root / "objects" / "chairs" / "chair_hash.object_config.json",
        {"category": "chair", "dimensions": [0.7, 0.95, 0.55], "user_defined": {"semantic_category": "dining_chair"}},
    )
    write_json(
        hssd_root / "stages" / "semantic_stage.stage_config.json",
        {"floor_polygon": [[0, 0], [3, 0], [3, 2], [0, 2]], "wall_height": 2.7},
    )
    write_json(
        hssd_root / "semantics" / "scenes" / "semantic_scene.semantic_config.json",
        {
            "region_annotations": [
                {
                    "name": "kitchen_region",
                    "label": "kitchen",
                    "floor_height": 0.0,
                    "extrusion_height": 2.8,
                    "min_bounds": [0, 0, 0],
                    "max_bounds": [2, 2.8, 3],
                    "poly_loop": [[0, 0, 0], [2, 0, 0], [2, 0, 3], [0, 0, 3]],
                }
            ]
        },
    )
    (hssd_root / "semantics" / "scenes" / "semantic_scene_floor_plan.png").parent.mkdir(parents=True, exist_ok=True)
    (hssd_root / "semantics" / "scenes" / "semantic_scene_floor_plan.png").write_bytes(b"not-a-real-png")
    write_json(hssd_root / "scene_filter_files" / "semantic.rec_filter.json", {"receptacles": []})
    write_json(hssd_root / "semantics" / "hssd-hab_semantic_lexicon.json", {"chair": {"label": "chair"}})
    write_json(hssd_root / "hssd-hab-uncluttered.scene_dataset_config.json", {"scene_instances": {"paths": {".json": ["scenes-uncluttered"]}}})
    metadata_dir = hssd_root / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "objects.csv").write_text("template_name,category\nchair_hash,chair\n", encoding="utf-8")

    paths = convert_hssd_hab(
        hssd_root=hssd_root,
        out_dir=tmp_path / "cases",
        levels=["structured_basic"],
        bbox_from_scale=True,
    )

    case = read_json(paths[0])
    assert case["room"]["boundary_source"] == "hssd_semantic_config.region_annotations.poly_loop"
    assert case["room"]["boundary_source_kind"] == "hssd_semantic_region_polygon"
    assert case["room"]["geometry_fidelity"] == "semantic_floor_plan"
    assert case["room"]["is_proxy_geometry"] is False
    assert case["room"]["boundary_role"] == "aggregate_proxy"
    assert case["room"]["wall_height"] == 2.7
    assert case["room"]["floor_plan"]["primary_representation"] == "regions"
    assert case["room"]["floor_plan"]["aggregate_boundary_role"] == "compatibility_proxy"
    assert case["room"]["floor_plan"]["regions"][0]["label"] == "kitchen"
    assert case["room"]["floor_plan"]["regions"][0]["floor_polygon"] == [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]]
    assert case["room"]["supporting_visuals"][0]["kind"] == "floor_plan_or_map"
    assert case["objects"][0]["category"] == "chair"
    assert case["objects"][0]["semantic_category"] == "chair"
    assert case["objects"][0]["bbox_size_source"] == "object_config.dimensions"
    assert case["objects"][0]["layout_center_hint"] == [0.0, 0.0, 0.475]
    assert case["objects"][0]["layout_center_hint_source"] == "hssd_translation_xz_plus_height_center_hint"
    assert case["objects"][0]["source_region_id"] == "kitchen_region"
    assert case["objects"][0]["region_assignment_source"] == "semantic_region_polygon"
    assert case["objects"][0]["region_assignment_confidence"] == 1.0
    assert case["objects"][0]["source_asset_references"]["object_config.source_config_path"] == "objects/chairs/chair_hash.object_config.json"
    assert case["source"]["scene_instance_fields"] == [
        "object_instances",
        "scene_id",
        "semantic_scene_instance",
        "stage_instance",
        "user_defined",
    ]
    assert case["source"]["room_layout_source"] == "semantics/scenes/*.semantic_config.json region_annotations"
    assert case["source"]["metadata_inclusion"]["stage_config"] is True
    assert case["source"]["metadata_inclusion"]["semantic_scene_config"] is True
    assert case["source"]["metadata_inclusion"]["scene_filter"] is True
    assert case["source"]["metadata_inclusion"]["scene_dataset_configs"] == 1
    assert case["source"]["mesh_free_import"] is True
    assert case["source"]["mesh_imported"] is False
    assert case["source"]["mesh_asset_references_kept"] is True
    assert case["source"]["room_boundary_source_kind"] == "hssd_semantic_region_polygon"
    assert case["source"]["room_geometry_fidelity"] == "semantic_floor_plan"
    assert case["source"]["missing_metadata"] == []


def test_hssd_converter_can_store_full_metadata_budgeted_mode(tmp_path: Path) -> None:
    hssd_root = tmp_path / "hssd"
    scene_dir = hssd_root / "scenes"
    scene_dir.mkdir(parents=True)
    write_json(
        scene_dir / "full.scene_instance.json",
        {
            "scene_id": "full_scene",
            "stage_instance": {"template_name": "stages/full_stage"},
            "object_instances": [
                {
                    "template_name": "chair_hash",
                    "translation": [1.0, 0.5, 2.0],
                    "rotation": [0, 0, 0, 1],
                    "non_uniform_scale": [0.6, 0.9, 0.5],
                    "motion_type": "static",
                },
            ],
        },
    )
    write_json(hssd_root / "stages" / "full_stage.stage_config.json", {"render_asset": "full_stage.glb"})

    paths = convert_hssd_hab(
        hssd_root=hssd_root,
        out_dir=tmp_path / "cases",
        levels=["structured_basic"],
        input_representation_mode="full_metadata_budgeted",
        preserve_raw_metadata=True,
        bbox_from_scale=True,
    )

    case = read_json(paths[0])
    assert case["scene_representation_mode"] == "full_metadata_budgeted"
    assert case["source"]["input_representation_mode"] == "full_metadata_budgeted"
    assert case["source"]["mesh_asset_references_kept"] is True
    assert case["source"]["stage_asset_references"]["render_asset"] == "full_stage.glb"
    assert case["objects"][0]["source_rotation"] == [0, 0, 0, 1]
    assert case["objects"][0]["source_non_uniform_scale"] == [0.6, 0.9, 0.5]
    assert case["objects"][0]["raw_hssd_instance"]["template_name"] == "chair_hash"
