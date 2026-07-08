from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmark.data.scene_adapters import layout_to_scene, normalize_scene, scene_adapter_summary, scene_to_layout
from benchmark.data.local_assets import resolve_local_asset_ref
from benchmark.data.local_scenes import load_local_scene, load_local_scene_index, resolve_local_scene_ref
from benchmark.pipeline import PipelineResources, evaluate_scene_pipeline, run_case_pipeline
from benchmark.utils.io import read_json, write_json
from benchmark.workflow.evaluate import evaluate_scene


ROOT = Path(__file__).resolve().parents[1]
HSSD_CASE = ROOT / "data" / "benchmark_cases" / "hssd_small" / "102343992_structured_relation.json"
LOCAL_ASSET_ROOT = ROOT / "Assets" / "imaginarium_assets"
LOCAL_SCENE_JSON = ROOT / "Scenes" / "converted_scenes" / "scene_000000_03.json"


def _require_local_assets() -> None:
    if not (LOCAL_ASSET_ROOT / "0_SM_Chair_1").exists():
        pytest.skip("local Assets/ directory is not checked into this repo")


def _require_local_scenes() -> None:
    if not LOCAL_SCENE_JSON.exists():
        pytest.skip("local Scenes/ directory is not checked into this repo")


def _has_local_assets() -> bool:
    return (LOCAL_ASSET_ROOT / "0_SM_Chair_1").exists()


def _scene_schema() -> dict:
    return read_json(ROOT / "schemas" / "scene.schema.json")


def _assert_valid_scene(scene: dict) -> None:
    errors = sorted(Draft202012Validator(_scene_schema()).iter_errors(scene), key=lambda item: list(item.path))
    assert [error.message for error in errors] == []


def test_scene_schema_accepts_minimal_scene() -> None:
    _assert_valid_scene(
        {
            "scene_id": "minimal_scene",
            "assets": [{"asset_id": "a001", "category": "chair"}],
        }
    )



def test_local_repo_asset_ref_resolves_existing_asset() -> None:
    _require_local_assets()
    ref = resolve_local_asset_ref({"source": "local_repo", "asset_id": "0_SM_Chair_1"})

    assert ref is not None
    assert ref["collection"] == "imaginarium_assets"
    assert ref["repo_path"] == "Assets/imaginarium_assets/0_SM_Chair_1"
    assert ref["mesh_path"].endswith("0_SM_Chair_1.fbx")
    assert ref["pointcloud_path"].endswith("0_SM_Chair_1.ply")
    assert ref["metadata_path"].endswith("0_SM_Chair_1_metadata.json")
    assert ref["dimensions"] == [0.646512, 0.637464, 0.850842]


def test_normalize_scene_enriches_local_repo_asset_ref() -> None:
    _require_local_assets()
    scene = normalize_scene(
        {
            "scene_id": "local_asset_scene",
            "assets": [
                {
                    "asset_id": "chair_instance_1",
                    "category": "chair",
                    "asset_ref": {"source": "local_repo", "asset_id": "0_SM_Chair_1"},
                }
            ],
        }
    )

    asset = scene["assets"][0]
    assert asset["dimensions"] == [0.646512, 0.637464, 0.850842]
    assert asset["asset_ref"]["repo_path"] == "Assets/imaginarium_assets/0_SM_Chair_1"
    assert asset["asset_ref"]["mesh_path"].endswith("0_SM_Chair_1.fbx")
    assert asset["asset_ref"]["pointcloud_path"].endswith("0_SM_Chair_1.ply")
    _assert_valid_scene(scene)


def test_local_repo_scene_ref_resolves_existing_scene() -> None:
    _require_local_scenes()
    ref = resolve_local_scene_ref({"source": "local_repo", "scene_id": "scene_000000_03"})

    assert ref is not None
    assert ref["collection"] == "Scenes"
    assert ref["scene_id"] == "scene_000000_03"
    assert ref["scene_json_path"] == "Scenes/converted_scenes/scene_000000_03.json"
    assert ref["repo_path"] == "Scenes/converted_scenes"
    assert ref["scene_type"] == "living room"
    assert ref["asset_count"] > 0
    assert "scene_000000_03" in load_local_scene_index(ROOT)


def test_load_local_scene_reads_existing_scene_json() -> None:
    _require_local_scenes()
    loaded = load_local_scene({"source": "local_repo", "scene_json_path": "Scenes/converted_scenes/scene_000000_03.json"})

    assert loaded is not None
    assert loaded["scene_id"] == "scene_000000_03"
    assert loaded["scene_ref"]["scene_json_path"] == "Scenes/converted_scenes/scene_000000_03.json"
    assert loaded["objects"][0]["jid"] == "b_58"


def test_normalize_scene_loads_local_repo_scene_ref() -> None:
    _require_local_scenes()
    scene = normalize_scene(
        {
            "scene_id": "scene_000000_03",
            "scene_ref": {"source": "local_repo", "scene_id": "scene_000000_03"},
        }
    )

    assert scene["scene_id"] == "scene_000000_03"
    assert scene["scene_ref"]["scene_json_path"] == "Scenes/converted_scenes/scene_000000_03.json"
    assert scene["room"]["boundary"][2] == [7.03, 10.64]
    assert scene["room"]["wall_height"] == 2.957
    asset = scene["assets"][0]
    assert asset["asset_id"] == "b_58"
    assert asset["object_id"] == "b_58"
    assert asset["placement"]["position"] == [1.86, 8.26, 0.5]
    assert asset["placement"]["yaw_degrees"] == 90.0
    assert len(scene["assets"]) == scene["scene_ref"]["asset_count"]
    summary = scene_adapter_summary(scene)
    assert summary["local_scene_ref_available"] is True
    assert summary["local_scene_id"] == "scene_000000_03"
    assert summary["local_scene_json_path"] == "Scenes/converted_scenes/scene_000000_03.json"
    _assert_valid_scene(scene)


def test_scene_schema_accepts_local_scene_ref_input() -> None:
    _assert_valid_scene(
        {
            "scene_id": "scene_000000_03",
            "scene_ref": {"source": "local_repo", "scene_id": "scene_000000_03"},
        }
    )


def test_scene_schema_accepts_local_scene_objects_import_shape() -> None:
    _require_local_scenes()
    raw_scene = read_json(LOCAL_SCENE_JSON)

    _assert_valid_scene(
        {
            "scene_id": raw_scene["scene_id"],
            "scene_type": raw_scene["scene_type"],
            "boundary": raw_scene["boundary"],
            "scene_height": raw_scene["scene_height"],
            "objects": [raw_scene["objects"][0]],
        }
    )

def test_scene_schema_accepts_placement_dimensions_and_flexible_asset_ref() -> None:
    _assert_valid_scene(
        {
            "scene_id": "asset_ref_scene",
            "unit": "meter",
            "assets": [
                {
                    "asset_id": "chair_1",
                    "category": "chair",
                    "placement": {"position": [1, 2, 0.45], "yaw_degrees": 15},
                    "dimensions": [0.6, 0.6, 0.9],
                    "support_parent": "floor",
                    "region_id": "work_zone",
                    "asset_ref": {
                        "source": "local_repo",
                        "collection": "imaginarium_assets",
                        "asset_id": "0_SM_Chair_1",
                        "repo_path": "Assets/imaginarium_assets/0_SM_Chair_1",
                        "mesh_path": "Assets/imaginarium_assets/0_SM_Chair_1/0_SM_Chair_1.fbx",
                        "pointcloud_path": "Assets/imaginarium_assets/0_SM_Chair_1/0_SM_Chair_1.ply",
                        "metadata_path": "Assets/imaginarium_assets/0_SM_Chair_1/0_SM_Chair_1_metadata.json",
                        "metadata": {"note": "repo asset already exists"},
                    },
                }
            ],
        }
    )


def test_scene_schema_rejects_bbox_on_canonical_asset() -> None:
    scene = {
        "scene_id": "bbox_scene",
        "assets": [
            {
                "asset_id": "chair_1",
                "category": "chair",
                "bbox": {"center": [1, 2, 0.45], "size": [0.6, 0.6, 0.9], "yaw": 15},
            }
        ],
    }
    errors = sorted(Draft202012Validator(_scene_schema()).iter_errors(scene), key=lambda item: list(item.path))
    assert errors


def test_layout_to_scene_converts_legend_objects_to_assets() -> None:
    layout = {
        "scene_id": "legacy_scene",
        "unit": "meter",
        "objects": [
            {
                "object_id": "chair_1",
                "category": "chair",
                "center": [1, 2, 0.45],
                "size": [0.6, 0.6, 0.9],
                "yaw": 0,
                "support_parent": "floor",
                "region_id": "work_zone",
                "canonical_object_id": "chair_canonical",
                "model_object_id": "chair_alias_1",
            }
        ],
    }
    case = {
        "case_id": "legacy_scene",
        "room": {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        "objects": [
            {
                "id": "chair_1",
                "category": "chair",
                "asset_ref": {"source": "local_repo", "asset_id": "0_SM_Chair_1"},
            }
        ],
        "relations": [{"id": "rel_1", "type": "near", "subject": "chair_1", "object": "desk_1"}],
    }

    scene = layout_to_scene(layout, case)

    asset = scene["assets"][0]
    assert scene["scene_id"] == "legacy_scene"
    assert scene["room"] == case["room"]
    assert scene["relations"] == case["relations"]
    assert asset["asset_id"] == "chair_1"
    assert asset["object_id"] == "chair_1"
    assert asset["placement"] == {"position": [1, 2, 0.45], "yaw_degrees": 0}
    assert asset["dimensions"] == [0.6, 0.6, 0.9]
    assert asset["support_parent"] == "floor"
    assert asset["region_id"] == "work_zone"
    assert asset["canonical_object_id"] == "chair_canonical"
    assert asset["model_object_id"] == "chair_alias_1"
    assert asset["asset_ref"]["source"] == "local_repo"
    assert asset["asset_ref"]["asset_id"] == "0_SM_Chair_1"
    if _has_local_assets():
        assert asset["asset_ref"]["mesh_path"].endswith("0_SM_Chair_1.fbx")
    assert asset["metadata"]["source_layout_object"]["object_id"] == "chair_1"


def test_scene_to_layout_converts_asset_geometry_to_legend_objects() -> None:
    scene = {
        "scene_id": "candidate_scene",
        "assets": [
            {
                "asset_id": "desk_1",
                "category": "desk",
                "placement": {
                    "position": [2, 1, 0.4],
                    "yaw_degrees": 90,
                    "support_parent": "floor",
                    "region_id": "work_zone",
                },
                "dimensions": [1.2, 0.6, 0.8],
                "asset_ref": {"source": "local_repo", "asset_id": "0_SM_Desk_Fir001"},
            }
        ],
    }

    layout = scene_to_layout(scene)

    assert layout["scene_id"] == "candidate_scene"
    assert layout["coordinate_system"]["rotation_unit"] == "degree"
    obj = layout["objects"][0]
    assert obj["object_id"] == "desk_1"
    assert obj["category"] == "desk"
    assert obj["center"] == [2, 1, 0.4]
    assert obj["size"] == [1.2, 0.6, 0.8]
    assert obj["yaw"] == 90
    assert obj["asset_id"] == "desk_1"
    assert obj["asset_ref"]["source"] == "local_repo"
    assert obj["asset_ref"]["asset_id"] == "0_SM_Desk_Fir001"
    if _has_local_assets():
        assert obj["asset_ref"]["mesh_path"].endswith("0_SM_Desk_Fir001.fbx")
    assert obj["support_parent"] == "floor"
    assert obj["region_id"] == "work_zone"


def test_asset_without_geometry_is_preserved_in_scene_and_skipped_in_layout() -> None:
    scene = {
        "scene_id": "mixed_scene",
        "assets": [
            {
                "asset_id": "box_1",
                "category": "box",
                "placement": {"position": [0, 0, 0.5], "yaw_degrees": 0},
                "dimensions": [1, 1, 1],
            },
            {
                "asset_id": "mesh_only_1",
                "category": "plant",
                "asset_ref": {"source": "local_repo", "asset_id": "0_SM_Chair_1"},
                "metadata": {"note": "no placement yet"},
            },
        ],
    }

    layout = scene_to_layout(scene)

    assert len(scene["assets"]) == 2
    assert [obj["object_id"] for obj in layout["objects"]] == ["box_1"]
    assert layout["_non_geometric_assets"][0]["asset_id"] == "mesh_only_1"
    assert layout["_non_geometric_assets"][0]["reason"] == "asset has no complete placement/dimensions geometry"


def test_evaluate_scene_accepts_scene_without_generation(tmp_path: Path) -> None:
    scene = {
        "scene_id": "direct_eval_scene",
        "room": {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "floor_z": 0.0, "wall_height": 3.0},
        "assets": [
            {
                "asset_id": "desk_1",
                "category": "desk",
                "placement": {"position": [1, 1, 0.4], "yaw_degrees": 0},
                "dimensions": [1.0, 0.6, 0.8],
            },
            {
                "asset_id": "chair_1",
                "category": "chair",
                "placement": {"position": [1.8, 1, 0.45], "yaw_degrees": 180},
                "dimensions": [0.5, 0.5, 0.9],
            },
        ],
        "relations": [{"id": "rel_1", "type": "near", "subject": "chair_1", "object": "desk_1"}],
    }

    report, metrics = evaluate_scene(
        scene,
        out_dir=tmp_path,
        benchmark_config={"evaluation": {"vlm_judge": "mock"}},
    )

    assert report["evaluation_input"]["input_type"] == "scene"
    assert report["evaluation_input"]["geometry_asset_count"] == 2
    assert report["vlm_judge_input_mode"] == "json_only"
    assert report["render_evidence_used"] is False
    assert report["json_scene_used"] is True
    assert report["geometry_available_rate"] == 1.0
    assert report["overall_valid"] is True
    assert metrics["evaluation_input_type"] == "scene"
    assert metrics["scene_asset_count"] == 2
    assert metrics["geometry_available_rate"] == 1.0
    assert (tmp_path / "case_metrics_iter_0.json").exists()
    assert not (tmp_path / "views").exists()
    assert (tmp_path / "vlm_judge" / "iter_000" / "judge_prompt.json").exists()
    assert (tmp_path / "vlm_judge" / "iter_000" / "judge_input_manifest.json").exists()
    assert (tmp_path / "vlm_judge" / "iter_000" / "judge_request_metadata.json").exists()
    assert read_json(tmp_path / "vlm_judge" / "iter_000" / "judge_image_manifest.json") == []


def test_evaluate_scene_json_only_allows_missing_geometry_assets(tmp_path: Path) -> None:
    scene = {
        "scene_id": "metadata_only_scene",
        "room": {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "floor_z": 0.0, "wall_height": 3.0},
        "assets": [
            {
                "asset_id": "plant_1",
                "category": "plant",
                "asset_ref": {"source": "local_repo", "asset_id": "0_SM_Chair_1", "metadata": {"raw": "kept"}},
            }
        ],
    }

    report, metrics = evaluate_scene(
        scene,
        out_dir=tmp_path,
        benchmark_config={"evaluation": {"vlm_judge": "mock"}},
    )

    assert report["overall_valid"] is True
    assert report["hard_failures"] == []
    assert report["geometry_available_rate"] == 0.0
    assert metrics["geometry_available_rate"] == 0.0
    assert metrics["asset_ref_asset_count"] == 1
    assert metrics["asset_ref_available_rate"] == 1.0
    assert metrics["local_asset_ref_count"] == 1
    assert metrics["local_asset_available_rate"] == 1.0
    assert report["debug_evidence"]["geometry_missing_assets"][0]["asset_id"] == "plant_1"
    assert report["debug_evidence"]["render_skipped_objects"][0]["type"] == "geometry_missing_asset"
    assert report["render_evidence"]["global_views"] == []
    prompt = read_json(tmp_path / "vlm_judge" / "iter_000" / "judge_prompt.json")
    assert "plant_1" in prompt["user"]
    assert "chair under table" in prompt["system"]


def test_generation_workflow_still_runs_with_mock_model_through_candidate_scene(tmp_path: Path) -> None:
    resources = PipelineResources(
        model_config={"models": {"mock": {"provider": "mock", "name": "mock"}}},
        benchmark_config={"benchmark": {"save_viewer_scene": False}, "evaluation": {"vlm_judge": "mock"}},
        layout_schema=read_json(ROOT / "schemas" / "layout.schema.json"),
    )

    state = run_case_pipeline(
        case_path=HSSD_CASE,
        out_dir=tmp_path,
        model_name="mock",
        resources=resources,
        max_repair_iterations=0,
    )

    assert state["current_layout"]["objects"]
    assert state["current_scene"]["assets"]
    assert state["current_evaluation"]["evaluation_input"]["input_type"] == "scene"
    assert state["pipeline_mode"] == "generation"
    assert state["generation_used"] is True
    assert state["current_evaluation"]["pipeline_mode"] == "generation"
    assert state["current_evaluation"]["generation_used"] is True
    assert state["per_case_result"]["pipeline_mode"] == "generation"
    assert state["per_case_result"]["generation_used"] is True
    assert state["per_case_result"]["generated_layout_path"].endswith("initial_layout.json")
    assert state["per_case_result"]["candidate_scene_path"].endswith("candidate_scene.json")
    assert state["per_case_result"]["evaluation_report_path"].endswith("evaluation_report.json")
    assert state["per_case_result"]["metrics_path"].endswith("case_metrics.json")
    assert state["current_evaluation"]["vlm_judge_input_mode"] == "json_only"
    assert state["current_evaluation"]["render_evidence_used"] is False
    assert state["case_metrics"]["primary_score"] >= 0
    assert state["case_metrics"]["pipeline_mode"] == "generation"
    assert state["case_metrics"]["generation_used"] is True
    assert "feedback_issue_count" in state["case_metrics"]
    assert (tmp_path / "initial_layout.json").exists()
    assert (tmp_path / "candidate_scene.json").exists()
    assert (tmp_path / "evaluation_report.json").exists()


def test_evaluate_scene_pipeline_writes_direct_evaluation_artifacts(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    write_json(scene_path, _direct_scene())
    resources = _direct_eval_resources()

    state = evaluate_scene_pipeline(
        scene_path=scene_path,
        out_dir=tmp_path / "out",
        resources=resources,
        judge_model_name="mock",
        vlm_judge_input_mode="json_only",
    )

    out_dir = tmp_path / "out"
    report = read_json(out_dir / "evaluation_report.json")
    feedback = read_json(out_dir / "feedback.json")
    metrics = read_json(out_dir / "case_metrics.json")
    prompt = read_json(out_dir / "vlm_judge" / "iter_000" / "judge_prompt.json")

    assert state["pipeline_mode"] == "evaluation"
    assert state["generation_used"] is False
    assert report["pipeline_mode"] == "evaluation"
    assert report["generation_used"] is False
    assert report["vlm_judge_input_mode"] == "json_only"
    assert report["render_evidence_used"] is False
    assert report["scene_schema_version"] == "2.1.0"
    assert report["normalized_scene_path"] == "normalized_scene.json"
    assert report["feedback_path"] == "feedback.json"
    assert metrics["pipeline_mode"] == "evaluation"
    assert metrics["generation_used"] is False
    assert metrics["scene_schema_version"] == "2.1.0"
    assert metrics["scene_schema_valid"] is True
    assert metrics["input_schema_type"] == "scene"
    assert metrics["scene_id"] == "direct_pipeline_scene"
    assert metrics["scene_asset_count"] == 2
    assert metrics["geometry_asset_count"] == 2
    assert metrics["asset_ref_asset_count"] == 2
    assert metrics["asset_ref_available_rate"] == 1.0
    assert metrics["local_asset_ref_count"] == 2
    assert metrics["local_asset_available_rate"] == 1.0
    assert metrics["feedback_issue_count"] == len(feedback["issues"])
    assert feedback["scene_id"] == "direct_pipeline_scene"
    assert feedback["advisory"] is True
    assert {"overall_valid", "score", "issues", "repair_hints", "physical_evidence", "vlm_judge_feedback", "suggested_actions"} <= set(feedback)
    assert "asset_ref" in prompt["user"]
    assert "0_SM_Chair_1" in prompt["user"]
    assert read_json(out_dir / "normalized_scene.json")["scene_id"] == "direct_pipeline_scene"
    assert state["current_scene_path"].endswith("normalized_scene.json")
    assert (out_dir / "viewer_scene.json").exists()
    assert not (out_dir / "initial_layout.json").exists()
    assert not (out_dir / "generation_prompt.txt").exists()
    assert not (out_dir / "generation_raw_response.txt").exists()
    assert not (out_dir / "candidate_scene.json").exists()


def test_evaluate_scene_pipeline_accepts_legend_layout_json(tmp_path: Path) -> None:
    layout_path = tmp_path / "layout.json"
    write_json(
        layout_path,
        {
            "scene_id": "legacy_direct_layout",
            "unit": "meter",
            "coordinate_system": {
                "origin": "front-left floor corner",
                "x_axis": "room width",
                "y_axis": "room depth",
                "z_axis": "height",
                "rotation_unit": "degree",
            },
            "objects": [
                {
                    "object_id": "box_1",
                    "category": "box",
                    "center": [1, 1, 0.5],
                    "size": [1, 1, 1],
                    "yaw": 0,
                }
            ],
        },
    )

    state = evaluate_scene_pipeline(
        scene_path=layout_path,
        out_dir=tmp_path / "out",
        resources=_direct_eval_resources(),
        judge_model_name="mock",
        vlm_judge_input_mode="json_only",
    )

    metrics = read_json(tmp_path / "out" / "case_metrics.json")
    normalized_scene = read_json(tmp_path / "out" / "normalized_scene.json")
    assert state["pipeline_mode"] == "evaluation"
    assert metrics["input_schema_type"] == "legend_layout"
    assert metrics["scene_schema_valid"] is False
    assert normalized_scene["scene_id"] == "legacy_direct_layout"
    assert normalized_scene["assets"][0]["asset_id"] == "box_1"
    assert not (tmp_path / "out" / "generation_prompt.txt").exists()


def test_evaluate_scene_pipeline_json_plus_render_writes_views(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    write_json(scene_path, _direct_scene())
    resources = _direct_eval_resources(mode="json_plus_render")

    state = evaluate_scene_pipeline(
        scene_path=scene_path,
        out_dir=tmp_path / "out",
        resources=resources,
        judge_model_name="mock",
        vlm_judge_input_mode="json_plus_render",
    )

    report = state["current_evaluation"]
    out_dir = tmp_path / "out"
    image_manifest = read_json(out_dir / "vlm_judge" / "iter_000" / "judge_image_manifest.json")

    assert report["vlm_judge_input_mode"] == "json_plus_render"
    assert report["render_evidence_used"] is True
    assert image_manifest
    assert (out_dir / "views" / "global" / "topdown_global_xy.png").exists()
    assert (out_dir / "viewer_scene.json").exists()


def test_evaluate_scene_cli_runs_direct_json_only_without_generation(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    out_dir = tmp_path / "cli_out"
    write_json(scene_path, _direct_scene())

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_scene.py"),
            "--scene",
            str(scene_path),
            "--out",
            str(out_dir),
            "--judge-model",
            "mock",
            "--no-render",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = read_json(out_dir / "evaluation_report.json")
    assert report["pipeline_mode"] == "evaluation"
    assert report["generation_used"] is False
    assert report["vlm_judge_input_mode"] == "json_only"
    assert (out_dir / "normalized_scene.json").exists()
    assert (out_dir / "feedback.json").exists()
    assert (out_dir / "case_metrics.json").exists()
    assert not (out_dir / "initial_layout.json").exists()


def _direct_eval_resources(mode: str = "json_only") -> PipelineResources:
    return PipelineResources(
        model_config={"models": {"mock": {"provider": "mock", "name": "mock"}}},
        benchmark_config={
            "benchmark": {"save_viewer_scene": True},
            "evaluation": {"vlm_judge": "mock", "vlm_judge_input_mode": mode},
        },
        layout_schema=read_json(ROOT / "schemas" / "layout.schema.json"),
        scene_schema=read_json(ROOT / "schemas" / "scene.schema.json"),
    )


def _direct_scene() -> dict:
    return {
        "scene_id": "direct_pipeline_scene",
        "unit": "meter",
        "room": {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "floor_z": 0.0, "wall_height": 3.0},
        "assets": [
            {
                "asset_id": "chair_1",
                "category": "chair",
                "asset_ref": {"source": "local_repo", "asset_id": "0_SM_Chair_1"},
                "placement": {"position": [1.0, 1.0, 0.45], "yaw_degrees": 180},
                "dimensions": [0.6, 0.6, 0.9],
            },
            {
                "asset_id": "desk_1",
                "category": "desk",
                "asset_ref": {"source": "local_repo", "asset_id": "0_SM_Desk_Fir001"},
                "placement": {"position": [2.1, 1.0, 0.4], "yaw_degrees": 0},
                "dimensions": [1.2, 0.6, 0.8],
            },
        ],
        "relations": [{"id": "rel_1", "type": "near", "subject": "chair_1", "object": "desk_1"}],
    }
