from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.data import DATASET_ADAPTERS, discover_and_normalize_cases
from benchmark.models import MODEL_ADAPTERS, create_model
from benchmark.pipeline import run_case_pipeline
from benchmark.run_config import load_resolved_run_config, pipeline_resources_from_resolved
from benchmark.utils.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def test_component_experiment_resolves_and_saves_config(tmp_path: Path) -> None:
    resolved = load_resolved_run_config(
        ROOT,
        experiment_config_path="configs/experiments/hssd_small_room_mock.yaml",
    )

    assert resolved.data["experiment_name"] == "hssd_small_room_mock"
    assert resolved.data["config_refs"]["dataset"].endswith("configs/datasets/hssd_downloaded_small.yaml")
    assert resolved.data["config_hash"]
    assert resolved.model_config["models"]["mock"]["judge_evidence_budgeting"] is False
    assert resolved.benchmark_config["evaluation"]["evaluator"] == "vlm_as_judge_v1"

    resources = pipeline_resources_from_resolved(ROOT, resolved)
    dataset_cases = discover_and_normalize_cases(_absolute_dataset_config(resolved.dataset_config))
    case_ref, input_json = dataset_cases[0]
    state = run_case_pipeline(
        case_path=case_ref.path,
        input_json=input_json,
        out_dir=tmp_path,
        model_name=resolved.model_name,
        resources=resources,
        max_repair_iterations=0,
    )

    report = state["current_evaluation"]
    assert (tmp_path / "resolved_run_config.json").exists()
    assert (tmp_path / "config_hash.txt").exists()
    assert report["config_hash"] == resolved.data["config_hash"]
    assert report["config_refs"]["model"].endswith("configs/models/mock.yaml")
    assert (tmp_path / "viewer_scene.json").exists()


def test_cli_style_overrides_patch_selected_model() -> None:
    resolved = load_resolved_run_config(
        ROOT,
        experiment_config_path="configs/experiments/hssd_small_room_qwen3vl32b_h200.yaml",
        model_overrides={
            "model_endpoint": "http://127.0.0.1:9999/v1",
            "model_id": "local/override",
            "temperature": 0.25,
            "max_tokens": 123,
            "timeout_seconds": 456,
            "response_format_json": False,
        },
    )

    model_def = resolved.model_config["models"]["qwen3vl_sglang_32b"]
    assert model_def["endpoint"] == "http://127.0.0.1:9999/v1"
    assert model_def["model"] == "local/override"
    assert model_def["temperature"] == 0.25
    assert model_def["max_tokens"] == 123
    assert model_def["timeout_seconds"] == 456
    assert model_def["response_format_json"] is False


def test_dataset_registry_json_folder_and_hssd_normalization(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    write_json(case_path, {"case_id": "case", "room": {"boundary": [[0, 0], [1, 0], [1, 1], [0, 1]]}, "objects": []})

    assert "json_folder" in DATASET_ADAPTERS
    json_cases = discover_and_normalize_cases({"source_type": "json_folder", "path": str(case_path)})
    assert json_cases[0][1]["task_id"] == "case"
    assert json_cases[0][1]["scene_representation_mode"] == "compact_objects"

    hssd_path = tmp_path / "demo.scene_instance.json"
    write_json(
        hssd_path,
        {
            "stage_instance": {"template_name": "demo_stage"},
            "object_instances": [
                {"name": "chair_1", "template_name": "chair.object_config.json", "translation": [1, 0.5, 2], "scale": [0.5, 0.6, 0.8]}
            ],
        },
    )
    hssd_cases = discover_and_normalize_cases(
        {
            "source_type": "hssd_scene_instance_json",
            "path": str(hssd_path),
            "scene_representation_mode": "full_metadata_budgeted",
        }
    )
    normalized = hssd_cases[0][1]
    assert normalized["case_id"] == "demo"
    assert normalized["scene_representation_mode"] == "full_metadata_budgeted"
    assert normalized["objects"][0]["id"] == "chair_1"
    assert normalized["objects"][0]["source_floor_position"] == [1.0, 2.0]
    assert normalized["room"]["boundary_source_kind"] == "object_position_extent_fallback"
    assert normalized["room"]["geometry_fidelity"] == "proxy_rectangle"
    assert normalized["source"]["mesh_imported"] is False
    assert normalized["source"]["room_geometry_fidelity"] == "proxy_rectangle"


def test_unknown_dataset_and_model_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset source_type"):
        discover_and_normalize_cases({"source_type": "missing", "path": "."})

    assert "mock" in MODEL_ADAPTERS
    assert "openai_compatible" in MODEL_ADAPTERS
    with pytest.raises(ValueError, match="Unsupported model provider"):
        create_model("bad", {"models": {"bad": {"provider": "missing"}}})


def test_missing_component_defaults_are_rejected(tmp_path: Path) -> None:
    experiment_path = tmp_path / "bad.yaml"
    experiment_path.write_text("name: bad\ndefaults:\n  model: mock\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires defaults.dataset"):
        load_resolved_run_config(ROOT, experiment_config_path=experiment_path)


def _absolute_dataset_config(dataset_config: dict) -> dict:
    config = dict(dataset_config)
    for key in ["path", "root", "cases_dir", "case", "source_path"]:
        if config.get(key):
            path = Path(config[key])
            config[key] = str(path if path.is_absolute() else ROOT / path)
            break
    return config
