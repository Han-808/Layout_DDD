from __future__ import annotations

from pathlib import Path

from benchmark.pipeline import PipelineResources, run_case_pipeline
from benchmark.utils.io import read_json


ROOT = Path(__file__).resolve().parents[1]
HSSD_CASE = ROOT / "data" / "benchmark_cases" / "hssd_small" / "102343992_structured_relation.json"


def test_shared_pipeline_writes_v0_outputs(tmp_path: Path) -> None:
    resources = PipelineResources(
        model_config={"models": {"mock": {"provider": "mock", "name": "mock"}}},
        benchmark_config={"benchmark": {"save_viewer_scene": True}, "evaluation": {"vlm_judge": "mock", "vlm_judge_input_mode": "json_plus_render"}},
        layout_schema=read_json(ROOT / "schemas" / "layout.schema.json"),
    )

    state = run_case_pipeline(
        case_path=HSSD_CASE,
        out_dir=tmp_path,
        model_name="mock",
        resources=resources,
        max_repair_iterations=0,
    )

    assert (tmp_path / "evaluation_report.json").exists()
    assert (tmp_path / "case_metrics.json").exists()
    assert (tmp_path / "normalized_case.json").exists()
    assert (tmp_path / "prompt_payload.json").exists()
    assert (tmp_path / "object_alias_map.json").exists()
    assert (tmp_path / "eval_context_summary.json").exists()
    assert (tmp_path / "visibility_audit.json").exists()
    assert (tmp_path / "input_quality.json").exists()
    assert (tmp_path / "viewer_scene.json").exists()
    assert (tmp_path / "workflow_trace.json").exists()
    assert state["current_evaluation"]["evaluator"] == "vlm_as_judge_v1"
    assert (tmp_path / "views" / "global" / "topdown_global_xy.png").exists()
    assert (tmp_path / "views" / "groups" / "group_001" / "group_001_xy.png").exists()
    assert state["case_metrics"]["primary_score"] >= 0
    result = read_json(tmp_path / "per_case_result.json")
    prompt_payload = read_json(tmp_path / "prompt_payload.json")
    eval_context_summary = read_json(tmp_path / "eval_context_summary.json")
    assert result["input_level"] == "structured_relation"
    assert result["scene_representation_mode"] == "compact_objects_with_estimated_relations"
    assert result["input_source_summary"]["dataset"] == "hssd-hab"
    assert prompt_payload["scene_representation_mode"] == "compact_objects_with_estimated_relations"
    assert prompt_payload["object_aliasing"]["enabled"] is True
    assert "spatial_cues" in prompt_payload
    assert result["object_aliasing"]["aliasing_enabled"] is True
    assert eval_context_summary["visibility_audit"]["spatial_cues_visible_to_model"] is True


def test_pipeline_can_use_configured_separate_judge_model(tmp_path: Path) -> None:
    resources = PipelineResources(
        model_config={
            "judge": {"model": "judge_mock"},
            "models": {
                "mock": {"provider": "mock", "name": "generator-mock"},
                "judge_mock": {"provider": "mock", "name": "judge-mock"},
            },
        },
        benchmark_config={"benchmark": {"save_viewer_scene": True}, "evaluation": {"vlm_judge": "same_model", "vlm_judge_input_mode": "json_plus_render"}},
        layout_schema=read_json(ROOT / "schemas" / "layout.schema.json"),
    )

    state = run_case_pipeline(
        case_path=HSSD_CASE,
        out_dir=tmp_path,
        model_name="mock",
        resources=resources,
        max_repair_iterations=0,
    )

    report = state["current_evaluation"]
    assert state["judge_model_name"] == "judge_mock"
    assert report["generator_metadata"]["model_id"] == "generator-mock"
    assert report["evaluator_metadata"]["model_id"] == "judge-mock"
    assert report["evaluator_metadata"]["same_as_generator"] is False
