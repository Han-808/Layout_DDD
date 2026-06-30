from __future__ import annotations

from pathlib import Path

from benchmark.models.mock_model import MockModel
from benchmark.utils.io import read_json
from benchmark.workflow import run_workflow


ROOT = Path(__file__).resolve().parents[1]
HSSD_CASE = ROOT / "data" / "benchmark_cases" / "hssd_small" / "102343992_structured_relation.json"


def _base_state(tmp_path: Path, model: MockModel, max_repair_iterations: int) -> dict:
    return {
        "case_path": str(HSSD_CASE),
        "out_dir": str(tmp_path),
        "model": model,
        "model_name": model.name,
        "layout_schema": read_json(ROOT / "schemas" / "layout.schema.json"),
        "benchmark_config": {"benchmark": {"save_viewer_scene": True}},
        "max_repair_iterations": max_repair_iterations,
    }


def test_graph_one_shot_path(tmp_path: Path) -> None:
    state = run_workflow(_base_state(tmp_path, MockModel(), 0))

    assert len(state["history"]) == 1
    assert state["history"][0]["overall_valid"] is True
    assert state["current_evaluation"]["evaluator"] == "vlm_as_judge_v1"
    assert state["case_metrics"]["primary_score"] >= 0
    assert "layout" in state["history"][0]
    assert "evaluation" in state["history"][0]
    assert "layout" not in state["per_case_result"]["history"][0]
    assert (tmp_path / "evaluation_report.json").exists()
    assert (tmp_path / "case_metrics.json").exists()
    assert (tmp_path / "workflow_trace.json").exists()
    assert (tmp_path / "workflow_graph.mmd").exists()
    assert Path(state["per_case_result_path"]).exists()
    assert (tmp_path / "viewer_scene.json").exists()
    graph_text = (tmp_path / "workflow_graph.mmd").read_text(encoding="utf-8")
    assert "repair_layout --> evaluate_layout" in graph_text
    assert "VLM-as-judge final validity source" in graph_text

    trace = read_json(tmp_path / "workflow_trace.json")
    evaluate_artifacts = trace["nodes"][2]["artifacts"]
    assert evaluate_artifacts["global_views"]
    assert evaluate_artifacts["group_views"]
    assert evaluate_artifacts["vlm_judge"]["prompt_path"] == "vlm_judge/iter_000/judge_prompt.json"

    viewer_scene = read_json(tmp_path / "viewer_scene.json")
    assert viewer_scene["group_evidence"]
    assert viewer_scene["group_evidence"][0]["views"]["xy"]["path"].endswith(".png")
    assert viewer_scene["scene"]["task_id"] == state["task_id"]
    assert viewer_scene["groups"] == viewer_scene["group_evidence"]
    assert viewer_scene["workflow_steps"] == viewer_scene["workflow"]["artifacts"]
    assert viewer_scene["artifacts"] == viewer_scene["workflow"]["artifacts"]
    assert viewer_scene["viewer_options"]["json_preview"]["lazy_load"] is True
    assert viewer_scene["global_views"]
    assert viewer_scene["vlm_judge_artifacts"]["raw_response_path"] == "vlm_judge/iter_000/judge_raw_response.txt"
    assert viewer_scene["runtime_evidence_config"]["render"]["backend"] == "perspective_bbox_zbuffer"
    group_diag = viewer_scene["group_evidence"][0]["diagnostics"]["xy"]
    assert group_diag["resolved_config"]["render"]["backend"] == "perspective_bbox_zbuffer"
    assert "data" not in viewer_scene["workflow"]["artifacts"][1]


def test_graph_vlm_evaluator_does_not_repair_minor_debug_overlap(tmp_path: Path) -> None:
    state = run_workflow(_base_state(tmp_path, MockModel(behavior="colliding_then_repair"), 1))

    assert len(state["history"]) == 1
    assert state["history"][0]["overall_valid"] is True
    assert state["history"][0]["metrics"]["validity_gate"] is True
    assert state["history"][0]["feedback_path"] == ""
    assert not (tmp_path / "feedback_iter_0.json").exists()
    assert not (tmp_path / "repaired_layout_iter_1.json").exists()

    viewer_scene = read_json(tmp_path / "viewer_scene.json")
    assert len(viewer_scene["iterations"]) == 1
    assert viewer_scene["iterations"][0]["label"] == "initial"
    assert viewer_scene["iterations"][0]["evaluation_report_path"].endswith("evaluation_report.json")
    assert [item["step"] for item in viewer_scene["workflow"]["artifacts"][:3]] == [
        "input",
        "generate",
        "evaluate",
    ]
    assert viewer_scene["workflow"]["artifacts"][1]["status"] == "valid"


def test_graph_uses_configured_per_case_filename(tmp_path: Path) -> None:
    state_config = _base_state(tmp_path, MockModel(), 0)
    state_config["benchmark_config"] = {
        "benchmark": {"save_viewer_scene": False},
        "outputs": {"per_case_filename": "custom_case_result.json"},
    }

    state = run_workflow(state_config)

    assert Path(state["per_case_result_path"]).name == "custom_case_result.json"
    assert (tmp_path / "custom_case_result.json").exists()
    assert not (tmp_path / "viewer_scene.json").exists()


def test_graph_can_skip_intermediate_artifact_files(tmp_path: Path) -> None:
    state_config = _base_state(tmp_path, MockModel(behavior="colliding_then_repair"), 1)
    state_config["benchmark_config"] = {
        "benchmark": {"save_intermediate_artifacts": False, "save_viewer_scene": True}
    }

    state = run_workflow(state_config)

    assert len(state["history"]) == 1
    assert state["history"][0]["layout_path"] == ""
    assert not (tmp_path / "initial_layout.json").exists()
    assert not (tmp_path / "evaluation_report_iter_0.json").exists()
    assert (tmp_path / "evaluation_report.json").exists()
    assert (tmp_path / "case_metrics.json").exists()

    viewer_scene = read_json(tmp_path / "viewer_scene.json")
    assert len(viewer_scene["iterations"]) == 1
    assert "data" not in viewer_scene["workflow"]["artifacts"][1]
    assert viewer_scene["workflow"]["artifacts"][1]["path"] == ""
