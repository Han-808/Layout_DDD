from __future__ import annotations

from pathlib import Path

from benchmark.pipeline import PipelineResources, run_case_pipeline
from benchmark.utils.io import read_json


ROOT = Path(__file__).resolve().parents[1]


def test_shared_pipeline_writes_v0_outputs(tmp_path: Path) -> None:
    resources = PipelineResources(
        model_config={"models": {"mock": {"provider": "mock", "name": "mock"}}},
        benchmark_config={"benchmark": {"save_viewer_scene": True}, "evaluation": {"room_judge": "mock", "pair_judge": "mock"}},
        layout_schema=read_json(ROOT / "schemas" / "layout.schema.json"),
    )

    state = run_case_pipeline(
        case_path=ROOT / "data" / "benchmark_cases" / "bm_instance_001.json",
        out_dir=tmp_path,
        model_name="mock",
        resources=resources,
        max_repair_iterations=0,
    )

    assert (tmp_path / "evaluation_report.json").exists()
    assert (tmp_path / "case_metrics.json").exists()
    assert (tmp_path / "viewer_scene.json").exists()
    assert (tmp_path / "workflow_trace.json").exists()
    assert (tmp_path / "views" / "room" / "topdown_room.png").exists()
    assert state["case_metrics"]["primary_score"] >= 0
