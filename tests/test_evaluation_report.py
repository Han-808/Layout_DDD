from __future__ import annotations

from pathlib import Path

from benchmark.workflow.evaluation import evaluate_layout_v0


def test_evaluation_report_sections_and_case_metrics(tmp_path: Path) -> None:
    case = {
        "case_id": "eval_case",
        "schema_version": "2.0",
        "input_level": "prompt_only",
        "description": {"text": "Create a room."},
    }
    layout = {
        "scene_id": "eval_case",
        "unit": "meter",
        "objects": [{"object_id": "box_001", "category": "box", "center": [0, 0, 0.5], "size": [1, 1, 1], "yaw": 0}],
    }

    report, metrics = evaluate_layout_v0(case=case, layout=layout, out_dir=tmp_path, model_name="mock")

    assert report["evaluator"] == "layered_vlm_room_pair_evaluator_v0"
    assert report["validity_gate"]["passed"]
    assert report["room_consistency"]["score"] == 3
    assert report["object_presence"]["evaluated"] is False
    assert report["specified_relations"]["evaluated"] is False
    assert report["specified_attachments"]["evaluated"] is False
    assert report["debug_evidence"]["physical_flags"] == []
    assert metrics["primary_score"] == 0.75
    assert (tmp_path / "case_metrics.json").exists()
    assert metrics["primary_score"] == report["metrics"]["primary_score"]
