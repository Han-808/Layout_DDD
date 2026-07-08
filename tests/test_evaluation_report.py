from __future__ import annotations

from pathlib import Path

from benchmark.workflow.evaluate import evaluate_layout_vlm_as_judge_v1


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

    report, metrics = evaluate_layout_vlm_as_judge_v1(
        case=case,
        layout=layout,
        out_dir=tmp_path,
        model_name="mock",
        benchmark_config={"evaluation": {"vlm_judge_input_mode": "json_plus_render"}},
    )

    assert report["evaluator"] == "vlm_as_judge_v1"
    assert report["evaluator_identity"] == "vlm_as_judge_v1"
    assert report["evaluation_policy"]["overall_valid_source"] == "vlm_judgement.valid"
    assert report["evaluation_policy"]["deterministic_flags_affect_validity"] is False
    assert report["scene_summary"]["case_id"] == "eval_case"
    assert report["layout_summary"]["num_layout_objects"] == 1
    assert report["text_budget_used"]["prompt_chars"] > 0
    assert report["judgement_status"] == "valid_judgement"
    assert report["insufficient_evidence"] is False
    assert report["vlm_judgement"]["brief_reasoning"]
    assert report["generator_metadata"]["model_id"] == "mock"
    assert report["evaluator_metadata"]["same_as_generator"] is True
    assert report["validity_gate"]["passed"]
    assert "vlm_judgement" in report
    assert report["vlm_judge_artifacts"]["prompt_path"] == "vlm_judge/iter_000/judge_prompt.json"
    assert report["room_consistency"]["score"] == 3
    assert report["object_presence"]["evaluated"] is False
    assert report["specified_relations"]["evaluated"] is False
    assert report["specified_attachments"]["evaluated"] is False
    assert report["debug_evidence"]["physical_flags"] == []
    assert report["debug_evidence"]["object_groups"][0]["object_ids"] == ["box_001"]
    assert report["debug_evidence"]["resolved_grouping_config"]["num_renderable_objects"] == 1
    assert "omitted_grouping_edges" in report["debug_evidence"]
    assert "cross_group_relations" in report["debug_evidence"]
    assert metrics["primary_score"] == 0.75
    assert report["case_metrics_path"] == "case_metrics_iter_0.json"
    assert (tmp_path / "case_metrics_iter_0.json").exists()
    assert not (tmp_path / "case_metrics.json").exists()
    assert metrics["primary_score"] == report["metrics"]["primary_score"]
    assert (tmp_path / "views" / "global" / "topdown_global_xy.png").exists()
    assert (tmp_path / "views" / "groups" / "group_001" / "group_001_xy.png").exists()
    assert (tmp_path / "vlm_judge" / "iter_000" / "judge_prompt.json").exists()
    assert (tmp_path / "vlm_judge" / "iter_000" / "judge_image_manifest.json").exists()
    assert (tmp_path / "vlm_judge" / "iter_000" / "judge_raw_response.txt").exists()
    assert (tmp_path / "vlm_judge" / "iter_000" / "judge_parsed_response.json").exists()
