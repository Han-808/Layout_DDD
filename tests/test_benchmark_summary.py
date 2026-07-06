from __future__ import annotations

from benchmark.metrics.aggregate import aggregate_case_results


def test_benchmark_summary_separates_failure_types_by_input_mode() -> None:
    summary = aggregate_case_results(
        [
            {
                "case_id": "ok",
                "input_level": "structured_basic",
                "scene_representation_mode": "compact_objects",
                "task_error": False,
                "parse_success": True,
                "validity_gate": True,
                "renderable": True,
                "judge_success": True,
                "vlm_valid": True,
                "overall_valid": True,
                "vlm_score": 3,
                "vlm_confidence": 0.66,
                "object_presence_rate": 1.0,
                "evidence_flag_counts": {"serious_collision": 2},
                "generation_truncated": False,
            },
            {
                "case_id": "bad_json",
                "input_level": "structured_basic",
                "scene_representation_mode": "compact_objects",
                "task_error": False,
                "parse_success": False,
                "validity_gate": False,
                "renderable": False,
                "judge_success": False,
                "vlm_valid": False,
                "overall_valid": False,
                "generation_error": True,
                "malformed_json": False,
                "generation_truncated": True,
                "parse_error_kind": "truncated_json",
            },
            {
                "case_id": "vlm_bad",
                "input_level": "structured_relation",
                "scene_representation_mode": "compact_objects_with_estimated_relations",
                "task_error": False,
                "parse_success": True,
                "validity_gate": True,
                "renderable": True,
                "judge_success": True,
                "vlm_valid": False,
                "overall_valid": False,
                "vlm_score": 1,
                "generation_truncated": False,
            },
        ]
    )

    assert summary["overall"]["num_cases"] == 3
    assert summary["failure_breakdown"]["generation_error"]["count"] == 1
    assert summary["failure_breakdown"]["vlm_invalid"]["count"] == 1
    assert summary["overall"]["generation_truncated_rate"] == 1 / 3
    assert summary["overall"]["malformed_json_rate"] == 0.0
    assert summary["by_input_mode"]["compact_objects"]["num_cases"] == 2
    assert summary["evidence_flag_rates"]["serious_collision"] == 1 / 3


def test_benchmark_summary_includes_physical_repair_diagnostics() -> None:
    summary = aggregate_case_results(
        [
            {
                "case_id": "improved",
                "room_boundary_delta": -1,
                "serious_collision_delta": -2,
                "above_wall_height_delta": -1,
                "below_floor_delta": 0,
                "fallback_metadata_conflict_count": 1,
                "dense_collision_cluster_count": 1,
                "floating_count_final": 1,
            },
            {
                "case_id": "worse",
                "room_boundary_delta": 0,
                "serious_collision_delta": 1,
                "above_wall_height_delta": 0,
                "below_floor_delta": 0,
                "fallback_physical_flag_count": 1,
                "fallback_metadata_conflict_count": 0,
            },
        ]
    )

    overall = summary["overall"]
    assert overall["mean_collision_delta"] == -0.5
    assert overall["cases_collision_improved"] == 1
    assert overall["cases_collision_worsened"] == 1
    assert overall["cases_boundary_improved"] == 1
    assert overall["cases_height_improved"] == 1
    assert overall["cases_with_fallback_metadata_conflict"] == 1
    assert overall["cases_with_dense_collision_cluster"] == 1
    assert overall["cases_with_floating_evidence"] == 1
