from __future__ import annotations

from benchmark.visualization import export_viewer_scene


def _case() -> dict:
    return {
        "case_id": "viewer_case",
        "room": {
            "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
            "floor_plan": {
                "source": "semantics/scenes/*.semantic_config.json region_annotations",
                "primary_representation": "regions",
                "regions": [{"id": "kitchen", "label": "kitchen", "floor_polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
                "aggregate_boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "aggregate_boundary_role": "compatibility_proxy",
            },
            "floor_z": 0.0,
            "wall_height": 3.0,
        },
    }


def _layout(center: list[float]) -> dict:
    return {
        "scene_id": "viewer_case",
        "unit": "meter",
        "objects": [
            {"object_id": "desk_1", "category": "desk", "center": center, "size": [1, 1, 0.8], "yaw": 0, "support_parent": "floor"}
        ],
    }


def _report(iteration: int) -> dict:
    return {
        "iteration": iteration,
        "overall_valid": False,
        "summary": {"schema_valid": True, "physical_valid": None, "spatial_relation_valid": None},
        "judgement_status": "valid_judgement",
        "room_consistency": {"view_artifacts": [{"id": "topdown_global_xy", "path": "views/global/topdown_global_xy.png"}]},
        "debug_evidence": {
            "layout_normalization": {"object_set_normalization": {"object_set_normalization_used": False}},
            "object_groups": [
                {
                    "group_id": "group_001",
                    "object_ids": ["desk_1"],
                    "sent_to_judge": True,
                    "selection_score": 2,
                    "selection_reasons": ["explicit relation"],
                    "formation_edges": [{"source": "desk_1", "target": "desk_1", "reason": "must_link", "strength": "must"}],
                }
            ],
            "group_view_artifacts": [],
            "judge_input_manifest": {"judge_evidence_budgeting": True},
        },
        "vlm_judge_artifacts": {"prompt_path": "vlm_judge/iter_000/judge_prompt.json"},
    }


def test_viewer_scene_contract_and_iteration_diff() -> None:
    initial = _layout([1.0, 1.0, 0.4])
    repaired = _layout([1.4, 1.0, 0.4])
    history = [
        {"iteration": 0, "layout_path": "initial_layout.json", "evaluation_path": "evaluation_report.json", "layout": initial, "evaluation": _report(0)},
        {
            "iteration": 1,
            "layout_path": "repaired_layout_iter_1.json",
            "evaluation_path": "evaluation_report_iter_1.json",
            "layout": repaired,
            "evaluation": _report(1),
        },
    ]

    scene = export_viewer_scene(_case(), repaired, _report(1), history)

    assert scene["scene"]["task_id"] == "viewer_case"
    assert scene["room"]["floor_plan"]["primary_representation"] == "regions"
    assert scene["room"]["floor_plan"]["region_count"] == 1
    assert scene["viewer_options"]["overlays"]["show_room_proxy"] is False
    assert scene["object_set_normalization"]["object_set_normalization_used"] is False
    assert scene["groups"] == scene["group_evidence"]
    assert scene["group_evidence"][0]["group_index"] == 0
    assert scene["group_evidence"][0]["group_color_key"] == "group_001"
    assert scene["group_evidence"][0]["sent_to_judge"] is True
    assert scene["group_evidence"][0]["formation_edges"][0]["reason"] == "must_link"
    assert scene["objects"][0]["group_id"] == "group_001"
    assert scene["judge_evidence"]["manifest"]["judge_evidence_budgeting"] is True
    assert scene["viewer_options"]["diff"]["position_tolerance_m"] == 0.01
    assert scene["iterations"][0]["label"] == "initial"
    assert scene["iterations"][1]["label"] == "repair_1"
    assert scene["iterations"][1]["evaluation_report_path"] == "evaluation_report_iter_1.json"
    assert scene["iterations"][1]["diff_from_initial"]["changed_object_ids"] == ["desk_1"]
