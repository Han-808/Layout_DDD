from __future__ import annotations

import json

from benchmark.workflow.judge_summaries import (
    build_judge_prompt_payload,
    build_layout_summary,
    build_scene_summary,
    text_budget_config,
)
from benchmark.workflow.vlm_judge import normalize_vlm_judgement


def _case() -> dict:
    return {
        "case_id": "case_a",
        "task_id": "case_a",
        "input_level": "structured_basic",
        "description": {"text": "Place a desk, chair, and lamp in a compact office."},
        "room": {"boundary": [[0, 0], [5, 0], [5, 4], [0, 4]], "wall_height": 3.0},
        "objects": [{"id": "desk_1", "category": "desk"}, {"id": "chair_1", "category": "chair"}, {"id": "lamp_1", "category": "lamp"}],
        "source": {"dataset": "hssd-hab", "scene_instance": "scenes/example.scene_instance.json"},
    }


def _layout() -> dict:
    return {
        "scene_id": "case_a",
        "objects": [
            {"object_id": "desk_1", "category": "desk", "center": [1.12345, 1, 0.4], "size": [1, 1, 0.8], "yaw": 0},
            {"object_id": "chair_1", "category": "chair", "center": [2, 1, 0.45], "size": [0.5, 0.5, 0.9], "yaw": 180},
            {"object_id": "lamp_1", "category": "lamp", "center": [1, 1, 1.1], "size": [0.2, 0.2, 0.4], "yaw": 0},
        ],
    }


def _groups() -> list[dict]:
    return [
        {"group_id": "group_001", "object_ids": ["desk_1", "chair_1"], "num_objects": 2, "edge_reasons": ["explicit_relation"]},
        {"group_id": "group_002", "object_ids": ["lamp_1"], "num_objects": 1, "edge_reasons": ["derived_support_geometry"]},
    ]


def test_scene_summary_is_deterministic_and_case_derived() -> None:
    budget = text_budget_config({})
    first = build_scene_summary(_case(), "structured_basic", budget)
    second = build_scene_summary(_case(), "structured_basic", budget)

    assert first == second
    assert first["case_id"] == "case_a"
    assert first["room_proxy"]["type"] == "synthetic_proxy_rectangle"
    assert first["room_proxy"]["width"] == 5.0
    assert first["dataset_source"] == "hssd-hab"
    assert "floor_plan.regions" in " ".join(first["notes"])


def test_scene_summary_prefers_hssd_multi_region_floor_plan() -> None:
    case = _case()
    case["room"] = {
        "floor_plan": {
            "source": "hssd_semantic_config.region_annotations.poly_loop",
            "coordinate_mapping": "HSSD semantic poly_loop [x, y, z] is imported as benchmark floor polygon [x, z].",
            "primary_representation": "regions",
            "aggregate_boundary_role": "compatibility_proxy",
            "regions": [
                {"id": "hallway", "label": "hallway", "floor_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                {"id": "kitchen", "label": "kitchen", "floor_polygon": [[1, 0], [2, 0], [2, 1], [1, 1]]},
            ],
            "aggregate_boundary": [[0, 0], [2, 0], [2, 1], [0, 1]],
        },
        "boundary": [[0, 0], [2, 0], [2, 1], [0, 1]],
        "wall_height": 2.8,
    }

    summary = build_scene_summary(case, "structured_basic", text_budget_config({}))

    assert summary["room_proxy"]["type"] == "multi_region_floor_plan"
    assert summary["room_proxy"]["source"] == "hssd_semantic_config.region_annotations.poly_loop"
    assert summary["room_proxy"]["boundary_check_source"] == "floor_plan.regions polygon union"
    assert summary["room_proxy"]["region_count"] == 2
    assert summary["room_proxy"]["region_labels"] == ["hallway", "kitchen"]


def test_layout_summary_is_deterministic_and_budgeted_groups_are_compact() -> None:
    budget = text_budget_config({})
    selection = {
        "budgeting_enabled": True,
        "selected_groups": [{"group_id": "group_001", "selection_score": 10}],
        "omitted_groups": [{"group_id": "group_002", "selection_score": 1}],
    }

    summary = build_layout_summary(
        layout=_layout(),
        renderable_layout={"objects": _layout()["objects"][:2]},
        layout_normalization_summary={"removed_optional_null_fields": []},
        object_groups=_groups(),
        sanity_flags=[],
        physical_flags=[{"type": "serious_collision", "objects": ["desk_1", "chair_1"], "message": "overlap"}],
        view_flags=[],
        render_skipped_objects=[{"type": "render_skipped_object", "object_id": "bad"}],
        evidence_selection=selection,
        text_budget=budget,
    )

    assert summary["num_layout_objects"] == 3
    assert summary["num_renderable_objects"] == 2
    assert summary["selected_groups_count"] == 1
    assert summary["omitted_groups_count"] == 1
    assert summary["evidence_budgeting_policy"]["mode"] == "budgeted"
    assert summary["evidence_budgeting_policy"]["omitted_groups_are_missing_objects"] is False
    assert summary["evidence_budgeting_policy"]["omitted_groups_reason"] == "prompt/image budget, not object absence"
    assert [item["group_id"] for item in summary["selected_group_details"]] == ["group_001"]
    assert summary["omitted_groups_summary"][0]["group_id"] == "group_002"
    assert summary["flag_summary"]["physical_flags"]["by_type"]["serious_collision"] == 1


def test_text_budget_caps_record_truncation() -> None:
    config = {
        "vlm_judge": {
            "text_budget": {
                "max_total_chars": 4500,
                "max_scene_summary_chars": 300,
                "max_layout_summary_chars": 700,
                "max_selected_group_objects": 2,
                "max_objects_per_group_in_prompt": 1,
                "max_flag_examples_per_type": 1,
                "numeric_precision": 2,
            }
        }
    }
    case = _case()
    case["description"] = {"text": "long prompt " * 200}
    layout = _layout()
    layout["objects"] = layout["objects"] * 8
    groups = [{"group_id": "group_001", "object_ids": [obj["object_id"] for obj in layout["objects"]], "num_objects": len(layout["objects"])}]

    bundle = build_judge_prompt_payload(
        case=case,
        layout=layout,
        renderable_layout=layout,
        input_level="structured_basic",
        layout_normalization_summary={},
        object_groups=groups,
        sanity_flags=[{"type": "layout_sanity", "message": "x"} for _ in range(4)],
        physical_flags=[],
        view_flags=[],
        render_skipped_objects=[],
        relation_specs=[],
        attachment_specs=[],
        evidence_selection={"budgeting_enabled": True, "selected_groups": [{"group_id": "group_001"}], "omitted_groups": []},
        image_manifest=[],
        benchmark_config=config,
    )

    assert bundle["scene_summary"]["truncated"] is True
    assert bundle["layout_summary"]["selected_group_details"][0]["truncated"] is True
    assert bundle["layout_summary"]["selected_group_details"][0]["total"] >= 1
    assert bundle["layout_summary"]["flag_summary"]["schema_flags"]["examples_by_type"]["layout_sanity"]["shown"] == 1
    assert bundle["text_budget_used"]["truncated"] is True
    assert len(json.dumps(bundle["prompt_payload"], separators=(",", ":"))) <= 4500


def test_prompt_payload_contains_temporary_rubric_and_output_schema() -> None:
    bundle = build_judge_prompt_payload(
        case=_case(),
        layout=_layout(),
        renderable_layout=_layout(),
        input_level="structured_basic",
        layout_normalization_summary={},
        object_groups=_groups(),
        sanity_flags=[],
        physical_flags=[],
        view_flags=[],
        render_skipped_objects=[],
        relation_specs=[],
        attachment_specs=[],
        evidence_selection={"budgeting_enabled": False},
        image_manifest=[{"id": "topdown_global_xy", "scope": "global", "path": "view.png", "included_in_prompt": True}],
        benchmark_config={},
    )

    payload = bundle["prompt_payload"]
    criteria = [item["id"] for item in payload["rubric"]["criteria"]]
    assert criteria == ["parseability", "completeness", "boundary", "height", "collision", "support", "evidence"]
    assert payload["required_output_schema"]["judgement_status"] == "valid_judgement | insufficient_evidence | judge_error"
    assert payload["evidence_manifest"][0]["meaning"] == "global top-down proxy room view"
    assert "budget-omitted groups" in payload["evaluation_policy"]["evidence_budgeting"]
    assert payload["evaluation_policy"]["completeness_source"].startswith("Judge object completeness")


def test_new_judge_output_schema_normalizes_to_compatible_fields() -> None:
    parsed = normalize_vlm_judgement(
        {
            "valid": False,
            "score": 1,
            "confidence": "low",
            "judgement_status": "insufficient_evidence",
            "brief_reasoning": "Side views are missing.",
            "issues": [
                {
                    "group_id": None,
                    "issue_type": "evidence",
                    "severity": "major",
                    "object_ids": [],
                    "evidence": "No group views.",
                    "repair_hint": "Render group tri-views.",
                }
            ],
            "insufficient_evidence": True,
        }
    )

    assert parsed["judgement_status"] == "insufficient_evidence"
    assert parsed["insufficient_evidence"] is True
    assert parsed["brief_reasoning"] == "Side views are missing."
    assert parsed["short_reason"] == "Side views are missing."
    assert parsed["issues"][0]["issue_type"] == "evidence"
