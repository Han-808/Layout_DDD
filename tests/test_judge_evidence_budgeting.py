from __future__ import annotations

import json
from pathlib import Path

from benchmark.models.factory import create_model
from benchmark.visualization import export_viewer_scene
from benchmark.workflow.evaluation import evaluate_layout_vlm_as_judge_v1
from benchmark.workflow.judge_evidence_selector import evidence_budgeting_config, select_judge_evidence
from benchmark.workflow.layout_normalization import sanitize_layout_optional_nulls
from benchmark.workflow.trace import build_workflow_trace


class _FakeBudgetedVLM:
    endpoint = "http://localhost:8000/v1"
    model_id = "Qwen/Qwen3-VL-32B-Instruct"
    runtime_profile = "hyak_h200_qwen3vl32b"
    temperature = 0.0
    max_tokens = 2048
    timeout_seconds = 300
    response_format_json = True

    def __init__(self, *, judge_evidence_budgeting: bool) -> None:
        self.judge_evidence_budgeting = judge_evidence_budgeting
        self.messages = []
        self.last_request_metadata = {}

    def chat_messages(self, messages: list[dict], *, response_format_json: bool | None = None) -> str:
        self.messages = messages
        image_count = sum(
            1
            for message in messages
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
        self.last_request_metadata = {
            "endpoint": self.endpoint,
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "response_format_json": response_format_json,
            "message_count": len(messages),
            "image_count": image_count,
        }
        return json.dumps(
            {
                "valid": True,
                "score": 3,
                "score_norm": 0.75,
                "short_reason": "Budgeted evidence is judgeable.",
                "global_assessment": "OK.",
                "group_results": [],
                "relation_results": [],
                "attachment_results": [],
            }
        )


def _budget_config() -> dict:
    return {
        "vlm_judge": {
            "evidence_budget": {
                "max_input_tokens": 60000,
                "include_global_view": True,
                "base_max_groups_for_judge": 3,
                "budget_raise_ratio": 0.5,
                "max_groups_for_judge_cap": 5,
                "group_views": ["xy", "yz", "xz"],
                "include_full_object_list": False,
                "include_compact_scene_summary": True,
                "include_selected_group_details": True,
                "summarize_schema_flags": True,
                "summarize_physical_flags": True,
                "summarize_view_flags": True,
            },
            "profiles": {
                "hyak_h200_qwen3vl32b": {
                    "judge_generation": {"max_tokens": 2048, "timeout_seconds": 300},
                }
            },
        },
        "grouping": {
            "proximity": {"min_gap_m": 0.0, "max_gap_m": 0.1, "max_normalized_gap": 0.1},
            "diameter": {"ratio_of_room_diagonal": 0.01, "min_m": 1.0, "max_m": 1.0},
            "object_count": {"additive_margin": 0, "min_objects_per_group": 1, "max_objects_per_group": 1},
        },
        "view_validation": {"min_foreground_ratio": 0.0, "min_visible_object_ratio": 0.0, "max_camera_retries": 0},
    }


def _case() -> dict:
    return {
        "case_id": "budget_case",
        "input_level": "structured_basic",
        "description": {"text": "Place four objects in a room."},
        "room": {"boundary": [[0, 0], [12, 0], [12, 4], [0, 4]], "floor_z": 0.0, "wall_height": 3.0},
        "objects": [{"id": f"obj_{idx}", "category": "box"} for idx in range(1, 5)],
    }


def _layout() -> dict:
    return {
        "scene_id": "budget_case",
        "unit": "meter",
        "objects": [
            {"object_id": "obj_1", "category": "box", "region_id": None, "center": [1, 1, 0.5], "size": [0.5, 0.5, 1.0], "yaw": 0},
            {"object_id": "obj_2", "category": "box", "region_id": None, "center": [4, 1, 0.5], "size": [0.5, 0.5, 1.0], "yaw": 0},
            {"object_id": "obj_3", "category": "box", "region_id": None, "center": [7, 1, 0.5], "size": [0.5, 0.5, 1.0], "yaw": 0},
            {"object_id": "obj_4", "category": "box", "region_id": None, "center": [10, 1, 0.5], "size": [0.5, 0.5, 1.0], "yaw": 0},
        ],
    }


def test_missing_judge_evidence_budgeting_defaults_to_false() -> None:
    model = create_model(
        "local",
        {
            "models": {
                "local": {
                    "provider": "openai_compatible",
                    "endpoint": "http://localhost:8000/v1",
                    "model": "local/test",
                }
            }
        },
    )

    assert getattr(model, "judge_evidence_budgeting") is False
    assert evidence_budgeting_config({}, judge_evidence_budgeting=getattr(model, "judge_evidence_budgeting"))["enabled"] is False


def test_judge_evidence_budgeting_true_uses_budget_config() -> None:
    config = evidence_budgeting_config(
        _budget_config(),
        judge_evidence_budgeting=True,
        runtime_profile="hyak_h200_qwen3vl32b",
    )
    assert config["enabled"] is True
    assert config["judge_evidence_budgeting"] is True
    assert config["base_max_groups_for_judge"] == 3
    assert config["budget_raise_ratio"] == 0.5
    assert config["max_groups_for_judge_cap"] == 5


def test_judge_evidence_budgeting_false_uses_full_mode() -> None:
    config = evidence_budgeting_config(
        _budget_config(),
        judge_evidence_budgeting=False,
        runtime_profile="hyak_h200_qwen3vl32b",
    )
    assert config["enabled"] is False
    assert config["judge_evidence_budgeting"] is False
    assert config["base_max_groups_for_judge"] == 3


def test_deterministic_group_selection_topk_and_tiebreak() -> None:
    groups = [
        {"group_id": "group_002", "object_ids": ["b"]},
        {"group_id": "group_001", "object_ids": ["a"]},
        {"group_id": "group_003", "object_ids": ["c"], "formation_edges": [{"source": "c", "target": "d", "reason": "explicit_relation"}]},
    ]
    selection = select_judge_evidence(
        global_view_artifacts=[{"id": "topdown_global_xy", "path": "global.png"}],
        group_view_artifacts=[],
        object_groups=groups,
        physical_flags=[],
        view_flags=[],
        render_skipped_objects=[],
        config={
            "enabled": True,
            "base_max_groups_for_judge": 2,
            "budget_raise_ratio": 0.0,
            "max_groups_for_judge_cap": 2,
            "group_views": ["xy"],
        },
        runtime_profile="hyak_h200_qwen3vl32b",
    )

    assert [item["group_id"] for item in selection["selected_groups"]] == ["group_003", "group_001"]
    assert selection["omitted_groups"][0]["group_id"] == "group_002"


def test_group_selection_prioritizes_room_boundary_flags() -> None:
    groups = [
        {"group_id": "group_boundary", "object_ids": ["outside_obj"]},
        {"group_id": "group_large", "object_ids": ["a", "b", "c", "d", "e", "f"]},
    ]
    selection = select_judge_evidence(
        global_view_artifacts=[{"id": "topdown_global_xy", "path": "global.png"}],
        group_view_artifacts=[],
        object_groups=groups,
        physical_flags=[{"type": "room_boundary", "objects": ["outside_obj"]}],
        view_flags=[],
        render_skipped_objects=[],
        config={
            "enabled": True,
            "base_max_groups_for_judge": 1,
            "budget_raise_ratio": 0.0,
            "max_groups_for_judge_cap": 1,
            "group_views": ["xy"],
        },
        runtime_profile="hyak_h200_qwen3vl32b",
    )

    assert selection["selected_groups"][0]["group_id"] == "group_boundary"
    assert "room_boundary" in selection["selected_groups"][0]["selection_reasons"]


def test_budget_raise_resolves_to_five_groups_and_sixteen_images() -> None:
    groups = [{"group_id": f"group_{idx:03d}", "object_ids": [f"obj_{idx}"]} for idx in range(1, 7)]
    group_artifacts = [
        {"id": f"{group['group_id']}_{projection}", "path": f"{group['group_id']}_{projection}.png"}
        for group in groups
        for projection in ["xy", "yz", "xz"]
    ]

    selection = select_judge_evidence(
        global_view_artifacts=[{"id": "topdown_global_xy", "path": "global.png"}],
        group_view_artifacts=group_artifacts,
        object_groups=groups,
        physical_flags=[],
        view_flags=[],
        render_skipped_objects=[],
        config={
            "enabled": True,
            "base_max_groups_for_judge": 3,
            "budget_raise_ratio": 0.5,
            "max_groups_for_judge_cap": 5,
            "group_views": ["xy", "yz", "xz"],
        },
        runtime_profile="hyak_h200_qwen3vl32b",
    )

    assert selection["budget"]["base_max_groups_for_judge"] == 3
    assert selection["budget"]["budget_raise_ratio"] == 0.5
    assert selection["budget"]["effective_max_groups_for_judge"] == 5
    assert selection["budget"]["effective_max_images"] == 16
    assert selection["budget"]["selected_images"] == 16
    assert len(selection["selected_groups"]) == 5
    assert len(selection["omitted_groups"]) == 1


def test_budget_raise_cap_is_respected() -> None:
    groups = [{"group_id": f"group_{idx:03d}", "object_ids": [f"obj_{idx}"]} for idx in range(1, 7)]
    selection = select_judge_evidence(
        global_view_artifacts=[{"id": "topdown_global_xy", "path": "global.png"}],
        group_view_artifacts=[],
        object_groups=groups,
        physical_flags=[],
        view_flags=[],
        render_skipped_objects=[],
        config={
            "enabled": True,
            "base_max_groups_for_judge": 4,
            "budget_raise_ratio": 1.0,
            "max_groups_for_judge_cap": 4,
            "group_views": ["xy"],
        },
        runtime_profile="hyak_h200_qwen3vl32b",
    )

    assert selection["budget"]["effective_max_groups_for_judge"] == 4


def test_budgeted_evaluation_manifest_viewer_marks_and_compact_prompt(tmp_path: Path) -> None:
    fake = _FakeBudgetedVLM(judge_evidence_budgeting=True)
    report, _ = evaluate_layout_vlm_as_judge_v1(
        case=_case(),
        layout=_layout(),
        out_dir=tmp_path,
        model_name="qwen3vl_sglang_32b",
        benchmark_config=_budget_config(),
        generator_model=fake,
        judge_model=fake,
    )

    manifest = report["debug_evidence"]["judge_input_manifest"]
    assert manifest["judge_evidence_budgeting"] is True
    assert manifest["budgeting_enabled"] is True
    assert manifest["mode"] == "budgeted"
    assert manifest["base_max_groups_for_judge"] == 3
    assert manifest["budget_raise_ratio"] == 0.5
    assert manifest["effective_max_groups_for_judge"] == 5
    assert manifest["max_groups_for_judge_cap"] == 5
    assert manifest["effective_max_images"] == 16
    assert manifest["budget"]["selected_images"] == 13
    assert len(manifest["selected_images"]) == 13
    assert len(manifest["selected_groups"]) == 4
    assert len(manifest["omitted_groups"]) == 0
    assert report["vlm_judge_artifacts"]["input_manifest_path"] == "vlm_judge/iter_000/judge_input_manifest.json"
    assert (tmp_path / "vlm_judge" / "iter_000" / "judge_input_manifest.json").exists()
    assert fake.last_request_metadata["image_count"] == 13

    user_prompt = fake.messages[1]["content"][0]["text"]
    prompt_payload = json.loads(user_prompt)
    assert "layout_objects" not in prompt_payload
    assert prompt_payload["flag_summary"]["schema_flags"]["total"] == 0
    assert prompt_payload["scene_summary"]["num_input_objects"] == 4
    assert prompt_payload["layout_summary"]["num_layout_objects"] == 4
    assert prompt_payload["layout_summary"]["selected_groups_count"] == 4

    viewer_scene = export_viewer_scene(_case(), _layout(), report)
    assert viewer_scene["judge_input_manifest"]["judge_evidence_budgeting"] is True
    sent = [group for group in viewer_scene["group_evidence"] if group["sent_to_judge"] is True]
    omitted = [group for group in viewer_scene["group_evidence"] if group["sent_to_judge"] is False]
    assert len(sent) == 4
    assert len(omitted) == 0
    assert all("selection_score" in group for group in viewer_scene["group_evidence"])

    trace = build_workflow_trace({"task_id": "budget_case", "current_evaluation": report}, tmp_path)
    evaluate_node = next(node for node in trace["nodes"] if node["id"] == "evaluate_layout")
    assert evaluate_node["artifacts"]["vlm_judge"]["input_manifest_path"] == "vlm_judge/iter_000/judge_input_manifest.json"


def test_full_evaluation_sends_all_evidence_without_budget_manifest(tmp_path: Path) -> None:
    fake = _FakeBudgetedVLM(judge_evidence_budgeting=False)
    report, _ = evaluate_layout_vlm_as_judge_v1(
        case=_case(),
        layout=_layout(),
        out_dir=tmp_path,
        model_name="qwen3vl_sglang_32b",
        benchmark_config=_budget_config(),
        generator_model=fake,
        judge_model=fake,
    )

    assert report["debug_evidence"]["judge_input_manifest"] == {}
    assert "input_manifest_path" not in report["vlm_judge_artifacts"]
    assert not (tmp_path / "vlm_judge" / "iter_000" / "judge_input_manifest.json").exists()
    assert fake.last_request_metadata["image_count"] == 13

    user_prompt = fake.messages[1]["content"][0]["text"]
    prompt_payload = json.loads(user_prompt)
    assert "scene_summary" in prompt_payload
    assert "layout_summary" in prompt_payload
    assert prompt_payload["layout_summary"]["full_group_details_included"] is True
    assert len(prompt_payload["layout_summary"]["selected_group_details"]) == 4

    viewer_scene = export_viewer_scene(_case(), _layout(), report)
    assert viewer_scene["judge_input_manifest"] == {}
    assert all("sent_to_judge" not in group for group in viewer_scene["group_evidence"])


def test_optional_null_fields_are_removed_from_sanitized_layout() -> None:
    sanitized, metadata = sanitize_layout_optional_nulls(_layout())

    assert metadata["removed_optional_null_fields"] == [{"field": "region_id", "count": 4}]
    assert all("region_id" not in obj for obj in sanitized["objects"])
