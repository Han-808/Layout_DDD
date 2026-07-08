from __future__ import annotations

import json
from pathlib import Path

from benchmark.workflow.evaluate import evaluate_layout_v0
from benchmark.workflow.vlm_judge import OpenAICompatibleVLMJudge


class _FakeVLM:
    def __init__(self, response: dict | str) -> None:
        self.response = response
        self.messages = []
        self.response_format_json = None
        self.last_request_metadata = {}

    def chat_messages(self, messages: list[dict], *, response_format_json: bool | None = None) -> str:
        self.messages = messages
        self.response_format_json = response_format_json
        self.last_request_metadata = {
            "endpoint": "http://localhost:8000/v1",
            "model": "fake-vlm",
            "message_count": len(messages),
            "image_count": sum(
                1
                for message in messages
                for part in (message.get("content") if isinstance(message.get("content"), list) else [])
                if isinstance(part, dict) and part.get("type") == "image_url"
            ),
        }
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


def _case() -> dict:
    return {
        "case_id": "judge_case",
        "input_level": "structured_relation",
        "description": {"text": "Create a tiny office."},
        "room": {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        "objects": [{"id": "desk_1", "category": "desk"}, {"id": "chair_1", "category": "chair"}],
        "relations": [{"id": "rel_1", "type": "near", "subject": "chair_1", "object": "desk_1"}],
    }


def _layout() -> dict:
    return {
        "scene_id": "judge_case",
        "unit": "meter",
        "coordinate_system": {
            "origin": "front-left floor corner",
            "x_axis": "room width",
            "y_axis": "room depth",
            "z_axis": "height",
            "rotation_unit": "degree",
        },
        "objects": [
            {"object_id": "desk_1", "category": "desk", "center": [1, 1, 0.4], "size": [1, 1, 0.8], "yaw": 0},
            {"object_id": "chair_1", "category": "chair", "center": [2, 1, 0.45], "size": [0.5, 0.5, 0.9], "yaw": 0},
        ],
    }


def test_openai_compatible_vlm_judge_sends_multimodal_payload(tmp_path: Path) -> None:
    image_path = tmp_path / "view.png"
    image_path.write_bytes(b"fake-png")
    fake = _FakeVLM(
        {
            "valid": True,
            "score": 4,
            "short_reason": "Looks coherent.",
            "global_assessment": "Good.",
            "group_results": [],
            "relation_results": [{"id": "rel_1", "pass": True, "reason": "Near."}],
        }
    )
    judge = OpenAICompatibleVLMJudge(fake)

    result = judge.judge(
        case=_case(),
        layout=_layout(),
        input_level="structured_relation",
        sanity_flags=[],
        physical_flags=[],
        view_flags=[],
        render_skipped_objects=[],
        object_groups=[],
        global_view_artifacts=[{"id": "topdown_global_xy", "abs_path": str(image_path), "path": "view.png"}],
        group_view_artifacts=[],
        relation_specs=[{"id": "rel_1", "type": "near", "subject": "chair_1", "object": "desk_1"}],
        attachment_specs=[],
        judge_input_mode="json_plus_render",
        artifact_dir=tmp_path / "vlm_judge",
    )

    user_content = fake.messages[1]["content"]
    assert fake.response_format_json is True
    assert any(part.get("type") == "image_url" for part in user_content)
    assert result["score"] == 4
    assert result["relation_results"][0]["pass"]
    assert result["_judge_artifacts"]["prompt_path"] == "vlm_judge/judge_prompt.json"
    assert result["_judge_artifacts"]["request_metadata_path"] == "vlm_judge/judge_request_metadata.json"
    assert (tmp_path / "vlm_judge" / "judge_image_manifest.json").exists()
    request_metadata = json.loads((tmp_path / "vlm_judge" / "judge_request_metadata.json").read_text(encoding="utf-8"))
    assert request_metadata["image_count"] == 1


def test_openai_compatible_vlm_judge_json_only_sends_scene_without_images(tmp_path: Path) -> None:
    image_path = tmp_path / "view.png"
    image_path.write_bytes(b"fake-png")
    fake = _FakeVLM(
        {
            "valid": True,
            "score": 3,
            "brief_reasoning": "Scene JSON is enough.",
            "group_results": [],
            "relation_results": [{"id": "rel_1", "pass": True, "reason": "Near in JSON."}],
        }
    )
    judge = OpenAICompatibleVLMJudge(fake)

    result = judge.judge(
        case=_case(),
        layout=_layout(),
        scene={
            "scene_id": "judge_case",
            "assets": [
                {
                    "asset_id": "desk_1",
                    "category": "desk",
                    "placement": {"position": [1, 1, 0.4], "yaw_degrees": 0},
                    "dimensions": [1, 1, 0.8],
                }
            ],
        },
        input_level="structured_relation",
        sanity_flags=[],
        physical_flags=[{"type": "serious_collision", "objects": ["desk_1"], "message": "example"}],
        view_flags=[],
        render_skipped_objects=[],
        object_groups=[],
        global_view_artifacts=[{"id": "topdown_global_xy", "abs_path": str(image_path), "path": "view.png"}],
        group_view_artifacts=[],
        relation_specs=[{"id": "rel_1", "type": "near", "subject": "chair_1", "object": "desk_1"}],
        attachment_specs=[],
        artifact_dir=tmp_path / "vlm_judge_json",
    )

    user_content = fake.messages[1]["content"]
    prompt_payload = json.loads(user_content[0]["text"])
    assert not any(part.get("type") == "image_url" for part in user_content)
    assert prompt_payload["vlm_judge_input_mode"] == "json_only"
    assert prompt_payload["scene"]["scene_id"] == "judge_case"
    assert prompt_payload["structured_evidence"]["physical_flags"][0]["type"] == "serious_collision"
    assert result["_judge_input_manifest"]["render_evidence_used"] is False
    image_manifest = json.loads((tmp_path / "vlm_judge_json" / "judge_image_manifest.json").read_text(encoding="utf-8"))
    assert image_manifest == []


def test_malformed_vlm_judge_response_creates_invalid_report(tmp_path: Path) -> None:
    report, metrics = evaluate_layout_v0(
        case=_case(),
        layout=_layout(),
        out_dir=tmp_path,
        model_name="fake",
        layout_schema=None,
        judge_model=_FakeVLM({}),
    )

    assert report["evaluator"] == "vlm_as_judge_v1"
    assert report["overall_valid"] is False
    assert report["judge_error"]
    assert metrics["primary_score"] == 0.0
