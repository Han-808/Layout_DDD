from __future__ import annotations

import json
import urllib.request

import pytest

from benchmark.models.base_model import ModelResponseError, build_generation_prompt, build_repair_prompt, compact_bm_instance_for_model, parse_json_object
from benchmark.models.factory import create_model
from benchmark.models.openai_compatible_model import OpenAICompatibleModel
from benchmark.models.prompt_budget import PromptBudgetError
from benchmark.object_aliasing import ALIAS_MAP_KEY
from benchmark.pipeline import apply_model_overrides
from benchmark.workflow.payloads import build_input_payloads


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_model_parses_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        layout = {
            "scene_id": "tiny",
            "unit": "meter",
            "coordinate_system": {
                "origin": "front-left floor corner",
                "x_axis": "room width",
                "y_axis": "room depth",
                "z_axis": "height",
                "rotation_unit": "degree",
            },
            "objects": [],
            "relations": [],
            "hierarchy": {"regions": [], "floor_objects": [], "supported_objects": []},
        }
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(layout)}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatibleModel(
        name="local",
        endpoint="http://localhost:8000/v1",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        timeout_seconds=12,
    )

    result = model.generate_layout({"case_id": "tiny"}, {"type": "object"})

    assert result["scene_id"] == "tiny"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["payload"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert model.last_request_metadata["endpoint"] == "http://localhost:8000/v1"
    assert model.last_request_metadata["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert model.last_request_metadata["timeout_seconds"] == 12
    assert model.last_request_metadata["message_count"] == 2
    assert model.last_request_metadata["image_count"] == 0
    assert model.last_request_metadata["prompt_chars"] == len(model.last_prompt_text) + len("You generate valid 3D room layout JSON. Return one JSON object only, with no Markdown or explanation.")
    assert model.last_request_metadata["content_chars"] > 0


def test_factory_requires_model_id_for_openai_compatible() -> None:
    with pytest.raises(ValueError, match="concrete model/model_id"):
        create_model(
            "local",
            {
                "models": {
                    "local": {
                        "provider": "openai_compatible",
                        "endpoint": "http://localhost:8000/v1",
                    }
                }
            },
        )


def test_model_overrides_apply_only_selected_model() -> None:
    config = {
        "models": {
            "qwen3vl_sglang": {
                "provider": "openai_compatible",
                "endpoint": "http://localhost:8000/v1",
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "temperature": 0.0,
                "max_tokens": 2048,
                "timeout_seconds": 300,
                "response_format_json": True,
                "judge_evidence_budgeting": True,
            },
            "ollama": {
                "provider": "openai_compatible",
                "endpoint": "http://localhost:11434/v1",
                "model": "qwen2.5:7b-instruct",
            },
        }
    }

    apply_model_overrides(
        config,
        "qwen3vl_sglang",
        {
            "endpoint": "http://localhost:9000/v1",
            "model_id": "local/test-model",
            "temperature": 0.2,
            "max_tokens": 777,
            "timeout_seconds": 33,
            "response_format_json": False,
        },
    )
    model = create_model("qwen3vl_sglang", config)

    assert isinstance(model, OpenAICompatibleModel)
    assert model.endpoint == "http://localhost:9000/v1"
    assert model.model_id == "local/test-model"
    assert model.temperature == 0.2
    assert model.max_tokens == 777
    assert model.timeout_seconds == 33
    assert model.response_format_json is False
    assert model.judge_evidence_budgeting is True
    assert config["models"]["ollama"]["endpoint"] == "http://localhost:11434/v1"


def test_empty_model_overrides_preserve_yaml_defaults() -> None:
    config = {
        "models": {
            "qwen3vl_sglang": {
                "provider": "openai_compatible",
                "endpoint": "http://localhost:8000/v1",
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "temperature": 0.0,
                "max_tokens": 2048,
                "timeout_seconds": 300,
                "response_format_json": True,
            }
        }
    }

    apply_model_overrides(
        config,
        "qwen3vl_sglang",
        {"endpoint": None, "model_id": None, "temperature": None, "max_tokens": None, "timeout_seconds": None, "response_format_json": None},
    )
    model = create_model("qwen3vl_sglang", config)

    assert isinstance(model, OpenAICompatibleModel)
    assert model.endpoint == "http://localhost:8000/v1"
    assert model.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert model.temperature == 0.0
    assert model.max_tokens == 2048
    assert model.timeout_seconds == 300
    assert model.response_format_json is True


def test_factory_applies_shared_api_retry_defaults() -> None:
    config = {
        "api": {"max_retries": 2, "retry_backoff_seconds": 0.25, "retry_on_status": [503]},
        "models": {
            "qwen3vl_sglang": {
                "provider": "openai_compatible",
                "endpoint": "http://localhost:8000/v1",
                "model": "Qwen/Qwen3-VL-8B-Instruct",
            }
        },
    }

    model = create_model("qwen3vl_sglang", config)

    assert isinstance(model, OpenAICompatibleModel)
    assert model.max_retries == 2
    assert model.retry_backoff_seconds == 0.25
    assert model.retry_on_status == [503]
    assert model.judge_evidence_budgeting is False


def test_parse_json_object_wraps_malformed_extracted_json() -> None:
    raw = 'prefix {"scene_id":"bad","objects":[{"object_id":"a" "category":"chair"}]} suffix'

    with pytest.raises(ModelResponseError, match="malformed JSON"):
        parse_json_object(raw)


def test_openai_compatible_model_keeps_raw_response_on_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = '{"scene_id":"bad","objects":[{"object_id":"a" "category":"chair"}]}'

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        return _FakeResponse({"choices": [{"message": {"content": malformed}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatibleModel(
        name="local",
        endpoint="http://localhost:8000/v1",
        model_id="Qwen/Qwen3-VL-32B-Instruct",
    )

    with pytest.raises(ModelResponseError, match="malformed JSON"):
        model.generate_layout({"case_id": "bad"}, {"type": "object"})

    assert model.last_response_text == malformed
    assert model.last_request_metadata["content_chars"] == len(malformed)


def test_repair_uses_safe_repair_token_budget_instead_of_large_generation_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        layout = {
            "scene_id": "case_1",
            "unit": "meter",
            "coordinate_system": {"origin": "dataset", "x_axis": "x", "y_axis": "y", "z_axis": "z", "rotation_unit": "degree"},
            "objects": [{"object_id": "chair_1", "category": "chair", "center": [0, 0, 0.5], "size": [1, 1, 1], "yaw": 0}],
        }
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(layout)}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatibleModel(
        name="local",
        endpoint="http://localhost:8000/v1",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        max_tokens=80000,
        context_length=131072,
        response_format_json=True,
    )

    model.repair_layout(
        {"case_id": "case_1", "objects": [{"id": "chair_1", "category": "chair", "bbox_size": [1, 1, 1]}]},
        {"scene_id": "case_1", "objects": [{"object_id": "chair_1", "category": "chair", "center": [0, 0, 0], "size": [1, 1, 1]}]},
        {"task_id": "case_1", "iteration": 0, "repair_targets": ["chair_1"], "violations": []},
        {"type": "object"},
    )

    assert captured["payload"]["max_tokens"] == 16000
    assert model.last_request_metadata["prompt_budget_report"]["max_tokens_source"] == "min(max_tokens, safe_repair_default)"


def test_prompt_budget_fail_fast_prevents_endpoint_call(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        nonlocal called
        called = True
        return _FakeResponse({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatibleModel(
        name="local",
        endpoint="http://localhost:8000/v1",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        max_tokens=100,
        context_length=200,
        prompt_safety_margin_tokens=50,
    )

    with pytest.raises(PromptBudgetError, match="prompt_budget_exceeded"):
        model.generate_layout({"case_id": "huge", "description": {"text": "x" * 5000}}, {"type": "object"})

    assert called is False
    assert model.last_request_metadata["prompt_budget_exceeded"] is True
    assert model.last_request_metadata["prompt_budget_report"]["fits_context"] is False


def test_generation_prompt_explains_hssd_coordinate_mapping() -> None:
    prompt = build_generation_prompt(
        {
            "case_id": "hssd_case",
            "objects": [
                {
                    "id": "obj_1",
                    "category": "chair",
                    "bbox_size": [1, 1, 1],
                    "source_position": [1, 2, 3],
                    "source_floor_position": [1, 3],
                    "source_height_position": 2,
                    "layout_center_hint": [1, 3, 2],
                }
            ],
        },
        {"type": "object"},
    )

    assert "layout_center_hint" in prompt
    assert "source_position" not in prompt.split("Benchmark instance:")[1]
    assert "Never copy source_position into center" in prompt
    assert "layout center is [x, room_depth_y, height_z]" in prompt


def test_model_prompt_compaction_respects_scene_representation_mode() -> None:
    case = {
        "case_id": "hssd_case",
        "input_level": "structured_relation",
        "scene_representation_mode": "compact_objects",
        "objects": [
            {
                "id": "obj_1",
                "category": "chair",
                "semantic_category": "chair",
                "bbox_size": [1, 1, 1],
                "source_position": [1, 2, 3],
                "source_template_name": "very_long_template",
                "hssd_semantic": {"category": "chair", "support": "floor"},
                "layout_center_hint": [1, 3, 2],
            }
        ],
        "relations": [{"id": "rel_001", "type": "near", "subject": "obj_1", "object": "obj_2", "source": "estimated"}],
        "source": {
            "dataset": "hssd-hab",
            "scene_instance": "scenes/demo.scene_instance.json",
            "mesh_imported": False,
            "room_geometry_fidelity": "proxy_rectangle",
            "metadata_inclusion": {"stage_config": True},
        },
    }

    compact = compact_bm_instance_for_model(case)

    assert compact["scene_representation_mode"] == "compact_objects"
    assert "relations" not in compact
    assert compact["required_object_rows_columns"] == ["object_id", "category", "bbox_size", "layout_center_hint"]
    assert compact["required_object_rows"][0][3] == [1, 3, 2]
    assert "source_position" not in str(compact["required_object_rows"])
    assert "source_template_name" not in str(compact["required_object_rows"])
    assert "hssd_semantic" not in str(compact["required_object_rows"])
    assert compact["source"]["room_geometry_fidelity"] == "proxy_rectangle"
    assert "metadata_inclusion" not in compact["source"]

    full_case = {**case, "scene_representation_mode": "full_metadata_budgeted"}
    full = compact_bm_instance_for_model(full_case)
    assert "relations" in full
    assert full["objects"][0]["source_position"] == [1, 2, 3]
    assert full["objects"][0]["source_template_name"] == "very_long_template"
    assert full["objects"][0]["hssd_semantic"]["category"] == "chair"
    assert full["source"]["metadata_inclusion"]["stage_config"] is True


def test_repair_prompt_compacts_debug_evidence() -> None:
    feedback = {
        "task_id": "case_1",
        "repair_targets": ["chair_1"],
        "locked_objects": ["table_1"],
        "violations": [{"category": "room_consistency", "type": "vlm_room_judge", "message": "bad layout"}],
        "debug_evidence": {
            "physical_flags": [{"type": "below_floor", "objects": ["chair_1"], "diagnostics": {"huge": "x" * 5000}}],
            "view_flags": [{"type": "view_warning", "group_id": "group_001", "diagnostics": {"object_pixel_counts": {"chair_1": 12}}}],
            "judge_input_manifest": {
                "selected_groups": [{"group_id": "group_001", "object_ids": ["chair_1"], "views_sent": {"xy": "view.png"}}],
            },
        },
    }

    prompt = build_repair_prompt(
        {"case_id": "case_1", "objects": [{"id": "chair_1", "category": "chair", "bbox_size": [1, 1, 1]}]},
        {"scene_id": "case_1", "objects": [{"object_id": "chair_1", "category": "chair", "center": [0, 0, 0], "size": [1, 1, 1]}]},
        feedback,
        {"type": "object", "properties": {"oversized_schema": "y" * 5000}},
    )

    assert "below_floor" in prompt
    assert "group_001" in prompt
    assert "object_pixel_counts" not in prompt
    assert "views_sent" not in prompt
    assert "oversized_schema" not in prompt
    assert "primary repair plan" in prompt
    assert "moves every repair target by less than 0.01m is not a valid repair" in prompt
    assert "candidate_warning" in prompt
    assert "alleviate large implausible bbox intersections" in prompt
    assert "Do not optimize solely for zero bbox overlap" in prompt
    assert "wall-mounted TVs/art/mirrors/shelves/curtains/panels" in prompt
    assert "objects contained inside storage" in prompt
    assert "strongly prefer the provided suggested_center_for_move_object" in prompt
    assert "Do not blindly follow one suggested center" in prompt
    assert "The suggested vectors and target centers are advisory. They are not an automatic script. Use them to guide a coherent repaired layout." in prompt
    assert "Do not blindly apply every vector" in prompt
    assert "Moving both objects in a colliding pair together usually does not resolve the collision" in prompt
    assert "Do not fix above-wall by moving objects below the floor" in prompt
    assert "Preserve every required object id/category/size" in prompt
    assert "Return full JSON layout, not compact rows" in prompt
    assert "primary actionable repair plan" not in prompt
    assert len(prompt) < 6200


def test_repair_prompt_aliases_feedback_ids_and_messages_when_alias_map_is_available() -> None:
    canonical_id = "20b73dd1f91dd128fb928fb7a032af2a47e79882_001"
    case = {
        "case_id": "case_1",
        "input_level": "structured_basic",
        "scene_representation_mode": "compact_objects",
        "objects": [{"id": canonical_id, "category": canonical_id, "semantic_category": "chair", "bbox_size": [1, 1, 1]}],
    }
    payloads = build_input_payloads(case)
    payload = {**payloads["prompt_payload"], ALIAS_MAP_KEY: payloads["eval_context"][ALIAS_MAP_KEY]}
    current_layout = {
        "scene_id": "case_1",
        "objects": [
            {
                "object_id": canonical_id,
                "model_object_id": "o001",
                "category": canonical_id,
                "model_category": "chair",
                "center": [0, 0, 1.2],
                "size": [1, 1, 1],
                "yaw": 0,
            }
        ],
    }
    feedback = {
        "task_id": "case_1",
        "repair_targets": [canonical_id],
        "locked_objects": [],
        "violations": [
            {
                "category": "physical_debug_flag",
                "type": "above_wall_height",
                "objects": [canonical_id],
                "message": f"{canonical_id} extends above wall_height.",
            }
        ],
        "repair_actions": [
            {
                "action": "lower_below_wall_height",
                "object_id": canonical_id,
                "suggested_center": [0, 0, 0.5],
                "reason": f"lower {canonical_id}",
            }
        ],
    }

    prompt = build_repair_prompt(payload, current_layout, feedback, {"type": "object"})

    assert canonical_id not in prompt
    assert "o001 extends above wall_height" in prompt
    assert '"object_id":"o001"' in prompt
    assert '"repair_targets":["o001"]' in prompt


def test_repair_prompt_uses_compact_rows_and_top_k_feedback() -> None:
    feedback = {
        "task_id": "case_1",
        "repair_targets": [f"obj_{i}" for i in range(60)],
        "locked_objects": [f"locked_{i}" for i in range(100)],
        "violations": [
            {"category": "physical_debug_flag", "type": "serious_collision", "objects": [f"obj_{i}", f"obj_{i+1}"], "message": "collision"}
            for i in range(80)
        ],
        "repair_actions": [
            {"action": "separate_collision_pair", "object_ids": [f"obj_{i}", f"obj_{i+1}"], "suggested_center_for_move_object": [i, 0.12345, 0.5]}
            for i in range(80)
        ],
    }
    current_layout = {
        "scene_id": "case_1",
        "objects": [
            {"object_id": f"obj_{i}", "category": "box", "center": [i + 0.12345, 0, 0.5], "size": [1, 1, 1], "yaw": 0}
            for i in range(5)
        ],
    }
    case = {"case_id": "case_1", "objects": [{"id": f"obj_{i}", "category": "box", "bbox_size": [1, 1, 1]} for i in range(5)]}

    prompt = build_repair_prompt(case, current_layout, feedback, {"type": "object"})

    assert '"columns":["object_id","category","center","size","yaw"]' in prompt
    assert '"omitted_repair_action_count":56' in prompt
    assert '"omitted_locked_object_count":80' in prompt
    assert "Current layout is shown in compact rows" in prompt
    assert "Output must be full JSON layout, not rows" in prompt
