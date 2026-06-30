from __future__ import annotations

import json
import urllib.request

import pytest

from benchmark.models.base_model import ModelResponseError, build_generation_prompt, build_repair_prompt, parse_json_object
from benchmark.models.factory import create_model
from benchmark.models.openai_compatible_model import OpenAICompatibleModel
from benchmark.pipeline import apply_model_overrides


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
    assert "advisory hints, not mandatory edits" in prompt
    assert "candidate_warning" in prompt
    assert "alleviate large implausible bbox intersections" in prompt
    assert "Do not optimize solely for zero bbox overlap" in prompt
    assert "wall-mounted TVs/art/mirrors/shelves/curtains/panels" in prompt
    assert "objects contained inside storage" in prompt
    assert "strongly prefer the provided suggested_center_for_move_object" in prompt
    assert "Do not blindly follow one suggested center" in prompt
    assert "primary actionable repair plan" not in prompt
    assert len(prompt) < 5000
