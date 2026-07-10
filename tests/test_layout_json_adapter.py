from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import pytest

from benchmark.adapters import get_adapter
from benchmark.adapters.layout_json.converter import convert_layout_json_to_scene, validate_layout_json
from benchmark.nl_scene.generation_input import build_generation_input, build_scene_request
from benchmark.scene_io.validate import ArtifactValidationError, validate_generated_scene
from benchmark.utils.io import read_json
from generate import run_generate
from scripts.run_scene_harness import run_scene_harness


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _object_plan(request_id: str = "smoke") -> dict:
    return {
        "request_id": request_id,
        "scene_type": "bedroom",
        "scene_description": "A room with a red bed.",
        "objects": [
            {
                "id": "hidden_bed_id",
                "role": "",
                "category": "bed",
                "description": "red bed",
                "count": 1,
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
                "metadata": {},
            }
        ],
        "global_constraints": [],
        "relations": [],
    }


def _generation_input(*, structure: bool = False, request_id: str = "smoke") -> dict:
    request = build_scene_request(
        request_id=request_id,
        instruction="A room with a red bed in the middle.",
        scene_type="bedroom",
        room={"boundary": [[0, 0], [8, 0], [8, 8], [0, 8]], "height": 2.8, "unit": "meter"},
        structure=structure,
    )
    return build_generation_input(scene_request=request, object_plan=_object_plan(request_id))


def _layout_json() -> dict:
    return {
        "schema_version": "layout_json_v1",
        "scene_type": "bedroom",
        "room": {"size": [8, 8, 2.8]},
        "objects": [
            {
                "id": "bed_1",
                "category": "bed",
                "description": "red bed",
                "center": [4, 4, 0.5],
                "size": [2, 2, 1],
                "rotation": [0, 0, 0],
            },
            {
                "id": "drawer_1",
                "category": "drawer",
                "description": "wooden drawer",
                "center": [5.5, 4, 0.4],
                "size": [0.8, 0.6, 0.8],
                "rotation": [0, 0, 0],
            },
        ],
        "relationships": [
            {"subject": "drawer_1", "predicate": "right", "object": "bed_1"}
        ],
    }


def test_direct_layout_json_prompt_exposes_only_natural_language(tmp_path: Path) -> None:
    adapter = get_adapter("layout_json")
    method_input_path = adapter.prepare_input(_generation_input(structure=False), tmp_path)
    method_input = read_json(method_input_path)
    prompt = method_input["messages"][1]["content"]

    assert method_input["output_schema"] == "layout_json_v1"
    assert method_input["input_mode"] == "natural_language_direct"
    assert "A room with a red bed in the middle." in prompt
    assert "hidden_bed_id" not in prompt
    assert "object_plan" not in prompt


def test_layout_json_converter_builds_canonical_proxy_scene() -> None:
    scene = convert_layout_json_to_scene(_layout_json(), _generation_input())

    assert validate_generated_scene(scene)
    assert scene["request_id"] == "smoke"
    assert scene["boundary"] == [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]]
    assert scene["objects"][0]["jid"] == "layout_json_proxy:bed_1"
    assert scene["objects"][0]["metadata"]["asset_resolution"] == "unresolved"
    assert scene["objects"][0]["asset_proxy"]["bbox_size"] == [2.0, 2.0, 1.0]
    assert scene["relations"] == [{"subject_id": "drawer_1", "type": "right", "object_id": "bed_1"}]


def test_layout_json_validation_rejects_missing_object_size() -> None:
    malformed = _layout_json()
    malformed["objects"][0].pop("size")

    with pytest.raises(ArtifactValidationError, match="layout_json_v1 validation failed"):
        validate_layout_json(malformed)


def test_layout_json_adapter_calls_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(_layout_json())}, "finish_reason": "stop"}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = run_generate(
        generation_input=_generation_input(),
        adapter_name="layout_json",
        out_dir=tmp_path,
        run_generation=True,
        adapter_config={
            "endpoint": "http://127.0.0.1:8298/v1",
            "model": "Qwen3-VL-32B-Instruct-64K",
            "timeout_seconds": 12,
        },
    )

    scene = read_json(result["generated_scene"])
    metadata = read_json(result["adapter_metadata"])
    assert captured["url"] == "http://127.0.0.1:8298/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["payload"]["model"] == "Qwen3-VL-32B-Instruct-64K"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert scene["metadata"]["generator_output_schema"] == "layout_json_v1"
    assert metadata["generation_run"]["model"] == "Qwen3-VL-32B-Instruct-64K"
    assert (tmp_path / "model_response.txt").exists()
    assert (tmp_path / "layout_json_output.json").exists()
    assert (tmp_path / "model_request_metadata.json").exists()


def test_layout_json_adapter_translates_reflection_into_repair_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(_layout_json())}, "finish_reason": "stop"}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    run_generate(
        generation_input=_generation_input(),
        adapter_name="layout_json",
        out_dir=tmp_path,
        run_generation=True,
        adapter_config={"endpoint": "http://127.0.0.1:8298/v1", "model": "Qwen3-VL-32B-Instruct-64K"},
        evaluation_report={"overall_score": 0.5, "reports": {"generic_validity": {"overall_score": 0.5}}},
        previous_generated_scene={"scene_id": "previous", "objects": []},
        iteration=1,
    )

    prompt = captured["payload"]["messages"][1]["content"]
    assert "repair_context" in prompt
    assert "previous_generated_scene" in prompt
    assert "previous_evaluation" in prompt


def test_layout_json_runs_through_scene_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        nonlocal calls
        calls += 1
        if calls == 1:
            plan = _object_plan("harness_smoke")
            return _FakeResponse({"choices": [{"message": {"content": json.dumps(plan)}, "finish_reason": "stop"}]})
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(_layout_json())}, "finish_reason": "stop"}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out_dir = tmp_path / "harness_smoke"
    manifest = run_scene_harness(
        instruction="A room with a red bed in the middle.",
        scene_type="bedroom",
        structure=False,
        asset_mode="off",
        adapter="layout_json",
        adapter_config={"endpoint": "http://127.0.0.1:8298/v1", "model": "Qwen3-VL-32B-Instruct-64K"},
        run_generation=True,
        out_dir=out_dir,
    )

    assert manifest["status"] == "generated_scene_available"
    assert manifest["adapter"]["generator_output_schema"] == "layout_json_v1"
    assert manifest["asset_resolution"]["mode"] == "off"
    assert manifest["converter"]["model_config_source"] == "generator_adapter_fallback"
    assert manifest["converter"]["model"] == "Qwen3-VL-32B-Instruct-64K"
    assert calls == 2
    assert read_json(out_dir / "generation_input.json")["generation_contract"]["input_mode"] == "natural_language_direct"
    assert read_json(out_dir / "generated_scene.json")["metadata"]["output_adapter"] == "layout_json"
    assert (out_dir / "evaluation_report.json").exists()
