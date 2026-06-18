from __future__ import annotations

import json
import urllib.request

import pytest

from benchmark.models.factory import create_model
from benchmark.models.openai_compatible_model import OpenAICompatibleModel


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
