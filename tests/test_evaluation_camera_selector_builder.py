from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image
import pytest

from benchmark.api import evaluation
from benchmark.models import OpenAICompatibleModel
from benchmark.visual_judge import OpenAICompatibleCameraSelector


def test_evaluation_cli_uses_camera_builder_for_legacy_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selector_config = tmp_path / "camera_selector.json"
    selector_config.write_text(
        json.dumps(
            {
                "endpoint": "https://selector.example.test/v1",
                "model": "selector-model",
                "require_api_key": False,
                "max_images": 2,
            }
        ),
        encoding="utf-8",
    )
    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}\n", encoding="utf-8")
    preview_path = tmp_path / "candidate.png"
    Image.new("RGB", (16, 16), (60, 90, 140)).save(preview_path)

    model_calls: list[dict] = []

    def fake_chat_messages(self, messages, **kwargs):
        model_calls.append(
            {
                "model": self.model_id,
                "endpoint": self.endpoint,
                "messages": messages,
                "kwargs": kwargs,
            }
        )
        return json.dumps(
            {
                "selected_view_ids": ["candidate_00"],
                "action": None,
                "reason": "best trusted candidate",
            }
        )

    monkeypatch.setattr(
        OpenAICompatibleModel,
        "chat_messages",
        fake_chat_messages,
    )
    # Keep this test focused on the CLI construction boundary; the checked-in
    # legacy profile does not require the separated L3 renderer stack.
    monkeypatch.setattr(
        evaluation,
        "is_legacy_game_profile",
        lambda _profile: True,
    )

    observed: dict = {}

    def fake_run_evaluate(**kwargs):
        transport = kwargs["camera_selector"]
        assert isinstance(transport, OpenAICompatibleCameraSelector)
        result = transport.select_camera_views(
            {
                "metric": "collision",
                "object_ids": ["object-a"],
                "candidates": [
                    {
                        "id": "trusted-view",
                        "image_path": str(preview_path),
                        "render_status": "ok",
                        "pose": {
                            "location": [2.0, 1.0, 2.0],
                            "target": [0.0, 0.0, 0.5],
                            "lens_mm": 50.0,
                            "camera_type": "PERSP",
                        },
                    }
                ],
                "max_views": 1,
                "allow_adjustment": False,
                "allowed_actions": [],
                "preview_role": "highlighted_focus",
            }
        )
        observed["transport"] = transport
        observed["result"] = result
        return {
            "benchmark_score": 1.0,
            "reports": {"fixture": {}},
        }

    monkeypatch.setattr(evaluation, "run_evaluate", fake_run_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.api.evaluation",
            "--scene",
            str(scene_path),
            "--out",
            str(tmp_path / "report.json"),
            "--camera-selector-config",
            str(selector_config),
        ],
    )

    evaluation.main()

    transport = observed["transport"]
    result = observed["result"]
    assert transport.production_camera_selector_transport is True
    assert result["selected_view_ids"] == ["trusted-view"]
    assert result["vlm_role"] == "vlm_camera_selector"
    assert transport.last_request_metadata["selection_mode"] == (
        "legacy_query_cov"
    )
    assert len(model_calls) == 1
    assert model_calls[0]["model"] == "selector-model"
    assert model_calls[0]["endpoint"] == (
        "https://selector.example.test/v1"
    )
    assert model_calls[0]["kwargs"]["call_type"] == (
        "vlm_camera_pose.query_cov"
    )
