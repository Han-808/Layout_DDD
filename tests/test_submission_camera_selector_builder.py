from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

from PIL import Image
import pytest

from benchmark.api import submission
from benchmark.models import OpenAICompatibleModel
from benchmark.visual_judge import (
    ActiveVLMCameraSelector,
    OpenAICompatibleCameraSelector,
)
from benchmark.visual_judge.camera_dsl import CameraConstraintSet
from benchmark.visual_judge.interfaces.camera import CameraSelectionRequest


def test_submission_cli_builds_camera_transport_and_executes_selection(
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
                "max_preview_images": 2,
            }
        ),
        encoding="utf-8",
    )
    preview_path = tmp_path / "trusted_view.png"
    Image.new("RGB", (16, 16), (40, 70, 120)).save(preview_path)

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
                "selected_view_ids": ["trusted-view"],
                "reason": "best trusted evidence",
            }
        )

    monkeypatch.setattr(
        OpenAICompatibleModel,
        "chat_messages",
        fake_chat_messages,
    )
    monkeypatch.setattr(
        submission,
        "load_case_bundle",
        lambda _path: SimpleNamespace(),
    )

    observed: dict = {}

    def fake_evaluate_submission(**kwargs):
        transport = kwargs["camera_selector"]
        assert isinstance(transport, OpenAICompatibleCameraSelector)
        selector = ActiveVLMCameraSelector(
            transport,
            selection_mode="candidate_only",
        )
        constraints = CameraConstraintSet(
            target_ids=("object-a",),
            required_observations=("target_visible",),
            metric="scale_consistency",
            view_goal="show object-a at a useful local scale",
        )
        candidate = {
            "id": "trusted-view",
            "image_path": str(preview_path),
            "render_status": "ok",
            "technical_feasibility": True,
            "target_ids": ["object-a"],
            "pose": {
                "location": [2.0, 1.0, 2.0],
                "target": [0.0, 0.0, 0.5],
                "lens_mm": 50.0,
                "camera_type": "PERSP",
            },
        }
        result = selector.select(
            CameraSelectionRequest(
                task="scale_consistency",
                metric="scale_consistency",
                target_ids=("object-a",),
                scene={
                    "scene_id": "submission-cli-boundary",
                    "objects": [{"id": "object-a"}],
                },
                evidence_goal={"view_goal": constraints.view_goal},
                existing_visual_evidence=(),
                budget={
                    "max_views_per_round": 1,
                    "candidate_budget": 1,
                },
                constraints=constraints.to_dict(),
                candidate_views=(candidate,),
                context={
                    "known_target_ids": ["object-a"],
                    "vlm_selection_mode": "candidate_only",
                },
            )
        )
        observed["result"] = result
        observed["transport"] = transport
        return {
            "evaluation_report": {
                "benchmark_score": 1.0,
                "official_submission": False,
            }
        }

    monkeypatch.setattr(
        submission,
        "evaluate_submission",
        fake_evaluate_submission,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.api.submission",
            "--case-bundle",
            str(tmp_path / "case_bundle"),
            "--scene",
            str(tmp_path / "scene.json"),
            "--out-dir",
            str(tmp_path / "output"),
            "--camera-selector-config",
            str(selector_config),
            "--diagnostic",
        ],
    )

    submission.main()

    result = observed["result"]
    transport = observed["transport"]
    assert result.selected_view_ids == ("trusted-view",)
    assert result.backend == "vlm_active"
    assert transport.production_camera_selector_transport is True
    assert transport.last_request_metadata["selection_mode"] == (
        "candidate_only"
    )
    assert len(model_calls) == 1
    assert model_calls[0]["model"] == "selector-model"
    assert model_calls[0]["endpoint"] == (
        "https://selector.example.test/v1"
    )
    assert model_calls[0]["kwargs"]["call_type"] == (
        "camera_selector_candidate_only"
    )
