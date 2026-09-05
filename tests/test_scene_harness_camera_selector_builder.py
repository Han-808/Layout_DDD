from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

from benchmark.models import OpenAICompatibleModel
from benchmark.visual_judge import (
    ActiveVLMCameraSelector,
    OpenAICompatibleCameraSelector,
)
from benchmark.visual_judge.camera_dsl import CameraConstraintSet
from benchmark.visual_judge.interfaces.camera import CameraSelectionRequest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_scene_harness_camera_selector_builder",
    ROOT / "scripts" / "run_scene_harness.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cli_config_builds_production_camera_selector_and_calls_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "camera_selector.json"
    config_path.write_text(
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
    preview = Image.new("RGB", (16, 16), (20, 30, 40))
    preview.putpixel((0, 0), (220, 80, 30))
    preview.save(preview_path)

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
        if kwargs["call_type"] == "vlm_camera_pose.query_cov":
            return json.dumps(
                {
                    "selected_view_ids": ["candidate_00"],
                    "action": None,
                    "reason": "best bounded legacy candidate",
                }
            )
        return json.dumps(
            {
                "selected_view_ids": ["trusted-view"],
                "reason": "best trusted local evidence",
            }
        )

    monkeypatch.setattr(
        OpenAICompatibleModel,
        "chat_messages",
        fake_chat_messages,
    )

    observed: dict = {}

    def fake_run_scene_harness(**kwargs):
        transport = kwargs["camera_active_selector"]
        assert isinstance(
            transport,
            OpenAICompatibleCameraSelector,
        )
        assert kwargs["l3_vlm_camera_selector"] is transport

        # This is the production adapter used by the Controller for an
        # OpenAI-compatible camera-selector transport.
        selector = ActiveVLMCameraSelector(
            transport,
            selection_mode="candidate_only",
        )
        constraints = CameraConstraintSet(
            target_ids=("object-a",),
            required_observations=("target_visible",),
            metric="scale_consistency",
            view_goal="show the selected object at a useful local scale",
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
                    "scene_id": "builder-boundary",
                    "objects": [{"id": "object-a"}],
                },
                evidence_goal={
                    "view_goal": constraints.view_goal,
                },
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
        observed["legacy_result"] = transport.select_camera_views(
            {
                "metric": "collision",
                "object_ids": ["object-a"],
                "candidates": [
                    {
                        "id": "trusted-view",
                        "image_path": str(preview_path),
                        "render_status": "ok",
                        "pose": candidate["pose"],
                    }
                ],
                "max_views": 1,
                "allow_adjustment": False,
                "allowed_actions": [],
                "preview_role": "highlighted_focus",
            }
        )
        observed["transport"] = transport
        return {
            "status": "complete",
            "artifacts": {
                "run_manifest": str(tmp_path / "run_manifest.json"),
            },
        }

    monkeypatch.setattr(
        MODULE,
        "run_scene_harness",
        fake_run_scene_harness,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scene_harness.py",
            "--instruction",
            "Create a room.",
            "--camera-selector-config",
            str(config_path),
            "--out-dir",
            str(tmp_path / "output"),
        ],
    )

    MODULE.main()

    result = observed["result"]
    legacy_result = observed["legacy_result"]
    transport = observed["transport"]
    assert result.selected_view_ids == ("trusted-view",)
    assert result.backend == "vlm_active"
    assert legacy_result["selected_view_ids"] == ["trusted-view"]
    assert legacy_result["vlm_role"] == "vlm_camera_selector"
    assert transport.production_camera_selector_transport is True
    assert transport.last_request_metadata["selection_mode"] == (
        "legacy_query_cov"
    )
    assert len(model_calls) == 2
    assert model_calls[0]["model"] == "selector-model"
    assert model_calls[0]["endpoint"] == (
        "https://selector.example.test/v1"
    )
    assert model_calls[0]["kwargs"]["call_type"] == (
        "camera_selector_candidate_only"
    )
    assert model_calls[1]["kwargs"]["call_type"] == (
        "vlm_camera_pose.query_cov"
    )
