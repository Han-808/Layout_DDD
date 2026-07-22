from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.visual_judge import OpenAICompatibleVLMJudge, adjudicate_p0b_event


class _FakeModel:
    model_id = "judge"
    endpoint = "http://127.0.0.1:8298/v1"

    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []
        self.last_request_metadata = {"image_count": 1}

    def chat_messages(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return json.dumps(self.response)


def _scene() -> dict:
    return {
        "scene_id": "scene",
        "boundary": [[0, 0], [7, 0], [7, 5], [0, 5]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "bed",
                "category": "bed",
                "description": "red bed",
                "center": [3.5, 2.5, 0.5],
                "size": [2.0, 1.5, 1.0],
                "rotation": [0, 0, 0],
                "geometry_provenance": "bbox_proxy",
            },
            {
                "id": "cabinet",
                "category": "cabinet",
                "description": "dark cabinet",
                "center": [4.4, 2.5, 0.5],
                "size": [0.8, 0.8, 1.0],
                "rotation": [0, 0, 0],
                "geometry_provenance": "asset_mesh",
            },
        ],
    }


def test_p0b_request_contains_rich_context_and_injected_local_view(tmp_path: Path) -> None:
    local_view = tmp_path / "bed_cabinet_local.png"
    local_view.write_bytes(b"png")
    overview = tmp_path / "overview.png"
    overview.write_bytes(b"png")
    provider_calls: list[dict] = []

    def local_view_provider(request: dict) -> list[Path]:
        provider_calls.append(request)
        request["scene"]["objects"].clear()
        return [local_view]

    judge_calls: list[dict] = []

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return {"verdict": "invalid", "confidence": 0.9, "reason": "rigid penetration"}

    scene = _scene()
    report = adjudicate_p0b_event(
        metric="collision",
        event={"object_a": "bed", "object_b": "cabinet", "xy_overlap_area": 0.2},
        prompt="Place a dark cabinet beside the red bed.",
        relationships=[{"subject": "cabinet", "predicate": "beside", "object": "bed"}],
        scene=scene,
        detector_evidence={"z_overlap": 0.8, "normalized_overlap": 0.25},
        judge=judge,
        overview_render_evidence=[str(overview)],
        local_view_provider=local_view_provider,
    )

    request = judge_calls[0]
    assert report["verdict"] == "invalid"
    assert report["score"] == 0.0
    assert provider_calls[0]["object_ids"] == ["bed", "cabinet"]
    assert provider_calls[0]["detector_evidence"]["normalized_overlap"] == 0.25
    assert provider_calls[0]["natural_language_prompt"].startswith("Place a dark cabinet")
    assert provider_calls[0]["access"] == "read_only_evidence_request"
    assert len(scene["objects"]) == 2
    assert request["natural_language_prompt"].startswith("Place a dark cabinet")
    assert request["extracted_relationships"][0]["predicate"] == "beside"
    assert [item["category"] for item in request["objects"]] == ["bed", "cabinet"]
    assert request["architecture"]["elements"] == ["floor", "walls", "ceiling"]
    assert request["detector_evidence"]["normalized_overlap"] == 0.25
    assert "unintended physical surface interpenetration" in request["metric_rubric"]
    assert "no verdict prior" in request["metric_rubric"]
    assert request["candidate_selection_policy"] == "high_recall_candidate_no_label_prior"
    assert request["local_render_evidence"] == [str(local_view)]
    assert request["render_evidence"] == [str(local_view), str(overview)]


def test_openai_p0b_judge_requires_binary_verdict(tmp_path: Path) -> None:
    image = tmp_path / "local.png"
    image.write_bytes(b"png")
    model = _FakeModel({"verdict": "valid", "confidence": 0.8, "reason": "intended contact"})

    result = OpenAICompatibleVLMJudge(model).adjudicate_p0b(
        {
            "metric": "collision",
            "event": {"object_a": "bed", "object_b": "cabinet"},
            "natural_language_prompt": "Put the cabinet beside the bed.",
            "extracted_relationships": [],
            "objects": [],
            "architecture": {"boundary": [[0, 0], [7, 0], [7, 5], [0, 5]], "height": 3.0},
            "detector_evidence": {"overlap": 0.01},
            "render_evidence": [str(image)],
        }
    )

    assert result["verdict"] == "valid"
    assert model.calls[0]["kwargs"]["call_type"] == "vlm_judge.p0b.collision"
    assert "insufficient-evidence" in model.calls[0]["messages"][0]["content"]

    invalid_model = _FakeModel({"verdict": "insufficient_evidence", "confidence": 0.2, "reason": "occluded"})
    with pytest.raises(ValueError, match="exactly 'valid' or 'invalid'"):
        OpenAICompatibleVLMJudge(invalid_model).adjudicate_p0b(
            {
                "metric": "support",
                "render_evidence": [],
            }
        )


def test_support_rubric_requires_grounded_ancestry_for_local_contact() -> None:
    calls: list[dict] = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"verdict": "invalid", "confidence": 1.0, "reason": "floating stack"}

    adjudicate_p0b_event(
        metric="support",
        event={"object_id": "top"},
        prompt="Stack the objects.",
        relationships=[],
        scene={"objects": [{"id": "top"}]},
        detector_evidence={"certified_grounded_support": False},
        judge=judge,
    )

    rubric = calls[0]["metric_rubric"]
    assert "full support chain" in rubric
    assert "local contact with a floating or ungrounded support is not" in rubric


def test_oob_rubric_requires_positive_structural_exception_evidence() -> None:
    from benchmark.visual_judge.p0b import P0B_METRIC_RUBRICS

    rubric = P0B_METRIC_RUBRICS["oob"]
    assert "authoritative facts" in rubric
    assert "Return invalid by default" in rubric
    assert "against_wall" in rubric
    assert "never exempt ordinary furniture" in rubric
    assert "If qualifying structural evidence is absent, return invalid" in rubric


def test_oob_uses_fixed_global_before_deterministic_local(tmp_path: Path) -> None:
    global_top = tmp_path / "standardized_top.png"
    global_perspective = tmp_path / "standardized_perspective.png"
    local = tmp_path / "oob_plane_normal.png"
    for path in (global_top, global_perspective, local):
        path.write_bytes(b"png")

    calls: list[dict] = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"verdict": "invalid", "confidence": 1.0, "reason": "measured crossing"}

    adjudicate_p0b_event(
        metric="oob",
        event={"object_id": "bed", "plane_flags": {"east_oob": True}},
        prompt="Place the bed near the east wall.",
        relationships=[{"type": "near_wall", "subject_id": "bed", "target": "east_wall"}],
        scene=_scene(),
        detector_evidence={
            "plane_flags": {"east_oob": True},
            "numerical_eps": 1.0e-6,
        },
        judge=judge,
        object_ids=["bed"],
        overview_render_evidence=[str(global_top), str(global_perspective)],
        local_view_provider=lambda request: [str(local)],
    )

    request = calls[0]
    assert request["render_evidence"] == [
        str(global_top),
        str(global_perspective),
        str(local),
    ]
    assert request["visual_evidence_policy"] == {
        "default_bundle": "fixed_global_plus_deterministic_local",
        "image_order": ["fixed_global", "deterministic_local"],
        "fixed_global_view_count": 2,
        "deterministic_local_view_count": 1,
        "local_camera_mode": "visibility_ranked",
        "pose_selector": "deterministic",
    }
