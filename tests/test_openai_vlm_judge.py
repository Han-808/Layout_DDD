from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from benchmark.visual_judge import OpenAICompatibleVLMJudge, build_openai_compatible_vlm_judge, evaluate_vlm_category
from benchmark.visual_judge.openai_compatible import _selector_candidate_order_key


class FakeMultimodalModel:
    model_id = "Qwen3-VL-32B-Instruct-64K"
    endpoint = "http://127.0.0.1:8298/v1"

    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = []
        self.last_request_metadata = {"image_count": 1}

    def chat_messages(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return json.dumps(self.response)


def _request(image_path: Path) -> dict:
    return {
        "category": "visual_quality",
        "prompt": None,
        "scene_summary": {
            "scene_id": "scene",
            "objects": [{"id": "bed", "rotation_degrees": [0, 0, 5]}],
        },
        "deterministic_evidence": {"generic_validity": {"score": 0.9}},
        "render_evidence": [str(image_path)],
    }


def test_openai_compatible_vlm_judge_sends_images_and_structured_prior(tmp_path: Path) -> None:
    image_path = tmp_path / "standardized_top.png"
    image_path.write_bytes(b"fake png")
    model = FakeMultimodalModel(
        {"applicable": True, "score": 0.75, "confidence": 0.8, "summary": "plausible", "issues": [], "evidence": ["top view"]}
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.evaluate(_request(image_path))

    content = model.calls[0]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert "generic_validity" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert model.calls[0]["kwargs"]["response_format_json"] is True
    assert result["score"] == 0.75
    assert result["model"] == "Qwen3-VL-32B-Instruct-64K"


def test_vlm_category_supports_not_applicable_response(tmp_path: Path) -> None:
    image_path = tmp_path / "view.png"
    image_path.write_bytes(b"fake png")
    model = FakeMultimodalModel(
        {"applicable": False, "score": None, "confidence": 0.2, "summary": "object is occluded", "issues": [], "evidence": []}
    )

    report = evaluate_vlm_category(
        category="visual_quality",
        prompt=None,
        scene={"scene_id": "scene", "objects": []},
        render_evidence=[str(image_path)],
        judge=OpenAICompatibleVLMJudge(model),
    )

    assert report["status"] == "not_evaluable"
    assert report["score"] is None
    assert report["reason"] == "vlm_judge_insufficient_evidence"


def test_vlm_category_preserves_canonical_description_and_proxy_status(tmp_path: Path) -> None:
    image_path = tmp_path / "view.png"
    image_path.write_bytes(b"fake png")
    model = FakeMultimodalModel(
        {"applicable": True, "score": 0.5, "confidence": 0.7, "summary": "layout only", "issues": [], "evidence": []}
    )

    evaluate_vlm_category(
        category="prompt_fidelity",
        prompt="Place a red bed in the room.",
        scene={
            "scene_id": "scene",
            "objects": [
                {
                    "id": "bed_1",
                    "category": "bed",
                    "description": "red bed",
                    "center": [1, 1, 0.5],
                    "size": [2, 1.5, 1],
                    "asset_proxy": {"type": "obb", "bbox_size": [2, 1.5, 1]},
                    "metadata": {"asset_resolution": "unresolved"},
                }
            ],
        },
        render_evidence=[str(image_path)],
        judge=OpenAICompatibleVLMJudge(model),
    )

    context = model.calls[0]["messages"][1]["content"][0]["text"]
    assert '"description":"red bed"' in context
    assert '"asset_resolution":"unresolved"' in context
    assert '"proxy_type":"obb"' in context


def test_judge_builder_supports_mnet_and_remote_api_config() -> None:
    local = build_openai_compatible_vlm_judge(
        {"endpoint": "http://127.0.0.1:8298/v1", "model": "Qwen3-VL-32B-Instruct-64K"}
    )
    remote = build_openai_compatible_vlm_judge(
        {"endpoint": "https://api.example.com/v1", "model": "remote-vlm", "api_key_env": "REMOTE_API_KEY"}
    )

    assert local.model.api_key_env is None
    assert remote.model.api_key_env == "REMOTE_API_KEY"
    assert remote.model.endpoint == "https://api.example.com/v1"


def test_relation_judge_is_binary_and_receives_prompt_claim_and_image(tmp_path: Path) -> None:
    image_path = tmp_path / "local_relation_view.png"
    image_path.write_bytes(b"fake png")
    model = FakeMultimodalModel(
        {"verdict": "valid", "confidence": 0.85, "reason": "the requested relation is visible"}
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.adjudicate_relation(
        {
            "family": "oor",
            "relation": {"type": "mirrors", "subject_id": "a", "object_id": "b"},
            "natural_language_prompt": "A mirrors B.",
            "involved_objects": [{"id": "a"}, {"id": "b"}],
            "scene_summary": {"scene_id": "scene", "objects": []},
            "detector_evidence": {"route": "requires_vlm", "proxy": "obb"},
            "render_evidence": [str(image_path)],
        }
    )

    content = model.calls[0]["messages"][1]["content"]
    assert "A mirrors B." in content[0]["text"]
    assert '"type":"mirrors"' in content[0]["text"]
    assert '"route":"requires_vlm"' in content[0]["text"]
    assert '"proxy":"obb"' in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert model.calls[0]["kwargs"]["call_type"] == "vlm_judge.relationship.oor"
    assert result["verdict"] == "valid"
    assert result["confidence"] == 0.85


def test_spatial_fidelity_judge_is_binary_and_treats_rarity_as_routing_only(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "spatial_pair.png"
    image_path.write_bytes(b"fake png")
    model = FakeMultimodalModel(
        {
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "unusual but coherent for a studio bedroom",
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.adjudicate_spatial_fidelity(
        {
            "metric": "cooccurrence_plausibility",
            "event": {
                "type": "rare_category_cooccurrence",
                "object_ids": ["bed", "fridge"],
            },
            "detector_evidence": {
                "best_directional_probability": 0.005,
                "candidate_route": "requires_vlm",
            },
            "involved_objects": [{"id": "bed"}, {"id": "fridge"}],
            "scene_summary": {"scene_id": "studio", "objects": []},
            "natural_language_prompt": "Create a compact studio bedroom.",
            "render_evidence": [str(image_path)],
        }
    )

    system = model.calls[0]["messages"][0]["content"]
    content = model.calls[0]["messages"][1]["content"]
    assert "dataset rarity or absence alone is never an error" in system
    assert '"best_directional_probability":0.005' in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert model.calls[0]["kwargs"]["call_type"] == (
        "vlm_judge.spatial_fidelity.cooccurrence_plausibility"
    )
    assert result["verdict"] == "valid"
    assert result["confidence"] == 0.9


def test_relation_context_budget_preserves_detector_packet_ahead_of_large_scene(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "relation.png"
    image_path.write_bytes(b"fake png")
    model = FakeMultimodalModel(
        {"verdict": "invalid", "confidence": 0.9, "reason": "proxy and image disagree"}
    )
    judge = OpenAICompatibleVLMJudge(model, max_context_chars=1000)

    judge.adjudicate_relation(
        {
            "family": "oar",
            "relation": {"type": "mounted_on_wall", "subject_id": "painting", "wall": "north"},
            "natural_language_prompt": "Mount the painting on the north wall. " + "detail " * 1000,
            "involved_objects": [{"id": "painting"}],
            "detector_evidence": {
                "route": "requires_vlm",
                "proxy_checks_passed": False,
                "signed_wall_clearance_m": 0.42,
            },
            "scene_summary": {
                "scene_id": "large",
                "objects": [{"id": f"object_{index}", "description": "x" * 300} for index in range(50)],
            },
            "render_evidence": [str(image_path)],
        }
    )

    text = model.calls[0]["messages"][1]["content"][0]["text"]
    context_text = text.split("\n", 1)[1]
    context = json.loads(context_text)
    assert len(context_text) <= 1000
    assert context["explicit_relation_claim"]["type"] == "mounted_on_wall"
    assert context["detector_evidence"]["signed_wall_clearance_m"] == 0.42
    assert context["_benchmark_context_budget"]["truncated"] is True


def test_p0b_context_budget_reserves_every_priority_field(tmp_path: Path) -> None:
    image_path = tmp_path / "p0b.png"
    image_path.write_bytes(b"fake png")
    model = FakeMultimodalModel(
        {"verdict": "valid", "confidence": 0.9, "reason": "test"}
    )
    judge = OpenAICompatibleVLMJudge(model, max_context_chars=1000)
    huge = "x" * 5000

    judge.adjudicate_p0b(
        {
            "metric": "collision",
            "event": {"event_detail": huge},
            "detector_evidence": {"detector_sentinel": huge},
            "natural_language_prompt": huge,
            "metric_rubric": huge,
            "candidate_selection_policy": "high_recall_candidate_no_label_prior",
            "collision_evidence_style_guide": huge,
            "local_render_evidence_metadata": {"metadata": huge},
            "objects": [{"description": huge}],
            "architecture": {"description": huge},
            "render_evidence": [str(image_path)],
        }
    )

    text = model.calls[0]["messages"][1]["content"][0]["text"]
    context_text = text.split("\n", 1)[1]
    context = json.loads(context_text)
    assert len(context_text) <= 1000
    for key in (
        "metric",
        "event",
        "detector_evidence",
        "natural_language_prompt",
        "metric_rubric",
        "candidate_selection_policy",
        "collision_evidence_style_guide",
        "view_names",
        "view_evidence",
    ):
        assert key in context
    assert "detector_sentinel" in context["detector_evidence"]["json_prefix"]


def test_query_cov_camera_selector_chooses_views_without_metric_verdict(tmp_path: Path) -> None:
    first = tmp_path / "candidate_a.png"
    second = tmp_path / "candidate_b.png"
    private_metadata = PngImagePlugin.PngInfo()
    private_metadata.add_text("private_source", "must-not-leave-process")
    Image.new("RGBA", (3, 2), (255, 0, 0, 64)).save(first, pnginfo=private_metadata)
    Image.new("RGBA", (3, 2), (0, 0, 255, 128)).save(second)
    candidates = [
        {
            "id": "generator_first_private_id",
            "pose": {
                "location": [1, 2, 3],
                "target": [0, 0, 0],
                "azimuth_degrees": 10.0,
                "elevation_degrees": 20.0,
                "target_object_ids": ["bed", "cabinet"],
            },
            "image_path": str(first),
        },
        {
            "id": "generator_second_private_id",
            "pose": {
                "location": [3, 2, 1],
                "target": [0, 0, 0],
                "azimuth_degrees": 200.0,
                "elevation_degrees": 15.0,
                "target_object_ids": ["bed", "cabinet"],
            },
            "image_path": str(second),
        },
    ]
    digest_order = sorted(candidates, key=_selector_candidate_order_key)
    candidates = list(reversed(digest_order))
    assert candidates[0]["id"] != digest_order[0]["id"]
    model = FakeMultimodalModel(
        {
            "selected_view_ids": ["candidate_00"],
            "action": {"view_id": "candidate_00", "type": "orbit_left"},
            "reason": "the overlap is visible but needs a small lateral change",
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.select_camera_views(
        {
            "metric": "collision",
            "event": {"object_a": "bed", "object_b": "cabinet"},
            "object_ids": ["bed", "cabinet"],
            "detector_evidence": {"normalized_overlap": 0.2, "secret": "detector-secret"},
            "natural_language_prompt": "private-prompt-sentinel",
            "extracted_relationships": [{"private-relation-sentinel": True}],
            "candidates": candidates,
            "max_views": 1,
            "allow_adjustment": True,
            "allowed_actions": ["orbit_left", "orbit_right"],
            "preview_role": "highlighted_focus",
            "color_legend": [
                {
                    "id": "bed",
                    "category": "private-category",
                    "role": "object_a",
                    "color": [1.0, 0.0, 0.0],
                    "representation": "asset_mesh",
                }
            ],
            # This is the field emitted by the active CameraEvidenceProvider.
            "preview_degradation": "focus_preview_failed: RuntimeError: incomplete highlight coverage",
        }
    )

    system = model.calls[0]["messages"][0]["content"]
    assert "Do not judge whether the metric event is valid or invalid" in system
    assert model.calls[0]["kwargs"]["call_type"] == "vlm_camera_pose.query_cov"
    assert result["selected_view_ids"] == [digest_order[0]["id"]]
    assert result["action"]["view_id"] == digest_order[0]["id"]
    assert result["action"]["type"] == "orbit_left"
    assert result["images_used"] == ["candidate_00", "candidate_01"]
    assert result["request_metadata"]["selector_candidate_order_policy"] == "stable_pose_image_digest_v1"
    assert result["request_metadata"]["selector_candidate_alias_policy"] == "per_request_sequential_alias_v1"

    outbound = model.calls[0]["messages"][1]["content"]
    context = json.loads(outbound[0]["text"].split("\n", 1)[1])
    assert set(context) == {
        "candidates",
        "max_views",
        "allow_adjustment",
        "allowed_actions",
        "metric_family",
        "preview_role",
        "preview_warning_class",
        "color_legend",
    }
    assert [item["id"] for item in context["candidates"]] == ["candidate_00", "candidate_01"]
    assert context["metric_family"] == "collision"
    assert context["preview_warning_class"] == "incomplete_target_visibility"
    assert context["color_legend"] == [
        {"role": "object_a", "rgb": [1.0, 0.0, 0.0], "representation": "mesh"}
    ]
    serialized = json.dumps(context)
    for private_value in (
        "bed",
        "cabinet",
        "detector-secret",
        "private-prompt-sentinel",
        "private-relation-sentinel",
        "private-category",
        "generator_first_private_id",
        "generator_second_private_id",
        first.name,
        second.name,
        str(tmp_path),
    ):
        assert private_value not in serialized
    assert all("location" not in item["pose"] and "target" not in item["pose"] for item in context["candidates"])

    encoded = outbound[1]["image_url"]["url"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as normalized:
        assert normalized.mode == "RGB"
        assert normalized.info == {}


def test_query_cov_camera_selector_excludes_blank_candidate_previews(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    usable = tmp_path / "usable.png"
    blank.write_bytes(b"blank")
    Image.new("RGB", (2, 2), (12, 34, 56)).save(usable)
    model = FakeMultimodalModel(
        {
            "selected_view_ids": ["candidate_00"],
            "action": None,
            "reason": "the usable candidate exposes the target",
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.select_camera_views(
        {
            "metric": "support",
            "candidates": [
                {
                    "id": "blank",
                    "pose": {"location": [1, 2, 3]},
                    "image_path": str(blank),
                    "render_status": "blank",
                },
                {
                    "id": "usable",
                    "pose": {"location": [3, 2, 1]},
                    "image_path": str(usable),
                    "render_status": "ok",
                },
            ],
            "max_views": 1,
            "allow_adjustment": False,
        }
    )

    assert result["selected_view_ids"] == ["usable"]
    assert result["images_used"] == ["candidate_00"]
    context = model.calls[0]["messages"][1]["content"][0]["text"]
    assert '"id": "blank"' not in context


def test_query_cov_camera_selector_rejects_invalid_preview_image(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not-an-image")
    judge = OpenAICompatibleVLMJudge(
        FakeMultimodalModel({"selected_view_ids": ["candidate_00"], "action": None})
    )

    with pytest.raises(ValueError, match="candidate_00 is not a valid decodable image"):
        judge.select_camera_views(
            {
                "metric": "support",
                "candidates": [
                    {"id": "private_internal_id", "pose": {}, "image_path": str(invalid)}
                ],
                "max_views": 1,
                "allow_adjustment": False,
            }
        )
