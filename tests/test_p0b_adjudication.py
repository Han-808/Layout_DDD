from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from benchmark.visual_judge import OpenAICompatibleVLMJudge, adjudicate_p0b_event
from benchmark.visual_judge.runtime import EvidenceControlUnresolvedError
from benchmark.task_contract import architecture_contract_for_room


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


def _non_vlm_stub(call):
    call.vlm_control_enabled = False
    return call


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

    @_non_vlm_stub
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
    assert set(request["architecture"]) == {
        "logical_boundary",
        "floor",
        "ceiling",
    }
    assert "physical_walls" not in request["architecture"]
    assert request["judge_context_policy_version"] == (
        "p0b_single_room_baseline_plus_physical_walls_v2"
    )
    assert request["detector_evidence"]["normalized_overlap"] == 0.25
    assert "unintended physical surface interpenetration" in request["metric_rubric"]
    assert "no verdict prior" in request["metric_rubric"]
    assert request["candidate_selection_policy"] == "high_recall_candidate_no_label_prior"
    assert request["local_render_evidence"] == [str(local_view)]
    assert request["render_evidence"] == [str(local_view), str(overview)]
    assert request["vlm_role"] == "judge"
    assert request["decision_contract"] == "p0b_binary_v1"
    assert request["judge_method"] == "adjudicate_p0b"


def test_promptless_wall_context_is_physical_only_for_support_and_oob() -> None:
    scene = _scene()
    scene["objects"][1]["metadata"] = {
        "task_slot": {
            "description": "private-wall-mounted-generator-claim",
        }
    }
    scene["metadata"] = {
        "architecture_contract": architecture_contract_for_room(
            {
                "boundary": scene["boundary"],
                "height": scene["scene_height"],
            },
            physical_wall_policy="explicit_only",
            active_wall_ids=("north_wall",),
            policy_source="private-generator-policy-source",
        )
    }
    calls: list[dict] = []

    @_non_vlm_stub
    def judge(request: dict) -> dict:
        calls.append(request)
        return {"verdict": "valid", "confidence": 0.9, "reason": "test"}

    support_evidence = {
        "support_instruction": "duplicated instruction",
        "evaluated_object": {"id": "cabinet"},
        "routing_reasons": ["possible_architecture_attachment"],
        "active_physical_wall_ids": ["north_wall"],
        "architecture_wall_surface_model": {
            "policy": "private-policy",
            "active_wall_ids": ["north_wall"],
            "physical_wall_center_plane": "logical_room_boundary",
            "physical_wall_thickness_m": 0.08,
            "physical_wall_thickness_source": "private-source",
            "inner_surface_offset_m": 0.04,
        },
        "architecture_contact_candidates": [
            {
                "plane": "north",
                "signed_clearance_m": 0.01,
                "mode": "wall_attachment",
            }
        ],
    }
    adjudicate_p0b_event(
        metric="support",
        event={
            "object_id": "cabinet",
            "active_physical_wall_ids": ["north_wall"],
            "architecture_contact_candidates": ["private-duplicate"],
        },
        prompt="",
        relationships=[],
        scene=scene,
        detector_evidence=support_evidence,
        judge=judge,
    )
    support_request = calls[-1]
    assert "natural_language_prompt" not in support_request
    assert "extracted_relationships" not in support_request
    assert support_request["architecture"]["physical_walls"] == {
        "active_wall_ids": ["north_wall"],
        "wall_thickness_m": 0.08,
        "center_plane": "logical_room_boundary",
        "inner_surface_offset_m": 0.04,
    }
    assert support_request["detector_evidence"][
        "architecture_contact_candidates"
    ] == [{"plane": "north", "signed_clearance_m": 0.01}]
    serialized = json.dumps(support_request, sort_keys=True)
    assert "private-wall-mounted-generator-claim" not in serialized
    assert "private-generator-policy-source" not in serialized
    assert "private-policy" not in serialized
    assert "private-source" not in serialized
    assert "duplicated instruction" not in serialized
    assert "private-duplicate" not in serialized

    adjudicate_p0b_event(
        metric="oob",
        event={
            "object_id": "cabinet",
            "architecture_element": "room_envelope",
        },
        prompt="",
        relationships=[],
        scene=scene,
        detector_evidence={"plane_flags": ["north_oob"]},
        judge=judge,
    )
    oob_request = calls[-1]
    assert oob_request["architecture"]["physical_walls"] == {
        "active_wall_ids": ["north_wall"],
        "wall_thickness_m": 0.08,
        "center_plane": "logical_room_boundary",
        "inner_surface_offset_m": 0.04,
    }
    assert oob_request["architecture"]["logical_boundary"]["boundary"] == (
        scene["boundary"]
    )
    oob_serialized = json.dumps(oob_request, sort_keys=True)
    assert "private-wall-mounted-generator-claim" not in oob_serialized
    assert "private-generator-policy-source" not in oob_serialized


def test_support_zero_local_evidence_forces_binary_choice() -> None:
    model = _FakeModel(
        {
            "verdict": "valid",
            "confidence": 0.7,
            "reason": "deterministic context supports a binary conclusion",
        }
    )

    result = adjudicate_p0b_event(
        metric="support",
        event={"object_id": "bed"},
        prompt="Place the bed in the room.",
        relationships=[],
        scene=_scene(),
        detector_evidence={"gap_m": 0.0},
        judge=OpenAICompatibleVLMJudge(model),
        object_ids=["bed"],
        overview_render_evidence=[],
        local_view_provider=None,
    )

    assert result["verdict"] == "valid"
    assert len(model.calls) == 1
    assert (
        model.calls[0]["kwargs"]["call_type"]
        == "vlm_judge.p0b.support.forced_choice"
    )
    forced = result["judgement"]["budget_exhaustion_forced_choice"]
    assert forced["applied"] is True
    assert forced["available_image_count"] == 0
    assert "degraded_evidence" not in json.dumps(result)
    assert "support_zero_local_evidence" in json.dumps(
        model.calls[0]["messages"]
    )


def test_support_one_local_raw_view_uses_normal_binary_judge(
    tmp_path: Path,
) -> None:
    local = tmp_path / "support-local.png"
    global_top = tmp_path / "support-global-top.png"
    for path, color in (
        (local, (30, 80, 120)),
        (global_top, (100, 70, 40)),
    ):
        rendered = Image.new("RGB", (8, 8), color)
        rendered.putpixel((0, 0), (220, 180, 60))
        rendered.save(path)

    model = _FakeModel(
        {
            "verdict": "invalid",
            "confidence": 0.8,
            "reason": "the available local view establishes a support gap",
        }
    )

    result = adjudicate_p0b_event(
        metric="support",
        event={"object_id": "bed"},
        prompt="Place the bed in the room.",
        relationships=[],
        scene=_scene(),
        detector_evidence={"gap_m": 0.1},
        judge=OpenAICompatibleVLMJudge(model),
        object_ids=["bed"],
        local_view_provider=lambda _request: [
            {
                "path": str(global_top),
                "role": "metric_highlighted_global",
                "view_id": "global_top",
            },
            {
                "path": str(local),
                "role": "metric_local_rgb",
                "view_id": "local_1",
            },
        ],
    )

    assert result["verdict"] == "invalid"
    assert len(model.calls) == 1
    assert (
        model.calls[0]["kwargs"]["call_type"]
        == "vlm_judge.p0b.support"
    )
    assert (
        result["visual_evidence_policy"]["config_id"]
        == "support_local_raw_global_top_budget2_v3"
    )
    assert (
        result["judgement"]["budget_exhaustion_forced_choice"]["applied"]
        is False
    )


def test_support_zero_local_uses_available_global_view_when_forced(
    tmp_path: Path,
) -> None:
    global_top = tmp_path / "support-global-top.png"
    rendered = Image.new("RGB", (8, 8), (80, 60, 40))
    rendered.putpixel((0, 0), (210, 170, 90))
    rendered.save(global_top)
    model = _FakeModel(
        {
            "verdict": "valid",
            "confidence": 0.6,
            "reason": "the remaining evidence supports a binary conclusion",
        }
    )

    result = adjudicate_p0b_event(
        metric="support",
        event={"object_id": "bed"},
        prompt="",
        relationships=[],
        scene=_scene(),
        detector_evidence={"gap_m": 0.0},
        judge=OpenAICompatibleVLMJudge(model),
        object_ids=["bed"],
        local_view_provider=lambda _request: [
            {
                "path": str(global_top),
                "role": "metric_highlighted_global",
                "view_id": "global_top",
            }
        ],
    )

    assert result["verdict"] == "valid"
    forced = result["judgement"]["budget_exhaustion_forced_choice"]
    assert forced["applied"] is True
    assert forced["available_image_count"] == 1
    assert len(model.calls[0]["messages"][1]["content"]) == 2


def test_openai_p0b_judge_requires_binary_verdict(tmp_path: Path) -> None:
    image = tmp_path / "local.png"
    rendered = Image.new("RGB", (4, 4), (64, 64, 64))
    rendered.putpixel((0, 0), (128, 64, 32))
    rendered.save(image)
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
    assert "need_more_evidence" in model.calls[0]["messages"][0]["content"]
    assert "status" not in result
    assert "evidence_request" not in result

    invalid_model = _FakeModel({"verdict": "insufficient_evidence", "confidence": 0.2, "reason": "occluded"})
    with pytest.raises(EvidenceControlUnresolvedError):
        OpenAICompatibleVLMJudge(invalid_model).adjudicate_p0b(
            {
                "metric": "support",
                "render_evidence": [],
            }
        )
    assert invalid_model.calls == []


def test_p0b_event_raw_openai_public_path_gates_empty_evidence_before_model():
    model = _FakeModel(
        {
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "must not run",
        }
    )

    with pytest.raises(EvidenceControlUnresolvedError):
        adjudicate_p0b_event(
            metric="collision",
            event={"object_a": "bed", "object_b": "cabinet"},
            prompt="Put the cabinet beside the bed.",
            relationships=[],
            scene=_scene(),
            detector_evidence={"overlap": 0.01},
            judge=OpenAICompatibleVLMJudge(model),
            object_ids=["bed", "cabinet"],
            overview_render_evidence=[],
            local_view_provider=None,
        )

    assert model.calls == []


def test_p0b_event_gates_unmarked_judge_without_provider():
    class _UnmarkedJudge:
        def __init__(self):
            self.calls = 0

        def adjudicate_p0b(self, request):
            del request
            self.calls += 1
            return {
                "verdict": "valid",
                "confidence": 1.0,
                "reason": "must not run",
            }

    judge = _UnmarkedJudge()
    with pytest.raises(EvidenceControlUnresolvedError):
        adjudicate_p0b_event(
            metric="collision",
            event={"object_a": "bed", "object_b": "cabinet"},
            prompt="Put the cabinet beside the bed.",
            relationships=[],
            scene=_scene(),
            detector_evidence={"overlap": 0.01},
            judge=judge,
            object_ids=["bed", "cabinet"],
            overview_render_evidence=[],
            local_view_provider=None,
        )

    assert judge.calls == 0


def test_p0b_event_raw_openai_path_does_not_repair_from_gate_metadata(
    tmp_path: Path,
):
    global_view = tmp_path / "global.png"
    local_bad = tmp_path / "local-bad.png"
    local_ready = tmp_path / "local-ready.png"
    for path, color in (
        (global_view, (20, 20, 20)),
        (local_bad, (40, 40, 40)),
        (local_ready, (80, 80, 80)),
    ):
        rendered = Image.new("RGB", (8, 8), color)
        rendered.putpixel((0, 0), (200, 100, 50))
        rendered.save(path)

    provider_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        if len(provider_calls) == 1:
            return [
                {
                    "path": str(local_bad),
                    "role": "metric_local_highlight",
                    "view_id": "local-oob",
                    "target_ids": ["chair-1"],
                    "target_visible": False,
                    "visibility": {
                        "target_visible": False,
                        "target_pixel_fractions": {"chair-1": 0.0},
                        "region_pixel_fractions": {
                            "architecture_plane": 0.2,
                        },
                    },
                }
            ]
        return [
            {
                "path": str(local_ready),
                "role": "metric_local_highlight",
                "view_id": "local-oob",
                "target_ids": ["chair-1"],
                "visibility": {
                    "target_pixel_fractions": {"chair-1": 0.02},
                    "region_pixel_fractions": {
                        "architecture_plane": 0.2,
                    },
                },
            }
        ]

    model = _FakeModel(
        {
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "boundary condition is visible",
        }
    )

    result = adjudicate_p0b_event(
        metric="oob",
        event={"object_id": "chair-1"},
        prompt="Place a chair near the wall.",
        relationships=[],
        scene=_scene(),
        detector_evidence={"flagged_planes": ["x_max"]},
        judge=OpenAICompatibleVLMJudge(model),
        object_ids=["chair-1"],
        overview_render_evidence=[str(global_view)],
        local_view_provider=provider,
        visual_config_policy="passthrough",
    )

    assert result["verdict"] == "valid"
    # Visibility metadata is not a Gate-owned sufficiency signal. The Judge
    # concluded on the initial packet, so no second acquisition is justified.
    assert len(provider_calls) == 1
    assert len(model.calls) == 1


def test_support_rubric_requires_grounded_ancestry_for_local_contact() -> None:
    calls: list[dict] = []

    @_non_vlm_stub
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


def test_oob_rubric_carries_no_invalid_prior_and_separates_floor_tolerance() -> None:
    from benchmark.visual_judge.p0b import P0B_METRIC_RUBRICS

    rubric = P0B_METRIC_RUBRICS["oob"]
    assert "authoritative facts" in rubric
    # The asymmetric contract: routing is a request for adjudication, not an
    # automatic invalid prior, and a measured crossing alone is insufficient.
    assert "carries no " in rubric
    assert "not by itself sufficient to return invalid" in rubric
    assert "Return invalid only when" in rubric
    assert "Return invalid by default" not in rubric
    # numerical robustness stays separate from the semantic floor-contact tolerance.
    assert "numerical_eps" in rubric
    assert "floor_contact_tolerance_m" in rubric
    assert "against_wall" in rubric
    assert "never exempt ordinary furniture" in rubric


def test_oob_uses_fixed_global_before_deterministic_local(tmp_path: Path) -> None:
    global_top = tmp_path / "standardized_top.png"
    global_perspective = tmp_path / "standardized_perspective.png"
    local = tmp_path / "oob_plane_normal.png"
    for path in (global_top, global_perspective, local):
        path.write_bytes(b"png")

    calls: list[dict] = []

    @_non_vlm_stub
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
