from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from benchmark.visual_judge import (
    ActiveVLMCameraSelector,
    CAMERA_SELECTOR_PROMPT_VERSION,
    CameraCandidatePreviewRenderer,
    OpenAICompatibleCameraSelector,
)
from benchmark.visual_judge.adapters.deterministic_camera import (
    DeterministicCameraRepairSolver,
)
from benchmark.visual_judge.camera_dsl import CameraConstraintSet
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionRequest,
)


class _Model:
    model_id = "camera-model"
    endpoint = "https://example.test/v1"
    response_format_json = True

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []
        self.last_request_metadata: dict = {}

    def chat_messages(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return json.dumps(self.response)


def _candidate(path: Path) -> dict:
    image = Image.new("RGB", (16, 16), (20, 30, 40))
    image.putpixel((0, 0), (220, 80, 30))
    image.save(path)
    return {
        "id": "trusted-view",
        "location": [2.0, 1.0, 2.0],
        "target": [0.0, 0.0, 0.5],
        "lens_mm": 50.0,
        "pose": {
            "location": [2.0, 1.0, 2.0],
            "target": [0.0, 0.0, 0.5],
            "lens_mm": 50.0,
            "camera_type": "PERSP",
        },
        "target_ids": ["a", "b"],
        "group_id": "group_001",
        "technical_feasibility": True,
        "target_visibility_estimate": True,
        "joint_visibility_estimate": True,
        "projected_coverage_estimate": 0.2,
        "view_family": "oblique",
        "image_path": str(path),
        "render_status": "ok",
        "preview_metadata": {"quality": "preview"},
    }


def _request(
    *,
    candidates=(),
    mode: str,
    plans=(),
) -> CameraSelectionRequest:
    constraints = CameraConstraintSet(
        target_ids=("a", "b"),
        required_observations=(
            "interaction_side_visible"
            if mode == "candidate_only"
            else "joint_visibility",
        ),
        relaxable_constraints=(
            ("joint_visibility",)
            if mode == "repair_plan"
            else ()
        ),
        metric="functional_consistency",
        view_goal="show the local group",
    )
    return CameraSelectionRequest(
        task="functional_consistency",
        metric="functional_consistency",
        target_ids=("a", "b"),
        scene={"objects": [{"id": "a"}, {"id": "b"}]},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={
            "max_views_per_round": 1,
            "candidate_budget": 8,
        },
        constraints=constraints.to_dict(),
        candidate_views=tuple(candidates),
        context={
            "known_target_ids": ["a", "b"],
            "vlm_selection_mode": mode,
            "camera_repair_plans": list(plans),
            "group_scope": {
                "group_id": "group_001",
                "member_ids": ["a", "b"],
            },
        },
    )


def test_production_candidate_only_transport_and_active_contract(
    tmp_path: Path,
) -> None:
    model = _Model(
        {
            "selected_view_ids": ["trusted-view"],
            "reason": "best interaction-side preview",
        }
    )
    transport = OpenAICompatibleCameraSelector(model)
    active = ActiveVLMCameraSelector(
        transport,
        selection_mode="repair_plan",
        repair_solver=DeterministicCameraRepairSolver(),
    )
    candidate = _candidate(tmp_path / "preview.png")

    result = active.select(
        _request(candidates=(candidate,), mode="candidate_only")
    )

    assert result.selected_view_ids == ("trusted-view",)
    assert result.selected_views == (candidate,)
    assert result.provenance["selection_mode"] == "candidate_only"
    assert transport.last_request_metadata["preview_aliases"] == [
        "trusted-view"
    ]
    assert transport.last_request_metadata["prompt_version"] == (
        CAMERA_SELECTOR_PROMPT_VERSION
    )
    assert transport.last_request_metadata["vlm_role"] == (
        "vlm_camera_selector"
    )
    assert transport.last_request_metadata["decision_contract"] == (
        "camera_selection_v1"
    )
    system_prompt = model.calls[0]["messages"][0]["content"]
    assert "Select the smallest number of views" in system_prompt
    assert "candidate_views order" in system_prompt
    assert "Metadata values marked null" in system_prompt
    assert "nearest authoritative logical boundary" in system_prompt
    assert "do not require or\ninvent a wall" in system_prompt
    user_content = model.calls[0]["messages"][1]["content"]
    assert any(item.get("type") == "image_url" for item in user_content)


def test_legacy_query_cov_compatibility_does_not_construct_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from benchmark.visual_judge.openai_compatible import (
        OpenAICompatibleVLMJudge,
    )

    def fail_judge_construction(*args, **kwargs):
        raise AssertionError(
            "camera transport must not construct a metric Judge"
        )

    monkeypatch.setattr(
        OpenAICompatibleVLMJudge,
        "__init__",
        fail_judge_construction,
    )
    transport = OpenAICompatibleCameraSelector(
        _Model(
            {
                "selected_view_ids": ["candidate_00"],
                "action": None,
                "reason": "bounded trusted candidate",
            }
        )
    )
    candidate = _candidate(tmp_path / "legacy-preview.png")

    result = transport.select_camera_views(
        {
            "metric": "collision",
            "candidates": [candidate],
            "max_views": 1,
            "allow_adjustment": False,
            "allowed_actions": [],
            "preview_role": "highlighted_focus",
        }
    )

    assert result["selected_view_ids"] == ["trusted-view"]
    assert result["vlm_role"] == "vlm_camera_selector"
    assert transport.last_request_metadata["selection_mode"] == (
        "legacy_query_cov"
    )


def test_legacy_query_cov_accepts_functional_probe_metric(
    tmp_path: Path,
) -> None:
    model = _Model(
        {
            "selected_view_ids": ["candidate_00"],
            "action": None,
            "reason": "usable side and outward context are visible",
        }
    )
    transport = OpenAICompatibleCameraSelector(model)
    candidate = _candidate(tmp_path / "functional-preview.png")

    result = transport.select_camera_views(
        {
            "metric": "functional_consistency",
            "candidates": [candidate],
            "max_views": 1,
            "allow_adjustment": False,
            "allowed_actions": [],
            "preview_role": "rgb",
            "functional_probe": {
                "probe_id": "functional_probe_01",
                "kind": "functional_frontage",
                "target_ids": ["a"],
                "related_target_ids": [],
                "required_observations": [
                    "interaction_side_visible",
                    "approach_zone_visible",
                ],
                "view_goal": "show the usable face and outward context",
            },
        }
    )

    assert result["selected_view_ids"] == ["trusted-view"]
    outbound = model.calls[0]["messages"][1]["content"]
    context = json.loads(outbound[0]["text"].split("\n", 1)[1])
    assert context["metric_family"] == "functional_consistency"
    assert context["functional_probe"]["kind"] == (
        "functional_frontage"
    )
    legacy_prompt = model.calls[0]["messages"][0]["content"]
    assert "nearest authoritative logical boundary" in legacy_prompt
    assert "do not require or invent a" in legacy_prompt


def test_functional_probe_planner_uses_minimal_objects_and_no_verdict(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    Image.new("RGB", (32, 32), (120, 130, 140)).save(
        global_image
    )
    model = _Model(
        {
            "probe_units": [
                {
                    "kind": "functional_correspondence",
                    "target_ids": ["sofa"],
                    "related_target_ids": ["television"],
                    "required_observations": [
                        "target_visible",
                        "joint_visibility",
                        "interaction_side_visible",
                        "front_back_disambiguated",
                        "approach_zone_visible",
                        "group_context_visible",
                        "limited_local_context",
                    ],
                    "priority": 1,
                    "reason": "show both usable sides together",
                }
            ],
            "reason": "bounded visual evidence coverage",
        }
    )
    selector = OpenAICompatibleCameraSelector(model)

    result = selector.plan_functional_evidence(
        {
            "metric": "functional_consistency",
            "scene_id": "scene",
            "scene_type": "living_room",
            "global_image_path": str(global_image),
            "architecture_context": {
                "source": "scene_architecture_contract",
                "logical_boundary_enabled": True,
                "logical_boundary_xy": [
                    [0.0, 0.0],
                    [5.0, 0.0],
                    [5.0, 4.0],
                    [0.0, 4.0],
                ],
                "physical_walls_rendered": False,
                "physical_wall_ids": [],
            },
            "objects": [
                {"id": "sofa", "category": "sofa"},
                {"id": "television", "category": "television"},
            ],
            "max_probe_units": 4,
        }
    )

    assert result["schema_version"] == "functional_probe_plan_v2"
    assert result["planner_role"] == (
        "visual_evidence_only_no_metric_verdict"
    )
    assert result["probe_units"][0]["probe_id"] == (
        "functional_probe_01"
    )
    assert "verdict" not in result
    assert model.calls[0]["kwargs"]["call_type"] == (
        "vlm_camera_pose.functional_probe_plan"
    )
    assert model.calls[0]["kwargs"]["max_tokens"] == 3072
    assert model.calls[0]["kwargs"]["max_tokens_source"] == (
        "functional_probe_planner_minimum"
    )
    content = model.calls[0]["messages"][1]["content"]
    text = next(
        item["text"] for item in content if item["type"] == "text"
    )
    assert '"id":"sofa"' in text
    assert '"category":"sofa"' in text
    assert "rotation" not in text
    assert any(item["type"] == "image_url" for item in content)
    planner_prompt = model.calls[0]["messages"][0]["content"]
    assert "Do not fill a quota" in planner_prompt
    assert "unique contiguous priorities" in planner_prompt
    assert "physical distance\n   is not a disqualifier" in planner_prompt
    assert "different local groups" in planner_prompt
    assert "Distance must affect camera framing" in planner_prompt
    assert "close enough that one wider local view" not in planner_prompt
    assert "Never merge spatially distant objects" not in planner_prompt
    assert "logical_boundary_xy" in planner_prompt
    assert "missing physical wall mesh" in planner_prompt
    assert "Use these exact required_observations templates" in (
        planner_prompt
    )
    outbound_context = json.loads(text.split("\n", 1)[1])
    assert outbound_context["vlm_role"] == "functional_evidence_planner"
    assert (
        outbound_context["decision_contract"]
        == "functional_probe_plan_v2"
    )
    assert outbound_context["architecture_context"] == {
        "source": "scene_architecture_contract",
        "logical_boundary_enabled": True,
        "logical_boundary_xy": [
            [0.0, 0.0],
            [5.0, 0.0],
            [5.0, 4.0],
            [0.0, 4.0],
        ],
        "physical_walls_rendered": False,
        "physical_wall_ids": [],
    }
    assert result["request_metadata"][
        "functional_probe_prompt_version"
    ] == "functional_probe_planner_v4"
    assert result["request_metadata"]["vlm_role"] == (
        "functional_evidence_planner"
    )
    assert result["request_metadata"]["decision_contract"] == (
        "functional_probe_plan_v2"
    )
    assert selector.last_request_metadata["selection_mode"] == (
        "functional_probe_plan"
    )


def test_functional_probe_planner_rejects_metric_decision_authority(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    Image.new("RGB", (16, 16), (20, 30, 40)).save(global_image)
    selector = OpenAICompatibleCameraSelector(
        _Model(
            {
                "probe_units": [],
                "reason": "also judged the scene",
                "verdict": "invalid",
            }
        )
    )

    with pytest.raises(ValueError, match="may not return"):
        selector.plan_functional_evidence(
            {
                "metric": "functional_consistency",
                "scene_id": "scene",
                "scene_type": "living_room",
                "global_image_path": str(global_image),
                "objects": [
                    {"id": "sofa", "category": "sofa"}
                ],
                "max_probe_units": 4,
            }
        )


def test_functional_probe_planner_may_return_no_probe_units(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global-clear.png"
    Image.new("RGB", (16, 16), (20, 30, 40)).save(global_image)
    selector = OpenAICompatibleCameraSelector(
        _Model(
            {
                "probe_units": [],
                "reason": (
                    "The global image already establishes the required "
                    "functional observations."
                ),
            }
        )
    )

    result = selector.plan_functional_evidence(
        {
            "metric": "functional_consistency",
            "scene_id": "scene",
            "scene_type": "living_room",
            "global_image_path": str(global_image),
            "objects": [{"id": "sofa", "category": "sofa"}],
            "max_probe_units": 4,
        }
    )

    assert result["schema_version"] == "functional_probe_plan_v2"
    assert result["probe_units"] == []


def test_functional_probe_planner_accepts_boundary_aware_frontage(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global-boundary.png"
    Image.new("RGB", (16, 16), (20, 30, 40)).save(global_image)
    selector = OpenAICompatibleCameraSelector(
        _Model(
            {
                "probe_units": [
                    {
                        "kind": "functional_frontage",
                        "target_ids": ["toilet"],
                        "related_target_ids": [],
                        "required_observations": [
                            "target_visible",
                            "interaction_side_visible",
                            "front_back_disambiguated",
                            "approach_zone_visible",
                            "architecture_plane_visible",
                            "global_context_preserved",
                            "limited_local_context",
                        ],
                        "priority": 1,
                        "reason": (
                            "show the usable side, logical boundary, and "
                            "interior approach region together"
                        ),
                    }
                ],
                "reason": "boundary relation is not established",
            }
        )
    )

    result = selector.plan_functional_evidence(
        {
            "metric": "functional_consistency",
            "scene_id": "scene",
            "scene_type": "bathroom",
            "global_image_path": str(global_image),
            "architecture_context": {
                "source": "scene_architecture_contract",
                "logical_boundary_enabled": True,
                "logical_boundary_xy": [
                    [0.0, 0.0],
                    [3.5, 0.0],
                    [3.5, 3.0],
                    [0.0, 3.0],
                ],
                "physical_walls_rendered": False,
                "physical_wall_ids": [],
            },
            "objects": [{"id": "toilet", "category": "toilet"}],
            "max_probe_units": 4,
        }
    )

    observations = result["probe_units"][0][
        "required_observations"
    ]
    assert "architecture_plane_visible" in observations
    assert "global_context_preserved" in observations


def test_functional_probe_planner_rejects_invalid_enabled_boundary(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global-invalid-boundary.png"
    Image.new("RGB", (16, 16), (20, 30, 40)).save(global_image)
    selector = OpenAICompatibleCameraSelector(
        _Model({"probe_units": [], "reason": "unused"})
    )

    with pytest.raises(ValueError, match="at least three XY points"):
        selector.plan_functional_evidence(
            {
                "metric": "functional_consistency",
                "scene_id": "scene",
                "scene_type": "bathroom",
                "global_image_path": str(global_image),
                "architecture_context": {
                    "source": "scene_architecture_contract",
                    "logical_boundary_enabled": True,
                    "logical_boundary_xy": [[0.0, 0.0], [1.0, 0.0]],
                    "physical_walls_rendered": False,
                    "physical_wall_ids": [],
                },
                "objects": [{"id": "toilet", "category": "toilet"}],
                "max_probe_units": 4,
            }
        )


def test_production_repair_plan_transport_contract() -> None:
    plan = {
        "plan_id": "relax_joint",
        "objective": "relax joint visibility",
        "preserved_constraints": [],
        "relaxed_constraints": ["joint_visibility"],
        "preferred_view_families": [],
        "required_view_count": 1,
        "estimated_cost": {},
        "provenance": {},
    }
    model = _Model(
        {
            "selected_plan_id": "relax_joint",
            "reason": "the trusted bounded relaxation resolves the conflict",
        }
    )
    transport = OpenAICompatibleCameraSelector(model)

    raw = transport.select(
        {
            **_request(
                mode="repair_plan",
                plans=(plan,),
            ).to_dict(),
            "selection_mode": "repair_plan",
            "trusted_repair_plans": [plan],
        }
    )

    assert raw["selected_plan_id"] == "relax_joint"
    assert not any(
        item.get("type") == "image_url"
        for item in model.calls[0]["messages"][1]["content"]
    )


def test_production_candidate_only_rejects_untrusted_model_id(
    tmp_path: Path,
) -> None:
    model = _Model(
        {
            "selected_view_ids": ["invented"],
            "reason": "invented view",
        }
    )
    selector = OpenAICompatibleCameraSelector(model)
    request = _request(
        candidates=(_candidate(tmp_path / "preview.png"),),
        mode="candidate_only",
    )

    with pytest.raises(ValueError, match="untrusted candidate"):
        selector.select(
            {
                **request.to_dict(),
                "selection_mode": "candidate_only",
            }
        )


@pytest.mark.parametrize(
    "forbidden_response",
    [
        {
            "selected_view_ids": ["trusted-view"],
            "reason": "also judged it",
            "verdict": "valid",
        },
        {
            "selected_view_ids": ["trusted-view"],
            "reason": "also scored it",
            "score": 1.0,
        },
        {
            "selected_view_ids": ["trusted-view"],
            "reason": "also changed membership",
            "member_ids": ["a"],
        },
    ],
)
def test_production_selector_rejects_decision_or_group_authority(
    tmp_path: Path,
    forbidden_response: dict,
) -> None:
    selector = OpenAICompatibleCameraSelector(
        _Model(forbidden_response)
    )
    request = _request(
        candidates=(_candidate(tmp_path / "preview.png"),),
        mode="candidate_only",
    )

    with pytest.raises(
        ValueError,
        match="forbidden or unknown fields",
    ):
        selector.select(
            {
                **request.to_dict(),
                "selection_mode": "candidate_only",
            }
        )


@pytest.mark.parametrize(
    ("preview_kind", "message"),
    [
        ("blank", "blank"),
        ("corrupt", "corrupt or undecodable"),
    ],
)
def test_candidate_preview_renderer_rejects_unusable_images(
    tmp_path: Path,
    preview_kind: str,
    message: str,
) -> None:
    class _PreviewProcess:
        def render_camera_views(
            self,
            *,
            blend_file,
            out_dir,
            camera_views,
            preview=False,
        ):
            del blend_file
            assert preview is True
            destination = Path(out_dir)
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / "candidate.png"
            if preview_kind == "blank":
                Image.new("RGB", (16, 16), (40, 40, 40)).save(path)
            else:
                path.write_bytes(b"not an image")
            return {
                "views": [
                    {
                        "id": camera_views[0]["id"],
                        "path": str(path),
                    }
                ]
            }

    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"stub")
    candidate = _candidate(tmp_path / "seed.png")
    candidate.pop("image_path")
    candidate.pop("render_status")
    renderer = CameraCandidatePreviewRenderer(
        renderer=_PreviewProcess(),
        blend_file=blend,
        out_dir=tmp_path / "previews",
    )

    with pytest.raises(ValueError, match=message):
        renderer.render(
            _request(
                candidates=(candidate,),
                mode="candidate_only",
            )
        )
