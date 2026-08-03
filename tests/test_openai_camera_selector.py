from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from benchmark.visual_judge import (
    ActiveVLMCameraSelector,
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
