from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PIL import Image

from benchmark.api.evaluation import run_evaluate
from benchmark.visual_judge.adapters.legacy_renderer import (
    CameraViewEvidenceRenderer,
)
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionResult,
)
from benchmark.visual_judge.interfaces.evidence import (
    EvidenceRenderRequest,
)
from benchmark.visual_judge.interfaces.judge import JudgeRequest
from benchmark.visual_judge.orchestration.camera_acquisition import (
    render_duration,
)


def _image(path: Path, color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (24, 24), color)
    image.putpixel(
        (0, 0),
        tuple(min(255, channel + 40) for channel in color),
    )
    image.save(path)
    return str(path)


def _scene() -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "l3_camera_scene",
        "request_id": "l3_camera_request",
        "scene_type": "home office",
        "boundary": [[0, 0], [5, 0], [5, 4], [0, 4]],
        "scene_height": 2.8,
        "objects": [
            {
                "id": "desk",
                "category": "desk",
                "description": "wood desk",
                "size": [1.4, 0.7, 0.75],
                "center": [2.0, 2.0, 0.375],
                "rotation": [0, 0, 0],
                "metadata": {},
            },
            {
                "id": "chair",
                "category": "chair",
                "description": "office chair",
                "size": [0.6, 0.6, 1.0],
                "center": [2.0, 1.1, 0.5],
                "rotation": [0, 0, 0],
                "metadata": {},
            },
        ],
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


def _grouping() -> dict:
    return {
        "status": "complete",
        "grouping_backend": "vlm",
        "grouping_policy_id": "vlm_visual_evidence_scope_v2",
        "object_groups": [
            {
                "group_id": "group_001",
                "object_ids": ["desk", "chair"],
                "label": "workstation",
            }
        ],
    }


class _Judge:
    vlm_control_enabled = True

    def __init__(self) -> None:
        self.scene_quality_requests: list[dict] = []

    def adjudicate_scene_quality(self, request: dict) -> dict:
        self.scene_quality_requests.append(deepcopy(request))
        if len(self.scene_quality_requests) == 1:
            return {
                "evidence_status": "insufficient",
                "verdict": "ambiguous",
                "confidence": 0.2,
                "reason": "The group context is not yet observable.",
                "missing_evidence": [],
                "defects": [],
                "evidence_request": {
                    "target_ids": ["desk", "chair"],
                    "missing_observations": ["group_context_visible"],
                    "view_goal": "show the workstation as one local context",
                    "metadata": {},
                },
            }
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.91,
            "reason": "The group scale is visually coherent.",
            "missing_evidence": [],
            "defects": [],
        }

    def adjudicate_p0b(self, request: dict) -> dict:
        del request
        return {
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "deterministic event accepted",
        }

    def adjudicate_relation(self, request: dict) -> dict:
        del request
        return {
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "relation accepted",
        }


class _NoFeasibleDeterministicSelector:
    backend = "deterministic_test"

    def __init__(self) -> None:
        self.requests = []

    def select(self, request):
        self.requests.append(request)
        return {
            "outcome": "no_feasible_candidate",
            "attempted_candidate_ids": [],
            "rejected_candidates": [],
            "reason_codes": [
                "observation_not_supported_by_deterministic_selector"
            ],
            "reason": "group_context_visible is not deterministically verified",
            "provenance": {"stage": "deterministic"},
        }


class _VLMSelector:
    backend = "vlm_test"
    validated_internal_candidate_bank = True

    def __init__(self) -> None:
        self.requests = []

    def select(self, request):
        self.requests.append(request)
        pose = {
            "id": "vlm_group_view",
            "location": [2.0, -2.0, 2.1],
            "target": [2.0, 1.6, 0.6],
            "lens_mm": 45.0,
            "sensor_width_mm": 36.0,
            "geometry_feasible": True,
            "geometry_feasibility_verified": True,
            "target_visibility_estimate": True,
            "joint_visibility_estimate": True,
            "projected_coverage_estimate": 0.3,
            "target_object_ids": ["desk", "chair"],
        }
        return CameraSelectionResult(
            outcome="selected",
            selected_view_ids=("vlm_group_view",),
            selected_views=(pose,),
            reason_codes=("vlm_group_repair_selected",),
            reason="selected one trusted group-local pose",
            backend=self.backend,
            evidence_round=request.evidence_round,
            provenance={"stage": "vlm"},
        )


class _ExistingRenderer:
    def __init__(self, *, report_gpu_time: bool = True) -> None:
        self.calls: list[dict] = []
        self.report_gpu_time = bool(report_gpu_time)

    def render_camera_views(
        self,
        *,
        blend_file,
        out_dir,
        camera_views,
        preview=False,
    ):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append(
            {
                "blend_file": str(blend_file),
                "camera_views": deepcopy(camera_views),
                "preview": preview,
            }
        )
        views = []
        for pose in camera_views:
            path = destination / f"{pose['id']}.png"
            _image(path, (65, 125, 185))
            views.append(
                {
                    "id": pose["id"],
                    "path": str(path),
                    "pose": deepcopy(pose),
                }
            )
        result = {"views": views}
        if self.report_gpu_time:
            result["render_gpu_time_seconds"] = 0.01
        return result


def test_camera_renderer_omits_unreported_gpu_time_provenance(
    tmp_path: Path,
) -> None:
    blend_path = tmp_path / "scene.blend"
    blend_path.write_bytes(b"blend")
    renderer = CameraViewEvidenceRenderer(
        renderer=_ExistingRenderer(report_gpu_time=False),
        blend_file=blend_path,
        out_dir=tmp_path / "controller_renders",
    )
    pose = {
        "id": "group_view",
        "location": [1.0, 1.0, 1.0],
        "target": [0.0, 0.0, 0.0],
        "lens_mm": 45.0,
    }
    rendered = renderer.render(
        EvidenceRenderRequest(
            judge_request=JudgeRequest(
                task="l3_scene_quality",
                metric="functional_consistency",
                claim_or_event={},
                scene_context={},
                deterministic_evidence={},
                visual_evidence=(),
                rubric={},
                context={"target_object_ids": ["a", "b"]},
            ),
            selection=CameraSelectionResult(
                selected_view_ids=("group_view",),
                selected_views=(pose,),
                backend="deterministic",
                evidence_round=1,
            ),
            evidence_goal={"target_ids": ["a", "b"]},
            previous_visual_evidence=(),
            evidence_round=1,
            budget={"max_views_per_round": 1},
            context={
                "group_scope": {
                    "group_id": "group_001",
                    "member_ids": ["a", "b"],
                }
            },
        )
    )

    assert "render_gpu_time_seconds" not in rendered.provenance
    assert rendered.provenance["render_gpu_time_source"] == "not_reported"
    assert render_duration(rendered.provenance) == 0.0


def test_canonical_l3_group_judge_repair_reaches_vlm_and_renders(
    tmp_path: Path,
) -> None:
    global_path = _image(tmp_path / "global.png", (30, 60, 90))
    initial_local_path = _image(
        tmp_path / "initial_group.png",
        (90, 60, 30),
    )
    blend_path = tmp_path / "scene.blend"
    blend_path.write_bytes(b"blend")
    renderer_backend = _ExistingRenderer()
    renderer = CameraViewEvidenceRenderer(
        renderer=renderer_backend,
        blend_file=blend_path,
        out_dir=tmp_path / "controller_renders",
    )
    deterministic = _NoFeasibleDeterministicSelector()
    vlm = _VLMSelector()
    judge = _Judge()
    initial_provider_requests: list[dict] = []

    def initial_provider(request: dict) -> list[dict]:
        initial_provider_requests.append(deepcopy(request))
        return [
            {
                "path": initial_local_path,
                "role": "group_local",
                "group_id": request["group_scope"]["group_id"],
                "member_ids": list(request["object_ids"]),
            }
        ]

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "evaluation_report.json",
        scene_request={
            "request_id": "l3_camera_request",
            "instruction": "Create a coherent home office.",
            "scene_type": "home office",
        },
        render_evidence=[global_path],
        object_grouping_report=_grouping(),
        l3_initial_evidence_provider=initial_provider,
        deterministic_camera_selector=deterministic,
        vlm_camera_selector=vlm,
        evidence_renderer=renderer,
        vlm_judge=judge,
        asset_policy={
            "mode": "generated_or_open_assets",
            "identity_owner": "generator",
            "category_selection_owner": "generator",
            "scale_owner": "generator",
            "appearance_owner": "generator",
            "arrangement_owner": "generator",
        },
        scene_quality_config={
            "metrics": {
                "scale_consistency": {
                    "enabled": True,
                    # This test isolates the post-Judge Controller repair
                    # cascade; JSON-first routing is covered separately.
                    "evidence_plan": {
                        "evidence_strategy": "global_and_local",
                        "router_options": None,
                    },
                },
                "object_pairing_consistency": {"enabled": False},
                "style_consistency": {"enabled": False},
                "functional_consistency": {"enabled": False},
                "semantic_placement_consistency": {"enabled": False},
            }
        },
        vlm_evaluation_control={
            "camera_acquisition": {
                "policy": "deterministic_then_vlm",
                "total": {
                    "max_evidence_rounds": 2,
                    "max_total_images": 6,
                    "max_selector_calls": 2,
                    "max_camera_actions": 2,
                },
            }
        },
    )

    metric = report["reports"]["scene_quality"]["metrics"][
        "scale_consistency"
    ]
    assert metric["judgement"]["verdict"] == "valid"
    assert len(judge.scene_quality_requests) == 2
    assert len(initial_provider_requests) == 1
    assert len(deterministic.requests) == 1
    assert len(vlm.requests) == 1
    assert len(renderer_backend.calls) == 1
    assert renderer_backend.calls[0]["preview"] is False
    assert deterministic.requests[0].context["group_scope"]["group_id"] == (
        "group_001"
    )
    assert vlm.requests[0].target_ids == ("desk", "chair")
    final_paths = judge.scene_quality_requests[-1]["render_evidence"]
    assert any("vlm_group_view.png" in path for path in final_paths)

    controlled = report["evaluation_config"]["vlm_evaluation_control"][
        "integration"
    ]["runtime"]["controlled_calls"]
    scale_call = next(
        item for item in controlled if item["metric"] == "scale_consistency"
    )
    stages = [entry["stage"] for entry in scale_call["audit"]["trace"]]
    assert stages == [
        "evidence_gate",
        "judge",
        "acquisition_planner",
        "camera_selector",
        "camera_escalation",
        "camera_selector",
        "render",
        "evidence_gate",
        "judge",
    ]
