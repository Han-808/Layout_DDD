from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PIL import Image

from benchmark.visual_judge import (
    CameraCandidatePreviewRenderer,
    CameraViewEvidenceRenderer,
    DeterministicLocalCameraSelector,
    OpenAICompatibleCameraSelector,
    OpenAICompatibleVLMJudge,
    VLMEvaluationController,
    resolve_vlm_evaluation_control,
)
from benchmark.visual_judge.adapters.legacy_judge import (
    ExistingJudgeAdapter,
)
from benchmark.visual_judge.interfaces.judge import JudgeRequest


class _SequentialJudgeModel:
    model_id = "judge-model"
    endpoint = "https://example.test/v1"
    response_format_json = True

    def __init__(self) -> None:
        self.calls = 0
        self.last_request_metadata = {}

    def chat_messages(self, messages, **kwargs):
        del messages, kwargs
        self.calls += 1
        self.last_request_metadata = {"call": self.calls}
        if self.calls == 1:
            return json.dumps(
                {
                    "evidence_status": "insufficient",
                    "verdict": "ambiguous",
                    "confidence": 0.2,
                    "reason": "Need a view showing group context.",
                    "missing_evidence": [],
                    "defects": [],
                    "evidence_request": {
                        "target_ids": ["a", "b"],
                        "missing_observations": [
                            "group_context_visible"
                        ],
                        "view_goal": (
                            "show both objects and their bounded local context"
                        ),
                        "metadata": {},
                    },
                }
            )
        return json.dumps(
            {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 0.9,
                "reason": "The selected final view resolves the group.",
                "missing_evidence": [],
                "defects": [],
                "evidence_request": None,
            }
        )


class _CandidateSelectorModel:
    model_id = "selector-model"
    endpoint = "https://example.test/v1"
    response_format_json = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat_messages(self, messages, **kwargs):
        self.calls.append({"messages": deepcopy(messages), "kwargs": kwargs})
        text = messages[1]["content"][0]["text"]
        payload = json.loads(text.split("\n", 1)[1])
        return json.dumps(
            {
                "selected_view_ids": [
                    payload["candidate_views"][0]["id"]
                ],
                "reason": "best trusted group-context preview",
            }
        )


class _StubBlenderCameraProcess:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render_camera_views(
        self,
        *,
        blend_file,
        out_dir,
        camera_views,
        preview=False,
    ):
        del blend_file
        self.calls.append(
            {
                "preview": preview,
                "candidate_ids": [
                    str(item["id"]) for item in camera_views
                ],
            }
        )
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        views = []
        for index, candidate in enumerate(camera_views):
            path = destination / f"{candidate['id']}.png"
            image = Image.new(
                "RGB",
                (24, 24),
                (
                    30 + index * 7 % 180,
                    80 + index * 11 % 150,
                    120,
                ),
            )
            image.putpixel((0, 0), (240, 30, 20))
            image.save(path)
            views.append(
                {
                    "id": candidate["id"],
                    "path": str(path),
                    "preview": preview,
                }
            )
        manifest_path = destination / "camera_render_manifest.json"
        manifest = {
            "views": views,
            "camera_evidence": {
                "preview": preview,
                "source_blend_modified": False,
            },
            "render_gpu_time_seconds": 0.01,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest


def _image(path: Path) -> str:
    image = Image.new("RGB", (24, 24), (60, 90, 120))
    image.putpixel((0, 0), (200, 40, 30))
    image.save(path)
    return str(path)


def test_faithful_production_group_semantic_camera_path(
    tmp_path: Path,
) -> None:
    initial = _image(tmp_path / "initial_group.png")
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"stub blend")
    scene = {
        "scene_id": "production-path",
        "boundary": [[0, 0], [6, 0], [6, 5], [0, 5]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "a",
                "center": [2.0, 2.0, 0.5],
                "size": [1.2, 0.8, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "b",
                "center": [3.2, 2.0, 0.5],
                "size": [0.8, 0.8, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            },
        ],
    }
    judge_model = _SequentialJudgeModel()
    production_judge = OpenAICompatibleVLMJudge(judge_model)
    judge = ExistingJudgeAdapter(
        production_judge,
        method_name="_adjudicate_scene_quality_raw",
        decision_contract="canonical_metric_v1",
    )
    selector_model = _CandidateSelectorModel()
    production_selector = OpenAICompatibleCameraSelector(
        selector_model
    )
    render_process = _StubBlenderCameraProcess()
    controller = VLMEvaluationController(
        judge=judge,
        renderer=CameraViewEvidenceRenderer(
            renderer=render_process,
            blend_file=blend,
            out_dir=tmp_path / "final",
        ),
        deterministic_camera_selector=(
            DeterministicLocalCameraSelector()
        ),
        vlm_camera_selector=production_selector,
        candidate_preview_renderer=CameraCandidatePreviewRenderer(
            renderer=render_process,
            blend_file=blend,
            out_dir=tmp_path / "preview",
        ),
        control=resolve_vlm_evaluation_control(
            {
                "camera_acquisition": {
                    "policy": "deterministic_then_vlm",
                    "vlm": {"selection_mode": "repair_plan"},
                }
            }
        ),
    )
    group_scope = {
        "group_id": "group_001",
        "member_ids": ["a", "b"],
        "target_bounds": {
            "min": [1.4, 1.6, 0.0],
            "max": [3.6, 2.4, 1.0],
        },
        "focus_center": [2.5, 2.0, 0.5],
        "extent": [2.2, 0.8, 1.0],
    }
    request = JudgeRequest(
        task="object_pairing_consistency",
        metric="object_pairing_consistency",
        claim_or_event={"object_ids": ["a", "b"]},
        scene_context=scene,
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=(
            {
                "path": initial,
                "role": "group_local",
                "view_id": "initial",
                "group_id": "group_001",
            },
        ),
        rubric={"scope": "object_pairing_consistency"},
        context={
            "group_scope": group_scope,
            "member_ids": ["a", "b"],
            "target_object_ids": ["a", "b"],
            "target_bounds": group_scope["target_bounds"],
            "focus_center": group_scope["focus_center"],
            "target_extent": group_scope["extent"],
        },
    )

    result = controller.run(request)

    assert result.status == "valid"
    assert judge_model.calls == 2
    assert len(selector_model.calls) == 1
    assert [call["preview"] for call in render_process.calls] == [
        True,
        False,
    ]
    stages = [item["stage"] for item in result.audit["trace"]]
    assert stages.index("trusted_candidate_bank") < stages.index(
        "candidate_preview_render"
    )
    assert stages.index("candidate_preview_render") < stages.index(
        "render"
    )
    assert result.audit["semantic_selection_triggered"] is True
    assert result.audit["effective_vlm_selection_mode"] == (
        "candidate_only"
    )
    assert result.audit["preview_render_count"] == 8
    assert result.audit["final_render_count"] == 1
    assert result.audit["production_camera_selector_backend"] == (
        "openai_compatible_camera_selector"
    )
    assert result.audit["trusted_candidate_count"] == 8
    assert result.audit["group_id"] == "group_001"
    assert result.audit["focus_target_ids"] == ["a", "b"]
    assert result.audit["authoritative_group_member_ids"] == [
        "a",
        "b",
    ]
