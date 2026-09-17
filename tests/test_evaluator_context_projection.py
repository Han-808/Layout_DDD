from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PIL import Image

from benchmark.api.evaluation import run_evaluate
from benchmark.evaluator.context_projection import (
    project_scene_for_evaluator_context,
)
from benchmark.evaluator.scene_quality.interfaces import (
    _judge_request,
    _request_scene_quality_evidence,
)
from benchmark.visual_judge.p0b import build_p0b_local_evidence_request


def _scene() -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "intent_projection_test",
        "request_id": "intent_projection_request",
        "scene_type": "room",
        "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
        "scene_height": 2.8,
        "objects": [
            {
                "id": "asset_instance",
                "category": "microwave_oven",
                "description": "countertop oven asset",
                "short_desc": "compact countertop oven",
                "center": [2.0, 2.7, 1.45],
                "size": [0.8, 0.6, 0.5],
                "rotation": [0, 0, 0],
                "metadata": {
                    "task_slot": {
                        "intended_category": "microwave_oven",
                        "intended_role": "private_object_role",
                    },
                    "asset_fact": "preserved",
                },
            }
        ],
        "metadata": {
            "generation_prompt": "private generation prompt",
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            },
            "instance_registry": {
                "instances": [
                    {
                        "id": "asset_instance",
                        "task_slot": {
                            "intended_category": "microwave_oven",
                            "intended_role": "private_registry_role",
                        },
                        "geometry_fact": "preserved",
                    }
                ]
            },
        },
    }


def _serialized(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _contains_key(value: object, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_projection_recursively_removes_generator_private_intent() -> None:
    source = _scene()
    frozen = deepcopy(source)

    projected = project_scene_for_evaluator_context(source)

    assert source == frozen
    assert "task_slot" not in _serialized(projected)
    assert "private_object_role" not in _serialized(projected)
    assert "private_registry_role" not in _serialized(projected)
    assert "private generation prompt" not in _serialized(projected)
    assert projected["objects"][0]["metadata"]["asset_fact"] == "preserved"
    assert projected["objects"][0]["category"] == "compact countertop oven"
    assert (
        projected["metadata"]["instance_registry"]["instances"][0][
            "geometry_fact"
        ]
        == "preserved"
    )
    assert projected["objects"][0]["center"] == source["objects"][0]["center"]


def test_p0b_camera_request_excludes_instance_registry_task_slot() -> None:
    request = build_p0b_local_evidence_request(
        metric="support",
        event={"object_id": "asset_instance"},
        prompt="",
        relationships=None,
        scene=_scene(),
        detector_evidence={"gap_band": "strong_positive_clearance"},
        object_ids=["asset_instance"],
    )

    assert "task_slot" not in _serialized(request)
    assert "private_registry_role" not in _serialized(request)


def test_scene_quality_judge_and_camera_requests_exclude_task_slots() -> None:
    scene = _scene()
    judge_request = _judge_request(
        metric_name="scale_consistency",
        scene=scene,
        prompt=None,
        render_evidence=[],
        selected_object_ids=["asset_instance"],
        selected_group_ids=[],
        groups=[],
        authorized_deviations=[],
        visual_style_spec=None,
    )
    captured: list[dict] = []

    def provider(request: dict) -> list[str]:
        captured.append(request)
        return []

    _request_scene_quality_evidence(
        provider,
        metric_name="scale_consistency",
        policy={"camera_scope": "scene_global"},
        scene=scene,
        prompt=None,
        selected_object_ids=["asset_instance"],
        selected_group_ids=[],
        selected_groups=[],
    )

    assert "task_slot" not in _serialized(judge_request)
    assert captured
    assert "task_slot" not in _serialized(captured[0])
    assert "private_registry_role" not in _serialized(captured[0])


class _FullWorkflowJudge:
    vlm_control_enabled = False

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def adjudicate_p0b(self, request: dict) -> dict:
        self.requests.append(deepcopy(request))
        return {
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "mock structural verdict",
        }

    def screen_scene_quality(self, request: dict) -> dict:
        return self.adjudicate_scene_quality(request)

    def adjudicate_scene_quality(self, request: dict) -> dict:
        self.requests.append(deepcopy(request))
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "mock scene-quality verdict",
            "missing_evidence": [],
            "defects": [],
            "evidence_request": None,
        }

    def adjudicate_relation(self, request: dict) -> dict:
        self.requests.append(deepcopy(request))
        return {
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "mock relation verdict",
        }


def test_canonical_full_workflow_excludes_task_slots_from_all_requests(
    tmp_path: Path,
) -> None:
    scene = _scene()
    # Force one ambiguous Support route so L1 exercises the P0b request path.
    scene["objects"][0]["center"][2] = 1.7
    image_path = tmp_path / "evidence.png"
    Image.new("RGB", (32, 32), (80, 100, 120)).save(image_path)
    judge = _FullWorkflowJudge()
    provider_requests: list[dict] = []

    def provider(request: dict) -> list[str]:
        provider_requests.append(deepcopy(request))
        return [str(image_path)]

    report = run_evaluate(
        scene=scene,
        out=tmp_path / "evaluation_report.json",
        render_evidence=[str(image_path)],
        object_grouping_report={
            "status": "complete",
            "grouping_backend": "mock",
            "grouping_policy_id": "vlm_visual_evidence_scope_v2",
            "object_groups": [
                {
                    "group_id": "group_001",
                    "object_ids": ["asset_instance"],
                    "label": "fixture",
                }
            ],
        },
        p0b_local_view_provider=provider,
        l3_initial_evidence_provider=provider,
        functional_probe_evidence_provider=provider,
        vlm_judge=judge,
        asset_policy={
            "mode": "generated_or_open_assets",
            "identity_owner": "generator",
            "category_selection_owner": "generator",
            "scale_owner": "generator",
            "appearance_owner": "generator",
            "arrangement_owner": "generator",
        },
    )

    assert report["reports"]["generic_validity"]["status"] == "ok"
    assert judge.requests
    assert provider_requests
    serialized = _serialized(
        {
            "judge_requests": judge.requests,
            "provider_requests": provider_requests,
        }
    )
    assert not _contains_key(judge.requests, "task_slot")
    assert not _contains_key(provider_requests, "task_slot")
    assert "private_object_role" not in serialized
    assert "private_registry_role" not in serialized
