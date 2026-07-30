from __future__ import annotations

import json
from copy import deepcopy

import pytest

from benchmark.visual_judge.control_config import (
    resolve_vlm_evaluation_control,
)
from benchmark.visual_judge.control_loop import (
    EvidenceRenderResult,
    ExistingEvidenceRendererAdapter,
    VLMEvaluationController,
)
from benchmark.visual_judge.contracts import (
    validate_camera_selection_response,
)
from benchmark.visual_judge.evidence_gate import DeterministicEvidenceGate
from benchmark.visual_judge.interfaces import (
    CameraSelectionRequest,
    CameraSelector,
    DeterministicCameraSelector,
    EvidenceGateRequest,
    EvidenceGateResult,
    EvidenceRequest,
    ExistingJudgeAdapter,
    HybridCameraSelector,
    JudgeRequest,
    JudgeResult,
    VLMCameraSelector,
    build_camera_selector,
    camera_selection_result_from_value,
)
from benchmark.visual_judge.roles import DecisionContract, VLMRole
from benchmark.visual_judge.runtime import (
    ControlledVLMJudge,
    EvidenceControlUnresolvedError,
    build_controlled_vlm_judge,
)


def _judge_request(*, evidence: tuple[object, ...] = ("initial.png",)):
    return JudgeRequest(
        task="collision",
        metric="collision",
        claim_or_event={"event_id": "event-1", "object_ids": ["a", "b"]},
        scene_context={"scene_id": "scene-1"},
        deterministic_evidence={"detector": "unresolved"},
        visual_evidence=evidence,
        rubric={"scope": "collision"},
    )


def _gate_result(
    *,
    ready: bool,
    camera_repairable: bool = False,
) -> EvidenceGateResult:
    deficiencies = (
        ()
        if ready
        else (
            {
                "code": "target_not_visible",
                "repairability": (
                    "camera" if camera_repairable else "rerender"
                ),
            },
        )
    )
    return EvidenceGateResult(
        ready=ready,
        camera_repairable=camera_repairable,
        reason_codes=("evidence_ready",) if ready else ("target_not_visible",),
        deficiencies=deficiencies,
    )


def _valid_result() -> dict:
    return {
        "status": "valid",
        "confidence": 0.9,
        "reason": "the scoped claim is visually supported",
        "defects": [],
    }


def _need_more_result() -> dict:
    return {
        "status": "need_more_evidence",
        "confidence": 0.4,
        "reason": "the contact region is occluded",
        "defects": [],
        "evidence_request": {
            "target_ids": ["a", "b"],
            "missing_observations": ["support_contact_region"],
            "view_goal": "show the support contact region",
        },
    }


class _Gate:
    def __init__(self, results, calls):
        self.results = list(results)
        self.calls = calls
        self.requests = []

    def check(self, request):
        self.calls.append("gate")
        self.requests.append(request)
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


class _Judge:
    def __init__(self, results, calls):
        self.results = list(results)
        self.calls = calls
        self.requests = []

    def judge(self, request):
        self.calls.append("judge")
        self.requests.append(request)
        return self.results.pop(0)


class _Selector:
    def __init__(self, result, calls):
        self.result = result
        self.calls = calls
        self.requests = []

    def select(self, request):
        self.calls.append("selector")
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return deepcopy(self.result)


class _Renderer:
    def __init__(self, result, calls):
        self.result = result
        self.calls = calls
        self.requests = []

    def render(self, request):
        self.calls.append("render")
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return deepcopy(self.result)


def _selection(*, action: dict | None = None) -> dict:
    return {
        "selected_view_ids": ["view-1"],
        "action": action,
        "reason": "best available scoped view",
    }


def _run(
    *,
    gate_results,
    judge_results,
    render_result=None,
    selector_result=None,
    control=None,
    evidence=("initial.png",),
    manifest_path=None,
):
    calls = []
    gate = _Gate(gate_results, calls)
    judge = _Judge(judge_results, calls)
    selector = _Selector(selector_result or _selection(), calls)
    renderer = _Renderer(
        render_result
        or {
            "visual_evidence": ["repair.png"],
            "merge_policy": "append",
        },
        calls,
    )
    controller = VLMEvaluationController(
        judge=judge,
        camera_selector=selector,
        evidence_gate=gate,
        renderer=renderer,
        control=control,
    )
    result = controller.run(
        _judge_request(evidence=tuple(evidence)),
        candidate_views=({"id": "view-1"},),
        allowed_actions=("orbit",),
        control_manifest_path=manifest_path,
    )
    return result, calls, gate, judge, selector, renderer


def test_ready_evidence_calls_judge_without_camera():
    result, calls, _, judge, selector, renderer = _run(
        gate_results=[_gate_result(ready=True)],
        judge_results=[_valid_result()],
    )

    assert result.status == "valid"
    assert calls == ["gate", "judge"]
    assert len(judge.requests) == 1
    assert not selector.requests
    assert not renderer.requests


def test_vlm_role_enum_has_only_judge_and_optional_vlm_selector():
    assert {role.value for role in VLMRole} == {
        "judge",
        "vlm_camera_selector",
    }


def test_camera_repairable_evidence_runs_selector_render_gate_judge():
    result, calls, gate, _, _, _ = _run(
        gate_results=[
            _gate_result(ready=False, camera_repairable=True),
            _gate_result(ready=True),
        ],
        judge_results=[_valid_result()],
    )

    assert result.status == "valid"
    assert calls == ["gate", "selector", "render", "gate", "judge"]
    assert len(gate.requests) == 2
    assert list(result.visual_evidence) == ["initial.png", "repair.png"]


def test_renderer_cannot_exceed_max_views_per_round_before_judge():
    result, calls, _, judge, _, _ = _run(
        gate_results=[
            _gate_result(ready=False, camera_repairable=True),
            _gate_result(ready=True),
        ],
        judge_results=[_valid_result()],
        render_result={
            "visual_evidence": ["repair-a.png", "repair-b.png"],
            "merge_policy": "replace",
        },
        control=resolve_vlm_evaluation_control(
            {
                "budgets": {
                    "max_views_per_round": 1,
                    "max_total_images": 6,
                }
            }
        ),
    )

    assert result.status == "unresolved"
    assert result.stop_reason == "renderer_followup_contract_invalid"
    assert calls == ["gate", "selector", "render", "gate"]
    assert judge.requests == []
    assert result.audit["trace"][2]["rendered_view_count"] == 2


def test_same_pose_render_bundle_counts_as_one_view_per_round():
    result, calls, _, judge, _, _ = _run(
        gate_results=[
            _gate_result(ready=False, camera_repairable=True),
            _gate_result(ready=True),
        ],
        judge_results=[_valid_result()],
        render_result={
            "visual_evidence": [
                {
                    "view_id": "repair-view",
                    "pair_id": "repair-view",
                    "path": "repair-raw.png",
                    "role": "collision_rgb",
                },
                {
                    "view_id": "repair-view",
                    "pair_id": "repair-view",
                    "path": "repair-contour.png",
                    "role": "metric_local_contour",
                },
            ],
            "merge_policy": "replace",
        },
        control=resolve_vlm_evaluation_control(
            {"budgets": {"max_views_per_round": 1}}
        ),
    )

    assert result.status == "valid"
    assert calls == ["gate", "selector", "render", "gate", "judge"]
    assert len(judge.requests) == 1
    assert result.audit["trace"][2]["rendered_view_count"] == 1


def test_non_camera_repairable_failure_never_calls_judge():
    result, calls, _, judge, selector, renderer = _run(
        gate_results=[_gate_result(ready=False)],
        judge_results=[],
    )

    assert result.status == "unresolved"
    assert result.stop_reason == "evidence_not_camera_repairable"
    assert calls == ["gate"]
    assert not judge.requests
    assert not selector.requests
    assert not renderer.requests


def test_judge_need_more_evidence_runs_full_next_round():
    result, calls, gate, judge, selector, _ = _run(
        gate_results=[
            _gate_result(ready=True),
            _gate_result(ready=True),
        ],
        judge_results=[_need_more_result(), _valid_result()],
    )

    assert result.status == "valid"
    assert calls == ["gate", "judge", "selector", "render", "gate", "judge"]
    assert len(gate.requests) == 2
    assert len(judge.requests) == 2
    assert selector.requests[0].target_ids == ("a", "b")
    assert (
        selector.requests[0].evidence_goal["view_goal"]
        == "show the support contact region"
    )
    assert selector.requests[0].evidence_round == 1


def test_zero_evidence_round_budget_stops_before_selector():
    control = resolve_vlm_evaluation_control(
        {"budgets": {"max_evidence_rounds": 0}}
    )
    result, calls, _, judge, selector, _ = _run(
        gate_results=[_gate_result(ready=False, camera_repairable=True)],
        judge_results=[],
        control=control,
    )

    assert result.stop_reason == "max_evidence_rounds_exhausted"
    assert calls == ["gate"]
    assert not judge.requests
    assert not selector.requests


def test_image_budget_stops_need_more_loop_without_rejudging():
    control = resolve_vlm_evaluation_control(
        {"budgets": {"max_total_images": 1}}
    )
    result, calls, _, judge, selector, _ = _run(
        gate_results=[_gate_result(ready=True)],
        judge_results=[_need_more_result()],
        control=control,
    )

    assert result.stop_reason == "max_total_images_exhausted"
    assert calls == ["gate", "judge"]
    assert len(judge.requests) == 1
    assert not selector.requests


def test_selector_call_budget_prevents_infinite_need_more_loop():
    control = resolve_vlm_evaluation_control(
        {
            "budgets": {
                "max_evidence_rounds": 2,
                "max_selector_calls": 1,
            }
        }
    )
    result, calls, _, judge, selector, _ = _run(
        gate_results=[
            _gate_result(ready=True),
            _gate_result(ready=True),
        ],
        judge_results=[_need_more_result(), _need_more_result()],
        control=control,
    )

    assert result.stop_reason == "max_selector_calls_exhausted"
    assert calls == [
        "gate",
        "judge",
        "selector",
        "render",
        "gate",
        "judge",
    ]
    assert len(judge.requests) == 2
    assert len(selector.requests) == 1


def test_camera_action_budget_is_checked_before_render():
    control = resolve_vlm_evaluation_control(
        {"budgets": {"max_camera_actions": 0}}
    )
    result, calls, _, _, _, renderer = _run(
        gate_results=[_gate_result(ready=False, camera_repairable=True)],
        judge_results=[],
        selector_result=_selection(
            action={"type": "orbit", "view_id": "view-1"}
        ),
        control=control,
    )

    assert result.stop_reason == "max_camera_actions_exhausted"
    assert calls == ["gate", "selector"]
    assert not renderer.requests


def test_rendered_packet_over_image_budget_is_gated_before_stopping():
    control = resolve_vlm_evaluation_control(
        {"budgets": {"max_total_images": 1}}
    )
    result, calls, gate, judge, _, _ = _run(
        gate_results=[
            _gate_result(ready=False, camera_repairable=True),
            _gate_result(ready=True),
        ],
        judge_results=[],
        evidence=(),
        render_result={
            "visual_evidence": ["one.png", "two.png"],
            "merge_policy": "replace",
        },
        control=control,
    )

    assert result.stop_reason == "max_total_images_exhausted"
    assert calls == ["gate", "selector", "render", "gate"]
    assert len(gate.requests) == 2
    assert not judge.requests


def test_selector_failure_keeps_previous_evidence_and_does_not_rejudge():
    result, calls, _, judge, _, _ = _run(
        gate_results=[_gate_result(ready=True)],
        judge_results=[_need_more_result()],
        selector_result=RuntimeError("selector unavailable"),
    )

    assert result.status == "unresolved"
    assert result.stop_reason == "camera_selector_failed"
    assert list(result.visual_evidence) == ["initial.png"]
    assert calls == ["gate", "judge", "selector"]
    assert len(judge.requests) == 1


def test_unchanged_rendered_packet_is_gated_then_stops():
    result, calls, gate, judge, _, _ = _run(
        gate_results=[
            _gate_result(ready=False, camera_repairable=True),
            _gate_result(ready=False, camera_repairable=True),
        ],
        judge_results=[],
        render_result={
            "visual_evidence": ["initial.png"],
            "merge_policy": "replace",
        },
    )

    assert result.stop_reason == "evidence_packet_unchanged"
    assert calls == ["gate", "selector", "render", "gate"]
    assert len(gate.requests) == 2
    assert not judge.requests


def test_replace_mode_still_consumes_cumulative_image_budget():
    control = resolve_vlm_evaluation_control(
        {
            "budgets": {
                "max_total_images": 2,
                "max_evidence_rounds": 2,
            }
        }
    )
    result, calls, _, judge, selector, _ = _run(
        gate_results=[
            _gate_result(ready=True),
            _gate_result(ready=True),
        ],
        judge_results=[_need_more_result(), _need_more_result()],
        render_result={
            "visual_evidence": ["replacement.png"],
            "merge_policy": "replace",
        },
        control=control,
    )

    assert result.stop_reason == "max_total_images_exhausted"
    assert calls == [
        "gate",
        "judge",
        "selector",
        "render",
        "gate",
        "judge",
    ]
    assert len(result.visual_evidence) == 1
    assert result.audit["current_packet_image_count"] == 1
    assert result.audit["total_images_acquired"] == 2
    assert len(judge.requests) == 2
    assert len(selector.requests) == 1


def test_renderer_cannot_underreport_selected_camera_action():
    result, calls, _, judge, _, _ = _run(
        gate_results=[
            _gate_result(ready=False, camera_repairable=True),
        ],
        judge_results=[],
        selector_result=_selection(
            action={"type": "orbit", "view_id": "view-1"}
        ),
        render_result={
            "visual_evidence": ["repair.png"],
            "camera_actions_executed": 0,
        },
    )

    assert result.stop_reason == "render_failed"
    assert calls == ["gate", "selector", "render", "gate"]
    assert not judge.requests


def test_invalid_post_render_candidates_are_gated_before_safe_stop():
    result, calls, gate, judge, _, _ = _run(
        gate_results=[
            _gate_result(ready=False, camera_repairable=True),
            _gate_result(ready=True),
        ],
        judge_results=[],
        render_result={
            "visual_evidence": ["repair.png"],
            "next_candidate_views": [{"id": ""}],
        },
    )

    assert result.stop_reason == "renderer_followup_contract_invalid"
    assert calls == ["gate", "selector", "render", "gate"]
    assert len(gate.requests) == 2
    assert not judge.requests


def test_same_image_pixels_with_changed_metadata_are_not_rejudged(tmp_path):
    evidence = tmp_path / "same.png"
    evidence.write_bytes(b"same pixels")
    initial = {"path": str(evidence), "view_id": "old", "timestamp": 1}
    replacement = {
        "path": str(evidence),
        "view_id": "new",
        "timestamp": 2,
    }
    result, calls, gate, judge, _, _ = _run(
        gate_results=[
            _gate_result(ready=True),
            _gate_result(ready=True),
        ],
        judge_results=[_need_more_result()],
        evidence=(initial,),
        render_result={
            "visual_evidence": [replacement],
            "merge_policy": "append",
        },
    )

    assert result.stop_reason == "evidence_packet_unchanged"
    assert calls == ["gate", "judge", "selector", "render", "gate"]
    assert len(gate.requests) == 2
    assert len(judge.requests) == 1


def test_control_manifest_records_resolved_parameters(tmp_path):
    manifest = tmp_path / "vlm_control_manifest.json"
    control = resolve_vlm_evaluation_control(
        {"budgets": {"max_evidence_rounds": 1}}
    )
    result, _, _, _, _, _ = _run(
        gate_results=[_gate_result(ready=True)],
        judge_results=[_valid_result()],
        control=control,
        manifest_path=manifest,
    )

    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert result.manifest_path == str(manifest)
    assert recorded["control"]["requested"]["budgets"][
        "max_evidence_rounds"
    ] == 1
    assert recorded["control"]["effective"]["budgets"][
        "max_evidence_rounds"
    ] == 1
    assert recorded["control"]["sources"][
        "budgets.max_evidence_rounds"
    ] == "config"
    assert recorded["rounds_used"] == 0
    assert recorded["unique_rendered_evidence_count"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {
            "selected_view_ids": ["unknown"],
            "reason": "unknown candidate",
        },
        {
            "selected_view_ids": ["view-1"],
            "reason": "bad role",
            "verdict": "valid",
        },
        {
            "selected_view_ids": ["view-1"],
            "reason": "bad role",
            "score": 1.0,
        },
        {
            "selected_view_ids": ["view-1"],
            "reason": "illegal action",
            "action": {"type": "dolly", "view_id": "view-1"},
        },
        {
            "selected_view_ids": [],
            "reason": "unvalidated proposal",
            "camera_proposal": {"proposal_id": "unknown"},
        },
        {
            "selected_view_ids": ["view-1"],
            "reason": "scene mutation",
            "scene_patch": {"objects": []},
        },
    ],
)
def test_selector_validation_rejects_unsafe_results(payload):
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        candidate_views=({"id": "view-1"},),
        allowed_actions=("orbit",),
    )

    with pytest.raises(ValueError):
        camera_selection_result_from_value(
            payload,
            request=request,
            backend="test",
        )


def test_selector_accepts_bounded_proposal_without_selected_view():
    trusted_proposal = {
        "proposal_id": "repair-1",
        "validated": True,
        "result_pose": {"location": [1, 2, 3]},
    }
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        context={
            "corrective_proposals": [trusted_proposal]
        },
    )

    result = camera_selection_result_from_value(
        {
            "selected_view_ids": [],
            "camera_proposal": {"proposal_id": "repair-1"},
            "reason": "bounded corrective proposal",
        },
        request=request,
        backend="test",
    )

    assert result.selected_view_ids == ()
    assert result.camera_proposal == trusted_proposal
    assert camera_selection_result_from_value(
        result,
        request=request,
        backend="test",
    ).camera_proposal == trusted_proposal


def test_selector_cannot_spoof_backend_or_evidence_round():
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        candidate_views=({"id": "view-1"},),
        evidence_round=2,
    )

    result = camera_selection_result_from_value(
        {
            "selected_view_ids": ["view-1"],
            "reason": "known candidate",
            "backend": "spoofed",
            "evidence_round": 999,
        },
        request=request,
        backend="deterministic",
    )

    assert result.backend == "deterministic"
    assert result.evidence_round == 2
    assert result.provenance["reported_backend"] == "spoofed"
    assert result.provenance["reported_evidence_round"] == 999


@pytest.mark.parametrize(
    "selected_view",
    [
        {},
        {"id": "view-1", "pose": {"location": [9, 9, 9]}},
    ],
)
def test_selector_rejects_missing_or_tampered_selected_view(selected_view):
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        candidate_views=(
            {"id": "view-1", "pose": {"location": [1, 2, 3]}},
        ),
    )

    with pytest.raises(ValueError, match="exactly match trusted"):
        camera_selection_result_from_value(
            {
                "selected_view_ids": ["view-1"],
                "selected_views": [selected_view],
                "reason": "claim a selected view",
            },
            request=request,
            backend="test",
        )


@pytest.mark.parametrize(
    "proposal",
    [
        {"proposal_id": "repair-1", "pose": {"location": [9, 9, 9]}},
        {"proposal_id": "repair-1", "scene_patch": {"objects": []}},
    ],
)
def test_selector_rejects_bounded_proposal_payload_tampering(proposal):
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        context={
            "corrective_proposals": [
                {
                    "proposal_id": "repair-1",
                    "validated": True,
                    "result_pose": {"location": [1, 2, 3]},
                }
            ]
        },
    )

    with pytest.raises(ValueError):
        camera_selection_result_from_value(
            {
                "selected_view_ids": [],
                "camera_proposal": proposal,
                "reason": "tampered proposal",
            },
            request=request,
            backend="test",
        )


def test_selector_rejects_explicitly_invalid_trusted_proposal():
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        context={
            "corrective_proposals": [
                {"proposal_id": "repair-1", "validated": False}
            ]
        },
    )

    with pytest.raises(ValueError, match="failed validation"):
        camera_selection_result_from_value(
            {
                "selected_view_ids": [],
                "camera_proposal": {"proposal_id": "repair-1"},
                "reason": "invalid proposal",
            },
            request=request,
            backend="test",
        )


@pytest.mark.parametrize(
    "action",
    [
        {},
        {"view_id": "view-1", "type": "orbit", "pose": {"x": 1}},
        {"view_id": "unknown", "type": "orbit"},
        {"view_id": "view-1", "type": "dolly"},
    ],
)
def test_selector_rejects_empty_unknown_or_untrusted_camera_actions(action):
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a",),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        candidate_views=({"id": "view-1"},),
        allowed_actions=("orbit",),
    )

    with pytest.raises(ValueError):
        camera_selection_result_from_value(
            {
                "selected_view_ids": ["view-1"],
                "camera_actions": [action],
                "reason": "unsafe action",
            },
            request=request,
            backend="test",
        )


def test_selector_reconstructs_bounded_action_from_trusted_proposal():
    trusted = {
        "proposal_id": "repair-1",
        "parent_view_id": "view-1",
        "action_primitive": "orbit",
        "family": "contact_left",
        "validated": True,
        "result_pose": {"location": [1, 2, 3]},
    }
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a",),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        candidate_views=({"id": "view-1"},),
        allowed_actions=("orbit",),
        context={"corrective_proposals": [trusted]},
    )

    result = camera_selection_result_from_value(
        {
            "selected_view_ids": ["view-1"],
            "camera_actions": [{"proposal_id": "repair-1"}],
            "reason": "bounded repair",
        },
        request=request,
        backend="test",
    )

    assert result.camera_actions == (
        {
            "view_id": "view-1",
            "type": "orbit",
            "proposal_id": "repair-1",
            "family": "contact_left",
        },
    )
    assert camera_selection_result_from_value(
        result,
        request=request,
        backend="test",
    ).camera_actions == result.camera_actions


def test_selector_rejects_selected_views_with_camera_proposal():
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a",),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        candidate_views=({"id": "view-1"},),
        context={
            "corrective_proposals": [
                {"proposal_id": "repair-1", "validated": True}
            ]
        },
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        camera_selection_result_from_value(
            {
                "selected_view_ids": ["view-1"],
                "camera_proposal": {"proposal_id": "repair-1"},
                "reason": "ambiguous output",
            },
            request=request,
            backend="test",
        )


def test_public_camera_validator_requires_nonempty_reason():
    with pytest.raises(ValueError, match="non-empty reason"):
        validate_camera_selection_response(
            {
                "selected_view_ids": ["view-1"],
                "reason": " ",
            },
            available_view_ids=("view-1",),
            max_views=1,
        )


def test_freeform_pose_requires_independent_validation():
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={"scene_id": "scene-1"},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        allow_freeform_pose=True,
    )
    payload = {
        "selected_view_ids": [],
        "camera_proposal": {"pose": {"location": [1, 2, 3]}},
        "reason": "new pose",
    }

    with pytest.raises(ValueError, match="injected pose validator"):
        camera_selection_result_from_value(
            payload,
            request=request,
            backend="experimental",
        )

    validated_request = CameraSelectionRequest(
        **{
            **request.__dict__,
            "context": {
                "pose_validator": (
                    lambda proposal, scene: (
                        proposal["pose"]["location"] == [1, 2, 3]
                        and scene["scene_id"] == "scene-1"
                    )
                )
            },
        }
    )
    result = camera_selection_result_from_value(
        payload,
        request=validated_request,
        backend="experimental",
    )

    assert result.camera_proposal == payload["camera_proposal"]


def test_selector_backends_share_one_protocol():
    class _LegacySelector:
        def __init__(self):
            self.requests = []

        def select_camera_views(self, request):
            self.requests.append(request)
            return {
                "selected_view_ids": ["view-1"],
                "reason": "legacy selector",
            }

    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a",),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        candidate_views=({"id": "view-1"},),
    )
    deterministic = DeterministicCameraSelector()
    legacy = _LegacySelector()
    vlm = VLMCameraSelector(legacy)
    hybrid = HybridCameraSelector(vlm, deterministic)

    assert isinstance(deterministic, CameraSelector)
    assert isinstance(vlm, CameraSelector)
    assert isinstance(hybrid, CameraSelector)
    assert deterministic.select(request).selected_view_ids == ("view-1",)
    vlm_result = vlm.select(request)
    assert vlm_result.selected_view_ids == ("view-1",)
    assert vlm_result.provenance["vlm_role"] == "vlm_camera_selector"
    assert legacy.requests[0]["vlm_role"] == "vlm_camera_selector"
    assert (
        legacy.requests[0]["decision_contract"]
        == "camera_selection_v1"
    )
    assert hybrid.select(request).selected_view_ids == ("view-1",)


def test_vlm_factory_wraps_stable_selector_and_injects_audit_metadata():
    class _Model:
        model_id = "selector-model"
        endpoint = "https://selector.invalid/v1"

    class _StableSelector:
        model = _Model()

        def __init__(self):
            self.requests = []

        def select(self, request):
            self.requests.append(request)
            return {
                "selected_view_ids": ["view-1"],
                "reason": "best view",
                "images_used": ["preview-00"],
            }

    stable = _StableSelector()
    selector = build_camera_selector(backend="vlm", vlm=stable)
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a",),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        candidate_views=({"id": "view-1"},),
    )

    assert selector is not stable
    result = selector.select(request)

    assert isinstance(stable.requests[0], CameraSelectionRequest)
    assert stable.requests[0].context["vlm_role"] == "vlm_camera_selector"
    assert (
        stable.requests[0].context["decision_contract"]
        == "camera_selection_v1"
    )
    assert stable.requests[0].context["judge_method"] == "select"
    assert result.provenance["vlm_role"] == "vlm_camera_selector"
    assert result.provenance["decision_contract"] == "camera_selection_v1"
    assert result.provenance["judge_method"] == "select"
    assert result.provenance["model"] == "selector-model"
    assert result.provenance["endpoint"] == "https://selector.invalid/v1"
    assert result.provenance["images_used"] == ["preview-00"]


def test_vlm_selector_rejects_conflicting_role_provenance():
    class _Selector:
        def select(self, request):
            del request
            return {
                "selected_view_ids": ["view-1"],
                "reason": "spoofed audit identity",
                "provenance": {"vlm_role": "judge"},
            }

    selector = VLMCameraSelector(_Selector())
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a",),
        scene={},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1},
        candidate_views=({"id": "view-1"},),
    )

    with pytest.raises(ValueError, match=r"provenance\.vlm_role conflicts"):
        selector.select(request)


def test_selector_factory_supports_existing_deterministic_vlm_and_hybrid():
    class _LegacySelector:
        def select_camera_views(self, request):
            return {
                "selected_view_ids": ["view-1"],
                "reason": "legacy selector",
            }

    deterministic = build_camera_selector(backend="deterministic")
    existing = build_camera_selector(
        backend="existing",
        existing=_LegacySelector(),
    )
    vlm = build_camera_selector(backend="vlm", vlm=_LegacySelector())
    hybrid = build_camera_selector(
        backend="hybrid",
        vlm=_LegacySelector(),
    )

    assert isinstance(deterministic, CameraSelector)
    assert isinstance(existing, CameraSelector)
    assert isinstance(vlm, CameraSelector)
    assert isinstance(hybrid, CameraSelector)
    assert (
        type(build_camera_selector(backend="existing")).__name__
        == "DeterministicCameraSelector"
    )


def test_controller_accepts_an_already_built_hybrid_selector():
    calls = []
    primary = _Selector(_selection(), calls)
    fallback = _Selector(_selection(), calls)
    hybrid = HybridCameraSelector(primary, fallback)
    controller = VLMEvaluationController(
        judge=_Judge([_valid_result()], calls),
        renderer=_Renderer([], calls),
        camera_selector=hybrid,
        evidence_gate=_Gate([_gate_result(ready=True)], calls),
        control=resolve_vlm_evaluation_control(
            {"camera_selector": {"backend": "hybrid"}}
        ),
    )

    result = controller.run(_judge_request())

    assert result.status == "valid"
    assert calls == ["gate", "judge"]


def test_deterministic_evidence_gate_has_no_model_or_metric_verdict(tmp_path):
    evidence = tmp_path / "view.png"
    evidence.write_bytes(b"not-empty")
    gate = DeterministicEvidenceGate()

    result = gate.check(
        EvidenceGateRequest(
            task="style_consistency",
            metric="style_consistency",
            target_ids=("a",),
            scene={},
            visual_evidence=(str(evidence),),
        )
    )

    assert result.ready is True
    assert not hasattr(gate, "model")
    assert "verdict" not in result.to_dict()
    assert "score" not in result.to_dict()
    with pytest.raises(ValueError, match="must not return"):
        EvidenceGateResult.from_value(
            {
                "ready": True,
                "camera_repairable": False,
                "reason_codes": [],
                "deficiencies": [],
                "verdict": "valid",
            }
        )


def test_deterministic_evidence_gate_rejects_empty_render_file(tmp_path):
    evidence = tmp_path / "empty.png"
    evidence.touch()

    result = DeterministicEvidenceGate().check(
        EvidenceGateRequest(
            task="style_consistency",
            metric="style_consistency",
            target_ids=("a",),
            scene={},
            visual_evidence=(str(evidence),),
        )
    )

    assert result.ready is False
    assert result.camera_repairable is False
    assert "empty_render_file" in result.reason_codes


def test_evidence_gate_accepts_sufficient_metadata_subset_in_mixed_oob_packet(
    tmp_path,
):
    global_view = tmp_path / "global.png"
    local_view = tmp_path / "local.png"
    global_view.write_bytes(b"global pixels")
    local_view.write_bytes(b"local pixels")

    result = DeterministicEvidenceGate().check(
        EvidenceGateRequest(
            task="oob",
            metric="oob",
            target_ids=("chair-1",),
            scene={},
            visual_evidence=(
                str(global_view),
                {
                    "path": str(local_view),
                    "role": "metric_local_highlight",
                    "view_id": "local-oob",
                    "target_ids": ["chair-1"],
                    "visibility": {
                        "target_pixel_fractions": {"chair-1": 0.02},
                        "region_pixel_fractions": {
                            "architecture_plane": 0.2,
                        },
                    },
                },
            ),
        )
    )

    assert result.ready is True
    assert result.provenance["assessment_scope"] == "metadata_subset"
    assert result.provenance["metadata_evidence_count"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "ready": True,
            "camera_repairable": False,
            "reason_codes": ["target_not_visible"],
            "deficiencies": [],
        },
        {
            "ready": False,
            "camera_repairable": True,
            "reason_codes": ["target_not_visible"],
            "deficiencies": [],
        },
        {
            "ready": False,
            "camera_repairable": True,
            "reason_codes": ["render_failed"],
            "deficiencies": [
                {"code": "render_failed", "repairability": "rerender"}
            ],
        },
    ],
)
def test_evidence_gate_result_rejects_inconsistent_readiness(payload):
    with pytest.raises(ValueError):
        EvidenceGateResult.from_value(payload)


def test_judge_need_more_evidence_requires_structured_request():
    with pytest.raises(ValueError, match="structured evidence_request"):
        JudgeResult.from_value(
            {
                "status": "need_more_evidence",
                "confidence": 0.5,
                "reason": "need another view",
                "defects": [],
            }
        )
    with pytest.raises(ValueError, match="target_ids"):
        EvidenceRequest.from_value(
            {
                "target_ids": [],
                "missing_observations": ["global_composition"],
                "view_goal": "show the full scene",
            }
        )
    request = EvidenceRequest.from_value(
        {
            "target_ids": ["scene"],
            "missing_observations": ["global_composition"],
            "view_goal": "show the full scene",
        }
    )
    assert request.target_ids == ("scene",)
    with pytest.raises(ValueError, match="view_goal"):
        JudgeResult.from_value(
            {
                "status": "need_more_evidence",
                "confidence": 0.5,
                "reason": "need another view",
                "defects": [],
                "evidence_request": {
                    "target_ids": ["a"],
                    "missing_observations": ["contact"],
                },
            }
        )


def test_existing_judge_adapter_adds_explicit_audit_contract():
    class _LegacyJudge:
        def __init__(self):
            self.requests = []

        def adjudicate_relation(self, request):
            self.requests.append(request)
            return {
                "verdict": "valid",
                "confidence": 0.8,
                "reason": "relation visible",
            }

    legacy = _LegacyJudge()
    adapter = ExistingJudgeAdapter(
        legacy,
        method_name="adjudicate_relation",
        decision_contract=DecisionContract.RELATION_BINARY,
    )

    result = adapter.judge(_judge_request())

    assert result.status == "valid"
    assert legacy.requests[0]["vlm_role"] == "judge"
    assert (
        legacy.requests[0]["decision_contract"]
        == "relation_binary_v1"
    )
    assert legacy.requests[0]["judge_method"] == "adjudicate_relation"
    assert result.provenance["vlm_role"] == "judge"
    assert result.provenance["decision_contract"] == "relation_binary_v1"
    assert adapter.last_raw_response == {
        "verdict": "valid",
        "confidence": 0.8,
        "reason": "relation visible",
    }


@pytest.mark.parametrize(
    "response",
    [
        {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": 0.8,
            "reason": "defect claimed without a record",
            "missing_evidence": [],
            "defects": [],
        },
        {
            "evidence_status": "insufficient",
            "verdict": "valid",
            "confidence": 0.4,
            "reason": "inconsistent evidence state",
            "missing_evidence": ["contact"],
            "defects": [],
        },
    ],
)
def test_existing_judge_adapter_enforces_canonical_contract(response):
    class _LegacyJudge:
        def adjudicate_scene_quality(self, request):
            return response

    adapter = ExistingJudgeAdapter(
        _LegacyJudge(),
        method_name="adjudicate_scene_quality",
        decision_contract=DecisionContract.CANONICAL_METRIC,
    )

    with pytest.raises(ValueError):
        adapter.judge(_judge_request())


def test_native_canonical_status_invalid_still_requires_defects():
    class _LegacyJudge:
        def adjudicate_scene_quality(self, request):
            del request
            return {
                "status": "invalid",
                "confidence": 0.8,
                "reason": "claimed invalid without a defect",
                "defects": [],
            }

    adapter = ExistingJudgeAdapter(
        _LegacyJudge(),
        method_name="adjudicate_scene_quality",
        decision_contract=DecisionContract.CANONICAL_METRIC,
    )

    with pytest.raises(ValueError, match="requires.*defect"):
        adapter.judge(_judge_request())


def test_native_judge_status_rejects_conflicting_legacy_fields():
    class _LegacyJudge:
        def adjudicate_scene_quality(self, request):
            del request
            return {
                "status": "valid",
                "evidence_status": "insufficient",
                "verdict": "ambiguous",
                "confidence": 0.8,
                "reason": "self-contradictory",
                "missing_evidence": ["local_view"],
                "defects": [],
            }

    adapter = ExistingJudgeAdapter(
        _LegacyJudge(),
        method_name="adjudicate_scene_quality",
        decision_contract=DecisionContract.CANONICAL_METRIC,
    )

    with pytest.raises(ValueError, match="conflicts"):
        adapter.judge(_judge_request())


def test_native_judge_rejects_conflicting_role_provenance():
    class _LegacyJudge:
        def adjudicate_relation(self, request):
            del request
            return {
                "status": "valid",
                "confidence": 0.8,
                "reason": "spoofed audit identity",
                "defects": [],
                "provenance": {
                    "vlm_role": "vlm_camera_selector",
                },
            }

    adapter = ExistingJudgeAdapter(
        _LegacyJudge(),
        method_name="adjudicate_relation",
        decision_contract=DecisionContract.RELATION_BINARY,
    )

    with pytest.raises(ValueError, match=r"provenance\.vlm_role conflicts"):
        adapter.judge(_judge_request())


def test_typed_canonical_result_cannot_bypass_metric_contract():
    class _LegacyJudge:
        def adjudicate_scene_quality(self, request):
            del request
            return JudgeResult(
                status="invalid",
                confidence=0.8,
                reason="typed but missing required defect",
                defects=(),
            )

    adapter = ExistingJudgeAdapter(
        _LegacyJudge(),
        method_name="adjudicate_scene_quality",
        decision_contract=DecisionContract.CANONICAL_METRIC,
    )

    with pytest.raises(ValueError, match="requires.*defect"):
        adapter.judge(_judge_request())


def test_typed_need_more_result_still_requires_structured_request():
    class _LegacyJudge:
        def adjudicate_relation(self, request):
            del request
            return JudgeResult(
                status="need_more_evidence",
                confidence=0.4,
                reason="typed but missing request",
                evidence_request=None,
            )

    adapter = ExistingJudgeAdapter(
        _LegacyJudge(),
        method_name="adjudicate_relation",
        decision_contract=DecisionContract.RELATION_BINARY,
    )

    with pytest.raises(ValueError, match="structured evidence_request"):
        adapter.judge(_judge_request())


def test_existing_judge_adapter_accepts_functional_insufficient_compatibility():
    class _LegacyJudge:
        def adjudicate_functional_semantic(self, request):
            return {
                "evidence_status": "insufficient",
                "verdict": "insufficient_evidence",
                "canonical_verdict": "ambiguous",
                "confidence": 0.4,
                "reason": "support contact is hidden",
                "missing_evidence": ["support_contact_region"],
                "defects": [],
            }

    adapter = ExistingJudgeAdapter(
        _LegacyJudge(),
        method_name="adjudicate_functional_semantic",
        decision_contract=DecisionContract.CANONICAL_METRIC,
    )

    result = adapter.judge(_judge_request())

    assert result.status == "need_more_evidence"
    assert result.evidence_request is not None
    assert result.evidence_request.target_ids == ("a", "b")
    assert result.evidence_request.missing_observations == (
        "support_contact_region",
    )


@pytest.mark.parametrize(
    "contract",
    [
        DecisionContract.P0B_BINARY,
        DecisionContract.RELATION_BINARY,
        DecisionContract.SPATIAL_FIDELITY_BINARY,
    ],
)
def test_existing_judge_adapter_rejects_third_binary_verdict(contract):
    class _LegacyJudge:
        def adjudicate(self, request):
            return {
                "verdict": "ambiguous",
                "confidence": 0.5,
                "reason": "uncertain",
            }

    adapter = ExistingJudgeAdapter(
        _LegacyJudge(),
        method_name="adjudicate",
        decision_contract=contract,
    )

    with pytest.raises(ValueError, match="valid.*invalid"):
        adapter.judge(_judge_request())


def test_existing_judge_adapter_preserves_status_native_raw_result():
    raw = {
        "status": "valid",
        "confidence": 0.7,
        "reason": "native result",
        "defects": [],
    }

    class _Judge:
        def adjudicate(self, request):
            return raw

    adapter = ExistingJudgeAdapter(
        _Judge(),
        method_name="adjudicate",
        decision_contract=DecisionContract.RELATION_BINARY,
    )

    assert adapter.judge(_judge_request()).status == "valid"
    raw["reason"] = "mutated after return"
    assert adapter.last_raw_response["reason"] == "native result"


def test_existing_judge_adapter_replaces_stale_render_evidence_each_round():
    seen = []

    class _LegacyJudge:
        def adjudicate_relation(self, request):
            seen.append(request)
            return {
                "verdict": "valid",
                "confidence": 0.8,
                "reason": "new evidence used",
            }

    adapter = ExistingJudgeAdapter(
        _LegacyJudge(),
        method_name="adjudicate_relation",
        decision_contract=DecisionContract.RELATION_BINARY,
    )
    request = JudgeRequest(
        task="relation",
        metric="relation",
        claim_or_event={"target_ids": ["a", "b"]},
        scene_context={},
        deterministic_evidence={},
        visual_evidence=(
            {"image_path": "round-2-a.png"},
            {"path": "round-2-b.png"},
        ),
        rubric={},
        context={
            "render_evidence": ["stale-round-1.png"],
            "vlm_role": "metric_judge",
            "decision_contract": "stale_contract",
            "judge_method": "stale_method",
        },
    )

    adapter.judge(request)

    assert seen[0]["render_evidence"] == [
        "round-2-a.png",
        "round-2-b.png",
    ]
    assert seen[0]["vlm_role"] == "judge"
    assert seen[0]["decision_contract"] == "relation_binary_v1"
    assert seen[0]["judge_method"] == "adjudicate_relation"


def test_existing_metric_method_runs_through_gate_and_judge_adapter(tmp_path):
    evidence = tmp_path / "style.png"
    evidence.write_bytes(b"render")

    class _LegacyMetricJudge:
        def __init__(self):
            self.requests = []

        def adjudicate_scene_quality(self, request):
            self.requests.append(request)
            return {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 0.9,
                "reason": "style is coherent",
                "missing_evidence": [],
                "defects": [],
            }

    legacy = _LegacyMetricJudge()
    judge = ExistingJudgeAdapter(
        legacy,
        method_name="adjudicate_scene_quality",
        decision_contract=DecisionContract.CANONICAL_METRIC,
    )
    controller = VLMEvaluationController(
        judge=judge,
        renderer=_Renderer([], []),
    )
    result = controller.run(
        JudgeRequest(
            task="style_consistency",
            metric="style_consistency",
            claim_or_event={"target_ids": ["scene"]},
            scene_context={"scene_id": "scene-1"},
            deterministic_evidence={},
            visual_evidence=(str(evidence),),
            rubric={"scope": "style_consistency"},
        )
    )

    assert result.status == "valid"
    assert legacy.requests[0]["vlm_role"] == "judge"
    assert (
        legacy.requests[0]["decision_contract"]
        == "canonical_metric_v1"
    )


def test_evidence_renderer_rejects_metric_output():
    with pytest.raises(ValueError, match="must not return"):
        EvidenceRenderResult.from_value(
            {
                "visual_evidence": ["view.png"],
                "verdict": "invalid",
            }
        )


def test_existing_evidence_provider_adapter_preserves_packet():
    seen = []

    def provider(request):
        seen.append(request)
        return ["new-a.png", "new-b.png"]

    adapter = ExistingEvidenceRendererAdapter(provider)
    selection = camera_selection_result_from_value(
        {
            "selected_view_ids": ["view-1"],
            "reason": "existing selector",
        },
        request=CameraSelectionRequest(
            task="collision",
            metric="collision",
            target_ids=("a",),
            scene={},
            evidence_goal={},
            existing_visual_evidence=(),
            budget={"max_views_per_round": 1},
            candidate_views=({"id": "view-1"},),
        ),
        backend="existing",
    )
    from benchmark.visual_judge.control_loop import EvidenceRenderRequest

    rendered = adapter.render(
        EvidenceRenderRequest(
            judge_request=_judge_request(),
            selection=selection,
            evidence_goal={},
            previous_visual_evidence=("old.png",),
            evidence_round=1,
            budget={},
        )
    )

    assert rendered.visual_evidence == ("new-a.png", "new-b.png")
    assert rendered.merge_policy == "replace"
    assert rendered.backend == "existing"
    assert rendered.provenance["adapter"].endswith(
        ".ExistingEvidenceRendererAdapter"
    )
    assert seen[0]["selected_view_ids"] == ["view-1"]


def test_controlled_public_metric_method_uses_gate_and_exact_legacy_result(
    tmp_path,
):
    evidence = tmp_path / "style.png"
    evidence.write_bytes(b"style pixels")

    class _ModelBackedJudge:
        vlm_control_enabled = True

        def __init__(self):
            self.requests = []

        def adjudicate_scene_quality(self, request):
            self.requests.append(request)
            return {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 0.9,
                "reason": "style is coherent",
                "missing_evidence": [],
                "defects": [],
            }

    legacy = _ModelBackedJudge()
    wrapper = ControlledVLMJudge(
        legacy,
        control=resolve_vlm_evaluation_control(),
    )
    expected = {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "confidence": 0.9,
        "reason": "style is coherent",
        "missing_evidence": [],
        "defects": [],
    }

    actual = wrapper.adjudicate_scene_quality(
        {
            "category": "scene_quality_interfaces",
            "metric": "style_consistency",
            "scene_summary": {"scene_id": "scene-1"},
            "render_evidence": [str(evidence)],
            "judgment_scope": {"included": ["style_consistency"]},
        }
    )

    assert actual == expected
    assert len(legacy.requests) == 1
    assert legacy.requests[0]["vlm_role"] == "judge"
    stages = [
        item["stage"]
        for item in wrapper.audit_records[0]["audit"]["trace"]
    ]
    assert stages == ["evidence_gate", "judge"]


def test_controlled_public_metric_need_more_runs_provider_gate_and_judge(
    tmp_path,
):
    initial = tmp_path / "initial.png"
    repair = tmp_path / "repair.png"
    initial.write_bytes(b"initial pixels")
    repair.write_bytes(b"new repair pixels")

    class _ModelBackedJudge:
        vlm_control_enabled = True

        def __init__(self):
            self.requests = []

        def adjudicate_scene_quality(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return {
                    "evidence_status": "insufficient",
                    "verdict": "ambiguous",
                    "confidence": 0.3,
                    "reason": "need the contact region",
                    "missing_evidence": ["contact_region"],
                    "defects": [],
                }
            return {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 0.8,
                "reason": "contact is visible",
                "missing_evidence": [],
                "defects": [],
            }

    provider_calls = []

    def provider(request):
        provider_calls.append(request)
        return [str(repair)]

    legacy = _ModelBackedJudge()
    wrapper = ControlledVLMJudge(
        legacy,
        control=resolve_vlm_evaluation_control(),
        camera_provider=provider,
    )

    actual = wrapper.adjudicate_scene_quality(
        {
            "category": "scene_quality_interfaces",
            "metric": "style_consistency",
            "scene_summary": {"scene_id": "scene-1"},
            "target_object_ids": ["scene"],
            "render_evidence": [str(initial)],
            "judgment_scope": {"included": ["style_consistency"]},
        }
    )

    assert actual["verdict"] == "valid"
    assert len(legacy.requests) == 2
    assert legacy.requests[1]["render_evidence"] == [
        str(initial),
        str(repair),
    ]
    assert len(provider_calls) == 1
    stages = [
        item["stage"]
        for item in wrapper.audit_records[0]["audit"]["trace"]
    ]
    assert stages == [
        "evidence_gate",
        "judge",
        "camera_selector",
        "render",
        "evidence_gate",
        "judge",
    ]
    audit = wrapper.audit_records[0]["audit"]
    assert audit["total_images_acquired"] == 2
    assert audit["selector_backend"] == "existing"
    selector_event = audit["trace"][2]["result"]
    render_event = audit["trace"][3]["result"]
    assert selector_event["selected_view_ids"] == [
        "existing_provider_acquisition"
    ]
    assert (
        render_event["provenance"]["selected_acquisition"]
        == selector_event["selected_view_ids"][0]
    )


def test_controlled_binary_method_never_calls_judge_when_gate_blocks():
    class _ModelBackedJudge:
        vlm_control_enabled = True

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

    legacy = _ModelBackedJudge()
    wrapper = ControlledVLMJudge(
        legacy,
        control=resolve_vlm_evaluation_control(),
    )

    with pytest.raises(EvidenceControlUnresolvedError):
        wrapper.adjudicate_p0b(
            {
                "category": "p0b_structural_adjudication",
                "metric": "collision",
                "event": {"object_ids": ["a", "b"]},
                "render_evidence": ["/definitely/missing/evidence.png"],
                "detector_evidence": {},
            }
        )

    assert legacy.calls == 0
    assert wrapper.audit_records[0]["stop_reason"] == (
        "evidence_not_camera_repairable"
    )


def test_builder_controls_legacy_backend_without_model_marker():
    class _LegacyBackend:
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

    legacy = _LegacyBackend()
    wrapper = build_controlled_vlm_judge(
        legacy,
        control=resolve_vlm_evaluation_control(),
    )

    assert wrapper.strict is True
    with pytest.raises(EvidenceControlUnresolvedError):
        wrapper.adjudicate_p0b(
            {
                "metric": "collision",
                "event": {"object_ids": ["a", "b"]},
                "render_evidence": [],
            }
        )
    assert legacy.calls == 0


def test_builder_gates_generic_legacy_backend_without_model_marker():
    class _LegacyBackend:
        def __init__(self):
            self.calls = 0

        def evaluate(self, request):
            del request
            self.calls += 1
            return {
                "applicable": True,
                "score": 1.0,
                "confidence": 1.0,
                "summary": "must not run",
                "issues": [],
                "evidence": [],
            }

    legacy = _LegacyBackend()
    wrapper = build_controlled_vlm_judge(
        legacy,
        control=resolve_vlm_evaluation_control(),
    )

    result = wrapper.evaluate(
        {
            "category": "visual_quality",
            "render_evidence": [],
        }
    )

    assert result["applicable"] is False
    assert result["score"] is None
    assert legacy.calls == 0
    assert wrapper.audit_records[0]["stop_reason"] == (
        "evidence_not_camera_repairable"
    )


def test_generic_wrapper_preserves_legacy_response_without_confidence(
    tmp_path,
):
    evidence = tmp_path / "generic.png"
    evidence.write_bytes(b"generic pixels")

    class _LegacyBackend:
        def __init__(self):
            self.calls = 0

        def evaluate(self, request):
            del request
            self.calls += 1
            return {
                "applicable": False,
                "score": None,
                "summary": "not enough task-specific visual evidence",
                "issues": [],
                "evidence": [],
            }

    legacy = _LegacyBackend()
    wrapper = build_controlled_vlm_judge(
        legacy,
        control=resolve_vlm_evaluation_control(),
    )

    result = wrapper.evaluate(
        {
            "category": "visual_quality",
            "render_evidence": [str(evidence)],
        }
    )

    assert result == {
        "applicable": False,
        "score": None,
        "summary": "not enough task-specific visual evidence",
        "issues": [],
        "evidence": [],
    }
    assert legacy.calls == 1
    assert wrapper.audit_records[0]["stop_reason"] == "judge_conclusion"


def test_builder_allows_only_explicit_strict_opt_out():
    class _NonVLMCompatibilityBackend:
        def __init__(self):
            self.calls = 0

        def adjudicate_p0b(self, request):
            del request
            self.calls += 1
            return {
                "verdict": "valid",
                "confidence": 1.0,
                "reason": "explicit non-VLM compatibility path",
            }

    backend = _NonVLMCompatibilityBackend()
    wrapper = build_controlled_vlm_judge(
        backend,
        control=resolve_vlm_evaluation_control(),
        strict=False,
    )

    result = wrapper.adjudicate_p0b(
        {"metric": "collision", "render_evidence": []}
    )

    assert wrapper.strict is False
    assert result["verdict"] == "valid"
    assert backend.calls == 1


def test_controlled_oob_mixed_packet_preserves_local_metadata_and_blocks_judge(
    tmp_path,
):
    global_view = tmp_path / "global.png"
    local_view = tmp_path / "local.png"
    global_view.write_bytes(b"global pixels")
    local_view.write_bytes(b"local pixels")

    class _ModelBackedJudge:
        vlm_control_enabled = True

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

    legacy = _ModelBackedJudge()
    wrapper = ControlledVLMJudge(
        legacy,
        control=resolve_vlm_evaluation_control(),
    )

    with pytest.raises(EvidenceControlUnresolvedError):
        wrapper.adjudicate_p0b(
            {
                "category": "p0b_structural_adjudication",
                "metric": "oob",
                "event": {
                    "object_id": "chair-1",
                    "object_ids": ["chair-1"],
                },
                "render_evidence": [
                    str(global_view),
                    str(local_view),
                ],
                "local_render_evidence_metadata": [
                    {
                        "path": str(local_view),
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
                ],
                "detector_evidence": {},
            }
        )

    assert legacy.calls == 0
    gate_event = wrapper.audit_records[0]["audit"]["trace"][0]
    assert gate_event["stage"] == "evidence_gate"
    assert gate_event["images_used"] == [
        str(global_view),
        "local-oob",
    ]
    assert gate_event["result"]["ready"] is False
    assert "target_not_visible" in gate_event["result"]["reason_codes"]
    provenance = gate_event["result"]["provenance"]
    assert provenance["assessment_scope"] == "metadata_subset"
    assert provenance["metadata_evidence_count"] == 1
    assert (
        "metric_specific_visibility_and_packet_sufficiency"
        in provenance["checks_applied"]
    )


def test_controlled_provider_honors_explicit_vlm_backend_and_actual_usage(
    tmp_path,
):
    initial = tmp_path / "initial.png"
    repair = tmp_path / "repair.png"
    initial.write_bytes(b"initial")
    repair.write_bytes(b"repair")

    class _Selector:
        def __init__(self):
            self.calls = 0

        def select_camera_views(self, request):
            self.calls += 1
            return {
                "selected_view_ids": [
                    request["candidates"][0]["id"]
                ],
                "action": None,
                "reason": "matching provider selector",
            }

    class _Provider:
        camera_selector_backend = "vlm"
        policy_config = {
            "max_selector_calls": 1,
            "max_camera_actions": 2,
        }

        def __init__(self, selector):
            self.selector = selector
            self.calls = 0
            self.last_call_usage = None

        def __call__(self, request):
            self.calls += 1
            self.selector.select_camera_views(
                {
                    "candidates": [{"id": "provider-view"}],
                }
            )
            self.last_call_usage = {
                "call_id": f"provider-{self.calls}",
                "metric": request["metric"],
                "cache_hit": False,
                "evidence_refs": [str(repair)],
                "manifest_path": None,
                "selector_calls": 1,
                "camera_actions": 1,
            }
            return [str(repair)]

    class _Judge:
        vlm_control_enabled = True

        def __init__(self):
            self.calls = 0

        def adjudicate_scene_quality(self, request):
            self.calls += 1
            if self.calls == 1:
                return {
                    "evidence_status": "insufficient",
                    "verdict": "ambiguous",
                    "confidence": 0.2,
                    "reason": "need another view",
                    "missing_evidence": ["local_view"],
                    "defects": [],
                }
            return {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 0.9,
                "reason": "ready",
                "missing_evidence": [],
                "defects": [],
            }

    selector = _Selector()
    provider = _Provider(selector)
    judge = _Judge()
    wrapper = ControlledVLMJudge(
        judge,
        control=resolve_vlm_evaluation_control(
            {
                "camera_selector": {"backend": "vlm"},
                "budgets": {
                    "max_selector_calls": 3,
                    "max_camera_actions": 2,
                },
            }
        ),
        camera_provider=provider,
        camera_selector=selector,
    )

    result = wrapper.adjudicate_scene_quality(
        {
            "metric": "style_consistency",
            "scene_summary": {"scene_id": "scene"},
            "render_evidence": [str(initial)],
            "judgment_scope": {"included": ["style_consistency"]},
        }
    )

    assert result["verdict"] == "valid"
    assert provider.calls == 1
    assert selector.calls == 1
    audit = wrapper.audit_records[0]["audit"]
    assert audit["requested_selector_backend"] == "vlm"
    assert audit["selector_backend"] == "vlm"
    assert audit["selector_calls_used"] == 1
    assert audit["camera_actions_used"] == 1
    assert (
        audit["trace"][2]["result"]["provenance"][
            "max_internal_camera_actions"
        ]
        == 2
    )
    assert (
        audit["trace"][3]["result"]["provenance"][
            "actual_camera_actions"
        ]
        == 1
    )


@pytest.mark.parametrize("failure_mode", ["raise", "empty"])
def test_controlled_provider_failure_charges_observed_usage(
    tmp_path,
    failure_mode,
):
    initial = tmp_path / "initial.png"
    initial.write_bytes(b"initial")

    class _Provider:
        policy_config = {
            "max_selector_calls": 1,
            "max_camera_actions": 1,
        }

        def __init__(self):
            self.calls = 0
            self.last_call_usage = None

        def __call__(self, request):
            self.calls += 1
            self.last_call_usage = {
                "call_id": f"failed-provider-{self.calls}",
                "metric": request["metric"],
                "cache_hit": False,
                "evidence_refs": [],
                "manifest_path": None,
                "selector_calls": 1,
                "camera_actions": 1,
            }
            if failure_mode == "empty":
                return []
            raise RuntimeError("render failed after camera work")

    class _Judge:
        def __init__(self):
            self.calls = 0

        def adjudicate_scene_quality(self, request):
            del request
            self.calls += 1
            return {
                "evidence_status": "insufficient",
                "verdict": "ambiguous",
                "confidence": 0.2,
                "reason": "need another view",
                "missing_evidence": ["local_view"],
                "defects": [],
            }

    provider = _Provider()
    judge = _Judge()
    wrapper = ControlledVLMJudge(
        judge,
        control=resolve_vlm_evaluation_control(),
        camera_provider=provider,
    )

    result = wrapper.adjudicate_scene_quality(
        {
            "metric": "style_consistency",
            "scene_summary": {"scene_id": "scene"},
            "render_evidence": [str(initial)],
            "judgment_scope": {"included": ["style_consistency"]},
        }
    )

    assert result["verdict"] == "ambiguous"
    assert judge.calls == 1
    assert provider.calls == 1
    audit = wrapper.audit_records[0]["audit"]
    assert audit["selector_calls_used"] == 1
    assert audit["camera_actions_used"] == 1
    render_event = audit["trace"][3]
    assert render_event["status"] == "failed"
    assert render_event["observed_internal_selector_calls"] == 1
    assert render_event["observed_camera_actions"] == 1
    assert (
        render_event["provenance"]["provider_usage"]["call_id"]
        == "failed-provider-1"
    )
    assert wrapper.audit_records[0]["stop_reason"] == "render_failed"


def test_controlled_provider_rejects_unbound_explicit_vlm_backend(
    tmp_path,
):
    evidence = tmp_path / "initial.png"
    evidence.write_bytes(b"initial")

    class _Selector:
        def select_camera_views(self, request):
            raise AssertionError(request)

    class _Judge:
        vlm_control_enabled = True

        def __init__(self):
            self.calls = 0

        def adjudicate_scene_quality(self, request):
            self.calls += 1
            raise AssertionError(request)

    judge = _Judge()
    wrapper = ControlledVLMJudge(
        judge,
        control=resolve_vlm_evaluation_control(
            {"camera_selector": {"backend": "vlm"}}
        ),
        camera_provider=lambda request: [str(evidence)],
        camera_selector=_Selector(),
    )

    with pytest.raises(ValueError, match="incompatible"):
        wrapper.adjudicate_scene_quality(
            {
                "metric": "style_consistency",
                "scene_summary": {"scene_id": "scene"},
                "render_evidence": [str(evidence)],
            }
        )
    assert judge.calls == 0


def test_controlled_initial_provider_usage_is_charged_before_binary_judge(
    tmp_path,
):
    evidence = tmp_path / "initial.png"
    evidence.write_bytes(b"initial")

    class _Provider:
        policy_config = {
            "max_selector_calls": 2,
            "max_camera_actions": 1,
        }
        last_call_usage = {
            "call_id": "initial-provider-call",
            "metric": "collision",
            "cache_hit": False,
            "evidence_refs": [],
            "manifest_path": None,
            "selector_calls": 2,
            "camera_actions": 1,
        }

        def __call__(self, request):
            raise AssertionError(request)

    provider = _Provider()
    provider.last_call_usage["evidence_refs"] = [str(evidence)]

    class _Judge:
        vlm_control_enabled = True

        def __init__(self):
            self.calls = 0

        def adjudicate_p0b(self, request):
            self.calls += 1
            return {
                "verdict": "valid",
                "confidence": 1.0,
                "reason": "must not be called",
            }

    judge = _Judge()
    wrapper = ControlledVLMJudge(
        judge,
        control=resolve_vlm_evaluation_control(
            {
                "camera_selector": {"backend": "existing"},
                "budgets": {
                    "max_selector_calls": 1,
                    "max_camera_actions": 1,
                },
            }
        ),
        camera_provider=provider,
    )

    with pytest.raises(EvidenceControlUnresolvedError) as exc:
        wrapper.adjudicate_p0b(
            {
                "metric": "collision",
                "event": {"object_ids": ["a", "b"]},
                "render_evidence": [str(evidence)],
            }
        )

    assert exc.value.result.stop_reason == "max_selector_calls_exhausted"
    assert judge.calls == 0
    audit = wrapper.audit_records[0]["audit"]
    assert audit["selector_calls_used"] == 2
    assert audit["camera_actions_used"] == 1
    assert (
        audit["initial_camera_usage"]["call_id"]
        == "initial-provider-call"
    )


def test_controlled_composite_provider_cannot_return_two_independent_views(
    tmp_path,
):
    initial = tmp_path / "initial.png"
    repair_a = tmp_path / "repair-a.png"
    repair_b = tmp_path / "repair-b.png"
    for path, content in (
        (initial, b"initial"),
        (repair_a, b"repair-a"),
        (repair_b, b"repair-b"),
    ):
        path.write_bytes(content)

    class _Judge:
        vlm_control_enabled = True

        def __init__(self):
            self.calls = 0

        def adjudicate_scene_quality(self, request):
            self.calls += 1
            return {
                "evidence_status": "insufficient",
                "verdict": "ambiguous",
                "confidence": 0.2,
                "reason": "need another view",
                "missing_evidence": ["local_view"],
                "defects": [],
            }

    provider_calls = []

    def provider(request):
        provider_calls.append(request)
        return [str(repair_a), str(repair_b)]

    judge = _Judge()
    wrapper = ControlledVLMJudge(
        judge,
        control=resolve_vlm_evaluation_control(
            {
                "budgets": {
                    "max_views_per_round": 1,
                    "max_total_images": 6,
                }
            }
        ),
        camera_provider=provider,
    )

    result = wrapper.adjudicate_scene_quality(
        {
            "metric": "style_consistency",
            "scene_summary": {"scene_id": "scene"},
            "render_evidence": [str(initial)],
            "judgment_scope": {"included": ["style_consistency"]},
        }
    )

    assert result["verdict"] == "ambiguous"
    assert judge.calls == 1
    assert len(provider_calls) == 1
    audit_record = wrapper.audit_records[0]
    assert audit_record["status"] == "unresolved"
    assert (
        audit_record["stop_reason"]
        == "renderer_followup_contract_invalid"
    )
    assert [
        event["stage"] for event in audit_record["audit"]["trace"]
    ] == [
        "evidence_gate",
        "judge",
        "camera_selector",
        "render",
        "evidence_gate",
    ]
