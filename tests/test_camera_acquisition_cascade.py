from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark.visual_judge.adapters.active_camera import (
    ActiveVLMCameraSelector,
)
from benchmark.visual_judge.adapters.deterministic_camera import (
    DeterministicCameraRepairSolver,
    DeterministicLocalCameraSelector,
)
from benchmark.visual_judge.control_config import (
    resolve_vlm_evaluation_control,
)
from benchmark.visual_judge.evidence_gate import (
    DeterministicEvidenceGate,
)
from benchmark.visual_judge.interfaces import (
    EvidenceRenderFailure,
    EvidenceGateResult,
    JudgeRequest,
)
from benchmark.visual_judge.interfaces.camera import (
    TrustedCameraCandidateBank,
)
from benchmark.visual_judge.orchestration.controller import (
    VLMEvaluationController,
)


def _request() -> JudgeRequest:
    return JudgeRequest(
        task="collision",
        metric="collision",
        claim_or_event={"object_ids": ["a", "b"]},
        scene_context={"objects": [{"id": "a"}, {"id": "b"}]},
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=("initial.png",),
        rubric={"scope": "collision"},
    )


def _gate(*, ready: bool) -> EvidenceGateResult:
    return EvidenceGateResult(
        ready=ready,
        camera_repairable=False,
        reason_codes=(
            ("evidence_ready",)
            if ready
            else ("blank_render",)
        ),
        deficiencies=(
            ()
            if ready
            else (
                {
                    "code": "blank_render",
                    "repairability": "rerender",
                },
            )
        ),
    )


def _valid() -> dict:
    return {
        "status": "valid",
        "confidence": 0.9,
        "reason": "the event is visible",
        "defects": [],
    }


def _need_more() -> dict:
    return {
        "status": "need_more_evidence",
        "confidence": 0.2,
        "reason": "the contact is hidden",
        "defects": [],
        "evidence_request": {
            "target_ids": ["a", "b"],
            "missing_observations": ["contact_surface_visible"],
            "view_goal": "show the contact surface",
        },
    }


def _need_more_for_metric(metric: str) -> dict:
    observations = {
        "object_pairing_consistency": ["joint_visibility"],
        "style_consistency": ["global_context_preserved"],
        "scale_consistency": ["target_visible"],
        "functional_semantic_fidelity": ["group_context_visible"],
        "oob": ["architecture_plane_visible"],
        "support": ["support_contact_region"],
    }.get(metric, ["contact_surface_visible"])
    result = _need_more()
    result["evidence_request"]["missing_observations"] = observations
    if metric == "style_consistency":
        result["evidence_request"]["target_ids"] = ["scene"]
    return result


class _Gate:
    def __init__(self, results, calls):
        self.results = list(results)
        self.calls = calls

    def check(self, request):
        self.calls.append("gate")
        if len(self.results) > 1:
            result = self.results.pop(0)
        else:
            result = self.results[0]
        return result


class _Judge:
    def __init__(self, results, calls):
        self.results = list(results)
        self.calls = calls

    def judge(self, request):
        self.calls.append("judge")
        return deepcopy(self.results.pop(0))


class _Selector:
    def __init__(self, name, results, calls):
        self.backend = name
        self.results = list(results)
        self.calls = calls
        self.requests = []

    def select(self, request):
        self.calls.append(self.backend)
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)


class _Renderer:
    backend = "test_renderer"

    def __init__(self, results, calls):
        self.results = list(results)
        self.calls = calls
        self.requests = []

    def render(self, request):
        self.calls.append("render")
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)


def _selected(view_id: str) -> dict:
    return {
        "outcome": "selected",
        "selected_view_ids": [view_id],
        "reason": f"select {view_id}",
    }


def _no_feasible() -> dict:
    return {
        "outcome": "no_feasible_candidate",
        "attempted_candidate_ids": ["det-view"],
        "rejected_candidates": [
            {
                "candidate_id": "det-view",
                "reason_codes": ["target_not_visible"],
            }
        ],
        "reason_codes": ["no_feasible_candidate"],
        "reason": "no deterministic candidate can see both targets",
    }


def _rendered(name: str) -> dict:
    return {
        "visual_evidence": [f"{name}.png"],
        "merge_policy": "append",
        "provenance": {
            "preview_render_count": 0,
            "full_render_count": 1,
            "render_gpu_time_seconds": 0.25,
        },
    }


def _control(policy: str, **total_overrides):
    total = {
        "max_evidence_rounds": 2,
        "max_total_images": 6,
        "max_selector_calls": 2,
        "max_camera_actions": 2,
        **total_overrides,
    }
    return resolve_vlm_evaluation_control(
        {
            "camera_acquisition": {
                "policy": policy,
                "total": total,
            }
        }
    )


def _run(
    *,
    policy: str,
    gates,
    judge_results=(),
    deterministic_results=(),
    vlm_results=(),
    render_results=(),
    total_overrides=None,
):
    calls = []
    # Cascade tests start from an integrity-valid packet whose Judge requests
    # a metric-scoped repair unless a test supplies an explicit Judge script.
    judge_results = tuple(judge_results) or (_need_more(),)
    deterministic = _Selector(
        "deterministic",
        deterministic_results,
        calls,
    )
    vlm = _Selector("vlm", vlm_results, calls)
    renderer = _Renderer(render_results, calls)
    controller = VLMEvaluationController(
        judge=_Judge(judge_results, calls),
        renderer=renderer,
        deterministic_camera_selector=deterministic,
        vlm_camera_selector=vlm,
        evidence_gate=_Gate(gates, calls),
        control=_control(policy, **(total_overrides or {})),
    )
    result = controller.run(
        _request(),
        candidate_views=(
            {"id": "det-view"},
            {"id": "vlm-view"},
        ),
    )
    return result, calls, deterministic, vlm, renderer


def test_fixed_policy_insufficient_never_calls_selector() -> None:
    result, calls, deterministic, vlm, _ = _run(
        policy="fixed",
        gates=[_gate(ready=True)],
    )

    assert result.stop_reason == "fixed_views_insufficient"
    assert calls == ["gate", "judge"]
    assert not deterministic.requests
    assert not vlm.requests


def test_deterministic_only_never_calls_vlm() -> None:
    result, calls, _, vlm, _ = _run(
        policy="deterministic_only",
        gates=[_gate(ready=True), _gate(ready=True)],
        judge_results=[_need_more(), _valid()],
        deterministic_results=[_selected("det-view")],
        render_results=[_rendered("det")],
    )

    assert result.status == "valid"
    assert calls == [
        "gate", "judge", "deterministic", "render", "gate", "judge"
    ]
    assert not vlm.requests


def test_vlm_only_never_calls_deterministic() -> None:
    result, calls, deterministic, _, _ = _run(
        policy="vlm_only",
        gates=[_gate(ready=True), _gate(ready=True)],
        judge_results=[_need_more(), _valid()],
        vlm_results=[_selected("vlm-view")],
        render_results=[_rendered("vlm")],
    )

    assert result.status == "valid"
    assert calls == [
        "gate", "judge", "vlm", "render", "gate", "judge"
    ]
    assert not deterministic.requests


def test_vlm_only_without_explicit_vlm_selector_fails_closed() -> None:
    calls = []
    controller = VLMEvaluationController(
        judge=_Judge([_need_more()], calls),
        renderer=_Renderer([], calls),
        evidence_gate=_Gate([_gate(ready=True)], calls),
        control=_control("vlm_only"),
    )

    result = controller.run(_request())

    assert controller.vlm_camera_selector is None
    assert result.stop_reason == "camera_selector_unavailable"
    assert calls == ["gate", "judge"]


def test_vlm_selector_dispatch_without_model_inference_is_counted_separately():
    calls = []

    class _RawVLMSelector:
        def __init__(self):
            self.calls = 0

        def select_camera_views(self, payload):
            self.calls += 1
            raise AssertionError(payload)

    raw = _RawVLMSelector()
    control = resolve_vlm_evaluation_control(
        {
            "camera_acquisition": {
                "policy": "vlm_only",
                "vlm": {"selection_mode": "candidate_only"},
            }
        }
    )
    controller = VLMEvaluationController(
        judge=_Judge([_need_more(), _valid()], calls),
        renderer=_Renderer([], calls),
        vlm_camera_selector=raw,
        evidence_gate=_Gate([_gate(ready=True)], calls),
        control=control,
    )

    result = controller.run(_request(), candidate_views=())

    assert result.stop_reason == (
        "vlm_no_feasible_candidate_forced_choice"
    )
    assert result.status == "valid"
    assert raw.calls == 0
    telemetry = result.audit["experiment_telemetry"]
    assert telemetry["vlm_selector_dispatches"] == 1
    assert telemetry["vlm_selector_calls"] == 0


def test_raw_vlm_selector_is_wrapped_with_configured_candidate_mode() -> None:
    calls = []

    class _RawVLMSelector:
        def __init__(self) -> None:
            self.payloads = []

        def select_camera_views(self, payload):
            calls.append("vlm_model")
            self.payloads.append(payload)
            return {
                "selected_view_ids": ["vlm-view"],
                "reason": "trusted candidate is the best repair",
            }

    raw = _RawVLMSelector()
    control = resolve_vlm_evaluation_control(
        {
            "camera_acquisition": {
                "policy": "vlm_only",
                "vlm": {"selection_mode": "candidate_only"},
                "total": {
                    "max_evidence_rounds": 2,
                    "max_total_images": 6,
                    "max_selector_calls": 2,
                    "max_camera_actions": 2,
                },
            }
        }
    )
    controller = VLMEvaluationController(
        judge=_Judge([_need_more(), _valid()], calls),
        renderer=_Renderer([_rendered("vlm")], calls),
        vlm_camera_selector=raw,
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=control,
    )

    result = controller.run(
        _request(),
        candidate_views=({"id": "vlm-view"},),
    )

    assert result.status == "valid"
    assert calls == [
        "gate", "judge", "vlm_model", "render", "gate", "judge"
    ]
    assert raw.payloads[0]["selection_mode"] == "candidate_only"
    assert raw.payloads[0]["vlm_role"] == "vlm_camera_selector"


def test_deterministic_capability_gap_preserves_candidates_for_vlm() -> None:
    calls = []
    payloads = []
    need_interaction_side = _need_more()
    need_interaction_side["evidence_request"][
        "missing_observations"
    ] = ["interaction_side_visible"]
    need_interaction_side["evidence_request"][
        "view_goal"
    ] = "show the usable interaction side"

    def choose_candidate(payload):
        calls.append("vlm_model")
        payloads.append(payload)
        return {
            "selected_view_ids": ["vlm-view"],
            "reason": "use the trusted candidate to inspect contact",
        }

    controller = VLMEvaluationController(
        judge=_Judge([need_interaction_side, _valid()], calls),
        renderer=_Renderer([_rendered("vlm")], calls),
        deterministic_camera_selector=DeterministicLocalCameraSelector(),
        vlm_camera_selector=ActiveVLMCameraSelector(
            choose_candidate,
            selection_mode="candidate_only",
        ),
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=resolve_vlm_evaluation_control(
            {
                "camera_acquisition": {
                    "policy": "deterministic_then_vlm",
                    "vlm": {"selection_mode": "candidate_only"},
                }
            }
        ),
    )

    result = controller.run(
        JudgeRequest(
            task="functional_semantic_fidelity",
            metric="functional_semantic_fidelity",
            claim_or_event={"object_ids": ["a", "b"]},
            scene_context={
                "objects": [{"id": "a"}, {"id": "b"}]
            },
            deterministic_evidence={"status": "unresolved"},
            visual_evidence=("initial.png",),
            rubric={"scope": "functional_semantic_fidelity"},
        ),
        candidate_views=({"id": "vlm-view"},),
    )

    assert result.status == "valid"
    assert calls == [
        "gate",
        "judge",
        "vlm_model",
        "render",
        "gate",
        "judge",
    ]
    assert payloads[0]["attempted_candidate_ids"] == []
    assert [
        candidate["id"] for candidate in payloads[0]["candidate_views"]
    ] == ["vlm-view"]
    assert payloads[0]["selection_mode"] == "candidate_only"
    assert not payloads[0].get("trusted_repair_plans")
    assert (
        payloads[0]["deterministic_rejected_candidates"][0][
            "reason_codes"
        ]
        == ["semantic_selection_required"]
    )


def test_default_existing_backend_uses_local_deterministic_stage_with_vlm_di():
    class _RawVLMSelector:
        def select_camera_views(self, payload):
            raise AssertionError(payload)

    controller = VLMEvaluationController(
        judge=_Judge([], []),
        renderer=_Renderer([], []),
        vlm_camera_selector=_RawVLMSelector(),
        control=resolve_vlm_evaluation_control(),
    )

    assert isinstance(
        controller.deterministic_camera_selector,
        DeterministicLocalCameraSelector,
    )
    assert isinstance(
        controller.vlm_camera_selector,
        ActiveVLMCameraSelector,
    )
    assert (
        controller.effective_camera_acquisition_policy
        == "deterministic_then_vlm"
    )


def test_validated_freeform_pose_runs_through_controller_and_gate() -> None:
    calls = []

    def propose_pose(payload):
        calls.append("vlm_model")
        assert payload["selection_mode"] == "freeform_pose"
        return {
            "camera_proposal": {
                "location": [1.0, 2.0, 3.0],
                "target": [0.0, 0.0, 0.0],
                "lens_mm": 50.0,
            },
            "reason": "the validated pose separates the targets",
        }

    class _PoseValidator:
        def validate(self, proposal, request):
            calls.append("pose_validator")
            assert request.scene["objects"]
            return {
                "valid": True,
                "pose": proposal,
                "checks": {
                    "camera_scene_boundary_feasibility": True,
                    "frustum_validation": True,
                    "collision_avoidance": True,
                    "target_visibility_prediction": True,
                    "pose_diversity_validation": True,
                },
            }

    selector = ActiveVLMCameraSelector(
        propose_pose,
        selection_mode="freeform_pose",
        pose_validator=_PoseValidator(),
        allow_freeform_pose=True,
    )
    control = resolve_vlm_evaluation_control(
        {
            "camera_selector": {"allow_freeform_pose": True},
            "camera_acquisition": {
                "policy": "vlm_only",
                "vlm": {"selection_mode": "freeform_pose"},
            },
        }
    )
    controller = VLMEvaluationController(
        judge=_Judge([_need_more(), _valid()], calls),
        renderer=_Renderer([_rendered("freeform")], calls),
        vlm_camera_selector=selector,
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=control,
    )

    result = controller.run(_request())

    assert result.status == "valid"
    assert calls == [
        "gate",
        "judge",
        "vlm_model",
        "pose_validator",
        "render",
        "gate",
        "judge",
    ]


def test_deterministic_render_ready_does_not_call_vlm() -> None:
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True), _gate(ready=True)],
        judge_results=[_need_more(), _valid()],
        deterministic_results=[_selected("det-view")],
        render_results=[_rendered("det")],
    )

    assert result.status == "valid"
    assert not vlm.requests
    assert calls[-2:] == ["gate", "judge"]


def test_no_feasible_candidate_escalates_without_extra_gate() -> None:
    result, calls, deterministic, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True), _gate(ready=True)],
        judge_results=[_need_more(), _valid()],
        deterministic_results=[_no_feasible()],
        vlm_results=[_selected("vlm-view")],
        render_results=[_rendered("vlm")],
    )

    assert result.status == "valid"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "vlm",
        "render",
        "gate",
        "judge",
    ]
    assert len(deterministic.requests) == len(vlm.requests) == 1
    escalations = [
        event
        for event in result.audit["trace"]
        if event["stage"] == "camera_escalation"
    ]
    assert escalations[0]["reason"] == "no_feasible_candidate"
    assert escalations[0]["attempted_candidate_ids"] == ["det-view"]


def test_empty_trusted_bank_reaches_cascade_then_forced_choice() -> None:
    calls: list[str] = []

    class EmptyBank:
        def build(self, request, *, constraints):
            del request, constraints
            return TrustedCameraCandidateBank(
                candidates=(),
                rejected_candidates=(
                    {
                        "candidate_id": "candidate-a",
                        "reason_codes": ["geometry_infeasible"],
                    },
                ),
                backend="test_empty_bank",
                provenance={"candidate_count": 1},
            )

    controller = VLMEvaluationController(
        judge=_Judge([_need_more(), _valid()], calls),
        renderer=_Renderer([], calls),
        deterministic_camera_selector=_Selector(
            "deterministic",
            [
                {
                    "outcome": "no_feasible_candidate",
                    "attempted_candidate_ids": [],
                    "rejected_candidates": [],
                    "reason_codes": ["no_feasible_candidate"],
                    "reason": "the technical bank is empty",
                }
            ],
            calls,
        ),
        vlm_camera_selector=_Selector(
            "vlm",
            [
                {
                    "outcome": "no_feasible_candidate",
                    "attempted_candidate_ids": [],
                    "rejected_candidates": [],
                    "reason_codes": ["no_trusted_repair_plan"],
                    "reason": "there is no trusted VLM acquisition option",
                }
            ],
            calls,
        ),
        candidate_bank_builder=EmptyBank(),
        evidence_gate=_Gate([_gate(ready=True)], calls),
        control=_control("deterministic_then_vlm"),
    )

    result = controller.run(_request())

    assert result.status == "valid"
    assert result.stop_reason == (
        "vlm_no_feasible_candidate_forced_choice"
    )
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "vlm",
        "judge",
    ]
    escalation = next(
        event
        for event in result.audit["trace"]
        if event.get("stage") == "camera_escalation"
    )
    assert escalation["from_stage"] == "deterministic"
    assert escalation["to_stage"] == "vlm"
    assert result.audit["terminal_forced_choice"] == {
        "applied": True,
        "ambiguity_before_forcing": True,
        "trigger_stop_reason": "vlm_no_feasible_candidate",
        "original_evidence_request": {
            "target_ids": ["a", "b"],
            "missing_observations": ["contact_surface_visible"],
            "view_goal": "show the contact surface",
            "metadata": {},
        },
        "final_status": "valid",
    }


def test_deterministic_constraint_conflict_has_explicit_escalation_reason():
    conflict = _no_feasible()
    conflict["rejected_candidates"][0]["failed_constraints"] = [
        "contact_surface_visible",
        "joint_visibility",
    ]
    request = _need_more()
    request["evidence_request"]["missing_observations"] = [
        "contact_surface_visible",
        "joint_visibility",
    ]
    result, _, _, _, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True), _gate(ready=True)],
        judge_results=[request, _valid()],
        deterministic_results=[conflict],
        vlm_results=[_selected("vlm-view")],
        render_results=[_rendered("vlm")],
    )

    escalation = next(
        event
        for event in result.audit["trace"]
        if event["stage"] == "camera_escalation"
    )
    assert escalation["reason"] == "camera_constraint_conflict"


def test_repeated_judge_request_escalates_only_after_deterministic_failure() -> None:
    result, calls, _, _, renderer = _run(
        policy="deterministic_then_vlm",
        gates=[
            _gate(ready=True),
            _gate(ready=True),
            _gate(ready=True),
        ],
        judge_results=[_need_more(), _need_more(), _valid()],
        deterministic_results=[
            _selected("det-view"),
            _no_feasible(),
        ],
        vlm_results=[_selected("vlm-view")],
        render_results=[_rendered("det"), _rendered("vlm")],
        total_overrides={"max_selector_calls": 3},
    )

    assert result.status == "valid"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "render",
        "gate",
        "judge",
        "deterministic",
        "vlm",
        "render",
        "gate",
        "judge",
    ]
    assert len(renderer.requests) == 2
    escalation = next(
        event
        for event in result.audit["trace"]
        if event["stage"] == "camera_escalation"
    )
    assert escalation["reason"] == "no_feasible_candidate"
    assert "evidence_gate_deficiencies" not in escalation
    assert result.audit["evaluation"] == {
        "case_id": None,
        "task": "collision",
        "metric": "collision",
        "final_status": "valid",
        "final_confidence": 0.9,
        "deterministic_outcome": "unresolved",
        "evidence_recovery_outcome": "recovered",
        "final_evidence_request": None,
    }
    telemetry = result.audit["experiment_telemetry"]
    assert telemetry["selected_view_count"] == 2
    selector_events = [
        event
        for event in telemetry["events"]
        if event["kind"] == "camera_selection"
    ]
    assert selector_events[0]["evidence_round"] == 1
    assert selector_events[0]["episode_index"] == 1
    assert selector_events[0]["selector_backend"] == "deterministic"
    assert selector_events[2]["selection_mode"] == "candidate_only"
    gate_events = [
        event
        for event in telemetry["events"]
        if event["kind"] == "evidence_gate"
    ]
    assert gate_events[0]["deficiencies"] == []


def test_unchanged_deterministic_render_does_not_escalate_to_vlm() -> None:
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[
            _gate(ready=True),
            _gate(ready=True),
        ],
        judge_results=[],
        deterministic_results=[_selected("det-view")],
        render_results=[
            {
                "visual_evidence": ["initial.png"],
                "merge_policy": "replace",
            },
        ],
    )

    assert result.status == "unresolved"
    assert result.stop_reason == "evidence_packet_unchanged"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "render",
        "gate",
    ]
    assert not vlm.requests
    assert not any(
        event["stage"] == "camera_escalation"
        for event in result.audit["trace"]
    )


def test_unchanged_vlm_render_stops_without_returning_to_deterministic():
    result, calls, deterministic, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True), _gate(ready=True)],
        deterministic_results=[_no_feasible()],
        vlm_results=[_selected("vlm-view")],
        render_results=[
            {
                "visual_evidence": ["initial.png"],
                "merge_policy": "replace",
            }
        ],
    )

    assert result.stop_reason == "evidence_packet_unchanged"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "vlm",
        "render",
        "gate",
    ]
    assert len(deterministic.requests) == 1
    assert len(vlm.requests) == 1


def test_replacement_candidate_bank_keeps_attempt_history_and_escalates():
    calls = []

    class _RawVLM:
        def __init__(self):
            self.payloads = []

        def select_camera_views(self, payload):
            calls.append("vlm")
            self.payloads.append(deepcopy(payload))
            return {
                "selected_view_ids": ["vlm-view"],
                "reason": "select from the replacement candidate bank",
            }

    raw_vlm = _RawVLM()
    replacement_failure = _no_feasible()
    replacement_failure["attempted_candidate_ids"] = ["vlm-view"]
    replacement_failure["rejected_candidates"][0][
        "candidate_id"
    ] = "vlm-view"
    controller = VLMEvaluationController(
        judge=_Judge([_need_more(), _need_more(), _valid()], calls),
        renderer=_Renderer(
            [
                {
                    **_rendered("det"),
                    "next_candidate_views": [{"id": "vlm-view"}],
                    "replaces_candidate_views": True,
                },
                _rendered("vlm"),
            ],
            calls,
        ),
        deterministic_camera_selector=_Selector(
            "deterministic",
            [_selected("det-view"), replacement_failure],
            calls,
        ),
        vlm_camera_selector=raw_vlm,
        evidence_gate=_Gate(
            [
                _gate(ready=True),
                _gate(ready=True),
                _gate(ready=True),
            ],
            calls,
        ),
        control=resolve_vlm_evaluation_control(
            {
                "camera_acquisition": {
                    "policy": "deterministic_then_vlm",
                    "vlm": {"selection_mode": "candidate_only"},
                    "total": {
                        "max_evidence_rounds": 2,
                        "max_total_images": 6,
                        "max_selector_calls": 3,
                        "max_camera_actions": 2,
                    },
                }
            }
        ),
    )

    result = controller.run(
        _request(),
        candidate_views=({"id": "det-view"},),
    )

    assert result.status == "valid"
    assert raw_vlm.payloads[0]["attempted_candidate_ids"] == [
        "det-view"
    ]
    assert [
        candidate["id"]
        for candidate in raw_vlm.payloads[0]["candidate_views"]
    ] == ["vlm-view"]


def test_selector_exception_is_not_normal_escalation() -> None:
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        deterministic_results=[RuntimeError("selector broke")],
    )

    assert result.stop_reason == "camera_selector_failed"
    assert calls == ["gate", "judge", "deterministic"]
    assert not vlm.requests
    assert not any(
        event["stage"] == "camera_escalation"
        for event in result.audit["trace"]
    )
    telemetry = result.audit["experiment_telemetry"]
    assert telemetry["deterministic_selector_calls"] == 1
    assert telemetry["vlm_selector_calls"] == 0
    failure = next(
        event
        for event in telemetry["events"]
        if event["kind"] == "camera_selection"
    )
    assert failure["outcome"] == "selector_exception"


def test_invalid_selector_response_is_counted_without_vlm_escalation() -> None:
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        deterministic_results=[_selected("unknown-view")],
    )

    assert result.stop_reason == "camera_selector_failed"
    assert calls == ["gate", "judge", "deterministic"]
    assert not vlm.requests
    telemetry = result.audit["experiment_telemetry"]
    assert telemetry["deterministic_selector_calls"] == 1
    event = next(
        item
        for item in telemetry["events"]
        if item["kind"] == "camera_selection"
    )
    assert event["outcome"] == "invalid_selector_response"


def test_inactive_failed_constraint_is_invalid_response_not_escalation():
    invalid = _no_feasible()
    invalid["rejected_candidates"][0]["failed_constraints"] = [
        "support_chain_visible"
    ]
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        deterministic_results=[invalid],
    )

    assert result.stop_reason == "camera_selector_failed"
    assert calls == ["gate", "judge", "deterministic"]
    assert not vlm.requests
    assert not any(
        event["stage"] == "camera_escalation"
        for event in result.audit["trace"]
    )
    telemetry = result.audit["experiment_telemetry"]
    assert telemetry["deterministic_selector_calls"] == 1
    assert next(
        item["outcome"]
        for item in telemetry["events"]
        if item["kind"] == "camera_selection"
    ) == "invalid_selector_response"


@pytest.mark.parametrize("decision_key", ["verdict", "score"])
def test_nested_selector_metric_decision_is_rejected(decision_key) -> None:
    selected = _selected("det-view")
    selected["provenance"] = {
        "nested": {decision_key: "invalid" if decision_key == "verdict" else 0.5}
    }
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        deterministic_results=[selected],
    )

    assert result.stop_reason == "camera_selector_failed"
    assert calls == ["gate", "judge", "deterministic"]
    assert not vlm.requests
    assert not any(
        event["stage"] == "camera_escalation"
        for event in result.audit["trace"]
    )


@pytest.mark.parametrize(
    "invalid_duration",
    [-0.1, "slow"],
)
def test_invalid_selector_telemetry_provenance_is_contract_failure(
    invalid_duration,
) -> None:
    selected = _selected("det-view")
    selected["provenance"] = {
        "candidate_generation_time_seconds": invalid_duration
    }
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        deterministic_results=[selected],
    )

    assert result.stop_reason == "camera_selector_failed"
    assert calls == ["gate", "judge", "deterministic"]
    assert not vlm.requests
    event = next(
        item
        for item in result.audit["experiment_telemetry"]["events"]
        if item["kind"] == "camera_selection"
    )
    assert event["outcome"] == "invalid_selector_response"


def test_nested_nonfinite_selector_provenance_is_safe_contract_failure():
    selected = _selected("det-view")
    selected["provenance"] = {"nested": {"value": float("inf")}}

    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        deterministic_results=[selected],
    )

    assert result.stop_reason == "camera_selector_failed"
    assert calls == ["gate", "judge", "deterministic"]
    assert not vlm.requests
    event = next(
        item
        for item in result.audit["experiment_telemetry"]["events"]
        if item["kind"] == "camera_selection"
    )
    assert event["outcome"] == "invalid_selector_response"


def test_render_failure_is_not_normal_escalation() -> None:
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        deterministic_results=[_selected("det-view")],
        render_results=[RuntimeError("render broke")],
    )

    assert result.stop_reason == "render_failed"
    assert calls == ["gate", "judge", "deterministic", "render"]
    assert not vlm.requests


def test_failed_render_cost_and_rejected_gate_are_recorded() -> None:
    failure = EvidenceRenderFailure(
        "partial render failed",
        visual_evidence=("partial.png",),
        provenance={
            "preview_render_count": 2,
            "full_render_count": 1,
            "render_gpu_time_seconds": 0.75,
        },
    )
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True), _gate(ready=True)],
        deterministic_results=[_selected("det-view")],
        render_results=[failure],
    )

    telemetry = result.audit["experiment_telemetry"]
    assert result.stop_reason == "render_failed"
    assert calls == ["gate", "judge", "deterministic", "render", "gate"]
    assert not vlm.requests
    assert telemetry["preview_render_count"] == 2
    assert telemetry["full_render_count"] == 1
    assert telemetry["render_gpu_time_seconds"] == 0.75
    assert any(
        event["kind"] == "evidence_gate"
        and event["phase"] == "post_render_rejected"
        for event in telemetry["events"]
    )


def test_nonfinite_render_cost_is_structured_render_failure() -> None:
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True), _gate(ready=True)],
        deterministic_results=[_selected("det-view")],
        render_results=[
            {
                "visual_evidence": ["partial.png"],
                "merge_policy": "append",
                "provenance": {
                    "render_gpu_time_seconds": float("nan"),
                },
            }
        ],
    )

    assert result.stop_reason == "render_failed"
    assert calls == ["gate", "judge", "deterministic", "render", "gate"]
    assert not vlm.requests
    assert (
        result.audit["experiment_telemetry"][
            "render_gpu_time_seconds"
        ]
        == 0.0
    )


def test_repeated_judge_request_forces_choice_at_round_budget() -> None:
    result, calls, _, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True), _gate(ready=True)],
        judge_results=[_need_more(), _need_more(), _valid()],
        deterministic_results=[_selected("det-view")],
        render_results=[_rendered("det")],
        total_overrides={"max_evidence_rounds": 1},
    )

    assert result.status == "valid"
    assert (
        result.stop_reason
        == "max_evidence_rounds_exhausted_forced_choice"
    )
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "render",
        "gate",
        "judge",
        "judge",
    ]
    assert not vlm.requests


def test_unsubstantiated_conflict_reason_does_not_forge_conflict_escalation():
    deterministic = _no_feasible()
    deterministic["reason_codes"] = ["camera_constraint_conflict"]
    result, _, _, _, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        judge_results=[_need_more(), _valid()],
        deterministic_results=[deterministic],
        vlm_results=[_no_feasible()],
    )

    escalation = next(
        event
        for event in result.audit["trace"]
        if event["stage"] == "camera_escalation"
    )
    assert escalation["reason"] == "no_feasible_candidate"


def test_invalid_post_render_manifest_does_not_escalate_to_vlm(
    tmp_path,
) -> None:
    global_path = tmp_path / "global.png"
    local_path = tmp_path / "local.png"
    manifest_path = tmp_path / "render-manifest.json"
    global_path.write_bytes(b"global")
    local_path.write_bytes(b"local")
    manifest_path.write_text("{invalid", encoding="utf-8")
    calls = []

    class _InitialThenDeterministicGate:
        def __init__(self):
            self.calls = 0
            self.delegate = DeterministicEvidenceGate()

        def check(self, request):
            calls.append("gate")
            self.calls += 1
            if self.calls == 1:
                return _gate(ready=True)
            return self.delegate.check(request)

    request = JudgeRequest(
        task="style_consistency",
        metric="style_consistency",
        claim_or_event={},
        scene_context={"objects": [{"id": "a"}]},
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=(
            {
                "path": str(global_path),
                "role": "metric_global",
                "view_id": "global-anchor",
            },
        ),
        rubric={"scope": "style_consistency"},
    )
    vlm = _Selector("vlm", [], calls)
    controller = VLMEvaluationController(
        judge=_Judge(
            [_need_more_for_metric("style_consistency")],
            calls,
        ),
        renderer=_Renderer(
            [
                {
                    "visual_evidence": [
                        {
                            "path": str(local_path),
                            "role": "metric_local",
                            "view_id": "local-repair",
                        }
                    ],
                    "merge_policy": "replace",
                    "manifest_path": str(manifest_path),
                }
            ],
            calls,
        ),
        deterministic_camera_selector=_Selector(
            "deterministic",
            [_selected("det-view")],
            calls,
        ),
        vlm_camera_selector=vlm,
        evidence_gate=_InitialThenDeterministicGate(),
        control=_control("deterministic_then_vlm"),
    )

    result = controller.run(
        request,
        candidate_views=({"id": "det-view"},),
    )

    assert result.stop_reason == "manifest_failure"
    assert calls == ["gate", "judge", "deterministic", "render", "gate"]
    assert not vlm.requests
    assert not any(
        event["stage"] == "camera_escalation"
        for event in result.audit["trace"]
    )


def test_corrupt_post_render_evidence_does_not_escalate_to_vlm(
    tmp_path,
) -> None:
    global_path = tmp_path / "global.png"
    local_path = tmp_path / "local.png"
    global_path.write_bytes(b"global")
    local_path.write_bytes(b"local")
    calls = []

    class _InitialThenDeterministicGate:
        def __init__(self):
            self.calls = 0
            self.delegate = DeterministicEvidenceGate()

        def check(self, request):
            calls.append("gate")
            self.calls += 1
            if self.calls == 1:
                return _gate(ready=True)
            return self.delegate.check(request)

    request = JudgeRequest(
        task="style_consistency",
        metric="style_consistency",
        claim_or_event={},
        scene_context={"objects": [{"id": "a"}]},
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=(
            {
                "path": str(global_path),
                "role": "metric_global",
                "view_id": "global-anchor",
                "redundant_view": False,
            },
        ),
        rubric={"scope": "style_consistency"},
    )
    vlm = _Selector("vlm", [], calls)
    controller = VLMEvaluationController(
        judge=_Judge(
            [_need_more_for_metric("style_consistency")],
            calls,
        ),
        renderer=_Renderer(
            [
                {
                    "visual_evidence": [
                        {
                            "path": str(local_path),
                            "role": "metric_local",
                            "view_id": "local-repair",
                            "render_status": "corrupt",
                            "redundant_view": False,
                        }
                    ],
                    "merge_policy": "replace",
                }
            ],
            calls,
        ),
        deterministic_camera_selector=_Selector(
            "deterministic",
            [_selected("det-view")],
            calls,
        ),
        vlm_camera_selector=vlm,
        evidence_gate=_InitialThenDeterministicGate(),
        control=_control("deterministic_then_vlm"),
    )

    result = controller.run(
        request,
        candidate_views=({"id": "det-view"},),
    )

    assert result.stop_reason == "corrupt_evidence"
    assert calls == ["gate", "judge", "deterministic", "render", "gate"]
    assert not vlm.requests


def test_vlm_render_always_runs_gate_again() -> None:
    result, calls, _, _, _ = _run(
        policy="vlm_only",
        gates=[_gate(ready=True), _gate(ready=True)],
        judge_results=[_need_more(), _valid()],
        vlm_results=[_selected("vlm-view")],
        render_results=[_rendered("vlm")],
    )

    assert result.status == "valid"
    assert calls == [
        "gate", "judge", "vlm", "render", "gate", "judge"
    ]


def test_judge_need_more_starts_new_episode_from_deterministic() -> None:
    result, calls, deterministic, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True), _gate(ready=True)],
        judge_results=[_need_more(), _valid()],
        deterministic_results=[_selected("det-view")],
        render_results=[_rendered("det")],
    )

    assert result.status == "valid"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "render",
        "gate",
        "judge",
    ]
    assert deterministic.requests[0].context[
        "camera_acquisition_stage"
    ] == "deterministic"
    assert not vlm.requests
    state = result.audit["camera_acquisition"]["state"]
    assert state["episode_index"] == 1


def test_judge_cannot_expand_camera_targets_beyond_original_scene_scope():
    request = _need_more()
    request["evidence_request"]["target_ids"] = ["invented"]
    result, calls, deterministic, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        judge_results=[request],
    )

    assert result.stop_reason == "camera_constraint_contract_invalid"
    assert calls == ["gate", "judge"]
    assert not deterministic.requests
    assert not vlm.requests
    failure = result.audit["trace"][-1]
    assert failure["failure_kind"] == "scene_contract_failure"
    assert "unknown target IDs" in failure["error"]


def test_group_scoped_judge_repair_preserves_context_and_subset_focus() -> None:
    calls = []
    request = JudgeRequest(
        task="functional_consistency",
        metric="functional_consistency",
        claim_or_event={"group_id": "work"},
        scene_context={
            "objects": [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        },
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=("initial.png",),
        rubric={"scope": "functional_consistency"},
        context={
            "group_scope": {
                "group_id": "work",
                "member_ids": ["a", "b"],
                "target_bounds": {
                    "min": [0.0, 0.0, 0.0],
                    "max": [2.0, 1.0, 1.0],
                },
                "focus_center": [1.0, 0.5, 0.5],
                "extent": [2.0, 1.0, 1.0],
            }
        },
    )
    need_more = {
        "status": "need_more_evidence",
        "confidence": 0.2,
        "reason": "The interaction side is not visible.",
        "defects": [],
        "evidence_request": {
            "target_ids": ["a"],
            "missing_observations": [
                "interaction_side_visible"
            ],
            "view_goal": "show the interaction side",
        },
    }
    deterministic = _Selector(
        "deterministic",
        [_selected("det-view")],
        calls,
    )
    controller = VLMEvaluationController(
        judge=_Judge([need_more, _valid()], calls),
        renderer=_Renderer([_rendered("det")], calls),
        deterministic_camera_selector=deterministic,
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=_control("deterministic_then_vlm"),
    )

    result = controller.run(
        request,
        candidate_views=({"id": "det-view"},),
    )

    assert result.status == "valid"
    assert deterministic.requests[0].target_ids == ("a",)
    assert deterministic.requests[0].constraints["target_ids"] == ["a"]
    assert deterministic.requests[0].evidence_goal["target_ids"] == ["a"]
    assert deterministic.requests[0].context["group_scope"][
        "member_ids"
    ] == ["a", "b"]
    assert deterministic.requests[0].context["target_bounds"] == {
        "min": [0.0, 0.0, 0.0],
        "max": [2.0, 1.0, 1.0],
    }
    assert deterministic.requests[0].context["focus_center"] == [
        1.0,
        0.5,
        0.5,
    ]
    assert result.audit["focus_target_ids"] == ["a"]
    assert result.audit["authoritative_group_member_ids"] == [
        "a",
        "b",
    ]
    assert deterministic.requests[0].context["target_extent"] == [
        2.0,
        1.0,
        1.0,
    ]


def test_group_scoped_judge_may_request_full_group_focus() -> None:
    calls = []
    request = JudgeRequest(
        task="scale_consistency",
        metric="scale_consistency",
        claim_or_event={"group_id": "work"},
        scene_context={
            "objects": [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        },
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=("initial.png",),
        rubric={"scope": "scale_consistency"},
        context={
            "group_scope": {
                "group_id": "work",
                "member_ids": ["a", "b"],
            }
        },
    )
    need_more = {
        "status": "need_more_evidence",
        "confidence": 0.2,
        "reason": "Both objects need a clearer view.",
        "defects": [],
        "evidence_request": {
            "target_ids": ["a", "b"],
            "missing_observations": ["joint_visibility"],
            "view_goal": "show the complete group together",
        },
    }
    deterministic = _Selector(
        "deterministic",
        [_selected("det-view")],
        calls,
    )
    controller = VLMEvaluationController(
        judge=_Judge([need_more, _valid()], calls),
        renderer=_Renderer([_rendered("det")], calls),
        deterministic_camera_selector=deterministic,
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=_control("deterministic_then_vlm"),
    )

    result = controller.run(
        request,
        candidate_views=({"id": "det-view"},),
    )

    assert result.status == "valid"
    selection_request = deterministic.requests[0]
    assert selection_request.target_ids == ("a", "b")
    assert selection_request.constraints["target_ids"] == ["a", "b"]
    assert selection_request.context["group_scope"]["member_ids"] == [
        "a",
        "b",
    ]


def test_group_scoped_judge_cannot_request_another_group_member() -> None:
    calls = []
    request = JudgeRequest(
        task="functional_consistency",
        metric="functional_consistency",
        claim_or_event={"group_id": "work"},
        scene_context={
            "objects": [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        },
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=("initial.png",),
        rubric={"scope": "functional_consistency"},
        context={
            "group_scope": {
                "group_id": "work",
                "member_ids": ["a", "b"],
            }
        },
    )
    need_more = {
        "status": "need_more_evidence",
        "confidence": 0.2,
        "reason": "Need a different target.",
        "defects": [],
        "evidence_request": {
            "target_ids": ["c"],
            "missing_observations": ["target_visible"],
            "view_goal": "show c",
        },
    }
    deterministic = _Selector("deterministic", [], calls)
    controller = VLMEvaluationController(
        judge=_Judge([need_more], calls),
        renderer=_Renderer([], calls),
        deterministic_camera_selector=deterministic,
        evidence_gate=_Gate([_gate(ready=True)], calls),
        control=_control("deterministic_then_vlm"),
    )

    result = controller.run(request)

    assert result.status == "unresolved"
    assert (
        result.stop_reason
        == "judge_evidence_request_outside_group_scope"
    )
    assert calls == ["gate", "judge"]
    assert not deterministic.requests


@pytest.mark.parametrize("unobservable_id", ["scene-1", "asset-a", "Chair"])
def test_scene_operational_ids_and_names_cannot_authorize_targets(
    unobservable_id,
) -> None:
    calls = []
    request = JudgeRequest(
        task="collision",
        metric="collision",
        claim_or_event={"object_ids": ["a", "b"]},
        scene_context={
            "scene_id": "scene-1",
            "objects": [
                {"id": "a", "asset_id": "asset-a", "name": "Chair"},
                {"id": "b"},
            ],
        },
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=("initial.png",),
        rubric={"scope": "collision"},
        context={"request_id": "request-1"},
    )
    evidence_request = _need_more()
    evidence_request["evidence_request"]["target_ids"] = [
        unobservable_id
    ]
    deterministic = _Selector("deterministic", [], calls)
    vlm = _Selector("vlm", [], calls)
    controller = VLMEvaluationController(
        judge=_Judge([evidence_request], calls),
        renderer=_Renderer([], calls),
        deterministic_camera_selector=deterministic,
        vlm_camera_selector=vlm,
        evidence_gate=_Gate([_gate(ready=True)], calls),
        control=_control("deterministic_then_vlm"),
    )

    result = controller.run(request)

    assert result.stop_reason == "camera_constraint_contract_invalid"
    assert calls == ["gate", "judge"]
    assert not deterministic.requests
    assert not vlm.requests


def test_local_replace_render_preserves_required_global_anchor(tmp_path) -> None:
    global_path = tmp_path / "global.png"
    local_path = tmp_path / "local.png"
    global_path.write_bytes(b"global")
    local_path.write_bytes(b"local")
    calls = []
    request = JudgeRequest(
        task="style_consistency",
        metric="style_consistency",
        claim_or_event={},
        scene_context={"objects": [{"id": "a"}]},
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=(
            {
                "path": str(global_path),
                "role": "metric_global",
                "view_id": "global-anchor",
            },
        ),
        rubric={"scope": "style_consistency"},
    )
    renderer = _Renderer(
        [
            {
                "visual_evidence": [
                    {
                        "path": str(local_path),
                        "role": "metric_local",
                        "view_id": "local-repair",
                    }
                ],
                "merge_policy": "replace",
            }
        ],
        calls,
    )
    controller = VLMEvaluationController(
        judge=_Judge(
            [_need_more_for_metric("style_consistency"), _valid()],
            calls,
        ),
        renderer=renderer,
        deterministic_camera_selector=_Selector(
            "deterministic",
            [_selected("det-view")],
            calls,
        ),
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=_control("deterministic_only"),
    )

    result = controller.run(
        request,
        candidate_views=({"id": "det-view"},),
    )

    assert result.status == "valid"
    assert [item["view_id"] for item in result.visual_evidence] == [
        "global-anchor",
        "local-repair",
    ]
    assert renderer.requests[0].context["camera_constraints"][
        "require_global_anchor"
    ] is True


def test_same_path_overwrite_is_compared_against_pre_render_bytes(
    tmp_path,
) -> None:
    evidence_path = tmp_path / "repair.png"
    evidence_path.write_bytes(b"before")
    calls = []

    class _OverwriteRenderer:
        backend = "overwrite_renderer"

        def render(self, request):
            calls.append("render")
            evidence_path.write_bytes(b"after")
            return {
                "visual_evidence": [
                    {
                        "path": str(evidence_path),
                        "role": "metric_local",
                        "view_id": "repair-view",
                    }
                ],
                "merge_policy": "replace",
            }

    request = JudgeRequest(
        task="collision",
        metric="collision",
        claim_or_event={"object_ids": ["a", "b"]},
        scene_context={"objects": [{"id": "a"}, {"id": "b"}]},
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=(
            {
                "path": str(evidence_path),
                "role": "metric_local",
                "view_id": "repair-view",
            },
        ),
        rubric={"scope": "collision"},
    )
    controller = VLMEvaluationController(
        judge=_Judge([_need_more(), _valid()], calls),
        renderer=_OverwriteRenderer(),
        deterministic_camera_selector=_Selector(
            "deterministic",
            [_selected("det-view")],
            calls,
        ),
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=_control("deterministic_only"),
    )

    result = controller.run(
        request,
        candidate_views=({"id": "det-view"},),
    )

    assert result.status == "valid"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "render",
        "gate",
        "judge",
    ]


def test_metadata_only_repair_does_not_change_evidence_integrity(
    tmp_path,
) -> None:
    evidence_path = tmp_path / "repair.png"
    evidence_path.write_bytes(b"same pixels")
    calls = []
    request = JudgeRequest(
        task="collision",
        metric="collision",
        claim_or_event={"object_ids": ["a", "b"]},
        scene_context={"objects": [{"id": "a"}, {"id": "b"}]},
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=(
            {
                "path": str(evidence_path),
                "role": "metric_local",
                "view_id": "repair-view",
            },
        ),
        rubric={"scope": "collision"},
    )
    controller = VLMEvaluationController(
        judge=_Judge([_need_more(), _valid()], calls),
        renderer=_Renderer(
            [
                {
                    "visual_evidence": [
                        {
                            "path": str(evidence_path),
                            "role": "metric_local",
                            "view_id": "repair-view",
                            "visibility": {
                                "target_visible": True,
                                "jointly_visible": True,
                            },
                        }
                    ],
                    "merge_policy": "append",
                }
            ],
            calls,
        ),
        deterministic_camera_selector=_Selector(
            "deterministic",
            [_selected("det-view")],
            calls,
        ),
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=_control("deterministic_only"),
    )

    result = controller.run(
        request,
        candidate_views=({"id": "det-view"},),
    )

    assert result.status == "unresolved"
    assert result.stop_reason == "evidence_packet_unchanged"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "render",
        "gate",
    ]


def test_vlm_stage_failure_does_not_return_to_deterministic() -> None:
    result, calls, deterministic, vlm, _ = _run(
        policy="deterministic_then_vlm",
        gates=[_gate(ready=True)],
        judge_results=[_need_more(), _valid()],
        deterministic_results=[_no_feasible()],
        vlm_results=[
            {
                "outcome": "no_feasible_candidate",
                "attempted_candidate_ids": ["vlm-view"],
                "rejected_candidates": [
                    {
                        "candidate_id": "vlm-view",
                        "reason_codes": ["target_not_visible"],
                    }
                ],
                "reason_codes": ["no_trusted_repair_plan"],
                "reason": "VLM stage has no verifiable repair",
            }
        ],
    )

    assert result.stop_reason == (
        "vlm_no_feasible_candidate_forced_choice"
    )
    assert result.status == "valid"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "vlm",
        "judge",
    ]
    assert len(deterministic.requests) == len(vlm.requests) == 1
    forced = result.audit["terminal_forced_choice"]
    assert forced["applied"] is True
    assert forced["ambiguity_before_forcing"] is True
    assert forced["trigger_stop_reason"] == (
        "vlm_no_feasible_candidate"
    )
    assert forced["original_evidence_request"]["target_ids"] == [
        "a",
        "b",
    ]


def test_total_selector_budget_is_shared_across_judge_requests() -> None:
    result, calls, deterministic, _, _ = _run(
        policy="deterministic_only",
        gates=[_gate(ready=True), _gate(ready=True)],
        judge_results=[_need_more(), _need_more(), _valid()],
        deterministic_results=[_selected("det-view")],
        render_results=[_rendered("det")],
        total_overrides={"max_selector_calls": 1},
    )

    assert result.status == "valid"
    assert (
        result.stop_reason
        == "max_selector_calls_exhausted_forced_choice"
    )
    assert len(deterministic.requests) == 1
    telemetry = result.audit["experiment_telemetry"]
    assert telemetry["deterministic_selector_calls"] == 1
    assert telemetry["vlm_selector_calls"] == 0
    assert telemetry["judge_calls"] == 3
    assert calls.count("render") == 1


def test_no_conflict_uses_candidate_only_instead_of_inventing_repair_plan():
    calls = []
    deterministic = _Selector(
        "deterministic",
        [_no_feasible()],
        calls,
    )

    def choose_candidate(payload):
        calls.append("vlm_model")
        assert payload["selection_mode"] == "candidate_only"
        assert not payload.get("trusted_repair_plans")
        return {
            "selected_view_ids": ["vlm-view"],
            "reason": "select one trusted candidate preview",
        }

    vlm = ActiveVLMCameraSelector(
        choose_candidate,
        selection_mode="repair_plan",
        repair_solver=DeterministicCameraRepairSolver(),
    )
    controller = VLMEvaluationController(
        judge=_Judge([_need_more(), _valid()], calls),
        renderer=_Renderer([_rendered("repair-plan")], calls),
        deterministic_camera_selector=deterministic,
        vlm_camera_selector=vlm,
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=_control("deterministic_then_vlm"),
    )

    result = controller.run(
        _request(),
        candidate_views=(
            {
                "id": "det-view",
                "target_ids": ["a", "b"],
                "location": [2.0, 0.0, 2.0],
                "target": [0.0, 0.0, 0.0],
                "lens_mm": 50.0,
                "feasible": True,
                "visibility": {
                    "target_visible": True,
                    "jointly_visible": True,
                    "projected_coverage": 0.2,
                },
            },
            {
                "id": "vlm-view",
                "target_ids": ["a", "b"],
                "location": [-2.0, 0.0, 2.0],
                "target": [0.0, 0.0, 0.0],
                "lens_mm": 50.0,
                "feasible": True,
                "visibility": {
                    "target_visible": True,
                    "jointly_visible": True,
                    "projected_coverage": 0.2,
                },
            },
        ),
    )

    assert result.status == "valid"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "vlm_model",
        "render",
        "gate",
        "judge",
    ]
    selection = next(
        event["result"]
        for event in result.audit["trace"]
        if event.get("selection_stage") == "vlm"
    )
    assert selection["outcome"] == "selected"
    assert selection["selected_view_ids"] == ["vlm-view"]
    assert selection["selected_plan_id"] is None


def test_conflict_repair_plan_is_reachable_from_controller(tmp_path) -> None:
    global_path = tmp_path / "global.png"
    local_path = tmp_path / "local.png"
    global_path.write_bytes(b"global")
    local_path.write_bytes(b"local")
    calls = []
    deterministic_failure = {
        "outcome": "no_feasible_candidate",
        "attempted_candidate_ids": ["candidate-a"],
        "rejected_candidates": [
            {
                "candidate_id": "candidate-a",
                "reason_codes": ["constraint_conflict"],
                "failed_constraints": [
                    "joint_visibility",
                    "global_context_preserved",
                ],
            }
        ],
        "reason_codes": ["no_feasible_candidate"],
        "reason": "local coverage conflicts with global framing",
    }

    def choose_plan(payload):
        calls.append("vlm_model")
        assert payload["trusted_repair_plans"]
        plan = payload["trusted_repair_plans"][0]
        assert plan["relaxed_constraints"] == [
            "global_context_preserved"
        ]
        return {
            "selected_plan_id": plan["plan_id"],
            "reason": "retain the packet anchor and relax only local framing",
        }

    evidence_request = _need_more_for_metric(
        "object_pairing_consistency"
    )
    evidence_request["evidence_request"]["missing_observations"] = [
        "joint_visibility",
        "global_context_preserved",
    ]
    controller = VLMEvaluationController(
        judge=_Judge([evidence_request, _valid()], calls),
        renderer=_Renderer(
            [
                {
                    "visual_evidence": [
                        {
                            "path": str(local_path),
                            "role": "metric_local",
                            "view_id": "candidate-a",
                        }
                    ],
                    "merge_policy": "replace",
                }
            ],
            calls,
        ),
        deterministic_camera_selector=_Selector(
            "deterministic",
            [deterministic_failure],
            calls,
        ),
        vlm_camera_selector=ActiveVLMCameraSelector(
            choose_plan,
            selection_mode="repair_plan",
            repair_solver=DeterministicCameraRepairSolver(),
        ),
        evidence_gate=_Gate(
            [_gate(ready=True), _gate(ready=True)],
            calls,
        ),
        control=_control("deterministic_then_vlm"),
    )
    request = JudgeRequest(
        task="object_pairing_consistency",
        metric="object_pairing_consistency",
        claim_or_event={"object_ids": ["a", "b"]},
        scene_context={"objects": [{"id": "a"}, {"id": "b"}]},
        deterministic_evidence={"status": "unresolved"},
        visual_evidence=(
            {
                "path": str(global_path),
                "role": "metric_global",
                "view_id": "global-anchor",
            },
        ),
        rubric={"scope": "object_pairing_consistency"},
    )

    result = controller.run(
        request,
        candidate_views=(
            {
                "id": "candidate-a",
                "view_family": "metric_local",
                "target_ids": ["a", "b"],
                "location": [2.0, 0.0, 2.0],
                "target": [0.0, 0.0, 0.0],
                "lens_mm": 50.0,
                "feasible": True,
                "visibility": {
                    "target_visible": True,
                    "jointly_visible": True,
                    "projected_coverage": 0.2,
                },
            },
        ),
    )

    assert result.status == "valid"
    assert calls == [
        "gate",
        "judge",
        "deterministic",
        "vlm_model",
        "render",
        "gate",
        "judge",
    ]
    assert [
        item["view_id"] for item in result.visual_evidence
    ] == ["global-anchor", "candidate-a"]
    escalation = next(
        event
        for event in result.audit["trace"]
        if event["stage"] == "camera_escalation"
    )
    assert escalation["reason"] == "camera_constraint_conflict"
    telemetry = result.audit["experiment_telemetry"]
    assert telemetry["vlm_selector_calls"] == 1
    assert telemetry["vlm_selector_dispatches"] == 1
