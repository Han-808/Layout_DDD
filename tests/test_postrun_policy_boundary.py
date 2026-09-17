"""V2 evidence policy boundaries, exercised without live inference."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from benchmark.visual_judge.evidence_gap_v2 import EvidenceGapError, FALLBACK_POLICY
from benchmark.visual_judge.evidence_resolution import (
    ADAPTIVE_METRICS, ADAPTIVE_POLICY, LEGACY_POLICY,
    adaptive_enabled, attach_resolution, bind_resolution, evidence_policy_scope,
    failure_record, policy_of, prepare_adaptive_request, resolution_of, style_policy_result,
)
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.adapters.legacy_judge import EvidenceControlUnresolvedError


class NonRectangularCameraEvidenceExhausted(RuntimeError):
    """Synthetic external provider boundary, usable in either source lane."""


def request(metric, paths=()):
    obj = {"id": "chair", "category": "chair", "center": [1, 1, 1], "size": [1, 1, 1]}
    return {
        "metric": metric, "evidence_resolution_policy": FALLBACK_POLICY,
        "render_evidence": list(paths), "selected_object_ids": ["chair"],
        "scene_summary": {"objects": [obj]},
    }


def test_explicit_policy_scope_restores_previous_mode():
    assert policy_of() == LEGACY_POLICY
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        assert policy_of() == FALLBACK_POLICY and adaptive_enabled()
        with evidence_policy_scope({"evidence_resolution_policy": ADAPTIVE_POLICY}):
            assert policy_of() == ADAPTIVE_POLICY
        assert policy_of() == FALLBACK_POLICY
    assert policy_of() == LEGACY_POLICY


@pytest.mark.parametrize("metric", sorted(ADAPTIVE_METRICS - {"scale_consistency", "object_pairing_consistency"}))
def test_no_visual_generic_judge_cannot_invent_geometry_score(metric):
    req = request(metric)
    original = deepcopy(req)
    with pytest.raises(EvidenceGapError) as caught:
        prepare_adaptive_request(req, terminal=True, trigger="bounded_bank_empty")
    assert caught.value.audit["policy"] == FALLBACK_POLICY
    assert caught.value.audit["available_images"] == []
    assert req == original


def test_style_no_appearance_policy_does_not_default_valid_in_v2():
    req = request("style_consistency")
    assert style_policy_result(req) is None
    req["evidence_resolution_policy"] = ADAPTIVE_POLICY
    old = prepare_adaptive_request(req, terminal=True, trigger="no_visual")
    assert style_policy_result(old)["verdict"] == "valid"


def test_existing_visual_is_retained_with_v2_provenance(tmp_path):
    path = tmp_path / "evidence.png"
    picture = Image.new("RGB", (20, 20), "white")
    picture.putpixel((2, 3), (0, 0, 0))
    picture.save(path)
    req = prepare_adaptive_request(request("style_consistency", [str(path)]),
                                   terminal=True, trigger="budget_exhausted")
    assert req["render_evidence"] == [str(path.resolve())]
    assert req["adaptive_evidence"]["policy"] == FALLBACK_POLICY
    result = attach_resolution({"verdict": "valid", "images_used": [str(path.resolve())]}, req)
    assert resolution_of(result)["policy"] == FALLBACK_POLICY
    assert not resolution_of(result)["visual_observation_complete"]
    record = {}
    bind_resolution(record, result)
    assert record["evidence_resolution_policy"] == FALLBACK_POLICY


@pytest.mark.parametrize("error,category,recoverable", [
    (EvidenceGapError("no_observation"), "evidence_gap", True),
    (NonRectangularCameraEvidenceExhausted("bounded"), "evidence_unavailable", True),
    (RuntimeError("no_feasible_candidate: forged prose"), "implementation_or_input_failure", False),
    (FileNotFoundError("missing"), "input_integrity_failure", False),
    (PermissionError("denied"), "permission_failure", False),
    (TimeoutError("timeout"), "model_service_failure", False),
    (ConnectionError("offline"), "model_service_failure", False),
    (OSError("renderer unavailable"), "infrastructure_failure", False),
])
def test_only_typed_normal_exhaustion_is_recoverable(error, category, recoverable):
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        record = failure_record(error, phase="acquisition")
    assert record["failure_category"] == category
    assert record["recoverable_acquisition"] is recoverable
    assert "error" not in record


@pytest.mark.parametrize("reason,recoverable", [
    ("max_total_images_exhausted", True),
    ("trusted_candidate_bank_empty", True),
    ("render_failed", False),
    ("camera_selector_failed", False),
])
def test_controller_uses_structured_stop_reason_not_generic_error_text(reason, recoverable):
    error = EvidenceControlUnresolvedError(SimpleNamespace(stop_reason=reason, audit={}))
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        record = failure_record(error, phase="acquisition")
    assert record["recoverable_acquisition"] is recoverable


def test_typed_wrapper_cannot_hide_service_cause():
    error = NonRectangularCameraEvidenceExhausted("bounded")
    error.__cause__ = ConnectionError("private connection details")
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        record = failure_record(error, phase="acquisition")
    assert record["failure_category"] == "model_service_failure"
    assert not record["recoverable_acquisition"]
    assert "private" not in repr(record)




@pytest.mark.parametrize("gate,expected", [
    ({"reason_codes": ["visual_evidence_missing"], "provenance": {"evidence_item_count": 0}}, True),
    ({"reason_codes": ["visual_evidence_missing"], "provenance": {"evidence_item_count": 1}}, False),
    ({"reason_codes": ["file_missing"], "provenance": {"evidence_item_count": 0}}, False),
    ({"reason_codes": ["image_corrupt"], "provenance": {"evidence_item_count": 1}}, False),
])
def test_empty_packet_certificate_cannot_hide_missing_or_corrupt_file(gate, expected):
    audit = {"trace": [{"stage": "evidence_gate", "result": gate}],
             "judge_request": {"visual_evidence": []}}
    error = EvidenceControlUnresolvedError(SimpleNamespace(stop_reason="evidence_missing", audit=audit))
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        classified = failure_record(error, phase="acquisition")
    assert classified["recoverable_acquisition"] is expected


class Model:
    model_id = "offline-mock"
    endpoint = "https://offline.invalid/v1"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.last_request_metadata = {}

    def chat_messages(self, messages, **kwargs):
        import json
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        return json.dumps(next(self.responses))


def visual_request(tmp_path):
    path = tmp_path / "packet.png"
    image = Image.new("RGB", (20, 20), "white")
    image.putpixel((2, 3), (0, 0, 0))
    image.save(path)
    req = request("scale_consistency", [str(path)])
    req.update(adaptive_terminal=True, evidence_phase="final", decision_mode="final",
               metric_rubric="Apply the original scale consistency rules.")
    return req


def valid_response():
    return {
        "evidence_status": "sufficient", "verdict": "valid", "confidence": 0.8,
        "reason": "Observed evidence supports this judgement.",
        "defects": [], "missing_evidence": [], "evidence_request": None,
    }


def test_real_raw_adapter_retains_terminal_ambiguity_without_forced_retry(tmp_path):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.openai_compatible import _budget_exhaustion_finalization
    req = visual_request(tmp_path)
    req["budget_exhaustion_finalization"] = {
        "required": True, "trigger_stop_reason": "max_total_images_exhausted",
    }
    reply = valid_response()
    reply.update(evidence_status="insufficient", verdict="ambiguous",
                 reason="The necessary target observation is still missing.",
                 missing_evidence=["target_visible"],
                 evidence_request={"target_ids": ["chair"], "missing_observations": ["target_visible"],
                                   "view_goal": "Inspect the unresolved target.", "metadata": {}})
    model = Model([reply])
    # Low-level request-policy integration; full runtime mode is not released.
    value = OpenAICompatibleVLMJudge(model)._adjudicate_scene_quality_raw(req)
    assert value["verdict"] == "ambiguous" and value["evidence_status"] == "insufficient"
    assert value["evidence_resolution"]["accepted"] is False
    assert value["evidence_resolution"]["policy"] == FALLBACK_POLICY
    assert len(model.calls) == 1
    assert ".forced_choice" not in model.calls[0][1]["call_type"]
    assert "never guess a binary verdict" in model.calls[0][0][0]["content"]
    assert _budget_exhaustion_finalization(req) is None


def test_real_raw_adapter_keeps_supported_visual_conclusion(tmp_path):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    model = Model([valid_response()])
    result = OpenAICompatibleVLMJudge(model)._adjudicate_scene_quality_raw(visual_request(tmp_path))
    assert result["verdict"] == "valid"
    assert result["evidence_resolution"]["accepted"]
    assert result["evidence_resolution"]["images_used"]
    assert not result["evidence_resolution"]["visual_observation_complete"]
    assert len(model.calls) == 1


def test_real_raw_adapter_v2_repair_receives_specific_diagnostic(tmp_path):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    reply = valid_response()
    reply["confidence"] = 2
    model = Model([reply, valid_response()])
    result = OpenAICompatibleVLMJudge(model)._adjudicate_scene_quality_raw(visual_request(tmp_path))
    assert result["verdict"] == "valid"
    assert len(model.calls) == 2
    feedback = model.calls[1][0][-1]["content"]
    assert "validation_diagnostic" in feedback and "confidence" in feedback


def test_real_raw_adapter_without_images_does_not_call_model():
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    model = Model([])
    req = request("style_consistency")
    req["adaptive_terminal"] = True
    with pytest.raises(EvidenceGapError):
        OpenAICompatibleVLMJudge(model)._adjudicate_scene_quality_raw(req)
    assert not model.calls


def test_real_bounded_wrapper_can_retain_normal_runtime_cause():
    error = NonRectangularCameraEvidenceExhausted("bounded")
    error.__cause__ = RuntimeError("empty list")
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        record = failure_record(error, phase="acquisition")
    assert record["failure_category"] == "evidence_unavailable"


def test_old_failure_classification_is_unchanged():
    with evidence_policy_scope({"evidence_resolution_policy": ADAPTIVE_POLICY}):
        record = failure_record(RuntimeError("legacy"), phase="acquisition")
    assert record["failure_category"] == "evidence_unavailable"


@pytest.mark.parametrize("cause,category", [
    (ValueError("bad field"), "judge_response_failure"),
    (ConnectionError("private endpoint"), "model_service_failure"),
])
def test_response_wrapper_preserves_schema_or_service_failure(cause, category):
    error = ResponseSchemaRepairError("repair failed", schema_audit={})
    error.__cause__ = cause
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        record = failure_record(error, phase="judge")
    assert record["failure_category"] == category
    assert not record["recoverable_acquisition"]


def v2_control(**overrides):
    # Candidate integration test only: the public config stays disabled until
    # metric/report integration is finished and validated.
    from dataclasses import replace
    from benchmark.visual_judge.control_config import resolve_vlm_evaluation_control
    return replace(resolve_vlm_evaluation_control(overrides), evidence_resolution_policy=FALLBACK_POLICY)


def ambiguous_response():
    value = valid_response()
    value.update(
        evidence_status="insufficient", verdict="ambiguous", confidence=0.3,
        missing_evidence=["target_visible"],
        evidence_request={"target_ids": ["chair"], "missing_observations": ["target_visible"],
                          "view_goal": "Inspect target", "metadata": {}},
    )
    return value


@pytest.mark.parametrize("metric", ["scale_consistency", "support"])
def test_controlled_terminal_v2_validates_then_records_null_gap(metric, tmp_path):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    req = visual_request(tmp_path)
    req["metric"] = metric
    req["event"] = {"object_id": "chair"}
    reply = ambiguous_response() if metric == "scale_consistency" else {
        "status": "need_more_evidence", "confidence": 0.3,
        "reason": "Contact remains unseen", "defects": [],
        "evidence_request": {"target_ids": ["chair"], "missing_observations": ["support_contact_region"],
                             "view_goal": "Inspect contact", "metadata": {}},
    }
    model = Model([reply])
    wrapper = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control())
    call = wrapper.adjudicate_p0b if metric == "support" else wrapper.adjudicate_scene_quality
    with pytest.raises(EvidenceGapError) as caught:
        call(req)
    gap = caught.value.audit["coverage_gap"]
    assert gap["score"] is None and not gap["accepted"]
    assert gap["verdict"] == "unknown" and gap["status"] == "not_evaluable"
    assert caught.value.audit["evidence_resolution"]["policy"] == FALLBACK_POLICY
    assert len(model.calls) == 1
    assert "never guess a binary verdict" in model.calls[0][0][0]["content"]
    assert policy_of() == LEGACY_POLICY


def test_controller_v2_budget_stop_never_forces_binary(tmp_path):
    from benchmark.visual_judge.orchestration.controller import VLMEvaluationController
    from test_vlm_control_loop import _Gate, _Judge, _Renderer, _gate_result, _judge_request, _need_more_result
    calls = []
    judge = _Judge([_need_more_result()], calls)
    controller = VLMEvaluationController(
        judge=judge, renderer=_Renderer(AssertionError("no rendering"), calls),
        evidence_gate=_Gate([_gate_result(ready=True)], calls),
        control=v2_control(budgets={"max_evidence_rounds": 0}),
    )
    result = controller.run(_judge_request(context={"evidence_resolution_policy": FALLBACK_POLICY}))
    assert result.status == "unresolved"
    assert result.stop_reason == "max_evidence_rounds_exhausted"
    assert calls.count("judge") == 1 and "render" not in calls
    assert result.judge_result.status == "need_more_evidence"
    assert any(row.get("outcome") == "deferred_to_evidence_gap_review" for row in result.audit["trace"])
    assert not result.audit.get("budget_exhaustion_forced_choice", {}).get("applied")


@pytest.mark.parametrize("reply", [valid_response(), ambiguous_response()])
def test_controlled_acquisition_exhaustion_uses_bounded_terminal_review(reply, tmp_path):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    model = Model([ambiguous_response(), reply])
    provider_calls = []
    def provider(req):
        provider_calls.append(req)
        raise AssertionError("zero acquisition rounds")
    wrapper = ControlledVLMJudge(
        OpenAICompatibleVLMJudge(model), control=v2_control(budgets={"max_evidence_rounds": 0}),
        camera_provider=provider,
    )
    req = visual_request(tmp_path)
    req["adaptive_terminal"] = False
    req["target_ids"] = ["chair"]
    if reply["verdict"] == "ambiguous":
        with pytest.raises(EvidenceGapError) as caught:
            wrapper.adjudicate_scene_quality(req)
        assert caught.value.audit["coverage_gap"]["score"] is None
        assert caught.value.audit["acquisition_stop_reason"] == "max_evidence_rounds_exhausted"
    else:
        result = wrapper.adjudicate_scene_quality(req)
        assert result["verdict"] == "valid"
        assert result["evidence_resolution"]["policy"] == FALLBACK_POLICY
    assert len(model.calls) == 2 and not provider_calls
    assert all(".forced_choice" not in kwargs.get("call_type", "") for _, kwargs in model.calls)


@pytest.mark.parametrize("error,category", [
    (ConnectionError("offline"), "model_service_failure"),
    (RuntimeError("no feasible candidate in error prose"), "implementation_or_input_failure"),
    (FileNotFoundError("missing"), "input_integrity_failure"),
])
def test_v2_controller_provider_fault_is_not_gap_or_terminal_guess(error, category, tmp_path):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    from benchmark.visual_judge.evidence_resolution import AdaptiveEvidenceError
    model = Model([ambiguous_response()])
    def provider(req):
        raise error
    wrapper = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control(), camera_provider=provider)
    req = visual_request(tmp_path)
    req.update(adaptive_terminal=False, target_ids=["chair"])
    with pytest.raises(AdaptiveEvidenceError) as caught:
        wrapper.adjudicate_scene_quality(req)
    assert not isinstance(caught.value, EvidenceGapError)
    assert caught.value.failure_category == category
    assert len(model.calls) == 1


@pytest.mark.parametrize("verdict", ["valid", "invalid"])
def test_controlled_v2_p0b_keeps_supported_binary_result(verdict, tmp_path):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    model = Model([{"status": verdict, "confidence": 0.9, "reason": "Contact is visible.", "defects": []}])
    wrapper = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control())
    req = visual_request(tmp_path)
    req.update(metric="support", event={"object_id": "chair"})
    result = wrapper.adjudicate_p0b(req)
    assert result["verdict"] == verdict
    assert result["evidence_resolution"]["accepted"]
    assert result["evidence_resolution"]["policy"] == FALLBACK_POLICY
    assert len(model.calls) == 1
