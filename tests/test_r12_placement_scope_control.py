"""Placement visibility requests are control flow, not malformed verdicts."""
from copy import deepcopy
import json
import pytest
from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
from benchmark.visual_judge.acquisition_outcome import AcquisitionExhausted
from benchmark.visual_judge.evidence_gap_v2 import EvidenceGapError, FALLBACK_POLICY
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.evaluator.scene_quality.placement_checks import validate_placement_check_results
from test_evidence_adaptive_judgement import Model, picture, response
from test_placement_proposal_handoff import global_request
from test_postrun_policy_boundary import v2_control


def occluded_scope():
    # Same response structure as the retained Fable failure; no invented check.
    return {"evidence_status": "insufficient", "verdict": "ambiguous",
            "confidence": 0.98, "reason": "Foreground geometry obscures the room context.",
            "missing_evidence": ["global_context_preserved", "group_context_visible", "target_visible"],
            "defects": [], "evidence_request": {"target_ids": ["scene"],
                "missing_observations": ["global_context_preserved", "group_context_visible", "target_visible"],
                "view_goal": "Show the complete room and contextual placement.", "metadata": {}}}


def scope_request(picture):
    req = global_request(picture)
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    return req


def test_scope_request_keeps_legacy_strict_and_does_not_certify_empty_checks():
    value = occluded_scope()
    with pytest.raises(ValueError):
        validate_placement_check_results(value, required_checks=[])
    result = validate_placement_check_results(value, required_checks=[], allow_scope_evidence_request=True)
    assert result["complete"] is False and result["scope_evidence_unresolved"] is True
    assert result["required_check_count"] == 0 and result["rows"] == []
    assert value == occluded_scope()


def test_raw_adapter_admits_first_response_without_schema_repair(picture):
    model = Model(occluded_scope())
    result = OpenAICompatibleVLMJudge(model)._adjudicate_scene_quality_raw(scope_request(picture))
    assert len(model.calls) == 1
    assert result["verdict"] == "ambiguous" and result["evidence_request"]["target_ids"] == ["scene"]
    assert result["request_metadata"]["response_schema_validation"]["attempt_count"] == 1


@pytest.mark.parametrize("ending", ["valid", "gap"])
def test_real_controller_reaches_terminal_after_occluded_scope(picture, ending):
    acquired = []
    def provider(request):
        acquired.append(deepcopy(request))
        raise AcquisitionExhausted("trusted_candidate_bank_empty")
    model = Model([occluded_scope(), response() if ending == "valid" else occluded_scope()])
    wrapper = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control(), camera_provider=provider)
    if ending == "gap":
        with pytest.raises(EvidenceGapError) as caught:
            wrapper.adjudicate_scene_quality(scope_request(picture))
        assert caught.value.audit["coverage_gap"]["score"] is None
        audit = wrapper.audit_records[-1]["evidence_resolution"]
    else:
        result = wrapper.adjudicate_scene_quality(scope_request(picture))
        assert result["verdict"] == "valid"
        audit = result["evidence_resolution"]
    assert acquired and len(model.calls) == 2
    assert audit["model_invoked"] is True and audit["images_used"] == [picture]
    context = json.loads(model.calls[-1][1]["content"][0]["text"].split("\n", 1)[1])
    assert context["adaptive_evidence"]["terminal"] is True
    assert all(len(messages) == 2 for messages in model.calls)  # No schema-repair conversation.


@pytest.mark.parametrize("bad", ["no_request", "empty_targets", "empty_observations", "no_goal", "unknown_target", "unknown_observation"])
def test_not_a_generic_invalid_response_bypass(picture, bad):
    value = occluded_scope()
    if bad == "no_request":
        value["evidence_request"] = None
    else:
        patch = {"empty_targets": {"target_ids": []}, "empty_observations": {"missing_observations": []},
                 "no_goal": {"view_goal": ""}, "unknown_target": {"target_ids": ["foreign"]},
                 "unknown_observation": {"missing_observations": ["fabricated_token"]}}[bad]
        value["evidence_request"].update(patch)
    with pytest.raises(ResponseSchemaRepairError):
        OpenAICompatibleVLMJudge(Model(value))._adjudicate_scene_quality_raw(scope_request(picture))


def test_existing_obligation_cannot_be_omitted():
    check = {"check_id": "zone", "check_type": "scene_zone", "subject_id": "chair",
             "context_ids": [], "owner_stage": "scene_global"}
    with pytest.raises(ValueError, match="placement_check_results"):
        validate_placement_check_results(occluded_scope(), required_checks=[check], allow_scope_evidence_request=True)


def test_terminal_model_failure_still_is_not_a_score(picture):
    model = Model(ConnectionError("offline service fault"))
    wrapper = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control())
    req = scope_request(picture)
    req["adaptive_terminal"] = True
    with pytest.raises(Exception) as caught:
        wrapper.adjudicate_scene_quality(req)
    assert getattr(caught.value, "failure_category", None) == "model_service_failure"


@pytest.mark.parametrize("error", [RuntimeError("bug"), FileNotFoundError("missing artifact"),
                                  PermissionError("denied"), ConnectionError("service")])
def test_provider_hard_failure_never_becomes_terminal_score(picture, error):
    def provider(request):
        raise error
    model = Model([occluded_scope(), response()])
    wrapper = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control(), camera_provider=provider)
    with pytest.raises(Exception) as caught:
        wrapper.adjudicate_scene_quality(scope_request(picture))
    assert not isinstance(caught.value, EvidenceGapError)
    assert len(model.calls) == 1


@pytest.mark.parametrize("observed", [False, True])
def test_exhaustion_wrapper_requires_successful_usage_observation(observed):
    from benchmark.visual_judge.interfaces.evidence import EvidenceRenderFailure
    from benchmark.visual_judge.evidence_gap_v2 import classify_failure
    provenance = {"provider_usage": {}} if observed else {"usage_observation_error": "missing telemetry"}
    wrapper = EvidenceRenderFailure("wrapped", provenance=provenance)
    wrapper.__cause__ = AcquisitionExhausted("trusted_candidate_bank_empty")
    result = classify_failure(wrapper, phase="acquisition")
    assert result["recoverable_acquisition"] is observed
