"""Opt-in final decisions, through the real provider/controller and Judge."""
from copy import deepcopy
import json
import pytest

from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.best_effort_terminal import POLICY
from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
from benchmark.visual_judge.acquisition_outcome import AcquisitionExhausted
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.evidence_gap_v2 import EvidenceGapError, FALLBACK_POLICY
from benchmark.evaluator.scene_quality.adaptive_acceptance import terminalize_adaptive_scope
from test_evidence_adaptive_judgement import Model, picture, request, response
from test_r12_placement_scope_control import occluded_scope, scope_request
from test_postrun_policy_boundary import v2_control


def judge(model):
    return OpenAICompatibleVLMJudge(model, evidence_resolution_policy=FALLBACK_POLICY,
                                   terminal_evidence_policy=POLICY)


def test_controller_exhaustion_requires_new_decision_with_same_evidence(picture):
    acquired = []
    def provider(req):
        acquired.append(deepcopy(req))
        raise AcquisitionExhausted("trusted_candidate_bank_empty")
    model = Model([occluded_scope(), occluded_scope(), response()])
    wrapper = ControlledVLMJudge(judge(model), control=v2_control(), camera_provider=provider)
    result = wrapper.adjudicate_scene_quality(scope_request(picture))
    assert acquired and len(model.calls) == 3
    assert result["verdict"] == "valid"
    audit = result["request_metadata"]["response_schema_validation"]
    assert audit["repair_retry_count"] == 0 and audit["decision_retry_count"] == 1
    assert audit["recovered"]
    assert model.calls[-1][:2] == model.calls[-2]
    assert len(model.calls[-1]) == 4
    resolution = result["evidence_resolution"]
    assert resolution["images_used"] == [picture]
    assert resolution["terminal_evidence_policy"] == POLICY
    assert resolution["visual_observation_complete"] is False
    assert resolution["model_judgement_completed"] is True


@pytest.mark.parametrize("ending", ["valid", "invalid"])
def test_binary_terminal_native_response_is_model_owned(picture, ending):
    req = request("collision", [picture])
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    first = {"status": "need_more_evidence", "confidence": 0.3,
             "reason": "Need an additional contact view.", "defects": [],
             "evidence_request": {"target_ids": ["chair"],
                 "missing_observations": ["contact_surface_visible"],
                 "view_goal": "Inspect contact.", "metadata": {}}}
    final = {"status": ending, "confidence": 0.6,
             "reason": "Best-supported inference from retained contact evidence.",
             "defects": [], "evidence_request": None}
    model = Model([first, final])
    result = judge(model)._adjudicate_p0b_raw(req)
    assert result["verdict"] == ending and len(model.calls) == 2
    assert result["request_metadata"]["response_schema_validation"]["decision_retry_count"] == 1
    record = terminalize_adaptive_scope(
        {"status": "evaluated", "score": float(ending == "valid"), "judgement": result}, phase="test")
    assert record["score"] == float(ending == "valid")


def test_persistent_abstention_is_bounded_response_fault(picture):
    req = scope_request(picture)
    req["adaptive_terminal"] = True
    model = Model(occluded_scope())
    with pytest.raises(ResponseSchemaRepairError) as caught:
        judge(model)._adjudicate_scene_quality_raw(req)
    assert len(model.calls) == 2
    assert caught.value.schema_audit["decision_retry_count"] == 1


@pytest.mark.parametrize("fault", [ConnectionError("offline"), PermissionError("denied"),
                                  FileNotFoundError("missing"), RuntimeError("bug")])
def test_real_provider_fault_does_not_trigger_final_decision(picture, fault):
    def provider(req):
        raise fault
    model = Model([occluded_scope(), response()])
    wrapper = ControlledVLMJudge(judge(model), control=v2_control(), camera_provider=provider)
    with pytest.raises(Exception) as caught:
        wrapper.adjudicate_scene_quality(scope_request(picture))
    assert not isinstance(caught.value, EvidenceGapError)
    assert len(model.calls) == 1


def test_empty_evidence_does_not_default_valid():
    model = Model(response())
    req = request("style_consistency", appearance=False)
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    with pytest.raises(EvidenceGapError):
        judge(model)._adjudicate_scene_quality_raw(req)
    assert not model.calls

@pytest.mark.parametrize("metric", ["functional_consistency", "semantic_placement_consistency"])
def test_typed_obligations_are_delivered_and_resolved_by_new_decision(picture, metric):
    from test_evidence_adaptive_judgement import complete_required_rows
    req = request(metric, [picture])
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    if metric == "functional_consistency":
        field = "functional_check_results"
        required = "required_functional_checks"
        req[required] = [{"check_id": "use", "check_type": "usability",
                          "target_ids": ["chair"], "required_observations": ["target_visible"]}]
        rows = [{"check_id": "use", "target_ids": ["chair"], "observation_status": "missing",
                 "conclusion": "unresolved", "reason": "A closer view would help."}]
    else:
        field = "placement_check_results"
        required = "required_placement_checks"
        req[required] = [{"check_id": "zone", "check_type": "scene_zone", "subject_id": "chair",
                         "context_ids": [], "owner_stage": "scene_global", "group_ids": [],
                         "required_observations": ["target_visible", "global_context_preserved"]}]
        rows = [{"check_id": "zone", "subject_id": "chair", "context_ids": [],
                 "observation_status": "missing", "conclusion": "unresolved",
                 "reason": "A closer view would help."}]
    first = occluded_scope()
    first["missing_evidence"] = ["target_visible"]
    first["evidence_request"].update(target_ids=["chair"], missing_observations=["target_visible"])
    first[field] = rows
    model = Model([first, complete_required_rows])
    result = judge(model)._adjudicate_scene_quality_raw(req)
    assert len(model.calls) == 2 and result["verdict"] == "valid"
    assert [row["check_id"] for row in result[field]] == [req[required][0]["check_id"]]
    assert all(row["observation_status"] == "inferred_under_budget" for row in result[field])
    for messages in model.calls:
        context = json.loads(messages[1]["content"][0]["text"].split("\n", 1)[1])
        assert [check["check_id"] for check in context[required]] == [req[required][0]["check_id"]]


def test_terminal_format_repair_still_cannot_reverse_verdict(picture):
    req = request("scale_consistency", [picture])
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    broken = response()
    broken["confidence"] = "not a number"
    model = Model([broken, response("invalid")])
    with pytest.raises(ResponseSchemaRepairError) as caught:
        judge(model)._adjudicate_scene_quality_raw(req)
    assert len(model.calls) == 2
    assert not caught.value.schema_audit.get("decision_retry_count")


def test_new_decision_cannot_reverse_previously_resolved_typed_row():
    from benchmark.visual_judge.best_effort_terminal import preserve_resolved_rows
    first = {"functional_check_results": [{"check_id": "kept", "target_ids": ["chair"],
              "conclusion": "invalid", "observation_status": "observed"}]}
    after = deepcopy(first)
    after["functional_check_results"][0]["conclusion"] = "valid"
    with pytest.raises(ValueError, match="already resolved"):
        preserve_resolved_rows(first, after)

@pytest.mark.parametrize("metric", ["functional_consistency", "semantic_placement_consistency"])
def test_public_metric_report_completes_from_terminal_inferences(picture, metric):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    from test_evidence_adaptive_judgement import complete_required_rows
    scene = request(metric)["scene_summary"]
    scene["objects"].append({**deepcopy(scene["objects"][0]), "id": "desk", "category": "desk", "center": [3, 3, 1]})
    def answer(messages):
        value = complete_required_rows(messages)
        if len(messages) > 2:
            return value
        context = json.loads(messages[1]["content"][0]["text"].split("\n", 1)[1])
        value.update(evidence_status="insufficient", verdict="ambiguous",
                     missing_evidence=["target_visible"], reason="Another view would reduce uncertainty.",
                     evidence_request={"target_ids": ["chair"], "missing_observations": ["target_visible"],
                                       "view_goal": "Inspect target.", "metadata": {}})
        for field in ("functional_check_results", "placement_check_results"):
            for row in value.get(field, []):
                row.update(observation_status="missing", conclusion="unresolved")
        return value
    model = Model(answer)
    wrapper = ControlledVLMJudge(judge(model), control=v2_control(
        budgets={"max_evidence_rounds": 0}))
    report = evaluate_scene_quality_interfaces(
        scene, config={"enabled": True, "metrics": {name: {"enabled": name == metric}
                      for name in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={"object_groups": [{"group_id": "work", "object_ids": ["chair", "desk"]}]},
        render_evidence=[picture], vlm_judge=wrapper,
        metric_applicability={metric: {"applicability": "relevant"}})
    result = report["metrics"][metric]
    assert result["status"] == "evaluated", {k: result.get(k) for k in ("status", "reason", "failure")}
    assert result["score"] == 1.0
    assert model.calls and any(len(messages) == 4 for messages in model.calls)
