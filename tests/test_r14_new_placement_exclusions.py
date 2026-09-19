"""New findings must retain exact Function ownership through real judgement."""
from copy import deepcopy
import json
import pytest
from test_evidence_adaptive_judgement import Model, picture, response, complete_required_rows
from test_placement_audit_contract import judge_request, ownership
from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.best_effort_terminal import POLICY
from benchmark.visual_judge.evidence_gap_v2 import FALLBACK_POLICY
from benchmark.evaluator.scene_quality.placement_checks import validate_placement_check_results


def answer():
    return {**response(), "judge_originated_placement_results": [{
        "proposal_id": "new-zone", "subject_id": "chair", "context_ids": [],
        "check_type": "scene_zone", "observation_goal": "Assess the static location independently.",
        "observation_status": "inferred_under_budget", "conclusion": "excluded_function_owned",
        "reason": "The independently invalid static placement is the same final Functional event.",
        "severity": "implausible", "function_event_ref": "function-1", "same_physical_event": True}]}


def request(picture):
    req = judge_request()
    req.update(required_placement_checks=[], render_evidence=[picture],
               evidence_resolution_policy=FALLBACK_POLICY, evidence_phase="global_discovery")
    return req


def test_new_exclusion_reaches_strict_typed_validator(picture):
    model = Model(answer())
    raw = OpenAICompatibleVLMJudge(model, terminal_evidence_policy=POLICY)._adjudicate_scene_quality_raw(request(picture))
    checks = raw["judge_originated_placement_check_registrations"]
    result = validate_placement_check_results(raw, required_checks=checks, function_events=ownership())
    assert result["complete"] and len(result["excluded_function_owned_check_ids"]) == 1
    assert len(model.calls) == 1 and raw["verdict"] == "valid" and raw["defects"] == []
    assert raw["placement_check_results"][0]["function_event_ref"] == "function-1"
    assert checks[0]["source_discovery_refs"] == ["new-zone"]


@pytest.mark.parametrize("bad", ["unknown_event", "false_same_event", "wrong_subject_event", "missing_event", "unknown_subject", "unresolved", "defect", "wrong_stage"])
def test_exclusions_cannot_bypass_ownership_or_scope(picture, bad):
    value, req = answer(), request(picture)
    row = value["judge_originated_placement_results"][0]
    if bad == "unknown_event": row["function_event_ref"] = "foreign"
    elif bad == "false_same_event": row["same_physical_event"] = False
    elif bad == "wrong_subject_event":
        req["functional_ownership_ledger"]["events"][0].update(scoring_target_ids=[], affected_object_ids=[], causal_object_ids=[])
    elif bad == "missing_event": req.pop("functional_ownership_ledger")
    elif bad == "unknown_subject": row["subject_id"] = "foreign"
    elif bad == "unresolved": row.update(conclusion="unresolved", observation_status="missing")
    elif bad == "defect": value["defects"] = [{"target_ids": ["chair"], "check_id": "new-zone", "scope": "object", "reason": "Must not score a duplicate."}]
    else: req.update(evidence_phase="group_local_review", group_scope={"group_id": "g", "member_ids": ["chair"]})
    model = Model(value)
    if bad == "wrong_stage":
        # Existing Open-space transport explicitly rehomes with an audit;
        # R14 does not change that historical lane behavior.
        result = OpenAICompatibleVLMJudge(model, terminal_evidence_policy=POLICY)._adjudicate_scene_quality_raw(req)
        assert result["judge_originated_placement_check_registrations"][0]["owner_stage"] == "scene_global"
        return
    with pytest.raises(Exception):
        OpenAICompatibleVLMJudge(model, terminal_evidence_policy=POLICY)._adjudicate_scene_quality_raw(req)
    assert len(model.calls) <= 2


def test_legacy_nonrect_policy_is_unchanged(picture):
    # Open-space already supported this transport before R14.
    import inspect
    from benchmark.evaluator.scene_quality.placement_checks import normalize_judge_originated_placement_results
    if "allow_function_exclusions" not in inspect.signature(normalize_judge_originated_placement_results).parameters:
        return
    with pytest.raises(Exception):
        OpenAICompatibleVLMJudge(Model(answer()))._adjudicate_scene_quality_raw(request(picture))


def test_public_metric_report_preserves_registered_exclusion(picture, monkeypatch):
    from benchmark.evaluator.scene_quality import interfaces
    from benchmark.visual_judge.runtime import build_controlled_vlm_judge
    from test_postrun_policy_boundary import v2_control
    req = request(picture)
    # Fixture represents a preceding, already-final Function result.
    ledger = deepcopy(req["functional_ownership_ledger"])
    monkeypatch.setattr(interfaces, "_resolved_functional_ownership_for_placement", lambda *a, **kw: deepcopy(ledger))
    emitted = []
    class Planner:
        def discover_placement_evidence(self, req):
            return {"schema_version": "placement_discovery_v2", "considered_object_ids": ["chair"], "decision_authority": "none", "reason": "No prior check.", "candidates": []}
    def respond(messages):
        context = json.loads(messages[1]["content"][0]["text"].split("\n", 1)[1])
        if context.get("evidence_phase") == "global_discovery" and not emitted:
            emitted.append(True)
            return answer()
        return complete_required_rows(messages)
    wrapper = build_controlled_vlm_judge(OpenAICompatibleVLMJudge(Model(respond), terminal_evidence_policy=POLICY), control=v2_control(), camera_provider=lambda req: {"status": "available", "paths": [picture]})
    metric = "semantic_placement_consistency"
    report = interfaces.evaluate_scene_quality_interfaces(req["scene_summary"],
        config={"enabled": True, "metrics": {name: {"enabled": name == metric} for name in interfaces.SCENE_QUALITY_INTERFACE_METRICS}},
        render_evidence=[picture], functional_evidence_planner=Planner(),
        camera_evidence_provider=lambda req: {"status": "available", "paths": [picture]},
        vlm_judge=wrapper, metric_applicability={metric: {"applicability": "relevant"}})
    value = report["metrics"][metric]
    assert emitted and value["status"] == "evaluated", value.get("reason")
    assert value["score"] == 1.0
    checks = value["placement_check_ledger"]["checks"]
    assert len(checks) == 1 and checks[0]["check_conclusion"] == "excluded_function_owned"
