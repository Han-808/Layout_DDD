"""A stage mismatch is a routed obligation, never a score in the wrong owner."""
from copy import deepcopy
import json
import pytest
from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.best_effort_terminal import POLICY
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.evidence_gap_v2 import FALLBACK_POLICY
from benchmark.visual_judge.evidence_resolution import evidence_policy_scope
from benchmark.visual_judge.orchestration.controller import _register_pending_placement_check
from benchmark.visual_judge.adapters.legacy_judge import _judge_request
from benchmark.visual_judge.interfaces.judge import EvidenceRequest
from test_evidence_adaptive_judgement import Model, picture, complete_required_rows
from test_placement_proposal_handoff import global_request, proposal
from test_postrun_policy_boundary import v2_control


def finding():
    return {"evidence_status": "sufficient", "verdict": "invalid", "confidence": 0.8,
            "reason": "A significant local semantic support mismatch.", "missing_evidence": [],
            "evidence_request": None, "judge_originated_placement_results": [
                {**proposal(), "observation_status": "inferred_under_budget", "conclusion": "invalid",
                 "reason": "The support is semantically unusual.", "severity": "atypical"}],
            "defects": [{"scope": "semantically_inappropriate_support_surface",
                "target_ids": ["chair"], "check_id": proposal()["proposal_id"],
                "check_type": "support_and_height", "relation": "support_and_height",
                "category": "semantic_surface_mismatch", "attribution_mode": "unary",
                "reason": "The support is semantically unusual.", "severity": "atypical"}]}


def judge(model, enabled=True):
    return OpenAICompatibleVLMJudge(model, terminal_evidence_policy=POLICY if enabled else "bounded_abstention_v2")


def test_validated_finding_becomes_pending_not_a_global_score(picture):
    req = global_request(picture)
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    model = Model(finding())
    result = judge(model)._adjudicate_scene_quality_raw(req)
    assert len(model.calls) == 1
    assert result["verdict"] == "ambiguous" and result["defects"] == []
    assert result["evidence_request"]["metadata"]["placement_check_proposal"] == proposal()
    with evidence_policy_scope(req):
        registered, check = _register_pending_placement_check(
            _judge_request(req), raw_response=result, evidence_request=EvidenceRequest(**result["evidence_request"]))
    assert check["owner_stage"] == "group_local" and check["judge_status"] != "resolved"
    assert check["prior_stage_finding"]["prior_finding"]["conclusion"] == "invalid"
    assert check in registered.context["deferred_placement_checks"]


def test_old_policy_keeps_old_open_space_rehoming(picture):
    req = global_request(picture)
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    model = Model(finding())
    result = judge(model, enabled=False)._adjudicate_scene_quality_raw(req)
    assert len(model.calls) == 1 and result["verdict"] == "invalid"
    assert result["evidence_request"] is None


@pytest.mark.parametrize("bad", ["confidence", "unknown_subject", "bad_severity", "missing_defect", "invalid_conclusion"])
def test_routing_cannot_hide_invalid_findings(picture, bad):
    req = global_request(picture)
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    value = finding()
    if bad == "confidence":
        value["confidence"] = "bad"
    elif bad == "unknown_subject":
        value["judge_originated_placement_results"][0]["subject_id"] = "foreign"
    elif bad == "bad_severity":
        value["defects"][0]["severity"] = "made-up"
        value["judge_originated_placement_results"][0]["severity"] = "made-up"
    elif bad == "missing_defect":
        value["defects"] = []
    else:
        value["judge_originated_placement_results"][0]["conclusion"] = "valid"
    with pytest.raises(ResponseSchemaRepairError):
        judge(Model(value))._adjudicate_scene_quality_raw(req)


@pytest.mark.parametrize("verdict", ["valid", "invalid"])
def test_public_report_requires_new_judgement_from_rightful_owner(picture, verdict):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    from benchmark.visual_judge.runtime import build_controlled_vlm_judge
    metric = "semantic_placement_consistency"
    req = global_request(picture)
    seen = []
    emitted = False
    class Planner:
        def discover_placement_evidence(self, req):
            return {"schema_version": "placement_discovery_v2", "considered_object_ids": ["chair", "desk"],
                    "decision_authority": "none", "reason": "No prior local check.", "candidates": []}
    def respond(messages):
        nonlocal emitted
        context = json.loads(messages[1]["content"][0]["text"].split("\n", 1)[1])
        seen.append(deepcopy(context))
        if context.get("evidence_phase") == "global_discovery" and not emitted:
            emitted = True
            return finding()
        value = complete_required_rows(messages)
        checks = context.get("required_placement_checks") or []
        if context.get("evidence_phase") == "group_local_review" and checks and verdict == "invalid":
            value["verdict"] = "invalid"
            for row in value["placement_check_results"]:
                row["conclusion"] = "invalid"
            value["defects"] = [{**finding()["defects"][0], "check_id": c["check_id"],
                                "target_ids": [c["subject_id"]]} for c in checks]
        return value
    model = Model(respond)
    wrapper = build_controlled_vlm_judge(judge(model), control=v2_control(),
        camera_provider=lambda req: {"status": "available", "paths": [picture]})
    report = evaluate_scene_quality_interfaces(
        req["scene_summary"], config={"enabled": True, "metrics": {
            name: {"enabled": name == metric} for name in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={"object_groups": req["object_groups"]},
        render_evidence=[picture], functional_evidence_planner=Planner(),
        camera_evidence_provider=lambda req: {"status": "available", "paths": [picture]},
        vlm_judge=wrapper, metric_applicability={metric: {"applicability": "relevant"}})
    value = report["metrics"][metric]
    assert emitted and value["status"] == "evaluated", value.get("reason")
    checks = value["placement_check_ledger"]["checks"]
    assert len(checks) == 1 and checks[0]["owner_stage"] == "group_local"
    assert checks[0]["judge_status"] == "resolved"
    assert any(c.get("evidence_phase") == "group_local_review" and c.get("required_placement_checks") for c in seen)
    assert (value["score"] == 1.0) is (verdict == "valid")
