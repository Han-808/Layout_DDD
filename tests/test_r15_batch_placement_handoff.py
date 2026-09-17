"""Multiple out-of-stage findings are obligations, not erased or scored."""
from copy import deepcopy
import json
import pytest
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.evidence_gap_v2 import FALLBACK_POLICY
from benchmark.visual_judge.evidence_resolution import evidence_policy_scope
from benchmark.visual_judge.orchestration.controller import _register_pending_placement_check
from benchmark.visual_judge.adapters.legacy_judge import _judge_request
from benchmark.visual_judge.interfaces.judge import EvidenceRequest
from test_evidence_adaptive_judgement import Model, picture, complete_required_rows
from test_placement_proposal_handoff import global_request
from test_r13_placement_stage_handoff import finding, judge
from test_postrun_policy_boundary import v2_control


def packet(picture):
    req = global_request(picture)
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    for name in ("lamp", "cup"):
        req["scene_summary"]["objects"].append({**deepcopy(req["scene_summary"]["objects"][0]), "id": name})
    req["object_groups"][0]["object_ids"] += ["lamp", "cup"]
    req["selected_object_ids"] = ["chair", "desk", "lamp", "cup"]
    value = finding()
    first_item, first_defect = deepcopy(value["judge_originated_placement_results"][0]), deepcopy(value["defects"][0])
    value["judge_originated_placement_results"], value["defects"] = [], []
    for name in ("chair", "lamp", "cup"):
        value["judge_originated_placement_results"].append(
            {**deepcopy(first_item), "subject_id": name, "proposal_id": "proposal_" + name})
        value["defects"].append({**deepcopy(first_defect), "target_ids": [name], "check_id": "proposal_" + name})
    return req, value


def test_all_claims_registered_atomically(picture):
    req, value = packet(picture)
    model = Model(value)
    response = judge(model)._adjudicate_scene_quality_raw(req)
    assert len(model.calls) == 1 and response["verdict"] == "ambiguous" and not response["defects"]
    original = _judge_request(req)
    with evidence_policy_scope(req):
        updated, _ = _register_pending_placement_check(
            original, raw_response=response, evidence_request=EvidenceRequest(**response["evidence_request"]))
    checks = updated.context["deferred_placement_checks"]
    assert len(checks) == 3
    assert {c["subject_id"] for c in checks} == {"chair", "lamp", "cup"}
    assert all(c["owner_stage"] == "group_local" and c["judge_status"] != "resolved" for c in checks)
    assert {c["prior_stage_finding"]["prior_finding"]["proposal_id"] for c in checks} == {
        "proposal_chair", "proposal_lamp", "proposal_cup"}
    assert not original.context.get("deferred_placement_checks")


@pytest.mark.parametrize("bad", ["unknown", "missing_defect", "severity", "conclusion", "duplicate"])
def test_one_bad_child_rejects_the_whole_response(picture, bad):
    req, value = packet(picture)
    item = value["judge_originated_placement_results"][-1]
    if bad == "unknown":
        item["subject_id"] = "foreign"
    elif bad == "missing_defect":
        value["defects"].pop()
    elif bad == "severity":
        value["defects"][-1]["severity"] = "invented"
    elif bad == "conclusion":
        item["conclusion"] = "valid"
    else:
        item["proposal_id"] = "proposal_chair"
    with pytest.raises(ResponseSchemaRepairError):
        judge(Model(value))._adjudicate_scene_quality_raw(req)


@pytest.mark.parametrize("bad", ["nested", "missing_target", "unknown_subject"])
def test_registration_cannot_partially_mutate_request(picture, bad):
    req, value = packet(picture)
    response = judge(Model(value))._adjudicate_scene_quality_raw(req)
    evidence = response["evidence_request"]
    child = evidence["metadata"]["placement_check_handoffs"][-1]
    if bad == "nested":
        child["metadata"]["placement_check_handoffs"] = []
    elif bad == "missing_target":
        evidence["target_ids"] = ["chair"]
    else:
        child["metadata"]["placement_check_proposal"]["subject_id"] = "foreign"
    original = _judge_request(req)
    before = deepcopy(original.context)
    with evidence_policy_scope(req), pytest.raises(ValueError):
        _register_pending_placement_check(original, raw_response=response,
                                         evidence_request=EvidenceRequest(**evidence))
    assert original.context == before


@pytest.mark.parametrize("verdict", ["valid", "invalid"])
def test_public_metric_delivers_every_check_to_owner(picture, verdict):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    from benchmark.visual_judge.runtime import build_controlled_vlm_judge
    req, findings = packet(picture)
    metric = "semantic_placement_consistency"
    seen, emitted = [], False
    class Planner:
        def discover_placement_evidence(self, req):
            return {"schema_version": "placement_discovery_v2",
                    "considered_object_ids": ["chair", "desk", "lamp", "cup"],
                    "decision_authority": "none", "reason": "No prior checks.", "candidates": []}
    def respond(messages):
        nonlocal emitted
        context = json.loads(messages[1]["content"][0]["text"].split("\n", 1)[1])
        if context.get("evidence_phase") == "global_discovery" and not emitted:
            emitted = True
            return findings
        value = complete_required_rows(messages)
        checks = context.get("required_placement_checks") or []
        if context.get("evidence_phase") == "group_local_review" and checks:
            seen.extend(deepcopy(checks))
            if verdict == "invalid":
                value["verdict"] = "invalid"
                for row in value["placement_check_results"]:
                    row["conclusion"] = "invalid"
                value["defects"] = [{**deepcopy(findings["defects"][0]), "check_id": c["check_id"],
                                     "target_ids": [c["subject_id"]]} for c in checks]
        return value
    wrapper = build_controlled_vlm_judge(judge(Model(respond)), control=v2_control(),
        camera_provider=lambda req: {"status": "available", "paths": [picture]})
    report = evaluate_scene_quality_interfaces(req["scene_summary"],
        config={"enabled": True, "metrics": {name: {"enabled": name == metric}
                                           for name in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={"object_groups": req["object_groups"]},
        render_evidence=[picture], functional_evidence_planner=Planner(),
        camera_evidence_provider=lambda req: {"status": "available", "paths": [picture]},
        vlm_judge=wrapper, metric_applicability={metric: {"applicability": "relevant"}})
    value = report["metrics"][metric]
    assert emitted and value["status"] == "evaluated", value.get("reason")
    checks = value["placement_check_ledger"]["checks"]
    assert len(checks) == 3 and all(c["judge_status"] == "resolved" for c in checks)
    assert {c["subject_id"] for c in seen} == {"chair", "lamp", "cup"}
    assert (value["score"] == 1.0) is (verdict == "valid")
