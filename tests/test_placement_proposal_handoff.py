"""Incoming global-to-local proposals must reach real controller registration.

Synthetic counterpart of the S100 lamp/table proposal; no API or Blender.
"""
from copy import deepcopy
import json

import pytest
from PIL import Image

from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.evidence_resolution import ADAPTIVE_POLICY, LEGACY_POLICY
from benchmark.visual_judge.openai_compatible import _validate_pending_placement_proposal
from test_evidence_adaptive_judgement import Model, complete_required_rows, request, response

METRIC = "semantic_placement_consistency"


def proposal():
    return {
        "proposal_id": "local-support-from-global",
        "subject_id": "chair", "context_ids": ["desk"],
        "check_type": "support_and_height",
        "observation_goal": "Inspect the subject's support surface and height.",
    }


def needs_local():
    value = response()
    value.update(evidence_status="insufficient", verdict="ambiguous",
                 reason="A local support observation is required.",
                 missing_evidence=["contact_surface_visible"],
                 evidence_request={
                     "target_ids": ["chair", "desk"],
                     "missing_observations": ["contact_surface_visible"],
                     "view_goal": "Show the subject and its support surface.",
                     "metadata": {"placement_check_proposal": proposal()},
                 })
    return value


def global_request(picture, phase="global_discovery"):
    req = request(METRIC, [picture], terminal=False)
    scene = req["scene_summary"]
    scene["objects"].append({**deepcopy(scene["objects"][0]), "id": "desk", "category": "desk", "center": [3, 3, 1]})
    req.update(evidence_phase=phase, decision_mode="final",
               object_groups=[{"group_id": "work", "object_ids": ["chair", "desk"]}],
               required_placement_checks=[])
    return req


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "synthetic.png"
    img = Image.new("RGB", (20, 20), "white")
    img.putpixel((2, 3), (0, 0, 0))
    img.save(path)
    return str(path)


@pytest.mark.parametrize("phase", ["global_discovery", "residual_global_placement_review"])
@pytest.mark.parametrize("policy", [ADAPTIVE_POLICY, LEGACY_POLICY])
def test_real_adapter_accepts_pending_handoff_without_verdict_or_repair(picture, phase, policy):
    req = global_request(picture, phase)
    req["evidence_resolution_policy"] = policy
    pending = needs_local()
    if phase == "residual_global_placement_review":
        pending["group_global_observations"] = [{
            "group_id": "work", "object_ids": ["chair", "desk"], "related_group_ids": [],
            "global_position_observation": "Synthetic coherent work zone.",
            "inter_group_observation": "Only one group is present.",
            "evidence_sufficiency": "sufficient", "residual_issue_candidate": "none",
        }]
    model = Model(pending)
    value = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=policy)._adjudicate_scene_quality_raw(req)
    assert len(model.calls) == 1
    assert value["verdict"] == "ambiguous"
    assert value["evidence_status"] == "insufficient"
    assert value["defects"] == []
    assert not value.get("placement_check_results")
    assert value["evidence_request"]["metadata"]["placement_check_proposal"]["subject_id"] == "chair"


@pytest.mark.parametrize("bad", ["subject", "context", "check_type", "extra_field", "incomplete_partition", "no_partition", "unowned_phase", "reverse_direction"])
def test_handoff_does_not_weaken_identity_or_stage_boundaries(picture, bad):
    req, value = global_request(picture), needs_local()
    p = value["evidence_request"]["metadata"]["placement_check_proposal"]
    if bad == "subject":
        p["subject_id"] = "unknown"
    elif bad == "context":
        p["context_ids"] = ["unknown"]
    elif bad == "check_type":
        p["check_type"] = "physical_collision"
    elif bad == "extra_field":
        p["verdict"] = "valid"
    elif bad == "incomplete_partition":
        req["object_groups"][0]["object_ids"] = ["desk"]
    elif bad == "no_partition":
        req["object_groups"] = []
    elif bad == "unowned_phase":
        req["evidence_phase"] = "unknown"
    else:
        req["evidence_phase"] = "group_local_review"
        p["check_type"] = "scene_zone"
    with pytest.raises(ValueError):
        _validate_pending_placement_proposal(value, request=req)


@pytest.mark.parametrize("conclusion", ["valid", "invalid", "missing"])
@pytest.mark.parametrize("new_evidence", [False, True])
@pytest.mark.parametrize("proposal_count", [1, 2])
def test_public_workflow_routes_incoming_proposal_and_requires_local_resolution(picture, tmp_path, conclusion, new_evidence, proposal_count):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    from benchmark.visual_judge.control_config import resolve_vlm_evaluation_control
    from benchmark.visual_judge.runtime import build_controlled_vlm_judge

    req = global_request(picture)
    seen = []
    emitted = 0

    class Planner:
        def discover_placement_evidence(self, req):
            return {"schema_version": "placement_discovery_v2", "considered_object_ids": ["chair", "desk"],
                    "decision_authority": "none", "reason": "Synthetic discovery omits the local support check.",
                    "candidates": []}

    def respond(messages):
        nonlocal emitted
        context = json.loads(messages[1]["content"][0]["text"].split("\n", 1)[1])
        seen.append(deepcopy(context))
        if context.get("evidence_phase") == "global_discovery" and emitted < proposal_count and not context.get("adaptive_evidence", {}).get("terminal"):
            emitted += 1
            return needs_local()
        value = complete_required_rows(messages)
        checks = context.get("required_placement_checks") or []
        if context.get("evidence_phase") == "group_local_review" and checks:
            if conclusion == "missing":
                raise RuntimeError("synthetic local Judge failure")
            if conclusion == "invalid":
                value["verdict"] = "invalid"
                for row in value["placement_check_results"]:
                    row["conclusion"] = "invalid"
                value["defects"] = [{
                    "scope": "semantically_inappropriate_support_surface", "target_ids": [c["subject_id"]],
                    "check_id": c["check_id"], "check_type": c["check_type"], "relation": c["check_type"],
                    "category": "semantic_surface_mismatch", "attribution_mode": "unary",
                    "reason": "Synthetic inappropriate semantic support.", "severity": "atypical",
                } for c in checks]
        return value

    model = Model(respond)
    acquisitions = []
    def provider(req):
        acquisitions.append(deepcopy(req))
        path = picture
        if new_evidence:
            path = str(tmp_path / f"acquired_{len(acquisitions)}.png")
            with Image.open(picture) as source:
                acquired = source.copy()
            acquired.putpixel((4, 5), (len(acquisitions), 80, 120))
            acquired.save(path)
        return {"status": "available", "paths": [path]}
    judge = build_controlled_vlm_judge(
        OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY),
        control=resolve_vlm_evaluation_control({"evidence_resolution_policy": ADAPTIVE_POLICY}), camera_provider=provider,
    )
    report = evaluate_scene_quality_interfaces(
        req["scene_summary"], config={"enabled": True, "metrics": {
            name: {"enabled": name == METRIC} for name in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={"object_groups": req["object_groups"]},
        render_evidence=[picture], functional_evidence_planner=Planner(),
        camera_evidence_provider=provider, vlm_judge=judge,
        metric_applicability={METRIC: {"applicability": "relevant"}},
    )
    metric = report["metrics"][METRIC]
    assert emitted
    if new_evidence:
        assert emitted == proposal_count
    local_checks = [c for ctx in seen if ctx.get("evidence_phase") == "group_local_review"
                    for c in ctx.get("required_placement_checks") or []]
    assert local_checks, json.dumps({"status": metric["status"], "calls": [
        {k: c.get(k) for k in ("evidence_phase", "required_placement_checks", "deferred_placement_checks")}
        for c in seen], "ledger": metric.get("placement_check_ledger"),
        "audit": [{k: a.get(k) for k in ("stage", "stop_reason", "judge_request")} for a in judge.audit_records]}, default=str)[:8000]
    assert {c["subject_id"] for c in local_checks} == {"chair"}
    check_ids = {c["check_id"] for c in local_checks}
    assert len(check_ids) == 1
    assert all(c["owner_stage"] == "group_local" for c in local_checks)
    ledger = [c for c in metric["placement_check_ledger"]["checks"] if c["check_id"] in check_ids]
    assert len(ledger) == 1
    if conclusion == "missing":
        assert metric["status"] == "failed"
        assert report["resolution_coverage"]["complete"] is False
    else:
        def validation_errors(value):
            if isinstance(value, dict):
                return ([value["validation_error"]] if value.get("validation_error") else []) + [e for v in value.values() for e in validation_errors(v)]
            if isinstance(value, list):
                return [e for v in value for e in validation_errors(v)]
            return []
        assert metric["status"] == "evaluated", validation_errors(metric)
        assert report["resolution_coverage"]["complete"]
        assert (report["score"] == 1.0) if conclusion == "valid" else (report["score"] < 1.0)
