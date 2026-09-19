"""V2 existence/context/ownership separation and bounded reverse handoff."""
from copy import deepcopy
import json

import pytest
from PIL import Image

from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, evidence_policy_scope
from benchmark.visual_judge.openai_compatible import _validate_pending_placement_proposal, _canonical_phase_instruction
from benchmark.visual_judge.orchestration.controller import _register_pending_placement_check
from benchmark.visual_judge.adapters.legacy_judge import _judge_request
from benchmark.visual_judge.interfaces.judge import EvidenceRequest
from test_placement_proposal_handoff import global_request, needs_local, METRIC


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "packet.png"
    im = Image.new("RGB", (20, 20), "white")
    im.putpixel((2, 3), (0, 0, 0))
    im.save(path)
    return str(path)


def local_request(picture):
    req = global_request(picture, "group_local_review")
    req["evidence_resolution_policy"] = FALLBACK_POLICY
    req["placement_scene_groups"] = [
        {"group_id": "seat", "object_ids": ["chair"]},
        {"group_id": "work", "object_ids": ["desk"]},
    ]
    req["object_groups"] = [req["placement_scene_groups"][0]]
    req["group_scope"] = {"group_id": "seat", "member_ids": ["chair"]}
    value = needs_local()
    value["evidence_request"]["metadata"]["placement_check_proposal"]["check_type"] = "contextual_anchor"
    return req, value


def test_real_external_context_routes_without_changing_identity(picture):
    req, value = local_request(picture)
    before = deepcopy(value)
    with evidence_policy_scope(req):
        _validate_pending_placement_proposal(value, request=req)
        incoming = value["evidence_request"]
        request, check = _register_pending_placement_check(
            _judge_request(req), raw_response=value,
            evidence_request=EvidenceRequest(**incoming),
        )
        assert check["subject_id"] == "chair"
        assert check["context_ids"] == ["desk"]
        assert check["check_type"] == "contextual_anchor"
        assert check["owner_stage"] == "scene_global"
        assert check["handoff_status"] == "deferred_to_scene_global"
        assert request.context["deferred_placement_checks"] == [check]
        assert not request.context.get("required_placement_checks")
        assert "scene_zone" in _canonical_phase_instruction(
            metric=METRIC, evidence_phase="group_local_review", decision_mode="final")
    assert value == before


@pytest.mark.parametrize("bad", ["unknown_context", "foreign_subject", "missing_target", "incomplete_partition"])
def test_v2_scope_rejects_invalid_identity_or_ownership(picture, bad):
    req, value = local_request(picture)
    proposal = value["evidence_request"]["metadata"]["placement_check_proposal"]
    if bad == "unknown_context":
        proposal["context_ids"] = ["invented"]
    elif bad == "foreign_subject":
        proposal.update(subject_id="desk", context_ids=["chair"])
    elif bad == "missing_target":
        value["evidence_request"]["target_ids"] = ["desk"]
    else:
        req["placement_scene_groups"] = req["object_groups"]
    with evidence_policy_scope(req), pytest.raises(ValueError):
        _validate_pending_placement_proposal(value, request=req)


def test_reverse_handoff_survives_scope_consumer_and_gets_owning_review(picture):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.runtime import build_controlled_vlm_judge
    from test_postrun_policy_boundary import v2_control
    from test_evidence_adaptive_judgement import Model, complete_required_rows

    req = global_request(picture)
    seen = []
    emitted = False
    class Planner:
        def discover_placement_evidence(self, request):
            return {"schema_version": "placement_discovery_v2", "considered_object_ids": ["chair", "desk"],
                    "decision_authority": "none", "reason": "Synthetic empty prior.", "candidates": []}
    def respond(messages):
        nonlocal emitted
        ctx = json.loads(messages[1]["content"][0]["text"].split("\n", 1)[1])
        seen.append(deepcopy(ctx))
        if ctx.get("evidence_phase") == "group_local_review" and not emitted:
            emitted = True
            value = needs_local()
            value["evidence_request"]["metadata"]["placement_check_proposal"]["check_type"] = "scene_zone"
            return value
        return complete_required_rows(messages)
    model = Model(respond)
    judge = build_controlled_vlm_judge(OpenAICompatibleVLMJudge(model), control=v2_control(),
        camera_provider=lambda request: {"status": "available", "paths": [picture]})
    report = evaluate_scene_quality_interfaces(
        req["scene_summary"], config={"enabled": True, "metrics": {
            name: {"enabled": name == METRIC} for name in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={"object_groups": req["object_groups"]},
        render_evidence=[picture], functional_evidence_planner=Planner(),
        camera_evidence_provider=lambda request: {"status": "available", "paths": [picture]},
        vlm_judge=judge, metric_applicability={METRIC: {"applicability": "relevant"}},
    )
    metric = report["metrics"][METRIC]
    checks = (metric.get("placement_check_ledger") or {}).get("checks") or []
    assert emitted, metric
    assert len(checks) == 1, metric
    assert checks[0]["check_type"] == "scene_zone"
    assert checks[0]["owner_stage"] == "scene_global"
    assert metric.get("placement_global_handoff_reviews"), metric
    assert checks[0]["judge_status"] == "resolved", metric
    assert any(ctx.get("evidence_phase") == "global_discovery" and ctx.get("required_placement_checks")
               for ctx in seen)
    assert metric["status"] == "evaluated", metric.get("resolution_coverage")
    assert metric["resolution_coverage"]["policy"] == FALLBACK_POLICY
    assert report["resolution_coverage"]["complete"]
