"""Offline metric isolation, typed failures and fixed pre-execution scope plan."""
import json

import pytest

from benchmark.visual_judge.evidence_gap_v2 import EvidenceGapError, FALLBACK_POLICY
from benchmark.visual_judge.evidence_resolution import evidence_policy_scope
from benchmark.evaluator.scene_quality.consistency_acceptance_v2 import summarize_metric, terminalize_scope
from test_postrun_placement_scope import picture


def accepted():
    return {"status": "evaluated", "score": 1.0, "evidence_resolution": {
        "policy": FALLBACK_POLICY, "accepted": True, "unit_key": "synthetic",
        "visual_observation_complete": True, "model_judgement_completed": True,
    }}


def test_missing_original_group_cannot_disappear_from_denominator():
    report = {**accepted(), "group_results": [{"group_id": "a", **accepted()}]}
    coverage = summarize_metric(report, ["group:a", "group:b"])
    assert coverage["planned_count"] == 2
    assert coverage["missing_ids"] == ["group:b"]
    assert coverage["evaluation_fraction"] == 0.5
    assert not coverage["complete"]
    assert not coverage["execution_complete"]


@pytest.mark.parametrize("category,status,terminal", [
    ("evidence_gap", "not_evaluable", "evidence_gap"),
    ("evidence_unavailable", "not_evaluable", "evidence_gap"),
    ("model_service_failure", "failed", "infrastructure_failure"),
    ("input_integrity_failure", "failed", "infrastructure_failure"),
])
def test_terminal_gap_is_not_infrastructure(category, status, terminal):
    record = {"status": "unresolved", "score": None,
              "judgement": {"failure": {"failure_category": category}}}
    result = terminalize_scope(record, phase="scope")
    assert result["status"] == status
    assert result["terminal_state"] == terminal
    assert result["score"] is None
    assert result["execution_complete"]


@pytest.mark.parametrize("fault", ["ambiguous", "service"])
def test_public_metrics_continue_after_local_failure(picture, fault):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.runtime import build_controlled_vlm_judge
    from test_postrun_policy_boundary import v2_control
    from test_evidence_adaptive_judgement import Model, complete_required_rows
    from test_placement_proposal_handoff import global_request, needs_local
    req = global_request(picture)
    seen = []
    def respond(messages):
        ctx = json.loads(messages[1]["content"][0]["text"].split("\n", 1)[1])
        metric = ctx["metric"]
        seen.append(metric)
        if metric == "style_consistency":
            if fault == "service":
                raise ConnectionError("synthetic service failure")
            value = needs_local()
            value["evidence_request"]["metadata"] = {}
            value["missing_evidence"] = ["target_visible"]
            value["evidence_request"]["missing_observations"] = ["target_visible"]
            return value
        return complete_required_rows(messages)
    model = Model(respond)
    provider = lambda request: {"status": "available", "paths": [picture]}
    judge = build_controlled_vlm_judge(OpenAICompatibleVLMJudge(model),
        control=v2_control(), camera_provider=provider)
    active = {"scale_consistency", "style_consistency", "object_pairing_consistency"}
    report = evaluate_scene_quality_interfaces(
        req["scene_summary"], config={"enabled": True, "metrics": {
            name: {"enabled": name in active} for name in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={"object_groups": req["object_groups"]}, render_evidence=[picture],
        camera_evidence_provider=provider, vlm_judge=judge,
        metric_applicability={name: {"applicability": "relevant"} for name in active},
    )
    metrics = report["metrics"]
    assert metrics["scale_consistency"]["status"] == "evaluated", metrics["scale_consistency"]
    assert metrics["object_pairing_consistency"]["status"] == "evaluated", metrics["object_pairing_consistency"]
    style = metrics["style_consistency"]
    assert style["status"] == ("not_evaluable" if fault == "ambiguous" else "failed"), json.dumps(style, default=str)
    assert style["score"] is None
    assert report["score"] is None
    assert report["execution_complete"]
    assert report["resolution_coverage"]["planned_count"] == 3
    assert report["resolution_coverage"]["evaluation_count"] == 2
    assert not report["resolution_coverage"]["complete"]
    assert set(seen) == active
