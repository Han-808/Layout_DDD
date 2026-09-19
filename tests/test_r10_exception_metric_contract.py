"""Public aggregation regression for exceptions escaping a metric workflow."""
from functools import wraps
import json

import pytest

from benchmark.evaluator.scene_quality import interfaces
from benchmark.evaluator.scene_quality.adaptive_acceptance import adaptive_metric_result
from benchmark.visual_judge.evidence_gap_v2 import EvidenceGapError
from test_postrun_placement_scope import picture


@pytest.mark.parametrize("values,expected", [
    ([], False), ([False, False], False), ([False, True], True),
    ([None], None), ([None, False], None), ([None, True], True),
])
def test_invocation_summary_preserves_unknown(values, expected):
    assert interfaces._any_reported_invocation(values) is expected


@pytest.mark.parametrize("failed_metric", interfaces.SUPPORTED_SCENE_QUALITY_METRICS)
@pytest.mark.parametrize("fault", ["gap", "service", "implementation"])
def test_exception_terminal_keeps_public_scene_contract(monkeypatch, picture, failed_metric, fault):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.runtime import build_controlled_vlm_judge
    from test_postrun_policy_boundary import v2_control
    from test_evidence_adaptive_judgement import Model, complete_required_rows
    from test_placement_proposal_handoff import global_request

    original = interfaces._evaluate_metric.__wrapped__
    called = []
    @wraps(original)
    def injected(**kwargs):
        called.append(kwargs["metric_name"])
        if kwargs["metric_name"] == failed_metric:
            error = {"gap": EvidenceGapError, "service": ConnectionError,
                     "implementation": KeyError}[fault]
            raise error("synthetic-private-detail-must-not-be-persisted")
        return original(**kwargs)
    monkeypatch.setattr(interfaces, "_evaluate_metric", adaptive_metric_result(injected))
    req = global_request(picture)
    model = Model(complete_required_rows)
    provider = lambda request: {"status": "available", "paths": [picture]}
    judge = build_controlled_vlm_judge(OpenAICompatibleVLMJudge(model),
        control=v2_control(), camera_provider=provider)
    report = interfaces.evaluate_scene_quality_interfaces(
        req["scene_summary"],
        config={"enabled": True},
        object_grouping_report={"object_groups": req["object_groups"]},
        render_evidence=[picture], camera_evidence_provider=provider, vlm_judge=judge,
        metric_applicability={name: {"applicability": "relevant"}
                              for name in interfaces.SUPPORTED_SCENE_QUALITY_METRICS},
    )
    assert set(called) == set(interfaces.SUPPORTED_SCENE_QUALITY_METRICS)
    assert len(called) == 5
    metric = report["metrics"][failed_metric]
    assert metric["status"] == ("not_evaluable" if fault == "gap" else "failed")
    assert metric["score"] is None
    assert metric["weight"] == report["scoring"]["runtime_config_metric_weights"][failed_metric]
    assert metric["evidence_request"]["provider_invoked"] is None
    assert metric["vlm_invoked"] is None
    assert metric["invocation_audit_complete"] is False
    assert metric["exception_diagnostics"]["frames"][-1]["function"] == "injected"
    assert "synthetic-private-detail" not in json.dumps(metric)
    assert report["execution_complete"]
    assert report["resolution_coverage"]["planned_count"] == 5
    assert report["score"] is None
    assert not report["resolution_coverage"]["complete"]
    # Other metrics keep independently established results, not skipped rows.
    siblings = [value for name, value in report["metrics"].items() if name != failed_metric]
    assert any(value["status"] == "evaluated" for value in siblings)
    assert all(value.get("execution_complete") is True for value in siblings)
