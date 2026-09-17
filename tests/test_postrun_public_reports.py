"""Public opt-in policy, null persistence, and original metric inventories."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY
from test_evidence_adaptive_judgement import Model, response, request, complete_required_rows


@pytest.mark.parametrize("metric", ["collision", "support", "oob"])
def test_public_l1_no_images_reviews_detector_facts_but_keeps_model_gap(metric):
    from benchmark.evaluator.generic_validity.collision import check_collision
    from benchmark.evaluator.generic_validity.oob import check_oob
    from benchmark.evaluator.generic_validity.support import check_support
    scene = request(metric)["scene_summary"]
    scene["scene_height"] = 3
    obj = scene["objects"][0]
    if metric == "collision":
        scene["objects"].append({**deepcopy(obj), "id": "table", "center": [1.1, 1.1, 1.1]})
    elif metric == "oob":
        obj["center"] = [4.1, 1, 1]
    model = Model({"status": "need_more_evidence", "confidence": 0.0,
        "reason": "Detector facts do not resolve the missing surface observation.", "defects": [],
        "evidence_request": {"target_ids": ["chair"], "missing_observations": ["target_visible"],
                             "view_goal": "Inspect the target surface", "metadata": {}}})
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=FALLBACK_POLICY)
    result = {"collision": check_collision, "support": check_support, "oob": check_oob}[metric](
        scene, vlm_judge=judge, render_evidence=[], local_view_provider=lambda _: [],
        config={"official_mode": True},
    )
    assert model.calls
    assert result["status"] == "not_evaluable", result
    assert result["score"] is None
    assert not result["resolution_coverage"]["complete"]
    assert result["resolution_coverage"]["gap_count"] > 0


def test_public_canonical_v2_gap_report_is_persisted_and_schema_valid(tmp_path):
    from benchmark.api.evaluation import run_evaluate
    from test_current_evaluation_profile import _scene
    from jsonschema import Draft202012Validator
    model = Model(complete_required_rows)
    output = tmp_path / "v2.json"
    result = run_evaluate(
        scene=_scene(), out=output,
        vlm_judge=OpenAICompatibleVLMJudge(model, evidence_resolution_policy=FALLBACK_POLICY),
        asset_policy={"mode": "generated_or_open_assets", "arrangement_owner": "generator",
                      "category_selection_owner": "generator", "scale_owner": "generator", "appearance_owner": "generator"},
        render_evidence=[], vlm_evaluation_control={"evidence_resolution_policy": FALLBACK_POLICY,
                                                   "budgets": {"max_evidence_rounds": 0}},
    )
    assert result["evidence_resolution_policy"] == FALLBACK_POLICY
    assert result["execution_complete"]
    assert result["benchmark_score"] is None
    assert result["resolution_coverage"]["planned_count"] == 8
    assert not result["resolution_coverage"]["complete"]
    assert json.loads(output.read_text())["resolution_coverage"] == result["resolution_coverage"]
    schema = json.loads((Path(__file__).parents[1] / "schemas/evaluation_report.schema.json").read_text())
    errors = list(Draft202012Validator({"$ref": "#/$defs/canonicalReportV2", "$defs": schema["$defs"]}).iter_errors(result))
    assert not errors, [(list(error.absolute_path), error.message[:300]) for error in errors[:8]]


def test_nonrect_terminal_v2_retains_all_room_metrics_with_null_gaps(tmp_path):
    pytest.importorskip("benchmark.non_rectangular.runtime", reason="Legacy workflow not shipped by uniform geometry adapter")
    from benchmark.api.evaluation import run_evaluate
    from benchmark.non_rectangular import prepare_non_rectangular_evaluation
    from test_non_rectangular_polygon_evaluator import _input, _grouping_runtime
    inputs = _input()
    model = Model(complete_required_rows)
    output = tmp_path / "nonrect-v2.json"
    result = run_evaluate(
        evaluation_mode="non_rectangular_multi_room", evaluation_input=inputs, out=output,
        vlm_judge=OpenAICompatibleVLMJudge(model, evidence_resolution_policy=FALLBACK_POLICY),
        runtime_by_room=_grouping_runtime(prepare_non_rectangular_evaluation(inputs)),
        vlm_evaluation_control={"evidence_resolution_policy": FALLBACK_POLICY, "budgets": {"max_evidence_rounds": 0}},
    )
    assert result["schema_version"] == "non_rectangular_evaluation_report_v2", result
    assert result["terminal_status"] == "finished_with_gaps"
    assert result["execution_complete"]
    assert result["aggregate"]["overall_score"] is None
    assert result["resolution_coverage"]["planned_count"] == len(result["rooms"]) * 8
    for room in result["rooms"].values():
        assert len(room["report"]["metrics"]) == 8
        assert room["report"]["status"] == "finished_with_gaps"
        assert any(item["status"] == "not_evaluable" and item["score"] is None
                   for item in room["report"]["metrics"].values())
        assert not room["infrastructure_failures"]
    assert json.loads(output.read_text())["aggregate"]["overall_score"] is None


def test_fixed_summary_keeps_seven_scores_and_eighth_gap():
    from benchmark.evaluator.consistency_audit_v2 import finish_persisted
    names = ["collision", "oob", "support", "scale_consistency", "style_consistency",
             "object_pairing_consistency", "functional_consistency", "semantic_placement_consistency"]
    result = {"metrics": [{"metric": name, "layer": "L1" if i < 3 else "L3",
                           "local_weight": 1, "overall_weight": 0.125}
                          for i, name in enumerate(names)], "layers": [{"layer": "L1"}, {"layer": "L3"}]}
    sources = {name: {"status": "checked", "score": 0.75,
                      "resolution_coverage": {"complete": True}} for name in names}
    sources[names[4]] = {"status": "not_evaluable", "score": None,
                         "resolution_coverage": {"complete": False}}
    finish_persisted(result, sources)
    assert result["combined_score_100"] is None
    assert result["resolution_coverage"]["planned_count"] == 8
    assert result["resolution_coverage"]["accepted_count"] == 7
    assert sum(item["score"] == 0.75 for item in result["metrics"]) == 7
    assert result["layers"][0]["score"] == 0.75
    assert result["layers"][1]["score"] is None


def test_public_existing_empty_pair_rule_is_deterministic_not_model_fallback():
    from benchmark.evaluator.generic_validity.collision import check_collision
    scene = request("collision")["scene_summary"]
    model = Model(response())
    result = check_collision(scene, vlm_judge=OpenAICompatibleVLMJudge(
        model, evidence_resolution_policy=FALLBACK_POLICY), render_evidence=[],
        config={"official_mode": True})
    assert not model.calls
    assert result["score"] == 1.0
    assert result["resolution_coverage"]["complete"]
    unit = result["resolution_coverage"]["units"][0]
    assert unit["decision_source"] == "deterministic_rule"
    assert not unit["visual_observation_complete"]


def test_v2_retained_target_does_not_force_or_relabel_judgement(tmp_path):
    from benchmark.evaluator.scene_quality.target_scoped import evaluate_target_scoped_judgements
    from benchmark.evaluator.scene_quality.target_scope import build_target_camera_scope
    from benchmark.visual_judge.evidence_gap_v2 import EvidenceGapError
    from benchmark.visual_judge.evidence_resolution import evidence_policy_scope
    from test_target_centered_scope import _scene, _decodable_image
    scene = _scene()
    scope = build_target_camera_scope(scene, target_id="chair",
                                     metric="semantic_placement_consistency", explicit_context_ids=["desk"])
    path = _decodable_image(tmp_path, "partial")
    calls = []
    def judge(_, req):
        calls.append(req)
        raise EvidenceGapError("target-local observation is still missing")
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        records = evaluate_target_scoped_judgements(
            metric_name="semantic_placement_consistency", scene=scene, prompt=None,
            packets=[{"target_id": "chair", "context_ids": ["desk"], "framing_ids": list(scope.framing_ids),
                      "target_scope": scope, "paths": [path],
                      "resolution": {"scope_satisfied": False, "global_anchor_satisfied": True,
                                     "local_scope_satisfied": False, "provider_invoked": True,
                                     "provider_status": "insufficient", "provider_reason": "no_feasible_candidate"}}],
            vlm_judge=object(), authorized_deviations=[], visual_style_spec=None,
            build_judge_request=lambda **kwargs: kwargs, call_judge=judge,
            apply_prompt_exemptions=lambda value, **_: value,
            normalize_judgement=lambda value, **_: value)
    assert len(calls) == 1 and calls[0]["render_evidence"] == [path]
    assert "budget_exhaustion_finalization" not in calls[0]
    assert not records[0].get("retained_global_forced_final")
    assert records[0]["evidence_degradation"]["forced_final_judge"] is False
    assert records[0]["score"] is None
    assert records[0]["status"] == "not_evaluable"


def test_nonrect_all_room_failures_still_keep_v2_original_inventory():
    pytest.importorskip("benchmark.non_rectangular.workflow", reason="Legacy workflow not shipped by uniform geometry adapter")
    from benchmark.non_rectangular.workflow import execute_non_rectangular_workflow
    from benchmark.non_rectangular.report import build_non_rectangular_evaluation_report
    from test_non_rectangular_workflow import _preflight
    class BrokenEvaluator:
        evidence_resolution_policy = FALLBACK_POLICY
        def evaluate(self, unit):
            raise ConnectionError("offline service failure")
    execution = execute_non_rectangular_workflow(_preflight(), room_evaluator=BrokenEvaluator())
    report = build_non_rectangular_evaluation_report(execution)
    assert report["schema_version"] == "non_rectangular_evaluation_report_v2"
    assert report["execution_complete"] and report["terminal_status"] == "finished_with_gaps"
    assert len(report["rooms"]) == 2
    assert report["resolution_coverage"]["planned_count"] == 16
    assert len(report["resolution_coverage"]["missing_ids"]) == 16
    assert report["resolution_coverage"]["accepted_count"] == 0
    assert report["coverage"]["infrastructure_failure_count"] == 2
    assert all(metric["score"] is None for metric in report["aggregate"]["metrics"].values())
