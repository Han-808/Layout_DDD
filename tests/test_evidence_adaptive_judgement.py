from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from PIL import Image

from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.evidence_resolution import (
    ADAPTIVE_METRICS, ADAPTIVE_POLICY, EvidenceUnavailableError,
    ContextCapacityError, evidence_policy_scope, prepare_adaptive_request,
    failure_record, style_policy_result,
)
from benchmark.evaluator.scene_quality.adaptive_acceptance import terminalize_adaptive_scope


class Model:
    model_id = "offline-mock"
    endpoint = "http://offline.invalid/v1"
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = []
        self.last_request_metadata = {}
    def chat_messages(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        value = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value(messages)
        return json.dumps(value)


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "evidence.png"
    img = Image.new("RGB", (20, 20), "white")
    img.putpixel((2, 3), (0, 0, 0))
    img.save(path)
    return str(path)


def request(metric, paths=(), *, appearance=True, terminal=True):
    obj = {"id": "chair", "category": "chair", "center": [1, 1, 1], "size": [1, 1, 1]}
    if appearance:
        obj["asset_metadata"] = {"material": "wood", "color": "brown"}
    return {
        "metric": metric, "evidence_resolution_policy": ADAPTIVE_POLICY,
        "adaptive_terminal": terminal, "render_evidence": list(paths),
        "objects": [obj], "scene_summary": {"objects": [obj], "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        "selected_object_ids": ["chair"], "event": {"object_id": "chair"},
        "metric_rubric": "Apply the original metric definition.",
    }


def response(verdict="valid"):
    return {
        "evidence_status": "sufficient", "verdict": verdict, "confidence": 0.73,
        "reason": "Metric evidence supports the conclusion.",
        "missing_evidence": [], "evidence_request": None,
        "defects": [] if verdict == "valid" else [
            {"scope": "object", "target_ids": ["chair"], "reason": "A significant metric defect.",
             "relation": "significant metric inconsistency",
             "severity": "major", "category": "inconsistent_scale"}
        ],
    }


@pytest.mark.parametrize("metric", sorted(ADAPTIVE_METRICS))
@pytest.mark.parametrize("tier", ["full_visual", "partial_visual", "structured_fallback"])
def test_eight_metrics_actual_evidence_delivery(metric, tier, picture):
    paths = [] if tier == "structured_fallback" else [picture]
    req = request(metric, paths, terminal=tier != "full_visual")
    model = Model(response())
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    raw = (judge._adjudicate_p0b_raw(req) if metric in {"collision", "support", "oob"}
           else judge._adjudicate_scene_quality_raw(req))
    assert raw["verdict"] == "valid"
    assert len(model.calls) == 1
    audit = raw["evidence_resolution"]
    assert audit["evidence_tier"] == tier
    assert audit["model_judgement_completed"] is True
    assert audit["visual_observation_complete"] is (tier == "full_visual")
    images = [item for msg in model.calls[0] if isinstance(msg["content"], list)
              for item in msg["content"] if item.get("type") == "image_url"]
    assert len(images) == len(paths)


@pytest.mark.parametrize("metric", sorted(ADAPTIVE_METRICS))
def test_structured_invalid_is_accepted_not_defaulted(metric):
    req = request(metric)
    answer = response("invalid")
    if metric == "semantic_placement_consistency":
        req["required_placement_checks"] = [{
            "check_id": "zone", "check_type": "scene_zone", "subject_id": "chair",
            "context_ids": [], "owner_stage": "scene_global", "group_ids": [],
            "required_observations": ["target_visible", "global_context_preserved"],
        }]
        answer["placement_check_results"] = [{
            "check_id": "zone", "subject_id": "chair", "context_ids": [],
            "observation_status": "inferred_under_budget", "conclusion": "invalid",
            "reason": "The room zone is inappropriate for this object.",
        }]
        answer["defects"] = [{
            "scope": "semantically_inappropriate_scene_zone", "target_ids": ["chair"],
            "relation": "scene_zone", "reason": "The room zone is inappropriate for this object.",
            "severity": "material_contextual_mismatch", "check_id": "zone", "check_type": "scene_zone",
        }]
    model = Model(answer)
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    raw = (judge._adjudicate_p0b_raw(req) if metric in {"collision", "support", "oob"}
           else judge._adjudicate_scene_quality_raw(req))
    assert raw["verdict"] == "invalid"
    record = terminalize_adaptive_scope({"status": "evaluated", "score": 0.0, "judgement": raw}, phase="test")
    assert record["status"] == "evaluated"
    assert record["score"] == 0.0
    assert record["evidence_coverage"]["grounded"] is False


def test_style_without_appearance_skips_model():
    model = Model(AssertionError("must not call the model"))
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    raw = judge._adjudicate_scene_quality_raw(request("style_consistency", appearance=False))
    assert not model.calls
    assert raw["confidence"] is None
    assert raw["evidence_resolution"]["decision_source"] == "style_policy_no_deduction"
    assert raw["evidence_resolution"]["model_judgement_completed"] is False


def test_no_facts_is_not_valid():
    with pytest.raises(EvidenceUnavailableError):
        prepare_adaptive_request({"metric": "collision"}, terminal=True, trigger="no_camera")
    with pytest.raises(EvidenceUnavailableError):
        prepare_adaptive_request({"metric": "style_consistency"}, terminal=True, trigger="missing_scene")


def test_model_transport_failure_is_not_valid():
    model = Model(ConnectionError("offline"))
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    with pytest.raises(ConnectionError):
        judge._adjudicate_scene_quality_raw(request("scale_consistency"))
    assert len(model.calls) == 1


def test_error_classifier_does_not_scan_reason():
    assert failure_record(RuntimeError("schema contract HTTP auth"), phase="acquisition")["recoverable_acquisition"]
    assert failure_record(ConnectionError("normal"), phase="judge")["failure_category"] == "model_service_failure"


def test_required_terminal_cannot_default_valid():
    with evidence_policy_scope({"evidence_resolution_policy": ADAPTIVE_POLICY}):
        from benchmark.evaluator.scene_quality.terminal import terminalize_required_scope
        record = terminalize_required_scope({"status": "unresolved", "score": None, "reason": "camera empty"}, phase="test")
    assert record["status"] == "failed"
    assert record["score"] is None


@pytest.mark.parametrize("metric", ["scale_consistency", "object_pairing_consistency", "style_consistency"])
def test_mock_public_l3_zero_images_scores_fallback(metric):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    config = {"enabled": True, "metrics": {
        name: {"enabled": name == metric} for name in SCENE_QUALITY_INTERFACE_METRICS
    }}
    config["metrics"][metric]["evidence_plan"] = {"evidence_strategy": "global_and_local", "router_options": None}
    scene = request(metric)["scene_summary"]
    scene["objects"].append({**deepcopy(scene["objects"][0]), "id": "desk", "category": "desk"})
    model = Model(response())
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    report = evaluate_scene_quality_interfaces(
        scene, config=config, render_evidence=[],
        object_grouping_report={"object_groups": [{"group_id": "work", "object_ids": ["chair", "desk"]}]},
        vlm_judge=judge, metric_applicability={metric: {"applicability": "relevant"}},
    )
    result = report["metrics"][metric]
    assert result["status"] == "evaluated", result
    assert result["score"] == 1.0
    assert result["resolution_coverage"]["complete"]
    assert result["resolution_coverage"]["visual_evidence_fraction"] == 0.0
    assert report["status"] == "evaluated", report


@pytest.mark.parametrize("metric", ["collision", "support", "oob"])
@pytest.mark.parametrize("verdict", ["valid", "invalid"])
def test_mock_l1_zero_images_and_actual_score(metric, verdict):
    from benchmark.evaluator.generic_validity.collision import check_collision
    from benchmark.evaluator.generic_validity.support import check_support
    from benchmark.evaluator.generic_validity.oob import check_oob
    scene = request(metric)["scene_summary"]
    scene["scene_height"] = 3
    obj = scene["objects"][0]
    if metric == "collision":
        scene["objects"].append({**deepcopy(obj), "id": "table", "center": [1.1, 1.1, 1.1]})
    elif metric == "oob":
        obj["center"] = [4.1, 1, 1]
    model = Model(response(verdict))
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    result = {"collision": check_collision, "support": check_support, "oob": check_oob}[metric](
        scene, vlm_judge=judge, render_evidence=[], local_view_provider=lambda _: [],
    )
    assert model.calls, result
    assert result["status"] == "checked", result
    assert result["resolution_coverage"]["complete"], result
    expected = 1.0 if verdict == "valid" else 0.5 if metric == "collision" else 0.0
    assert result["score"] == expected, result
    assert result["resolution_coverage"]["visual_evidence_fraction"] == 0.0


@pytest.mark.parametrize("metric", ["functional_consistency", "semantic_placement_consistency"])
def test_mock_l3_direct_structured_flow(metric):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    config = {"enabled": True, "metrics": {
        name: {"enabled": name == metric} for name in SCENE_QUALITY_INTERFACE_METRICS
    }}
    config["metrics"][metric].update(
        evidence_plan={"evidence_strategy": "global_and_local", "router_options": None},
        evidence_policy={"camera_scope": "global"},
    )
    scene = request(metric)["scene_summary"]
    model = Model(response())
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    report = evaluate_scene_quality_interfaces(
        scene, config=config, render_evidence=[], vlm_judge=judge,
        metric_applicability={metric: {"applicability": "relevant"}},
    )
    assert report["metrics"][metric]["status"] == "evaluated", report["metrics"][metric]
    assert report["score"] == 1.0
    assert report["resolution_coverage"]["complete"]
    assert report["resolution_coverage"]["visual_evidence_fraction"] == 0.0


def test_scene_scoring_accepts_fallback_without_reweighting():
    from benchmark.api.evaluation import _strict_scoring_profile_score, _strict_scoring_profile_coverage
    reports = {
        "L1": {"status": "evaluated", "score": 0.4, "resolution_coverage": {"complete": True},
               "coverage": {"fraction": 0.0}},
        "L3": {"status": "evaluated", "score": 0.8, "resolution_coverage": {"complete": True},
               "coverage": {"fraction": 0.2}},
    }
    with evidence_policy_scope({"evidence_resolution_policy": ADAPTIVE_POLICY}):
        score = _strict_scoring_profile_score(reports, {"L1": 0.5, "L3": 0.5})
        coverage = _strict_scoring_profile_coverage({}, scoring_reports=reports, layer_weights={"L1": 0.5, "L3": 0.5})
    assert score == pytest.approx(0.6)
    assert coverage["complete"]
    assert not coverage["score_grounding_complete"]
    assert coverage["grounded_score_fraction"] == pytest.approx(0.1)


def test_budget_context_preserves_ownership_checks_and_enums():
    from benchmark.visual_judge.adaptive_context import budget_adaptive_context
    context = {
        "metric": "semantic_placement_consistency",
        "required_placement_checks": [{"check_id": "check1", "target_ids": ["chair"]}],
        "functional_ownership_ledger": {"events": [{"event_id": "event", "target_ids": ["chair"]}]},
        "response_contract": {"observation_status": ["observed", "inferred_under_budget"]},
        "optional_audit": "z" * 5000,
    }
    payload = json.loads(budget_adaptive_context(context, 800))
    for key in ("required_placement_checks", "functional_ownership_ledger", "response_contract"):
        assert payload[key] == context[key]
    assert "json_prefix" not in json.dumps(payload)
    with pytest.raises(ContextCapacityError):
        budget_adaptive_context(context, 100)


def test_atomic_shards_exact_once_and_ownership_kept():
    from benchmark.visual_judge.adaptive_context import call_with_atomic_shards
    req = request("functional_consistency")
    req["required_functional_checks"] = [{"check_id": value, "target_ids": ["chair"]} for value in ["c", "a", "b"]]
    req["functional_ownership_ledger"] = {"events": [{"event_id": "do_not_drop"}]}
    calls = []
    def invoke(part):
        checks = part["required_functional_checks"]
        if len(checks) > 1:
            raise ContextCapacityError("test-sized context")
        assert part["functional_ownership_ledger"] == req["functional_ownership_ledger"]
        calls.append(checks[0]["check_id"])
        return {**response(), "functional_check_results": [
            {"check_id": check["check_id"], "target_ids": ["chair"], "conclusion": "valid",
             "observation_status": "inferred_under_budget", "reason": "inferred"} for check in checks
        ]}
    result = call_with_atomic_shards(req, invoke)
    assert calls == ["a", "b", "c"]
    assert sorted(row["check_id"] for row in result["functional_check_results"]) == ["a", "b", "c"]


def test_duplicate_input_check_ids_fail_before_any_model_call():
    from benchmark.visual_judge.evidence_resolution import EvidenceIntegrityError
    req = request("functional_consistency")
    req["required_functional_checks"] = [{"check_id": "same"}, {"check_id": "same"}]
    model = Model(response())
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    with pytest.raises(EvidenceIntegrityError):
        judge._adjudicate_scene_quality_raw(req)
    assert not model.calls


def test_bad_image_is_quarantined_remaining_image_delivered(picture, tmp_path):
    req = request("scale_consistency", [str(tmp_path / "missing.png"), picture])
    model = Model(response())
    raw = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(req)
    assert raw["images_used"] == [picture]
    assert raw["evidence_resolution"]["excluded_images"]


def test_terminal_repair_once_preserves_images_and_valid_conclusion(picture):
    from benchmark.visual_judge.contracts import ResponseSchemaRepairError
    broken = response()
    broken["confidence"] = "invalid number"
    model = Model([broken, response()])
    raw = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(
        request("scale_consistency", [picture]))
    assert raw["verdict"] == "valid"
    assert len(model.calls) == 2
    assert model.calls[1][:len(model.calls[0])] == model.calls[0]
    model = Model([broken, response("invalid")])
    with pytest.raises(ResponseSchemaRepairError):
        OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(
            request("scale_consistency", [picture]))
    assert len(model.calls) == 2


def test_collision_wrong_dsl_is_repaired_inside_one_boundary(picture):
    initial = {"status": "need_more_evidence", "confidence": 0.4, "reason": "Need a relation view",
               "defects": [], "evidence_request": {"target_ids": ["chair"],
               "missing_observations": ["orientation_visible"], "view_goal": "observe", "metadata": {}}}
    repaired = deepcopy(initial)
    repaired["evidence_request"]["missing_observations"] = ["joint_visibility"]
    model = Model([initial, repaired])
    result = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_p0b_raw(
        request("collision", [picture], terminal=False), _allow_need_more_evidence=True)
    assert result["status"] == "need_more_evidence"
    assert len(model.calls) == 2
    assert not result["evidence_resolution"]["accepted"]


def test_default_style_weak_evidence_skips_terminal_model():
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    metric = "style_consistency"
    model = Model(AssertionError("no appearance model call allowed"))
    report = evaluate_scene_quality_interfaces(
        request(metric, appearance=False)["scene_summary"],
        config={"enabled": True, "metrics": {name: {"enabled": name == metric} for name in SCENE_QUALITY_INTERFACE_METRICS}},
        render_evidence=[], vlm_judge=OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY),
        metric_applicability={metric: {"applicability": "relevant"}},
    )
    assert report["score"] == 1.0, report["metrics"][metric]
    assert not model.calls
    assert report["resolution_coverage"]["style_policy_fraction"] == 1.0


def complete_required_rows(messages):
    text = messages[1]["content"][0]["text"]
    context = json.loads(text.split("\n", 1)[1])
    result = response()
    result["functional_check_results"] = [
        {"check_id": check["check_id"], "target_ids": check["target_ids"],
         "observation_status": "inferred_under_budget", "conclusion": "valid", "reason": "Supported by structured facts."}
        for check in context.get("required_functional_checks") or []
    ]
    result["placement_check_results"] = [
        {"check_id": check["check_id"], "subject_id": check["subject_id"], "context_ids": check.get("context_ids") or [],
         "observation_status": "inferred_under_budget", "conclusion": "valid", "reason": "Supported by structured facts."}
        for check in context.get("required_placement_checks") or []
    ]
    result["group_global_observations"] = [
        {"group_id": group["group_id"], "object_ids": group["object_ids"], "related_group_ids": [],
         "global_position_observation": "The room position is inferred from canonical coordinates.",
         "inter_group_observation": "Relative positions are inferred from canonical coordinates.",
         "evidence_sufficiency": "partial_but_usable", "residual_issue_candidate": "none"}
        for group in context.get("object_groups") or []
    ]
    return result


@pytest.mark.parametrize("metric", ["functional_consistency", "semantic_placement_consistency"])
def test_mock_default_l3_workflow_zero_images(metric):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    scene = request(metric)["scene_summary"]
    scene["objects"].append({**deepcopy(scene["objects"][0]), "id": "desk", "category": "desk", "center": [3, 3, 1]})
    model = Model(complete_required_rows)
    report = evaluate_scene_quality_interfaces(
        scene, config={"enabled": True, "metrics": {name: {"enabled": name == metric} for name in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={"object_groups": [{"group_id": "work", "object_ids": ["chair", "desk"]}]},
        render_evidence=[], vlm_judge=OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY),
        metric_applicability={metric: {"applicability": "relevant"}},
    )
    result = report["metrics"][metric]
    assert result["status"] == "evaluated", {key: result.get(key) for key in ("reason", "judgement", "resolution_coverage")}
    assert report["score"] == 1.0
    assert report["resolution_coverage"]["complete"]
    assert report["resolution_coverage"]["visual_evidence_fraction"] == 0.0
    assert model.calls


def test_persisted_scores_keep_new_acceptance_and_old_visual_coverage():
    from benchmark.camera_cal_scene_level.persisted_scoring import case_scoring_summary, SCORING_METRIC_ORDER
    from benchmark.evaluator.adaptive_audit import coverage_from_units
    metric = {
        "status": "evaluated", "score": 0.8, "evidence_resolution_policy": ADAPTIVE_POLICY,
        "resolution_coverage": coverage_from_units([{
            "accepted": True, "model_judgement_completed": True, "visual_observation_complete": False,
            "evidence_tier": "structured_fallback", "decision_source": "model",
        }], complete=True),
    }
    l1 = {"metrics": {name: deepcopy(metric) for layer, name, _ in SCORING_METRIC_ORDER if layer == "L1"}}
    l3 = {"metrics": {name: {**deepcopy(metric), "weight": 0.2} for layer, name, _ in SCORING_METRIC_ORDER if layer == "L3"}}
    summary = case_scoring_summary(
        case_id="mock", case_manifest={"scoring_profile": {"layer_weights": {
            "l1_physical_plausibility": 0.5, "l3_scene_quality": 0.5,
        }}}, l1_report=l1, l3_report=l3,
    )
    assert summary["combined_status"] == "complete"
    assert summary["combined_score_100"] == pytest.approx(80.0)
    assert summary["combined_coverage_fraction"] == 0.0
    assert summary["resolution_coverage"]["model_judgement_fraction"] == 1.0


def test_legacy_policy_does_not_relabel_persisted_reports():
    from benchmark.camera_cal_scene_level.persisted_scoring import case_scoring_summary
    value = case_scoring_summary(case_id="old", case_manifest={}, l1_report={}, l3_report={})
    assert "evidence_resolution_policy" not in value


@pytest.mark.parametrize("metric", ["support", "oob"])
def test_public_partial_packet_preserves_single_raw_view(metric, picture):
    from benchmark.visual_judge.p0b import adjudicate_p0b_event
    model = Model(response("invalid"))
    result = adjudicate_p0b_event(
        metric=metric, event={"object_id": "chair"}, prompt="", relationships=[],
        scene=request(metric)["scene_summary"], detector_evidence={"positive_gap": 0.3},
        judge=OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY),
        local_view_provider=lambda _: [{"path": picture, "role": "metric_local_rgb", "metric": metric}],
    )
    assert result["verdict"] == "invalid"
    assert result["evidence_resolution"]["evidence_tier"] == "partial_visual"
    assert result["evidence_resolution"]["images_used"] == [picture]
    assert len(model.calls) == 1


@pytest.mark.parametrize("metric", ["collision", "style_consistency"])
def test_exhausted_control_budget_does_not_reacquire(metric, picture):
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    from benchmark.visual_judge.control_config import resolve_vlm_evaluation_control
    provider_calls = []
    def provider(req):
        provider_calls.append(req)
        raise AssertionError("zero remaining rounds cannot render")
    model = Model(response("invalid"))
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    control = resolve_vlm_evaluation_control({
        "evidence_resolution_policy": ADAPTIVE_POLICY, "budgets": {"max_evidence_rounds": 0},
    })
    wrapper = ControlledVLMJudge(judge, control=control, camera_provider=provider)
    req = request(metric, appearance=False, terminal=False)
    req["decision_mode"] = "final"
    result = (wrapper.adjudicate_p0b(req) if metric == "collision" else wrapper.adjudicate_scene_quality(req))
    assert not provider_calls
    assert result["evidence_resolution"]["accepted"]
    assert not result["evidence_resolution"]["visual_observation_complete"]
    assert len(model.calls) == (1 if metric == "collision" else 0)
    assert wrapper.control.max_evidence_rounds == 0


def test_style_policy_unit_does_not_clear_other_confirmed_defect():
    from benchmark.evaluator.scene_quality.adaptive_acceptance import adaptive_metric_result
    from benchmark.visual_judge.evidence_resolution import attach_resolution
    strong = request("style_consistency", appearance=True)
    weak = request("style_consistency", appearance=False)
    strong["event"] = {"scope": "strong"}
    weak["event"] = {"scope": "weak"}
    bad = attach_resolution(response("invalid"), prepare_adaptive_request(strong, terminal=True, trigger="test"))
    policy = style_policy_result(prepare_adaptive_request(weak, terminal=True, trigger="test"))
    original = {"status": "evaluated", "score": 0.5, "group_judgements": [
        {"status": "evaluated", "score": 0.0, "judgement": bad},
        {"status": "evaluated", "score": 1.0, "judgement": policy},
    ]}
    with evidence_policy_scope({"evidence_resolution_policy": ADAPTIVE_POLICY}):
        result = adaptive_metric_result(lambda: deepcopy(original))()
    assert result["score"] == 0.5
    assert result["group_judgements"][0]["judgement"]["defects"] == bad["defects"]
    assert result["resolution_coverage"]["style_policy_fraction"] == 0.5
    assert result["resolution_coverage"]["model_judgement_fraction"] == 0.5


def test_unjudged_check_stays_in_denominator():
    from benchmark.evaluator.scene_quality.adaptive_acceptance import summarize_metric_resolution
    from benchmark.visual_judge.evidence_resolution import attach_resolution
    req = prepare_adaptive_request(request("functional_consistency"), terminal=True, trigger="test")
    raw = attach_resolution({**response(), "functional_check_results": [{
        "check_id": "done", "target_ids": ["chair"], "conclusion": "valid", "observation_status": "inferred_under_budget",
    }]}, req)
    summary = summarize_metric_resolution({"status": "evaluated", "score": 1.0, "judgement": raw,
        "functional_check_ledger": {"checks": [{"check_id": "done"}, {"check_id": "missing"}]}})
    assert summary["eligible_count"] == 2
    assert summary["acceptance_fraction"] == 0.5
    assert not summary["complete"]


@pytest.mark.parametrize("kind", ["duplicate_object", "unknown_target", "wrong_hash", "missing_check"])
def test_integrity_errors_are_not_fallback(kind, picture):
    from benchmark.visual_judge.evidence_resolution import EvidenceIntegrityError
    req = request("functional_consistency", [picture])
    if kind == "duplicate_object":
        req["scene_summary"]["objects"].append(deepcopy(req["objects"][0]))
    elif kind == "unknown_target":
        req["selected_object_ids"] = ["unknown"]
    elif kind == "wrong_hash":
        req["render_evidence"] = [{"path": picture, "sha256": "0" * 64}]
    else:
        req["required_functional_checks"] = [{"target_ids": ["chair"]}]
    model = Model(response())
    with pytest.raises(EvidenceIntegrityError):
        OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(req)
    assert not model.calls


def test_repair_transport_failure_keeps_service_category():
    broken = response()
    broken["confidence"] = "not a number"
    model = Model([broken, ConnectionError("offline")])
    with pytest.raises(Exception) as error:
        OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(
            request("scale_consistency"))
    assert failure_record(error.value, phase="judge")["failure_category"] == "model_service_failure"
    assert len(model.calls) == 2


def test_public_api_mock_end_to_end_policy_and_persistence(tmp_path):
    from benchmark.api.evaluation import run_evaluate
    from test_current_evaluation_profile import _scene
    model = Model(complete_required_rows)
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)
    output = tmp_path / "report.json"
    report = run_evaluate(
        scene=_scene(), out=output, vlm_judge=judge,
        asset_policy={"mode": "generated_or_open_assets", "arrangement_owner": "generator",
                      "category_selection_owner": "generator", "scale_owner": "generator",
                      "appearance_owner": "generator"}, render_evidence=[],
        vlm_evaluation_control={"evidence_resolution_policy": ADAPTIVE_POLICY,
                                "budgets": {"max_evidence_rounds": 0}},
    )
    control = report["evaluation_config"]["vlm_evaluation_control"]
    assert control["effective"]["evidence_resolution_policy"] == ADAPTIVE_POLICY
    assert control["effective"]["budgets"]["max_evidence_rounds"] == 0
    assert control["evidence_resolution_implementation"]["source_tree_sha256"]
    assert report["benchmark_score"] is not None, report["reports"]["scene_quality"]
    assert json.loads(output.read_text())["benchmark_score"] == report["benchmark_score"]
    assert model.calls
    assert report["coverage"]["grounded_score_fraction"] == 0.0
    assert not report["coverage"]["score_grounding_complete"]
    assert report["resolution_coverage"]["complete"]
    assert report["resolution_coverage"]["visual_evidence_fraction"] == 0.0
    from jsonschema import Draft202012Validator
    metric_calls = [messages for messages in model.calls if any(
        isinstance(message.get("content"), list) and any(
            '"adaptive_evidence"' in part.get("text", "") for part in message["content"]
        ) for message in messages
    )]
    assert report["scoring_reliability"]["judge_episode_count"] == len(metric_calls)
    schema = json.loads((Path(__file__).parents[1] / "schemas/evaluation_report.schema.json").read_text())
    errors = list(Draft202012Validator({"$ref": "#/$defs/canonicalReportV2", "$defs": schema["$defs"]}).iter_errors(report))
    assert not errors, [(list(error.absolute_path), error.validator, error.message[:300]) for error in errors[:8]]



def test_real_model_context_shards_large_check_set_losslessly():
    req = request("functional_consistency")
    req["required_functional_checks"] = [
        {"check_id": f"check-{i:02d}", "target_ids": ["chair"], "check_type": "object_clearance",
         "owner_stage": "scene_global", "observation_goals": ["resolve " + "complete context " * 90]}
        for i in range(12)
    ]
    req["functional_ownership_ledger"] = {"schema_version": "test-ledger", "events": [
        {"event_id": "owner", "owning_metric": "functional_consistency", "scoring_target_ids": ["chair"],
         "reason": "This ownership event must survive every shard."}
    ]}
    model = Model(complete_required_rows)
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY, max_context_chars=10000)
    raw = judge._adjudicate_scene_quality_raw(req)
    assert len(model.calls) > 1
    ids = []
    for messages in model.calls:
        context_text = messages[1]["content"][0]["text"].split("\n", 1)[1]
        assert len(context_text) <= 10000
        assert "json_prefix" not in context_text
        context = json.loads(context_text)
        assert context["functional_ownership_ledger"]["events"][0]["event_id"] == "owner"
        ids.extend(check["check_id"] for check in context["required_functional_checks"])
        for check in context["required_functional_checks"]:
            assert check["observation_goals"] == req["required_functional_checks"][0]["observation_goals"]
    assert ids == [check["check_id"] for check in req["required_functional_checks"]]
    assert raw["evidence_resolution"]["model_call_count"] == len(model.calls)


def test_unavailable_l3_packet_keeps_good_image_and_reports_bad_one(picture, tmp_path):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    metric = "scale_consistency"
    model = Model(response())
    report = evaluate_scene_quality_interfaces(
        request(metric)["scene_summary"],
        config={"enabled": True, "metrics": {name: {"enabled": name == metric} for name in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={"object_groups": [{"group_id": "single", "object_ids": ["chair"]}]},
        render_evidence=[],
        camera_evidence_provider=lambda _: {"status": "unavailable", "paths": [str(tmp_path / "missing.png"), picture]},
        vlm_judge=OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY),
        metric_applicability={metric: {"applicability": "relevant"}},
    )
    assert report["score"] == 1.0
    units = report["resolution_coverage"]["units"]
    assert units[0]["images_used"] == [picture]
    assert units[0]["evidence_tier"] == "partial_visual"
    assert units[0]["excluded_images"]
    assert report["resolution_coverage"]["visual_evidence_fraction"] == 0.0


def test_l3_camera_service_error_never_becomes_valid():
    from benchmark.evaluator.scene_quality.interfaces import _request_scene_quality_evidence
    def provider(_):
        raise ConnectionError("offline")
    with evidence_policy_scope({"evidence_resolution_policy": ADAPTIVE_POLICY}):
        with pytest.raises(ConnectionError):
            _request_scene_quality_evidence(
                provider, metric_name="scale_consistency", policy={"camera_scope": "global"},
                scene=request("scale_consistency")["scene_summary"], prompt=None,
                selected_object_ids=["chair"], selected_group_ids=[], selected_groups=[],
            )
