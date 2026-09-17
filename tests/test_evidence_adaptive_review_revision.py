"""Public-boundary regressions from the independent r1 cross-review.

All providers/models are local mocks. No external rendering or API is used.
"""

from copy import deepcopy
from pathlib import Path
import hashlib

import pytest
from PIL import Image

from test_evidence_adaptive_judgement import Model, complete_required_rows, request, response
from test_current_evaluation_profile import _scene
from benchmark.api.evaluation import run_evaluate
from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
from benchmark.visual_judge.control_config import resolve_vlm_evaluation_control
from benchmark.visual_judge.evidence_resolution import (
    ADAPTIVE_POLICY, LEGACY_POLICY, adaptive_enabled, evidence_policy_scope,
    policy_of, with_evidence_policy,
    AdaptiveEvidenceError, EvidenceIntegrityError, usable_visuals, failure_record,
)


ASSET_POLICY = {
    "mode": "generated_or_open_assets", "arrangement_owner": "generator",
    "category_selection_owner": "generator", "scale_owner": "generator",
    "appearance_owner": "generator",
}


@pytest.fixture
def image_paths(tmp_path):
    paths = []
    for name, pixel in (("initial", (0, 0, 0)), ("new", (255, 0, 0)), ("extra", (0, 0, 255))):
        path = tmp_path / (name + ".png")
        image = Image.new("RGB", (20, 20), "white")
        image.putpixel((2, 3), pixel)
        image.save(path)
        paths.append(str(path))
    return paths


def _l3_scale(model, *, provider=None, evidence=()):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    metric = "scale_consistency"
    return evaluate_scene_quality_interfaces(
        request(metric)["scene_summary"],
        config={"enabled": True, "metrics": {
            name: {"enabled": name == metric} for name in SCENE_QUALITY_INTERFACE_METRICS
        }},
        object_grouping_report={"object_groups": [{"group_id": "single", "object_ids": ["chair"]}]},
        render_evidence=list(evidence), camera_evidence_provider=provider,
        vlm_judge=OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY),
        metric_applicability={metric: {"applicability": "relevant"}},
    )


@pytest.mark.parametrize("key", ["render_evidence_items", "visual_evidence", "paths"])
def test_l3_provider_hash_is_checked_before_path_projection(image_paths, key):
    model = Model(response())
    with pytest.raises(EvidenceIntegrityError):
        _l3_scale(model, provider=lambda _: {"status": "available", key: [{
            "path": image_paths[0], "sha256": "0" * 64, "target_ids": ["chair"],
        }]})
    assert model.calls == []


def test_l3_provider_unrelated_image_is_quarantined_not_visual_credit(image_paths):
    model = Model(response())
    report = _l3_scale(model, provider=lambda _: {"status": "available", "render_evidence_items": [{
        "path": image_paths[0], "sha256": hashlib.sha256(Path(image_paths[0]).read_bytes()).hexdigest(),
        "target_ids": ["unrelated_object"], "role": "metric_local_rgb",
    }]})
    assert report["score"] == 1.0
    assert report["resolution_coverage"]["complete"]
    assert report["resolution_coverage"]["visual_evidence_fraction"] == 0.0
    unit = report["resolution_coverage"]["units"][0]
    assert unit["images_used"] == []
    assert any(item.get("reason") == "unrelated_target_scope" for item in unit["excluded_images"])
    assert len(model.calls) == 1  # Genuine structured Judge, not automatic valid.


def test_duplicate_path_cannot_hide_a_bad_provenance_claim(image_paths):
    with pytest.raises(EvidenceIntegrityError):
        usable_visuals([image_paths[0], {"path": image_paths[0], "sha256": "0" * 64}])


class AuthenticationError(RuntimeError):
    """Typed local mock, never an actual authentication request."""


@pytest.mark.parametrize("phase", ["judge", "acquisition"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_response_validation_failure_keeps_response_category(phase, wrapped):
    from benchmark.visual_judge.contracts import ResponseSchemaRepairError
    broken = response()
    broken["confidence"] = "not numeric"
    model = Model([broken, broken])
    with pytest.raises(ResponseSchemaRepairError) as raised:
        OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(
            request("scale_consistency"))
    assert isinstance(raised.value.__cause__, ValueError)
    error = raised.value
    if wrapped:
        error = RuntimeError("opaque renderer wrapper")
        error.__cause__ = raised.value
    record = failure_record(error, phase=phase)
    assert record["failure_category"] == "judge_response_failure"
    assert not record["recoverable_acquisition"]
    assert len(model.calls) == 2


@pytest.mark.parametrize("cause,category", [
    (ConnectionError("offline"), "model_service_failure"),
    (AuthenticationError("offline"), "model_service_failure"),
    (EvidenceIntegrityError("invalid source"), "input_integrity_failure"),
])
def test_response_wrapper_does_not_hide_typed_service_or_integrity_failure(cause, category):
    from benchmark.visual_judge.contracts import ResponseSchemaRepairError
    error = ResponseSchemaRepairError("repair failed", schema_audit={})
    error.__cause__ = cause
    outer = RuntimeError("opaque renderer wrapper")
    outer.__cause__ = error
    assert failure_record(error, phase="judge")["failure_category"] == category
    assert failure_record(outer, phase="acquisition")["failure_category"] == category


def _repair_judge(provider, *, verdict="valid", control=None):
    initial = {
        "evidence_status": "insufficient", "verdict": "ambiguous", "confidence": 0.3,
        "reason": "Need closer scale reference.", "missing_evidence": ["target_visible"], "defects": [],
        "evidence_request": {"target_ids": ["chair"], "missing_observations": ["target_visible"],
                             "view_goal": "view target", "metadata": {}},
    }
    model = Model([initial, response(verdict)])
    judge = ControlledVLMJudge(
        OpenAICompatibleVLMJudge(model),
        control=resolve_vlm_evaluation_control({"evidence_resolution_policy": ADAPTIVE_POLICY, **(control or {})}),
        camera_provider=provider,
    )
    return judge, model


@pytest.mark.parametrize("error_type,category", [
    (ConnectionError, "model_service_failure"), (TimeoutError, "model_service_failure"),
    (AuthenticationError, "model_service_failure"), (EvidenceIntegrityError, "input_integrity_failure"),
    (ValueError, "implementation_or_input_failure"),
])
def test_internal_repair_nonrecoverable_error_never_reaches_terminal_model(image_paths, error_type, category):
    calls = []
    def provider(value):
        calls.append(value)
        raise error_type("local negative fixture")
    judge, model = _repair_judge(provider)
    with pytest.raises(AdaptiveEvidenceError) as raised:
        judge.adjudicate_scene_quality(request("scale_consistency", [image_paths[0]], terminal=False))
    assert failure_record(raised.value, phase="acquisition")["failure_category"] == category
    assert len(calls) == 1
    assert len(model.calls) == 1
    audit = judge.audit_records[-1]["audit"]
    assert audit["failure"]["failure_category"] == category
    assert not audit["failure"]["recoverable_acquisition"]


def test_internal_repair_structured_service_failure_does_not_keep_returned_image(image_paths):
    judge, model = _repair_judge(lambda _: {"status": "unavailable",
        "failure_category": "model_service_failure", "paths": [image_paths[1]]})
    with pytest.raises(AdaptiveEvidenceError) as raised:
        judge.adjudicate_scene_quality(request("scale_consistency", [image_paths[0]], terminal=False))
    assert raised.value.failure_category == "model_service_failure"
    assert len(model.calls) == 1


def test_internal_repair_ordinary_unavailable_still_allows_a_real_fallback(image_paths):
    def provider(_):
        raise RuntimeError("schema contract HTTP auth text is not a typed service failure")
    judge, model = _repair_judge(provider, verdict="invalid")
    result = judge.adjudicate_scene_quality(request("scale_consistency", [image_paths[0]], terminal=False))
    assert result["verdict"] == "invalid"
    assert result["evidence_resolution"]["accepted"]
    assert len(model.calls) == 2
    assert result["acquisition_audit"]["failure"]["recoverable_acquisition"]


@pytest.mark.parametrize("status", ["available", "unavailable", "insufficient", "partial"])
@pytest.mark.parametrize("verdict", ["valid", "invalid"])
def test_internal_repair_keeps_partial_good_images_and_fault_audit(image_paths, tmp_path, status, verdict):
    calls = []
    missing = str(tmp_path / "missing.png")
    def provider(value):
        calls.append(value)
        return {"status": status, "render_evidence_items": [
            {"path": missing, "role": "metric_local_rgb", "target_ids": ["chair"]},
            {"path": image_paths[1], "role": "metric_local_rgb", "target_ids": ["chair"],
             "sha256": hashlib.sha256(Path(image_paths[1]).read_bytes()).hexdigest()},
        ]}
    judge, model = _repair_judge(provider, verdict=verdict)
    result = judge.adjudicate_scene_quality(request("scale_consistency", [image_paths[0]], terminal=False))
    assert result["verdict"] == verdict
    assert result["evidence_resolution"]["accepted"]
    assert result["evidence_resolution"]["evidence_tier"] == "partial_visual"
    assert set(result["images_used"]) == set(image_paths[:2])
    assert any(item.get("path") == missing for item in result["evidence_resolution"]["excluded_images"])
    assert len(calls) == 1
    assert len(model.calls) == 2
    images = [part for message in model.calls[-1] if isinstance(message["content"], list)
              for part in message["content"] if part.get("type") == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"] != images[1]["image_url"]


def test_internal_repair_image_hash_validation_precedes_projection(image_paths):
    judge, model = _repair_judge(lambda _: {"status": "unavailable", "paths": [{
        "path": image_paths[1], "sha256": "0" * 64, "target_ids": ["chair"],
    }]})
    with pytest.raises(AdaptiveEvidenceError) as raised:
        judge.adjudicate_scene_quality(request("scale_consistency", [image_paths[0]], terminal=False))
    assert raised.value.failure_category == "input_integrity_failure"
    assert len(model.calls) == 1


def test_partial_bundle_can_fit_remaining_image_budget_only_in_new_policy(image_paths):
    from benchmark.visual_judge.adapters.legacy_renderer import _fit_provider_evidence_to_budget
    images = [{"path": path, "pair_id": "same-pose"} for path in image_paths[1:]]
    selected, audit = _fit_provider_evidence_to_budget(images, remaining_images=1, allow_partial_bundles=True)
    assert selected == images[:1]
    assert audit["remaining_images"] == 1
    assert audit["dropped_evidence_count"] == 1
    with pytest.raises(RuntimeError):
        _fit_provider_evidence_to_budget(images, remaining_images=1)


@pytest.mark.parametrize("stage", ["selector", "render"])
@pytest.mark.parametrize("error_type", [ConnectionError, AuthenticationError, EvidenceIntegrityError])
def test_controller_preserves_failure_type_before_legacy_forced_choice(stage, error_type):
    from test_vlm_control_loop import _run, _gate_result, _need_more_result, _valid_result
    kwargs = {"selector_result" if stage == "selector" else "render_result": error_type("local fixture")}
    result, calls, _, judge, _, _ = _run(
        gate_results=[_gate_result(ready=True)], judge_results=[_need_more_result(), _valid_result()],
        control=resolve_vlm_evaluation_control({"evidence_resolution_policy": ADAPTIVE_POLICY}), **kwargs,
    )
    assert result.status == "unresolved"
    assert not result.audit["failure"]["recoverable_acquisition"]
    assert len(judge.requests) == 1


def test_controller_invalid_followup_contract_is_not_a_valid_fallback():
    from test_vlm_control_loop import _run, _gate_result, _need_more_result, _valid_result
    result, _, _, judge, _, _ = _run(
        gate_results=[_gate_result(ready=True), _gate_result(ready=True)],
        judge_results=[_need_more_result(), _valid_result()],
        render_result={"visual_evidence": ["repair.png"], "next_candidate_views": [{"id": ""}]},
        control=resolve_vlm_evaluation_control({"evidence_resolution_policy": ADAPTIVE_POLICY}),
    )
    assert result.status == "unresolved"
    assert result.audit["failure"]["failure_category"] == "input_integrity_failure"
    assert len(judge.requests) == 1


@pytest.mark.parametrize("batched", [False, True])
def test_preview_service_failure_keeps_failure_not_forced_choice(batched):
    from test_vlm_control_loop import (
        _PreviewReadinessSelector, _selection, _readiness_pass, _Judge, _functional_need_more_result,
        _valid_result, _Renderer, _PassThroughCandidateBankBuilder, _Gate, _gate_result, _functional_request,
    )
    from benchmark.visual_judge.control_loop import VLMEvaluationController
    calls = []
    judge = _Judge([_functional_need_more_result(), _valid_result()], calls)
    controller = VLMEvaluationController(
        judge=judge, vlm_camera_selector=_PreviewReadinessSelector(_selection(), calls, [_readiness_pass()]),
        candidate_preview_renderer=_Renderer(ConnectionError("local fixture"), calls),
        candidate_bank_builder=_PassThroughCandidateBankBuilder(),
        evidence_gate=_Gate([_gate_result(ready=True)], calls),
        renderer=_Renderer({"visual_evidence": ["repair.png"]}, calls),
        control=resolve_vlm_evaluation_control({"evidence_resolution_policy": ADAPTIVE_POLICY,
            "camera_acquisition": {"policy": "vlm_only", "vlm": {"selection_mode": "candidate_only"}}}),
    )
    result = controller.run(_functional_request(batched=batched),
                            candidate_views=({"id": "view-1"},), allowed_actions=("orbit",))
    assert result.status == "unresolved"
    assert result.audit["failure"]["failure_category"] == "model_service_failure"
    assert len(judge.requests) == 1
    preview = next(item for item in result.audit["trace"] if item["stage"] == "candidate_preview_render")
    assert preview["fallback"] is None


@pytest.mark.parametrize("judge_policy,control_policy,expected", [
    (None, None, LEGACY_POLICY),
    (None, ADAPTIVE_POLICY, ADAPTIVE_POLICY),
    (LEGACY_POLICY, ADAPTIVE_POLICY, ADAPTIVE_POLICY),
    (ADAPTIVE_POLICY, None, ADAPTIVE_POLICY),
    (ADAPTIVE_POLICY, LEGACY_POLICY, LEGACY_POLICY),
    (ADAPTIVE_POLICY, ADAPTIVE_POLICY, ADAPTIVE_POLICY),
])
def test_public_api_policy_priority_and_identity(tmp_path, judge_policy, control_policy, expected):
    model = Model(complete_required_rows)
    judge = OpenAICompatibleVLMJudge(model, **(
        {"evidence_resolution_policy": judge_policy} if judge_policy is not None else {}
    ))
    control = {"budgets": {"max_evidence_rounds": 0}}
    if control_policy is not None:
        control["evidence_resolution_policy"] = control_policy
    original = deepcopy(control)
    report = run_evaluate(
        scene=_scene(), out=tmp_path / "report.json", vlm_judge=judge,
        asset_policy=ASSET_POLICY, render_evidence=[], vlm_evaluation_control=control,
    )
    effective = report["evaluation_config"]["vlm_evaluation_control"]["effective"]
    assert effective.get("evidence_resolution_policy", LEGACY_POLICY) == expected
    assert effective["budgets"]["max_evidence_rounds"] == 0
    assert control == original
    assert judge.evidence_resolution_policy == (judge_policy or LEGACY_POLICY)
    assert policy_of() == LEGACY_POLICY  # Per-call policy did not escape its scope.
    if expected == ADAPTIVE_POLICY:
        assert report["benchmark_score"] == 1.0
        assert report["resolution_coverage"]["complete"]
        assert report["resolution_coverage"]["eligible_count"] > 0
        assert report["evidence_resolution_policy"] == ADAPTIVE_POLICY
    else:
        assert report.get("evidence_resolution_policy") is None
        for metric in report["reports"]["scene_quality"]["metrics"].values():
            assert metric.get("evidence_resolution_policy") is None


def test_controlled_judge_policy_does_not_proxy_the_underlying_default():
    judge = OpenAICompatibleVLMJudge(Model(response()))
    wrapper = ControlledVLMJudge(judge, control=resolve_vlm_evaluation_control(
        {"evidence_resolution_policy": ADAPTIVE_POLICY}
    ))
    assert policy_of(wrapper) == ADAPTIVE_POLICY
    assert judge.evidence_resolution_policy == LEGACY_POLICY


def test_nested_call_inherits_scope_and_explicit_control_can_override():
    judge = OpenAICompatibleVLMJudge(Model(response()))

    @with_evidence_policy
    def child(vlm_judge, vlm_evaluation_control=None):
        return policy_of(vlm_judge)

    with evidence_policy_scope({"evidence_resolution_policy": ADAPTIVE_POLICY}):
        assert child(judge) == ADAPTIVE_POLICY
        assert child(judge, {"budgets": {"max_evidence_rounds": 0}}) == ADAPTIVE_POLICY
        assert child(judge, {"evidence_resolution_policy": LEGACY_POLICY}) == LEGACY_POLICY
        assert adaptive_enabled()
    assert policy_of() == LEGACY_POLICY


@pytest.mark.skipif(not (Path(__file__).parents[1] / "src/benchmark/non_rectangular").is_dir(),
                    reason="Non-rect entry point belongs only to the Non-rect lane")
@pytest.mark.parametrize("judge_policy,control_policy", [
    (None, ADAPTIVE_POLICY), (ADAPTIVE_POLICY, None), (ADAPTIVE_POLICY, LEGACY_POLICY),
])
def test_nonrect_public_api_policy_priority(tmp_path, judge_policy, control_policy):
    from test_non_rectangular_polygon_evaluator import _input, _grouping_runtime
    from benchmark.non_rectangular import prepare_non_rectangular_evaluation

    inputs = _input()
    control = {"budgets": {"max_evidence_rounds": 0}}
    if control_policy is not None:
        control["evidence_resolution_policy"] = control_policy
    model = Model(complete_required_rows)
    report = run_evaluate(
        evaluation_mode="non_rectangular_multi_room", evaluation_input=inputs,
        out=tmp_path / "report.json",
        vlm_judge=OpenAICompatibleVLMJudge(model, **(
            {"evidence_resolution_policy": judge_policy} if judge_policy is not None else {}
        )),
        runtime_by_room=_grouping_runtime(prepare_non_rectangular_evaluation(inputs)),
        vlm_evaluation_control=control,
    )
    expected = control_policy or judge_policy or LEGACY_POLICY
    if expected == ADAPTIVE_POLICY:
        assert report["terminal_status"] == "complete"
        assert report["aggregate"]["overall_score"] == 1.0
        assert report["evidence_resolution_policy"] == ADAPTIVE_POLICY
        assert report["resolution_coverage"]["complete"]
    else:
        assert report.get("evidence_resolution_policy") is None
        for room in report["rooms"].values():
            assert room.get("evidence_resolution_policy") is None
