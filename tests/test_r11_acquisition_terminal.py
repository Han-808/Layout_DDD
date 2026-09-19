"""Exercise actual acquisition -> controlled Judge -> model delivery, offline."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from PIL import Image

from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.acquisition_outcome import AcquisitionExhausted, acquire_evidence
from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
from benchmark.visual_judge.evidence_gap_v2 import EvidenceGapError, FALLBACK_POLICY
from benchmark.visual_judge.evidence_resolution import (
    AdaptiveEvidenceError, bind_acquisition_resolution, evidence_policy_scope,
    prepare_adaptive_request,
)
from test_evidence_adaptive_judgement import Model, complete_required_rows, picture
from test_postrun_policy_boundary import (
    ambiguous_response, request, v2_control, valid_response,
)


def test_real_camera_empty_bank_emits_exhaustion_before_render(monkeypatch, tmp_path):
    from benchmark.visual_judge import render_views
    from test_camera_pose_modes import _FakeRenderer, _request
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"offline fixture")
    renderer = _FakeRenderer()
    provider = render_views.CameraEvidenceProvider(renderer=renderer, blend_file=blend,
        out_dir=tmp_path / "evidence", mode="bbox_track", max_views=2)
    monkeypatch.setattr(render_views, "generate_camera_pose_candidates", lambda *args, **kwargs: [])
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        with pytest.raises(AcquisitionExhausted):
            provider(_request())
    assert provider.last_call_usage["candidate_count_generated"] == 0
    assert renderer.calls == []


@pytest.mark.parametrize("metric", ["scale_consistency", "object_pairing_consistency", "style_consistency",
                                    "functional_consistency", "semantic_placement_consistency"])
def test_public_l3_initial_exhaustion_reaches_actual_terminal_model(picture, metric):
    from benchmark.evaluator.scene_quality import interfaces
    from test_placement_proposal_handoff import global_request
    req = global_request(picture)
    acquired = []
    def provider(value):
        acquired.append(deepcopy(value))
        raise AcquisitionExhausted("bounded_bank_empty", audit={
            "available_items": [picture] if metric == "object_pairing_consistency" else []})
    model = Model(complete_required_rows)
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control(), camera_provider=provider)
    report = interfaces.evaluate_scene_quality_interfaces(req["scene_summary"],
        config={"enabled": True, "metrics": {name: {"enabled": name == metric,
            "evidence_policy": {"camera_scope": "global" if name == "object_pairing_consistency" else "group_local"},
            "evidence_plan": {"evidence_strategy": "global_and_local", "router_options": None}}
            for name in interfaces.SUPPORTED_SCENE_QUALITY_METRICS}},
        object_grouping_report={"object_groups": req["object_groups"]},
        render_evidence=[] if metric == "object_pairing_consistency" else [picture],
        camera_evidence_provider=provider, vlm_judge=judge,
        metric_applicability={metric: {"applicability": "relevant"}})
    assert acquired
    terminals = [row for row in judge.audit_records if row.get("stage") == "adaptive_terminal"]
    assert terminals, report["metrics"][metric]
    assert any(row["resolution"]["images_used"] == [picture] for row in terminals)
    assert model.calls
    assert report["metrics"][metric]["status"] == "evaluated", report["metrics"][metric]


@pytest.mark.parametrize("metric", ["collision", "support", "oob"])
def test_p0b_initial_exhaustion_reuses_overview(picture, metric):
    from benchmark.visual_judge.p0b import adjudicate_p0b_event
    model = Model({"status": "valid", "confidence": 0.8, "reason": "Synthetic supported result", "defects": []})
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control())
    req = request(metric)
    def provider(value):
        raise AcquisitionExhausted("empty")
    result = adjudicate_p0b_event(metric=metric, event={"object_id": "chair"}, prompt="",
        relationships=[], scene=req["scene_summary"], detector_evidence={}, judge=judge,
        object_ids=["chair"], overview_render_evidence=[picture], local_view_provider=provider)
    assert len(model.calls) == 1
    assert judge.audit_records[-1]["stage"] == "adaptive_terminal"
    assert judge.audit_records[-1]["resolution"]["images_used"] == [picture]
    assert result["verdict"] == "valid"


def test_controller_repair_exhaustion_preserves_old_and_partial_new_images(picture, tmp_path):
    extra = tmp_path / "partial.png"
    Image.open(picture).save(extra)
    calls = []
    def provider(value):
        calls.append(value)
        raise AcquisitionExhausted("bounded", audit={"available_items": [str(extra)]})
    model = Model([ambiguous_response(), valid_response()])
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control(), camera_provider=provider)
    req = request("scale_consistency", [picture])
    req.update(adaptive_terminal=False, evidence_phase="final", target_ids=["chair"])
    result = judge.adjudicate_scene_quality(req)
    assert len(calls) == 1
    assert len(model.calls) == 2
    assert set(result["evidence_resolution"]["images_used"]) == {picture, str(extra)}


def test_terminal_reclaims_unsent_images_without_new_acquisition(picture, tmp_path):
    extra = tmp_path / "unsent.png"
    Image.open(picture).save(extra)
    req = request("style_consistency", [picture])
    bind_acquisition_resolution(req, {"scope_satisfied": False, "images_not_delivered": [str(extra)],
        "failure": {"failure_category": "evidence_unavailable", "recoverable_acquisition": True}})
    model = Model(valid_response())
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control())
    result = judge.adjudicate_scene_quality(req)
    assert len(model.calls) == 1
    assert set(result["evidence_resolution"]["images_used"]) == {picture, str(extra)}
    assert result["evidence_resolution"]["images_not_delivered"] == []
    images = [item for msg in model.calls[0] if isinstance(msg["content"], list)
              for item in msg["content"] if item.get("type") == "image_url"]
    assert len(images) == 2


@pytest.mark.parametrize("verdict", ["valid", "ambiguous"])
@pytest.mark.parametrize("metric", ["scale_consistency", "object_pairing_consistency", "style_consistency"])
def test_zero_image_judge_receives_metric_facts_and_can_keep_gap(verdict, metric):
    req = request(metric)
    if metric == "style_consistency":
        req["scene_summary"]["objects"][0]["asset_metadata"] = {"material": "wood"}
    req.update(adaptive_terminal=True, evidence_phase="final")
    model = Model(valid_response() if verdict == "valid" else ambiguous_response())
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control())
    if verdict == "ambiguous":
        with pytest.raises(EvidenceGapError) as caught:
            judge.adjudicate_scene_quality(req)
        assert caught.value.audit["coverage_gap"]["score"] is None
    else:
        result = judge.adjudicate_scene_quality(req)
        assert result["evidence_resolution"]["evidence_tier"] == "structured_fallback"
    assert len(model.calls) == 1
    assert "metric_structured_evidence_v1" in json.dumps(model.calls)


@pytest.mark.parametrize("error,category", [(ConnectionError("secret"), "model_service_failure"),
    (FileNotFoundError("secret"), "input_integrity_failure"), (ValueError("secret"), "implementation_or_input_failure"),
    (OSError("secret"), "infrastructure_failure")])
def test_hard_initial_failure_never_calls_terminal_model(picture, error, category):
    def provider(value):
        raise error
    req = request("scale_consistency", [picture])
    with evidence_policy_scope(req):
        outcome = acquire_evidence(provider, req)
    bind_acquisition_resolution(req, {"scope_satisfied": False, "failure": outcome.audit["failure"]})
    model = Model(AssertionError("must not call"))
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control())
    with pytest.raises(AdaptiveEvidenceError) as caught:
        judge.adjudicate_scene_quality(req)
    assert caught.value.failure_category == category
    assert model.calls == []
    assert "secret" not in repr(outcome.audit)


@pytest.mark.parametrize("metric", ["collision", "support", "oob", "functional_consistency", "semantic_placement_consistency", "style_consistency"])
def test_proxy_box_alone_is_not_metric_permission_for_text_only_judge(metric):
    with pytest.raises(EvidenceGapError):
        prepare_adaptive_request(request(metric), terminal=True, trigger="empty")


def test_provider_render_failure_is_not_normal_exhaustion(picture):
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        outcome = acquire_evidence(lambda _: {"status": "failed", "failure_category": "render_failure", "paths": [picture]}, {})
    assert outcome.items == [picture]
    assert not outcome.audit["failure"]["recoverable_acquisition"]
    with pytest.raises(AdaptiveEvidenceError):
        outcome.raise_if_failed()


@pytest.mark.parametrize("metric", ["style_consistency", "functional_consistency", "semantic_placement_consistency"])
def test_default_global_first_route_carries_initial_outcome(picture, metric):
    from benchmark.evaluator.scene_quality import interfaces
    from test_placement_proposal_handoff import global_request
    req = global_request(picture)
    def provider(value):
        raise AcquisitionExhausted("partial_bank", audit={"available_items": [picture]})
    model = Model(complete_required_rows)
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control(), camera_provider=provider)
    report = interfaces.evaluate_scene_quality_interfaces(req["scene_summary"],
        config={"enabled": True, "metrics": {name: {"enabled": name == metric}
            for name in interfaces.SUPPORTED_SCENE_QUALITY_METRICS}},
        object_grouping_report={"object_groups": req["object_groups"]}, render_evidence=[],
        camera_evidence_provider=provider, vlm_judge=judge,
        metric_applicability={metric: {"applicability": "relevant"}})
    terminals = [row for row in judge.audit_records if row.get("stage") == "adaptive_terminal"]
    assert terminals, report["metrics"][metric]
    assert any(picture in row["resolution"]["images_used"] for row in terminals)


def test_one_group_hard_acquisition_failure_does_not_erase_other_group(picture):
    from benchmark.evaluator.scene_quality import interfaces
    from test_placement_proposal_handoff import global_request
    req = global_request(picture)
    scene = req["scene_summary"]
    scene["objects"] += [{**deepcopy(obj), "id": obj["id"] + "2"} for obj in scene["objects"]]
    groups = [*req["object_groups"], {"group_id": "work2", "object_ids": ["chair2", "desk2"]}]
    def provider(value):
        if "chair" in value["object_ids"]:
            raise ConnectionError("secret")
        raise AcquisitionExhausted("empty")
    model = Model(complete_required_rows)
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control(), camera_provider=provider)
    report = interfaces.evaluate_scene_quality_interfaces(scene,
        config={"enabled": True, "metrics": {name: {"enabled": name == "scale_consistency"}
            for name in interfaces.SUPPORTED_SCENE_QUALITY_METRICS}},
        object_grouping_report={"object_groups": groups}, render_evidence=[picture],
        camera_evidence_provider=provider, vlm_judge=judge,
        metric_applicability={"scale_consistency": {"applicability": "relevant"}})
    metric = report["metrics"]["scale_consistency"]
    results = {row["group_id"]: row for row in metric["group_results"]}
    assert results["work"]["status"] == "failed"
    assert results["work2"]["status"] == "evaluated"
    assert metric["score"] is None
    assert len(model.calls) == 1


@pytest.mark.parametrize("error", [ValueError("bad input"), ConnectionError("offline"), OSError("renderer")])
def test_nonrect_empty_counter_cannot_relabel_hard_fault(error):
    pytest.importorskip("benchmark.non_rectangular.runtime", reason="Legacy continuity wrapper is not the uniform acquisition boundary")
    from benchmark.non_rectangular.runtime import _NonRectangularContinuityCameraProvider
    class Provider:
        last_call_usage = {"candidate_count_generated": 0}
        def __call__(self, request):
            raise error
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        with pytest.raises(type(error)) as caught:
            _NonRectangularContinuityCameraProvider(Provider())({})
    assert caught.value is error


@pytest.mark.parametrize("fault", ["gap", "service", "partial"])
def test_specialized_functional_probe_uses_same_outcome(picture, fault):
    from benchmark.evaluator.scene_quality.functional_probe import acquire_functional_probe_evidence
    discovery = {"schema_version": "functional_discovery_v3", "inspected_object_ids": ["a", "b"],
        "directed_surface_targets": [], "within_group_correspondences": [],
        "cross_group_correspondences": [{"discovery_id": "relation_01", "target_ids": ["a", "b"],
            "group_ids": ["group_a", "group_b"], "observation_kinds": ["mutual_orientation"],
            "observation_goal": "show a and b together"}],
        "approach_clearance_targets": [], "boundary_sensitive_targets": [],
        "unusual_unconfirmed": [], "reason": "synthetic", "provenance": {}}
    class Planner:
        def discover_functional_evidence(self, req):
            return deepcopy(discovery)
    def provider(req):
        if fault == "service":
            raise ConnectionError("secret")
        if fault == "gap":
            raise AcquisitionExhausted("bounded")
        return {"status": "insufficient", "paths": [picture]}
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        paths, audit = acquire_functional_probe_evidence(planner=Planner(), provider=provider,
            scene={"scene_id": "scene", "scene_type": "living_room",
                   "objects": [{"id": i, "category": "object"} for i in ["a", "b"]]},
            global_image_path=picture, max_probe_units=1,
            groups=[{"group_id": "group_" + i, "object_ids": [i]} for i in ["a", "b"]])
    row = audit["probe_results"][0]
    assert row["acquisition_outcome"]["schema_version"] == "acquisition_outcome_v1"
    if fault == "service":
        assert audit["status"] == "failed"
        assert audit["failure"]["failure_category"] == "model_service_failure"
        assert "secret" not in json.dumps(audit)
    else:
        assert audit["status"] != "failed"
    assert paths == ([picture] if fault == "partial" else [])


def test_terminal_context_limit_is_honest_about_remaining_unsent(picture, tmp_path):
    paths = [picture]
    for index in range(9):
        extra = tmp_path / f"view{index}.png"
        Image.open(picture).save(extra)
        paths.append(str(extra))
    req = request("style_consistency", paths[:2])
    req.update(adaptive_terminal=True, adaptive_unselected_images=paths[2:])
    model = Model(valid_response())
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model, max_images=3), control=v2_control())
    result = judge.adjudicate_scene_quality(req)
    resolution = result["evidence_resolution"]
    assert resolution["images_used"] == [paths[0], *paths[-2:]]
    assert len(resolution["images_not_delivered"]) == 7
    assert not resolution["visual_observation_complete"]


def test_terminal_missing_supplied_artifact_is_input_failure(picture, tmp_path):
    req = request("scale_consistency", [picture])
    req.update(adaptive_terminal=True, adaptive_unselected_images=[str(tmp_path / "missing.png")])
    model = Model(AssertionError("must not call"))
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control())
    with pytest.raises(AdaptiveEvidenceError) as caught:
        judge.adjudicate_scene_quality(req)
    assert caught.value.failure_category == "input_integrity_failure"
    assert not model.calls


def test_terminal_retains_initial_images_after_controller_window_replacement(monkeypatch, picture, tmp_path):
    from types import SimpleNamespace
    from benchmark.visual_judge.orchestration.controller import VLMEvaluationController
    extra = tmp_path / "replacement.png"
    Image.open(picture).save(extra)
    monkeypatch.setattr(VLMEvaluationController, "run", lambda *args, **kwargs: SimpleNamespace(
        status="unresolved", stop_reason="max_total_images_exhausted", audit={}, visual_evidence=[str(extra)]))
    model = Model(valid_response())
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model), control=v2_control())
    result = judge.adjudicate_scene_quality(request("scale_consistency", [picture]))
    assert set(result["evidence_resolution"]["images_used"]) == {picture, str(extra)}
    assert len(model.calls) == 1
    assert judge.audit_records[-1]["stage"] == "adaptive_terminal"


@pytest.mark.parametrize("malformed", [False, True])
def test_nonrect_action_feasibility_gap_is_distinct_from_malformed_pose(monkeypatch, malformed):
    camera = pytest.importorskip("benchmark.non_rectangular.camera")
    monkeypatch.setattr(camera, "polygon_geometry_from_scene", lambda scene: object())
    monkeypatch.setattr(camera, "_gate_candidate", lambda *args, **kwargs: None)
    pose = {"location": [1, 1, 1], "target": [0, 0] if malformed else [0, 0, 0]}
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        with pytest.raises(ValueError if malformed else AcquisitionExhausted):
            camera.revalidate_polygon_camera_pose(pose, scene={})


def test_functional_text_only_admission_requires_scoped_measurement_rows():
    req = request("functional_consistency")
    req["required_functional_checks"] = [{"check_id": "owned"}]
    req["functional_probe_evidence"] = {"functional_measurements": {"check_measurements": [
        {"check_id": "other", "status": "complete", "distance_m": 1.2}]}}
    with pytest.raises(EvidenceGapError):
        prepare_adaptive_request(req, terminal=True, trigger="bounded")
    req["functional_probe_evidence"]["functional_measurements"]["check_measurements"].append(
        {"check_id": "owned", "status": "partial", "distance_m": 1.5})
    value = prepare_adaptive_request(req, terminal=True, trigger="bounded")
    rows = value["adaptive_evidence"]["structured_evidence"]["functional_measurements"]["check_measurements"]
    assert [row["check_id"] for row in rows] == ["owned"]
