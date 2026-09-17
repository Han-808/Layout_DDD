"""Offline S100 regression: no live renderer, network or model calls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.evaluator.scene_quality.consistency_acceptance_v2 import summarize_metric, terminalize_scope
from benchmark.evaluator.scene_quality.adaptive_acceptance import _resolution_units
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, evidence_policy_scope, AdaptiveEvidenceError
from benchmark.visual_judge.acquisition_outcome import acquire_evidence
from benchmark.visual_judge.render_views import CameraEvidenceProvider, _blank_view_ids
from benchmark.scene_generation.feedback_loop.frozen_report import normalized_feedback_scores, compact_report, validate_acceptance, defect_content


def judge(verdict="valid", visual=True):
    return {"verdict": verdict, "terminal_state": "evaluated" if visual else "evaluated_degraded",
            "evidence_resolution": {"policy": FALLBACK_POLICY, "accepted": True,
                "model_judgement_completed": True, "visual_observation_complete": visual}}


def outcome(verdict="valid", visual=True):
    return {"status": "evaluated", "score": float(verdict == "valid"), "judgement": judge(verdict, visual)}


@pytest.mark.parametrize("key,label", [("scene_global_judgement", "scene_global"),
    ("residual_global_placement_judgement", "residual:0"),
    ("cross_group_relation_judgements", "relation:0"), ("target_scope_results", "target:0"),
    ("placement_global_handoff_reviews", "handoff:0")])
@pytest.mark.parametrize("verdict", ["valid", "invalid"])
@pytest.mark.parametrize("visual", [True, False])
def test_dynamic_bare_and_wrapped_judgements(key, label, verdict, visual):
    for row in (judge(verdict, visual), outcome(verdict, visual)):
        report = {**outcome(), "judgement": {key: row}}
        result = summarize_metric(report, [label])
        assert result["complete"]
        assert result["units"][0]["score"] == float(verdict == "valid")
        assert result["units"][0]["visual_observation_complete"] is visual


@pytest.mark.parametrize("change", ["naked", "unknown", "rejected", "unfinished", "failed",
                                   "explicit_none", "failure_record"])
def test_residual_never_infers_acceptance_from_verdict(change):
    row = judge()
    if change == "naked": row.pop("evidence_resolution")
    if change == "unknown": row["verdict"] = "unknown"
    if change == "rejected": row["evidence_resolution"]["accepted"] = False
    if change == "unfinished": row["evidence_resolution"]["model_judgement_completed"] = False
    if change == "failed": row["status"] = "failed"
    if change == "explicit_none": row["score"] = None
    if change == "failure_record": row["failure"] = {"failure_category": "render_failure"}
    report = {**outcome(), "judgement": {"residual_global_placement_judgement": row}}
    result = summarize_metric(report, ["residual:0"])
    assert result["accepted_count"] == 0
    assert result["units"][0]["score"] is None


def test_atomic_functional_aggregate_preserves_all_episode_receipts():
    record = {"status": "evaluated", "score": 0., "judgement": {"verdict": "invalid"},
              "check_episodes": [outcome(), outcome("invalid", False)]}
    terminalize_scope(record, phase="group")
    assert record["status"] == "evaluated"
    assert record["terminal_state"] == "evaluated_degraded"
    result = summarize_metric({**outcome(), "group_results": [{**record, "group_id": "g"}]}, ["group:g"])
    assert result["accepted_count"] == 1
    assert result["units"][0]["score"] == 0.


@pytest.mark.parametrize("kind", ["empty", "missing_receipt", "failed", "rejected"])
def test_atomic_aggregate_does_not_inherit_first_episode_acceptance(kind):
    record = {**outcome(), "check_episodes": [outcome(), outcome()]}
    if kind == "empty": record["check_episodes"] = []
    if kind == "missing_receipt": record["check_episodes"][1]["judgement"].pop("evidence_resolution")
    if kind == "failed": record["check_episodes"][1].update(status="failed", score=None)
    if kind == "rejected": record["check_episodes"][1]["judgement"]["evidence_resolution"]["accepted"] = False
    terminalize_scope(record, phase="group")
    assert record["status"] == "failed"
    assert record["score"] is None


class PreviewRenderer:
    def __init__(self, blank=("c0",), failure=None, final_blank=(), identity_blank=()):
        self.blank, self.failure = blank, failure
        self.final_blank, self.identity_blank = final_blank, identity_blank

    def render_focus_overlay_views(self, **kwargs):
        if self.failure: raise self.failure
        assert kwargs["allow_blank_views"] is True
        blank = self.blank if kwargs["preview"] else self.identity_blank
        return self.manifest(kwargs, blank)

    def render_camera_views(self, **kwargs):
        assert kwargs["allow_blank_views"] is True
        return self.manifest(kwargs, self.final_blank)

    @staticmethod
    def manifest(kwargs, blank):
        # Historical renderer records display names rather than ids.
        destination = Path(kwargs["out_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        for pose in kwargs["camera_views"]:
            (destination / (pose["id"] + ".png")).write_bytes(b"offline mock pixels")
        return {"views": [{"id": p["id"], "name": "name-" + p["id"],
                           "path": str(Path(kwargs["out_dir"]) / (p["id"] + ".png"))}
                          for p in kwargs["camera_views"]],
                "render_validation": {"blank_views": ["name-" + key for key in blank]}}


def provider(tmp_path, monkeypatch, renderer):
    selected = []
    (tmp_path / "scene.blend").write_bytes(b"offline fixture; never opened by Blender")
    def select(request):
        selected.append(request)
        return {"selected_view_ids": [c["id"] for c in request["candidates"]][:2], "action": None,
                "reason": "fixture"}
    p = CameraEvidenceProvider(renderer=renderer, blend_file=tmp_path / "scene.blend",
        out_dir=tmp_path, mode="query_cov", selector=SimpleNamespace(select_camera_views=select),
        max_views=2, candidate_count=4, max_steps=0)
    spec = {"targets": [{"id": "bookcase", "color": [1., 0., 0.], "required_for_visibility": True}]}
    monkeypatch.setattr(p, "_build_focus_spec", lambda request: deepcopy(spec))
    monkeypatch.setattr("benchmark.visual_judge.render_views.measure_focus_visibility",
                        lambda *a, **kw: {"target_pixel_fractions": {"bookcase": .5}})
    return p, selected, spec


def candidates():
    return [{"id": "c" + str(i)} for i in range(4)]


def test_one_blank_preview_keeps_three_candidates_and_final_rgb(tmp_path, monkeypatch):
    p, selected, spec = provider(tmp_path, monkeypatch, PreviewRenderer())
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        items = p._functional_probe_raw_evidence({"metric": "functional_consistency",
            "functional_probe": {}, "evidence_policy": {"scoped_image_budget": 2}}, candidates(), tmp_path,
            resolved_mode="query_cov", requested_candidate_count=4)
    assert [c["id"] for c in selected[0]["candidates"]] == ["c1", "c2", "c3"]
    assert [item["view_id"] for item in items] == ["c1", "c2"]
    manifest = json.loads((tmp_path / "camera_evidence_manifest.json").read_text())
    assert any(r["candidate_id"] == "c0" for r in manifest["selection"]["steps"][0]["candidate_rejections"])


def test_all_blank_preview_is_recoverable_without_selector_call(tmp_path, monkeypatch):
    p, selected, spec = provider(tmp_path, monkeypatch, PreviewRenderer(blank=("c0", "c1", "c2", "c3")))
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        result = acquire_evidence(lambda req: p._query_cov_selection(req, candidates(), tmp_path,
            overlay_spec=spec, require_visible_targets=True), {"metric": "functional_consistency"})
    result.raise_if_failed()
    assert result.audit["failure"]["recoverable_acquisition"] is True
    assert not selected


@pytest.mark.parametrize("error", [FileNotFoundError("fixture"), RuntimeError("renderer exited"),
                                 TimeoutError("fixture"), ValueError("corrupt manifest")])
def test_preview_hard_faults_remain_hard(tmp_path, monkeypatch, error):
    p, selected, spec = provider(tmp_path, monkeypatch, PreviewRenderer(failure=error))
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        result = acquire_evidence(lambda req: p._query_cov_selection(req, candidates(), tmp_path,
            overlay_spec=spec, require_visible_targets=True), {"metric": "functional_consistency"})
    with pytest.raises(AdaptiveEvidenceError): result.raise_if_failed()
    assert not selected


@pytest.mark.parametrize("rgb_blank,identity_blank,kept", [(("c1",), (), ["c2"]),
    ((), ("c1",), ["c2"]), (("c1", "c2"), (), []), ((), ("c1", "c2"), ["c1", "c2"])])
def test_final_blank_filter_and_partial_packet(tmp_path, monkeypatch, rgb_blank, identity_blank, kept):
    p, _, _ = provider(tmp_path, monkeypatch, PreviewRenderer(final_blank=rgb_blank, identity_blank=identity_blank))
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        result = acquire_evidence(lambda req: p._functional_probe_raw_evidence(req, candidates(), tmp_path,
            resolved_mode="query_cov", requested_candidate_count=4),
            {"metric": "functional_consistency", "evidence_policy": {"scoped_image_budget": 2}})
    result.raise_if_failed()
    assert [item["view_id"] for item in result.items] == kept
    if len(identity_blank) == 2:
        assert all(item["identity_grounded"] is False for item in result.items)
        assert result.audit["failure"]["recoverable_acquisition"] is True


def scoring():
    return {"combined_score_100": 60., "layers": [], "metrics": [
        {"metric": name, "layer": "L1" if i < 3 else "L3", "score": .6,
         "overall_weight": w, "coverage_fraction": 1., "status": "evaluated"}
        for i, (name, w) in enumerate(zip(("collision", "support", "oob", "functional_consistency",
                                           "semantic_placement_consistency"), (.1, .1, .1, .364, .196)))]}


def test_failed_metric_positive_coverage_serializes_without_score():
    raw = scoring()
    raw["metrics"][3].update(score=None, coverage_fraction=22/29, status="failed", reason="render_failure")
    raw["metrics"][4].update(score=None, coverage_fraction=11/12, status="failed")
    view, audit = normalized_feedback_scores(raw)
    assert view["feedback_score"] is None
    assert view["layer_scores"]["l3_scene_quality"]["score"] is None
    assert audit["unavailable_metric_details"][0]["reason"] == "render_failure"
    json.dumps((view, audit), allow_nan=False)


def test_success_normalization_and_unknown_zero_mass_unchanged():
    raw = scoring()
    assert normalized_feedback_scores(raw)[0]["feedback_score"] == pytest.approx(.6)
    raw["metrics"][3].update(score=None, coverage_fraction=0., status="not_evaluable")
    assert normalized_feedback_scores(raw)[0]["feedback_score"] == pytest.approx(.6)
    raw["metrics"][4]["score"] = float("nan")
    with pytest.raises(ValueError): normalized_feedback_scores(raw)


def test_ineligible_scene_compacts_but_never_gets_feedback_score():
    raw = scoring()
    raw["combined_score_100"] = None
    summary = {"schema_version": "model_judgement_coverage_v1", "eligible": False,
        "status": "infrastructure_failure", "judgement_coverage_fraction": .9,
        "minimum_judgement_coverage": .8, "infrastructure_failure_metrics": ["style_consistency"], "score": None}
    report = {"report_schema_version": "scene_evaluation_report_v2", "layer_reports": {}, "reports": {},
        "evaluation_status": "incomplete", "benchmark_score": None, "benchmark_score_status": "infrastructure_failure",
        "judgement_coverage_summary": summary, "runner_outcome": {"uniform_score_acceptance": deepcopy(summary)}}
    compact = compact_report(report, scoring_summary=raw)
    assert compact["feedback_scores"]["feedback_score"] is None
    assert compact["judgement_coverage_summary"]["infrastructure_failure_metrics"] == ["style_consistency"]
    with pytest.raises(RuntimeError): validate_acceptance(compact)
    json.dumps(compact, allow_nan=False)


def test_failed_diagnostic_keeps_cause_without_raw_exchange():
    content, _ = defect_content({'scene_quality': {'metrics': {'functional_consistency': {
        'status': 'failed', 'reason': 'required_scope_infrastructure_failure',
        'judgement': {'failure': {'failure_category': 'render_failure', 'error_type': 'FileNotFoundError',
                                  'raw_response': 'PRIVATE', 'headers': {'Authorization': 'PRIVATE'}}}}}}})
    encoded = json.dumps(content)
    assert 'required_scope_infrastructure_failure' in encoded
    assert 'render_failure' in encoded and 'FileNotFoundError' in encoded
    assert 'PRIVATE' not in encoded


def test_missing_atomic_episode_cannot_shrink_the_required_plan():
    from benchmark.evaluator.scene_quality.group_scoped import _combine_functional_check_episodes
    packet = {'group': {'group_id': 'g'}, 'functional_probe_evidence': {
        'required_checks': [{'check_id': 'a'}, {'check_id': 'b'}]}}
    with pytest.raises(ValueError, match='do not match the required ledger'):
        _combine_functional_check_episodes(packets=[packet], episode_results=[{
            **outcome(), 'group_id': 'g', 'functional_check_episode_id': 'a'}])


@pytest.mark.parametrize('ending', ['valid', 'invalid'])
def test_blank_acquisition_reaches_real_terminal_judge(tmp_path, monkeypatch, ending):
    from test_evidence_adaptive_judgement import Model, response
    from test_r12_placement_scope_control import occluded_scope, scope_request
    from test_r13_best_effort_terminal import judge as terminal_judge
    from test_postrun_policy_boundary import v2_control
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    from PIL import Image
    picture = tmp_path / 'retained.png'
    pixels = Image.new('RGB', (32, 32), (20, 120, 220))
    pixels.paste((240, 30, 40), (0, 0, 16, 32))
    pixels.save(picture)
    p, selected, spec = provider(tmp_path, monkeypatch, PreviewRenderer(blank=('c0', 'c1', 'c2', 'c3')))
    def acquire(request):
        return p._query_cov_selection(request, candidates(), tmp_path, overlay_spec=spec,
                                      require_visible_targets=True)
    model = Model([occluded_scope(), occluded_scope(), response(ending)])
    wrapper = ControlledVLMJudge(terminal_judge(model), control=v2_control(), camera_provider=acquire)
    request = scope_request(str(picture))
    request['metric'] = 'scale_consistency'
    request.pop('required_placement_checks')
    result = wrapper.adjudicate_scene_quality(request)
    assert result['verdict'] == ending
    assert result['evidence_resolution']['accepted'] is True
    assert result['evidence_resolution']['model_judgement_completed'] is True
    assert result['evidence_resolution']['images_used'] == [str(picture)]
    assert len(model.calls) == 3 and not selected


def test_renderer_validation_missing_file_is_not_a_blank_candidate(tmp_path):
    from benchmark.rendering.blender import _validate_render_views, BlenderRenderError
    with pytest.raises(BlenderRenderError, match='missing render files'):
        _validate_render_views({'views': [{'id': 'missing', 'path': str(tmp_path / 'missing.png')}]})
