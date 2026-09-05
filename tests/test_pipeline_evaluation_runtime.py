from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image
import pytest

from benchmark.generation_comparison.evaluation_runtime import CanonicalEvaluationRuntime
from benchmark.generation_comparison.pilot import (
    preflight_prepared_pilot, prepare_controlled_pilot, run_prepared_pilot,
)
from benchmark.models.openai_compatible_model import OpenAICompatibleModel
from benchmark.rendering import BlenderRenderer
from benchmark.utils.io import read_json, write_json
from benchmark.visual_judge import CameraEvidenceProvider
from benchmark.visual_judge.openai_compatible import OpenAICompatibleVLMJudge
from benchmark.visual_judge.openai_camera_selector import OpenAICompatibleCameraSelector
from test_controlled_generation_pilot import _asset_root, _pilot_spec


def _runtime_config() -> dict:
    return {
        "schema_version": "canonical_blender_evaluation_runtime_v1",
        "renderer": {"blender_bin": sys.executable},
        "judge": {"endpoint": "http://127.0.0.1:9999/v1", "model": "fixture-judge",
                  "api_key_env": "PIPELINE_TEST_JUDGE_KEY"},
        "camera_selector": {"endpoint": "http://127.0.0.1:9999/v1", "model": "fixture-selector",
                            "api_key_env": "PIPELINE_TEST_JUDGE_KEY"},
    }


@pytest.mark.parametrize("missing", ["renderer", "judge", "camera_selector", "credentials"])
def test_runtime_fails_closed_without_required_components(missing, monkeypatch):
    monkeypatch.delenv("PIPELINE_TEST_JUDGE_KEY", raising=False)
    config = _runtime_config()
    if missing != "credentials":
        config.pop(missing)
        monkeypatch.setenv("PIPELINE_TEST_JUDGE_KEY", "test-only")
    with pytest.raises(ValueError):
        CanonicalEvaluationRuntime(config)


def test_runtime_preflight_never_contacts_model(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleModel, "chat_messages", lambda *a, **k: pytest.fail("no-call preflight"))
    runtime = CanonicalEvaluationRuntime(_runtime_config(), require_credentials=False)
    assert runtime.readiness["ready"] is True
    assert runtime.readiness["service_contacted"] is False
    assert runtime.readiness["real_service_verified"] is False


def test_missing_runtime_blocks_before_ready_generator(tmp_path, monkeypatch):
    prepare_controlled_pilot(spec=_pilot_spec(methods=["catalog_placement"]),
                             asset_root=_asset_root(tmp_path / "assets"), out_dir=tmp_path / "pilot")
    monkeypatch.setenv("PIPELINE_TEST_GENERATOR_KEY", "test-only")
    monkeypatch.setattr("benchmark.generation_comparison.pilot.run_controlled_generation",
                        lambda **k: pytest.fail("missing evaluation runtime must block generation"))
    result = run_prepared_pilot(prepared_dir=tmp_path / "pilot", dry_run_only=True,
                               method_configs={"catalog_placement": {"adapter_config": {
                                   "endpoint": "http://127.0.0.1:9999/v1", "model": "fixture",
                                   "api_key_env": "PIPELINE_TEST_GENERATOR_KEY"}}})
    assert result["status"] == "blocked" and result["attempted_runs"] == 0
    row = json.loads((tmp_path / "pilot/results.jsonl").read_text())
    assert row["readiness"]["execution_readiness"]["reasons"] == ["evaluator_runtime_not_ready"]


def test_no_call_preflight_exposes_frozen_applicability_block_without_writes(tmp_path, monkeypatch):
    spec = _pilot_spec(methods=["catalog_placement"])
    spec["asset_selection_status"] = "human_approved"
    prepare_controlled_pilot(
        spec=spec, asset_root=_asset_root(tmp_path / "assets"), out_dir=tmp_path / "pilot",
        evaluation_runtime_config=_runtime_config(),
    )
    monkeypatch.setenv("PIPELINE_TEST_JUDGE_KEY", "test-only")
    config = {"catalog_placement": {"adapter_config": {
        "endpoint": "http://127.0.0.1:9999/v1", "model": "fixture",
        "api_key_env": "PIPELINE_TEST_JUDGE_KEY"}}}
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must never execute generation, rendering or a model")
    monkeypatch.setattr(OpenAICompatibleModel, "chat_messages", forbidden)
    monkeypatch.setattr(BlenderRenderer, "render_scene", forbidden)
    monkeypatch.setattr("benchmark.generation_comparison.pilot.run_controlled_generation", forbidden)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = preflight_prepared_pilot(prepared_dir=tmp_path / "pilot", method_configs=config)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert report["asset_selection_approved"]
    assert not report["ready_for_generation"]
    assert not report["service_contacted"] and not report["generation_executed"]
    assert report["evaluation_runtime"]["score_policy"]["unresolved_applicability_metrics"] == [
        "object_pairing_consistency", "style_consistency",
    ]
    result = run_prepared_pilot(prepared_dir=tmp_path / "pilot", method_configs=config)
    assert result["status"] == "blocked" and result["attempted_runs"] == 0


def _install_external_fixtures(monkeypatch, state):
    """Mock external observations/rendering only, never run_evaluate or scores."""
    def render(self, *, scene_path, out_dir, asset_root=None):
        scene = read_json(scene_path)
        state["scene"] = scene
        destination = Path(out_dir)
        destination.mkdir(parents=True)
        views = []
        for name in ("perspective", "top", "identity_map"):
            path = destination / f"{name}.png"
            image = Image.new("RGB", (32, 32), (90, 70, 80))
            image.putpixel((0, 0), (200, 30, 40))
            image.save(path)
            views.append({"name": name, "path": str(path)})
        blend = destination / "scene.blend"
        blend.write_bytes(b"external-renderer-fixture-not-a-real-blend")
        manifest = {"objects": [{"id": obj["id"], "representation": "asset_mesh"} for obj in scene["objects"]],
                    "views": views, "blend_file": str(blend),
                    "identity_legend": {f"O{i + 1}": obj["id"] for i, obj in enumerate(scene["objects"])},
                    "collision_geometry": {"objects": {}}}
        write_json(destination / "render_manifest.json", manifest)
        state["renders"].append(str(scene_path))
        state["image"] = views[0]["path"]
        return manifest

    def chat(self, messages, **kwargs):
        assert kwargs.get("call_type") == "vlm_grouping.partition"
        state["grouping_calls"] += 1
        return json.dumps({"object_groups": [{"object_ids": [obj["id"] for obj in state["scene"]["objects"]],
                                             "label": "fixture group", "reason": "fixture scope"}],
                           "reason": "fixture grouping"})

    def judge(self, request):
        state["judgements"].append(request)
        result = {"evidence_status": "sufficient", "verdict": "valid", "confidence": 0.9,
                  "reason": "fixture external observation", "missing_evidence": [], "defects": []}
        for prefix in ("functional", "placement"):
            checks = request.get(f"required_{prefix}_checks", [])
            if checks:
                result[f"{prefix}_check_results"] = [
                    {"check_id": item["check_id"], "observation_status": "observed",
                     "conclusion": "valid", "reason": "fixture observed check",
                     **({"target_ids": item.get("target_ids", [])} if prefix == "functional" else
                        {"subject_id": item.get("subject_id"), "context_ids": item.get("context_ids", [])})}
                    for item in checks
                ]
        return result

    def functional(self, request):
        state["functional_discovery"] += 1
        return {"inspected_object_ids": [obj["id"] for obj in state["scene"]["objects"]],
                "directed_surface_targets": [], "functional_correspondences": [],
                "approach_clearance_targets": [], "boundary_sensitive_targets": [],
                "unusual_unconfirmed": [], "reason": "fixture discovery"}

    def placement(self, request):
        return {"schema_version": "placement_discovery_v1",
                "considered_object_ids": [obj["id"] for obj in state["scene"]["objects"]],
                "candidates": [], "reason": "fixture discovery", "decision_authority": "none"}

    def local(self, request):
        state["local_requests"].append(request)
        return [{"path": state["image"], "role": "group_local"}]

    monkeypatch.setattr(BlenderRenderer, "render_scene", render)
    monkeypatch.setattr(OpenAICompatibleModel, "chat_messages", chat)
    monkeypatch.setattr(OpenAICompatibleVLMJudge, "adjudicate_scene_quality", judge)
    monkeypatch.setattr(OpenAICompatibleVLMJudge, "_adjudicate_scene_quality_raw", judge)
    monkeypatch.setattr(OpenAICompatibleVLMJudge, "adjudicate_p0b", judge)
    monkeypatch.setattr(OpenAICompatibleCameraSelector, "discover_functional_evidence", functional)
    monkeypatch.setattr(OpenAICompatibleCameraSelector, "discover_placement_evidence", placement)
    monkeypatch.setattr(CameraEvidenceProvider, "__call__", local)


def test_frozen_runtime_exposes_existing_applicability_coverage_gap(tmp_path, monkeypatch):
    state = {"renders": [], "judgements": [], "grouping_calls": 0,
             "functional_discovery": 0, "local_requests": []}
    _install_external_fixtures(monkeypatch, state)
    monkeypatch.setenv("PIPELINE_TEST_JUDGE_KEY", "test-only")
    prepared = prepare_controlled_pilot(
        spec=_pilot_spec(methods=["catalog_placement"]),
        asset_root=_asset_root(tmp_path / "assets"), out_dir=tmp_path / "pilot",
        evaluation_runtime_config=_runtime_config(),
    )
    native = write_json(tmp_path / "native.json", {
        "schema_version": "catalog_placement_v1", "instances": [{
            "instance_id": "chair_instance", "slot_id": "chair_0", "asset_id": "chair_asset",
            "center_m": [2, 2, 0.5], "uniform_scale": 1.0, "rotation_euler_xyz_deg": [0, 0, 0],
        }],
    })
    result = run_prepared_pilot(
        prepared_dir=tmp_path / "pilot", dry_run_only=True, allow_offline_artifacts=True,
        method_outputs={"catalog_placement": {prepared["cases"][0]["case_id"]: native}},
    )
    row = json.loads((tmp_path / "pilot/results.jsonl").read_text())
    summary = read_json(result["summary"])
    assert summary["methods"]["catalog_placement"]["mean_score"] is None
    assert summary["methods"]["catalog_placement"]["incomplete_evaluations"] == 1
    assert row["evaluation_report"], row
    report = read_json(row["evaluation_report"])
    assert report["workflow"] == "canonical_l0_l4"
    # This is an existing scoring/applicability limitation, not missing wiring:
    # honest fixed-asset ownership leaves Style/Pairing ungrounded while the
    # unchanged evaluator retains their weights. Never call that complete.
    assert report["benchmark_score_status"] == "partial_coverage", report
    assert report["reports"]["scene_quality"]["coverage"]["fraction"] == pytest.approx(0.84)
    assert result["status"] == "failed" and not row["evaluation_success"]
    assert row["score_available"] is True
    assert state["grouping_calls"] == 1
    assert state["functional_discovery"] > 0
    assert state["judgements"]
    assert len(state["renders"]) == 1
    assert not result["real_upstream_execution_performed"]
    manifest = read_json(row["run_manifest"])
    assert manifest["evaluator"]["actual_policy_sha256"] == prepared["evaluator_config_sha256"]
    assert Path(manifest["evaluator"]["runtime_manifest"]).is_file()


def test_each_native_iteration_builds_its_own_runtime_evidence(tmp_path, monkeypatch):
    from benchmark.api.generation import run_generate
    from benchmark.api.scene_weaver_iterations import evaluate_scene_weaver_iterations
    from test_external_harness_execution import _adapter_config, _fake_upstream_repo, _generation_input

    state = {"renders": [], "judgements": [], "grouping_calls": 0,
             "functional_discovery": 0, "local_requests": []}
    _install_external_fixtures(monkeypatch, state)
    runtime = CanonicalEvaluationRuntime(_runtime_config(), require_credentials=False)
    generated = run_generate(
        generation_input=_generation_input(), adapter_name="scene_weaver",
        out_dir=tmp_path / "generated", run_generation=True,
        adapter_config=_adapter_config("scene_weaver", _fake_upstream_repo(tmp_path / "upstream")),
    )
    summary = evaluate_scene_weaver_iterations(
        native_output=generated["native_output"], generation_input=_generation_input(),
        out_dir=tmp_path / "trajectory", evaluation_runtime=runtime,
    )
    assert summary["available_iterations"] == [0, 1]
    assert summary["native_trajectory_verified_unchanged"]
    assert summary["benchmark_feedback_used_by_native_loop"] is False
    assert len(state["renders"]) == 2 and len(set(state["renders"])) == 2
    hashes = [read_json(Path(path).parent / "runtime_manifest.json")["canonical_scene_sha256"]
              for path in state["renders"]]
    assert hashes[0] != hashes[1]


def test_failed_evaluation_can_be_recovered_without_generation_or_source_overwrite(tmp_path, monkeypatch):
    from benchmark.generation_comparison.reevaluation import reevaluate_prepared_unit

    state = {"renders": [], "judgements": [], "grouping_calls": 0,
             "functional_discovery": 0, "local_requests": []}
    _install_external_fixtures(monkeypatch, state)
    monkeypatch.setenv("PIPELINE_TEST_JUDGE_KEY", "test-only")
    prepared = prepare_controlled_pilot(
        spec=_pilot_spec(methods=["catalog_placement"]), asset_root=_asset_root(tmp_path / "assets"),
        out_dir=tmp_path / "pilot", evaluation_runtime_config=_runtime_config(),
    )
    native = write_json(tmp_path / "native.json", {
        "schema_version": "catalog_placement_v1", "instances": [{
            "instance_id": "chair_instance", "slot_id": "chair_0", "asset_id": "chair_asset",
            "center_m": [2, 2, 0.5], "uniform_scale": 1.0, "rotation_euler_xyz_deg": [0, 0, 0],
        }],
    })
    def broken_render(*args, **kwargs):
        raise RuntimeError("fixture renderer unavailable after generation")
    with monkeypatch.context() as failure_patch:
        failure_patch.setattr(BlenderRenderer, "render_scene", broken_render)
        result = run_prepared_pilot(
            prepared_dir=tmp_path / "pilot", dry_run_only=True, allow_offline_artifacts=True,
            method_outputs={"catalog_placement": {prepared["cases"][0]["case_id"]: native}},
        )
    assert result["status"] == "failed"
    unit = tmp_path / "pilot/cases/case_001/catalog_placement"
    proof = read_json(unit / "comparison/generation_manifest.json")
    assert proof["status"] == "GENERATED_AWAITING_EVALUATION"
    assert read_json(unit / "comparison/run_manifest.json")["status"] == "EVALUATION_FAILED"
    before = {p: p.read_bytes() for p in (tmp_path / "pilot").rglob("*") if p.is_file()}
    monkeypatch.setattr("benchmark.api.generation.run_generate", lambda **k: pytest.fail("must not regenerate"))
    replay = reevaluate_prepared_unit(prepared_dir=tmp_path / "pilot", case_id="case_001",
                                      method="catalog_placement", out_dir=tmp_path / "reevaluation")
    assert replay["source_artifacts_verified_unchanged"]
    assert replay["status"] == "INCOMPLETE_EVALUATION"  # same honest applicability gap
    assert not replay["generation_reexecuted"] and not replay["converter_reexecuted"]
    assert before == {p: p.read_bytes() for p in (tmp_path / "pilot").rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        reevaluate_prepared_unit(prepared_dir=tmp_path / "pilot", case_id="case_001",
                                  method="catalog_placement", out_dir=tmp_path / "reevaluation")
    canonical = Path(proof["canonical_scene"])
    canonical.write_bytes(canonical.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical_scene hash mismatch"):
        reevaluate_prepared_unit(prepared_dir=tmp_path / "pilot", case_id="case_001",
                                  method="catalog_placement", out_dir=tmp_path / "invalid_replay")
    assert not (tmp_path / "invalid_replay").exists()
