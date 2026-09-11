"""Offline publication checks; never use live evaluation inputs or model calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluator_release_tools import (
    EvaluatorReleaseError, relative_file, sha256_file, tree_sha256,
    verify_nonrect_core, verify_snapshot,
)
from scripts.run_floorplan_evaluator import resolve_release, run_release


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "single_room_sceneweaver_20260909_v1"
SNAPSHOT = ROOT / "evaluator_snapshots" / BASELINE
REGISTRY = ROOT / "configs/runners/floorplan_evaluator_baselines_v1.json"


def test_single_room_snapshot_matches_registered_origin_and_all_exported_files():
    result = verify_snapshot(ROOT, BASELINE)
    assert result["verified_file_count"] == 484
    assert result["origin_tree_sha256"] == "893b3a5c38751a553f348ac2e12a0142121cf489858abb28a66152f33ed069dd"
    assert result["published_tree_sha256"] == "a9caf5cea2a0201a6eac69a84a852f393bed5ddffc9132541798e708fac0109a"


def test_nonrect_core_stays_byte_identical_to_3186983():
    result = verify_nonrect_core(ROOT)
    assert result == {"source_commit": "31869837105d7ef10c3b3e382cb84eeb1f1f1efc", "verified_file_count": 183}


def test_current_mapping_and_multi_room_provenance_are_not_promoted():
    registry = json.loads(REGISTRY.read_text())
    assert registry["current"] == {
        "single_room": BASELINE,
        "multi_room": "multi_room_hy4_wall_20260902_v1",
        "non_rectangular_multi_room": "nonrect_3186983_execution_v4_20260910_v1",
    }
    multi = registry["baselines"][registry["current"]["multi_room"]]
    assert multi["code_identity"]["replay_ready"] is False
    result = resolve_release(ROOT, "multi_room")
    assert result["execution_available"] is False
    with pytest.raises(EvaluatorReleaseError, match="No other mode"):
        run_release(result, ["--help"])


def test_nonrect_selector_checks_core_and_retains_sealed_release_requirement():
    result = resolve_release(ROOT, "non_rectangular_multi_room")
    assert result["core_verification"]["verified_file_count"] == 183
    sealed = ROOT / "Support/artifacts/releases/complicated_eval_combined142_v1/run_combined.py"
    assert result["execution_available"] == sealed.is_file()
    if not sealed.is_file():
        assert result["missing_requirement"] == "sealed_combined142_release"


@pytest.mark.parametrize("relative", ["../outside.py", "/absolute.py", "a/../b.py", "a\\b.py", "./a.py", "a//b.py", ""])
def test_manifest_paths_cannot_escape_or_alias(tmp_path, relative):
    with pytest.raises(EvaluatorReleaseError):
        relative_file(tmp_path, relative)


def test_symlink_source_is_rejected(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("pass\n")
    (tmp_path / "alias.py").symlink_to(target)
    with pytest.raises(EvaluatorReleaseError, match="Symlinks"):
        relative_file(tmp_path, "alias.py")


@pytest.fixture
def tiny_snapshot(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "src").mkdir(parents=True)
    source = snapshot / "src/a.py"
    source.write_text("pass\n")
    entries = {"src/a.py": sha256_file(source)}
    manifest = {
        "schema_version": "evaluator_source_snapshot_v1", "baseline_id": "fixture",
        "mode": "single_room", "origin_files_sha256": entries, "origin_file_count": 1,
        "origin_tree_sha256": tree_sha256(entries), "published_files_sha256": entries,
        "published_file_count": 1, "published_tree_sha256": tree_sha256(entries),
    }
    manifest_path = snapshot / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    registry_path = tmp_path / "configs/runners/floorplan_evaluator_baselines_v1.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps({"baselines": {"fixture": {
        "mode": "single_room", "versions": {},
        "code_identity": {"file_count": 1, "tree_sha256": tree_sha256(entries)},
        "source_publication": {"manifest": "snapshot/source_manifest.json",
                               "manifest_sha256": sha256_file(manifest_path),
                               "tree_sha256": tree_sha256(entries)},
    }}}))
    return tmp_path, source, manifest_path


def test_changed_source_is_rejected(tiny_snapshot):
    root, source, _ = tiny_snapshot
    assert verify_snapshot(root, "fixture")["verified_file_count"] == 1
    source.write_text("raise RuntimeError('changed')\n")
    with pytest.raises(EvaluatorReleaseError, match="Source hash mismatch"):
        verify_snapshot(root, "fixture")


def test_changed_manifest_is_rejected(tiny_snapshot):
    root, _, manifest = tiny_snapshot
    manifest.write_text("{}")
    with pytest.raises(EvaluatorReleaseError, match="manifest hash mismatch"):
        verify_snapshot(root, "fixture")


def test_unlisted_source_is_rejected(tiny_snapshot):
    root, source, _ = tiny_snapshot
    source.with_name("extra.py").write_text("pass\n")
    with pytest.raises(EvaluatorReleaseError, match="Unexpected"):
        verify_snapshot(root, "fixture")


def test_run_requires_explicit_arguments_without_starting_a_process(monkeypatch):
    monkeypatch.setattr(subprocess, "call", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(EvaluatorReleaseError, match="explicit runner arguments"):
        run_release(resolve_release(ROOT, "single_room"), [])


def test_run_uses_fresh_interpreter_and_does_not_mutate_parent_environment(monkeypatch):
    observed = {}
    monkeypatch.setenv("PYTHONPATH", "/some/other/benchmark")
    def fake_call(command, *, env):
        observed.update(command=command, env=env)
        return 7
    monkeypatch.setattr(subprocess, "call", fake_call)
    result = resolve_release(ROOT, "single_room")
    assert run_release(result, ["--help"]) == 7
    assert observed["command"] == [sys.executable, "-B", str(SNAPSHOT / "scripts/run_camera_cal_scene_level.py"), "--help"]
    assert observed["env"]["PYTHONPATH"] == str(SNAPSHOT / "src")
    assert observed["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert os.environ["PYTHONPATH"] == "/some/other/benchmark"


def test_published_runner_help_works_in_a_clean_cwd_without_operator_assets(tmp_path):
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/run_floorplan_evaluator.py"),
         "--mode", "single_room", "--run", "--", "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset-root" in result.stdout
    assert f"Evaluator baseline: {BASELINE}" in result.stdout


def test_isolated_snapshot_versions_scoring_and_resources(tmp_path):
    # This process intentionally imports the root evaluator first. Only the
    # child should see v4/v12/v8/v31; namespace contamination is a regression.
    from benchmark.evaluator.generic_validity.collision import COLLISION_EVALUATOR_VERSION
    assert COLLISION_EVALUATOR_VERSION == "collision_p0b_v3"
    program = r'''
import json
from pathlib import Path
from benchmark.evaluator.generic_validity import collision, support
from benchmark.evaluator.scene_quality import definitions
from benchmark.visual_judge import l3_prompts
from benchmark.evaluator.scoring import project_metric_events, project_incomplete_metric_coverage
from benchmark.scoring_profiles import DEFAULT_L3_METRIC_WEIGHTS
from benchmark.resources import runtime_resource_path
event = {"event_id": "one", "burden": 0.4, "allocations": {"a": 0.4}}
score = project_metric_events("collision", ordered_object_ids=["a", "b", "c", "d"], events=[event])
assert abs(score["score"] - 0.6) < 1e-12
assert event == {"event_id": "one", "burden": 0.4, "allocations": {"a": 0.4}}
assert project_incomplete_metric_coverage(score, coverage_fraction=0.79)["score"] is None
assert project_incomplete_metric_coverage(score, coverage_fraction=0.8)["score"] == score["score"]
assert project_metric_events("collision", ordered_object_ids=[], events=[])["score"] is None
assert DEFAULT_L3_METRIC_WEIGHTS == {"scale_consistency": .04, "style_consistency": .07, "object_pairing_consistency": .09, "functional_consistency": .52, "semantic_placement_consistency": .28}
print(json.dumps({"versions": [collision.COLLISION_EVALUATOR_VERSION, support.SUPPORT_EVALUATOR_VERSION, definitions.SCENE_QUALITY_INTERFACE_VERSION, l3_prompts.L3_METRIC_PROMPT_VERSION], "module": collision.__file__, "resource": str(runtime_resource_path("configs/evaluation/metric_profile_canonical_v2.yaml"))}))
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", program], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(SNAPSHOT / "src"), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["versions"] == ["collision_p0b_v4", "support_p0b_v12", "scene_quality_v8", "l3_burden_categories_v31"]
    assert Path(value["module"]).is_relative_to(SNAPSHOT)
    assert Path(value["resource"]).is_relative_to(SNAPSHOT)


def test_root_and_packaged_profile_copies_match_in_snapshot():
    for name in ["metric_profile_canonical_v2.yaml", "metric_profile_game_canonical_v1.yaml"]:
        assert (SNAPSHOT / "configs/evaluation" / name).read_bytes() == (
            SNAPSHOT / "src/benchmark/_resources/configs/evaluation" / name).read_bytes()
