from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from benchmark.adapters import get_adapter
from benchmark.adapters.common.execution import (
    ExternalExecutionError,
    artifact_sha256,
    verify_preserved_native_artifact,
)
from benchmark.api.evaluation import run_evaluate
from benchmark.api.generation import run_generate
from benchmark.api.scene_weaver_iterations import evaluate_scene_weaver_iterations
from benchmark.io_contracts import O1_OBJECT_STATE
from benchmark.nl_scene.generation_input import (
    build_direct_natural_language_generation_input,
)
from benchmark.scene_io.validate import ArtifactValidationError, validate_generated_scene
from benchmark.utils.io import read_json, write_json
from scripts.run_scene_harness import run_scene_harness


EXECUTABLE_ADAPTERS = (
    "layout_gpt",
    "direct_layout",
    "layout_vlm",
    "respace",
    "scene_weaver",
)
PRIVATE_SENTINEL = "HIDDEN_EVAL_SENTINEL"


def test_only_priority_external_adapters_declare_executable_integration() -> None:
    assert {
        name
        for name in EXECUTABLE_ADAPTERS
        if get_adapter(name).executable_integration
    } == set(EXECUTABLE_ADAPTERS)
    assert get_adapter("holodeck").executable_integration is False
    assert get_adapter("scene_smith").executable_integration is False


@pytest.mark.parametrize("adapter_name", EXECUTABLE_ADAPTERS)
def test_external_subprocess_generation_preserves_and_evaluates_native_output(
    adapter_name: str,
    tmp_path: Path,
) -> None:
    repo = _fake_upstream_repo(tmp_path / f"repo_{adapter_name}")
    generation_input = _generation_input()
    config = _adapter_config(adapter_name, repo)
    result = run_generate(
        generation_input=generation_input,
        adapter_name=adapter_name,
        out_dir=tmp_path / f"run_{adapter_name}",
        adapter_config=config,
        run_generation=True,
    )

    assert result["status"]["status"] == "generated_scene_available"
    scene = read_json(result["generated_scene"])
    assert validate_generated_scene(scene)
    assert scene["metadata"]["output_adapter"] == adapter_name

    adapter_metadata = read_json(result["adapter_metadata"])
    assert adapter_metadata["executable_integration"] is True
    run_metadata = adapter_metadata["generation_run"]
    assert run_metadata["status"] == "completed"
    assert run_metadata["runner_kind"] == "subprocess"
    assert run_metadata["return_code"] == 0
    assert run_metadata["timed_out"] is False
    assert run_metadata["runtime_seconds"] >= 0.0
    assert run_metadata["started_at"].endswith("Z")
    assert run_metadata["ended_at"].endswith("Z")
    assert run_metadata["upstream_repo"] == repo.resolve().as_posix()
    assert len(run_metadata["upstream_commit"]) == 40
    provenance = run_metadata["runner_provenance"]
    assert provenance["status"] == "SOURCE_FINGERPRINTED"
    assert provenance["source_path"] == (repo / "fake_upstream.py").resolve().as_posix()
    assert provenance["source_sha256"] == hashlib.sha256(
        (repo / "fake_upstream.py").read_bytes()
    ).hexdigest()
    assert provenance["source_git_commit"] == run_metadata["upstream_commit"]
    assert provenance["source_git_tracked"] is True
    assert provenance["source_git_modified"] is False
    assert provenance["source_unchanged_during_execution"] is True
    assert provenance["control_verification"] == "NOT_VERIFIED"
    assert run_metadata["native_artifact_verified_after_conversion"] is True
    assert run_metadata["auxiliary_artifacts_verified_after_conversion"] is True
    assert Path(run_metadata["canonical_scene_path"]) == Path(
        result["generated_scene"]
    ).resolve()
    assert "fake stdout" in Path(run_metadata["stdout_path"]).read_text(
        encoding="utf-8"
    )
    assert "fake stderr" in Path(run_metadata["stderr_path"]).read_text(
        encoding="utf-8"
    )
    if adapter_name == "scene_weaver":
        assert run_metadata["sceneweaver_available_iterations"] == [0, 1]
        assert [
            item["iteration"]
            for item in run_metadata["sceneweaver_iteration_artifacts"]
        ] == [0, 1]
        assert all(
            any("render_" in path for path in item["related_artifacts"])
            for item in run_metadata["sceneweaver_iteration_artifacts"]
        )
        assert run_metadata["benchmark_feedback_used_by_native_loop"] is False

    source = Path(run_metadata["source_native_artifact_path"])
    preserved = Path(run_metadata["preserved_native_artifact_path"])
    assert source != preserved
    source_digest, source_entries = artifact_sha256(source)
    preserved_digest, preserved_entries = artifact_sha256(preserved)
    assert source_digest == preserved_digest == run_metadata["native_artifact_sha256"]
    assert source_entries == preserved_entries
    assert Path(result["native_output"]) == preserved
    assert Path(result["raw_native_artifact"]) == preserved

    execution_result = read_json(run_metadata["execution_result_path"])
    assert execution_result["native_artifact_verified_after_conversion"] is True
    assert execution_result["native_artifact_sha256"] == preserved_digest
    assert execution_result["conversion_status"] == "completed"
    assert Path(execution_result["converter_metadata"]["raw_artifact_path"]) == (
        preserved.resolve()
    )
    runner_config = read_json(run_metadata["runner_config_path"])
    assert runner_config["execution"]["environment"]["UPSTREAM_API_TOKEN"] == (
        "<redacted>"
    )
    assert PRIVATE_SENTINEL not in json.dumps(runner_config)
    assert any(
        token == "<redacted-env:UPSTREAM_API_TOKEN>"
        for token in execution_result["command"]
    )

    method_input = read_json(result["method_input"])
    native_input = read_json(method_input["execution_input"]["path"])
    assert PRIVATE_SENTINEL not in json.dumps(method_input)
    assert PRIVATE_SENTINEL not in json.dumps(native_input)
    assert "reference_annotation" not in method_input
    assert "evaluation_context" not in method_input
    assert "reference_annotation" not in json.dumps(native_input)
    assert "evaluation_context" not in json.dumps(native_input)
    _assert_native_input(adapter_name, native_input)

    source_artifact = scene["metadata"]["harness_compatibility"][
        "source_artifact"
    ]
    assert Path(source_artifact).resolve().is_relative_to(preserved.resolve()) or Path(
        source_artifact
    ).resolve() == preserved.resolve()

    report = run_evaluate(
        scene=scene,
        out=tmp_path / f"evaluation_{adapter_name}.json",
    )
    assert report["workflow"] == "canonical_l0_l4"
    assert report["request_id"] == "external-execution"

    offline_config: dict[str, Any] = {}
    if adapter_name == "layout_vlm":
        offline_config["scene_config_path"] = method_input["execution_input"]["path"]
    if adapter_name == "scene_weaver":
        offline_config["selected_iteration"] = 1
    offline_result = run_generate(
        generation_input=_generation_input(),
        adapter_name=adapter_name,
        out_dir=tmp_path / f"offline_{adapter_name}",
        method_output=source,
        adapter_config=offline_config,
    )
    assert offline_result["status"]["status"] == "generated_scene_available"
    assert Path(offline_result["native_output"]).resolve() != source.resolve()
    assert validate_generated_scene(read_json(offline_result["generated_scene"]))


@pytest.mark.parametrize(
    ("mode", "expected_message", "return_code", "timed_out"),
    [
        ("nonzero", "exited with code 7", 7, False),
        ("timeout", "timed out", None, True),
        ("missing", "expected native artifact", 0, False),
        ("ambiguous", "ambiguous", 0, False),
    ],
)
def test_external_process_failures_persist_execution_evidence(
    mode: str,
    expected_message: str,
    return_code: int | None,
    timed_out: bool,
    tmp_path: Path,
) -> None:
    repo = _fake_upstream_repo(tmp_path / f"failure_repo_{mode}")
    config = _adapter_config("direct_layout", repo, mode=mode)
    if mode == "timeout":
        config["execution"]["timeout_seconds"] = 0.05

    with pytest.raises(ExternalExecutionError, match=expected_message):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="direct_layout",
            out_dir=tmp_path / f"failure_{mode}",
            adapter_config=config,
            run_generation=True,
        )

    result = read_json(
        tmp_path
        / f"failure_{mode}"
        / "generator"
        / "execution"
        / "execution_result.json"
    )
    assert result["status"] == "failed"
    assert result["return_code"] == return_code
    assert result["timed_out"] is timed_out
    assert result["runner_provenance"]["status"] == "SOURCE_FINGERPRINTED"
    assert expected_message in result["error"]["message"]
    assert "fake stdout" in Path(result["stdout_path"]).read_text(encoding="utf-8")
    assert "fake stderr" in Path(result["stderr_path"]).read_text(encoding="utf-8")
    assert not (tmp_path / f"failure_{mode}" / "generated_scene.json").exists()


def test_missing_repo_and_executable_fail_clearly(tmp_path: Path) -> None:
    missing_repo_config = _adapter_config(
        "direct_layout",
        tmp_path / "missing_repo",
    )
    with pytest.raises(ExternalExecutionError, match="repo is missing"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="direct_layout",
            out_dir=tmp_path / "missing_repo_run",
            adapter_config=missing_repo_config,
            run_generation=True,
        )

    repo = _fake_upstream_repo(tmp_path / "missing_executable_repo")
    config = _adapter_config("direct_layout", repo)
    config["execution"]["command"][0] = "missing-upstream-executable"
    with pytest.raises(ExternalExecutionError, match="executable is missing"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="direct_layout",
            out_dir=tmp_path / "missing_executable_run",
            adapter_config=config,
            run_generation=True,
        )

    config = _adapter_config("direct_layout", repo)
    config["execution"]["command"][1] = "{repo_path}/missing_entrypoint.py"
    with pytest.raises(ExternalExecutionError, match="entrypoint is missing"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="direct_layout",
            out_dir=tmp_path / "missing_entrypoint_run",
            adapter_config=config,
            run_generation=True,
        )


def test_unsupported_or_ambiguous_execution_configuration_fails(
    tmp_path: Path,
) -> None:
    with pytest.raises(ExternalExecutionError, match="exactly one"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="direct_layout",
            out_dir=tmp_path / "unconfigured_run",
            adapter_config={},
            run_generation=True,
        )

    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> Path:
        del method_input_path, out_dir, config
        raise AssertionError("ambiguous runner must not execute")

    with pytest.raises(ExternalExecutionError, match="exactly one"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="direct_layout",
            out_dir=tmp_path / "ambiguous_config_run",
            adapter_config={
                "runner": runner,
                "raw_output_path": tmp_path / "unused.json",
            },
            run_generation=True,
        )


def test_malformed_native_output_is_preserved_but_never_converted_to_empty_scene(
    tmp_path: Path,
) -> None:
    repo = _fake_upstream_repo(tmp_path / "malformed_repo")
    with pytest.raises(ArtifactValidationError, match="placement array"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="direct_layout",
            out_dir=tmp_path / "malformed_run",
            adapter_config=_adapter_config(
                "direct_layout",
                repo,
                mode="malformed",
            ),
            run_generation=True,
        )

    execution_result = read_json(
        tmp_path
        / "malformed_run"
        / "generator"
        / "execution"
        / "execution_result.json"
    )
    assert execution_result["status"] == "completed"
    assert execution_result["conversion_status"] == "failed"
    assert "placement array" in execution_result["conversion_error"]["message"]
    assert execution_result["native_artifact_verified_after_conversion"] is False
    preserved = Path(execution_result["preserved_native_artifact_path"])
    assert preserved.is_file()
    assert read_json(preserved) == {}
    assert not (tmp_path / "malformed_run" / "generated_scene.json").exists()


def test_offline_method_output_is_snapshotted_before_conversion(tmp_path: Path) -> None:
    native = write_json(
        tmp_path / "offline_direct.json",
        _direct_layout_output(),
    )
    original_bytes = native.read_bytes()
    result = run_generate(
        generation_input=_generation_input(),
        adapter_name="direct_layout",
        out_dir=tmp_path / "offline_run",
        method_output=native,
    )

    assert native.read_bytes() == original_bytes
    preserved = Path(result["native_output"])
    assert preserved != native.resolve()
    assert preserved.read_bytes() == original_bytes
    metadata = read_json(result["adapter_metadata"])["generation_run"]
    assert metadata["runner_kind"] == "offline_supplied_artifact"
    assert metadata["runner_provenance"] == {"status": "NOT_EXECUTED"}
    assert metadata["source_native_artifact_path"] == native.resolve().as_posix()
    assert metadata["native_artifact_verified_after_conversion"] is True
    assert hashlib.sha256(original_bytes).hexdigest() == metadata[
        "native_artifact_sha256"
    ]


def test_offline_asset_binding_sidecar_is_snapshotted_and_used(tmp_path: Path) -> None:
    native = write_json(
        tmp_path / "layoutgpt_native.json",
        {
            "unit": "m",
            "object_list": [
                [
                    "chair",
                    {
                        "length": 0.8,
                        "width": 0.8,
                        "height": 1.0,
                        "left": 2.0,
                        "top": 2.0,
                        "depth": 0.5,
                        "orientation": 0.0,
                    },
                ]
            ],
        },
    )
    sidecar = write_json(
        tmp_path / "asset_ids.json",
        {"chair_1": "chair-asset"},
    )
    result = run_generate(
        generation_input=_generation_input(),
        adapter_name="layout_gpt",
        out_dir=tmp_path / "offline_sidecar_run",
        method_output=native,
        adapter_config={"asset_ids_path": sidecar.name},
    )

    scene = read_json(result["generated_scene"])
    assert scene["objects"][0]["asset_ref"]["asset_key"] == "chair-asset"
    metadata = read_json(result["adapter_metadata"])["generation_run"]
    preserved = metadata["preserved_auxiliary_artifacts"]["asset_ids"]
    assert Path(preserved["path"]).read_bytes() == sidecar.read_bytes()
    assert Path(preserved["path"]).resolve() != sidecar.resolve()
    assert len(preserved["sha256"]) == 64


def test_callback_runner_uses_same_preservation_boundary(tmp_path: Path) -> None:
    calls: list[Path] = []

    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> Path:
        del config
        calls.append(method_input_path)
        print("callback stdout")
        print("callback stderr", file=sys.stderr)
        return write_json(out_dir / "callback_native.json", _direct_layout_output())

    result = run_generate(
        generation_input=_generation_input(),
        adapter_name="direct_layout",
        out_dir=tmp_path / "callback_run",
        adapter_config={"runner": runner},
        run_generation=True,
    )

    metadata = read_json(result["adapter_metadata"])["generation_run"]
    assert calls == [Path(result["method_input"])]
    assert metadata["runner_kind"] == "callback"
    assert metadata["runner_provenance"]["source_path"] == Path(__file__).resolve().as_posix()
    assert metadata["runner_provenance"]["source_sha256"] == hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    assert metadata["return_code"] == 0
    assert "callback stdout" in Path(metadata["stdout_path"]).read_text(
        encoding="utf-8"
    )
    assert "callback stderr" in Path(metadata["stderr_path"]).read_text(
        encoding="utf-8"
    )
    assert Path(metadata["preserved_native_artifact_path"]) == Path(
        result["native_output"]
    )


def test_callback_failure_still_captures_output_and_result(tmp_path: Path) -> None:
    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> Path:
        del method_input_path, out_dir, config
        print("callback failed stdout")
        print("callback failed stderr", file=sys.stderr)
        raise RuntimeError("callback exploded")

    with pytest.raises(ExternalExecutionError, match="callback exploded"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="direct_layout",
            out_dir=tmp_path / "callback_failure",
            adapter_config={"runner": runner},
            run_generation=True,
        )

    result = read_json(
        tmp_path
        / "callback_failure"
        / "generator"
        / "execution"
        / "execution_result.json"
    )
    assert result["status"] == "failed"
    assert "callback exploded" in result["error"]["message"]
    assert "callback failed stdout" in Path(result["stdout_path"]).read_text(
        encoding="utf-8"
    )
    assert "callback failed stderr" in Path(result["stderr_path"]).read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("nested_cwd", [False, True])
@pytest.mark.parametrize("unbuffered", [False, True])
def test_relative_entrypoint_uses_upstream_cwd_not_benchmark_cwd(
    nested_cwd: bool, unbuffered: bool, tmp_path: Path
) -> None:
    repo = _fake_upstream_repo(tmp_path / "relative_entrypoint_repo")
    config = _adapter_config("direct_layout", repo)
    config["execution"]["command"][1] = "fake_upstream.py"
    if nested_cwd:
        (repo / "work").mkdir()
        config["execution"]["cwd"] = "work"
        config["execution"]["command"][1] = "../fake_upstream.py"
    if unbuffered:
        config["execution"]["command"].insert(1, "-u")
    result = run_generate(
        generation_input=_generation_input(), adapter_name="direct_layout",
        out_dir=tmp_path / "run", adapter_config=config, run_generation=True,
    )
    metadata = read_json(result["adapter_metadata"])["generation_run"]
    assert metadata["return_code"] == 0
    assert metadata["runner_provenance"]["source_path"] == (
        repo / "fake_upstream.py"
    ).resolve().as_posix()


@pytest.mark.parametrize("untracked", [False, True])
def test_shim_fingerprint_does_not_misrepresent_uncommitted_code(
    untracked: bool, tmp_path: Path
) -> None:
    repo = _fake_upstream_repo(tmp_path / "modified_upstream")
    script = repo / ("custom_bridge.py" if untracked else "fake_upstream.py")
    script.write_text(_FAKE_UPSTREAM_SCRIPT + "\n# local bridge change\n", encoding="utf-8")
    config = _adapter_config("direct_layout", repo)
    config["execution"]["command"][1] = script.name
    result = run_generate(
        generation_input=_generation_input(), adapter_name="direct_layout",
        out_dir=tmp_path / "run", adapter_config=config, run_generation=True,
    )
    provenance = read_json(result["adapter_metadata"])["generation_run"]["runner_provenance"]
    assert provenance["source_git_tracked"] is (not untracked)
    assert provenance["source_git_modified"] is (None if untracked else True)
    assert provenance["source_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert provenance["control_verification"] == "NOT_VERIFIED"


def test_entrypoint_hash_records_code_changes_during_execution(tmp_path: Path) -> None:
    repo = _fake_upstream_repo(tmp_path / "self_modifying_upstream")
    script = repo / "fake_upstream.py"
    script.write_text(
        _FAKE_UPSTREAM_SCRIPT + '\nPath(__file__).write_text("# changed by upstream\\n")\n',
        encoding="utf-8",
    )
    initial_digest = hashlib.sha256(script.read_bytes()).hexdigest()
    result = run_generate(
        generation_input=_generation_input(), adapter_name="direct_layout",
        out_dir=tmp_path / "run", adapter_config=_adapter_config("direct_layout", repo),
        run_generation=True,
    )
    provenance = read_json(result["adapter_metadata"])["generation_run"]["runner_provenance"]
    assert provenance["source_sha256"] == initial_digest
    assert provenance["source_sha256_after_execution"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert provenance["source_unchanged_during_execution"] is False


def test_module_execution_does_not_claim_discovered_source_identity(tmp_path: Path) -> None:
    repo = _fake_upstream_repo(tmp_path / "module_upstream")
    config = _adapter_config("direct_layout", repo)
    config["execution"]["command"][1:2] = ["-m", "fake_upstream"]
    result = run_generate(
        generation_input=_generation_input(), adapter_name="direct_layout",
        out_dir=tmp_path / "run", adapter_config=config, run_generation=True,
    )
    provenance = read_json(result["adapter_metadata"])["generation_run"]["runner_provenance"]
    assert provenance["status"] == "NOT_DISCOVERED"
    assert "source_sha256" not in provenance
    assert provenance["control_verification"] == "NOT_VERIFIED"


def test_unreadable_source_audit_does_not_hide_the_upstream_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmark.adapters.common import execution

    source = tmp_path / "runner.py"
    source.write_text("# test fixture\n", encoding="utf-8")
    provenance = {"source_path": str(source), "source_sha256": "initial"}

    def unavailable(path):
        raise PermissionError(path)

    monkeypatch.setattr(execution, "_file_sha256", unavailable)
    execution._finish_runner_source_provenance(provenance)
    assert provenance["source_unchanged_during_execution"] is None
    assert provenance["source_sha256_after_execution"] is None
    assert provenance["source_verification_error"] == "PermissionError"


@pytest.mark.parametrize("adapter_name", ["layout_gpt", "scene_weaver"])
def test_generation_sidecar_asset_bindings_are_preserved_and_consumed(
    adapter_name: str,
    tmp_path: Path,
) -> None:
    repo = _fake_upstream_repo(tmp_path / f"sidecar_repo_{adapter_name}")
    config = _adapter_config(adapter_name, repo, mode="sidecar")
    if adapter_name == "layout_gpt":
        config["execution"].update(
            {
                "native_artifact": "{upstream_output_dir}/layoutgpt.json",
                "auxiliary_artifacts": {
                    "asset_ids": "{upstream_output_dir}/asset_ids.json"
                },
            }
        )
    else:
        config["selected_iteration"] = 1
        config["execution"]["auxiliary_artifacts"] = {
            "asset_bindings": "{upstream_output_dir}/asset_bindings.json"
        }

    result = run_generate(
        generation_input=_generation_input(),
        adapter_name=adapter_name,
        out_dir=tmp_path / f"sidecar_run_{adapter_name}",
        adapter_config=config,
        run_generation=True,
    )

    scene = read_json(result["generated_scene"])
    assert scene["objects"][0]["asset_ref"]["asset_key"] == "chair-asset"
    run_metadata = read_json(result["adapter_metadata"])["generation_run"]
    auxiliary_name = "asset_ids" if adapter_name == "layout_gpt" else "asset_bindings"
    auxiliary = run_metadata["preserved_auxiliary_artifacts"][auxiliary_name]
    assert Path(auxiliary["path"]).is_file()
    assert len(auxiliary["sha256"]) == 64
    assert run_metadata["auxiliary_artifacts_verified_after_conversion"] is True


def test_native_artifact_verification_detects_post_conversion_mutation(
    tmp_path: Path,
) -> None:
    native = write_json(tmp_path / "native.json", _direct_layout_output())
    result = run_generate(
        generation_input=_generation_input(),
        adapter_name="direct_layout",
        out_dir=tmp_path / "immutability_run",
        method_output=native,
    )
    metadata = read_json(result["adapter_metadata"])["generation_run"]
    Path(metadata["preserved_native_artifact_path"]).write_text(
        "mutated",
        encoding="utf-8",
    )

    with pytest.raises(ExternalExecutionError, match="changed during"):
        verify_preserved_native_artifact(metadata)


def test_sceneweaver_preserves_and_evaluates_every_native_iteration(
    tmp_path: Path,
) -> None:
    repo = _fake_upstream_repo(tmp_path / "sceneweaver_repo")
    config = _adapter_config("scene_weaver", repo)
    generation_result = run_generate(
        generation_input=_generation_input(),
        adapter_name="scene_weaver",
        out_dir=tmp_path / "sceneweaver_run",
        adapter_config=config,
        run_generation=True,
    )
    native_root = Path(generation_result["native_output"])
    before, _ = artifact_sha256(native_root)
    summary = evaluate_scene_weaver_iterations(
        native_output=native_root,
        generation_input=_generation_input(),
        out_dir=tmp_path / "sceneweaver_iteration_evaluation",
    )
    after, _ = artifact_sha256(native_root)

    assert before == after == summary["native_trajectory_sha256"]
    assert summary["available_iterations"] == [0, 1]
    assert summary["evaluation_workflows"] == ["canonical_l0_l4"]
    assert summary["benchmark_feedback_used_by_native_loop"] is False
    assert summary["native_trajectory_verified_unchanged"] is True
    assert [row["iteration"] for row in summary["iterations"]] == [0, 1]
    for row in summary["iterations"]:
        assert Path(row["native_artifact"]).is_file()
        assert Path(row["canonical_scene"]).is_file()
        assert Path(row["converter_metadata"]).is_file()
        assert Path(row["evaluation_report"]).is_file()
        assert row["evaluation_workflow"] == "canonical_l0_l4"
        assert any("render_" in path for path in row["related_native_artifacts"])
    persisted = read_json(summary["summary_path"])
    assert "summary_path" not in persisted
    assert persisted["iterations"] == summary["iterations"]


def test_sceneweaver_rejects_benchmark_feedback_at_native_runner_boundary(
    tmp_path: Path,
) -> None:
    repo = _fake_upstream_repo(tmp_path / "feedback_repo")
    with pytest.raises(ArtifactValidationError, match="never accepts benchmark"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="scene_weaver",
            out_dir=tmp_path / "feedback_run",
            adapter_config=_adapter_config("scene_weaver", repo),
            run_generation=True,
            evaluation_report={"benchmark_score": 0.0},
            previous_generated_scene={},
            iteration=1,
        )


def test_full_harness_keeps_private_reference_out_of_external_process(
    tmp_path: Path,
) -> None:
    repo = _fake_upstream_repo(tmp_path / "private_isolation_repo")
    out_dir = tmp_path / "private_isolation"
    manifest = run_scene_harness(
        instruction="Design a room with one chair.",
        scene_type="room",
        out_dir=out_dir,
        room={"boundary": [[0, 0], [4, 0], [4, 5], [0, 5]], "height": 3.0},
        structure=False,
        reference_annotation=_reference_annotation(out_dir.name),
        adapter="direct_layout",
        adapter_config=_adapter_config("direct_layout", repo),
        run_generation=True,
    )

    method_input = read_json(out_dir / "generator" / "method_input.json")
    native_input = read_json(method_input["execution_input"]["path"])
    assert PRIVATE_SENTINEL not in json.dumps(method_input)
    assert PRIVATE_SENTINEL not in json.dumps(native_input)
    assert PRIVATE_SENTINEL in (out_dir / "reference_annotation.json").read_text(
        encoding="utf-8"
    )
    assert manifest["data_isolation"][
        "benchmark_private_artifacts_written_after_generation"
    ] is True
    assert manifest["evaluation"]["gate"]["workflow"] == "canonical_l0_l4"
    adapter_metadata = read_json(manifest["artifacts"]["adapter_metadata"])
    generation_run = adapter_metadata["generation_run"]
    assert generation_run["evaluation_workflow"] == "canonical_l0_l4"
    assert generation_run["evaluation_link_recorded_after_generation"] is True
    assert Path(generation_run["evaluation_report_path"]).is_file()
    execution_result = read_json(generation_run["execution_result_path"])
    assert execution_result["evaluation_report_path"] == generation_run[
        "evaluation_report_path"
    ]


def _adapter_config(
    adapter_name: str,
    repo: Path,
    *,
    mode: str = "success",
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "execution": {
            "repo_path": repo.as_posix(),
            "python_executable": sys.executable,
            "command": [
                "{python_executable}",
                "{repo_path}/fake_upstream.py",
                "{adapter}",
                "{native_input}",
                "{upstream_output_dir}",
                "{env:UPSTREAM_API_TOKEN}",
            ],
            "timeout_seconds": 5,
            "environment": {
                "FAKE_MODE": mode,
                "UPSTREAM_API_TOKEN": "super-secret-token",
            },
        }
    }
    if adapter_name == "layout_vlm":
        config["layout_vlm_scene_config"] = {
            "assets": {
                "chair-asset-0": {
                    "uid": "chair-asset",
                    "category": "chair",
                    "description": "chair",
                    "assetMetadata": {
                        "boundingBox": {"x": 0.8, "y": 0.8, "z": 1.0}
                    },
                }
            }
        }
    if adapter_name == "scene_weaver":
        config["selected_iteration"] = 1
    return config


def _generation_input() -> dict:
    return build_direct_natural_language_generation_input(
        request_id="external-execution",
        instruction="Design a room with one chair.",
        scene_type="room",
        room={"boundary": [[0, 0], [4, 0], [4, 5], [0, 5]], "height": 3.0},
        evaluator_output_type=O1_OBJECT_STATE,
    )


def _fake_upstream_repo(repo: Path) -> Path:
    repo.mkdir(parents=True)
    script = repo / "fake_upstream.py"
    script.write_text(_FAKE_UPSTREAM_SCRIPT, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "fake_upstream.py"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Harness Test",
            "-c",
            "user.email=harness@example.invalid",
            "commit",
            "-qm",
            "fake upstream",
        ],
        cwd=repo,
        check=True,
    )
    return repo


def _assert_native_input(adapter_name: str, native_input: Any) -> None:
    if adapter_name == "direct_layout":
        assert native_input == [
            ["Design a room with one chair."],
            [[4.0, 5.0, 3.0]],
        ]
        return
    if adapter_name == "layout_vlm":
        assert native_input["task_description"] == "Design a room with one chair."
        assert native_input["boundary"]["wall_height"] == 3.0
        assert "chair-asset-0" in native_input["assets"]
        return
    assert native_input["schema_version"]
    if adapter_name == "layout_gpt":
        assert native_input["query_id"] == "external-execution"
        assert native_input["room_dimensions_m"] == [4.0, 5.0, 3.0]
    elif adapter_name == "respace":
        assert native_input["scene"]["bounds_bottom"][2] == [4.0, 0.0, -5.0]
        assert native_input["operation"] == "full_scene_generation"
    elif adapter_name == "scene_weaver":
        assert native_input["benchmark_room"]["roomsize"] == [4.0, 5.0]
        assert native_input["feedback_source"] == "native_sceneweaver_only"


def _direct_layout_output() -> list[dict[str, Any]]:
    return [
        {
            "new_object_id": "chair_1",
            "category": "chair",
            "rotation": {"z_angle": 0.0},
            "size_in_meters": {"length": 0.8, "width": 0.8, "height": 1.0},
            "position": {"x": 2.0, "y": 2.0, "z": 0.5},
        }
    ]


def _reference_annotation(request_id: str) -> dict:
    return {
        "annotation_version": "reference_annotation_v1",
        "validation_status": "confirmed",
        "source": "manual",
        "request_id": request_id,
        "scene_type": "room",
        "inventory_policy": "open_world",
        "objects": [
            {
                "id": "chair",
                "category": "chair",
                "description": "chair",
                "count": 1,
                "claim_state": "confirmed",
            }
        ],
        "oor_relations": [],
        "oar_relations": [],
        "room_constraints": {"claim_state": "not_mentioned"},
        "provenance": {"private_marker": PRIVATE_SENTINEL},
    }


_FAKE_UPSTREAM_SCRIPT = r'''from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

adapter, native_input_path, output_dir, secret = sys.argv[1:5]
native_input = json.loads(Path(native_input_path).read_text(encoding="utf-8"))
if "HIDDEN_EVAL_SENTINEL" in json.dumps(native_input):
    raise SystemExit(91)
if secret != "super-secret-token":
    raise SystemExit(92)
mode = os.environ.get("FAKE_MODE", "success")
print(f"fake stdout: {adapter}", flush=True)
print(f"fake stderr: {adapter}", file=sys.stderr, flush=True)
if mode == "nonzero":
    raise SystemExit(7)
if mode == "timeout":
    time.sleep(2)
    raise SystemExit(0)
output = Path(output_dir)
output.mkdir(parents=True, exist_ok=True)
if mode == "missing":
    raise SystemExit(0)
if mode == "ambiguous":
    (output / "one.json").write_text("{}", encoding="utf-8")
    (output / "two.json").write_text("{}", encoding="utf-8")
    raise SystemExit(0)
if mode == "malformed":
    (output / "malformed.json").write_text("{}", encoding="utf-8")
    raise SystemExit(0)

if adapter == "layout_gpt":
    attributes = {
        "length": 0.8,
        "width": 0.8,
        "height": 1.0,
        "left": 2.0,
        "top": 2.0,
        "depth": 0.5,
        "orientation": 0.0,
    }
    if mode != "sidecar":
        attributes["asset"] = {"asset_key": "chair-asset", "category": "chair"}
    payload = {
        "query_id": native_input["query_id"],
        "unit": "m",
        "object_list": [["chair", attributes]],
    }
    (output / "layoutgpt.json").write_text(json.dumps(payload), encoding="utf-8")
    if mode == "sidecar":
        (output / "asset_ids.json").write_text(
            json.dumps({"chair_1": "chair-asset"}), encoding="utf-8"
        )
elif adapter == "direct_layout":
    payload = [{
        "new_object_id": "chair_1",
        "category": "chair",
        "rotation": {"z_angle": 0.0},
        "size_in_meters": {"length": 0.8, "width": 0.8, "height": 1.0},
        "position": {"x": 2.0, "y": 2.0, "z": 0.5},
    }]
    (output / "direct.json").write_text(json.dumps(payload), encoding="utf-8")
elif adapter == "layout_vlm":
    payload = {"chair-asset-0": {
        "position": [2.0, 2.0, 0.5],
        "rotation": [0.0, 0.0, 0.0],
    }}
    (output / "layout.json").write_text(json.dumps(payload), encoding="utf-8")
elif adapter == "respace":
    payload = dict(native_input["scene"])
    payload["objects"] = [{
        "id": "chair_1",
        "category": "chair",
        "sampled_asset_jid": "chair-asset",
        "pos": [2.0, 0.0, -2.0],
        "rot": [0.0, 0.0, 0.0, 1.0],
        "size": [0.8, 1.0, 0.8],
    }]
    (output / "scene.json").write_text(json.dumps(payload), encoding="utf-8")
elif adapter == "scene_weaver":
    root = output / "sceneweaver_native"
    record = root / "record_scene"
    record.mkdir(parents=True)
    for iteration in range(2):
        obj = {
            "location": [1.5 + iteration, 2.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "size": [0.8, 0.8, 1.0],
            "parent": [],
        }
        if mode != "sidecar":
            obj["asset_id"] = "chair-asset"
        payload = {
            "roomsize": [4, 5],
            "structure": {},
            "objects": {"chair_0": obj},
        }
        (record / f"layout_{iteration}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        (record / f"render_{iteration}.txt").write_text(
            f"render {iteration}", encoding="utf-8"
        )
    if mode == "sidecar":
        (output / "asset_bindings.json").write_text(
            json.dumps({"chair_0": {"asset_key": "chair-asset", "category": "chair"}}),
            encoding="utf-8",
        )
else:
    raise SystemExit(93)
'''
