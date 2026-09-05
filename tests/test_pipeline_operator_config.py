"""No-API operator config tests; production/native readiness is not simulated."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.generation_comparison.model_policy import api_base_sha256
from benchmark.generation_comparison import pilot
from benchmark.generation_comparison.public_brief import revise_frozen_public_brief
from benchmark.utils.io import read_json, write_json
from benchmark.models.openai_compatible_model import OpenAICompatibleModel
from benchmark.rendering import BlenderRenderer


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/generation_comparison"


@pytest.fixture
def compiler():
    spec = importlib.util.spec_from_file_location("pipeline_operator_ci", ROOT / "scripts/prepare_pipeline_operator_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    source, _ = revise_frozen_public_brief(
        read_json(CONFIGS / "frozen_imaginarium_scene10_v1.json"),
        read_json(CONFIGS / "frozen_imaginarium_scene10_public_brief_v2.json"),
    )
    spec_path = write_json(tmp_path / "source.json", source)
    icl = write_json(tmp_path / "icl.json", [{"role": "user", "content": "synthetic training example"}])
    runtime = read_json(CONFIGS / "canonical_evaluation_runtime.example.json")
    runtime["renderer"]["blender_bin"] = sys.executable
    for role in ("judge", "camera_selector"):
        runtime[role].update(endpoint="http://127.0.0.1:1/v1/chat/completions", model="fixture-evaluator")
    runtime_path = write_json(tmp_path / "runtime.json", runtime)
    bindings = read_json(CONFIGS / "pipeline_operator_bindings.example.json")
    bindings.update(source_spec_sha256=canonical_json_sha256(source), benchmark_python=sys.executable,
                    asset_root=str(tmp_path / "assets"), asset_bundle_root=str(tmp_path / "glbs"),
                    runs_root=str(tmp_path / "new_runs"), evaluation_runtime_config=str(runtime_path))
    bindings["shared_model"] = {"provider": "fixture", "model_id": "exact-observed-model", "deployment_id": "fixture-route", "api_base_url": "http://127.0.0.1:1/v1"}
    for name, host in bindings["upstreams"].items():
        host.update(repo_path=str(tmp_path / name), python_executable=sys.executable)
    bindings["catalog_placement"] = {"endpoint": "http://127.0.0.1:1/v1/chat/completions", "model": "separate-baseline", "api_key_env": "PIPELINE_TEST_BASELINE_KEY"}
    bindings["layoutgpt_icl"].update(path=str(icl), sha256=hashlib.sha256(icl.read_bytes()).hexdigest())
    binding_path = write_json(tmp_path / "bindings.json", bindings)
    return spec_path, binding_path, tmp_path / "bound"


def test_bind_same_model_routes_icl_and_cases_without_calls_or_source_mutation(compiler, inputs, monkeypatch):
    spec, binding, output = inputs
    before = {path: path.read_bytes() for path in spec.parent.glob("*.json")}
    def forbidden(*a, **k):
        pytest.fail("configuration binding cannot call services, generate, render or launch subprocesses")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(OpenAICompatibleModel, "chat_messages", forbidden)
    monkeypatch.setattr(BlenderRenderer, "render_scene", forbidden)
    monkeypatch.setattr(pilot, "run_controlled_generation", forbidden)
    manifest = compiler.bind_operator_config(spec_path=spec, bindings_path=binding, out_dir=output)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert manifest["status"] == "CONFIG_BOUND_NOT_PREFLIGHTED"
    assert manifest["cases_assets_scoring_unchanged"]
    original, revised = read_json(spec), read_json(output / "spec.json")
    for key in original.keys() - {"generation", "evaluator"}:
        assert revised[key] == original[key]
    assert revised["evaluator"] == {**original["evaluator"], "acceptance_policy": "frozen_assets_required_metrics_v1"}
    assert manifest["source_evaluator_policy_sha256"] == canonical_json_sha256(original["evaluator"])
    assert manifest["bound_evaluator_policy_sha256"] == canonical_json_sha256(revised["evaluator"])
    expected_generation = deepcopy(original["generation"])
    expected_generation["model_policy"] = revised["generation"]["model_policy"]
    expected_generation["harness_inputs"]["layout_gpt"] = revised["generation"]["harness_inputs"]["layout_gpt"]
    assert revised["generation"] == expected_generation
    methods = read_json(output / "methods.json")["methods"]
    policy = revised["generation"]["model_policy"]
    for name in compiler.HARNESSES:
        config = methods[name]["adapter_config"]
        assert config["model_identity"] == policy["required_identity"]
        assert config["model_deployment_id"] == policy["required_deployment_id"]
        env = config["execution"]["environment"]
        endpoint = next(iter(env.values()))
        assert api_base_sha256(endpoint, completion_endpoint=name == "layout_gpt") == policy["required_api_base_sha256"]
    assert methods["catalog_placement"]["adapter_config"]["model"] == "separate-baseline"
    icl = read_json(binding)["layoutgpt_icl"]
    assert (output / "layoutgpt_icl_messages.json").read_bytes() == Path(icl["path"]).read_bytes()
    assert revised["generation"]["harness_inputs"]["layout_gpt"]["icl_sha256"] == icl["sha256"]
    assert revised["generation"]["harness_inputs"]["layout_gpt"]["status"] == "human_approved"
    assert revised["generation"]["harness_inputs"]["layout_gpt"]["hidden_evaluator_data_used"] is False
    assert "fixture-evaluator" not in json.dumps(methods)
    assert "evaluation_runtime" not in json.dumps(methods)
    assert "acceptance_policy" not in json.dumps(methods)
    assert all(hashlib.sha256((output / name).read_bytes()).hexdigest() == digest for name, digest in manifest["files"].items())
    assert not manifest["service_contacted"] and not manifest["real_native_environment_qualified"]
    assert not Path(read_json(binding)["runs_root"]).exists()


def test_legacy_bindings_preserve_strict_complete_score_default(compiler, inputs):
    spec, binding, output = inputs
    data = read_json(binding)
    data.pop("evaluation_acceptance_policy")
    write_json(binding, data)
    manifest = compiler.bind_operator_config(spec_path=spec, bindings_path=binding, out_dir=output)
    assert read_json(output / "spec.json")["evaluator"] == read_json(spec)["evaluator"]
    assert manifest["evaluation_acceptance_policy"] == "complete_score_v1"
    assert "evaluator.acceptance_policy" not in manifest["changed_spec_fields"]


@pytest.mark.parametrize("value", [None, "allow_all_partial", 0.8])
def test_invalid_acceptance_binding_cannot_create_output(compiler, inputs, value):
    spec, binding, output = inputs
    data = read_json(binding)
    data["evaluation_acceptance_policy"] = value
    write_json(binding, data)
    with pytest.raises(ValueError, match="acceptance policy"):
        compiler.bind_operator_config(spec_path=spec, bindings_path=binding, out_dir=output)
    assert not output.exists()


def test_launch_plan_routes_existing_cli_and_all_three_full_repetitions(compiler, inputs, monkeypatch, capsys):
    spec, binding, output = inputs
    compiler.bind_operator_config(spec_path=spec, bindings_path=binding, out_dir=output)
    plan = read_json(output / "launch_plan.json")
    assert [stage["planned_units"] for stage in plan["stages"]] == [5, 5, 50, 50, 50]
    assert plan["formal_planned_units"] == 150
    assert len({stage["prepared_dir"] for stage in plan["stages"]}) == 5
    seen = []
    monkeypatch.setattr(pilot, "prepare_controlled_pilot", lambda **kwargs: seen.append(kwargs) or {})
    monkeypatch.setattr(pilot, "preflight_prepared_pilot", lambda **kwargs: {"ready_for_generation": False})
    monkeypatch.setattr(pilot, "run_prepared_pilot", lambda **kwargs: {"status": "blocked"})
    for stage in plan["stages"]:
        for key in ("prepare_argv", "preflight_argv", "run_argv"):
            argv = stage[key]
            assert argv[1:3] == ["-m", "benchmark.generation_comparison.pilot"]
            monkeypatch.setattr(sys, "argv", ["pilot", *argv[3:]])
            if key == "prepare_argv":
                pilot.main()
            else:
                with pytest.raises(SystemExit) as failure:
                    pilot.main()
                assert failure.value.code == 2
    assert [kwargs["case_ids"] for kwargs in seen] == [["S100"], ["S101"], None, None, None]
    assert all(kwargs["evaluation_runtime_config"] == str(output / "evaluation_runtime.json") for kwargs in seen)
    assert plan["reevaluation_may_spend_judge_calls"] and not plan["reevaluation_reexecutes_generation_or_conversion"]
    assert not plan["commands_executed"] and plan["requires_separate_real_execution_authorization"]
    capsys.readouterr()


@pytest.mark.parametrize("violation", ["source_hash", "icl_hash", "icl_approval", "literal_key", "runtime_key", "url_credentials", "base_not_endpoint", "placeholder", "unknown_method", "relative_path", "existing_runs", "unapproved_cohort", "plugin_pin"])
def test_invalid_bindings_fail_before_writes(compiler, inputs, violation, monkeypatch):
    spec, binding, output = inputs
    data = read_json(binding)
    if violation == "source_hash":
        data["source_spec_sha256"] = "0" * 64
    elif violation == "icl_hash":
        data["layoutgpt_icl"]["sha256"] = "0" * 64
    elif violation == "icl_approval":
        data["layoutgpt_icl"]["approval"] = "pending"
    elif violation == "literal_key":
        data["shared_model"]["api_key"] = "SECRET_SENTINEL"
    elif violation == "runtime_key":
        runtime = read_json(data["evaluation_runtime_config"])
        runtime["judge"]["api_key"] = "SECRET_SENTINEL"
        write_json(data["evaluation_runtime_config"], runtime)
    elif violation == "url_credentials":
        data["shared_model"]["api_base_url"] = "https://u:SECRET_SENTINEL@example.org/v1"
    elif violation == "base_not_endpoint":
        data["shared_model"]["api_base_url"] += "/chat/completions"
    elif violation == "placeholder":
        data["shared_model"]["model_id"] = "YOUR-MODEL"
    elif violation == "unknown_method":
        data["upstreams"]["respace"] = {}
    elif violation == "relative_path":
        data["asset_root"] = "relative"
    elif violation == "existing_runs":
        Path(data["runs_root"]).mkdir()
    elif violation == "unapproved_cohort":
        source = read_json(spec)
        source["methods"].remove("scene_weaver")
        write_json(spec, source)
        data["source_spec_sha256"] = canonical_json_sha256(source)
    else:
        actual_hash = compiler._hash
        monkeypatch.setattr(compiler, "_hash", lambda path: "0" * 64 if path.name == "scene_weaver_frozen_plugin.py" else actual_hash(path))
    write_json(binding, data)
    with pytest.raises(ValueError) as error:
        compiler.bind_operator_config(spec_path=spec, bindings_path=binding, out_dir=output)
    assert "SECRET_SENTINEL" not in str(error.value)
    assert not output.exists()


def test_configuration_is_append_only(compiler, inputs):
    spec, binding, output = inputs
    compiler.bind_operator_config(spec_path=spec, bindings_path=binding, out_dir=output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        compiler.bind_operator_config(spec_path=spec, bindings_path=binding, out_dir=output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_preserves_venv_interpreter_symlinks_in_methods_and_launch_plan(compiler, inputs):
    spec, binding, output = inputs
    python = spec.parent / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    data = read_json(binding)
    data["benchmark_python"] = str(python)
    for host in data["upstreams"].values():
        host["python_executable"] = str(python)
    write_json(binding, data)
    compiler.bind_operator_config(spec_path=spec, bindings_path=binding, out_dir=output)
    methods = read_json(output / "methods.json")["methods"]
    for name in compiler.HARNESSES:
        assert methods[name]["adapter_config"]["execution"]["python_executable"] == str(python)
    for stage in read_json(output / "launch_plan.json")["stages"]:
        assert stage["prepare_argv"][0] == str(python)


def test_example_identity_matches_approved_revised_source():
    source, _ = revise_frozen_public_brief(read_json(CONFIGS / "frozen_imaginarium_scene10_v1.json"),
                                          read_json(CONFIGS / "frozen_imaginarium_scene10_public_brief_v2.json"))
    assert read_json(CONFIGS / "pipeline_operator_bindings.example.json")["source_spec_sha256"] == canonical_json_sha256(source)
