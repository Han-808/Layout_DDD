#!/usr/bin/env python3
"""Bind the approved Scene10 experiment to one host; never launch a workflow.

This is a configuration compiler, not another runner. Its output uses the
existing controlled-pilot prepare/preflight/run and append-only reevaluation.
Only deployment/path/ICL bindings change; cases, assets and scoring do not.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from benchmark.generation_comparison.evaluation_runtime import CanonicalEvaluationRuntime
from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.generation_comparison.model_policy import api_base_sha256, normalize_model_identity
from benchmark.generation_comparison.pilot import _validate_pilot_spec, bridge_execution_hashes
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/generation_comparison"
HARNESSES = ("layout_gpt", "direct_layout", "layout_vlm", "scene_weaver")
METHODS = ("catalog_placement", *HARNESSES)


def _fields(value: Any, required: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != required:
        # Never print user-supplied values, which could include credentials.
        raise ArtifactValidationError(f"{label} requires exactly: {sorted(required)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ArtifactValidationError(f"{label} requires non-empty, trimmed text")
    if any(marker in value.upper() for marker in ("/ABSOLUTE/", "YOUR-", "YOUR_", "REPLACE_ME")):
        raise ArtifactValidationError(f"{label} still contains a placeholder")
    return value


def _path(value: Any, label: str, *, preserve_symlink: bool = False) -> Path:
    path = Path(_text(value, label))
    if not path.is_absolute():
        raise ArtifactValidationError(f"{label} must be an absolute path")
    # Resolving a venv's bin/python symlink selects the base interpreter and
    # silently loses its installed native dependencies. Keep executable paths.
    return path if preserve_symlink else path.resolve()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _no_literals(value: Any) -> None:
    if isinstance(value, dict):
        if any(str(key).lower() in {"api_key", "authorization", "headers", "token", "password", "secret"}
               for key in value):
            raise ArtifactValidationError("configuration must contain environment names, never credential literals")
        for child in value.values():
            _no_literals(child)
    elif isinstance(value, list):
        for child in value:
            _no_literals(child)


def _credential_name(value: Any) -> str:
    name = _text(value, "credential environment name")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ArtifactValidationError("invalid credential environment name")
    return name


def bind_operator_config(*, spec_path: Path, bindings_path: Path, out_dir: Path) -> dict:
    """Create new private config/plan files; does not prepare or execute runs."""
    output = out_dir.resolve()
    if output.exists():
        raise FileExistsError("operator configuration requires a fresh directory")
    spec = read_json(spec_path)
    original = deepcopy(spec)
    binding = read_json(bindings_path)
    _no_literals(binding)
    _fields(binding, {
        "schema_version", "source_spec_sha256", "benchmark_python", "asset_root",
        "asset_bundle_root", "runs_root", "shared_model", "upstreams",
        "layoutgpt_icl", "catalog_placement", "evaluation_runtime_config",
    }, "operator bindings")
    if binding["schema_version"] != "pipeline_operator_bindings_v1":
        raise ArtifactValidationError("unsupported operator bindings schema")
    if binding["source_spec_sha256"] != canonical_json_sha256(spec):
        raise ArtifactValidationError("source spec hash mismatch; do not rewrite an old prepared manifest")
    _validate_pilot_spec(spec)
    if (spec["methods"] != list(METHODS)
            or [case["case_id"] for case in spec["cases"]] != [f"S{i}" for i in range(100, 110)]
            or spec.get("asset_selection_status") != "human_approved"
            or spec.get("mode") != "frozen_assets"):
        raise ArtifactValidationError("operator plan requires the complete approved Scene10 five-method cohort")

    model = _fields(binding["shared_model"], {"provider", "model_id", "deployment_id", "api_base_url"}, "shared model")
    identity = normalize_model_identity({key: _text(model[key], key) for key in ("provider", "model_id")}, path="shared_model")
    base = _text(model["api_base_url"], "shared model API base").rstrip("/")
    if base.endswith("/chat/completions"):
        raise ArtifactValidationError("shared_model.api_base_url must be a base, not the completion endpoint")
    endpoint_hash = api_base_sha256(base)
    deployment = _text(model["deployment_id"], "deployment ID")
    upstreams = _fields(binding["upstreams"], set(HARNESSES), "upstreams")
    icl = _fields(binding["layoutgpt_icl"], {"path", "sha256", "approval", "provenance"}, "LayoutGPT ICL")
    icl_path = _path(icl["path"], "ICL path")
    icl_bytes = icl_path.read_bytes() if icl_path.is_file() else b""
    if not icl_bytes or hashlib.sha256(icl_bytes).hexdigest() != icl["sha256"] or icl["sha256"] == "0" * 64:
        raise ArtifactValidationError("LayoutGPT approved ICL file/hash mismatch")
    if icl["approval"] != "user_approved_released_training_snapshot":
        raise ArtifactValidationError("LayoutGPT ICL approval missing")
    provenance = _text(icl["provenance"], "ICL provenance")
    benchmark_python = _path(binding["benchmark_python"], "benchmark Python", preserve_symlink=True)
    asset_root = _path(binding["asset_root"], "asset root")
    bundle_root = _path(binding["asset_bundle_root"], "GLB bundle root")
    runs = _path(binding["runs_root"], "new runs root")
    if runs.exists() or runs == output or runs.is_relative_to(output) or output.is_relative_to(runs):
        raise ArtifactValidationError("runs root must be new and separate from the configuration directory")
    runtime_path = _path(binding["evaluation_runtime_config"], "evaluation runtime config")
    runtime = read_json(runtime_path)
    _no_literals(runtime)
    for role in ("judge", "camera_selector"):
        route = runtime.get(role, {})
        _text(route.get("model") or route.get("model_id"), f"{role} model")
        api_base_sha256(_text(route.get("endpoint") or route.get("base_url"), f"{role} endpoint"))
        _credential_name(route.get("api_key_env"))
    # Instantiates the existing local clients/renderer, without credentials,
    # service calls, rendering or any claim that native dependencies work.
    CanonicalEvaluationRuntime(runtime, require_credentials=False)

    methods_path = CONFIGS / "frozen_imaginarium_scene10_methods.example.json"
    methods = read_json(methods_path)
    for name in HARNESSES:
        host = _fields(upstreams[name], {"repo_path", "python_executable"}, f"{name} host")
        config = methods["methods"][name]["adapter_config"]
        config["model_identity"] = deepcopy(identity)
        config["model_deployment_id"] = deployment
        execution = config["execution"]
        execution["repo_path"] = str(_path(host["repo_path"], f"{name} repository"))
        execution["python_executable"] = str(_path(host["python_executable"], f"{name} Python", preserve_symlink=True))
        variables = execution["template_variables"]
        bridge = ROOT / "scripts/external_harness_bridges" / f"{name}_frozen.py"
        variables["bridge_script"] = str(bridge)
        hashes = bridge_execution_hashes(bridge)
        if any(execution[key] != hashes[key] for key in (
            "expected_entrypoint_sha256", "expected_bridge_bundle_sha256",
        )):
            raise ArtifactValidationError(f"{name} template/source pin mismatch; never silently repin")
        endpoint_name = "LAYOUT_DDD_API_ENDPOINT" if name == "layout_gpt" else "LAYOUT_DDD_API_BASE_URL"
        execution["environment"][endpoint_name] = base + "/chat/completions" if name == "layout_gpt" else base
        if name == "layout_gpt":
            variables.update(layoutgpt_icl_examples=str(output / "layoutgpt_icl_messages.json"), layoutgpt_icl_sha256=icl["sha256"])
        if name == "scene_weaver":
            plugin = bridge.with_name("scene_weaver_frozen_plugin.py")
            if _hash(plugin) != variables["frozen_plugin_sha256"]:
                raise ArtifactValidationError("SceneWeaver plugin/template pin mismatch")
            variables["frozen_plugin"] = str(plugin)
    baseline = _fields(binding["catalog_placement"], {"endpoint", "model", "api_key_env"}, "Catalog Placement")
    api_base_sha256(_text(baseline["endpoint"], "baseline endpoint"), completion_endpoint=True)
    _text(baseline["model"], "baseline model")
    _credential_name(baseline["api_key_env"])
    methods["methods"]["catalog_placement"]["adapter_config"].update(baseline)

    generation = spec["generation"]
    policy = generation["model_policy"]
    if policy["comparison_group"] != list(HARNESSES) or policy["excluded_baselines"] != ["catalog_placement"]:
        raise ArtifactValidationError("same-model cohort differs from the approved experiment")
    policy.update(required_identity=identity, required_deployment_id=deployment, required_api_base_sha256=endpoint_hash)
    generation["harness_inputs"]["layout_gpt"].update(
        icl_sha256=icl["sha256"], status="human_approved", provenance=provenance,
    )
    if generation["harness_inputs"]["layout_gpt"].get("hidden_evaluator_data_used") is not False:
        raise ArtifactValidationError("LayoutGPT ICL must retain the no-hidden-evaluator-data contract")
    _validate_pilot_spec(spec)

    stages = []
    prefix = [str(benchmark_python), "-m", "benchmark.generation_comparison.pilot"]
    for stage, selected in (("smoke", ["S100"]), ("dense_pilot", ["S101"]),
                            ("formal_r1", []), ("formal_r2", []), ("formal_r3", [])):
        target = runs / stage
        prepare = prefix + ["prepare", "--spec", str(output / "spec.json"),
                            "--asset-root", str(asset_root), "--asset-bundle-root", str(bundle_root),
                            "--method-configs", str(output / "methods.json"),
                            "--evaluation-runtime-config", str(output / "evaluation_runtime.json"),
                            "--out-dir", str(target)]
        for case_id in selected:
            prepare += ["--case-id", case_id]
        common = ["--prepared-dir", str(target), "--method-configs", str(output / "methods.json")]
        stages.append({
            "stage": stage, "case_ids": selected or [f"S{i}" for i in range(100, 110)],
            "planned_units": 5 if selected else 50, "prepared_dir": str(target),
            "prepare_argv": prepare, "preflight_argv": prefix + ["preflight", *common],
            "run_argv": prefix + ["run", *common],
        })
    plan = {
        "schema_version": "pipeline_operator_plan_v1", "cwd": str(ROOT),
        "environment": {"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        "stages": stages, "formal_planned_units": 150,
        "commands_executed": False, "requires_separate_real_execution_authorization": True,
        "preflight_is_no_call": True, "run_is_real_generation": True,
        "advance_policy": "all_five_qualified_no_score_selection",
        "resume_supported": False, "automatic_generation_retry": False,
        "reevaluation_argv_template": [str(benchmark_python), "-m", "benchmark.generation_comparison.reevaluation",
                                       "--prepared-dir", "{original_prepared_dir}", "--case-id", "{case_id}",
                                       "--method", "{method}", "--out-dir", "{fresh_outside_original_run}"],
        "reevaluation_may_spend_judge_calls": True,
        "reevaluation_reexecutes_generation_or_conversion": False,
        "reevaluation_selects_original_final_state_only": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"spec.json": spec, "methods.json": methods, "evaluation_runtime.json": runtime, "launch_plan.json": plan}
    for name, value in artifacts.items():
        write_json(output / name, value)
    # Preserve the exact approved ICL bytes, not a reserialized approximation.
    with (output / "layoutgpt_icl_messages.json").open("xb") as handle:
        handle.write(icl_bytes)
    manifest = {
        "schema_version": "pipeline_operator_configuration_v1", "status": "CONFIG_BOUND_NOT_PREFLIGHTED",
        "source_spec_sha256": canonical_json_sha256(original), "bound_spec_sha256": canonical_json_sha256(spec),
        "bindings_sha256": _hash(bindings_path), "compiler_sha256": _hash(Path(__file__)),
        "methods_template_sha256": _hash(methods_path), "source_icl_sha256": icl["sha256"],
        "changed_spec_fields": ["generation.model_policy.required_identity", "generation.model_policy.required_deployment_id",
                                "generation.model_policy.required_api_base_sha256", "generation.harness_inputs.layout_gpt"],
        "cases_sha256": canonical_json_sha256(spec["cases"]), "catalog_sha256": canonical_json_sha256(spec["catalog"]),
        "source_evaluator_policy_sha256": canonical_json_sha256(spec["evaluator"]),
        "cases_assets_scoring_unchanged": all(spec[key] == original[key] for key in ("cases", "catalog", "evaluator", "methods")),
        "files": {name: _hash(output / name) for name in (*artifacts, "layoutgpt_icl_messages.json")},
        "service_contacted": False, "generation_executed": False,
        "real_native_environment_qualified": False, "real_upstream_smoke_verified": False,
    }
    write_json(output / "configuration_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = bind_operator_config(spec_path=args.spec, bindings_path=args.bindings, out_dir=args.out_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
