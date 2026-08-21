from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest

from scripts import run_camera_cal_scene_level as runner


ROOT = Path(__file__).resolve().parents[1]
E0_CONTRACT = (
    ROOT / "tests/fixtures/camera_cal_scene_level_e0_contract.json"
)
REQUIRED_CASE_OUTPUTS = (
    "evaluation_report.json",
    "grouping.json",
    "l1_report.json",
    "l1_diagnostics.json",
    "scene_quality_report.json",
    "scene_comparison.json",
    "control_manifest.json",
)
pytestmark = pytest.mark.requires_git_history
EXTERNAL_RUNTIME_NAMES = (
    "BlenderRenderer",
    "CameraEvidenceProvider",
    "CameraViewEvidenceRenderer",
    "CameraCandidatePreviewRenderer",
    "DeterministicLocalCameraSelector",
    "build_openai_compatible_vlm_judge",
    "build_openai_compatible_camera_selector",
    "build_grouping_model",
    "load_collision_geometry_manifest",
    "run_evaluate",
)


def _historical_runner() -> ModuleType:
    """Load the E0 runner without importing the current compatibility facade."""

    contract = json.loads(E0_CONTRACT.read_text(encoding="utf-8"))
    source = subprocess.check_output(
        [
            "git",
            "show",
            (
                f"{contract['runner']['git_commit_sha']}:"
                f"{contract['runner']['path']}"
            ),
        ],
        cwd=ROOT,
    )
    module = ModuleType("camera_cal_scene_level_e0_execution")
    module.__file__ = str(ROOT / contract["runner"]["path"])
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _case_runtime_module() -> ModuleType:
    """Load the required E4 single-case runtime module."""

    module = importlib.import_module(
        "benchmark.camera_cal_scene_level.case_runtime"
    )
    assert hasattr(module, "CaseRuntimeDeps")
    assert hasattr(module, "run_case_impl")
    return module


def _package_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    return imports


def _call_keyword_order(
    source: str,
    *,
    function_name: str,
    call_name: str,
) -> list[str | None]:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )

    def dotted_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and dotted_name(node.func).split(".")[-1] == call_name
    ]
    assert len(calls) == 1
    return [keyword.arg for keyword in calls[0].keywords]


def _case_kwargs(output_root: Path) -> dict[str, Any]:
    case = runner.discover_cases(
        runner.DEFAULT_DATASET_ROOT,
        case_ids=["N001"],
    )[0]
    return {
        "case": case,
        "dataset_root": runner.DEFAULT_DATASET_ROOT,
        "output_root": output_root,
        "grouping_config_path": runner.DEFAULT_GROUPING_CONFIG,
        "route": {
            "endpoint": "https://example.invalid/v1",
            "model": "fixture-model",
            "api_key_env": "FIXTURE_KEY",
            "authorization_configured": True,
        },
        "metrics": ("functional_consistency",),
        "functional_group_local_granularity": "per_check",
        "functional_group_local_evidence_policy": "shared_group_bank",
        "deduction_multiplier": 2.0,
        "renderer_config": {
            "blender_bin": "/fixture/blender",
            "timeout_seconds": 10,
            "width": 64,
            "height": 64,
            "render_engine": "BLENDER_EEVEE_NEXT",
            "cycles_device": "CPU",
            "cycles_samples": 1,
            "cycles_denoising": False,
            "preview_render_engine": "BLENDER_EEVEE_NEXT",
            "preview_width": 64,
            "preview_height": 64,
            "preview_cycles_samples": 1,
        },
        "control_config": runner.resolved_control().to_dict(),
        "resume": True,
        "export_audit_graphs": False,
        "l3_only": False,
        "progress": None,
        "model_route_abort_signal": None,
    }


def _case_kwargs_for_module(module: ModuleType, output_root: Path) -> dict[str, Any]:
    """Build identical inputs through either the current or E0 module."""

    case = module.discover_cases(
        module.DEFAULT_DATASET_ROOT,
        case_ids=["N001"],
    )[0]
    return {
        "case": case,
        "dataset_root": module.DEFAULT_DATASET_ROOT,
        "output_root": output_root,
        "grouping_config_path": module.DEFAULT_GROUPING_CONFIG,
        "route": {
            "endpoint": "https://example.invalid/v1",
            "model": "fixture-model",
            "api_key_env": "FIXTURE_KEY",
            "authorization_configured": True,
        },
        "metrics": ("functional_consistency",),
        "functional_group_local_granularity": "per_check",
        "functional_group_local_evidence_policy": "shared_group_bank",
        "deduction_multiplier": 2.0,
        "renderer_config": {
            "blender_bin": "/fixture/blender",
            "timeout_seconds": 10,
            "width": 64,
            "height": 64,
            "render_engine": "BLENDER_EEVEE_NEXT",
            "cycles_device": "CPU",
            "cycles_samples": 1,
            "cycles_denoising": False,
            "preview_render_engine": "BLENDER_EEVEE_NEXT",
            "preview_width": 64,
            "preview_height": 64,
            "preview_cycles_samples": 1,
        },
        "control_config": module.resolved_control().to_dict(),
        "resume": True,
        "export_audit_graphs": False,
        "l3_only": False,
        "progress": None,
        "model_route_abort_signal": None,
    }


def _case_fingerprint(kwargs: dict[str, Any]) -> str:
    case = kwargs["case"]
    source_root = Path(str(case["case_root"])).resolve()
    manifest = runner.read_json(source_root / "case_manifest.json")
    paths = runner.case_paths(source_root, manifest)
    grouping_config = runner.read_yaml_object(kwargs["grouping_config_path"])
    return runner.case_input_fingerprint(
        case=case,
        case_manifest=manifest,
        paths=paths,
        route=kwargs["route"],
        metrics=kwargs["metrics"],
        functional_group_local_granularity=(
            kwargs["functional_group_local_granularity"]
        ),
        functional_group_local_evidence_policy=(
            kwargs["functional_group_local_evidence_policy"]
        ),
        deduction_multiplier=kwargs["deduction_multiplier"],
        grouping_config=grouping_config,
        renderer_config=kwargs["renderer_config"],
        control_config=kwargs["control_config"],
        l3_only=kwargs["l3_only"],
    )


def _write_resumable_case(kwargs: dict[str, Any]) -> Path:
    case_out = kwargs["output_root"] / "cases" / kwargs["case"]["case_id"]
    case_out.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_CASE_OUTPUTS:
        runner.atomic_write_json(case_out / name, {})
    runner.atomic_write_json(
        case_out / "case_run_manifest.json",
        {
            "schema_version": runner.CASE_SCHEMA_VERSION,
            "case_id": kwargs["case"]["case_id"],
            "status": "complete",
            "input_fingerprint": _case_fingerprint(kwargs),
            "elapsed_seconds": 1.25,
            "grouping_status": "complete",
            "l1_status": "evaluated",
            "l3_status": "evaluated",
            "final_decision_status": "resolved",
        },
    )
    return case_out


def _patch_external_guards(
    monkeypatch: pytest.MonkeyPatch,
    trace: list[tuple[str, str]],
) -> None:
    targets: list[Any] = [runner]
    case_runtime = _case_runtime_module()
    targets.append(case_runtime)

    def guard(name: str) -> Callable[..., Any]:
        def blocked(*_: Any, **__: Any) -> Any:
            trace.append(("forbidden_external", name))
            raise AssertionError(f"external runtime was touched: {name}")

        return blocked

    for target in targets:
        for name in EXTERNAL_RUNTIME_NAMES:
            if hasattr(target, name):
                monkeypatch.setattr(target, name, guard(name))


def _fixture_report() -> dict[str, Any]:
    metric = {
        "metric": "functional_consistency",
        "status": "evaluated",
        "score": 1.0,
        "coverage": {
            "complete": True,
            "eligible_count": 0,
            "resolved_count": 0,
        },
        "group_results": [],
    }
    return {
        "benchmark_score_status": "complete",
        "benchmark_score_100": 100.0,
        "scoring_reliability": {
            "schema_version": "scoring_reliability_v2",
            "terminal_state": "complete",
        },
        "reports": {
            "object_grouping": {
                "status": "complete",
                "object_groups": [],
            },
            "scene_quality": {
                "status": "evaluated",
                "metrics": {"functional_consistency": metric},
            },
        },
        "layer_reports": {
            runner.L1: {
                "status": "evaluated",
                "metrics": {},
            }
        },
        "evaluation_config": {
            "vlm_evaluation_control": {
                "integration": {"runtime": {"controlled_calls": []}}
            }
        },
    }


def _install_fake_execution(
    monkeypatch: pytest.MonkeyPatch,
    trace: list[tuple[str, str]],
    *,
    fail_evaluation: bool = False,
) -> None:
    """Patch current and, when present, proposed execution injection points."""

    targets: list[Any] = [runner]
    case_runtime = _case_runtime_module()
    targets.append(case_runtime)

    class FakeRenderer:
        def __init__(self, **_: Any) -> None:
            trace.append(("constructor", "BlenderRenderer"))

    class FakeEvidenceProvider:
        candidate_policy = "fixture_camera_policy"

        def __init__(self, **kwargs: Any) -> None:
            trace.append(
                (
                    "constructor",
                    f"CameraEvidenceProvider:{kwargs.get('mode')}",
                )
            )

    class FakeEvidenceRenderer:
        def __init__(self, **_: Any) -> None:
            trace.append(("constructor", "CameraViewEvidenceRenderer"))

    class FakePreviewRenderer:
        def __init__(self, **_: Any) -> None:
            trace.append(("constructor", "CameraCandidatePreviewRenderer"))

    def fake_selector(**kwargs: Any) -> Any:
        trace.append(("constructor", "DeterministicLocalCameraSelector"))
        return SimpleNamespace(config=kwargs)

    def fake_judge(config: Any) -> Any:
        trace.append(("constructor", "build_openai_compatible_vlm_judge"))
        return SimpleNamespace(config=config)

    def fake_camera_selector(config: Any) -> Any:
        trace.append(
            ("constructor", "build_openai_compatible_camera_selector")
        )
        return SimpleNamespace(config=config)

    def fake_grouping_model(route: Any) -> Any:
        trace.append(("constructor", "build_grouping_model"))
        return SimpleNamespace(route=route)

    def fake_model_config(route: Any, *, role: str) -> dict[str, Any]:
        trace.append(("constructor", f"model_config:{role}"))
        return {"route": route, "role": role}

    def fake_geometry(_: Any) -> dict[str, Any]:
        trace.append(("constructor", "load_collision_geometry_manifest"))
        return {"schema_version": "collision_geometry_v1"}

    def fake_run_evaluate(**kwargs: Any) -> dict[str, Any]:
        trace.append(("call", "run_evaluate"))
        if fail_evaluation:
            raise RuntimeError("fixture evaluation failure")
        report = _fixture_report()
        runner.atomic_write_json(Path(kwargs["out"]), report)
        return deepcopy(report)

    replacements = {
        "BlenderRenderer": FakeRenderer,
        "CameraEvidenceProvider": FakeEvidenceProvider,
        "CameraViewEvidenceRenderer": FakeEvidenceRenderer,
        "CameraCandidatePreviewRenderer": FakePreviewRenderer,
        "DeterministicLocalCameraSelector": fake_selector,
        "build_openai_compatible_vlm_judge": fake_judge,
        "build_openai_compatible_camera_selector": fake_camera_selector,
        "build_grouping_model": fake_grouping_model,
        "model_config": fake_model_config,
        "load_collision_geometry_manifest": fake_geometry,
        "run_evaluate": fake_run_evaluate,
    }
    for target in targets:
        for name, value in replacements.items():
            if hasattr(target, name):
                monkeypatch.setattr(target, name, value)

    original_atomic_write = runner.atomic_write_json

    def traced_atomic_write(path: Path, value: Any) -> None:
        trace.append(("write", Path(path).name))
        original_atomic_write(path, value)

    for target in targets:
        if hasattr(target, "atomic_write_json"):
            monkeypatch.setattr(target, "atomic_write_json", traced_atomic_write)

    fixed_timestamp = "2026-08-21T00:00:00+00:00"
    monkeypatch.setattr(runner, "utc_now", lambda: fixed_timestamp)
    if hasattr(case_runtime, "utc_now"):
        monkeypatch.setattr(case_runtime, "utc_now", lambda: fixed_timestamp)

    monotonic_calls = 0

    def fixed_monotonic() -> float:
        nonlocal monotonic_calls
        value = 100.0 if monotonic_calls == 0 else 101.5
        monotonic_calls += 1
        return value

    # The runner and the extracted execution module both use the stdlib time
    # module.  Patching this one clock gives deterministic elapsed_seconds for
    # either implementation without adding a production-only test hook.
    monkeypatch.setattr(runner.time, "monotonic", fixed_monotonic)


def _install_fake_historical_runtime(
    monkeypatch: pytest.MonkeyPatch,
    historical: ModuleType,
    trace: list[tuple[str, str]],
) -> None:
    """Patch the E0 module's own globals, without touching the current facade."""

    class FakeRenderer:
        def __init__(self, **_: Any) -> None:
            trace.append(("constructor", "BlenderRenderer"))

    class FakeEvidenceProvider:
        candidate_policy = "fixture_camera_policy"

        def __init__(self, **kwargs: Any) -> None:
            trace.append(
                (
                    "constructor",
                    f"CameraEvidenceProvider:{kwargs.get('mode')}",
                )
            )

    class FakeEvidenceRenderer:
        def __init__(self, **_: Any) -> None:
            trace.append(("constructor", "CameraViewEvidenceRenderer"))

    class FakePreviewRenderer:
        def __init__(self, **_: Any) -> None:
            trace.append(("constructor", "CameraCandidatePreviewRenderer"))

    def fake_selector(**kwargs: Any) -> Any:
        trace.append(("constructor", "DeterministicLocalCameraSelector"))
        return SimpleNamespace(config=kwargs)

    def fake_judge(config: Any) -> Any:
        trace.append(("constructor", "build_openai_compatible_vlm_judge"))
        return SimpleNamespace(config=config)

    def fake_camera_selector(config: Any) -> Any:
        trace.append(
            ("constructor", "build_openai_compatible_camera_selector")
        )
        return SimpleNamespace(config=config)

    def fake_grouping_model(route: Any) -> Any:
        trace.append(("constructor", "build_grouping_model"))
        return SimpleNamespace(route=route)

    def fake_model_config(route: Any, *, role: str) -> dict[str, Any]:
        trace.append(("constructor", f"model_config:{role}"))
        return {"route": route, "role": role}

    def fake_geometry(_: Any) -> dict[str, Any]:
        trace.append(("constructor", "load_collision_geometry_manifest"))
        return {"schema_version": "collision_geometry_v1"}

    def fake_run_evaluate(**kwargs: Any) -> dict[str, Any]:
        trace.append(("call", "run_evaluate"))
        report = _fixture_report()
        historical.atomic_write_json(Path(kwargs["out"]), report)
        return deepcopy(report)

    replacements = {
        "BlenderRenderer": FakeRenderer,
        "CameraEvidenceProvider": FakeEvidenceProvider,
        "CameraViewEvidenceRenderer": FakeEvidenceRenderer,
        "CameraCandidatePreviewRenderer": FakePreviewRenderer,
        "DeterministicLocalCameraSelector": fake_selector,
        "build_openai_compatible_vlm_judge": fake_judge,
        "build_openai_compatible_camera_selector": fake_camera_selector,
        "build_grouping_model": fake_grouping_model,
        "model_config": fake_model_config,
        "load_collision_geometry_manifest": fake_geometry,
        "run_evaluate": fake_run_evaluate,
    }
    for name, value in replacements.items():
        if hasattr(historical, name):
            monkeypatch.setattr(historical, name, value)

    original_atomic_write = historical.atomic_write_json

    def traced_atomic_write(path: Path, value: Any) -> None:
        trace.append(("write", Path(path).name))
        original_atomic_write(path, value)

    monkeypatch.setattr(historical, "atomic_write_json", traced_atomic_write)
    fixed_timestamp = "2026-08-21T00:00:00+00:00"
    monkeypatch.setattr(historical, "utc_now", lambda: fixed_timestamp)

    monotonic_calls = 0

    def fixed_monotonic() -> float:
        nonlocal monotonic_calls
        value = 100.0 if monotonic_calls == 0 else 101.5
        monotonic_calls += 1
        return value

    monkeypatch.setattr(historical.time, "monotonic", fixed_monotonic)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalized_resume_identity(value: Any) -> Any:
    normalized = deepcopy(value)
    if isinstance(normalized, dict) and "input_fingerprint" in normalized:
        normalized["input_fingerprint"] = "<versioned-runtime-identity>"
    return normalized


PARITY_ARTIFACTS = (
    "case_run_manifest.json",
    "evaluation_report.json",
    "grouping.json",
    "l1_report.json",
    "l1_diagnostics.json",
    "scene_quality_report.json",
    "scene_comparison.json",
    "control_manifest.json",
    "api_usage.json",
)


def _artifact_snapshot(output_root: Path) -> dict[str, tuple[bytes, bytes]]:
    case_root = output_root / "cases" / "N001"
    snapshot: dict[str, tuple[bytes, bytes]] = {}
    for name in PARITY_ARTIFACTS:
        path = case_root / name
        raw = path.read_bytes()
        snapshot[name] = (raw, _canonical_json(json.loads(raw)))
    progress_path = output_root / "progress.jsonl"
    progress_raw = progress_path.read_bytes()
    progress_values = [
        json.loads(line)
        for line in progress_raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    snapshot["progress.jsonl"] = (
        progress_raw,
        _canonical_json(progress_values),
    )
    return snapshot


def test_e4_package_import_direction_and_run_case_facade_signature() -> None:
    package_root = ROOT / "src/benchmark/camera_cal_scene_level"
    for path in sorted(package_root.glob("*.py")):
        imports = _package_imports(path)
        assert not any(
            name == "scripts" or name.startswith("scripts.")
            for name in imports
        ), path
        assert not any(
            name == "benchmark.evaluation_campaign"
            or name.startswith("benchmark.evaluation_campaign.")
            for name in imports
        ), path

    historical = _historical_runner()
    assert inspect.signature(runner.run_case) == inspect.signature(
        historical.run_case
    )
    assert runner.run_case.__name__ == "run_case"
    assert runner.run_case.__module__ == "scripts.run_camera_cal_scene_level"

    case_runtime = _case_runtime_module()
    deps_type = case_runtime.CaseRuntimeDeps
    assert hasattr(deps_type, "__dataclass_fields__")
    assert {"io", "resume", "policy", "external"}.issubset(
        deps_type.__dataclass_fields__
    )
    run_case_impl_signature = inspect.signature(case_runtime.run_case_impl)
    assert "deps" in run_case_impl_signature.parameters


def test_e4_run_evaluate_keyword_contract_matches_e0() -> None:
    contract = json.loads(E0_CONTRACT.read_text(encoding="utf-8"))
    historical_source = subprocess.check_output(
        [
            "git",
            "show",
            (
                f"{contract['runner']['git_commit_sha']}:"
                f"{contract['runner']['path']}"
            ),
        ],
        cwd=ROOT,
        text=True,
    )
    case_runtime = _case_runtime_module()
    current_source = Path(case_runtime.__file__).read_text(encoding="utf-8")

    assert _call_keyword_order(
        historical_source,
        function_name="run_case",
        call_name="run_evaluate",
    ) == _call_keyword_order(
        current_source,
        function_name="run_case_impl",
        call_name="run_evaluate",
    )


def test_e4_resume_checks_outputs_before_external_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _case_kwargs(tmp_path / "resumed")
    _write_resumable_case(kwargs)
    trace: list[tuple[str, str]] = []
    _patch_external_guards(monkeypatch, trace)

    result = runner.run_case(**kwargs)

    assert result["status"] == "resumed"
    assert not [event for event in trace if event[0] == "forbidden_external"]
    assert not [event for event in trace if event[0] == "call"]


def test_e4_stale_resume_fails_before_external_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _case_kwargs(tmp_path / "stale")
    case_out = _write_resumable_case(kwargs)
    manifest_path = case_out / "case_run_manifest.json"
    manifest = runner.read_json(manifest_path)
    manifest["input_fingerprint"] = "0" * 64
    runner.atomic_write_json(manifest_path, manifest)
    trace: list[tuple[str, str]] = []
    _patch_external_guards(monkeypatch, trace)

    with pytest.raises(RuntimeError, match="fingerprint does not match"):
        runner.run_case(**kwargs)

    assert not [event for event in trace if event[0] == "forbidden_external"]
    assert not [event for event in trace if event[0] == "call"]


def test_e4_success_trace_keeps_constructor_evaluation_and_write_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _case_kwargs(tmp_path / "success")
    trace: list[tuple[str, str]] = []
    _install_fake_execution(monkeypatch, trace)

    result = runner.run_case(**kwargs)

    case_out = kwargs["output_root"] / "cases" / "N001"
    assert result["status"] == "complete"
    assert result["elapsed_seconds"] == 1.5
    assert {
        *REQUIRED_CASE_OUTPUTS,
        "case_run_manifest.json",
        "api_usage.json",
    }.issubset({path.name for path in case_out.iterdir()})

    manifest = runner.read_json(case_out / "case_run_manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["started_at"] == "2026-08-21T00:00:00+00:00"
    assert manifest["completed_at"] == "2026-08-21T00:00:00+00:00"
    assert manifest["elapsed_seconds"] == 1.5

    constructor_indices = [
        index
        for index, event in enumerate(trace)
        if event[0] == "constructor"
    ]
    evaluation_index = trace.index(("call", "run_evaluate"))
    assert constructor_indices
    assert max(constructor_indices) < evaluation_index

    writes = [
        (index, name)
        for index, (kind, name) in enumerate(trace)
        if kind == "write"
    ]
    first_manifest = next(
        index for index, name in writes if name == "case_run_manifest.json"
    )
    first_report = next(
        index for index, name in writes if name == "evaluation_report.json"
    )
    assert first_manifest < evaluation_index
    assert evaluation_index < first_report
    assert writes[-1][1] == "case_run_manifest.json"

    expected_constructors = [
        "load_collision_geometry_manifest",
        "model_config:judge",
        "model_config:camera-selector",
        "build_grouping_model",
        "build_openai_compatible_vlm_judge",
        "build_openai_compatible_camera_selector",
        "BlenderRenderer",
        "CameraEvidenceProvider:auto",
        "CameraEvidenceProvider:visibility_ranked",
        "CameraEvidenceProvider:query_cov",
        "DeterministicLocalCameraSelector",
        "CameraViewEvidenceRenderer",
        "CameraCandidatePreviewRenderer",
    ]
    observed_constructors = [
        name for kind, name in trace if kind == "constructor"
    ]
    assert observed_constructors == expected_constructors
    assert sum(event == ("call", "run_evaluate") for event in trace) == 1


def test_e4_exception_trace_keeps_partial_artifacts_and_failure_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _case_kwargs(tmp_path / "failure")
    trace: list[tuple[str, str]] = []
    _install_fake_execution(monkeypatch, trace, fail_evaluation=True)

    with pytest.raises(RuntimeError, match="fixture evaluation failure"):
        runner.run_case(**kwargs)

    case_out = kwargs["output_root"] / "cases" / "N001"
    manifest_path = case_out / "case_run_manifest.json"
    usage_path = case_out / "api_usage.json"
    progress_path = kwargs["output_root"] / "progress.jsonl"
    assert manifest_path.is_file()
    assert usage_path.is_file()
    assert progress_path.is_file()

    manifest = runner.read_json(manifest_path)
    assert manifest["status"] in {"running", "failed"}
    events = [
        json.loads(line)["event"]
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "evaluation_failed" in events

    evaluation_index = trace.index(("call", "run_evaluate"))
    writes = [
        (index, name)
        for index, (kind, name) in enumerate(trace)
        if kind == "write"
    ]
    manifest_index = next(
        index for index, name in writes if name == "case_run_manifest.json"
    )
    assert manifest_index < evaluation_index
    assert writes[-1][1] == "api_usage.json"


def test_e4_current_facade_matches_e0_inline_run_case_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay both implementations with identical injected runtime behavior.

    Evaluator outputs remain byte-identical. The one deliberate compatibility
    change is the opaque resume fingerprint, which now includes the extracted
    runtime source identity and therefore differs from E0.
    """

    historical = _historical_runner()
    output_root = tmp_path / "same-output-root"

    with monkeypatch.context() as current_patch:
        current_trace: list[tuple[str, str]] = []
        _install_fake_execution(current_patch, current_trace)
        current_kwargs = _case_kwargs_for_module(runner, output_root)
        current_result = runner.run_case(**current_kwargs)
    current_snapshot = _artifact_snapshot(output_root)

    assert output_root.is_dir()
    assert not output_root.is_symlink()
    shutil.rmtree(output_root)
    assert not output_root.exists()

    with monkeypatch.context() as historical_patch:
        historical_trace: list[tuple[str, str]] = []
        _install_fake_historical_runtime(
            historical_patch,
            historical,
            historical_trace,
        )
        historical_kwargs = _case_kwargs_for_module(historical, output_root)
        historical_result = historical.run_case(**historical_kwargs)
    historical_snapshot = _artifact_snapshot(output_root)

    assert current_result["input_fingerprint"] != historical_result[
        "input_fingerprint"
    ]
    assert _canonical_json(
        _normalized_resume_identity(current_result)
    ) == _canonical_json(
        _normalized_resume_identity(historical_result)
    )
    assert current_trace == historical_trace
    assert set(current_snapshot) == set(historical_snapshot)
    for name in PARITY_ARTIFACTS + ("progress.jsonl",):
        current_raw, current_canonical = current_snapshot[name]
        historical_raw, historical_canonical = historical_snapshot[name]
        if name == "case_run_manifest.json":
            assert _canonical_json(
                _normalized_resume_identity(json.loads(current_raw))
            ) == _canonical_json(
                _normalized_resume_identity(json.loads(historical_raw))
            )
            assert current_raw != historical_raw
        else:
            assert current_canonical == historical_canonical, name
            assert current_raw == historical_raw, name


def test_e4_case_runtime_has_only_injected_runtime_dependencies() -> None:
    """Keep evaluator/renderer/model ownership in the compatibility facade."""

    case_runtime = _case_runtime_module()
    source = Path(case_runtime.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        name == "scripts" or name.startswith("scripts.")
        for name in imported_modules
    )
    assert not any(
        name == "benchmark.evaluation_campaign"
        or name.startswith("benchmark.evaluation_campaign.")
        for name in imported_modules
    )
    assert "benchmark.camera_cal_scene_level.io" not in imported_modules
    assert not any(
        name == "benchmark.api.evaluation"
        or name.startswith("benchmark.api.evaluation.")
        or name == "benchmark.models"
        or name.startswith("benchmark.models.")
        or name == "benchmark.rendering"
        or name.startswith("benchmark.rendering.")
        or name == "benchmark.evaluator"
        or name.startswith("benchmark.evaluator.")
        or name == "benchmark.visual_judge"
        or name.startswith("benchmark.visual_judge.")
        for name in imported_modules
    ), (
        "case_runtime must receive evaluator, renderer, model, and IO "
        "dependencies through CaseRuntimeDeps"
    )
