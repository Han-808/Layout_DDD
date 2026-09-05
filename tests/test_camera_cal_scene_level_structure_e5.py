from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from benchmark.camera_cal_scene_level import adapters, case_runtime, composition
from scripts import run_camera_cal_scene_level as runner


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src/benchmark/camera_cal_scene_level"
E0_CONTRACT = ROOT / "tests/fixtures/camera_cal_scene_level_e0_contract.json"
pytestmark = pytest.mark.requires_git_history
RUN_ARTIFACTS = (
    "experiment_plan.json",
    "endpoint_preflight.json",
    "summary.json",
    "run_manifest.json",
)


def _contract() -> dict[str, Any]:
    return json.loads(E0_CONTRACT.read_text(encoding="utf-8"))


def _historical_runner() -> ModuleType:
    contract = _contract()
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{contract['runner']['git_commit_sha']}:{contract['runner']['path']}",
        ],
        cwd=ROOT,
    )
    module = ModuleType("camera_cal_scene_level_e5_e0_runner")
    module.__file__ = str(ROOT / contract["runner"]["path"])
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _public_top_level_names(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes: list[str] = []
    functions: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                functions.append(node.name)
    return {"classes": classes, "functions": functions}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _module_imports(path: Path) -> set[str]:
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


def _main_ast() -> ast.FunctionDef:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(runner.__file__))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _main_args(
    output_root: Path,
    grouping_config: Path,
    blender_bin: Path,
    *,
    max_workers: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=runner.DEFAULT_DATASET_ROOT,
        output_root=output_root,
        grouping_config=grouping_config,
        case_id=["N001"],
        metric=["functional_consistency"],
        functional_group_local_granularity="per_check",
        functional_group_local_evidence_policy="shared_group_bank",
        deduction_multiplier=2.0,
        l3_only=False,
        max_cases=None,
        max_workers=max_workers,
        endpoint_preflight_attempts=1,
        endpoint_preflight_timeout_seconds=10,
        terminal_progress=False,
        resume=True,
        continue_on_error=False,
        export_audit_graphs=False,
        blender_bin=blender_bin,
        render_width=64,
        render_height=64,
        render_engine="BLENDER_EEVEE_NEXT",
        cycles_device="CPU",
        cycles_samples=1,
        cycles_denoising=False,
        preview_render_engine="BLENDER_EEVEE_NEXT",
        preview_width=64,
        preview_height=64,
        preview_cycles_samples=1,
        blender_timeout_seconds=10,
    )


def _main_fake_report() -> dict[str, Any]:
    usage = {
        "schema_version": "camera_cal_api_usage_v1",
        "api_calls_number": 0,
        "successful_api_calls": 0,
        "failed_api_calls": 0,
        "tokens_usage": None,
    }
    return {
        "schema_version": "camera_cal_scene_level_summary_v2",
        "status": "complete",
        "elapsed_seconds": 1.5,
        "totals": {
            "cases": 1,
            "successful": 1,
            "failed": 0,
            "cancelled": 0,
        },
        "api_usage": usage,
    }


def _install_main_fakes(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    output_root: Path,
    grouping_config: Path,
    blender_bin: Path,
    trace: list[tuple[str, str]],
    *,
    preflight_failure: bool = False,
    sequential_failure: bool = False,
    parallel_delegation: bool = False,
) -> None:
    args = _main_args(
        output_root,
        grouping_config,
        blender_bin,
        max_workers=2 if parallel_delegation else 1,
    )
    route = {
        "endpoint": "https://example.invalid/v1",
        "model": "fixture-model",
        "api_key_env": "FIXTURE_KEY",
        "authorization_configured": True,
    }
    cases = [
        {
            "case_id": "N001",
            "case_root": str(
                (runner.DEFAULT_DATASET_ROOT / "N001").resolve()
            ),
            "scene_type": "fixture",
            "object_count": 1,
        }
    ]
    stable_preflight = {
        "schema_version": "endpoint_stability_preflight_v1",
        "status": "complete",
        "attempts_required": 1,
        "api_invocations": 1,
    }
    stable_failure_report = {
        "schema_version": "endpoint_stability_preflight_v1",
        "status": "failed",
        "fatal_route_configuration": False,
        "completed_attempts": 0,
        "attempts_required": 1,
        "failures": [{"error": "fixture preflight failure"}],
    }

    class FakeProgress:
        def __init__(self, path: Path, *, terminal: bool = True) -> None:
            self.path = path.expanduser().resolve()
            self.terminal = bool(terminal)
            trace.append(("progress_init", self.path.name))

        def emit(
            self,
            event: str,
            *,
            case_id: str | None = None,
            **details: Any,
        ) -> dict[str, Any]:
            trace.append(("progress", str(event)))
            record = {
                "schema_version": "camera_cal_scene_level_progress_v1",
                "timestamp": "2026-08-22T00:00:00+00:00",
                "event": str(event),
                "case_id": str(case_id) if case_id else None,
                "details": deepcopy(details),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            return record

    def fake_run_case(*, case: dict[str, Any], **_: Any) -> dict[str, Any]:
        trace.append(("call", "run_case"))
        if sequential_failure:
            raise RuntimeError("fixture sequential failure")
        return {
            "case_id": str(case["case_id"]),
            "status": "complete",
            "elapsed_seconds": 1.5,
            "api_usage": _main_fake_report()["api_usage"],
        }

    def fake_failure(
        *,
        case: dict[str, Any],
        output_root: Path,
        error: Exception,
    ) -> dict[str, Any]:
        trace.append(("call", "record_case_failure"))
        return {
            "case_id": str(case["case_id"]),
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }

    def fake_parallel(**_: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        trace.append(("call", "run_cases_parallel"))
        record = {
            "case_id": "N001",
            "status": "complete",
            "elapsed_seconds": 1.5,
            "api_usage": _main_fake_report()["api_usage"],
        }
        return [record], []

    def fake_preflight(**_: Any) -> dict[str, Any]:
        trace.append(("call", "run_endpoint_stability_preflight"))
        if preflight_failure:
            raise module.EndpointStabilityPreflightError(stable_failure_report)
        return deepcopy(stable_preflight)

    def fake_plan(**_: Any) -> dict[str, Any]:
        trace.append(("call", "build_experiment_plan"))
        return {
            "schema_version": "camera_cal_scene_level_plan_v2",
            "cases": deepcopy(cases),
            "fixture": True,
        }

    def fake_summary(**_: Any) -> dict[str, Any]:
        trace.append(("call", "build_summary"))
        return deepcopy(_main_fake_report())

    def fake_control() -> Any:
        trace.append(("call", "resolved_control"))
        return SimpleNamespace(to_dict=lambda: {"fixture": True})

    def fake_renderer_config(*_: Any, **__: Any) -> dict[str, Any]:
        trace.append(("call", "renderer_config_from_args"))
        return {"blender_bin": str(blender_bin), "fixture": True}

    def fake_atomic_write(path: Path, value: Any) -> None:
        trace.append(("write", Path(path).name))
        original = getattr(module, "_e5_original_atomic_write", None)
        if original is None:
            original = module.atomic_write_json
        original(path, value)

    original_atomic = module.atomic_write_json
    monkeypatch.setattr(
        module,
        "_e5_original_atomic_write",
        original_atomic,
        raising=False,
    )
    monkeypatch.setattr(module, "atomic_write_json", fake_atomic_write)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "effective_model_route", lambda: deepcopy(route))
    monkeypatch.setattr(
        module,
        "normalize_metric_selection",
        lambda _: ("functional_consistency",),
    )
    monkeypatch.setattr(
        module,
        "discover_cases",
        lambda *_args, **_kwargs: deepcopy(cases),
    )
    monkeypatch.setattr(module, "renderer_config_from_args", fake_renderer_config)
    monkeypatch.setattr(module, "resolved_control", fake_control)
    monkeypatch.setattr(module, "build_experiment_plan", fake_plan)
    monkeypatch.setattr(
        module,
        "api_usage_summary",
        lambda _: deepcopy(_main_fake_report()["api_usage"]),
    )
    monkeypatch.setattr(module, "ProgressReporter", FakeProgress)
    monkeypatch.setattr(module, "_endpoint_preflight_image", lambda _: blender_bin)
    monkeypatch.setattr(module, "run_endpoint_stability_preflight", fake_preflight)
    monkeypatch.setattr(module, "record_case_failure", fake_failure)
    monkeypatch.setattr(module, "build_summary", fake_summary)
    monkeypatch.setattr(module, "run_case", fake_run_case)
    monkeypatch.setattr(module, "run_cases_parallel", fake_parallel)
    monkeypatch.setattr(module, "utc_now", lambda: "2026-08-22T00:00:00+00:00")

    monotonic_calls = 0

    def fixed_monotonic() -> float:
        nonlocal monotonic_calls
        value = 100.0 if monotonic_calls == 0 else 101.5
        monotonic_calls += 1
        return value

    monkeypatch.setattr(module.time, "monotonic", fixed_monotonic)


def _run_main(module: ModuleType) -> tuple[int, str, str]:
    try:
        result = module.main()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            code = 0
        if not isinstance(code, int):
            raise AssertionError(f"unexpected exit code: {code!r}")
        return code, "", ""
    return (0 if result is None else int(result)), "", ""


def _run_main_with_capture(module: ModuleType) -> tuple[int, bytes, bytes]:
    import contextlib
    import io

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code, _, _ = _run_main(module)
    return code, stdout.getvalue().encode(), stderr.getvalue().encode()


def _run_artifact_snapshot(output_root: Path) -> dict[str, tuple[bytes, bytes]]:
    snapshot: dict[str, tuple[bytes, bytes]] = {}
    for name in RUN_ARTIFACTS:
        path = output_root / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        snapshot[name] = (raw, _canonical_json(json.loads(raw)))
    progress = output_root / "progress.jsonl"
    raw_progress = progress.read_bytes()
    values = [
        json.loads(line)
        for line in raw_progress.decode("utf-8").splitlines()
        if line.strip()
    ]
    snapshot["progress.jsonl"] = (
        raw_progress,
        _canonical_json(values),
    )
    return snapshot


CASE_ARTIFACTS = (
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


def _case_run_kwargs(output_root: Path) -> dict[str, Any]:
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
        "functional_group_local_granularity": "per_check",
        "functional_group_local_evidence_policy": "shared_group_bank",
        "deduction_multiplier": 2.0,
        "export_audit_graphs": False,
        "l3_only": False,
        "progress": None,
        "model_route_abort_signal": None,
    }


def _package_case_run_kwargs(output_root: Path) -> dict[str, Any]:
    case = composition.discovery.discover_cases(
        composition.cli.DEFAULT_DATASET_ROOT,
        case_ids=["N001"],
    )[0]
    return {
        "case": case,
        "dataset_root": composition.cli.DEFAULT_DATASET_ROOT,
        "output_root": output_root,
        "grouping_config_path": composition.cli.DEFAULT_GROUPING_CONFIG,
        "route": {
            "endpoint": "https://example.invalid/v1",
            "model": "fixture-model",
            "api_key_env": "FIXTURE_KEY",
            "authorization_configured": True,
        },
        "metrics": ("functional_consistency",),
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
        "control_config": composition.planning.resolved_control().to_dict(),
        "resume": True,
        "functional_group_local_granularity": "per_check",
        "functional_group_local_evidence_policy": "shared_group_bank",
        "deduction_multiplier": 2.0,
        "export_audit_graphs": False,
        "l3_only": False,
        "progress": None,
        "model_route_abort_signal": None,
    }


def _case_fake_report() -> dict[str, Any]:
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


def _install_case_fakes(
    monkeypatch: pytest.MonkeyPatch,
    trace: list[tuple[str, str]],
) -> None:
    class FakeRenderer:
        def __init__(self, **_: Any) -> None:
            trace.append(("constructor", "BlenderRenderer"))

    class FakeProvider:
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

    def fake_grouping(route: Any) -> Any:
        trace.append(("constructor", "build_grouping_model"))
        return SimpleNamespace(route=route)

    def fake_geometry(_: Any) -> dict[str, Any]:
        trace.append(("constructor", "load_collision_geometry_manifest"))
        return {"schema_version": "collision_geometry_v1"}

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        trace.append(("call", "run_evaluate"))
        report = _case_fake_report()
        runner.atomic_write_json(Path(kwargs["out"]), report)
        return deepcopy(report)

    monkeypatch.setattr(runner, "BlenderRenderer", FakeRenderer)
    monkeypatch.setattr(runner, "CameraEvidenceProvider", FakeProvider)
    monkeypatch.setattr(
        runner,
        "CameraViewEvidenceRenderer",
        FakeEvidenceRenderer,
    )
    monkeypatch.setattr(
        runner,
        "CameraCandidatePreviewRenderer",
        FakePreviewRenderer,
    )
    monkeypatch.setattr(
        runner,
        "DeterministicLocalCameraSelector",
        fake_selector,
    )
    monkeypatch.setattr(
        runner,
        "build_openai_compatible_vlm_judge",
        fake_judge,
    )
    monkeypatch.setattr(
        runner,
        "build_openai_compatible_camera_selector",
        fake_camera_selector,
    )
    monkeypatch.setattr(runner, "build_grouping_model", fake_grouping)
    monkeypatch.setattr(
        runner,
        "load_collision_geometry_manifest",
        fake_geometry,
    )
    monkeypatch.setattr(runner, "run_evaluate", fake_evaluate)

    original_atomic_write = runner.atomic_write_json

    def traced_atomic_write(path: Path, value: Any) -> None:
        trace.append(("write", Path(path).name))
        original_atomic_write(path, value)

    monkeypatch.setattr(runner, "atomic_write_json", traced_atomic_write)
    monkeypatch.setattr(
        runner,
        "utc_now",
        lambda: "2026-08-22T00:00:00+00:00",
    )
    monotonic_calls = 0

    def fixed_monotonic() -> float:
        nonlocal monotonic_calls
        value = 100.0 if monotonic_calls == 0 else 101.5
        monotonic_calls += 1
        return value

    monkeypatch.setattr(runner.time, "monotonic", fixed_monotonic)


def _install_composition_case_fakes(
    monkeypatch: pytest.MonkeyPatch,
    trace: list[tuple[str, str]],
) -> None:
    """Patch package composition globals, never runner dependency builders."""

    class FakeRenderer:
        def __init__(self, **_: Any) -> None:
            trace.append(("constructor", "BlenderRenderer"))

    class FakeProvider:
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

    def fake_grouping(route: Any) -> Any:
        trace.append(("constructor", "build_grouping_model"))
        return SimpleNamespace(route=route)

    def fake_model_config(route: Any, *, role: str) -> dict[str, Any]:
        return {"route": route, "role": role}

    def fake_geometry(_: Any) -> dict[str, Any]:
        trace.append(("constructor", "load_collision_geometry_manifest"))
        return {"schema_version": "collision_geometry_v1"}

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        trace.append(("call", "run_evaluate"))
        report = _case_fake_report()
        composition.io.atomic_write_json(Path(kwargs["out"]), report)
        return deepcopy(report)

    monkeypatch.setattr(composition, "BlenderRenderer", FakeRenderer)
    monkeypatch.setattr(composition, "CameraEvidenceProvider", FakeProvider)
    monkeypatch.setattr(
        composition,
        "CameraViewEvidenceRenderer",
        FakeEvidenceRenderer,
    )
    monkeypatch.setattr(
        composition,
        "CameraCandidatePreviewRenderer",
        FakePreviewRenderer,
    )
    monkeypatch.setattr(
        composition,
        "DeterministicLocalCameraSelector",
        fake_selector,
    )
    monkeypatch.setattr(
        composition,
        "build_openai_compatible_vlm_judge",
        fake_judge,
    )
    monkeypatch.setattr(
        composition,
        "build_openai_compatible_camera_selector",
        fake_camera_selector,
    )
    monkeypatch.setattr(
        composition.planning,
        "model_config",
        fake_model_config,
    )
    monkeypatch.setattr(
        composition.planning,
        "build_grouping_model",
        fake_grouping,
    )
    monkeypatch.setattr(
        composition,
        "load_collision_geometry_manifest",
        fake_geometry,
    )
    monkeypatch.setattr(composition, "run_evaluate", fake_evaluate)

    original_io_atomic = composition.io.atomic_write_json

    def traced_atomic_write(path: Path, value: Any) -> None:
        trace.append(("write", Path(path).name))
        original_io_atomic(path, value)

    monkeypatch.setattr(composition.io, "atomic_write_json", traced_atomic_write)
    monkeypatch.setattr(
        composition.telemetry,
        "atomic_write_json",
        traced_atomic_write,
    )

    # The package tracker captures its IO defaults when the class is defined;
    # replace the imported tracker class with a thin test adapter so its
    # initial api_usage write uses the same traced package IO as later writes.
    package_tracker = composition.telemetry.APICallTracker

    class TracedAPICallTracker(package_tracker):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("write_json", composition.io.atomic_write_json)
            kwargs.setdefault("clock", composition.telemetry.utc_now)
            kwargs.setdefault("monotonic", composition.time.monotonic)
            super().__init__(**kwargs)

    monkeypatch.setattr(
        composition.telemetry,
        "APICallTracker",
        TracedAPICallTracker,
    )
    fixed_timestamp = "2026-08-22T00:00:00+00:00"
    monkeypatch.setattr(composition.io, "utc_now", lambda: fixed_timestamp)
    monkeypatch.setattr(
        composition.progress,
        "utc_now",
        lambda: fixed_timestamp,
    )
    monkeypatch.setattr(
        composition.telemetry,
        "utc_now",
        lambda: fixed_timestamp,
    )

    original_dependencies = composition.case_runtime_dependencies

    def traced_dependencies() -> Any:
        trace.append(("call", "composition.case_runtime_dependencies"))
        return original_dependencies()

    monkeypatch.setattr(
        composition,
        "case_runtime_dependencies",
        traced_dependencies,
    )
    monotonic_calls = 0

    def fixed_monotonic() -> float:
        nonlocal monotonic_calls
        value = 100.0 if monotonic_calls == 0 else 101.5
        monotonic_calls += 1
        return value

    monkeypatch.setattr(composition.time, "monotonic", fixed_monotonic)


def _case_artifact_snapshot(output_root: Path) -> dict[str, tuple[bytes, bytes]]:
    case_root = output_root / "cases" / "N001"
    snapshot: dict[str, tuple[bytes, bytes]] = {}
    for name in CASE_ARTIFACTS:
        path = case_root / name
        raw = path.read_bytes()
        snapshot[name] = (raw, _canonical_json(json.loads(raw)))
    progress_path = output_root / "progress.jsonl"
    raw_progress = progress_path.read_bytes()
    progress_values = [
        json.loads(line)
        for line in raw_progress.decode("utf-8").splitlines()
        if line.strip()
    ]
    snapshot["progress.jsonl"] = (
        raw_progress,
        _canonical_json(progress_values),
    )
    return snapshot


def test_e5_runtime_adapters_is_the_canonical_package_composition() -> None:
    assert case_runtime.RuntimeAdapters is adapters.AdapterBundle
    assert case_runtime.RuntimeAdapters.__module__ == adapters.__name__
    assert "scripts" not in _module_imports(
        Path(adapters.__file__)
    )
    assert "scripts" not in _module_imports(
        Path(case_runtime.__file__)
    )


def test_e5_package_case_runtime_matches_runner_facade_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare facade wiring with direct package composition on one case."""

    output_root = tmp_path / "same-case-output"
    with monkeypatch.context() as facade_patch:
        facade_kwargs = _case_run_kwargs(output_root)
        facade_trace: list[tuple[str, str]] = []
        _install_case_fakes(facade_patch, facade_trace)
        facade_result = runner.run_case(**facade_kwargs)
        facade_snapshot = _case_artifact_snapshot(output_root)

    assert output_root.is_dir()
    assert not output_root.is_symlink()
    shutil.rmtree(output_root)
    assert not output_root.exists()

    with monkeypatch.context() as package_patch:
        package_kwargs = _package_case_run_kwargs(output_root)
        package_trace: list[tuple[str, str]] = []
        _install_composition_case_fakes(package_patch, package_trace)
        package_result = composition.run_case(**package_kwargs)
        package_snapshot = _case_artifact_snapshot(output_root)

    assert _canonical_json(facade_result) == _canonical_json(package_result)
    assert package_trace[0] == (
        "call",
        "composition.case_runtime_dependencies",
    )
    assert package_trace[1:] == facade_trace
    assert set(facade_snapshot) == set(package_snapshot)
    for name in facade_snapshot:
        facade_raw, facade_canonical = facade_snapshot[name]
        package_raw, package_canonical = package_snapshot[name]
        assert facade_canonical == package_canonical, name
        assert facade_raw == package_raw, name


def test_e5_runner_public_surface_schema_and_signatures_remain_e0_exact() -> None:
    contract = _contract()
    historical = _historical_runner()
    runner_path = Path(runner.__file__)
    assert _public_top_level_names(runner_path) == contract["public_top_level"]

    actual_versions = {
        name: getattr(runner, name)
        for name in dir(runner)
        if name.endswith("_VERSION")
    }
    assert actual_versions == contract["schema_versions"]

    for name in contract["public_top_level"]["functions"]:
        assert inspect.signature(getattr(runner, name)) == inspect.signature(
            getattr(historical, name)
        ), name

    old_help = subprocess.run(
        [sys.executable, contract["runner"]["path"], "--help"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert old_help.returncode == contract["cli"]["help_exit_code"]
    assert all(
        token.encode() in old_help.stdout
        for token in contract["cli"]["required_help_tokens"]
    )


def test_e5_main_is_a_dynamic_orchestrator_facade() -> None:
    main = _main_ast()
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    call_names = [_dotted_name(node.func) for node in calls]
    assert any(
        "orchestrator" in name or name.endswith("run_main")
        for name in call_names
    ), "runner.main must delegate to the package orchestrator"
    assert any(
        any(
            keyword.arg in {"deps", "dependencies", "runtime"}
            for keyword in node.keywords
        )
        for node in calls
        if "orchestrator" in _dotted_name(node.func)
        or _dotted_name(node.func).endswith("run_main")
    ), "orchestrator delegation must receive an explicit dependency object"
    forbidden_direct_calls = {
        "run_case",
        "run_cases_parallel",
        "run_evaluate",
        "run_endpoint_stability_preflight",
        "build_summary",
    }
    assert not forbidden_direct_calls.intersection(
        name.split(".")[-1] for name in call_names
    )
    dependency_calls = [
        name
        for name in call_names
        if name.split(".")[-1].endswith("dependencies")
    ]
    assert dependency_calls, "main must construct dependencies on every call"


def test_e5_orchestrator_and_package_main_are_checkout_independent() -> None:
    orchestrator_path = PACKAGE_ROOT / "orchestrator.py"
    package_main_path = PACKAGE_ROOT / "__main__.py"
    assert orchestrator_path.is_file()
    assert package_main_path.is_file()

    for path in (orchestrator_path, package_main_path):
        imports = _module_imports(path)
        assert not any(
            name == "scripts" or name.startswith("scripts.")
            for name in imports
        ), path
        assert not any(
            name == "benchmark.evaluation_campaign"
            or name.startswith("benchmark.evaluation_campaign.")
            for name in imports
        ), path
        source = path.read_text(encoding="utf-8")
        assert "scripts/run_camera_cal_scene_level.py" not in source

    package_main_source = package_main_path.read_text(encoding="utf-8")
    assert (
        "benchmark.camera_cal_scene_level.orchestrator" in package_main_source
        or "benchmark.camera_cal_scene_level.cli" in package_main_source
    )
    assert "__name__" in package_main_source

    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        imports = _module_imports(path)
        assert not any(
            name == "scripts" or name.startswith("scripts.")
            for name in imports
        ), path


def test_e5_source_checkout_package_help_is_byte_identical_to_script_help() -> None:
    environment = dict(os.environ)
    source_path = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT), environment.get("PYTHONPATH", ""))
    )
    environment["PYTHONPATH"] = source_path
    old_help = subprocess.run(
        [sys.executable, "scripts/run_camera_cal_scene_level.py", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )
    package_help = subprocess.run(
        [sys.executable, "-m", "benchmark.camera_cal_scene_level", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )
    assert package_help.returncode == old_help.returncode == 0
    assert package_help.stdout == old_help.stdout
    assert package_help.stderr == old_help.stderr


def test_e5_package_composition_has_no_repository_script_dependency() -> None:
    packaging = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '[tool.setuptools.packages.find]' in packaging
    assert 'where = ["src"]' in packaging
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        imports = _module_imports(path)
        assert not any(
            name == "scripts" or name.startswith("scripts.")
            for name in imports
        ), path


@pytest.mark.parametrize(
    ("scenario", "preflight_failure", "sequential_failure", "parallel"),
    (
        ("preflight_success_sequential_failure", False, True, False),
        ("preflight_failure", True, False, False),
        ("parallel_delegation", False, False, True),
    ),
)
def test_e5_fixed_clock_main_matches_e0_main_for_terminal_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    preflight_failure: bool,
    sequential_failure: bool,
    parallel: bool,
) -> None:
    """Compare the final run-level contract before and after orchestration.

    The fake runtime deliberately patches each module's globals independently.
    No fields are removed during comparison; canonical JSON only sorts keys so
    semantic payloads remain fully visible.  The current facade is expected to
    fail this gate until ``orchestrator.py`` and package ``__main__.py`` land.
    """

    grouping_config = tmp_path / f"{scenario}-grouping.yaml"
    grouping_config.write_text("fixture: true\n", encoding="utf-8")
    blender_bin = tmp_path / f"{scenario}-blender"
    blender_bin.write_bytes(b"fixture blender")

    output_root = tmp_path / f"{scenario}-output"
    historical = _historical_runner()

    with monkeypatch.context() as current_patch:
        current_trace: list[tuple[str, str]] = []
        _install_main_fakes(
            current_patch,
            runner,
            output_root,
            grouping_config,
            blender_bin,
            current_trace,
            preflight_failure=preflight_failure,
            sequential_failure=sequential_failure,
            parallel_delegation=parallel,
        )
        current_code, current_stdout, current_stderr = _run_main_with_capture(
            runner
        )
    current_snapshot = _run_artifact_snapshot(output_root)

    assert output_root.is_dir()
    assert not output_root.is_symlink()
    shutil.rmtree(output_root)
    assert not output_root.exists()

    with monkeypatch.context() as historical_patch:
        historical_trace: list[tuple[str, str]] = []
        _install_main_fakes(
            historical_patch,
            historical,
            output_root,
            grouping_config,
            blender_bin,
            historical_trace,
            preflight_failure=preflight_failure,
            sequential_failure=sequential_failure,
            parallel_delegation=parallel,
        )
        historical_code, historical_stdout, historical_stderr = (
            _run_main_with_capture(historical)
        )
    historical_snapshot = _run_artifact_snapshot(output_root)

    assert current_code == historical_code, scenario
    assert current_stdout == historical_stdout, scenario
    assert current_stderr == historical_stderr, scenario
    assert current_trace == historical_trace, scenario
    assert set(current_snapshot) == set(historical_snapshot), scenario
    for name in current_snapshot:
        current_raw, current_canonical = current_snapshot[name]
        historical_raw, historical_canonical = historical_snapshot[name]
        assert current_canonical == historical_canonical, (scenario, name)
        assert current_raw == historical_raw, (scenario, name)

    if parallel:
        assert ("call", "run_cases_parallel") in current_trace
        assert ("call", "run_cases_parallel") in historical_trace
    else:
        assert ("call", "run_cases_parallel") not in current_trace
        assert ("call", "run_cases_parallel") not in historical_trace
    if preflight_failure:
        assert current_code == historical_code == 2
        assert ("call", "run_endpoint_stability_preflight") in current_trace
    elif sequential_failure:
        assert current_code == historical_code == 1
        assert ("call", "run_case") in current_trace
