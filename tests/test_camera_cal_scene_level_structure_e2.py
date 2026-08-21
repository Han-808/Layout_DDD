from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace
from typing import Any

from benchmark.camera_cal_scene_level import discovery, planning, scheduling
from scripts import run_camera_cal_scene_level as runner


ROOT = Path(__file__).resolve().parents[1]
E0_CONTRACT = (
    ROOT / "tests/fixtures/camera_cal_scene_level_e0_contract.json"
)


def _historical_runner() -> ModuleType:
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
    module = ModuleType("camera_cal_scene_level_e0_runtime")
    module.__file__ = str(ROOT / contract["runner"]["path"])
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _ready_case(root: Path, case_id: str) -> None:
    case_root = root / case_id
    for relative in (
        "scene/canonical_scene.json",
        "prepared/evaluation.blend",
        "annotation.json",
        "evidence/standardized_perspective.png",
        "evidence/standardized_top.png",
        "evidence/standardized_identity_map.png",
        "evidence/collision_geometry_manifest.json",
    ):
        path = case_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
    (case_root / "case_manifest.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "status": "ready",
                "scene_type": "fixture",
                "object_count": 1,
                "paths": {},
            }
        ),
        encoding="utf-8",
    )


def test_e2_package_never_imports_the_compatibility_script() -> None:
    for module in (discovery, planning, scheduling):
        source = Path(module.__file__).read_text(encoding="utf-8")
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


def test_cli_and_discovery_match_the_frozen_e0_runner(tmp_path: Path) -> None:
    historical = _historical_runner()
    argv = [
        "--output-root",
        str(tmp_path / "output"),
        "--metric",
        "functional_consistency",
        "--max-workers",
        "2",
        "--no-terminal-progress",
    ]
    assert vars(runner.parse_args(argv)) == vars(historical.parse_args(argv))

    dataset = tmp_path / "dataset"
    _ready_case(dataset, "S100")
    _ready_case(dataset, "S101")
    assert runner.discover_cases(dataset) == historical.discover_cases(dataset)
    assert runner.discover_cases(dataset, case_ids=["S101"]) == (
        historical.discover_cases(dataset, case_ids=["S101"])
    )


def test_planning_matches_the_frozen_e0_runner(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    historical = _historical_runner()
    fixed_time = "2026-08-21T00:00:00+00:00"
    monkeypatch.setattr(runner, "utc_now", lambda: fixed_time)
    historical.utc_now = lambda: fixed_time
    grouping = tmp_path / "grouping.yaml"
    grouping.write_text("fixture: true\n", encoding="utf-8")
    route = {
        "endpoint": "https://example.invalid/v1",
        "model": "gpt-5.6-sol",
        "api_key_env": "PRIVATE_KEY",
        "authorization_configured": True,
    }
    renderer_args = SimpleNamespace(
        blender_timeout_seconds=900,
        render_width=768,
        render_height=768,
        render_engine="BLENDER_EEVEE_NEXT",
        cycles_device="CPU",
        cycles_samples=16,
        cycles_denoising=False,
        preview_render_engine="BLENDER_EEVEE_NEXT",
        preview_width=256,
        preview_height=256,
        preview_cycles_samples=1,
    )
    blender = tmp_path / "blender"
    current_renderer = runner.renderer_config_from_args(
        renderer_args, blender_bin=blender
    )
    historical_renderer = historical.renderer_config_from_args(
        renderer_args, blender_bin=blender
    )
    assert current_renderer == historical_renderer
    current_control = runner.resolved_control().to_dict()
    historical_control = historical.resolved_control().to_dict()
    assert current_control == historical_control
    kwargs: dict[str, Any] = {
        "dataset_root": tmp_path / "dataset",
        "output_root": tmp_path / "output",
        "grouping_config_path": grouping,
        "route": route,
        "metrics": ("functional_consistency",),
        "functional_group_local_granularity": "per_check",
        "functional_group_local_evidence_policy": "shared_group_bank",
        "deduction_multiplier": 2.0,
        "cases": [{"case_id": "S100"}],
        "renderer_config": current_renderer,
        "control": current_control,
        "max_workers": 2,
        "endpoint_preflight_attempts": 10,
        "endpoint_preflight_timeout_seconds": 300,
        "resume": True,
        "continue_on_error": False,
        "export_audit_graphs": False,
        "l3_only": False,
    }
    assert runner.build_experiment_plan(**kwargs) == (
        historical.build_experiment_plan(**kwargs)
    )


def test_resume_and_scheduler_facades_use_extracted_runtime(tmp_path: Path) -> None:
    assert issubclass(
        runner.ModelRouteAbortSignal,
        scheduling.ModelRouteAbortSignal,
    )
    case_out = tmp_path / "case"
    for name in (
        "evaluation_report.json",
        "grouping.json",
        "l1_report.json",
        "l1_diagnostics.json",
        "scene_quality_report.json",
        "scene_comparison.json",
        "control_manifest.json",
    ):
        path = case_out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    manifest = {"status": "complete", "input_fingerprint": "fingerprint"}
    assert runner.resumable_case(
        manifest,
        expected_fingerprint="fingerprint",
        case_out=case_out,
    )
