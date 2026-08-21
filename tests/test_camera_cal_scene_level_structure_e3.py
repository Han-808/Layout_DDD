from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
from types import ModuleType

import pytest

from benchmark.camera_cal_scene_level import comparison, provenance, reports
from scripts import run_camera_cal_scene_level as runner


ROOT = Path(__file__).resolve().parents[1]
E0_CONTRACT = (
    ROOT / "tests/fixtures/camera_cal_scene_level_e0_contract.json"
)
pytestmark = pytest.mark.requires_git_history


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
    module = ModuleType("camera_cal_scene_level_e0_reports")
    module.__file__ = str(ROOT / contract["runner"]["path"])
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def test_e3_modules_do_not_import_the_compatibility_script() -> None:
    for module in (comparison, provenance, reports):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        modules.update(
            str(node.module or "")
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert not any(
            name == "scripts" or name.startswith("scripts.")
            for name in modules
        )


def test_comparison_and_summary_match_the_frozen_e0_runner() -> None:
    historical = _historical_runner()
    annotation = {
        "metrics": {
            "functional_consistency": {
                "anomaly": True,
                "unclear": False,
                "affected_object_ids": ["chair", "table", "chair"],
                "issue": "fixture",
            }
        }
    }
    quality = {
        "metrics": {
            "functional_consistency": {
                "status": "evaluated",
                "judgement": {"verdict": "invalid"},
                "final_defect_claims": [
                    {"target_ids": ["chair", "lamp"]}
                ],
            }
        }
    }
    kwargs = {
        "case_id": "S100",
        "annotation": annotation,
        "scene_quality_report": quality,
        "metrics": ("functional_consistency",),
    }
    assert runner.build_scene_comparison(**kwargs) == (
        historical.build_scene_comparison(**kwargs)
    )
    summary_kwargs = {
        "case_records": [],
        "metrics": ("functional_consistency",),
        "elapsed_seconds": 1.25,
    }
    assert runner.build_summary(**summary_kwargs) == (
        historical.build_summary(**summary_kwargs)
    )


def test_case_fingerprint_versions_the_extracted_runtime_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = _historical_runner()
    paths: dict[str, Path] = {}
    for name in (
        "scene",
        "annotation",
        "perspective",
        "top",
        "identity",
        "collision_geometry",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(f"{{\"name\": \"{name}\"}}\n", encoding="utf-8")
        paths[name] = path
    kwargs = {
        "case": {
            "case_id": "S100",
            "semantic_content_fingerprint": "semantic",
        },
        "case_manifest": {
            "critical_artifact_hashes": {"blend": "blend-hash"}
        },
        "paths": paths,
        "route": {
            "endpoint": "https://example.invalid/v1",
            "model": "gpt-5.6-sol",
            "api_key_env": "PRIVATE_KEY",
            "authorization_configured": True,
        },
        "metrics": ("functional_consistency",),
        "functional_group_local_granularity": "per_check",
        "functional_group_local_evidence_policy": "shared_group_bank",
        "deduction_multiplier": 2.0,
        "grouping_config": {"policy": "fixture"},
        "renderer_config": {
            "blender_bin": "/private/blender",
            "width": 768,
        },
        "control_config": {"policy": "fixture"},
        "l3_only": False,
    }
    current = runner.case_input_fingerprint(**kwargs)
    historical_value = historical.case_input_fingerprint(**kwargs)
    assert current != historical_value

    runtime_source = (
        ROOT / "src/benchmark/camera_cal_scene_level/case_runtime.py"
    ).resolve()
    original_file_sha256 = runner.file_sha256

    def changed_runtime_hash(path: Path) -> str:
        if Path(path).resolve() == runtime_source:
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(runner, "file_sha256", changed_runtime_hash)
    assert runner.case_input_fingerprint(**kwargs) != current


def test_route_and_resolution_facades_match_the_frozen_e0_runner() -> None:
    historical = _historical_runner()
    route = {
        "endpoint": "https://example.invalid/v1",
        "model": "gpt-5.6-sol",
        "api_key_env": "PRIVATE_KEY",
        "authorization_configured": True,
        "min_request_interval_seconds": 1,
        "secret": "must-not-appear",
    }
    assert runner.safe_route_manifest(route) == historical.safe_route_manifest(
        route
    )
    quality = {
        "metrics": {
            "functional_consistency": {
                "status": "evaluated",
                "score": 0.9,
                "coverage": {
                    "coverage_threshold_passed": True,
                    "eligible_count": 10,
                    "resolved_count": 9,
                },
            }
        }
    }
    assert runner.l3_resolution_audit(
        quality, metrics=("functional_consistency",)
    ) == historical.l3_resolution_audit(
        quality, metrics=("functional_consistency",)
    )
