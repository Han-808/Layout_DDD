"""F2 parity gates for persisted scoring projection extraction.

The historical implementation is loaded from the pinned pre-F2 commit rather
than imported from the mutable working tree.  This permanently preserves the
viewer implementation as the extraction oracle.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.requires_git_history


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_REF = "73cabb8"
VIEWER_PATH = ROOT / "scripts" / "build_vlm_evidence_viewer.py"
SELECTOR_PATH = ROOT / "scripts" / "select_first_publishable_scene_evaluations.py"
PACKAGE_MODULE = "benchmark.camera_cal_scene_level.persisted_scoring"


def _historical_viewer() -> ModuleType:
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{HISTORICAL_REF}:scripts/build_vlm_evidence_viewer.py",
        ],
        cwd=ROOT,
    )
    module = ModuleType("persisted_scoring_historical_viewer")
    module.__file__ = str(VIEWER_PATH)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _current_viewer() -> ModuleType:
    module_name = "persisted_scoring_current_viewer"
    spec = importlib.util.spec_from_file_location(module_name, VIEWER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package() -> ModuleType:
    return importlib.import_module(PACKAGE_MODULE)


def _raw_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


SCORING_METRICS = (
    ("L1", "collision", "Collision"),
    ("L1", "support", "Support"),
    ("L1", "oob", "Out of bounds"),
    ("L3", "scale_consistency", "Scale"),
    ("L3", "style_consistency", "Style"),
    ("L3", "object_pairing_consistency", "Object pairing"),
    ("L3", "functional_consistency", "Function"),
    ("L3", "semantic_placement_consistency", "Placement"),
)


def _case_fixture(
    *,
    coverage: float = 1.0,
    missing_score: str | None = None,
    l1_failure: bool = False,
    placement_components: bool = False,
    case_id: str = "S100",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    l1_metrics: dict[str, Any] = {}
    l3_metrics: dict[str, Any] = {}
    missing_score = missing_score or ("collision" if l1_failure else None)
    l1_weights = {metric: 1.0 / 3.0 for _, metric, _ in SCORING_METRICS[:3]}
    l3_weights = {metric: 0.2 for _, metric, _ in SCORING_METRICS[3:]}
    l3_weights["functional_consistency"] = 0.4

    for layer, metric, _ in SCORING_METRICS:
        score: float | None = 0.9 if layer == "L1" else 0.8
        if metric == missing_score:
            score = None
        metric_scoring: dict[str, Any] = {
            "event_count": 0,
            "events": [],
            "nominal_metric_weight": (
                l1_weights[metric] if layer == "L1" else l3_weights[metric]
            ),
            "coefficient_n_m": 2.0,
            "burden_total_b_m": 0.1,
            "p_max": 0.2,
            "metric_deduction": 0.1,
        }
        if metric == "oob":
            metric_scoring["events"] = [
                {
                    "category": "room_boundary_crossing",
                    "severity": "major",
                    "burden": 0.2,
                    "scoring_target_ids": ["chair"],
                }
            ]
            metric_scoring["event_count"] = 1
        if placement_components and metric == "semantic_placement_consistency":
            metric_scoring["placement_component_weights"] = {
                "scene_zone": 0.75,
                "contextual_anchor": 0.25,
            }
            metric_scoring["placement_components"] = {
                "scene_zone": {
                    "score": 0.7,
                    "metric_deduction": 0.3,
                    "event_count": 2,
                },
                "contextual_anchor": {
                    "score": 0.9,
                    "metric_deduction": 0.1,
                    "event_count": 1,
                },
            }
        metric_report: dict[str, Any] = {
            "status": "failed" if l1_failure and layer == "L1" else "evaluated",
            "score": score,
            "judgement": {"verdict": "valid", "reason": "fixture"},
            "coverage": {
                "score_grounding": {
                    "fraction": coverage,
                    "complete": coverage >= 1.0,
                }
            },
            "scoring": metric_scoring,
        }
        (l1_metrics if layer == "L1" else l3_metrics)[metric] = metric_report

    manifest = {
        "case_id": case_id,
        "final_decision_status": "unresolved" if l1_failure else "resolved",
        "scoring_profile": {
            "scoring_profile_id": "intrinsic_validity_v1",
            "scoring_spec_version": "object_equivalent_burden_v3",
            "deduction_multiplier": 2.0,
            "layer_weights": {
                "l1_physical_plausibility": 0.3,
                "l3_scene_quality": 0.7,
            },
        },
        "canonical_object_denominator": {
            "n_scene": 3,
            "ordered_object_ids": ["chair", "table", "lamp"],
        },
        "scoring_reliability": {
            "terminal_state": "infrastructure_failure" if l1_failure else "complete",
            "judge_episode_count": 3,
            "forced_binary_episode_count": 1 if l1_failure else 0,
            "evidence_ambiguous_episode_count": 0,
            "infrastructure_failures": (
                [{"metric": "collision", "error": "fixture failure"}]
                if l1_failure
                else []
            ),
        },
    }
    l1_report = {
        "status": "failed" if l1_failure else "evaluated",
        "coverage": {"complete": coverage >= 1.0},
        "metrics": l1_metrics,
        "scoring": {"metric_weights": l1_weights},
    }
    l3_report = {
        "status": "evaluated",
        "coverage": {"complete": coverage >= 1.0},
        "metrics": l3_metrics,
        "scoring": {"metric_weights": l3_weights},
    }
    diagnostics = {
        "engineering_failures": (
            [
                {
                    "metric": "collision",
                    "route": "vlm_adjudication_failed",
                    "error": "fixture failure",
                }
            ]
            if l1_failure
            else []
        )
    }
    return manifest, l1_report, l3_report, diagnostics


def _summary_pair(
    historical: ModuleType,
    package: ModuleType,
    fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, l1_report, l3_report, diagnostics = fixture
    kwargs = {
        "case_id": str(manifest["case_id"]),
        "case_manifest": deepcopy(manifest),
        "l1_report": deepcopy(l1_report),
        "l3_report": deepcopy(l3_report),
        "l1_diagnostics": deepcopy(diagnostics),
    }
    old = historical.case_scoring_summary(**deepcopy(kwargs))
    new = package.case_scoring_summary(**deepcopy(kwargs))
    return old, new


@pytest.mark.parametrize(
    ("label", "coverage", "missing_score", "l1_failure", "placement"),
    (
        ("complete", 1.0, None, False, False),
        ("partial", 0.9, None, False, False),
        ("below_threshold", 0.5, None, False, False),
        ("missing_score", 1.0, "functional_consistency", False, False),
        ("l1_failure", 1.0, None, True, False),
        ("placement_components", 1.0, None, False, True),
    ),
)
def test_f2_case_summary_matches_historical_blob_byte_for_byte(
    label: str,
    coverage: float,
    missing_score: str | None,
    l1_failure: bool,
    placement: bool,
) -> None:
    del label
    historical = _historical_viewer()
    package = _package()
    old, new = _summary_pair(
        historical,
        package,
        _case_fixture(
            coverage=coverage,
            missing_score=missing_score,
            l1_failure=l1_failure,
            placement_components=placement,
        ),
    )
    assert list(new) == list(old)
    assert _raw_json(new) == _raw_json(old)
    assert _canonical_json(new) == _canonical_json(old)
    assert new == old

    if placement:
        metric = next(
            row for row in new["metrics"]
            if row["metric"] == "semantic_placement_consistency"
        )
        assert metric["placement_component_weights"] == {
            "scene_zone": 0.75,
            "contextual_anchor": 0.25,
        }
        assert list(metric["placement_components"]) == [
            "scene_zone",
            "contextual_anchor",
        ]
    if l1_failure:
        assert new["engineering_failure_record_count"] == 1
        assert new["engineering_failures"]


def _aggregate_pair(
    historical: ModuleType,
    package: ModuleType,
    fixtures: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    ],
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_summaries: list[dict[str, Any]] = []
    new_summaries: list[dict[str, Any]] = []
    for fixture in fixtures:
        old, new = _summary_pair(historical, package, fixture)
        old_summaries.append(old)
        new_summaries.append(new)
    return (
        historical.run_scoring_aggregate(old_summaries),
        package.run_scoring_aggregate(new_summaries),
    )


@pytest.mark.parametrize(
    ("label", "fixtures"),
    (
        (
            "official_none",
            [
                _case_fixture(case_id="S100"),
                _case_fixture(coverage=0.5, case_id="S101"),
            ],
        ),
        (
            "official_complete",
            [
                _case_fixture(case_id="S100"),
                _case_fixture(case_id="S101", placement_components=True),
            ],
        ),
    ),
)
def test_f2_aggregate_matches_historical_key_order_and_bytes(
    label: str,
    fixtures: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    ],
) -> None:
    historical = _historical_viewer()
    package = _package()
    old, new = _aggregate_pair(historical, package, fixtures)
    assert list(new) == list(old)
    assert _raw_json(new) == _raw_json(old)
    assert _canonical_json(new) == _canonical_json(old)
    assert new == old
    assert any(row["metric"] == "semantic_placement_consistency" for row in new["metrics"])
    if label == "official_none":
        assert new["official_score_100"] is None
    else:
        assert new["official_score_100"] is not None


def test_f2_current_viewer_reexports_package_identity_and_selector_uses_package() -> None:
    package = _package()
    viewer = _current_viewer()
    assert viewer.case_scoring_summary is package.case_scoring_summary
    assert viewer.run_scoring_aggregate is package.run_scoring_aggregate
    assert viewer.MIN_PUBLISHABLE_SCORE_COVERAGE == package.MIN_PUBLISHABLE_SCORE_COVERAGE
    assert viewer.SCORING_METRIC_ORDER is package.SCORING_METRIC_ORDER
    assert inspect.signature(viewer.case_scoring_summary) == inspect.signature(
        package.case_scoring_summary
    )
    assert inspect.signature(viewer.run_scoring_aggregate) == inspect.signature(
        package.run_scoring_aggregate
    )

    tree = ast.parse(SELECTOR_PATH.read_text(encoding="utf-8"))
    imports = {
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "benchmark.camera_cal_scene_level.persisted_scoring" in imports
    assert not any(
        name == "build_vlm_evidence_viewer"
        or name.endswith(".build_vlm_evidence_viewer")
        for name in imports
    )


def test_f2_package_is_pure_and_does_not_own_viewer_or_io() -> None:
    package_path = ROOT / "src/benchmark/camera_cal_scene_level/persisted_scoring.py"
    assert package_path.is_file()
    source = package_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(package_path))
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
    assert not any(
        name == "scripts"
        or name.startswith("scripts.")
        or name == "html"
        or name.startswith("html.")
        or name == "io"
        or name.startswith("io.")
        or name == "pathlib"
        or name.startswith("pathlib.")
        or "build_vlm_evidence_viewer" in name
        for name in imports
    )
    lowered = source.lower()
    assert "<html" not in lowered
    assert "write_text(" not in lowered
    assert "read_text(" not in lowered
    assert "atomic_write" not in lowered


def _manifest_fixture(root: Path) -> tuple[Path, Path, Path]:
    files = (
        "scripts/run_camera_cal_scene_level.py",
        "scripts/select_first_publishable_scene_evaluations.py",
        "scripts/check_model_endpoint.py",
        "scripts/build_vlm_evidence_viewer.py",
        "src/benchmark/evaluation_campaign/__init__.py",
        "src/benchmark/camera_cal_scene_level/__init__.py",
        "src/benchmark/camera_cal_scene_level/persisted_scoring.py",
        "src/benchmark/api/evaluation.py",
        "src/benchmark/evaluator/module.py",
        "src/benchmark/models/module.py",
        "src/benchmark/rendering/module.py",
        "src/benchmark/visual_judge/module.py",
        "src/benchmark/grouping/module.py",
        "src/benchmark/scoring_profiles.py",
        "pyproject.toml",
        "uv.lock",
    )
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    yaml_files = (
        "configs/evaluation/policy.yaml",
        "configs/grouping/policy.yaml",
        "src/benchmark/_resources/configs/evaluation/policy.yaml",
        "src/benchmark/_resources/configs/grouping/policy.yaml",
    )
    for relative in yaml_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {relative}\n", encoding="utf-8")
    return (
        root / "src/benchmark/camera_cal_scene_level/persisted_scoring.py",
        root / "scripts/build_vlm_evidence_viewer.py",
        root,
    )


def test_f2_campaign_manifest_covers_persisted_scoring_but_not_viewer_mutation(
    tmp_path: Path,
) -> None:
    provenance = importlib.import_module("benchmark.evaluation_campaign.provenance")
    persisted, viewer, repo = _manifest_fixture(tmp_path / "repo")
    baseline = provenance.evaluation_source_manifest(repo)
    baseline_paths = {str(row["path"]) for row in baseline["files"]}
    assert "src/benchmark/camera_cal_scene_level/persisted_scoring.py" in baseline_paths

    viewer.write_text("VIEWER_ONLY_MUTATION = True\n", encoding="utf-8")
    viewer_changed = provenance.evaluation_source_manifest(repo)
    assert viewer_changed["manifest_sha256"] == baseline["manifest_sha256"]

    persisted.write_text("RUNTIME_SCORING_MUTATION = True\n", encoding="utf-8")
    runtime_changed = provenance.evaluation_source_manifest(repo)
    assert runtime_changed["manifest_sha256"] != viewer_changed["manifest_sha256"]
