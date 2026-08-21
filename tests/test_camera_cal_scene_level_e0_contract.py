from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_camera_cal_scene_level as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "tests" / "fixtures" / (
    "camera_cal_scene_level_e0_contract.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _public_top_level_names(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes: list[str] = []
    functions: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                functions.append(node.name)
    return {"classes": classes, "functions": functions}


@pytest.mark.requires_git_history
def test_runner_blob_public_surface_and_schema_versions_are_pinned() -> None:
    contract = _contract()
    runner_path = PROJECT_ROOT / contract["runner"]["path"]

    baseline_ref = (
        f"{contract['runner']['git_commit_sha']}:{contract['runner']['path']}"
    )
    baseline_bytes = subprocess.check_output(
        ["git", "show", baseline_ref], cwd=PROJECT_ROOT
    )
    assert hashlib.sha256(baseline_bytes).hexdigest() == contract["runner"][
        "sha256"
    ]
    blob = subprocess.check_output(
        ["git", "rev-parse", baseline_ref],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    assert blob == contract["runner"]["git_blob_sha"]
    assert subprocess.call(
        ["git", "cat-file", "-e", f"{contract['runner']['git_commit_sha']}^{{commit}}"],
        cwd=PROJECT_ROOT,
    ) == 0

    assert _public_top_level_names(runner_path) == contract["public_top_level"]

    actual_versions = {
        name: getattr(runner, name)
        for name in dir(runner)
        if name.endswith("_VERSION")
    }
    assert actual_versions == contract["schema_versions"]


def test_declared_frozen_sources_and_help_contract_are_offline_safe() -> None:
    contract = _contract()
    source_hashes = contract["source_sha256"]
    frozen_paths = {
        path
        for category in source_hashes.values()
        for path in category
    }
    expected_paths = {
        "src/benchmark/visual_judge/l3_prompts.py",
        "src/benchmark/evaluator/scene_quality/prompt_context.py",
        "src/benchmark/evaluator/scoring.py",
        "src/benchmark/scoring_profiles.py",
        *runner.FUNCTIONAL_PROBE_IMPLEMENTATION_FILES,
    }
    assert frozen_paths == expected_paths
    for category in source_hashes.values():
        for relative_path, expected_hash in category.items():
            path = PROJECT_ROOT / relative_path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    result = subprocess.run(
        [sys.executable, contract["runner"]["path"], "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == contract["cli"]["help_exit_code"]
    help_text = result.stdout
    assert all(token in help_text for token in contract["cli"]["required_help_tokens"])
