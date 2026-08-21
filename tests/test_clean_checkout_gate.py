from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import check_clean_checkout as gate


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _minimal_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "clean-gate@example.invalid")
    _git(repo, "config", "user.name", "Clean Gate Test")
    (repo / "tracked.py").write_text("VALUE = 'tracked'\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo


def test_safe_environment_does_not_inherit_credentials(tmp_path: Path) -> None:
    source = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": "/sensitive/home",
        "OPENAI_API_KEY": "secret",
        "FORGEAX_API_KEY": "secret",
        "JUDGE_ENDPOINT": "https://internal.invalid",
    }

    result = gate._safe_environment(
        source,
        checkout=tmp_path / "checkout",
        temporary_home=tmp_path / "home",
    )

    assert result["HOME"] == str(tmp_path / "home")
    assert result["PIP_NO_INDEX"] == "1"
    assert result["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "checkout" / "src"),
        str(tmp_path / "checkout"),
    ]
    assert "OPENAI_API_KEY" not in result
    assert "FORGEAX_API_KEY" not in result
    assert "JUDGE_ENDPOINT" not in result


def test_gate_verifies_commit_without_untracked_dependency(tmp_path: Path) -> None:
    repo = _minimal_repo(tmp_path)
    (repo / "local_only.py").write_text("VALUE = 'untracked'\n", encoding="utf-8")
    code = (
        "from pathlib import Path; import subprocess; "
        "assert Path('tracked.py').is_file(); "
        "assert not Path('local_only.py').exists(); "
        "assert subprocess.run(['git','config','--get','remote.origin.url'], "
        "capture_output=True).returncode != 0"
    )

    report = gate.verify_clean_checkout(
        repo_root=repo,
        source_ref="HEAD",
        python=sys.executable,
        skip_collection=True,
        skip_wheel=True,
        steps=(("boundary", (sys.executable, "-c", code)),),
    )

    assert report["status"] == "ok_ref_only"
    assert report["worktree_matches_verified_ref"] is False
    assert report["worktree_change_count"] == 1
    assert report["steps"][0]["ok"] is True
    assert report["source_origin_present"] is False
    assert report["network_isolation_enforced"] is False


def test_gate_reports_failure_from_clean_clone(tmp_path: Path) -> None:
    repo = _minimal_repo(tmp_path)
    (repo / "local_only.py").write_text("VALUE = 'untracked'\n", encoding="utf-8")

    report = gate.verify_clean_checkout(
        repo_root=repo,
        python=sys.executable,
        skip_collection=True,
        skip_wheel=True,
        steps=(
            (
                "must_not_see_local",
                (sys.executable, "-c", "import local_only"),
            ),
        ),
    )

    assert report["status"] == "failed"
    assert report["steps"][0]["ok"] is False
    assert "ModuleNotFoundError" in report["steps"][0]["stderr_tail"]


def test_require_worktree_match_rejects_dirty_source(tmp_path: Path) -> None:
    repo = _minimal_repo(tmp_path)
    (repo / "local_only.py").write_text("VALUE = 'untracked'\n", encoding="utf-8")

    with pytest.raises(gate.CleanCheckoutError, match="not present"):
        gate.verify_clean_checkout(
            repo_root=repo,
            skip_collection=True,
            skip_wheel=True,
            require_worktree_match=True,
            steps=(),
        )


def test_run_step_bounds_diagnostic_output(tmp_path: Path) -> None:
    result = gate._run_step(
        "large_failure",
        (
            sys.executable,
            "-c",
            "import sys; print('x' * 5000); print('y' * 5000, file=sys.stderr); sys.exit(3)",
        ),
        cwd=tmp_path,
        environ=gate._safe_environment(temporary_home=tmp_path),
    )

    assert result.exit_code == 3
    assert len(result.stdout_tail) <= 4003
    assert len(result.stderr_tail) <= 4003
    assert result.stdout_tail.startswith("...")
    assert result.stderr_tail.startswith("...")


def test_run_step_times_out_without_hanging_gate(tmp_path: Path) -> None:
    result = gate._run_step(
        "timeout",
        (sys.executable, "-c", "import time; time.sleep(5)"),
        cwd=tmp_path,
        environ=gate._safe_environment(temporary_home=tmp_path),
        timeout_seconds=0.05,
    )

    assert result.exit_code == 124
    assert result.timed_out is True


def test_gate_rejects_gitlink_before_archive_materialization(tmp_path: Path) -> None:
    repo = _minimal_repo(tmp_path)
    target = _git(repo, "rev-parse", "HEAD")
    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{target},vendor",
    )
    _git(repo, "commit", "--quiet", "-m", "add synthetic gitlink")

    with pytest.raises(gate.CleanCheckoutError, match="unsupported git tree entry"):
        gate.verify_clean_checkout(
            repo_root=repo,
            skip_collection=True,
            skip_canonical_tests=True,
            skip_wheel=True,
            steps=(),
        )


def test_gate_rejects_git_archive_content_substitution(tmp_path: Path) -> None:
    repo = _minimal_repo(tmp_path)
    (repo / ".gitattributes").write_text(
        "tracked.py export-subst\n", encoding="utf-8"
    )
    (repo / "tracked.py").write_text(
        "COMMIT = '$Format:%H$'\n", encoding="utf-8"
    )
    _git(repo, "add", ".gitattributes", "tracked.py")
    _git(repo, "commit", "--quiet", "-m", "add archive substitution")

    with pytest.raises(gate.CleanCheckoutError, match="entry_mismatches"):
        gate.verify_clean_checkout(
            repo_root=repo,
            skip_collection=True,
            skip_canonical_tests=True,
            skip_wheel=True,
            steps=(),
        )


def test_default_lifecycle_step_receives_parent_registry(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")
    steps = gate._default_steps(
        python=sys.executable,
        skip_collection=True,
        skip_canonical_tests=True,
        baseline_registry=baseline,
    )

    assert steps[0] == (
        "component_lifecycle",
        (
            sys.executable,
            "scripts/check_component_lifecycle.py",
            "--baseline-registry",
            str(baseline),
        ),
    )


def test_commit_blob_bytes_reads_exact_parent_registry(tmp_path: Path) -> None:
    repo = _minimal_repo(tmp_path)
    registry_path = repo / gate.LIFECYCLE_REGISTRY_PATH
    registry_path.parent.mkdir(parents=True)
    expected = b'{"schema_version":"component_lifecycle_v1"}\n'
    registry_path.write_bytes(expected)
    _git(repo, "add", gate.LIFECYCLE_REGISTRY_PATH)
    _git(repo, "commit", "--quiet", "-m", "add lifecycle registry")
    commit = _git(repo, "rev-parse", "HEAD")

    assert gate._commit_blob_bytes(
        repo_root=repo,
        commit=commit,
        path=gate.LIFECYCLE_REGISTRY_PATH,
        environ=gate._safe_environment(temporary_home=tmp_path / "home"),
        timeout_seconds=5,
    ) == expected
