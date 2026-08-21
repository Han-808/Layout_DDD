#!/usr/bin/env python3
"""Verify a committed Layout_DDD revision from an isolated tracked snapshot.

The checker materializes ``git archive`` into a fresh temporary Git index and
runs repository inventories plus the deterministic canonical test suite there.
It deliberately excludes the caller's untracked files, unreachable Git objects,
remote origin, and credential environment.  An optional wheel smoke check then
installs the wheel into a temporary target and probes the public package surface.

The default verifies ``HEAD`` even when the caller's worktree is dirty.  In
that case the JSON result is explicitly labelled ``ok_ref_only`` because the
uncommitted worktree was not part of the verified revision.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TREE_MODES = frozenset({"100644", "100755"})
LIFECYCLE_REGISTRY_PATH = "configs/repository/component_lifecycle_v1.json"


class CleanCheckoutError(RuntimeError):
    """Raised when the committed revision cannot pass the clean gate."""


@dataclass(frozen=True)
class StepResult:
    name: str
    command: tuple[str, ...]
    exit_code: int
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def _bounded_tail(value: str, *, max_chars: int = 4000) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def _safe_environment(
    source: Mapping[str, str] | None = None,
    *,
    checkout: Path | None = None,
    temporary_home: Path | None = None,
) -> dict[str, str]:
    """Return a minimal environment with no inherited credential values."""

    source = os.environ if source is None else source
    result: dict[str, str] = {}
    for name in ("PATH", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT"):
        value = source.get(name)
        if value:
            result[name] = str(value)
    if temporary_home is not None:
        result["HOME"] = str(temporary_home)
    result.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    if checkout is not None:
        result["PYTHONPATH"] = os.pathsep.join(
            (str(checkout / "src"), str(checkout))
        )
    return result


def _run_step(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    environ: Mapping[str, str],
    timeout_seconds: float = 300.0,
) -> StepResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=cwd,
            env=dict(environ),
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        return StepResult(
            name=name,
            command=tuple(str(item) for item in command),
            exit_code=124,
            elapsed_seconds=round(time.monotonic() - started, 3),
            stdout_tail=_bounded_tail(
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else str(exc.stdout or "")
            ),
            stderr_tail=_bounded_tail(
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr or "")
            ),
            timed_out=True,
        )
    return StepResult(
        name=name,
        command=tuple(str(item) for item in command),
        exit_code=int(completed.returncode),
        elapsed_seconds=round(time.monotonic() - started, 3),
        stdout_tail=_bounded_tail(completed.stdout),
        stderr_tail=_bounded_tail(completed.stderr),
        timed_out=False,
    )


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = _bounded_tail(completed.stderr or completed.stdout)
        raise CleanCheckoutError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _parse_stage_records(
    payload: bytes, *, label: str
) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, encoded_path = raw.split(b"\t", 1)
            fields = metadata.decode("ascii").split(" ")
            if len(fields) != 3:
                raise ValueError("stage metadata must contain mode, object, and stage")
            mode, object_id, stage = fields
            path = encoded_path.decode("utf-8", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise CleanCheckoutError(f"cannot parse {label} entry") from exc
        if stage != "0":
            raise CleanCheckoutError(f"unmerged {label} entry: {path}")
        result[path] = (mode, object_id)
    return result


def _commit_tree_entries(
    *,
    repo_root: Path,
    commit: str,
    environ: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, tuple[str, str]]:
    try:
        completed = subprocess.run(
            ("git", "ls-tree", "-r", "-z", commit),
            cwd=repo_root,
            env=dict(environ),
            check=False,
            capture_output=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise CleanCheckoutError(
            f"git ls-tree timed out after {timeout_seconds:g} seconds"
        ) from exc
    if completed.returncode != 0:
        raise CleanCheckoutError(
            "git ls-tree failed: "
            + _bounded_tail(completed.stderr.decode("utf-8", errors="replace"))
        )
    entries: dict[str, tuple[str, str]] = {}
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, encoded_path = raw.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = encoded_path.decode("utf-8", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise CleanCheckoutError("cannot parse git tree entry") from exc
        if mode not in ALLOWED_TREE_MODES or object_type != "blob":
            raise CleanCheckoutError(
                f"unsupported git tree entry: {path} mode={mode} type={object_type}"
            )
        entries[path] = (mode, object_id)
    return entries


def _git_object_format(
    *,
    repo_root: Path,
    environ: Mapping[str, str],
    timeout_seconds: float,
) -> str:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "--show-object-format"),
            cwd=repo_root,
            env=dict(environ),
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise CleanCheckoutError(
            f"git object-format query timed out after {timeout_seconds:g} seconds"
        ) from exc
    object_format = completed.stdout.strip()
    if completed.returncode != 0 or object_format not in {"sha1", "sha256"}:
        detail = _bounded_tail(completed.stderr or completed.stdout)
        raise CleanCheckoutError(f"cannot determine Git object format: {detail}")
    return object_format


def _first_parent(
    *,
    repo_root: Path,
    commit: str,
    environ: Mapping[str, str],
    timeout_seconds: float,
) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "rev-list", "--parents", "-n", "1", commit),
            cwd=repo_root,
            env=dict(environ),
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise CleanCheckoutError("git first-parent query timed out") from exc
    fields = completed.stdout.strip().split()
    if completed.returncode != 0 or not fields or fields[0] != commit:
        detail = _bounded_tail(completed.stderr or completed.stdout)
        raise CleanCheckoutError(f"cannot resolve commit parents: {detail}")
    return fields[1] if len(fields) > 1 else None


def _commit_blob_bytes(
    *,
    repo_root: Path,
    commit: str,
    path: str,
    environ: Mapping[str, str],
    timeout_seconds: float,
) -> bytes | None:
    try:
        listing = subprocess.run(
            ("git", "ls-tree", "-z", commit, "--", path),
            cwd=repo_root,
            env=dict(environ),
            check=False,
            capture_output=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise CleanCheckoutError("git baseline registry lookup timed out") from exc
    if listing.returncode != 0:
        detail = _bounded_tail(
            (listing.stderr or listing.stdout).decode("utf-8", errors="replace")
        )
        raise CleanCheckoutError(f"cannot inspect baseline registry: {detail}")
    records = [record for record in listing.stdout.split(b"\0") if record]
    if not records:
        return None
    if len(records) != 1:
        raise CleanCheckoutError("baseline registry lookup returned multiple entries")
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        actual_path = encoded_path.decode("utf-8", errors="strict")
    except (UnicodeError, ValueError) as exc:
        raise CleanCheckoutError("cannot parse baseline registry tree entry") from exc
    if (
        mode not in ALLOWED_TREE_MODES
        or object_type != "blob"
        or actual_path != path
    ):
        raise CleanCheckoutError(
            "baseline registry is not a regular blob: "
            f"path={actual_path}, mode={mode}, type={object_type}"
        )
    try:
        content = subprocess.run(
            ("git", "cat-file", "blob", object_id),
            cwd=repo_root,
            env=dict(environ),
            check=False,
            capture_output=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise CleanCheckoutError("git baseline registry read timed out") from exc
    if content.returncode != 0:
        detail = _bounded_tail(
            (content.stderr or content.stdout).decode("utf-8", errors="replace")
        )
        raise CleanCheckoutError(f"cannot read baseline registry: {detail}")
    return content.stdout


def _worktree_change_count(repo_root: Path) -> int:
    status = _git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    return len(status.splitlines()) if status else 0


def _safe_archive_member(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise CleanCheckoutError(f"unsafe git archive member: {name!r}")
    return path


def _extract_git_archive(payload: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            relative = _safe_archive_member(member.name)
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise CleanCheckoutError(
                    "tracked symlink, hardlink, device, or special archive member "
                    f"is not allowed by the clean gate: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise CleanCheckoutError(
                    f"cannot read git archive member: {member.name}"
                )
            with source, target.open("wb") as stream:
                while chunk := source.read(1024 * 1024):
                    stream.write(chunk)
            # Git trees preserve only the executable bit for regular files;
            # checkout materializes 100644/100755 rather than archive umask
            # variants such as 0664.
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


def _materialize_revision(
    *,
    repo_root: Path,
    destination: Path,
    commit: str,
    environ: Mapping[str, str],
    timeout_seconds: float,
) -> None:
    source_entries = _commit_tree_entries(
        repo_root=repo_root,
        commit=commit,
        environ=environ,
        timeout_seconds=timeout_seconds,
    )
    object_format = _git_object_format(
        repo_root=repo_root,
        environ=environ,
        timeout_seconds=timeout_seconds,
    )
    try:
        archive = subprocess.run(
            ("git", "archive", "--format=tar", commit),
            cwd=repo_root,
            env=dict(environ),
            check=False,
            capture_output=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise CleanCheckoutError(
            f"git archive timed out after {timeout_seconds:g} seconds"
        ) from exc
    if archive.returncode != 0:
        raise CleanCheckoutError(
            "git archive failed: "
            + _bounded_tail(
                (archive.stderr or archive.stdout).decode(
                    "utf-8", errors="replace"
                )
            )
        )
    _extract_git_archive(archive.stdout, destination)
    initialize = _run_step(
        "initialize_snapshot_index",
        ("git", "init", "--quiet", f"--object-format={object_format}"),
        cwd=destination,
        environ=environ,
        timeout_seconds=timeout_seconds,
    )
    if not initialize.ok:
        raise CleanCheckoutError(
            "snapshot git init failed: "
            f"{initialize.stderr_tail or initialize.stdout_tail}"
        )
    index = _run_step(
        "initialize_snapshot_index",
        ("git", "add", "--force", "--all"),
        cwd=destination,
        environ=environ,
        timeout_seconds=timeout_seconds,
    )
    if not index.ok:
        raise CleanCheckoutError(
            "snapshot git index failed: "
            f"{index.stderr_tail or index.stdout_tail}"
        )
    completed = subprocess.run(
        ("git", "ls-files", "--stage", "-z"),
        cwd=destination,
        env=dict(environ),
        check=False,
        capture_output=True,
        timeout=float(timeout_seconds),
    )
    if completed.returncode != 0:
        raise CleanCheckoutError(
            "snapshot index listing failed: "
            + _bounded_tail(completed.stderr.decode("utf-8", errors="replace"))
        )
    snapshot_entries = _parse_stage_records(
        completed.stdout, label="snapshot index"
    )
    if snapshot_entries != source_entries:
        missing = sorted(set(source_entries) - set(snapshot_entries))
        extra = sorted(set(snapshot_entries) - set(source_entries))
        entry_mismatches = {
            path: {
                "source": source_entries[path],
                "snapshot": snapshot_entries[path],
            }
            for path in sorted(set(source_entries) & set(snapshot_entries))
            if source_entries[path] != snapshot_entries[path]
        }
        raise CleanCheckoutError(
            "snapshot index differs from source tree: "
            f"missing={missing}, extra={extra}, entry_mismatches={entry_mismatches}"
        )


def _default_steps(
    *,
    python: str,
    skip_collection: bool,
    skip_canonical_tests: bool,
    baseline_registry: Path | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    inventory_command = [python, "scripts/check_pytest_inventory.py"]
    if skip_collection:
        inventory_command.append("--skip-collect")
    lifecycle_command = [python, "scripts/check_component_lifecycle.py"]
    if baseline_registry is not None:
        lifecycle_command.extend(
            ("--baseline-registry", str(baseline_registry))
        )
    steps = [
        (
            "component_lifecycle",
            tuple(lifecycle_command),
        ),
        (
            "active_runner_sources",
            (python, "scripts/check_active_runner_sources.py"),
        ),
        ("pytest_inventory", tuple(inventory_command)),
    ]
    if not skip_canonical_tests:
        steps.append(
            (
                "canonical_tests",
                (
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "-m",
                    "not requires_blender and not requires_loopback and not requires_local_data",
                ),
            )
        )
    return steps


def _wheel_steps(
    *,
    python: str,
    checkout: Path,
    scratch: Path,
    environ: Mapping[str, str],
    timeout_seconds: float,
) -> list[StepResult]:
    dist = scratch / "dist"
    dist.mkdir()
    build = _run_step(
        "wheel_build",
        (
            python,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist),
        ),
        cwd=checkout,
        environ=environ,
        timeout_seconds=timeout_seconds,
    )
    results = [build]
    if not build.ok:
        return results
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        results.append(
            StepResult(
                name="wheel_import",
                command=(),
                exit_code=2,
                elapsed_seconds=0.0,
                stdout_tail="",
                stderr_tail=f"expected exactly one wheel, found {len(wheels)}",
                timed_out=False,
            )
        )
        return results
    installed = scratch / "installed"
    install = _run_step(
        "wheel_install",
        (
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--target",
            str(installed),
            str(wheels[0]),
        ),
        cwd=scratch,
        environ=environ,
        timeout_seconds=timeout_seconds,
    )
    results.append(install)
    if not install.ok:
        return results
    isolated = scratch / "isolated"
    isolated.mkdir()
    wheel_env = dict(environ)
    wheel_env["PYTHONPATH"] = str(installed)
    import_code = (
        "import importlib.util, pathlib; "
        "import benchmark.api.generation as generation; "
        "import benchmark.api.evaluation as evaluation; "
        "import benchmark.api.submission as submission; "
        "import benchmark.scene_generation.frozen_two_stage as compatibility; "
        "from benchmark.resources import runtime_resource_path; "
        f"expected=pathlib.Path({str(installed)!r}).resolve(); "
        "origins=[generation.__file__, evaluation.__file__, submission.__file__, compatibility.__file__]; "
        "assert all(pathlib.Path(value).resolve().is_relative_to(expected) for value in origins); "
        "assert importlib.util.find_spec('benchmark.legend') is None; "
        "resource=runtime_resource_path('configs/evaluation/metric_profile_canonical_v2.yaml'); "
        "assert pathlib.Path(resource).is_file(); "
        "print({'origins': origins, 'resource': str(resource)})"
    )
    results.append(
        _run_step(
            "wheel_import",
            (python, "-c", import_code),
            cwd=isolated,
            environ=wheel_env,
            timeout_seconds=timeout_seconds,
        )
    )
    return results


def verify_clean_checkout(
    *,
    repo_root: Path,
    source_ref: str = "HEAD",
    python: str = sys.executable,
    skip_collection: bool = False,
    skip_canonical_tests: bool = False,
    skip_wheel: bool = False,
    require_worktree_match: bool = False,
    steps: Iterable[tuple[str, Sequence[str]]] | None = None,
    step_timeout_seconds: float = 300.0,
) -> dict[str, object]:
    repo_root = repo_root.expanduser().resolve()
    if (
        not math.isfinite(float(step_timeout_seconds))
        or float(step_timeout_seconds) <= 0.0
    ):
        raise CleanCheckoutError(
            "step_timeout_seconds must be finite and greater than zero"
        )
    commit = _git_output(repo_root, "rev-parse", "--verify", f"{source_ref}^{{commit}}")
    change_count = _worktree_change_count(repo_root)
    if require_worktree_match and change_count:
        raise CleanCheckoutError(
            f"worktree has {change_count} change(s) not present in {commit}"
        )

    results: list[StepResult] = []
    baseline_commit: str | None = None
    baseline_present = False
    with tempfile.TemporaryDirectory(prefix="layout-ddd-clean-checkout-") as raw:
        scratch = Path(raw).resolve()
        checkout = scratch / "checkout"
        temporary_home = scratch / "home"
        temporary_home.mkdir()
        base_env = _safe_environment(temporary_home=temporary_home)
        baseline_commit = _first_parent(
            repo_root=repo_root,
            commit=commit,
            environ=base_env,
            timeout_seconds=step_timeout_seconds,
        )
        baseline_registry_path: Path | None = None
        if baseline_commit is not None:
            baseline_bytes = _commit_blob_bytes(
                repo_root=repo_root,
                commit=baseline_commit,
                path=LIFECYCLE_REGISTRY_PATH,
                environ=base_env,
                timeout_seconds=step_timeout_seconds,
            )
            if baseline_bytes is not None:
                baseline_registry_path = scratch / "baseline_component_lifecycle.json"
                baseline_registry_path.write_bytes(baseline_bytes)
                baseline_present = True
        _materialize_revision(
            repo_root=repo_root,
            destination=checkout,
            commit=commit,
            environ=base_env,
            timeout_seconds=step_timeout_seconds,
        )
        check_env = _safe_environment(
            checkout=checkout,
            temporary_home=temporary_home,
        )
        selected_steps = list(
            steps
            if steps is not None
            else _default_steps(
                python=python,
                skip_collection=skip_collection,
                skip_canonical_tests=skip_canonical_tests,
                baseline_registry=baseline_registry_path,
            )
        )
        for name, command in selected_steps:
            result = _run_step(
                str(name),
                tuple(str(item) for item in command),
                cwd=checkout,
                environ=check_env,
                timeout_seconds=step_timeout_seconds,
            )
            results.append(result)
            if not result.ok:
                break
        if all(item.ok for item in results) and not skip_wheel:
            results.extend(
                _wheel_steps(
                    python=python,
                    checkout=checkout,
                    scratch=scratch,
                    environ=check_env,
                    timeout_seconds=step_timeout_seconds,
                )
            )

    failed = [item for item in results if not item.ok]
    successful_names = {item.name for item in results if item.ok}
    collection_checked = (
        not skip_collection and "pytest_inventory" in successful_names
    )
    canonical_tests_checked = (
        not skip_canonical_tests and "canonical_tests" in successful_names
    )
    wheel_checked = (
        not skip_wheel
        and {"wheel_build", "wheel_install", "wheel_import"}
        <= successful_names
    )
    return {
        "schema_version": "clean_checkout_gate_v1",
        "status": (
            "failed"
            if failed
            else ("ok" if change_count == 0 else "ok_ref_only")
        ),
        "verified_ref": source_ref,
        "verified_commit": commit,
        "worktree_matches_verified_ref": change_count == 0,
        "worktree_change_count": change_count,
        "source_origin_present": False,
        "lifecycle_baseline_commit": baseline_commit,
        "lifecycle_baseline_present": baseline_present,
        "credential_environment_filtered": True,
        "pip_index_disabled": True,
        "network_isolation_enforced": False,
        "test_collection_checked": collection_checked,
        "canonical_tests_checked": canonical_tests_checked,
        "wheel_checked": wheel_checked,
        "steps": [
            {**asdict(item), "ok": item.ok}
            for item in results
        ],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-canonical-tests", action="store_true")
    parser.add_argument("--skip-wheel", action="store_true")
    parser.add_argument("--require-worktree-match", action="store_true")
    parser.add_argument(
        "--step-timeout-seconds",
        type=float,
        default=300.0,
        help="Per subprocess/archive timeout (default: 300 seconds).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = verify_clean_checkout(
            repo_root=args.repo_root,
            source_ref=str(args.source_ref),
            python=str(args.python),
            skip_collection=bool(args.skip_collection),
            skip_canonical_tests=bool(args.skip_canonical_tests),
            skip_wheel=bool(args.skip_wheel),
            require_worktree_match=bool(args.require_worktree_match),
            step_timeout_seconds=float(args.step_timeout_seconds),
        )
    except CleanCheckoutError as exc:
        report = {
            "schema_version": "clean_checkout_gate_v1",
            "status": "failed",
            "error": str(exc),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") in {"ok", "ok_ref_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
