#!/usr/bin/env python3
"""Verify the immutable source inventory for active generation runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "configs" / "runners" / "active_generation_bundles_v1.json"
)
EXPECTED_SCHEMA_VERSION = "active_generation_bundles_v1"
IGNORED_NAMES = {".DS_Store"}
IGNORED_SUFFIXES = {".pyc"}
IGNORED_PARTS = {"__pycache__"}
SHARED_RUNTIME_ROOTS = {
    "configs/generation",
    "src/benchmark/scene_generation/frozen_two_stage",
    "src/benchmark/scene_generation/retrieval",
    "configs/retrieval",
}


class RunnerInventoryError(RuntimeError):
    """Raised when an active runner bundle differs from its frozen manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mode(path: Path) -> str:
    return format(stat.S_IMODE(path.stat().st_mode), "04o")


def _discover_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in IGNORED_NAMES
        and path.suffix not in IGNORED_SUFFIXES
        and not (set(path.parts) & IGNORED_PARTS)
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerInventoryError(f"cannot read runner manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerInventoryError("runner manifest root must be an object")
    if value.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise RunnerInventoryError(
            f"schema_version must be {EXPECTED_SCHEMA_VERSION!r}"
        )
    return value


def _tracked_paths(roots: list[str]) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--", *roots],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RunnerInventoryError(f"git ls-files failed: {detail}")
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def validate_manifest(
    manifest: dict[str, Any],
    *,
    require_tracked: bool = True,
) -> dict[str, int]:
    bundles = manifest.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise RunnerInventoryError("bundles must be a non-empty array")

    bundle_ids: set[str] = set()
    roots: set[str] = set()
    declared_repo_paths: set[str] = set()
    file_count = 0

    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, dict):
            raise RunnerInventoryError(f"bundles[{index}] must be an object")
        bundle_id = bundle.get("bundle_id")
        root_text = bundle.get("root")
        role = bundle.get("role")
        entrypoints = bundle.get("entrypoints")
        files = bundle.get("files")
        if not isinstance(bundle_id, str) or not bundle_id:
            raise RunnerInventoryError(f"bundles[{index}].bundle_id is invalid")
        if bundle_id in bundle_ids:
            raise RunnerInventoryError(f"duplicate bundle_id: {bundle_id}")
        bundle_ids.add(bundle_id)
        if not isinstance(root_text, str) or not (
            root_text.startswith("tools/") or root_text in SHARED_RUNTIME_ROOTS
        ):
            raise RunnerInventoryError(f"invalid root for {bundle_id}: {root_text!r}")
        if root_text in roots:
            raise RunnerInventoryError(f"duplicate bundle root: {root_text}")
        roots.add(root_text)
        if not isinstance(role, str) or not role:
            raise RunnerInventoryError(f"role is required for {bundle_id}")
        if not isinstance(entrypoints, list) or not entrypoints or any(
            not isinstance(item, str) or not item for item in entrypoints
        ):
            raise RunnerInventoryError(f"entrypoints are invalid for {bundle_id}")
        if not isinstance(files, list) or not files:
            raise RunnerInventoryError(f"files are invalid for {bundle_id}")

        root = (REPO_ROOT / root_text).resolve()
        if root_text.startswith("tools/"):
            tools_root = (REPO_ROOT / "tools").resolve()
            if root.parent != tools_root:
                raise RunnerInventoryError(
                    f"tool bundle root must be a direct child of tools/: {root_text}"
                )
        elif root_text not in SHARED_RUNTIME_ROOTS:
            raise RunnerInventoryError(
                f"shared runtime root is not allowlisted: {root_text}"
            )
        if not root.is_dir():
            raise RunnerInventoryError(f"bundle root does not exist: {root_text}")

        declared: dict[str, dict[str, str]] = {}
        for file_index, item in enumerate(files):
            if not isinstance(item, dict):
                raise RunnerInventoryError(
                    f"{bundle_id}.files[{file_index}] must be an object"
                )
            relative = item.get("path")
            sha256 = item.get("sha256")
            mode = item.get("mode")
            if not isinstance(relative, str) or not relative:
                raise RunnerInventoryError(f"invalid file path in {bundle_id}")
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RunnerInventoryError(
                    f"file path escapes bundle {bundle_id}: {relative}"
                )
            if relative in declared:
                raise RunnerInventoryError(
                    f"duplicate file in {bundle_id}: {relative}"
                )
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise RunnerInventoryError(
                    f"invalid sha256 for {bundle_id}/{relative}"
                )
            if not isinstance(mode, str) or not mode.startswith("0"):
                raise RunnerInventoryError(f"invalid mode for {bundle_id}/{relative}")
            declared[relative] = {"sha256": sha256, "mode": mode}

        actual_paths = _discover_files(root)
        actual_relatives = {path.relative_to(root).as_posix() for path in actual_paths}
        missing = sorted(set(declared) - actual_relatives)
        extra = sorted(actual_relatives - set(declared))
        if missing or extra:
            raise RunnerInventoryError(
                f"file inventory differs for {bundle_id}: "
                f"missing={missing}, extra={extra}"
            )

        for relative, expected in declared.items():
            path = root / relative
            if path.is_symlink():
                raise RunnerInventoryError(
                    f"active runner files may not be symlinks: {bundle_id}/{relative}"
                )
            actual_hash = _sha256(path)
            if actual_hash != expected["sha256"]:
                raise RunnerInventoryError(
                    f"sha256 mismatch for {bundle_id}/{relative}: "
                    f"expected={expected['sha256']}, actual={actual_hash}"
                )
            actual_mode = _mode(path)
            if actual_mode != expected["mode"]:
                raise RunnerInventoryError(
                    f"mode mismatch for {bundle_id}/{relative}: "
                    f"expected={expected['mode']}, actual={actual_mode}"
                )
            declared_repo_paths.add(f"{root_text}/{relative}")

        unknown_entrypoints = sorted(set(entrypoints) - set(declared))
        if unknown_entrypoints:
            raise RunnerInventoryError(
                f"entrypoints are not in file inventory for {bundle_id}: "
                f"{unknown_entrypoints}"
            )
        file_count += len(declared)

    if require_tracked:
        tracked = _tracked_paths(sorted(roots))
        missing_from_git = sorted(declared_repo_paths - tracked)
        extra_in_git = sorted(tracked - declared_repo_paths)
        if missing_from_git or extra_in_git:
            raise RunnerInventoryError(
                "Git tracked runner inventory differs from manifest: "
                f"missing={missing_from_git}, extra={extra_in_git}"
            )

    return {"bundle_count": len(bundles), "file_count": file_count}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help="verify bytes and modes without requiring files in the Git index",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = validate_manifest(
            _load_manifest(args.manifest),
            require_tracked=not args.allow_untracked,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "schema_version": EXPECTED_SCHEMA_VERSION,
                    "tracked_checked": not args.allow_untracked,
                    **summary,
                },
                sort_keys=True,
            )
        )
        return 0
    except RunnerInventoryError as exc:
        print(f"active runner inventory error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
