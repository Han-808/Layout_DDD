#!/usr/bin/env python3
"""Verify byte-identical source and package copies of runtime resources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "configs" / "resources" / "runtime_resources_v1.json"
)
EXPECTED_SCHEMA_VERSION = "runtime_resources_v1"
EXPECTED_CAMERA_IMPLEMENTATION_SHA256 = (
    "457ff95848300defe71e613c24e772aa2f541ee258b66471d894a566499f0929"
)
EXPECTED_RESOURCES = {
    "schemas/generator_catalog_placement_v1.schema.json",
    "schemas/generator_layout_v1.schema.json",
    "configs/evaluation/metric_profile_canonical_v2.yaml",
    "configs/evaluation/scene_generation_leaderboard_scoring_v1.json",
    "configs/evaluation/metric_profile_game_canonical_v1.yaml",
    "configs/grouping/vlm_visual_evidence_scope_v2.yaml",
    "configs/game/game_mode_canonical_v1.yaml",
    "configs/game/counter_strike_static_arena_style_v1.json",
    "configs/game/counter_strike/benchmark_v1.yaml",
    "configs/retrieval/profiles_v2.json",
    "configs/retrieval/golden/imaginarium_qwen3_0_6b_v2.json",
}
PACKAGE_PREFIX = "src/benchmark/_resources/"


class RuntimeResourceError(RuntimeError):
    """Raised when a package resource differs from its canonical source."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_repo_path(value: Any, *, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise RuntimeResourceError(f"{field} must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeResourceError(f"{field} must be repository-relative: {value!r}")
    resolved = (REPO_ROOT / relative).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeResourceError(f"{field} escapes repository: {value!r}") from exc
    return relative.as_posix(), resolved


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeResourceError(f"cannot read resource manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeResourceError("resource manifest root must be an object")
    if value.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise RuntimeResourceError(
            f"schema_version must be {EXPECTED_SCHEMA_VERSION!r}"
        )
    return value


def _tracked_paths(paths: list[str]) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--", *paths],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeResourceError(f"git ls-files failed: {detail}")
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def validate_manifest(
    manifest: dict[str, Any],
    *,
    require_tracked: bool = True,
) -> dict[str, int]:
    resources = manifest.get("resources")
    if not isinstance(resources, list) or not resources:
        raise RuntimeResourceError("resources must be a non-empty array")
    if (
        manifest.get("camera_evidence_implementation_sha256")
        != EXPECTED_CAMERA_IMPLEMENTATION_SHA256
    ):
        raise RuntimeResourceError(
            "camera_evidence_implementation_sha256 does not match the "
            "approved relocatable implementation"
        )

    declared: set[str] = set()
    tracked_candidates: set[str] = set()
    package_paths: set[str] = set()
    for index, item in enumerate(resources):
        if not isinstance(item, dict):
            raise RuntimeResourceError(f"resources[{index}] must be an object")
        resource = item.get("resource")
        if not isinstance(resource, str) or resource not in EXPECTED_RESOURCES:
            raise RuntimeResourceError(f"unexpected resource identifier: {resource!r}")
        if resource in declared:
            raise RuntimeResourceError(f"duplicate resource identifier: {resource}")
        declared.add(resource)

        source_text, source_path = _safe_repo_path(
            item.get("source_path"), field=f"resources[{index}].source_path"
        )
        package_text, package_path = _safe_repo_path(
            item.get("package_path"), field=f"resources[{index}].package_path"
        )
        if source_text != resource:
            raise RuntimeResourceError(
                f"source_path must equal resource identifier for {resource}"
            )
        expected_package = f"{PACKAGE_PREFIX}{resource}"
        if package_text != expected_package:
            raise RuntimeResourceError(
                f"package_path must be {expected_package!r} for {resource}"
            )
        if not source_path.is_file() or not package_path.is_file():
            raise RuntimeResourceError(
                f"source or package resource is missing for {resource}"
            )
        if source_path.read_bytes() != package_path.read_bytes():
            raise RuntimeResourceError(
                f"source and package bytes differ for {resource}"
            )
        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeResourceError(f"invalid sha256 for {resource}")
        actual_hash = _sha256(source_path)
        if actual_hash != expected_hash:
            raise RuntimeResourceError(
                f"sha256 mismatch for {resource}: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        tracked_candidates.update({source_text, package_text})
        package_paths.add(package_text)

    if declared != EXPECTED_RESOURCES:
        raise RuntimeResourceError(
            "resource inventory differs from required runtime set: "
            f"missing={sorted(EXPECTED_RESOURCES - declared)}, "
            f"extra={sorted(declared - EXPECTED_RESOURCES)}"
        )

    packaged_root = REPO_ROOT / "src" / "benchmark" / "_resources"
    actual_package_paths = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in packaged_root.rglob("*")
        if path.is_file()
        and path.name != "__init__.py"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if actual_package_paths != package_paths:
        raise RuntimeResourceError(
            "package resource directory differs from manifest: "
            f"missing={sorted(package_paths - actual_package_paths)}, "
            f"extra={sorted(actual_package_paths - package_paths)}"
        )

    if require_tracked:
        tracked = _tracked_paths(sorted(tracked_candidates))
        if tracked != tracked_candidates:
            raise RuntimeResourceError(
                "runtime resource files are not fully tracked: "
                f"missing={sorted(tracked_candidates - tracked)}, "
                f"extra={sorted(tracked - tracked_candidates)}"
            )

    return {"resource_count": len(resources)}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help="verify resource bytes without requiring files in the Git index",
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
    except RuntimeResourceError as exc:
        print(f"runtime resource error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
