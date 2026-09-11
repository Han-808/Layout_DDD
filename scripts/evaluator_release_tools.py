"""Read-only verification for the repository's versioned evaluator sources."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


REGISTRY_PATH = "configs/runners/floorplan_evaluator_baselines_v1.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EvaluatorReleaseError(ValueError):
    """The declared release is missing, unsafe, or no longer matches its pins."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluatorReleaseError(f"Expected a JSON object: {path.name}")
    return value


def relative_file(root: Path, name: str) -> Path:
    """Accept only canonical relative file names, never symlink escapes."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise EvaluatorReleaseError("Invalid release-relative path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts:
        raise EvaluatorReleaseError(f"Unsafe release-relative path: {name}")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise EvaluatorReleaseError(f"Symlinks are not release source files: {name}")
    if not path.is_file():
        raise EvaluatorReleaseError(f"Missing release file: {name}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(entries: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, sha in sorted(entries.items()):
        if not isinstance(name, str) or not isinstance(sha, str) or not SHA256.fullmatch(sha):
            raise EvaluatorReleaseError("Invalid source hash entry")
        relative = PurePosixPath(name)
        if (not name or "\\" in name or relative.is_absolute()
                or relative.as_posix() != name or ".." in relative.parts):
            raise EvaluatorReleaseError("Unsafe source hash entry")
        digest.update((name + "\0" + sha + "\n").encode("utf-8"))
    return digest.hexdigest()


def verify_files(root: Path, entries: Mapping[str, str]) -> None:
    for name, expected in entries.items():
        if sha256_file(relative_file(root, name)) != expected:
            raise EvaluatorReleaseError(f"Source hash mismatch: {name}")


def verify_snapshot(repo: Path, baseline_id: str) -> dict[str, Any]:
    registry = read_json(repo / REGISTRY_PATH)
    baseline = registry["baselines"][baseline_id]
    publication = baseline.get("source_publication", {})
    manifest_path = relative_file(repo, publication["manifest"])
    if sha256_file(manifest_path) != publication["manifest_sha256"]:
        raise EvaluatorReleaseError("Snapshot manifest hash mismatch")
    manifest = read_json(manifest_path)
    if (manifest.get("schema_version") != "evaluator_source_snapshot_v1"
            or manifest.get("baseline_id") != baseline_id
            or manifest.get("mode") != baseline["mode"]):
        raise EvaluatorReleaseError("Snapshot identity mismatch")
    origin = manifest["origin_files_sha256"]
    published = manifest["published_files_sha256"]
    identity = baseline["code_identity"]
    if (len(origin) != manifest["origin_file_count"]
            or len(origin) != identity["file_count"]
            or tree_sha256(origin) != identity["tree_sha256"]
            or manifest["origin_tree_sha256"] != identity["tree_sha256"]):
        raise EvaluatorReleaseError("Historical source-tree identity mismatch")
    if (not published or len(published) != manifest["published_file_count"]
            or tree_sha256(published) != manifest["published_tree_sha256"]
            or manifest["published_tree_sha256"] != publication["tree_sha256"]
            or any(origin.get(name) != sha for name, sha in published.items())):
        raise EvaluatorReleaseError("Published source is not the declared exact subset")
    root = manifest_path.parent
    verify_files(root, published)
    # Extra files in import/config/entrypoint roots can change execution despite
    # matching every listed file. Bytecode caches are not published source.
    actual = {
        path.relative_to(root).as_posix()
        for folder in ("src", "configs", "scripts")
        for path in (root / folder).rglob("*")
        if (path.is_file() or path.is_symlink())
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if actual != set(published):
        raise EvaluatorReleaseError("Unexpected or missing snapshot source files")
    return {
        "baseline_id": baseline_id,
        "mode": baseline["mode"],
        "source_root": str(root),
        "verified_file_count": len(published),
        "published_tree_sha256": manifest["published_tree_sha256"],
        "origin_tree_sha256": identity["tree_sha256"],
        "versions": baseline["versions"],
    }


def verify_nonrect_core(repo: Path) -> dict[str, Any]:
    """Verify existing Nonrect core without rewriting it to the Single-room version."""
    manifest = read_json(repo / "configs/runners/nonrect_3186983_core_manifest_v1.json")
    entries = manifest["files"]
    tree_sha256(entries)
    verify_files(repo, entries)
    return {"source_commit": manifest["source_commit"], "verified_file_count": len(entries)}
