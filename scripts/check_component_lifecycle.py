#!/usr/bin/env python3
"""Validate repository component lifecycles against the current Git index.

The registry is deliberately small and strict.  It is not a file catalogue and
it does not infer that a version is safe to remove merely because its name is
old.  Instead it records ownership and lifecycle intent, then checks the facts
that can be established without importing repository code:

* active generation bundles have an exact lifecycle owner;
* component roots do not overlap;
* tracked policy matches the current Git index;
* tracked tests do not import Python implementations that exist only locally;
* active components do not reach unsafe lifecycle states; and
* deletion candidates have both quarantine evidence and no inbound references;
* deleted code leaves an immutable, approval-backed tombstone; and
* registry transitions cannot silently discard lifecycle identifiers.

No repository module is imported and no experiment artifact is inspected.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    REPO_ROOT / "configs" / "repository" / "component_lifecycle_v1.json"
)
DEFAULT_ACTIVE_BUNDLES = (
    REPO_ROOT / "configs" / "runners" / "active_generation_bundles_v1.json"
)
EXPECTED_SCHEMA_VERSION = "component_lifecycle_v1"

LIFECYCLE_STATES = frozenset(
    {
        "active",
        "compatibility",
        "frozen_replay",
        "local_only",
        "quarantine",
        "deletion_candidate",
        "deleted",
    }
)
TRACKED_POLICIES = frozenset({"all_tracked", "no_tracked_files"})
RUNNABLE_STATES = frozenset({"active", "compatibility"})
UNSAFE_DEPENDENCY_STATES = frozenset(
    {"frozen_replay", "local_only", "quarantine", "deletion_candidate", "deleted"}
)
ALLOWED_GIT_MODES = frozenset({"100644", "100755"})
VERSION_PATH_PATTERN = re.compile(r"(?:^|[/_.-])v[0-9]+(?:[/_.-]|$)")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMPONENT_FIELDS = frozenset(
    {
        "component_id",
        "root",
        "state",
        "owner",
        "replacement",
        "reason",
        "tracked_policy",
        "entrypoints",
        "sources",
        "dependencies",
        "quarantine_evidence",
        "quarantine_started_at",
        "minimum_quarantine_days",
        "content_sha256",
        "deletion_approval_evidence",
    }
)
COMPONENT_REQUIRED_FIELDS = COMPONENT_FIELDS - {
    "quarantine_started_at",
    "minimum_quarantine_days",
    "content_sha256",
    "deletion_approval_evidence",
}
ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "path",
        "state",
        "owner",
        "replacement",
        "reason",
        "content_sha256",
        "mirror_of",
        "quarantine_evidence",
        "quarantine_started_at",
        "minimum_quarantine_days",
        "deletion_approval_evidence",
        "git_mode",
    }
)
ARTIFACT_REQUIRED_FIELDS = ARTIFACT_FIELDS - {
    "replacement",
    "content_sha256",
    "mirror_of",
    "quarantine_evidence",
    "quarantine_started_at",
    "minimum_quarantine_days",
    "deletion_approval_evidence",
    "git_mode",
}
IGNORED_FILE_NAMES = frozenset({".DS_Store"})
IGNORED_FILE_SUFFIXES = frozenset({".pyc"})
IGNORED_PATH_PARTS = frozenset({"__pycache__", ".pytest_cache"})
TEXT_SUFFIXES = frozenset(
    {
        ".cfg",
        ".ini",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)


class LifecycleError(RuntimeError):
    """Raised when lifecycle evidence is incomplete or contradictory."""


@dataclass(frozen=True)
class Component:
    component_id: str
    root: str
    state: str
    owner: str
    replacement: str | None
    reason: str
    tracked_policy: str
    entrypoints: tuple[str, ...]
    sources: tuple[str, ...]
    dependencies: tuple[str, ...]
    quarantine_evidence: tuple[str, ...]
    quarantine_started_at: date | None
    minimum_quarantine_days: int
    content_sha256: str | None
    deletion_approval_evidence: tuple[str, ...]


@dataclass(frozen=True)
class VersionedArtifact:
    artifact_id: str
    path: str
    state: str
    owner: str
    replacement: str | None
    reason: str
    content_sha256: str | None
    mirror_of: str | None
    quarantine_evidence: tuple[str, ...]
    quarantine_started_at: date | None
    minimum_quarantine_days: int
    deletion_approval_evidence: tuple[str, ...]
    git_mode: str | None


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"{label} root must be an object")
    return value


def _validate_registry_object(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise LifecycleError(
            f"schema_version must be {EXPECTED_SCHEMA_VERSION!r}"
        )
    unknown = sorted(set(value) - {"schema_version", "components", "artifacts"})
    if unknown:
        raise LifecycleError(f"unknown registry fields: {unknown}")
    return value


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load a registry without importing any component implementation."""

    return _validate_registry_object(
        _read_json_object(path, label="component lifecycle registry")
    )


def _load_registry_from_git_ref(
    *, repo_root: Path, ref: str, registry_repo_path: str
) -> dict[str, Any] | None:
    verify = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if verify.returncode != 0:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        if ref == "HEAD" and head.returncode != 0:
            return None
        detail = (verify.stderr or verify.stdout).decode(
            "utf-8", errors="replace"
        ).strip()
        raise LifecycleError(f"cannot resolve baseline ref {ref!r}: {detail}")

    listing = subprocess.run(
        ["git", "ls-tree", "-z", ref, "--", registry_repo_path],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if listing.returncode != 0:
        detail = (listing.stderr or listing.stdout).decode(
            "utf-8", errors="replace"
        ).strip()
        raise LifecycleError(f"cannot inspect baseline registry: {detail}")
    records = [record for record in listing.stdout.split(b"\0") if record]
    if not records:
        return None
    if len(records) != 1:
        raise LifecycleError("baseline registry lookup returned multiple entries")
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        path = encoded_path.decode("utf-8", errors="strict")
    except (UnicodeError, ValueError) as exc:
        raise LifecycleError("cannot parse baseline registry tree entry") from exc
    if (
        mode not in ALLOWED_GIT_MODES
        or object_type != "blob"
        or path != registry_repo_path
    ):
        raise LifecycleError(
            "baseline registry is not a regular tracked blob: "
            f"path={path}, mode={mode}, type={object_type}"
        )
    content = subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if content.returncode != 0:
        detail = (content.stderr or content.stdout).decode(
            "utf-8", errors="replace"
        ).strip()
        raise LifecycleError(f"cannot read baseline registry blob: {detail}")
    try:
        value = json.loads(content.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LifecycleError("baseline registry is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise LifecycleError("baseline registry root must be an object")
    return _validate_registry_object(value)


def load_active_bundles(path: Path = DEFAULT_ACTIVE_BUNDLES) -> dict[str, Any]:
    return _read_json_object(path, label="active generation bundle manifest")


def _git_index_records(repo_root: Path = REPO_ROOT) -> dict[str, tuple[str, str]]:
    """Return exact stage-zero index path -> (mode, blob OID) records."""

    completed = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).decode(
            "utf-8", errors="replace"
        ).strip()
        raise LifecycleError(f"git ls-files failed: {detail}")
    result: dict[str, tuple[str, str]] = {}
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, encoded_path = raw.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split(" ")
            path = encoded_path.decode("utf-8", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise LifecycleError("cannot parse git index entry") from exc
        if stage != "0":
            raise LifecycleError(f"unmerged git index entry is not allowed: {path}")
        result[path] = (mode, object_id)
    return result


def git_index_entries(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Return exact stage-zero index path -> mode entries."""

    return {
        path: mode
        for path, (mode, _object_id) in _git_index_records(repo_root).items()
    }


def git_tracked_paths(repo_root: Path = REPO_ROOT) -> set[str]:
    return set(git_index_entries(repo_root))


def _safe_index_path(path: str) -> PurePosixPath:
    candidate = PurePosixPath(path)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.as_posix() != path
    ):
        raise LifecycleError(f"unsafe Git index path: {path!r}")
    return candidate


def _materialize_index_blobs(
    *,
    repo_root: Path,
    snapshot: Path,
    records: Mapping[str, tuple[str, str]],
) -> None:
    """Write raw index blobs without checkout filters or worktree configuration."""

    ordered = sorted(records.items())
    for path, (mode, _object_id) in ordered:
        _safe_index_path(path)
        if mode not in ALLOWED_GIT_MODES:
            raise LifecycleError(
                f"unsupported Git index mode: {path}={mode}"
            )
    requested = b"".join(
        object_id.encode("ascii") + b"\n"
        for _path, (_mode, object_id) in ordered
    )
    try:
        completed = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=repo_root,
            input=requested,
            check=False,
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise LifecycleError("git cat-file timed out") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).decode(
            "utf-8", errors="replace"
        ).strip()
        raise LifecycleError(f"git cat-file failed: {detail}")

    payload = completed.stdout
    offset = 0
    for path, (mode, expected_object_id) in ordered:
        newline = payload.find(b"\n", offset)
        if newline < 0:
            raise LifecycleError(f"truncated git cat-file header for: {path}")
        try:
            object_id, object_type, size_text = payload[offset:newline].decode(
                "ascii"
            ).split(" ")
            size = int(size_text)
        except (UnicodeError, ValueError) as exc:
            raise LifecycleError(
                f"cannot parse git cat-file header for: {path}"
            ) from exc
        if object_id != expected_object_id or object_type != "blob" or size < 0:
            raise LifecycleError(
                "unexpected git cat-file object for "
                f"{path}: {object_id} {object_type} {size}"
            )
        start = newline + 1
        end = start + size
        if end >= len(payload) or payload[end : end + 1] != b"\n":
            raise LifecycleError(f"truncated git cat-file blob for: {path}")
        relative = _safe_index_path(path)
        target = snapshot.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload[start:end])
        target.chmod(0o755 if mode == "100755" else 0o644)
        offset = end + 1
    if payload[offset:]:
        raise LifecycleError("unexpected trailing git cat-file output")


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise LifecycleError(f"{field} must be an array of strings")
    result = tuple(item.strip() for item in value)
    if any(not item for item in result):
        raise LifecycleError(f"{field} may not contain empty strings")
    if len(result) != len(set(result)):
        raise LifecycleError(f"{field} contains duplicates")
    return result


def _optional_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise LifecycleError(f"{field} must be a lowercase SHA-256 digest or null")
    return value


def _optional_date(value: Any, *, field: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LifecycleError(f"{field} must be an ISO date or null")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise LifecycleError(f"{field} must be an ISO date or null") from exc
    if value != parsed.isoformat():
        raise LifecycleError(f"{field} must use canonical YYYY-MM-DD form")
    return parsed


def _nonnegative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LifecycleError(f"{field} must be a non-negative integer")
    return value


def _validate_lifecycle_evidence(
    *,
    identifier: str,
    state: str,
    content_sha256: str | None,
    quarantine_evidence: tuple[str, ...],
    quarantine_started_at: date | None,
    minimum_quarantine_days: int,
    deletion_approval_evidence: tuple[str, ...],
) -> None:
    if state in {"frozen_replay", "quarantine", "deletion_candidate", "deleted"} and (
        content_sha256 is None
    ):
        raise LifecycleError(
            f"{state} {identifier} requires content_sha256"
        )
    if state in {"quarantine", "deletion_candidate", "deleted"}:
        if not quarantine_evidence:
            raise LifecycleError(
                f"{state} {identifier} requires quarantine evidence"
            )
        if quarantine_started_at is None:
            raise LifecycleError(
                f"{state} {identifier} requires quarantine_started_at"
            )
        if minimum_quarantine_days < 1:
            raise LifecycleError(
                f"{state} {identifier} requires minimum_quarantine_days >= 1"
            )
    elif (
        quarantine_evidence
        or quarantine_started_at is not None
        or minimum_quarantine_days != 0
    ):
        raise LifecycleError(
            f"non-quarantine {identifier} may not declare quarantine evidence/time"
        )
    if state == "deleted":
        if not deletion_approval_evidence:
            raise LifecycleError(
                f"deleted {identifier} requires deletion approval evidence"
            )
    elif deletion_approval_evidence:
        raise LifecycleError(
            f"non-deleted {identifier} may not declare deletion approval evidence"
        )


def _exact_repo_path(value: Any, *, field: str) -> str:
    text = _string(value, field=field)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(char in text for char in "*?[]{}")
    ):
        raise LifecycleError(f"{field} must be an exact normalized repo path: {text!r}")
    return text


def _path_is_within(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return path_parts[: len(root_parts)] == root_parts


def _parse_components(registry: Mapping[str, Any]) -> list[Component]:
    raw = registry.get("components")
    if not isinstance(raw, list) or not raw:
        raise LifecycleError("components must be a non-empty array")

    components: list[Component] = []
    ids: set[str] = set()
    roots: set[str] = set()
    for index, item in enumerate(raw):
        field = f"components[{index}]"
        if not isinstance(item, dict):
            raise LifecycleError(f"{field} must be an object")
        unknown = sorted(set(item) - COMPONENT_FIELDS)
        missing = sorted(COMPONENT_REQUIRED_FIELDS - set(item))
        if unknown or missing:
            raise LifecycleError(
                f"{field} fields differ from schema: missing={missing}, unknown={unknown}"
            )

        component_id = _string(item["component_id"], field=f"{field}.component_id")
        if component_id in ids:
            raise LifecycleError(f"duplicate component_id: {component_id}")
        ids.add(component_id)
        root = _exact_repo_path(item["root"], field=f"{field}.root")
        if root in roots:
            raise LifecycleError(f"duplicate component root: {root}")
        roots.add(root)
        state = _string(item["state"], field=f"{field}.state")
        if state not in LIFECYCLE_STATES:
            raise LifecycleError(
                f"unknown lifecycle state for {component_id}: {state!r}"
            )
        owner = _string(item["owner"], field=f"{field}.owner")
        reason = _string(item["reason"], field=f"{field}.reason")
        tracked_policy = _string(
            item["tracked_policy"], field=f"{field}.tracked_policy"
        )
        if tracked_policy not in TRACKED_POLICIES:
            raise LifecycleError(
                f"unknown tracked policy for {component_id}: {tracked_policy!r}"
            )
        replacement_value = item["replacement"]
        if replacement_value is not None and not isinstance(replacement_value, str):
            raise LifecycleError(f"{field}.replacement must be a string or null")
        replacement = replacement_value.strip() if replacement_value else None
        entrypoints = tuple(
            _exact_repo_path(path, field=f"{field}.entrypoints")
            for path in _string_list(item["entrypoints"], field=f"{field}.entrypoints")
        )
        sources = tuple(
            _exact_repo_path(path, field=f"{field}.sources")
            for path in _string_list(item["sources"], field=f"{field}.sources")
        )
        if not sources:
            raise LifecycleError(f"{field}.sources must name at least one source")
        dependencies = _string_list(
            item["dependencies"], field=f"{field}.dependencies"
        )
        evidence = tuple(
            _exact_repo_path(path, field=f"{field}.quarantine_evidence")
            for path in _string_list(
                item["quarantine_evidence"], field=f"{field}.quarantine_evidence"
            )
        )
        deletion_approval_evidence = tuple(
            _exact_repo_path(
                path, field=f"{field}.deletion_approval_evidence"
            )
            for path in _string_list(
                item.get("deletion_approval_evidence", []),
                field=f"{field}.deletion_approval_evidence",
            )
        )
        quarantine_started_at = _optional_date(
            item.get("quarantine_started_at"),
            field=f"{field}.quarantine_started_at",
        )
        minimum_quarantine_days = _nonnegative_int(
            item.get("minimum_quarantine_days", 0),
            field=f"{field}.minimum_quarantine_days",
        )
        content_sha256 = _optional_sha256(
            item.get("content_sha256"), field=f"{field}.content_sha256"
        )
        _validate_lifecycle_evidence(
            identifier=component_id,
            state=state,
            content_sha256=content_sha256,
            quarantine_evidence=evidence,
            quarantine_started_at=quarantine_started_at,
            minimum_quarantine_days=minimum_quarantine_days,
            deletion_approval_evidence=deletion_approval_evidence,
        )
        for path in (*entrypoints, *sources):
            if not _path_is_within(path, root):
                raise LifecycleError(
                    f"{component_id} path is outside its exact root {root}: {path}"
                )
        components.append(
            Component(
                component_id=component_id,
                root=root,
                state=state,
                owner=owner,
                replacement=replacement,
                reason=reason,
                tracked_policy=tracked_policy,
                entrypoints=entrypoints,
                sources=sources,
                dependencies=dependencies,
                quarantine_evidence=evidence,
                quarantine_started_at=quarantine_started_at,
                minimum_quarantine_days=minimum_quarantine_days,
                content_sha256=content_sha256,
                deletion_approval_evidence=deletion_approval_evidence,
            )
        )

    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            if _path_is_within(left.root, right.root) or _path_is_within(
                right.root, left.root
            ):
                raise LifecycleError(
                    "component roots overlap: "
                    f"{left.component_id}={left.root}, {right.component_id}={right.root}"
                )
    return components


def _parse_artifacts(
    registry: Mapping[str, Any], *, components: Sequence[Component]
) -> list[VersionedArtifact]:
    raw = registry.get("artifacts")
    if not isinstance(raw, list):
        raise LifecycleError("artifacts must be an array")
    component_roots = {component.root for component in components}
    identifiers = {component.component_id for component in components}
    paths: set[str] = set()
    artifacts: list[VersionedArtifact] = []
    for index, item in enumerate(raw):
        field = f"artifacts[{index}]"
        if not isinstance(item, dict):
            raise LifecycleError(f"{field} must be an object")
        unknown = sorted(set(item) - ARTIFACT_FIELDS)
        missing = sorted(ARTIFACT_REQUIRED_FIELDS - set(item))
        if unknown or missing:
            raise LifecycleError(
                f"{field} fields differ from schema: missing={missing}, unknown={unknown}"
            )
        artifact_id = _string(item["artifact_id"], field=f"{field}.artifact_id")
        if artifact_id in identifiers:
            raise LifecycleError(f"duplicate lifecycle identifier: {artifact_id}")
        identifiers.add(artifact_id)
        path = _exact_repo_path(item["path"], field=f"{field}.path")
        if path in paths:
            raise LifecycleError(f"duplicate versioned artifact path: {path}")
        if any(_path_is_within(path, root) for root in component_roots):
            raise LifecycleError(
                f"artifact is already covered by a component root: {path}"
            )
        paths.add(path)
        state = _string(item["state"], field=f"{field}.state")
        if state not in LIFECYCLE_STATES:
            raise LifecycleError(
                f"unknown lifecycle state for {artifact_id}: {state!r}"
            )
        git_mode_value = item.get("git_mode")
        git_mode = (
            _string(git_mode_value, field=f"{field}.git_mode")
            if git_mode_value is not None
            else None
        )
        pinned_states = {
            "frozen_replay",
            "quarantine",
            "deletion_candidate",
            "deleted",
        }
        if state in pinned_states and git_mode is None:
            raise LifecycleError(f"{state} artifact {artifact_id} requires git_mode")
        if git_mode is not None and git_mode not in ALLOWED_GIT_MODES:
            raise LifecycleError(
                f"artifact {artifact_id} has unsupported git_mode: {git_mode}"
            )
        if state not in pinned_states and git_mode is not None:
            raise LifecycleError(
                f"non-pinned artifact {artifact_id} may not declare git_mode"
            )
        owner = _string(item["owner"], field=f"{field}.owner")
        reason = _string(item["reason"], field=f"{field}.reason")
        replacement_value = item.get("replacement")
        if replacement_value is not None and not isinstance(replacement_value, str):
            raise LifecycleError(f"{field}.replacement must be a string or null")
        replacement = replacement_value.strip() if replacement_value else None
        mirror_value = item.get("mirror_of")
        if mirror_value is not None:
            mirror_of = _exact_repo_path(mirror_value, field=f"{field}.mirror_of")
            if mirror_of == path:
                raise LifecycleError(f"artifact may not mirror itself: {path}")
        else:
            mirror_of = None
        content_sha256 = _optional_sha256(
            item.get("content_sha256"), field=f"{field}.content_sha256"
        )
        evidence = tuple(
            _exact_repo_path(value, field=f"{field}.quarantine_evidence")
            for value in _string_list(
                item.get("quarantine_evidence", []),
                field=f"{field}.quarantine_evidence",
            )
        )
        deletion_approval_evidence = tuple(
            _exact_repo_path(
                value, field=f"{field}.deletion_approval_evidence"
            )
            for value in _string_list(
                item.get("deletion_approval_evidence", []),
                field=f"{field}.deletion_approval_evidence",
            )
        )
        quarantine_started_at = _optional_date(
            item.get("quarantine_started_at"),
            field=f"{field}.quarantine_started_at",
        )
        minimum_quarantine_days = _nonnegative_int(
            item.get("minimum_quarantine_days", 0),
            field=f"{field}.minimum_quarantine_days",
        )
        _validate_lifecycle_evidence(
            identifier=artifact_id,
            state=state,
            content_sha256=content_sha256,
            quarantine_evidence=evidence,
            quarantine_started_at=quarantine_started_at,
            minimum_quarantine_days=minimum_quarantine_days,
            deletion_approval_evidence=deletion_approval_evidence,
        )
        artifacts.append(
            VersionedArtifact(
                artifact_id=artifact_id,
                path=path,
                state=state,
                owner=owner,
                replacement=replacement,
                reason=reason,
                content_sha256=content_sha256,
                mirror_of=mirror_of,
                quarantine_evidence=evidence,
                quarantine_started_at=quarantine_started_at,
                minimum_quarantine_days=minimum_quarantine_days,
                deletion_approval_evidence=deletion_approval_evidence,
                git_mode=git_mode,
            )
        )
    valid_replacements = identifiers
    state_by_id = {
        component.component_id: component.state for component in components
    }
    state_by_id.update(
        {artifact.artifact_id: artifact.state for artifact in artifacts}
    )
    for artifact in artifacts:
        if artifact.replacement == artifact.artifact_id:
            raise LifecycleError(
                f"artifact cannot replace itself: {artifact.artifact_id}"
            )
        if artifact.replacement and artifact.replacement not in valid_replacements:
            raise LifecycleError(
                f"unknown replacement for {artifact.artifact_id}: {artifact.replacement}"
            )
        if artifact.state == "compatibility":
            if not artifact.replacement:
                raise LifecycleError(
                    "compatibility artifact requires an active replacement: "
                    f"{artifact.artifact_id}"
                )
            replacement_state = state_by_id[artifact.replacement]
            if replacement_state != "active":
                raise LifecycleError(
                    "compatibility artifact replacement must be active: "
                    f"{artifact.artifact_id} -> {artifact.replacement} "
                    f"({replacement_state})"
                )
    return artifacts


def _validate_registry_transition(
    baseline_registry: Mapping[str, Any],
    *,
    components: Sequence[Component],
    artifacts: Sequence[VersionedArtifact],
) -> None:
    if baseline_registry.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise LifecycleError(
            "baseline component lifecycle registry has an incompatible schema"
        )
    previous_components = _parse_components(baseline_registry)
    previous_artifacts = _parse_artifacts(
        baseline_registry, components=previous_components
    )

    previous: dict[str, tuple[str, str, Component | VersionedArtifact]] = {
        component.component_id: ("component", component.root, component)
        for component in previous_components
    }
    previous.update(
        {
            artifact.artifact_id: ("artifact", artifact.path, artifact)
            for artifact in previous_artifacts
        }
    )
    current: dict[str, tuple[str, str, Component | VersionedArtifact]] = {
        component.component_id: ("component", component.root, component)
        for component in components
    }
    current.update(
        {
            artifact.artifact_id: ("artifact", artifact.path, artifact)
            for artifact in artifacts
        }
    )

    for identifier, (previous_kind, previous_path, previous_record) in previous.items():
        current_value = current.get(identifier)
        if current_value is None:
            raise LifecycleError(
                "lifecycle identifier disappeared without a deleted tombstone: "
                f"{identifier}"
            )
        current_kind, current_path, current_record = current_value
        if current_kind != previous_kind:
            raise LifecycleError(
                f"lifecycle identifier changed kind: {identifier} "
                f"{previous_kind} -> {current_kind}"
            )
        if current_path != previous_path:
            raise LifecycleError(
                f"lifecycle identifier changed path: {identifier} "
                f"{previous_path} -> {current_path}"
            )
        if previous_record.state == "deleted":
            if current_record != previous_record:
                raise LifecycleError(
                    f"deleted lifecycle tombstone is immutable: {identifier}"
                )
            continue
        if previous_record.state == "frozen_replay":
            if current_record.state == "frozen_replay":
                if current_record != previous_record:
                    raise LifecycleError(
                        f"frozen_replay lifecycle record is immutable: {identifier}"
                    )
                continue
            if current_record.state != "quarantine":
                raise LifecycleError(
                    "frozen_replay may transition only to quarantine: "
                    f"{identifier} -> {current_record.state}"
                )
            expected_quarantine = replace(
                previous_record,
                state="quarantine",
                quarantine_evidence=current_record.quarantine_evidence,
                quarantine_started_at=current_record.quarantine_started_at,
                minimum_quarantine_days=current_record.minimum_quarantine_days,
            )
            if current_record != expected_quarantine:
                raise LifecycleError(
                    "frozen_replay to quarantine must preserve metadata, mode, "
                    f"and content hash: {identifier}"
                )
            continue
        if current_record.state == "deletion_candidate":
            if previous_record.state == "deletion_candidate":
                if current_record != previous_record:
                    raise LifecycleError(
                        "deletion_candidate lifecycle record is immutable: "
                        f"{identifier}"
                    )
                continue
            if previous_record.state != "quarantine":
                raise LifecycleError(
                    "lifecycle entry may become deletion_candidate only from "
                    f"quarantine: {identifier} was {previous_record.state}"
                )
            expected_candidate = replace(
                previous_record,
                state="deletion_candidate",
            )
            if current_record != expected_candidate:
                raise LifecycleError(
                    "deletion_candidate transition must preserve quarantined "
                    f"metadata and content hash: {identifier}"
                )
            continue
        if previous_record.state in {"quarantine", "deletion_candidate"} and (
            current_record.state == previous_record.state
            and current_record != previous_record
        ):
            raise LifecycleError(
                f"{previous_record.state} lifecycle record is immutable: {identifier}"
            )
        if current_record.state == "deleted":
            if previous_record.state != "deletion_candidate":
                raise LifecycleError(
                    "lifecycle entry may become deleted only from "
                    f"deletion_candidate: {identifier} was {previous_record.state}"
                )
            if isinstance(previous_record, Component):
                expected_deleted: Component | VersionedArtifact = replace(
                    previous_record,
                    state="deleted",
                    tracked_policy="no_tracked_files",
                    deletion_approval_evidence=(
                        current_record.deletion_approval_evidence
                    ),
                )
            else:
                expected_deleted = replace(
                    previous_record,
                    state="deleted",
                    deletion_approval_evidence=(
                        current_record.deletion_approval_evidence
                    ),
                )
            if current_record != expected_deleted:
                raise LifecycleError(
                    "deleted transition must preserve candidate metadata and "
                    f"content hash: {identifier}"
                )


def _discover_component_files(root: Path) -> list[Path]:
    candidates = [root] if root.is_file() or root.is_symlink() else root.rglob("*")
    return sorted(
        path
        for path in candidates
        if (path.is_file() or path.is_symlink())
        and path.name not in IGNORED_FILE_NAMES
        and path.suffix not in IGNORED_FILE_SUFFIXES
        and not (set(path.parts) & IGNORED_PATH_PARTS)
    )


def _repo_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise LifecycleError(f"path escapes repository: {path}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_content_sha256(
    *,
    root: str,
    repo_root: Path,
    tracked: set[str],
    index_modes: Mapping[str, str] | None = None,
) -> str:
    paths = sorted(path for path in tracked if _path_is_within(path, root))
    if not paths:
        raise LifecycleError(f"cannot hash empty tracked component root: {root}")
    digest = hashlib.sha256()
    for path in paths:
        physical = repo_root / path
        if not physical.is_file() or physical.is_symlink():
            raise LifecycleError(f"cannot hash non-regular tracked file: {path}")
        mode = (
            index_modes.get(path)
            if index_modes is not None
            else ("100755" if physical.stat().st_mode & 0o111 else "100644")
        )
        if mode not in ALLOWED_GIT_MODES:
            raise LifecycleError(
                f"cannot hash unsupported tracked mode: {path}={mode}"
            )
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(mode.encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(physical)))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_paths_and_tracking(
    components: Sequence[Component],
    *,
    repo_root: Path,
    tracked: set[str],
    index_modes: Mapping[str, str] | None = None,
) -> None:
    required_tracked_states = {
        "active",
        "compatibility",
        "frozen_replay",
        "quarantine",
        "deletion_candidate",
    }
    for component in components:
        root = repo_root / component.root
        tracked_under_root = {
            path for path in tracked if _path_is_within(path, component.root)
        }
        if component.state == "deleted":
            if component.tracked_policy != "no_tracked_files":
                raise LifecycleError(
                    f"deleted component {component.component_id} must use "
                    "no_tracked_files"
                )
            if tracked_under_root:
                raise LifecycleError(
                    f"deleted component {component.component_id} still has tracked "
                    f"files: {sorted(tracked_under_root)}"
                )
            if root.exists() or root.is_symlink():
                raise LifecycleError(
                    f"deleted component root still exists: "
                    f"{component.component_id}={component.root}"
                )
            continue
        if not root.exists() and not root.is_symlink():
            # ``local_only`` records a developer-worktree component precisely so
            # tracked code cannot depend on it.  A clean checkout is expected to
            # omit that root entirely.
            if component.state == "local_only":
                if any(_path_is_within(path, component.root) for path in tracked):
                    raise LifecycleError(
                        f"absent local_only root still has tracked files: "
                        f"{component.component_id}"
                    )
                continue
            raise LifecycleError(
                f"component root does not exist: {component.component_id}={component.root}"
            )
        for path_text in (*component.entrypoints, *component.sources):
            path = repo_root / path_text
            if not path.is_file() and not path.is_symlink():
                raise LifecycleError(
                    f"component source/entrypoint does not exist: {path_text}"
                )
        if component.state in {"active", "compatibility", "frozen_replay"}:
            invalid_modes = {
                path: index_modes[path]
                for path in sorted(tracked_under_root)
                if index_modes is not None
                and index_modes.get(path) not in ALLOWED_GIT_MODES
            }
            if invalid_modes:
                raise LifecycleError(
                    f"{component.component_id} contains unsupported git modes: "
                    f"{invalid_modes}"
                )
        if component.tracked_policy == "all_tracked":
            files = {
                _repo_path(path, repo_root)
                for path in _discover_component_files(root)
            }
            missing = sorted(files - tracked)
            if missing:
                raise LifecycleError(
                    f"{component.component_id} requires all files tracked: {missing}"
                )
            absent = sorted(tracked_under_root - files)
            if absent:
                raise LifecycleError(
                    f"{component.component_id} has tracked files absent from worktree: "
                    f"{absent}"
                )
        else:
            # A local-only component may contain a virtual environment, vendor
            # cache, build tree, or nested Git repository.  Its lifecycle gate
            # is the Git index boundary, so do not recursively inventory local
            # contents after checking the declared source/entrypoint anchors.
            if tracked_under_root:
                raise LifecycleError(
                    f"{component.component_id} requires no tracked files: "
                    f"{sorted(tracked_under_root)}"
                )

        if component.state in required_tracked_states and (
            component.tracked_policy != "all_tracked"
        ):
            raise LifecycleError(
                f"{component.state} component {component.component_id} must be all_tracked"
            )
        if component.state == "local_only" and (
            component.tracked_policy != "no_tracked_files"
        ):
            raise LifecycleError(
                f"local_only component {component.component_id} must use no_tracked_files"
            )
        if component.state in {
            "frozen_replay",
            "quarantine",
            "deletion_candidate",
        }:
            actual_digest = _tree_content_sha256(
                root=component.root,
                repo_root=repo_root,
                tracked=tracked,
                index_modes=index_modes,
            )
            if actual_digest != component.content_sha256:
                raise LifecycleError(
                    f"{component.state} content hash mismatch for "
                    f"{component.component_id}: expected={component.content_sha256}, "
                    f"actual={actual_digest}"
                )


def _validate_component_links(components: Sequence[Component]) -> None:
    by_id = {component.component_id: component for component in components}
    for component in components:
        if component.replacement == component.component_id:
            raise LifecycleError(f"component cannot replace itself: {component.component_id}")
        if component.replacement and component.replacement not in by_id:
            raise LifecycleError(
                f"unknown replacement for {component.component_id}: {component.replacement}"
            )
        if component.state == "compatibility":
            if not component.replacement:
                raise LifecycleError(
                    "compatibility component requires an active replacement: "
                    f"{component.component_id}"
                )
            replacement_state = by_id[component.replacement].state
            if replacement_state != "active":
                raise LifecycleError(
                    "compatibility component replacement must be active: "
                    f"{component.component_id} -> {component.replacement} "
                    f"({replacement_state})"
                )
        for dependency in component.dependencies:
            if dependency == component.component_id:
                raise LifecycleError(
                    f"component cannot depend on itself: {component.component_id}"
                )
            if dependency not in by_id:
                raise LifecycleError(
                    f"unknown dependency for {component.component_id}: {dependency}"
                )

    def visit(origin: Component, current: Component, chain: tuple[str, ...]) -> None:
        for dependency_id in current.dependencies:
            if dependency_id in chain:
                cycle = " -> ".join((*chain, dependency_id))
                raise LifecycleError(f"component dependency cycle: {cycle}")
            target = by_id[dependency_id]
            if (
                origin.state in RUNNABLE_STATES
                and target.state in UNSAFE_DEPENDENCY_STATES
            ):
                path = " -> ".join((*chain, dependency_id))
                raise LifecycleError(
                    f"runnable component {origin.component_id} reaches unsafe "
                    f"{target.state} dependency: {path}"
                )
            visit(origin, target, (*chain, dependency_id))

    for component in components:
        visit(component, component, (component.component_id,))


def _validate_active_bundle_coverage(
    components: Sequence[Component], active_manifest: Mapping[str, Any]
) -> int:
    bundles = active_manifest.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise LifecycleError("active generation bundle manifest has no bundles")
    by_root = {component.root: component for component in components}
    covered = 0
    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, dict):
            raise LifecycleError(f"active bundle[{index}] must be an object")
        root = _exact_repo_path(bundle.get("root"), field=f"active bundle[{index}].root")
        component = by_root.get(root)
        if component is None:
            raise LifecycleError(f"active bundle root is absent from registry: {root}")
        if component.state not in {"active", "compatibility"}:
            raise LifecycleError(
                f"active bundle {root} has non-runnable lifecycle state {component.state}"
            )
        raw_entrypoints = bundle.get("entrypoints")
        if not isinstance(raw_entrypoints, list) or any(
            not isinstance(path, str) for path in raw_entrypoints
        ):
            raise LifecycleError(f"active bundle entrypoints are invalid: {root}")
        expected = {
            f"{root}/{PurePosixPath(path).as_posix()}" for path in raw_entrypoints
        }
        absent = sorted(expected - set(component.entrypoints))
        if absent:
            raise LifecycleError(
                f"active bundle entrypoints lack lifecycle coverage for {root}: {absent}"
            )
        covered += 1
    return covered


def _validate_versioned_artifacts(
    artifacts: Sequence[VersionedArtifact],
    *,
    repo_root: Path,
    tracked: set[str],
    index_modes: Mapping[str, str] | None,
) -> None:
    for artifact in artifacts:
        physical = repo_root / artifact.path
        if artifact.state == "deleted":
            if artifact.path in tracked:
                raise LifecycleError(
                    f"deleted artifact is still tracked: {artifact.path}"
                )
            if physical.exists() or physical.is_symlink():
                raise LifecycleError(
                    f"deleted artifact path still exists: {artifact.path}"
                )
            continue
        if artifact.path not in tracked:
            raise LifecycleError(
                f"versioned artifact is not tracked: {artifact.path}"
            )
        if not physical.is_file() or physical.is_symlink():
            raise LifecycleError(
                f"versioned artifact is not a regular file: {artifact.path}"
            )
        actual_mode = (
            index_modes.get(artifact.path)
            if index_modes is not None
            else ("100755" if physical.stat().st_mode & 0o111 else "100644")
        )
        if actual_mode not in ALLOWED_GIT_MODES:
            raise LifecycleError(
                f"versioned artifact has unsupported git mode: "
                f"{artifact.path}={actual_mode}"
            )
        if artifact.state in {
            "frozen_replay",
            "quarantine",
            "deletion_candidate",
        } and actual_mode != artifact.git_mode:
            raise LifecycleError(
                f"{artifact.state} artifact git mode mismatch for "
                f"{artifact.artifact_id}: expected={artifact.git_mode}, "
                f"actual={actual_mode}"
            )
        actual_digest = _file_sha256(physical)
        if (
            artifact.state in {
                "frozen_replay",
                "quarantine",
                "deletion_candidate",
            }
            and actual_digest != artifact.content_sha256
        ):
            raise LifecycleError(
                f"{artifact.state} artifact hash mismatch for "
                f"{artifact.artifact_id}: "
                f"expected={artifact.content_sha256}, actual={actual_digest}"
            )
        if artifact.mirror_of is not None:
            if artifact.mirror_of not in tracked:
                raise LifecycleError(
                    f"artifact mirror source is not tracked: {artifact.mirror_of}"
                )
            mirror = repo_root / artifact.mirror_of
            if not mirror.is_file() or mirror.is_symlink():
                raise LifecycleError(
                    f"artifact mirror source is not a regular file: "
                    f"{artifact.mirror_of}"
                )
            if actual_digest != _file_sha256(mirror):
                raise LifecycleError(
                    f"packaged artifact drift for {artifact.path}; "
                    f"mirror_of={artifact.mirror_of}"
                )


def _validate_version_path_coverage(
    components: Sequence[Component],
    artifacts: Sequence[VersionedArtifact],
    *,
    tracked: set[str],
) -> int:
    artifact_paths = {artifact.path for artifact in artifacts}
    uncovered = sorted(
        path
        for path in tracked
        if VERSION_PATH_PATTERN.search(path)
        and path not in artifact_paths
        and not any(_path_is_within(path, component.root) for component in components)
    )
    if uncovered:
        raise LifecycleError(
            f"tracked versioned paths lack lifecycle classification: {uncovered}"
        )
    return sum(
        VERSION_PATH_PATTERN.search(path) is not None for path in tracked
    )


def _module_candidates(module: str) -> tuple[str, ...]:
    relative = module.replace(".", "/")
    if module == "tools" or module.startswith("tools."):
        base = relative
    else:
        base = f"src/{relative}"
    return (f"{base}.py", f"{base}/__init__.py")


def _imported_modules(tree: ast.AST) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def _path_chain(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return [*_path_chain(node.left), *_path_chain(node.right)]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"Path", "PurePath", "PurePosixPath"}
    ):
        result: list[str] = []
        for argument in node.args:
            result.extend(_path_chain(argument))
        return result
    return []


def _literal_tools_paths(tree: ast.AST) -> Iterator[str]:
    seen: set[str] = set()
    for node in ast.walk(tree):
        chain = _path_chain(node)
        segments: list[str] = []
        for part in chain:
            segments.extend(PurePosixPath(part).parts)
        if "tools" not in segments:
            continue
        index = segments.index("tools")
        candidate = PurePosixPath(*segments[index:]).as_posix()
        seen.add(candidate)
    # ``ast.walk`` also visits every prefix of a chained ``Path(...) / ...``
    # expression.  Report only maximal paths so one dependency has one finding.
    for candidate in sorted(seen):
        if not any(
            other != candidate and other.startswith(f"{candidate}/")
            for other in seen
        ):
            yield candidate


def _module_tracking_problem(
    module: str, *, repo_root: Path, tracked: set[str]
) -> str | None:
    candidates = _module_candidates(module)
    existing = [path for path in candidates if (repo_root / path).is_file()]
    if existing:
        untracked = [path for path in existing if path not in tracked]
        if untracked and not any(path in tracked for path in existing):
            return f"{module} resolves only to untracked files {untracked}"
        return None

    package_dir = candidates[1].removesuffix("/__init__.py")
    physical_dir = repo_root / package_dir
    if physical_dir.is_dir():
        tracked_members = [path for path in tracked if _path_is_within(path, package_dir)]
        if not tracked_members:
            return f"{module} resolves only to untracked namespace {package_dir}"
        return None
    if module == "tools" or module.startswith("tools."):
        return f"{module} has no repository implementation"
    return None


def tracked_test_import_problems(
    *, repo_root: Path, tracked: set[str]
) -> list[str]:
    """Find imports/path loads where a tracked test relies on local-only code."""

    problems: list[str] = []
    test_paths = sorted(
        path
        for path in tracked
        if path.startswith("tests/")
        and PurePosixPath(path).name.startswith("test_")
        and path.endswith(".py")
    )
    for test_path in test_paths:
        physical = repo_root / test_path
        if not physical.is_file():
            problems.append(f"tracked test is absent from worktree: {test_path}")
            continue
        try:
            tree = ast.parse(physical.read_text(encoding="utf-8"), filename=test_path)
        except (OSError, UnicodeError, SyntaxError) as exc:
            problems.append(f"cannot statically inspect tracked test {test_path}: {exc}")
            continue
        for module in sorted(set(_imported_modules(tree))):
            if module != "tools" and not module.startswith("tools."):
                continue
            detail = _module_tracking_problem(module, repo_root=repo_root, tracked=tracked)
            if detail:
                problems.append(f"{test_path}: {detail}")
        for candidate in sorted(set(_literal_tools_paths(tree))):
            physical_candidate = repo_root / candidate
            if not physical_candidate.exists():
                continue
            if physical_candidate.is_file():
                has_tracked = candidate in tracked
            else:
                has_tracked = any(_path_is_within(path, candidate) for path in tracked)
            if not has_tracked:
                problems.append(
                    f"{test_path}: tools path exists only outside Git index: {candidate}"
                )
    return sorted(set(problems))


def _component_for_repo_path(
    path: str, components: Sequence[Component]
) -> Component | None:
    for component in components:
        if _path_is_within(path, component.root):
            return component
    return None


def _validate_discovered_runnable_dependencies(
    components: Sequence[Component], *, repo_root: Path, tracked: set[str]
) -> None:
    by_id = {component.component_id: component for component in components}
    for component in components:
        if component.state not in RUNNABLE_STATES:
            continue
        declared = set(component.dependencies)
        python_paths = sorted(
            path
            for path in tracked
            if path.endswith(".py") and _path_is_within(path, component.root)
        )
        for path in python_paths:
            try:
                tree = ast.parse(
                    (repo_root / path).read_text(encoding="utf-8"), filename=path
                )
            except (OSError, UnicodeError, SyntaxError) as exc:
                raise LifecycleError(f"cannot inspect active source {path}: {exc}") from exc
            for module in _imported_modules(tree):
                for candidate in _module_candidates(module):
                    target = _component_for_repo_path(candidate, components)
                    if target is None or target.component_id == component.component_id:
                        continue
                    if target.component_id not in declared:
                        raise LifecycleError(
                            f"runnable component {component.component_id} has undeclared "
                            f"import dependency {target.component_id} via {path}: {module}"
                        )
                    if target.state in UNSAFE_DEPENDENCY_STATES:
                        raise LifecycleError(
                            f"runnable component {component.component_id} imports unsafe "
                            f"{target.state} component {target.component_id} via {path}"
                        )
                    break
            for candidate in _literal_tools_paths(tree):
                target = _component_for_repo_path(candidate, components)
                if target is None or target.component_id == component.component_id:
                    continue
                if target.component_id not in declared:
                    raise LifecycleError(
                        f"runnable component {component.component_id} has undeclared "
                        f"path dependency {target.component_id} via {path}: {candidate}"
                    )
                if target.state in UNSAFE_DEPENDENCY_STATES:
                    raise LifecycleError(
                        f"runnable component {component.component_id} loads unsafe "
                        f"{target.state} component {target.component_id} via {path}"
                    )
        # Keep this access explicit so a typo in dependencies is caught even when
        # a component currently contains no Python source.
        for dependency in declared:
            _ = by_id[dependency]


def _text_mentions_component(text: str, component: Component) -> bool:
    needles = {component.root}
    path = PurePosixPath(component.root)
    if path.parts and path.parts[0] == "tools":
        needles.add(".".join(path.parts))
    elif path.parts and path.parts[0] == "src":
        needles.add(".".join(path.parts[1:]))
    return any(needle and needle in text for needle in needles)


def _inbound_references(
    candidate: Component,
    *,
    components: Sequence[Component],
    repo_root: Path,
    tracked: set[str],
    registry_path: Path,
) -> list[str]:
    inbound: set[str] = set()
    for component in components:
        if component.component_id != candidate.component_id and (
            candidate.component_id in component.dependencies
            or component.replacement == candidate.component_id
        ):
            inbound.add(f"registry:{component.component_id}")

    excluded = {
        _repo_path(registry_path, repo_root),
        *candidate.quarantine_evidence,
        *candidate.deletion_approval_evidence,
    }
    for path in sorted(tracked):
        if path in excluded or _path_is_within(path, candidate.root):
            continue
        if PurePosixPath(path).suffix not in TEXT_SUFFIXES:
            continue
        physical = repo_root / path
        if not physical.is_file() or physical.is_symlink():
            continue
        try:
            text = physical.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if _text_mentions_component(text, candidate):
            inbound.add(path)
    return sorted(inbound)


def _validate_deletion_candidates(
    components: Sequence[Component],
    *,
    repo_root: Path,
    tracked: set[str],
    registry_path: Path,
    today: date,
) -> int:
    count = 0
    for component in components:
        if component.state not in {"quarantine", "deletion_candidate", "deleted"}:
            continue
        _validate_quarantine_period_and_evidence(
            identifier=component.component_id,
            target_path=component.root,
            evidence_paths=component.quarantine_evidence,
            quarantine_started_at=component.quarantine_started_at,
            minimum_quarantine_days=component.minimum_quarantine_days,
            repo_root=repo_root,
            tracked=tracked,
            today=today,
            require_elapsed=component.state in {"deletion_candidate", "deleted"},
        )
        if component.state == "deleted":
            _validate_deletion_approval_evidence(
                identifier=component.component_id,
                target_path=component.root,
                evidence_paths=component.deletion_approval_evidence,
                repo_root=repo_root,
                tracked=tracked,
            )
        if component.state not in {"deletion_candidate", "deleted"}:
            continue
        if component.state == "deletion_candidate":
            count += 1
        inbound = _inbound_references(
            component,
            components=components,
            repo_root=repo_root,
            tracked=tracked,
            registry_path=registry_path,
        )
        if inbound:
            raise LifecycleError(
                f"deletion candidate {component.component_id} has inbound references: {inbound}"
            )
    return count


def _validate_quarantine_period_and_evidence(
    *,
    identifier: str,
    target_path: str,
    evidence_paths: Sequence[str],
    quarantine_started_at: date | None,
    minimum_quarantine_days: int,
    repo_root: Path,
    tracked: set[str],
    today: date,
    require_elapsed: bool,
) -> None:
    if quarantine_started_at is None:
        raise LifecycleError(f"quarantine start date is missing: {identifier}")
    eligible_on = quarantine_started_at + timedelta(days=minimum_quarantine_days)
    if quarantine_started_at > today:
        raise LifecycleError(
            f"quarantine start date is in the future for {identifier}: "
            f"started_at={quarantine_started_at.isoformat()}, today={today.isoformat()}"
        )
    if require_elapsed and today < eligible_on:
        raise LifecycleError(
            f"quarantine period has not elapsed for {identifier}: "
            f"eligible_on={eligible_on.isoformat()}, today={today.isoformat()}"
        )
    for evidence in evidence_paths:
        path = repo_root / evidence
        if not path.is_file() or path.is_symlink():
            raise LifecycleError(
                f"quarantine evidence does not exist as a regular file for "
                f"{identifier}: {evidence}"
            )
        if evidence not in tracked:
            raise LifecycleError(
                f"quarantine evidence is not tracked for {identifier}: {evidence}"
            )
        try:
            evidence_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise LifecycleError(
                f"cannot read quarantine evidence for {identifier}: "
                f"{evidence}: {exc}"
            ) from exc
        if identifier not in evidence_text and target_path not in evidence_text:
            raise LifecycleError(
                f"quarantine evidence does not identify {identifier} or "
                f"{target_path}: {evidence}"
            )


def _validate_deletion_approval_evidence(
    *,
    identifier: str,
    target_path: str,
    evidence_paths: Sequence[str],
    repo_root: Path,
    tracked: set[str],
) -> None:
    for evidence in evidence_paths:
        path = repo_root / evidence
        if not path.is_file() or path.is_symlink():
            raise LifecycleError(
                "deletion approval evidence does not exist as a regular file for "
                f"{identifier}: {evidence}"
            )
        if evidence not in tracked:
            raise LifecycleError(
                f"deletion approval evidence is not tracked for "
                f"{identifier}: {evidence}"
            )
        try:
            evidence_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise LifecycleError(
                f"cannot read deletion approval evidence for {identifier}: "
                f"{evidence}: {exc}"
            ) from exc
        normalized = evidence_text.casefold()
        if identifier not in evidence_text and target_path not in evidence_text:
            raise LifecycleError(
                f"deletion approval evidence does not identify {identifier} or "
                f"{target_path}: {evidence}"
            )
        if "approved" not in normalized and "approval" not in normalized:
            raise LifecycleError(
                f"deletion approval evidence does not record approval for "
                f"{identifier}: {evidence}"
            )


def _validate_artifact_quarantine_and_deletion(
    artifacts: Sequence[VersionedArtifact],
    *,
    repo_root: Path,
    tracked: set[str],
    registry_path: Path,
    today: date,
) -> int:
    count = 0
    registry_repo_path = _repo_path(registry_path, repo_root)
    for artifact in artifacts:
        if artifact.state not in {"quarantine", "deletion_candidate", "deleted"}:
            continue
        _validate_quarantine_period_and_evidence(
            identifier=artifact.artifact_id,
            target_path=artifact.path,
            evidence_paths=artifact.quarantine_evidence,
            quarantine_started_at=artifact.quarantine_started_at,
            minimum_quarantine_days=artifact.minimum_quarantine_days,
            repo_root=repo_root,
            tracked=tracked,
            today=today,
            require_elapsed=artifact.state in {"deletion_candidate", "deleted"},
        )
        if artifact.state == "deleted":
            _validate_deletion_approval_evidence(
                identifier=artifact.artifact_id,
                target_path=artifact.path,
                evidence_paths=artifact.deletion_approval_evidence,
                repo_root=repo_root,
                tracked=tracked,
            )
        if artifact.state not in {"deletion_candidate", "deleted"}:
            continue
        if artifact.state == "deletion_candidate":
            count += 1
        excluded = {
            registry_repo_path,
            artifact.path,
            *artifact.quarantine_evidence,
            *artifact.deletion_approval_evidence,
        }
        inbound: list[str] = []
        needles = {artifact.artifact_id, artifact.path}
        for path in sorted(tracked - excluded):
            if PurePosixPath(path).suffix not in TEXT_SUFFIXES:
                continue
            physical = repo_root / path
            if not physical.is_file() or physical.is_symlink():
                continue
            try:
                text = physical.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if any(needle in text for needle in needles):
                inbound.append(path)
        if inbound:
            raise LifecycleError(
                f"deletion candidate {artifact.artifact_id} has inbound references: "
                f"{inbound}"
            )
    return count


def validate_registry(
    registry: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    tracked: Iterable[str] | None = None,
    index_modes: Mapping[str, str] | None = None,
    active_manifest: Mapping[str, Any] | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
    active_manifest_path: Path | None = None,
    baseline_registry: Mapping[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Validate registry structure and repository evidence without imports."""

    repo_root = repo_root.resolve()
    if tracked is None:
        resolved_modes = git_index_entries(repo_root)
        tracked_set = set(resolved_modes)
    else:
        tracked_set = set(tracked)
        resolved_modes = dict(index_modes) if index_modes is not None else None
    resolved_registry_path = registry_path.resolve()
    registry_repo_path = _repo_path(resolved_registry_path, repo_root)
    if registry_repo_path not in tracked_set:
        raise LifecycleError(
            f"component lifecycle registry is not tracked: {registry_repo_path}"
        )
    if active_manifest_path is not None:
        active_repo_path = _repo_path(active_manifest_path.resolve(), repo_root)
        if active_repo_path not in tracked_set:
            raise LifecycleError(
                f"active generation bundle manifest is not tracked: {active_repo_path}"
            )
    components = _parse_components(registry)
    artifacts = _parse_artifacts(registry, components=components)
    if baseline_registry is not None:
        _validate_registry_transition(
            baseline_registry,
            components=components,
            artifacts=artifacts,
        )
    _validate_paths_and_tracking(
        components,
        repo_root=repo_root,
        tracked=tracked_set,
        index_modes=resolved_modes,
    )
    _validate_component_links(components)
    covered = _validate_active_bundle_coverage(
        components,
        active_manifest if active_manifest is not None else load_active_bundles(),
    )
    test_problems = tracked_test_import_problems(repo_root=repo_root, tracked=tracked_set)
    if test_problems:
        raise LifecycleError(
            "tracked tests depend on untracked Python modules/paths: "
            + " | ".join(test_problems)
        )
    _validate_discovered_runnable_dependencies(
        components, repo_root=repo_root, tracked=tracked_set
    )
    _validate_versioned_artifacts(
        artifacts,
        repo_root=repo_root,
        tracked=tracked_set,
        index_modes=resolved_modes,
    )
    versioned_path_count = _validate_version_path_coverage(
        components, artifacts, tracked=tracked_set
    )
    resolved_today = today or datetime.now(timezone.utc).date()
    deletion_candidates = _validate_deletion_candidates(
        components,
        repo_root=repo_root,
        tracked=tracked_set,
        registry_path=resolved_registry_path,
        today=resolved_today,
    )
    deletion_candidates += _validate_artifact_quarantine_and_deletion(
        artifacts,
        repo_root=repo_root,
        tracked=tracked_set,
        registry_path=resolved_registry_path,
        today=resolved_today,
    )
    state_counts = {
        state: sum(component.state == state for component in components)
        for state in sorted(LIFECYCLE_STATES)
    }
    artifact_state_counts = {
        state: sum(artifact.state == state for artifact in artifacts)
        for state in sorted(LIFECYCLE_STATES)
    }
    return {
        "component_count": len(components),
        "versioned_artifact_count": len(artifacts),
        "classified_versioned_path_count": versioned_path_count,
        "active_bundle_count": covered,
        "deletion_candidate_count": deletion_candidates,
        "tracked_test_count": sum(
            path.startswith("tests/")
            and PurePosixPath(path).name.startswith("test_")
            and path.endswith(".py")
            for path in tracked_set
        ),
        "state_counts": state_counts,
        "artifact_state_counts": artifact_state_counts,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--active-bundles", type=Path, default=DEFAULT_ACTIVE_BUNDLES
    )
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        help="Explicit prior registry snapshot used to validate state transitions.",
    )
    parser.add_argument(
        "--baseline-ref",
        default="HEAD",
        help="Git ref used as the transition baseline when no snapshot is supplied.",
    )
    return parser.parse_args(argv)


def validate_index_snapshot(
    *,
    repo_root: Path = REPO_ROOT,
    registry_path: Path = DEFAULT_REGISTRY,
    active_manifest_path: Path = DEFAULT_ACTIVE_BUNDLES,
    baseline_registry: Mapping[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Validate exactly the Git index bytes, never unstaged worktree content."""

    repo_root = repo_root.resolve()
    records = _git_index_records(repo_root)
    entries = {path: mode for path, (mode, _object_id) in records.items()}
    registry_repo_path = _repo_path(registry_path.resolve(), repo_root)
    active_repo_path = _repo_path(active_manifest_path.resolve(), repo_root)
    for label, path in (
        ("component lifecycle registry", registry_repo_path),
        ("active generation bundle manifest", active_repo_path),
    ):
        if path not in entries:
            raise LifecycleError(f"{label} is not tracked in the Git index: {path}")

    with tempfile.TemporaryDirectory(prefix="layout-ddd-lifecycle-index-") as raw:
        snapshot = Path(raw).resolve()
        _materialize_index_blobs(
            repo_root=repo_root,
            snapshot=snapshot,
            records=records,
        )
        snapshot_registry = snapshot / registry_repo_path
        snapshot_active = snapshot / active_repo_path
        return validate_registry(
            load_registry(snapshot_registry),
            repo_root=snapshot,
            tracked=set(entries),
            index_modes=entries,
            active_manifest=load_active_bundles(snapshot_active),
            registry_path=snapshot_registry,
            active_manifest_path=snapshot_active,
            baseline_registry=baseline_registry,
            today=today,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        repo_root = REPO_ROOT.resolve()
        registry_path = args.registry.resolve()
        registry_repo_path = _repo_path(registry_path, repo_root)
        if args.baseline_registry is not None:
            baseline_registry = load_registry(args.baseline_registry.resolve())
        else:
            baseline_registry = _load_registry_from_git_ref(
                repo_root=repo_root,
                ref=str(args.baseline_ref),
                registry_repo_path=registry_repo_path,
            )
        summary = validate_index_snapshot(
            repo_root=repo_root,
            registry_path=registry_path,
            active_manifest_path=args.active_bundles,
            baseline_registry=baseline_registry,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "schema_version": EXPECTED_SCHEMA_VERSION,
                    **summary,
                },
                sort_keys=True,
            )
        )
        return 0
    except LifecycleError as exc:
        print(f"component lifecycle error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
