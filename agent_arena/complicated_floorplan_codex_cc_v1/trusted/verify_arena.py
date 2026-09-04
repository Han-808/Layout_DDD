#!/usr/bin/env python3
"""Seal or verify the content-addressed standalone arena."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ARENA_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ARENA_ROOT / "arena.lock.json"
EXCLUDED_NAMES = frozenset({"arena.lock.json", ".DS_Store"})


class ArenaLockError(RuntimeError):
    pass


def controlled_files() -> list[Path]:
    files: list[Path] = []
    for path in ARENA_ROOT.rglob("*"):
        relative = path.relative_to(ARENA_ROOT)
        if not path.is_file() or path.name in EXCLUDED_NAMES:
            continue
        if relative.parts and relative.parts[0] == "episodes" and path.name != ".gitkeep":
            continue
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ArenaLockError(f"controlled file must not be a symlink: {relative}")
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ARENA_ROOT).as_posix())


def build_lock() -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for path in controlled_files():
        relative = path.relative_to(ARENA_ROOT).as_posix()
        payload = path.read_bytes()
        entries[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "mode": oct(path.stat().st_mode & 0o777),
        }
    canonical = json.dumps(
        entries, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": "sieve_agent_arena_lock_v1",
        "arena_id": "sieve-complicated-floorplan-coding-agents-v1",
        "file_count": len(entries),
        "files": entries,
        "content_root_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def write_lock() -> dict[str, Any]:
    if LOCK_PATH.exists() or LOCK_PATH.is_symlink():
        raise ArenaLockError("arena.lock.json already exists")
    value = build_lock()
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    LOCK_PATH.chmod(0o444)
    return value


def verify_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink():
        raise ArenaLockError("arena.lock.json is missing or linked")
    expected = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    observed = build_lock()
    if expected != observed:
        raise ArenaLockError("arena content differs from arena.lock.json")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    value = write_lock() if args.write_lock else verify_lock()
    from arena import verify_fixed_suite

    fixed = verify_fixed_suite()
    print(
        json.dumps(
            {
                "status": "valid",
                "content_root_sha256": value["content_root_sha256"],
                "controlled_file_count": value["file_count"],
                "scene_count": fixed["scene_count"],
                "room_count": fixed["room_count"],
                "wall_segment_count": fixed["wall_segment_count"],
                "database_snapshot_id": fixed["database_snapshot_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
