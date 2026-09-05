"""Source identity for the additive Agent-only generation track."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from benchmark.resources import runtime_resource_path


RUNTIME_VERSION = "1.0.0"


def agent_source_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix == ".pyc"
            or "__pycache__" in path.parts
        ):
            continue
        files.append(_record(path, path.relative_to(root).as_posix()))
    schema = runtime_resource_path(
        "schemas/non_rectangular_agent/submission_v1.schema.json"
    )
    files.append(
        _record(
            schema,
            "benchmark._resources/schemas/non_rectangular_agent/submission_v1.schema.json",
        )
    )
    payload = {
        "schema_version": "non_rectangular_agent_source_manifest_v1",
        "runtime_version": RUNTIME_VERSION,
        "logical_root": "benchmark.scene_generation.non_rectangular_agent",
        "files": sorted(files, key=lambda item: item["path"]),
    }
    return {**payload, "manifest_sha256": _sha256_mapping(payload)}


def _record(path: Path, logical: str) -> dict[str, Any]:
    return {
        "path": logical,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["RUNTIME_VERSION", "agent_source_manifest"]
