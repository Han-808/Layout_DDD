"""Path-free source identity for the shared retrieval runtime v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._common import canonical_json_bytes, sha256_bytes, sha256_file


RUNTIME_ROOT = Path(__file__).resolve().parent


def retrieval_source_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(RUNTIME_ROOT.glob("*.py")):
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": "generation_retrieval_source_manifest_v2",
        "logical_root": "benchmark.scene_generation.retrieval",
        "files": files,
    }
    return {
        **payload,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }
