"""Source provenance for the shared frozen-two-stage compatibility runtime.

The source inventory is part of the migration contract documented in
``docs/generation_transport_compatibility.md``.  It intentionally contains no
runtime configuration, credentials, prompts, responses, or evaluator sources.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RUNTIME_ROOT = Path(__file__).resolve().parent


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def compatibility_source_manifest() -> dict[str, Any]:
    """Hash every Python source in the shared generation compatibility layer."""

    files = []
    for source_path in sorted(RUNTIME_ROOT.rglob("*.py")):
        if "__pycache__" in source_path.parts:
            continue
        data = source_path.read_bytes()
        files.append(
            {
                "path": source_path.relative_to(RUNTIME_ROOT).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    payload = {
        "schema_version": "frozen_two_stage_compatibility_source_manifest_v1",
        "root": "src/benchmark/scene_generation/frozen_two_stage",
        "architecture_document": "docs/generation_transport_compatibility.md",
        "files": files,
    }
    return {
        **payload,
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
    }
