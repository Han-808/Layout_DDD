"""Artifact selection helpers for harness outputs that may be directories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


def read_json_source(path: Path, *, candidates: Iterable[str] = ()) -> tuple[object, Path]:
    source = Path(path)
    if source.is_file():
        return read_json(source), source
    if not source.is_dir():
        raise FileNotFoundError(f"harness output does not exist: {source}")
    for relative in candidates:
        candidate = source / relative
        if candidate.is_file():
            return read_json(candidate), candidate
    json_files = sorted(source.glob("*.json"))
    if len(json_files) == 1:
        return read_json(json_files[0]), json_files[0]
    raise ArtifactValidationError(
        f"harness output directory {source} does not contain any expected JSON artifact: "
        f"{list(candidates)}"
    )


def latest_numbered_json(directory: Path, *, prefix: str) -> Path:
    source = Path(directory)
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"harness output does not exist: {source}")
    pattern = re.compile(rf"^{re.escape(prefix)}_([0-9]+)\.json$")
    candidates: list[tuple[int, Path]] = []
    for candidate in source.glob(f"{prefix}_*.json"):
        match = pattern.match(candidate.name)
        if match:
            candidates.append((int(match.group(1)), candidate))
    if not candidates:
        nested = source / "record_scene"
        if nested.is_dir():
            return latest_numbered_json(nested, prefix=prefix)
        raise ArtifactValidationError(
            f"harness output directory {source} contains no {prefix}_<iteration>.json"
        )
    return max(candidates, key=lambda item: item[0])[1]


__all__ = ["latest_numbered_json", "read_json_source"]
