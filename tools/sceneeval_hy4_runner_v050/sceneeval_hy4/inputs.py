"""Sanitized SceneEval-100 input loading."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .constants import EXPECTED_SCENE_IDS
from .strict_json import StrictJSONError, loads_strict


@dataclass(frozen=True)
class SceneInput:
    scene_id: int
    description: str


@dataclass(frozen=True)
class InputBatch:
    scenes: tuple[SceneInput, ...]
    exact_bytes: bytes
    sha256: str


class InputError(ValueError):
    pass


def load_human100_jsonl(path: Path) -> InputBatch:
    """Load exactly IDs 0..99 and reject every field except id/description."""
    exact_bytes = path.read_bytes()
    try:
        text = exact_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InputError(f"input is not valid UTF-8: {exc}") from exc

    lines = text.splitlines()
    if not lines:
        raise InputError("input JSONL is empty")

    scenes: list[SceneInput] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise InputError(f"line {line_number}: blank lines are not allowed")
        try:
            value = loads_strict(line)
        except StrictJSONError as exc:
            raise InputError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise InputError(f"line {line_number}: value must be an object")
        if set(value) != {"id", "description"}:
            raise InputError(
                f"line {line_number}: fields must be exactly 'id' and 'description'"
            )
        scene_id = value["id"]
        description = value["description"]
        if not isinstance(scene_id, int) or isinstance(scene_id, bool):
            raise InputError(f"line {line_number}: id must be an integer")
        if not isinstance(description, str) or not description.strip():
            raise InputError(f"line {line_number}: description must be non-empty text")
        scenes.append(SceneInput(scene_id=scene_id, description=description))

    ids = tuple(scene.scene_id for scene in scenes)
    if ids != EXPECTED_SCENE_IDS:
        raise InputError(
            "input must contain exactly 100 rows ordered by integer id 0 through 99; "
            f"received {len(ids)} rows with ids beginning {ids[:10]!r}"
        )

    return InputBatch(
        scenes=tuple(scenes),
        exact_bytes=exact_bytes,
        sha256=hashlib.sha256(exact_bytes).hexdigest(),
    )
