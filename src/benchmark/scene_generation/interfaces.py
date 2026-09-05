"""Implementation-independent interfaces for local scene generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class InputBatchInfo:
    """Validated immutable input metadata."""

    path: Path
    sha256: str
    scene_count: int
    first_scene_id: int
    last_scene_id: int


@dataclass(frozen=True)
class SceneGenerationResult:
    """Terminal result returned by a single-scene generation call."""

    scene_id: int
    status: str
    attempt_count: int
    stop_batch: bool
    output_root: Path


@dataclass(frozen=True)
class BatchGenerationResult:
    """Terminal summary returned by a batch generation call."""

    summary: dict[str, Any]
    stopped: bool
    output_root: Path


@runtime_checkable
class SceneGenerator(Protocol):
    """Local callable boundary for scene generators.

    Implementations own their prompt, request construction, retry policy, and
    artifacts. Callers provide only scene input and an output location.
    """

    def validate_input(self, input_jsonl: str | Path) -> InputBatchInfo:
        """Validate a batch without invoking a model."""

    def run_scene(
        self,
        *,
        scene_id: int,
        description: str,
        output_root: str | Path,
    ) -> SceneGenerationResult:
        """Generate one scene through the implementation's frozen protocol."""

    def run_batch(
        self,
        *,
        input_jsonl: str | Path,
        output_root: str | Path,
        resume: bool = False,
    ) -> BatchGenerationResult:
        """Generate a validated batch through the same frozen protocol."""
