"""Write-once layout-level artifact paths and resume identity checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class NonRectangularGenerationArtifactError(RuntimeError):
    """Raised when output state is stale, partial, linked, or hash-drifted."""


@dataclass(frozen=True, slots=True)
class NonRectangularGenerationArtifacts:
    root: Path

    @property
    def run_manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def room_layout(self) -> Path:
        return self.root / "room_layout.json"

    @property
    def room_program(self) -> Path:
        return self.root / "room_program.json"

    @property
    def stage_a_dir(self) -> Path:
        return self.root / "stage_a"

    @property
    def stage_a_first_emission(self) -> Path:
        return self.stage_a_dir / "object_plan_first_emission.json"

    @property
    def object_plan(self) -> Path:
        return self.stage_a_dir / "object_plan.json"

    @property
    def object_plan_validation(self) -> Path:
        return self.stage_a_dir / "validation.json"

    @property
    def retrieval_dir(self) -> Path:
        return self.root / "retrieval"

    @property
    def retrieval_request(self) -> Path:
        return self.retrieval_dir / "request.json"

    @property
    def retrieval_results(self) -> Path:
        return self.retrieval_dir / "results.json"

    @property
    def asset_selection(self) -> Path:
        return self.retrieval_dir / "asset_selection.json"

    @property
    def stage_c_dir(self) -> Path:
        return self.root / "stage_c"

    @property
    def stage_c_input(self) -> Path:
        return self.stage_c_dir / "generation_input.json"

    @property
    def stage_c_first_emission(self) -> Path:
        return self.stage_c_dir / "placement_first_emission.json"

    @property
    def global_placement(self) -> Path:
        return self.stage_c_dir / "global_placement.json"

    @property
    def placement_validation(self) -> Path:
        return self.stage_c_dir / "validation.json"

    @property
    def generated_scene(self) -> Path:
        return self.root / "generated_scene.json"

    @property
    def compiled_architecture(self) -> Path:
        return self.root / "compiled_architecture.json"

    @property
    def evaluation_preflight(self) -> Path:
        return self.root / "evaluation_preflight.json"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    def initialize(
        self,
        *,
        core: Any,
        run_manifest: Mapping[str, Any],
        room_layout: Mapping[str, Any],
        room_program: Mapping[str, Any],
    ) -> None:
        if self.root.exists() or self.root.is_symlink():
            raise FileExistsError(f"generation output already exists: {self.root}")
        self.root.mkdir(parents=True, exist_ok=False)
        self.stage_a_dir.mkdir()
        self.retrieval_dir.mkdir()
        self.stage_c_dir.mkdir()
        core.write_json_exclusive(self.run_manifest, run_manifest)
        core.write_json_exclusive(self.room_layout, room_layout)
        core.write_json_exclusive(self.room_program, room_program)

    def verify_resume(
        self,
        *,
        run_manifest: Mapping[str, Any],
        room_layout: Mapping[str, Any],
        room_program: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not self.root.is_dir() or self.root.is_symlink():
            raise NonRectangularGenerationArtifactError(
                "resume root must be a real directory"
            )
        for path in (
            self.run_manifest,
            self.room_layout,
            self.room_program,
            self.stage_a_dir,
            self.retrieval_dir,
            self.stage_c_dir,
        ):
            if not path.exists() or path.is_symlink():
                raise NonRectangularGenerationArtifactError(
                    f"resume artifact missing or linked: {path.name}"
                )
        if _read_json(self.run_manifest) != dict(run_manifest):
            raise NonRectangularGenerationArtifactError(
                "resume run-manifest identity mismatch"
            )
        if _read_json(self.room_layout) != dict(room_layout):
            raise NonRectangularGenerationArtifactError(
                "resume room-layout identity mismatch"
            )
        if _read_json(self.room_program) != dict(room_program):
            raise NonRectangularGenerationArtifactError(
                "resume room-program identity mismatch"
            )
        return _read_json(self.summary) if self.summary.is_file() else None

    def reject_ambiguous_partial_stage(
        self,
        *,
        first_emission: Path,
        normalized_artifact: Path,
        stage: str,
    ) -> None:
        if first_emission.exists() and not normalized_artifact.exists():
            raise NonRectangularGenerationArtifactError(
                f"{stage} has an ambiguous captured emission and cannot be resent"
            )


def read_json(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NonRectangularGenerationArtifactError(
            f"cannot read artifact {path.name}: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise NonRectangularGenerationArtifactError(
            f"artifact {path.name} must be a JSON object"
        )
    return value


__all__ = [
    "NonRectangularGenerationArtifactError",
    "NonRectangularGenerationArtifacts",
    "read_json",
]
