"""Write-once and resumable artifacts for one Agent FloorPlan episode."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


class AgentArtifactError(RuntimeError):
    """Raised when an Agent episode tree is ambiguous or identity-drifted."""


class AgentEpisodeArtifacts:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().absolute()
        self.input_dir = self.root / "input"
        self.workspace = self.root / "workspace"
        self.agent_dir = self.root / "agent"
        self.attempts_dir = self.agent_dir / "attempts"
        self.normalized_dir = self.root / "normalized"
        self.run_manifest = self.root / "run_manifest.json"
        self.room_layout = self.input_dir / "room_layout.json"
        self.room_program = self.input_dir / "room_program.json"
        self.task_payload = self.input_dir / "task.json"
        self.task_prompt = self.workspace / "TASK.md"
        self.workspace_task = self.workspace / "task.json"
        self.workspace_layout = self.workspace / "room_layout.json"
        self.workspace_program = self.workspace / "room_program.json"
        self.final_submission = self.workspace / "final_submission.json"
        self.finalization = self.workspace / "finalization.json"
        self.tool_events = self.workspace / "tool_events.jsonl"
        self.object_plan = self.normalized_dir / "object_plan.json"
        self.asset_selection = self.normalized_dir / "asset_selection.json"
        self.global_placement = self.normalized_dir / "global_placement.json"
        self.generated_scene = self.normalized_dir / "generated_scene.json"
        self.compiled_architecture = self.normalized_dir / "compiled_architecture.json"
        self.evaluation_preflight = self.normalized_dir / "evaluation_preflight.json"
        self.submission_validation = self.normalized_dir / "submission_validation.json"
        self.summary = self.root / "summary.json"

    def initialize(
        self,
        *,
        run_manifest: Mapping[str, Any],
        room_layout: Mapping[str, Any],
        room_program: Mapping[str, Any],
        task_payload: Mapping[str, Any],
        task_prompt: str,
    ) -> None:
        if self.root.exists() or self.root.is_symlink():
            raise FileExistsError(f"Agent episode output already exists: {self.root}")
        self.root.mkdir(parents=True)
        self.input_dir.mkdir()
        self.workspace.mkdir()
        self.agent_dir.mkdir()
        self.attempts_dir.mkdir()
        self.normalized_dir.mkdir()
        write_json_exclusive(self.run_manifest, run_manifest)
        write_json_exclusive(self.room_layout, room_layout)
        write_json_exclusive(self.room_program, room_program)
        write_json_exclusive(self.task_payload, task_payload)
        write_json_exclusive(self.workspace_task, task_payload)
        write_json_exclusive(self.workspace_layout, room_layout)
        write_json_exclusive(self.workspace_program, room_program)
        with self.task_prompt.open("x", encoding="utf-8") as handle:
            handle.write(task_prompt)
            if not task_prompt.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def verify_resume(
        self,
        *,
        run_manifest: Mapping[str, Any],
        room_layout: Mapping[str, Any],
        room_program: Mapping[str, Any],
        task_payload: Mapping[str, Any],
        task_prompt: str,
    ) -> dict[str, Any] | None:
        if not self.root.is_dir() or self.root.is_symlink():
            raise AgentArtifactError("Agent episode root must be a real directory")
        for path in (
            self.input_dir,
            self.workspace,
            self.agent_dir,
            self.attempts_dir,
            self.normalized_dir,
            self.run_manifest,
            self.room_layout,
            self.room_program,
            self.task_payload,
            self.task_prompt,
        ):
            if not path.exists() or path.is_symlink():
                raise AgentArtifactError(f"resume artifact missing or linked: {path.name}")
        expected = (
            (self.run_manifest, run_manifest),
            (self.room_layout, room_layout),
            (self.room_program, room_program),
            (self.task_payload, task_payload),
        )
        for path, value in expected:
            if read_json(path) != dict(value):
                raise AgentArtifactError(f"resume identity mismatch: {path.name}")
        observed_prompt = self.task_prompt.read_text(encoding="utf-8")
        normalized_prompt = task_prompt if task_prompt.endswith("\n") else task_prompt + "\n"
        if observed_prompt != normalized_prompt:
            raise AgentArtifactError("resume task prompt identity mismatch")
        return read_json(self.summary) if self.summary.is_file() else None

    def next_attempt_dir(self) -> Path:
        existing: list[int] = []
        for path in self.attempts_dir.iterdir():
            if path.is_dir() and not path.is_symlink() and path.name.startswith("attempt_"):
                try:
                    existing.append(int(path.name.removeprefix("attempt_")))
                except ValueError:
                    continue
        target = self.attempts_dir / f"attempt_{max(existing, default=0) + 1:03d}"
        target.mkdir()
        return target


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentArtifactError(
            f"cannot read Agent artifact {resolved.name}: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise AgentArtifactError(f"Agent artifact {resolved.name} must be an object")
    return value


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with resolved.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_or_verify(path: str | Path, value: Mapping[str, Any]) -> None:
    resolved = Path(path)
    if resolved.exists():
        if not resolved.is_file() or resolved.is_symlink():
            raise AgentArtifactError(f"artifact is not a real file: {resolved.name}")
        if read_json(resolved) != dict(value):
            raise AgentArtifactError(f"artifact identity mismatch: {resolved.name}")
        return
    write_json_exclusive(resolved, value)


__all__ = [
    "AgentArtifactError",
    "AgentEpisodeArtifacts",
    "read_json",
    "write_json_exclusive",
    "write_json_or_verify",
]
