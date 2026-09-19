#!/usr/bin/env python3
"""Task-level concurrent launcher for the Astra generation conditions.

Generation policy is owned entirely by the registered campaigns; this launcher
only decides *which* campaign invocations run and *how many* run at once. It
changes no prompt, route, retrieval profile, retry policy or artifact contract,
and it never edits a campaign registry.

Concurrency model
-----------------
``FrozenTwoStageOrchestrator.run`` iterates its briefs serially and asserts that
the executed brief order equals the campaign contract, so a campaign invocation
cannot be split across workers. Independent *invocations* are therefore the unit
of parallelism, one OS process each with its own output directory:

* ``open-space``     - 1 task (one campaign, 10 briefs serial inside)
* ``multi-room``     - 1 task per floor-plan layout
* ``complex-layout`` - not runnable from this repository (see ``_complex_tasks``)

Process-per-task isolation is deliberate: it keeps any per-run in-process cache
private to one task, so raising ``--workers`` cannot make two tasks share
mutable state.

Usage
-----
  scripts/run_astra_generation_v1.py --condition open-space --check
  scripts/run_astra_generation_v1.py --condition multi-room --workers 6 \
      --output-base Support/artifacts/outputs/e2e_multi_room/astra_xhigh_r1 --run
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
# A git worktree has no `.venv` of its own, so fall back to the interpreter that
# is running this launcher rather than refusing to start.
_REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON_BIN = _REPO_VENV_PYTHON if _REPO_VENV_PYTHON.is_file() else Path(sys.executable)

OPEN_SPACE_CAMPAIGN = "api2-gpt6-astra-xhigh-scene10-v1"
MULTI_ROOM_CAMPAIGN = "api2-gpt6-astra-xhigh-multi-room-v1"
COMPLEX_LAYOUT_CAMPAIGN = "api2-gpt6-astra-xhigh-nonrect-v1"

DEFAULT_FLOOR_PLAN_ROOT = (
    REPO_ROOT / "output" / "multi_room_generation_floorplans_v2" / "runner_inputs_v1"
)
DEFAULT_GENERATION_BINDINGS = REPO_ROOT / ".runtime" / "generation_bindings.local.json"
DEFAULT_RESOURCE_BINDINGS = REPO_ROOT / ".runtime" / "retrieval_bindings.local.json"

# One campaign invocation holds one embedding encoder and one in-flight request,
# so the ceiling is upstream rate limit rather than local CPU. These are the
# reviewed values; the parser rejects anything else.
ALLOWED_WORKERS = (1, 3, 6)
CONDITIONS = ("open-space", "multi-room", "complex-layout")


def _subprocess_env() -> dict[str, str]:
    """Bind children to THIS tree's source, not whatever is installed.

    Without this a worktree silently executes the main checkout's ``benchmark``
    package, so a campaign registered here would look unregistered.
    """

    env = os.environ.copy()
    source_root = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{source_root}{os.pathsep}{existing}" if existing else source_root
    return env


@dataclass(frozen=True, slots=True)
class Task:
    """One campaign invocation."""

    task_id: str
    condition: str
    campaign_id: str
    floor_plan: Path | None = None
    expected_units: int = 0

    def command(self, *, output_dir: Path, args: argparse.Namespace) -> list[str]:
        command = [
            str(PYTHON_BIN),
            "-m",
            "benchmark.scene_generation",
            "run",
            "--campaign",
            self.campaign_id,
            "--output-dir",
            str(output_dir),
            "--generation-bindings",
            str(args.generation_bindings),
            "--resource-bindings",
            str(args.resource_bindings),
        ]
        if self.floor_plan is not None:
            command += ["--floor-plan", str(self.floor_plan)]
        return command


@dataclass
class TaskOutcome:
    task_id: str
    condition: str
    state: str
    attempts: int = 0
    duration_seconds: float = 0.0
    output_dir: str = ""
    units_written: int = 0
    units_complete: int = 0
    detail: str = ""
    blocked_reason: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "state": self.state,
            "attempts": self.attempts,
            "duration_seconds": round(self.duration_seconds, 1),
            "output_dir": self.output_dir,
            "units_written": self.units_written,
            "units_complete": self.units_complete,
            "detail": self.detail,
            "blocked_reason": self.blocked_reason,
        }


def _open_space_tasks() -> list[Task]:
    return [
        Task(
            task_id="open-space.scene10",
            condition="open-space",
            campaign_id=OPEN_SPACE_CAMPAIGN,
            expected_units=10,
        )
    ]


def _multi_room_tasks(floor_plan_root: Path) -> list[Task]:
    if not floor_plan_root.is_dir():
        raise FileNotFoundError(
            f"multi-room floor-plan root does not exist: {floor_plan_root}"
        )
    tasks: list[Task] = []
    for layout_dir in sorted(floor_plan_root.iterdir()):
        if not layout_dir.is_dir() or not layout_dir.name.startswith("layout_"):
            continue
        floor_plan = layout_dir / "floor_plan.json"
        if not floor_plan.is_file():
            continue
        tasks.append(
            Task(
                task_id=f"multi-room.{layout_dir.name}",
                condition="multi-room",
                campaign_id=MULTI_ROOM_CAMPAIGN,
                floor_plan=floor_plan,
            )
        )
    if not tasks:
        raise FileNotFoundError(
            f"no layout_*/floor_plan.json inputs under {floor_plan_root}"
        )
    return tasks


def _complex_tasks() -> list[Task]:
    # The non-rectangular generator is not part of this repository's `src/`: it
    # lives in the frozen release `floorplan_api_complexity_v1`, whose runner
    # hash-verifies every file against `release.lock.json`. Registering Astra
    # there means building a NEW release, which this launcher deliberately does
    # not do. Surfaced as a blocked task rather than skipped silently.
    return [
        Task(
            task_id="complex-layout.blocked",
            condition="complex-layout",
            campaign_id=COMPLEX_LAYOUT_CAMPAIGN,
        )
    ]


def build_tasks(conditions: Iterable[str], floor_plan_root: Path) -> list[Task]:
    tasks: list[Task] = []
    for condition in conditions:
        if condition == "open-space":
            tasks += _open_space_tasks()
        elif condition == "multi-room":
            tasks += _multi_room_tasks(floor_plan_root)
        elif condition == "complex-layout":
            tasks += _complex_tasks()
        else:  # pragma: no cover - argparse restricts the choices
            raise ValueError(f"unsupported condition: {condition!r}")
    return tasks


def _unit_counts(output_dir: Path) -> tuple[int, int]:
    """Return (terminalized units, units whose own status is complete)."""

    results = sorted(output_dir.glob("*/case.result.json"))
    complete = 0
    for path in results:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("status") == "complete":
            complete += 1
    return len(results), complete


def _already_finished(output_dir: Path) -> bool:
    return (output_dir / "run_manifest.json").is_file()


def _check_task(task: Task, args: argparse.Namespace) -> TaskOutcome:
    """Validate registration and inputs without network or generation."""

    outcome = TaskOutcome(task_id=task.task_id, condition=task.condition, state="")
    if task.condition == "complex-layout":
        outcome.state = "blocked"
        outcome.blocked_reason = (
            "non-rectangular generation lives in the frozen release "
            "floorplan_api_complexity_v1; adding Astra requires building a new "
            "release (build_release.py --destination NEW), not editing one"
        )
        return outcome
    command = [
        str(PYTHON_BIN),
        "-m",
        "benchmark.scene_generation",
        "check",
        "--campaign",
        task.campaign_id,
    ]
    if task.floor_plan is not None:
        command += ["--floor-plan", str(task.floor_plan)]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        outcome.state = "ready"
    else:
        outcome.state = "not_ready"
        outcome.detail = (completed.stdout + completed.stderr).strip().splitlines()[-1:][
            0
        ] if (completed.stdout or completed.stderr) else f"rc={completed.returncode}"
    return outcome


def _run_task(task: Task, args: argparse.Namespace, output_base: Path) -> TaskOutcome:
    outcome = TaskOutcome(task_id=task.task_id, condition=task.condition, state="")
    if task.condition == "complex-layout":
        return _check_task(task, args)

    output_dir = output_base / task.task_id.replace(".", "/")
    outcome.output_dir = str(output_dir)
    if _already_finished(output_dir):
        if not args.resume:
            outcome.state = "refused"
            outcome.detail = (
                "output directory already holds a terminal run_manifest.json; "
                "pass --resume to skip finished tasks"
            )
            return outcome
        written, complete = _unit_counts(output_dir)
        outcome.state = "resumed_complete"
        outcome.units_written = written
        outcome.units_complete = complete
        return outcome

    log_dir = output_base / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for attempt in range(1, args.max_attempts + 1):
        outcome.attempts = attempt
        if output_dir.exists() and not _already_finished(output_dir):
            # The campaign CLI owns its output directory and refuses to reuse a
            # partial one; a retry therefore needs a fresh sibling.
            output_dir = output_base / f"{task.task_id.replace('.', '/')}_attempt_{attempt:02d}"
            outcome.output_dir = str(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            task.command(output_dir=output_dir, args=args),
            cwd=REPO_ROOT,
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        stem = f"{task.task_id.replace('.', '_')}_attempt_{attempt:02d}"
        (log_dir / f"{stem}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (log_dir / f"{stem}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if _already_finished(output_dir):
            written, complete = _unit_counts(output_dir)
            outcome.units_written = written
            outcome.units_complete = complete
            # rc 2 means the campaign terminalized with failed units; that is a
            # recorded scientific outcome, not an infrastructure failure.
            outcome.state = "complete" if completed.returncode == 0 else "terminal_with_failures"
            outcome.detail = f"rc={completed.returncode}"
            break
        outcome.state = "failed"
        outcome.detail = f"rc={completed.returncode}; no run_manifest.json written"
        if attempt < args.max_attempts:
            time.sleep(args.retry_delay_seconds)
    outcome.duration_seconds = time.time() - started
    return outcome


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        dest="conditions",
        choices=CONDITIONS,
        help="repeatable; defaults to every condition",
    )
    parser.add_argument("--workers", type=int, choices=ALLOWED_WORKERS, default=3)
    parser.add_argument("--output-base", type=Path)
    parser.add_argument(
        "--generation-bindings", type=Path, default=DEFAULT_GENERATION_BINDINGS
    )
    parser.add_argument(
        "--resource-bindings", type=Path, default=DEFAULT_RESOURCE_BINDINGS
    )
    parser.add_argument("--floor-plan-root", type=Path, default=DEFAULT_FLOOR_PLAN_ROOT)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="read-only readiness check (the default) with no network or generation",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="explicit live generation; default is a read-only readiness check",
    )
    args = parser.parse_args(argv)
    if args.conditions is None:
        args.conditions = list(CONDITIONS)
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.run and args.output_base is None:
        parser.error("--run requires --output-base")
    # A git worktree has no `.runtime/` of its own, so the default paths can point
    # at files that do not exist. Fail here with the path rather than letting the
    # campaign CLI surface a redacted contract error per task.
    for label, path in (
        ("--generation-bindings", args.generation_bindings),
        ("--resource-bindings", args.resource_bindings),
    ):
        if not Path(path).expanduser().is_file():
            parser.error(f"{label} does not exist: {path}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not PYTHON_BIN.is_file():
        print(f"interpreter is missing: {PYTHON_BIN}", file=sys.stderr)
        return 2
    try:
        tasks = build_tasks(args.conditions, args.floor_plan_root.expanduser().resolve())
    except (FileNotFoundError, ValueError) as exc:
        print(f"task planning failed: {exc}", file=sys.stderr)
        return 2

    runnable = [task for task in tasks if task.condition != "complex-layout"]
    workers = min(args.workers, max(1, len(runnable)))

    if not args.run:
        outcomes = [_check_task(task, args) for task in tasks]
        print(
            json.dumps(
                {
                    "schema_version": "astra_generation_launch_plan_v1",
                    "mode": "check",
                    "conditions": args.conditions,
                    "requested_workers": args.workers,
                    "effective_workers": workers,
                    "task_count": len(tasks),
                    "tasks": [outcome.public_dict() for outcome in outcomes],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if all(o.state in {"ready", "blocked"} for o in outcomes) else 1

    output_base = args.output_base.expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    started = time.time()
    outcomes: list[TaskOutcome] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_task, task, args, output_base): task for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:  # one task's crash must not stop the batch
                outcome = TaskOutcome(
                    task_id=task.task_id,
                    condition=task.condition,
                    state="failed",
                    detail=f"{type(exc).__name__}",
                )
            outcomes.append(outcome)
            print(
                f"[{outcome.state}] {outcome.task_id} "
                f"units={outcome.units_complete}/{outcome.units_written} "
                f"{outcome.duration_seconds:.0f}s",
                flush=True,
            )

    outcomes.sort(key=lambda item: item.task_id)
    states: dict[str, int] = {}
    for outcome in outcomes:
        states[outcome.state] = states.get(outcome.state, 0) + 1
    manifest = {
        "schema_version": "astra_generation_launch_manifest_v1",
        "conditions": args.conditions,
        "requested_workers": args.workers,
        "effective_workers": workers,
        "max_attempts_per_task": args.max_attempts,
        "output_base": str(output_base),
        "elapsed_seconds": round(time.time() - started, 1),
        "state_counts": states,
        "units_written": sum(o.units_written for o in outcomes),
        "units_complete": sum(o.units_complete for o in outcomes),
        "tasks": [outcome.public_dict() for outcome in outcomes],
    }
    (output_base / "launch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    unhealthy = states.get("failed", 0) + states.get("refused", 0)
    return 2 if unhealthy else 0


if __name__ == "__main__":
    raise SystemExit(main())
