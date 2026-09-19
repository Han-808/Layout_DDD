"""Task planning and concurrency policy of the Astra generation launcher.

These tests cover the launcher's own decisions only. Generation policy belongs to
the registered campaigns and is asserted by the campaign suites.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

_MODULE_NAME = "astra_generation_launcher"
_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_astra_generation_v1.py"
)
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
launcher = importlib.util.module_from_spec(_spec)
# `dataclass` resolves annotations through `sys.modules`, so register before exec.
sys.modules[_MODULE_NAME] = launcher
_spec.loader.exec_module(launcher)


def _floor_plan_root(tmp_path: Path, layouts: int) -> Path:
    root = tmp_path / "runner_inputs_v1"
    root.mkdir()
    for index in range(1, layouts + 1):
        layout = root / f"layout_{index:02d}"
        layout.mkdir()
        (layout / "floor_plan.json").write_text("{}", encoding="utf-8")
    return root


def test_open_space_is_a_single_unsplittable_task(tmp_path: Path) -> None:
    tasks = launcher.build_tasks(["open-space"], _floor_plan_root(tmp_path, 10))
    assert len(tasks) == 1
    assert tasks[0].campaign_id == launcher.OPEN_SPACE_CAMPAIGN
    assert tasks[0].expected_units == 10
    assert tasks[0].floor_plan is None


def test_multi_room_is_one_task_per_layout(tmp_path: Path) -> None:
    tasks = launcher.build_tasks(["multi-room"], _floor_plan_root(tmp_path, 10))
    assert [task.task_id for task in tasks] == [
        f"multi-room.layout_{index:02d}" for index in range(1, 11)
    ]
    assert all(task.campaign_id == launcher.MULTI_ROOM_CAMPAIGN for task in tasks)
    assert all(task.floor_plan is not None for task in tasks)


def test_multi_room_requires_real_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        launcher.build_tasks(["multi-room"], tmp_path / "absent")
    empty = tmp_path / "runner_inputs_v1"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="floor_plan.json"):
        launcher.build_tasks(["multi-room"], empty)


def test_complex_layout_is_surfaced_as_blocked_not_skipped(tmp_path: Path) -> None:
    tasks = launcher.build_tasks(["complex-layout"], _floor_plan_root(tmp_path, 1))
    assert len(tasks) == 1
    outcome = launcher._check_task(tasks[0], launcher._parse_args(["--check"]))
    assert outcome.state == "blocked"
    assert "floorplan_api_complexity_v1" in outcome.blocked_reason


def test_only_reviewed_worker_counts_are_accepted() -> None:
    assert launcher.ALLOWED_WORKERS == (1, 3, 6)
    for workers in launcher.ALLOWED_WORKERS:
        assert launcher._parse_args(["--workers", str(workers)]).workers == workers
    with pytest.raises(SystemExit):
        launcher._parse_args(["--workers", "4"])


def test_run_requires_an_output_base() -> None:
    with pytest.raises(SystemExit):
        launcher._parse_args(["--run"])
    args = launcher._parse_args(["--run", "--output-base", "/tmp/astra-out"])
    assert args.run is True


def test_retry_budget_is_bounded_by_default() -> None:
    args = launcher._parse_args([])
    assert args.max_attempts == 2
    assert args.retry_delay_seconds == 30.0
    with pytest.raises(SystemExit):
        launcher._parse_args(["--max-attempts", "0"])


def test_default_mode_is_a_read_only_check() -> None:
    args = launcher._parse_args([])
    assert args.run is False
    assert args.conditions == list(launcher.CONDITIONS)


def test_unit_counts_reads_terminal_case_results(tmp_path: Path) -> None:
    for name, status in (
        ("brief_00", "complete"),
        ("brief_01", "stage_a_schema_invalid"),
        ("brief_02", "complete"),
    ):
        unit = tmp_path / name
        unit.mkdir()
        (unit / "case.result.json").write_text(
            '{"schema_version": "hy34_case_result_v2", "status": "%s"}' % status,
            encoding="utf-8",
        )
    written, complete = launcher._unit_counts(tmp_path)
    assert (written, complete) == (3, 2)


def test_finished_detection_requires_a_run_manifest(tmp_path: Path) -> None:
    assert launcher._already_finished(tmp_path) is False
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    assert launcher._already_finished(tmp_path) is True


def test_subprocess_env_binds_this_tree_source_first() -> None:
    env = launcher._subprocess_env()
    assert env["PYTHONPATH"].split(":")[0] == str(launcher.REPO_ROOT / "src")
