from __future__ import annotations

import json
from pathlib import Path

from benchmark.camera_cal_scene_level import io as runtime_io
from benchmark.camera_cal_scene_level import progress as runtime_progress
from benchmark.camera_cal_scene_level import telemetry as runtime_telemetry
from scripts import run_camera_cal_scene_level as runner


def test_runner_preserves_leaf_runtime_compatibility_facades() -> None:
    assert issubclass(runner.ProgressReporter, runtime_progress.ProgressReporter)
    assert issubclass(runner.APICallTracker, runtime_telemetry.APICallTracker)
    record = {
        "timestamp": "2026-08-21T01:02:03+00:00",
        "case_id": "S100",
        "event": "metric",
        "details": {"metric": "collision", "evidence_count": 2},
    }
    assert runner._format_progress_record(record) == (
        runtime_progress.format_progress_record(record)
    )


def test_runner_and_leaf_io_emit_identical_json(tmp_path: Path) -> None:
    value = {"unicode": "场景", "nested": {"value": 1}}
    runner_path = tmp_path / "runner.json"
    runtime_path = tmp_path / "runtime.json"
    runner.atomic_write_json(runner_path, value)
    runtime_io.atomic_write_json(runtime_path, value)
    assert runner_path.read_bytes() == runtime_path.read_bytes()
    assert runner.read_json(runner_path) == runtime_io.read_json(runtime_path)
    assert runner.json_sha256(value) == runtime_io.json_sha256(value)
    assert runner.file_sha256(runner_path) == runtime_io.file_sha256(runtime_path)


def test_runner_and_leaf_api_usage_are_deeply_equal() -> None:
    records = [
        {
            "role": "judge",
            "call_type": "vlm_camera_pose.functional_discovery.affordance",
            "status": "complete",
            "tokens_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        },
        {
            "role": "camera_selector",
            "call_type": "camera_selector_rank",
            "status": "failed",
            "tokens_usage": None,
        },
    ]
    left = runner.api_usage_summary(records)
    right = runtime_telemetry.api_usage_summary(records)
    assert json.loads(json.dumps(left, sort_keys=True)) == right
