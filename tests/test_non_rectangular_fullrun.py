from __future__ import annotations

from dataclasses import replace
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmark.scene_generation.non_rectangular_multi_room.cohort_runner import (
    NonRectangularFullrunError,
    execute_fullrun,
    prepare_fullrun,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = (
    ROOT
    / "configs/generation_extensions/non_rectangular_multi_room_v1/fullruns"
)
API2_PROFILE = PROFILE_ROOT / "spatiallm_selected10_api2_retry5_v2.json"
API3_PROFILE = PROFILE_ROOT / "spatiallm_selected10_api3_retry5_v2.json"


@pytest.mark.parametrize(
    ("profile_path", "provider", "model_order"),
    [
        (
            API2_PROFILE,
            "api2",
            ["glm-5.3", "gpt-5.6-sol", "kimi-k3"],
        ),
        (
            API3_PROFILE,
            "api3",
            [
                "claude-sonnet-5-aihub",
                "claude-opus-5-aihub",
                "claude-fable-5-aihub",
            ],
        ),
    ],
)
def test_provider_fullrun_profiles_are_exact_and_offline_ready(
    profile_path: Path,
    provider: str,
    model_order: list[str],
) -> None:
    prepared = prepare_fullrun(profile_path)
    public = prepared.public_dict()

    assert public["provider_family"] == provider
    assert public["model_order"] == model_order
    assert public["scene_count"] == 10
    assert public["case_count"] == 30
    assert public["room_count_per_model"] == 50
    assert public["stage_timeouts_seconds"] == {
        "stage_a": 2400.0,
        "stage_c": 3600.0,
    }
    assert public["retry_policy"] == {
        "max_infrastructure_retries": 5,
        "retry_ambiguous_timeouts": False,
    }
    assert public["credential_loaded"] is False
    assert public["network_used"] is False
    assert all(
        campaign.route.route_profile_id.startswith(f"{provider}-")
        for campaign in prepared.campaigns.values()
    )


def test_mock_api2_whole_run_is_30_cases_and_auto_resumes(
    tmp_path: Path,
) -> None:
    prepared = replace(
        prepare_fullrun(API2_PROFILE),
        preflight_retry_delay_seconds=0.0,
        model_cooldown_seconds=0.0,
        case_failure_delay_seconds=0.0,
    )
    activations: list[str] = []
    calls: list[tuple[str, str, bool]] = []

    def activate(campaign: Any, **kwargs: Any):
        activations.append(campaign.model.model_profile_id)
        return {"ok": True, "http_status": 200}, SimpleNamespace()

    def run_case(
        campaign: Any,
        activated: Any,
        *,
        output_root: Path,
        resume: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        output_root.mkdir(parents=True, exist_ok=resume)
        calls.append(
            (
                campaign.model.model_profile_id,
                campaign.room_layout["layout_id"],
                resume,
            )
        )
        return {"status": "complete"}

    output = tmp_path / "api2"
    first = execute_fullrun(
        prepared,
        command="run",
        output_base=output,
        fresh=True,
        activate=activate,
        run_case=run_case,
        sleeper=lambda value: None,
    )
    second = execute_fullrun(
        prepared,
        command="run",
        output_base=output,
        activate=activate,
        run_case=run_case,
        sleeper=lambda value: None,
    )

    assert first["status"] == "complete"
    assert first["complete_cases"] == 30
    assert second["status"] == "complete"
    assert len(activations) == 6
    assert len(calls) == 60
    assert all(resume is False for _, _, resume in calls[:30])
    assert all(resume is True for _, _, resume in calls[30:])
    manifest = json.loads(
        (output / "_runner_state/run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["provider_family"] == "api2"
    assert manifest["stage_timeouts_seconds"] == {
        "stage_a": 2400.0,
        "stage_c": 3600.0,
    }
    text = (output / "_runner_state/events.jsonl").read_text(encoding="utf-8")
    assert "credential" not in text
    assert "endpoint" not in text


def test_preflight_retries_transient_but_not_nonretryable_http() -> None:
    prepared = replace(
        prepare_fullrun(API3_PROFILE),
        preflight_retry_delay_seconds=0.0,
        model_cooldown_seconds=0.0,
    )
    attempts: dict[str, int] = {}

    def transient(campaign: Any, **kwargs: Any):
        key = campaign.model.model_profile_id
        attempts[key] = attempts.get(key, 0) + 1
        if key == prepared.models[0].model_profile_id and attempts[key] < 3:
            return {
                "ok": False,
                "http_status": 503,
                "failure_category": "transport_or_http",
            }, None
        return {"ok": True, "http_status": 200}, SimpleNamespace()

    result = execute_fullrun(
        prepared,
        command="preflight",
        activate=transient,
        sleeper=lambda value: None,
    )

    assert result["status"] == "complete"
    assert result["passed_models"] == 3
    assert result["models"][0]["preflight_attempts"] == 3

    unauthorized_attempts = 0

    def unauthorized(campaign: Any, **kwargs: Any):
        nonlocal unauthorized_attempts
        unauthorized_attempts += 1
        return {
            "ok": False,
            "http_status": 401,
            "failure_category": "transport_or_http",
        }, None

    result = execute_fullrun(
        replace(prepared, preflight_max_total_attempts=5),
        command="preflight",
        activate=unauthorized,
        sleeper=lambda value: None,
    )

    assert result["status"] == "partial"
    assert unauthorized_attempts == 3
    assert all(item["preflight_attempts"] == 1 for item in result["models"])


def test_resume_manifest_drift_and_duplicate_writer_fail_closed(
    tmp_path: Path,
) -> None:
    prepared = replace(
        prepare_fullrun(API2_PROFILE),
        preflight_retry_delay_seconds=0.0,
        model_cooldown_seconds=0.0,
        case_failure_delay_seconds=0.0,
    )

    def activate(campaign: Any, **kwargs: Any):
        return {"ok": True, "http_status": 200}, SimpleNamespace()

    def run_case(campaign: Any, activated: Any, *, output_root: Path, **kwargs: Any):
        output_root.mkdir(parents=True, exist_ok=True)
        return {"status": "complete"}

    output = tmp_path / "api2"
    execute_fullrun(
        prepared,
        command="run",
        output_base=output,
        fresh=True,
        activate=activate,
        run_case=run_case,
        sleeper=lambda value: None,
    )
    manifest_path = output / "_runner_state/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NonRectangularFullrunError, match="manifest identity"):
        execute_fullrun(
            prepared,
            command="run",
            output_base=output,
            activate=activate,
            run_case=run_case,
            sleeper=lambda value: None,
        )

    clean = tmp_path / "locked"
    state = clean / "_runner_state"
    state.mkdir(parents=True)
    lock_path = state / "runner.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(NonRectangularFullrunError, match="another writer"):
            execute_fullrun(
                prepared,
                command="run",
                output_base=clean,
                activate=activate,
                run_case=run_case,
                sleeper=lambda value: None,
            )


def test_fresh_refuses_existing_output_before_any_preflight(tmp_path: Path) -> None:
    prepared = prepare_fullrun(API3_PROFILE)
    output = tmp_path / "existing"
    output.mkdir()
    calls = 0

    def activate(campaign: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return {"ok": True}, SimpleNamespace()

    with pytest.raises(FileExistsError):
        execute_fullrun(
            prepared,
            command="run",
            output_base=output,
            fresh=True,
            activate=activate,
            sleeper=lambda value: None,
        )

    assert calls == 0
