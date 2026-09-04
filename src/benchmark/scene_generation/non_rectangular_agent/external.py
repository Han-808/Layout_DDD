"""Provider-neutral external Agent command adapter with safe retry semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import signal
import subprocess
import time
from typing import Any, Callable, Mapping

from .artifacts import AgentEpisodeArtifacts, write_json_exclusive
from .profiles import AgentBackendProfile
from .tool_client import SOCKET_ENV, TOKEN_ENV
from .tool_server import AgentToolServer


@dataclass(frozen=True, slots=True)
class AgentProcessResult:
    status: str
    attempts: int
    returncode: int | None
    timed_out: bool
    final_submission_sealed: bool
    elapsed_seconds: float = 0.0

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "non_rectangular_external_agent_result_v1",
            "status": self.status,
            "attempts": self.attempts,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "final_submission_sealed": self.final_submission_sealed,
            "elapsed_seconds": self.elapsed_seconds,
        }


def run_external_agent(
    *,
    profile: AgentBackendProfile,
    artifacts: AgentEpisodeArtifacts,
    tool_server: AgentToolServer,
    task_prompt: str,
    environ: Mapping[str, str] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> AgentProcessResult:
    """Run one Agent command; only declared retryable exits may be restarted."""

    source_environment = os.environ if environ is None else environ
    attempts = 0
    last_returncode: int | None = None
    episode_started = time.monotonic()
    for attempt in range(1, profile.max_process_attempts + 1):
        attempts = attempt
        attempt_dir = artifacts.next_attempt_dir()
        command = _expand_command(profile.command, artifacts=artifacts)
        safe_environment = _build_environment(
            profile,
            source_environment=source_environment,
            tool_server=tool_server,
        )
        write_json_exclusive(
            attempt_dir / "started.json",
            {
                "schema_version": "non_rectangular_external_agent_attempt_started_v1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
                "agent_id": profile.agent_id,
                "model_id": profile.model_id,
                "command_argv_count": len(command),
                "timeout_seconds": profile.timeout_seconds,
                "credential_values_recorded": False,
            },
        )
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        started = time.monotonic()
        timed_out = False
        launch_error_type: str | None = None
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=artifacts.workspace,
                    env=safe_environment,
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except OSError as exc:
                process = None
                launch_error_type = type(exc).__name__
            if process is not None:
                try:
                    process.communicate(
                        input=task_prompt.encode("utf-8"),
                        timeout=profile.timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_group(process)
                last_returncode = process.returncode
            else:
                last_returncode = None
        elapsed = max(0.0, time.monotonic() - started)
        sealed = artifacts.final_submission.is_file() and artifacts.finalization.is_file()
        if sealed:
            status = "complete"
        elif launch_error_type is not None:
            status = "nonretryable_process_launch_failure"
        elif timed_out:
            # A timed-out Agent may still have an in-flight model/tool operation.
            # Never restart it blindly inside the same episode.
            status = "ambiguous_timeout"
        elif last_returncode == 0:
            status = "submission_missing"
        elif last_returncode in profile.retryable_exit_codes:
            status = "retryable_infrastructure_failure"
        else:
            status = "nonretryable_process_failure"
        write_json_exclusive(
            attempt_dir / "result.json",
            {
                "schema_version": "non_rectangular_external_agent_attempt_result_v1",
                "attempt": attempt,
                "status": status,
                "returncode": last_returncode,
                "timed_out": timed_out,
                "launch_error_type": launch_error_type,
                "elapsed_seconds": elapsed,
                "final_submission_sealed": sealed,
                "stdout_bytes": stdout_path.stat().st_size,
                "stderr_bytes": stderr_path.stat().st_size,
            },
        )
        if status == "complete" or status != "retryable_infrastructure_failure":
            return AgentProcessResult(
                status=status,
                attempts=attempts,
                returncode=last_returncode,
                timed_out=timed_out,
                final_submission_sealed=sealed,
                elapsed_seconds=max(0.0, time.monotonic() - episode_started),
            )
        if attempt < profile.max_process_attempts:
            sleeper(profile.retry_delay_seconds)
    return AgentProcessResult(
        status="retryable_infrastructure_failure",
        attempts=attempts,
        returncode=last_returncode,
        timed_out=False,
        final_submission_sealed=False,
        elapsed_seconds=max(0.0, time.monotonic() - episode_started),
    )


def _expand_command(
    command: tuple[str, ...], *, artifacts: AgentEpisodeArtifacts
) -> list[str]:
    values = {
        "workspace": str(artifacts.workspace),
        "task_path": str(artifacts.workspace_task),
        "submission_path": str(artifacts.workspace / "submission.json"),
    }
    output: list[str] = []
    for item in command:
        try:
            output.append(item.format_map(values))
        except (KeyError, ValueError) as exc:
            raise ValueError("Agent command contains an unsupported placeholder") from exc
    if not output or not output[0]:
        raise ValueError("Agent command must contain an executable")
    return output


def _build_environment(
    profile: AgentBackendProfile,
    *,
    source_environment: Mapping[str, str],
    tool_server: AgentToolServer,
) -> dict[str, str]:
    names = {"PATH", "LANG", "LC_ALL"} | set(profile.pass_environment)
    environment = {
        name: str(source_environment[name])
        for name in names
        if name in source_environment
    }
    environment.update(
        {
            SOCKET_ENV: str(tool_server.socket_path),
            TOKEN_ENV: tool_server.token,
            "LAYOUT_DDD_AGENT_TRACK_ID": "complicated_floorplan_agent_track_v1",
            "LAYOUT_DDD_AGENT_TOOL_COMMAND": "./layout-ddd-agent-tool",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.0)


__all__ = ["AgentProcessResult", "run_external_agent"]
