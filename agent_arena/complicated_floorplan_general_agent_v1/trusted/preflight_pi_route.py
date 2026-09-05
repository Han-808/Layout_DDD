#!/usr/bin/env python3
"""Run a route preflight through the exact pinned Pi harness.

This is deliberately not a hand-written provider probe.  It launches the
frozen Pi runtime inside the same Seatbelt boundary used by an episode, lets
Pi serialize a real four-tool request, executes one stock ``read`` tool call,
and requires a second model turn.  Only sanitized booleans, counts, hashes,
and stable failure classes are returned to the experiment supervisor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import tempfile
import time
from typing import Any, Callable, Mapping

from api_profiles import ModelProfile, RouteProfile, RouteRuntimeBinding
from isolated_exec import IsolationResult, run_isolated
from model_gateway import (
    GatewayShutdownError,
    ScopedModelGateway,
    SharedCooldownGate,
    completion_requires_transport_recycle,
    verify_gateway_audit,
)
from pi_harness import (
    ARENA_ROOT,
    PiEpisodeConfig,
    ROUTE_PREFLIGHT_FIXTURE,
    ROUTE_PREFLIGHT_TASK_PROMPT,
    pi_system_prompt_binding,
    prepare_route_preflight,
    verify_prepared_episode_material,
)
from pi_tool_transcript import (
    project_pi_tool_transcript,
    verify_pi_tool_transcript,
)


PREFLIGHT_MARKER = "SIEVE_PREFLIGHT_OK"
PREFLIGHT_AGENT_ID = "route-preflight"
PORTABLE_ID_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)


class PiRoutePreflightError(RuntimeError):
    """A route or pinned-Pi compatibility failure."""


@dataclass(frozen=True)
class PiRoutePreflightReport:
    ok: bool
    model_profile_id: str
    api_family_id: str
    route_profile_id: str
    protocol: str
    audit_relative_path: str
    process_status: str | None
    process_returncode_zero: bool
    logical_requests: int
    gateway_stream_contract_complete: bool
    pinned_pi_tool_roundtrip_complete: bool
    tool_calls_started: int
    tool_calls_ended: int
    exactly_one_read_call: bool
    final_marker_exact: bool
    reasoning_replay_required: bool
    reasoning_replayed_on_tool_followup: bool | None
    response_identity_required: bool
    response_identity_matches: bool | None
    routing_identity_assurance: str
    transport_recycle_required: bool
    failure_category: str | None
    failure_code: str | None
    elapsed_seconds: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sieve_pi_route_preflight_report_v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **asdict(self),
            "real_pinned_pi_process_used": True,
            "handwritten_provider_payload_used": False,
            "raw_request_recorded": False,
            "raw_response_recorded": False,
            "hidden_reasoning_recorded": False,
            "credential_headers_or_endpoint_recorded": False,
        }


def run_pi_route_preflight(
    *,
    runtime_root: str | Path,
    audit_root: str | Path,
    experiment_id: str,
    experiment_sha256: str,
    profile_registry_sha256: str,
    route: RouteProfile,
    model: ModelProfile,
    runtime_binding: RouteRuntimeBinding,
    runtime_credential: str,
    cooldown_gate: SharedCooldownGate,
    wall_clock_seconds: int,
    managed_transport_ambiguity_probe: Callable[[bool], bool] | None = None,
) -> PiRoutePreflightReport:
    """Exercise Pi -> scoped gateway -> route -> Pi across one real tool turn."""

    started = time.monotonic()
    root = _initialize_audit_root(audit_root, model)
    host = root / "host"
    workspace = root / "workspace"
    _initialize_workspace(workspace)

    process: IsolationResult | None = None
    completion: dict[str, Any] | None = None
    transcript_summary: dict[str, Any] | None = None
    exactly_one_read = False
    final_marker_exact = False
    gateway_complete = False
    transcript_complete = False
    failure_category: str | None = None
    failure_code: str | None = None
    material: dict[str, Any] | None = None
    gateway_started = False
    transport_recycle_required = False

    try:
        prompt_binding = pi_system_prompt_binding(
            workspace,
            # prepare_route_preflight installs this exact fixed prompt.
            _route_preflight_system_prompt(),
        )
        with _unused_tool_socket() as (tool_socket, tool_token):
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=runtime_binding,
                runtime_credential=runtime_credential,
                max_requests=2,
                event_path=host / "model_gateway_events.jsonl",
                cooldown_gate=cooldown_gate,
                session_id=_session_id(experiment_id, model.model_profile_id),
                required_system_prompt_sha256s=tuple(prompt_binding),
                system_prompt_rewrites=prompt_binding,
                managed_transport_ambiguity_probe=(
                    managed_transport_ambiguity_probe
                ),
            ) as gateway:
                gateway_started = True
                material = prepare_route_preflight(
                    PiEpisodeConfig(
                        runtime_root=Path(runtime_root),
                        workspace=workspace,
                        gateway_base_url=gateway.base_url,
                        model_profile=model,
                        experiment_id=experiment_id,
                        experiment_sha256=experiment_sha256,
                        profile_registry_sha256=profile_registry_sha256,
                        max_model_requests=2,
                        wall_clock_seconds=wall_clock_seconds,
                    )
                )
                _write_json_exclusive(
                    host / "preflight_material.json",
                    material["preflight_record"],
                )
                _write_json_exclusive(
                    host / "model_gateway.json",
                    gateway.public_dict(),
                )
                process = run_isolated(
                    workspace=workspace,
                    runtime_root=runtime_root,
                    command=material["command"],
                    tool_socket=tool_socket,
                    tool_token=tool_token,
                    stdout_path=host / "agent.stdout.jsonl",
                    stderr_path=host / "agent.stderr.log",
                    stdin_text=material["stdin_text"],
                    timeout_seconds=wall_clock_seconds,
                    model_gateway=gateway.endpoint_address,
                    model_gateway_token=gateway.capability_token,
                    extra_environment={
                        "ARENA_AGENT_ID": PREFLIGHT_AGENT_ID,
                        "ARENA_MODEL_ID": model.model_profile_id,
                        "ARENA_RUN_ID": root.name,
                    },
                    harness_extension=material["harness_extension_path"],
                )
                _write_json_exclusive(host / "process_result.json", process.public_dict())
                transcript_summary = project_pi_tool_transcript(
                    source_path=host / "agent.stdout.jsonl",
                    output_path=host / "pi_tool_transcript.jsonl",
                    require_complete=process.status == "exited_zero",
                )
                _write_json_exclusive(
                    host / "pi_tool_transcript_summary.json",
                    transcript_summary,
                )
            completion = gateway.wait_for_completion_report()

        if completion is None:
            raise PiRoutePreflightError("gateway_completion_missing")
        _write_json_exclusive(host / "model_gateway_completion.json", completion)
        gateway_verification = verify_gateway_audit(
            host / "model_gateway_events.jsonl",
            completion,
            expected_api_family_id=model.api_family_id,
            expected_route_profile_id=route.route_profile_id,
            expected_model_profile_id=model.model_profile_id,
            expected_retry_policy=model.retry.public_dict(),
        )
        gateway_complete = bool(
            gateway_verification.get("all_logical_requests_complete")
        )
        if process is None or process.status != "exited_zero":
            status = process.status if process is not None else "missing"
            raise PiRoutePreflightError(f"pi_process_{status}")
        if material is None or transcript_summary is None:
            raise PiRoutePreflightError("preflight_material_missing")
        verify_prepared_episode_material(material)
        transcript_verification = verify_pi_tool_transcript(
            source_path=host / "agent.stdout.jsonl",
            transcript_path=host / "pi_tool_transcript.jsonl",
            summary=transcript_summary,
            require_complete=True,
        )
        transcript_complete = bool(transcript_verification.get("complete"))
        exactly_one_read = _exactly_one_successful_read(
            host / "pi_tool_transcript.jsonl"
        )
        final_marker_exact = _last_assistant_text(host / "agent.stdout.jsonl") == (
            PREFLIGHT_MARKER
        )
        _verify_completion_contract(
            completion=completion,
            model=model,
            exactly_one_read=exactly_one_read,
            final_marker_exact=final_marker_exact,
            gateway_complete=gateway_complete,
            transcript_complete=transcript_complete,
        )
    except GatewayShutdownError:
        raise
    except Exception as exc:
        if gateway_started and completion is None:
            # The gateway ran but no trustworthy terminal snapshot was
            # obtained. A managed intermediary must be recycled before later
            # work; direct transports treat the callback as a no-op.
            transport_recycle_required = True
        failure_category = _failure_category(exc)
        failure_code = _failure_code(exc)
    finally:
        for name in (
            "agent.stdout.jsonl",
            "agent.stderr.log",
            "model_gateway_events.jsonl",
        ):
            path = host / name
            if path.is_file() and not path.is_symlink():
                path.chmod(0o444)

    identity_matches = _identity_result(completion, model)
    reasoning_replayed = _reasoning_replay_result(completion)
    transport_recycle_required = bool(
        transport_recycle_required
        or completion_requires_transport_recycle(completion)
    )
    logical_requests = (
        int(completion.get("request_count", 0))
        if isinstance(completion, Mapping)
        else 0
    )
    process_status = process.status if process is not None else None
    ok = failure_code is None
    return PiRoutePreflightReport(
        ok=ok,
        model_profile_id=model.model_profile_id,
        api_family_id=model.api_family_id,
        route_profile_id=route.route_profile_id,
        protocol=route.pi_api_protocol,
        audit_relative_path=root.relative_to(ARENA_ROOT).as_posix(),
        process_status=process_status,
        process_returncode_zero=bool(process and process.returncode == 0),
        logical_requests=logical_requests,
        gateway_stream_contract_complete=gateway_complete,
        pinned_pi_tool_roundtrip_complete=transcript_complete,
        tool_calls_started=(
            int(transcript_summary.get("started_tool_calls", 0))
            if isinstance(transcript_summary, Mapping)
            else 0
        ),
        tool_calls_ended=(
            int(transcript_summary.get("ended_tool_calls", 0))
            if isinstance(transcript_summary, Mapping)
            else 0
        ),
        exactly_one_read_call=exactly_one_read,
        final_marker_exact=final_marker_exact,
        reasoning_replay_required=model.reasoning.preserve_across_tool_turns,
        reasoning_replayed_on_tool_followup=reasoning_replayed,
        response_identity_required=model.response_identity_required,
        response_identity_matches=identity_matches,
        routing_identity_assurance=(
            "request_and_response_identity_gated"
            if model.response_identity_required
            else "request_profile_only_unverified_response"
        ),
        transport_recycle_required=transport_recycle_required,
        failure_category=failure_category,
        failure_code=failure_code,
        elapsed_seconds=round(max(0.0, time.monotonic() - started), 6),
    )


def write_pi_route_preflight_report(
    path: str | Path, report: PiRoutePreflightReport
) -> None:
    _write_json_exclusive(Path(path), report.public_dict())


def _verify_completion_contract(
    *,
    completion: Mapping[str, Any],
    model: ModelProfile,
    exactly_one_read: bool,
    final_marker_exact: bool,
    gateway_complete: bool,
    transcript_complete: bool,
) -> None:
    if completion.get("request_count") != 2:
        raise PiRoutePreflightError("logical_request_count_mismatch")
    if completion.get("tool_history_by_request") != {"1": False, "2": True}:
        raise PiRoutePreflightError("tool_followup_history_missing")
    replay = completion.get("reasoning_replay_by_request")
    if not isinstance(replay, Mapping) or replay.get("1") is not None:
        raise PiRoutePreflightError("reasoning_replay_evidence_malformed")
    expected_replay = bool(model.reasoning.preserve_across_tool_turns)
    if replay.get("2") is not expected_replay:
        raise PiRoutePreflightError("reasoning_replay_contract_mismatch")
    identities = completion.get("identity_matches_by_request")
    expected_identity = True if model.response_identity_required else None
    if identities != {"1": expected_identity, "2": expected_identity}:
        raise PiRoutePreflightError("response_identity_contract_mismatch")
    if not gateway_complete:
        raise PiRoutePreflightError("gateway_stream_contract_incomplete")
    if not transcript_complete:
        raise PiRoutePreflightError("pi_tool_transcript_incomplete")
    if not exactly_one_read:
        raise PiRoutePreflightError("exact_read_tool_roundtrip_missing")
    if not final_marker_exact:
        raise PiRoutePreflightError("final_marker_mismatch")


def _exactly_one_successful_read(path: Path) -> bool:
    starts: list[dict[str, Any]] = []
    ends: list[dict[str, Any]] = []
    for value in _iter_json_objects(path):
        if value.get("event") == "tool_execution_start":
            starts.append(value)
        elif value.get("event") == "tool_execution_end":
            ends.append(value)
    if len(starts) != 1 or len(ends) != 1:
        return False
    start = starts[0]
    end = ends[0]
    arguments = start.get("arguments")
    return bool(
        start.get("tool_name") == "read"
        and end.get("tool_name") == "read"
        and start.get("tool_call_sha256") == end.get("tool_call_sha256")
        and isinstance(arguments, Mapping)
        and arguments.get("path") == "preflight_fixture.txt"
        and end.get("is_error") is False
        and end.get("result")
        == {
            "content": [
                {
                    "type": "text",
                    "text": ROUTE_PREFLIGHT_FIXTURE,
                }
            ]
        }
    )


def _last_assistant_text(path: Path) -> str | None:
    last: str | None = None
    for value in _iter_json_objects(path):
        if value.get("type") != "message_end":
            continue
        message = value.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        blocks = [
            block.get("text")
            for block in content
            if isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if blocks:
            last = "".join(blocks)
    return last


def _iter_json_objects(path: Path):
    if not path.is_file() or path.is_symlink():
        raise PiRoutePreflightError("preflight_audit_file_missing")
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PiRoutePreflightError("preflight_audit_record_malformed")
            yield value


def _identity_result(
    completion: Mapping[str, Any] | None, model: ModelProfile
) -> bool | None:
    if not model.response_identity_required:
        return None
    if not isinstance(completion, Mapping):
        return False
    identities = completion.get("identity_matches_by_request")
    return bool(
        isinstance(identities, Mapping)
        and identities.get("1") is True
        and identities.get("2") is True
    )


def _reasoning_replay_result(
    completion: Mapping[str, Any] | None,
) -> bool | None:
    if not isinstance(completion, Mapping):
        return None
    replay = completion.get("reasoning_replay_by_request")
    return replay.get("2") if isinstance(replay, Mapping) else None


def _initialize_audit_root(
    value: str | Path, model: ModelProfile
) -> Path:
    root = Path(value).expanduser().absolute()
    episodes = (ARENA_ROOT / "episodes").resolve(strict=True)
    try:
        relative = root.relative_to(episodes)
    except ValueError as exc:
        raise PiRoutePreflightError("preflight_audit_root_outside_episodes") from exc
    if (
        len(relative.parts) != 3
        or relative.parts[0] != PREFLIGHT_AGENT_ID
        or relative.parts[1] != model.model_profile_id
        or any(not _portable_id(part) for part in relative.parts)
    ):
        raise PiRoutePreflightError("preflight_audit_identity_mismatch")
    if root.exists() or root.is_symlink():
        raise PiRoutePreflightError("preflight_audit_root_already_exists")
    (root / "host").mkdir(parents=True, mode=0o700)
    (root / "workspace").mkdir(mode=0o700)
    return root.resolve(strict=True)


def _initialize_workspace(workspace: Path) -> None:
    (workspace / ".home").mkdir(mode=0o700)
    (workspace / ".tmp").mkdir(mode=0o700)
    task = {
        "schema_version": "sieve_pi_route_preflight_task_v1",
        "asset_database": {"snapshot_id": "route-preflight-no-database-access"},
        "public_validation_policy": {"version": "route-preflight-v1"},
        "tool_policy": {
            "max_asset_inspections": 0,
            "max_asset_searches": 0,
            "max_submission_validations": 0,
            "max_top_k": 1,
            "max_total_calls": 0,
        },
    }
    _write_bytes_exclusive(
        workspace / "TODO.md",
        ROUTE_PREFLIGHT_TASK_PROMPT.encode("utf-8"),
        0o444,
    )
    _write_bytes_exclusive(
        workspace / "task.json",
        _json_bytes(task),
        0o444,
    )
    for name in (
        "database-interface.json",
        "floorplan.json",
        "room_program.json",
        "submission.schema.json",
    ):
        _write_bytes_exclusive(workspace / name, b"{}\n", 0o444)
    _write_bytes_exclusive(
        workspace / "sieve-agent-tool",
        b"#!/bin/sh\nexit 126\n",
        0o555,
    )


class _unused_tool_socket:
    """Provide run_isolated's mandatory DB capability without serving data."""

    def __init__(self) -> None:
        self.root: Path | None = None
        self.path: Path | None = None
        self.server: socket.socket | None = None
        self.token = secrets.token_hex(32)

    def __enter__(self) -> tuple[Path, str]:
        self.root = Path(tempfile.mkdtemp(prefix="sieve-pi-route-preflight-"))
        self.path = self.root / "tool.sock"
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen(1)
        return self.path, self.token

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.server is not None:
            self.server.close()
        if self.path is not None and self.path.exists():
            self.path.unlink()
        if self.root is not None and self.root.exists():
            shutil.rmtree(self.root)
        self.token = ""


def _route_preflight_system_prompt() -> str:
    # Imported lazily through the module constant without duplicating its text.
    from pi_harness import ROUTE_PREFLIGHT_SYSTEM_PROMPT

    return ROUTE_PREFLIGHT_SYSTEM_PROMPT


def _session_id(experiment_id: str, model_profile_id: str) -> str:
    entropy = secrets.token_hex(16)
    return hashlib.sha256(
        f"{experiment_id}:{model_profile_id}:{entropy}".encode("utf-8")
    ).hexdigest()


def _portable_id(value: str) -> bool:
    return bool(value) and len(value) <= 127 and all(
        character in PORTABLE_ID_CHARS for character in value
    )


def _failure_category(exc: Exception) -> str:
    if isinstance(exc, PiRoutePreflightError):
        code = str(exc)
        if code.startswith("pi_process_"):
            return "agent_runtime"
        if "identity" in code:
            return "routing_identity"
        if "reasoning" in code:
            return "reasoning_compatibility"
        if "gateway" in code:
            return "transport_or_stream"
        if "tool" in code or "marker" in code:
            return "tool_roundtrip"
        return "route_compatibility"
    return "workflow_infrastructure"


def _failure_code(exc: Exception) -> str:
    if isinstance(exc, PiRoutePreflightError):
        value = str(exc)
        if value and len(value) <= 160 and all(
            character.isalnum() or character in "_.-" for character in value
        ):
            return value
    return type(exc).__name__


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, _json_bytes(dict(value)), 0o444)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_exclusive(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PiRoutePreflightError(f"refusing_to_overwrite_{path.name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)


__all__ = [
    "PREFLIGHT_MARKER",
    "PiRoutePreflightError",
    "PiRoutePreflightReport",
    "run_pi_route_preflight",
    "write_pi_route_preflight_report",
]
