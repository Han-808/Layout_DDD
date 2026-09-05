"""Budgeted local tool service for one isolated Agent episode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socketserver
import stat
import tempfile
import threading
import time
from typing import Any, Mapping

from .contracts import ValidatedAgentSubmission, validate_agent_submission


TOOL_PROTOCOL_VERSION = "non_rectangular_agent_tool_protocol_v1"
TOOL_EVENT_SCHEMA_VERSION = "non_rectangular_agent_tool_event_v3"
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_SUBMISSION_BYTES = 32 * 1024 * 1024
MAX_TOOL_EVENT_BYTES = 128 * 1024 * 1024
TOOL_METHODS = frozenset(
    {
        "get_task",
        "search_assets",
        "inspect_asset",
        "validate_submission",
        "finalize_submission",
    }
)


class AgentToolError(RuntimeError):
    """Raised for an unauthorized, over-budget, or malformed tool call."""


class AgentToolShutdownError(RuntimeError):
    """Raised when the trusted host cannot prove that tool handlers drained."""


@dataclass(frozen=True, slots=True)
class AgentToolPolicy:
    max_total_calls: int = 160
    max_asset_searches: int = 80
    max_asset_inspections: int = 80
    max_submission_validations: int = 30
    max_top_k: int = 12

    def __post_init__(self) -> None:
        for field in (
            "max_total_calls",
            "max_asset_searches",
            "max_asset_inspections",
            "max_submission_validations",
            "max_top_k",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "non_rectangular_agent_tool_policy_v1",
            "max_total_calls": self.max_total_calls,
            "max_asset_searches": self.max_asset_searches,
            "max_asset_inspections": self.max_asset_inspections,
            "max_submission_validations": self.max_submission_validations,
            "max_top_k": self.max_top_k,
            "evaluator_feedback_allowed": False,
            "external_asset_sources_allowed": False,
        }


class AgentToolSession:
    """Pure dispatcher plus an append-only audit log for one layout."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        room_layout: Mapping[str, Any],
        room_program: Mapping[str, Any],
        asset_catalog: Any,
        task_payload: Mapping[str, Any],
        policy: AgentToolPolicy,
        audit_path: str | Path | None = None,
        seal_record_path: str | Path | None = None,
        sealed_submission_path: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir() or self.workspace.is_symlink():
            raise AgentToolError("Agent workspace must be a real directory")
        self.room_layout = dict(room_layout)
        self.room_program = dict(room_program)
        self.asset_catalog = asset_catalog
        self.task_payload = dict(task_payload)
        self.policy = policy
        self.events_path = (
            Path(audit_path).expanduser().resolve()
            if audit_path is not None
            else self.workspace / "tool_events.jsonl"
        )
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.events_path.exists() and (
            not self.events_path.is_file() or self.events_path.is_symlink()
        ):
            raise AgentToolError("tool event journal must be a real file")
        self.seal_record_path = (
            Path(seal_record_path).expanduser().resolve()
            if seal_record_path is not None
            else None
        )
        if self.seal_record_path is not None:
            self.seal_record_path.parent.mkdir(parents=True, exist_ok=True)
            if self.seal_record_path.exists() and (
                not self.seal_record_path.is_file()
                or self.seal_record_path.is_symlink()
            ):
                raise AgentToolError("trusted seal record must be a real file")
        self.sealed_submission_path = (
            Path(sealed_submission_path).expanduser().resolve()
            if sealed_submission_path is not None
            else None
        )
        if (self.seal_record_path is None) != (self.sealed_submission_path is None):
            raise AgentToolError(
                "trusted seal record and sealed submission must be configured together"
            )
        if self.sealed_submission_path is not None:
            self.sealed_submission_path.parent.mkdir(parents=True, exist_ok=True)
            if self.sealed_submission_path.exists() and (
                not self.sealed_submission_path.is_file()
                or self.sealed_submission_path.is_symlink()
            ):
                raise AgentToolError("host-sealed submission must be a real file")
        self.final_submission_path = self.workspace / "final_submission.json"
        self.finalization_path = self.workspace / "finalization.json"
        self._finalized = any(
            path.exists() or path.is_symlink()
            for path in (
                self.final_submission_path,
                self.finalization_path,
                self.seal_record_path,
                self.sealed_submission_path,
            )
            if path is not None
        )
        self._counts = {
            "total": 0,
            "search_assets": 0,
            "inspect_asset": 0,
            "validate_submission": 0,
            "finalize_submission": 0,
            "get_task": 0,
        }
        self._lock = threading.RLock()
        self._last_event_sha256: str | None = None
        self._restore_counts()

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def dispatch(self, method: str, params: Mapping[str, Any] | None) -> Any:
        name = str(method or "").strip()
        arguments = dict(params or {})
        started = time.monotonic()
        status = "failed"
        error_type: str | None = None
        error_message: str | None = None
        charged = False
        method_charged = False
        result: Any = None
        try:
            with self._lock:
                charge_error, method_charged = self._charge(name)
                charged = True
                if charge_error is not None:
                    raise AgentToolError(charge_error)
                if name == "get_task":
                    _exact_keys(arguments, set(), label="get_task params")
                    result = dict(self.task_payload)
                elif name == "search_assets":
                    _exact_keys(
                        arguments,
                        {"query", "size_constraint", "top_k"},
                        label="search_assets params",
                    )
                    top_k = arguments["top_k"]
                    if (
                        isinstance(top_k, bool)
                        or not isinstance(top_k, int)
                        or top_k > self.policy.max_top_k
                    ):
                        raise AgentToolError("search top_k exceeds the episode policy")
                    result = self.asset_catalog.search(
                        str(arguments["query"]),
                        size_constraint=arguments["size_constraint"],
                        top_k=top_k,
                    )
                elif name == "inspect_asset":
                    _exact_keys(
                        arguments,
                        {"asset_id"},
                        label="inspect_asset params",
                    )
                    result = {
                        "schema_version": "non_rectangular_agent_asset_inspection_v1",
                        "catalog_snapshot_id": str(self.asset_catalog.snapshot_id),
                        "asset": self.asset_catalog.resolve(str(arguments["asset_id"])),
                    }
                elif name in {"validate_submission", "finalize_submission"}:
                    _exact_keys(
                        arguments,
                        {"submission_path"},
                        label=f"{name} params",
                    )
                    submission = self._read_workspace_submission(
                        str(arguments["submission_path"])
                    )
                    validated = validate_agent_submission(
                        submission,
                        room_layout=self.room_layout,
                        room_program=self.room_program,
                        asset_catalog=self.asset_catalog,
                    )
                    constraint_report = validate_task_submission_constraints(
                        validated,
                        task_payload=self.task_payload,
                    )
                    if name == "finalize_submission":
                        self._publish_final(
                            submission,
                            validated,
                            constraint_report=constraint_report,
                        )
                    result = {
                        **validated.public_dict(),
                        "task_constraint_validation": constraint_report,
                    }
                else:
                    raise AgentToolError(f"unsupported Agent tool method: {name!r}")
            status = "complete"
            return result
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = _public_error_message(exc)
            raise
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            if charged:
                self._event(
                    method=name if name in TOOL_METHODS else "invalid",
                    status=status,
                    error_type=error_type,
                    elapsed_seconds=elapsed,
                    charged=True,
                    method_charged=method_charged,
                    params=_safe_params(name, arguments),
                    result=result,
                    error_message=error_message,
                )

    def _charge(self, method: str) -> tuple[str | None, bool]:
        if self._counts["total"] >= self.policy.max_total_calls:
            raise AgentToolError("Agent episode exhausted total tool-call budget")
        self._counts["total"] += 1
        if method not in TOOL_METHODS:
            return f"unsupported Agent tool method: {method!r}", False
        limits = {
            "search_assets": self.policy.max_asset_searches,
            "inspect_asset": self.policy.max_asset_inspections,
            "validate_submission": self.policy.max_submission_validations,
            "finalize_submission": self.policy.max_submission_validations,
        }
        limit = limits.get(method)
        if limit is not None and self._counts[method] >= limit:
            return f"Agent episode exhausted {method} budget", False
        self._counts[method] += 1
        return None, True

    def _read_workspace_submission(self, relative: str) -> dict[str, Any]:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise AgentToolError("submission_path must be workspace-relative")
        resolved = (self.workspace / path).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise AgentToolError("submission_path escapes the Agent workspace") from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise AgentToolError("submission_path must name a real file")
        if resolved.stat().st_size > MAX_SUBMISSION_BYTES:
            raise AgentToolError("Agent submission exceeds size limit")
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AgentToolError("Agent submission is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise AgentToolError("Agent submission root must be an object")
        return value

    def _publish_final(
        self,
        submission: Mapping[str, Any],
        validated: ValidatedAgentSubmission,
        *,
        constraint_report: Mapping[str, Any],
    ) -> None:
        if self._finalized:
            raise AgentToolError("this Agent episode is already finalized")
        expected_submission = _json_file_bytes(submission)
        submission_sha256 = hashlib.sha256(expected_submission).hexdigest()
        if self.sealed_submission_path is not None:
            if self.sealed_submission_path.exists():
                _require_real_file_mode(
                    self.sealed_submission_path,
                    0o400,
                    "host-sealed submission",
                )
                if self.sealed_submission_path.read_bytes() != expected_submission:
                    raise AgentToolError(
                        "a different host-sealed submission already exists"
                    )
            else:
                _write_bytes_exclusive(
                    self.sealed_submission_path,
                    expected_submission,
                    mode=0o400,
                )
        if self.final_submission_path.exists():
            if _load_json(self.final_submission_path) != dict(submission):
                raise AgentToolError("a different final submission is already sealed")
        else:
            _write_json_exclusive(self.final_submission_path, submission)
        if self.final_submission_path.read_bytes() != expected_submission:
            raise AgentToolError("sealed submission bytes differ from validated content")
        finalization = {
            **validated.public_dict(),
            "status": "sealed",
            "catalog_snapshot_id": str(self.asset_catalog.snapshot_id),
            "submission_sha256": submission_sha256,
            "task_constraint_validation": dict(constraint_report),
            "tool_counts": self.counts(),
        }
        stable_current = dict(finalization)
        stable_current.pop("tool_counts", None)
        if self.finalization_path.exists():
            existing = _load_json(self.finalization_path)
            stable_existing = dict(existing)
            stable_existing.pop("tool_counts", None)
            if stable_existing != stable_current:
                raise AgentToolError("finalization identity differs on resume")
        else:
            _write_json_exclusive(self.finalization_path, finalization)
        if self.seal_record_path is not None:
            trusted_record = {
                "schema_version": "sieve_trusted_submission_seal_v2",
                "finalization": stable_current,
                "sealed_submission": {
                    "path": "sealed_submission.json",
                    "sha256": submission_sha256,
                    "size_bytes": len(expected_submission),
                    "mode": "0o400",
                },
            }
            if self.seal_record_path.exists():
                _require_real_file_mode(
                    self.seal_record_path, 0o400, "trusted submission seal"
                )
                if _load_json(self.seal_record_path) != trusted_record:
                    raise AgentToolError("trusted submission seal differs on resume")
            else:
                _write_bytes_exclusive(
                    self.seal_record_path,
                    _json_file_bytes(trusted_record),
                    mode=0o400,
                )
        self._finalized = True

    def _restore_counts(self) -> None:
        if not self.events_path.exists():
            return
        verification = verify_tool_event_journal(
            self.events_path,
            policy=self.policy,
            require_finalized=False,
            expected_mode=None,
        )
        restored = verification["charged_counts"]
        self._counts = {key: int(restored[key]) for key in self._counts}
        self._last_event_sha256 = verification["last_event_sha256"]

    def _event(self, **payload: Any) -> None:
        result = payload.pop("result", None)
        with self._lock:
            if self.events_path.is_symlink():
                raise AgentToolError("tool event journal must not be a symlink")
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "schema_version": TOOL_EVENT_SCHEMA_VERSION,
                "previous_event_sha256": self._last_event_sha256,
                **payload,
            }
            if result is not None:
                event["result"] = result
                event["result_sha256"] = hashlib.sha256(
                    _canonical_json_bytes(result)
                ).hexdigest()
            event["event_sha256"] = hashlib.sha256(
                _canonical_json_bytes(event)
            ).hexdigest()
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.events_path, flags, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            self._last_event_sha256 = str(event["event_sha256"])

    def seal_event_journal(self) -> dict[str, Any]:
        """Fsync, verify, and make the trusted tool journal read-only."""

        with self._lock:
            if not self.events_path.exists() and not self.events_path.is_symlink():
                _write_bytes_exclusive(self.events_path, b"", mode=0o444)
            else:
                if not self.events_path.is_file() or self.events_path.is_symlink():
                    raise AgentToolError("tool event journal must be a real file")
                descriptor = os.open(
                    self.events_path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fchmod(descriptor, 0o444)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            return verify_tool_event_journal(
                self.events_path,
                policy=self.policy,
                require_finalized=False,
                expected_mode=0o444,
            )


def validate_task_submission_constraints(
    validated: ValidatedAgentSubmission,
    *,
    task_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Enforce benchmark-owned hard constraints layered above the base schema."""

    complexity = task_payload.get("complexity_contract")
    geometry = task_payload.get("geometry_contract")
    if not isinstance(complexity, Mapping) or not isinstance(geometry, Mapping):
        return {
            "schema_version": "sieve_agent_task_constraint_validation_v1",
            "valid": True,
            "declared_checks": [],
        }
    ranges = complexity.get("room_instance_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise AgentToolError("task complexity contract lacks room instance ranges")
    plan_rooms = list(validated.object_plan["rooms"])
    plan_order = [str(room["room_id"]) for room in plan_rooms]
    range_order: list[str] = []
    normalized_ranges: dict[str, tuple[int, int]] = {}
    for row in ranges:
        if not isinstance(row, Mapping) or set(row) != {"room_id", "min", "max"}:
            raise AgentToolError("task room instance range is malformed")
        room_id = str(row["room_id"])
        lower = row["min"]
        upper = row["max"]
        if (
            not room_id
            or room_id in normalized_ranges
            or isinstance(lower, bool)
            or not isinstance(lower, int)
            or isinstance(upper, bool)
            or not isinstance(upper, int)
            or lower < 1
            or upper < lower
        ):
            raise AgentToolError("task room instance range is invalid")
        range_order.append(room_id)
        normalized_ranges[room_id] = (lower, upper)
    if range_order != plan_order:
        raise AgentToolError("task room instance range order differs from plan")
    task_total = task_payload.get("target_total_instances")
    if not isinstance(task_total, Mapping) or {
        "min": sum(item[0] for item in normalized_ranges.values()),
        "max": sum(item[1] for item in normalized_ranges.values()),
    } != dict(task_total):
        raise AgentToolError("task room ranges do not reproduce the scene total range")
    actual_counts = validated.plan_validation["room_instance_counts"]
    compliance: list[dict[str, Any]] = []
    for room_id in plan_order:
        lower, upper = normalized_ranges[room_id]
        actual = int(actual_counts[room_id])
        if not lower <= actual <= upper:
            raise AgentToolError(
                f"room {room_id!r} instance count {actual} is outside "
                f"the required range [{lower}, {upper}]"
            )
        compliance.append(
            {"room_id": room_id, "min": lower, "max": upper, "actual": actual}
        )
    scale_policy = geometry.get("uniform_scale")
    if scale_policy != {"policy": "exact", "value": 1.0}:
        raise AgentToolError("task uniform-scale policy is unsupported")
    for room in validated.global_placement["rooms"]:
        for instance in room["instances"]:
            if float(instance["uniform_scale"]) != 1.0:
                raise AgentToolError("every placed instance must use uniform_scale 1.0")
    return {
        "schema_version": "sieve_agent_task_constraint_validation_v1",
        "valid": True,
        "declared_checks": ["per_room_instance_ranges", "uniform_scale_exactly_one"],
        "room_instance_ranges": compliance,
        "uniform_scale": 1.0,
    }


class AgentToolServer:
    """Authenticated loopback JSON-line server around an AgentToolSession."""

    def __init__(
        self,
        session: AgentToolSession,
        *,
        close_timeout_seconds: float = 10.0,
    ) -> None:
        if (
            isinstance(close_timeout_seconds, bool)
            or not isinstance(close_timeout_seconds, (int, float))
            or not math.isfinite(float(close_timeout_seconds))
            or float(close_timeout_seconds) <= 0
        ):
            raise ValueError("close_timeout_seconds must be finite and positive")
        self.session = session
        self._token = secrets.token_hex(32)
        self._close_timeout_seconds = float(close_timeout_seconds)
        self._state = threading.Condition()
        self._active_handlers = 0
        self._started = False
        self._closing = False
        self._closed = False
        self._shutdown_error: AgentToolShutdownError | None = None
        self._audit_verification: dict[str, Any] | None = None
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                if not owner._enter_handler():
                    self._respond(
                        {"ok": False, "error_type": "service_closing"}
                    )
                    return
                try:
                    raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
                    if len(raw) > MAX_REQUEST_BYTES:
                        self._respond(
                            {"ok": False, "error_type": "request_too_large"}
                        )
                        return
                    try:
                        request = json.loads(raw.decode("utf-8"))
                        if not isinstance(request, dict):
                            raise AgentToolError("request root must be an object")
                        if request.get("protocol") != TOOL_PROTOCOL_VERSION:
                            raise AgentToolError("tool protocol version mismatch")
                        if not secrets.compare_digest(
                            str(request.get("token") or ""), owner._token
                        ):
                            raise AgentToolError("tool authentication failed")
                        params = request.get("params")
                        if params is not None and not isinstance(params, Mapping):
                            raise AgentToolError("tool params must be an object")
                        result = owner.session.dispatch(
                            str(request.get("method") or ""), params
                        )
                        self._respond({"ok": True, "result": result})
                    except Exception as exc:
                        self._respond(
                            {
                                "ok": False,
                                "error_type": type(exc).__name__,
                                "message": _public_error_message(exc),
                            }
                        )
                finally:
                    owner._leave_handler()

            def _respond(self, value: Mapping[str, Any]) -> None:
                try:
                    self.wfile.write(
                        (
                            json.dumps(
                                dict(value), ensure_ascii=False, sort_keys=True
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                except OSError:
                    return

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        self._socket_root = Path(
            tempfile.mkdtemp(prefix="layout-ddd-agent-tool-")
        )
        self.socket_path = self._socket_root / "tool.sock"
        self._server = Server(str(self.socket_path), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="nonrect-agent-tool-server",
            daemon=True,
        )

    def start(self) -> "AgentToolServer":
        with self._state:
            if self._closed or self._closing:
                raise AgentToolShutdownError("tool server is already closed")
            if self._started:
                raise AgentToolShutdownError("tool server was already started")
            try:
                self._thread.start()
            except BaseException as exc:
                # Thread.start() can fail after the Unix listener and capability
                # have already been created.  shutdown() is invalid here because
                # serve_forever() never began; revoke and close directly instead.
                self._closing = True
                start_error = exc
            else:
                self._started = True
                return self

        cleanup_failures: list[BaseException] = []
        try:
            self._server.server_close()
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            if self.socket_path.exists() or self.socket_path.is_symlink():
                self.socket_path.unlink()
            if self._socket_root.is_dir():
                self._socket_root.rmdir()
        except BaseException as exc:
            cleanup_failures.append(exc)
        verification: dict[str, Any] | None = None
        try:
            verification = self.session.seal_event_journal()
        except BaseException as exc:
            cleanup_failures.append(exc)
        error = AgentToolShutdownError(
            "tool server start-failure cleanup failed"
            if cleanup_failures
            else "tool server thread failed to start"
        )
        with self._state:
            self._token = ""
            self._closed = True
            self._audit_verification = verification
            self._shutdown_error = error
            self._state.notify_all()
        raise error from (cleanup_failures[-1] if cleanup_failures else start_error)

    @property
    def token(self) -> str:
        with self._state:
            if not self._token:
                raise AgentToolShutdownError("tool capability has been scrubbed")
            return self._token

    @property
    def audit_verification(self) -> dict[str, Any] | None:
        with self._state:
            return (
                None
                if self._audit_verification is None
                else dict(self._audit_verification)
            )

    def _enter_handler(self) -> bool:
        with self._state:
            if self._closing or self._closed:
                return False
            self._active_handlers += 1
            return True

    def _leave_handler(self) -> None:
        with self._state:
            self._active_handlers -= 1
            if self._active_handlers < 0:  # pragma: no cover - invariant guard
                self._active_handlers = 0
                self._shutdown_error = AgentToolShutdownError(
                    "tool handler accounting underflow"
                )
            self._state.notify_all()

    def close(self) -> dict[str, Any]:
        with self._state:
            if self._shutdown_error is not None:
                raise self._shutdown_error
            if self._closed:
                if self._audit_verification is None:  # pragma: no cover
                    raise AgentToolShutdownError("tool audit verification is missing")
                return dict(self._audit_verification)
            self._closing = True
            started = self._started

        deadline = time.monotonic() + self._close_timeout_seconds
        if started:
            self._server.shutdown()
        self._server.server_close()
        if started:
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._state:
            while self._active_handlers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._state.wait(remaining)
            thread_alive = started and self._thread.is_alive()
            active = self._active_handlers

        cleanup_error: OSError | None = None
        try:
            if self.socket_path.exists() or self.socket_path.is_symlink():
                self.socket_path.unlink()
            if self._socket_root.is_dir():
                self._socket_root.rmdir()
        except OSError as exc:
            cleanup_error = exc

        error: AgentToolShutdownError | None = None
        if thread_alive or active:
            error = AgentToolShutdownError(
                "tool server could not prove that all request handlers drained"
            )
        elif cleanup_error is not None:
            error = AgentToolShutdownError("tool server socket cleanup failed")

        verification: dict[str, Any] | None = None
        if error is None:
            try:
                verification = self.session.seal_event_journal()
            except Exception as exc:
                error = AgentToolShutdownError(
                    "tool event journal could not be sealed and verified"
                )
                error.__cause__ = exc

        with self._state:
            self._token = ""
            self._closed = True
            self._closing = True
            self._audit_verification = verification
            self._shutdown_error = error
            self._state.notify_all()
        if error is not None:
            raise error
        if verification is None:  # pragma: no cover - invariant guard
            raise AgentToolShutdownError("tool audit verification is missing")
        return dict(verification)

    def __enter__(self) -> "AgentToolServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def verify_tool_event_journal(
    path: str | Path,
    *,
    policy: AgentToolPolicy,
    require_finalized: bool,
    expected_mode: int | None = 0o444,
) -> dict[str, Any]:
    """Strictly verify the append-only host tool transcript and budgets."""

    source = Path(path).expanduser().absolute()
    if not source.is_file() or source.is_symlink():
        raise AgentToolError("tool event journal must be a real file")
    observed_mode = stat.S_IMODE(source.stat().st_mode)
    if expected_mode is not None and observed_mode != expected_mode:
        raise AgentToolError("tool event journal mode differs")
    size = source.stat().st_size
    if size > MAX_TOOL_EVENT_BYTES:
        raise AgentToolError("tool event journal exceeds its host limit")
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read(MAX_TOOL_EVENT_BYTES + 1)
    except BaseException:
        raise
    if len(raw) > MAX_TOOL_EVENT_BYTES:
        raise AgentToolError("tool event journal exceeds its host limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentToolError("tool event journal is not UTF-8") from exc
    if text and not text.endswith("\n"):
        raise AgentToolError("tool event journal has a truncated final record")

    counts = {
        "total": 0,
        "search_assets": 0,
        "inspect_asset": 0,
        "validate_submission": 0,
        "finalize_submission": 0,
        "get_task": 0,
    }
    previous_sha256: str | None = None
    previous_timestamp: datetime | None = None
    successful_finalizations = 0
    base_keys = {
        "timestamp",
        "schema_version",
        "previous_event_sha256",
        "method",
        "status",
        "error_type",
        "elapsed_seconds",
        "charged",
        "method_charged",
        "params",
        "error_message",
        "event_sha256",
    }
    for index, line in enumerate(text.splitlines(), start=1):
        try:
            event = json.loads(line, parse_constant=_reject_json_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise AgentToolError(
                f"tool event journal record {index} is invalid JSON"
            ) from exc
        if not isinstance(event, dict):
            raise AgentToolError("tool event journal entry must be an object")
        status_value = event.get("status")
        expected_keys = set(base_keys)
        if status_value == "complete":
            expected_keys.update({"result", "result_sha256"})
        if set(event) != expected_keys:
            raise AgentToolError("tool event journal field set differs")
        if event.get("schema_version") != TOOL_EVENT_SCHEMA_VERSION:
            raise AgentToolError("tool event journal schema differs")
        timestamp = _parse_utc_timestamp(event.get("timestamp"))
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise AgentToolError("tool event journal timestamps are not monotonic")
        previous_timestamp = timestamp
        method = event.get("method")
        if method not in TOOL_METHODS | {"invalid"}:
            raise AgentToolError("tool event journal method is invalid")
        if status_value not in {"complete", "failed"}:
            raise AgentToolError("tool event journal status is invalid")
        elapsed = event.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise AgentToolError("tool event elapsed time is invalid")
        if event.get("charged") is not True:
            raise AgentToolError("tool event must consume the total-call budget")
        method_charged = event.get("method_charged")
        if not isinstance(method_charged, bool):
            raise AgentToolError("tool event method charge flag is invalid")
        if method == "invalid" and method_charged:
            raise AgentToolError("invalid tool method consumed a method budget")
        params = event.get("params")
        if not isinstance(params, dict):
            raise AgentToolError("tool event safe params are malformed")
        _verify_safe_params(str(method), params)
        error_type = event.get("error_type")
        error_message = event.get("error_message")
        if status_value == "complete":
            if error_type is not None or error_message is not None:
                raise AgentToolError("complete tool event contains an error")
            result = event.get("result")
            if not isinstance(result, dict):
                raise AgentToolError("complete tool event result is malformed")
            result_sha256 = event.get("result_sha256")
            if (
                not isinstance(result_sha256, str)
                or len(result_sha256) != 64
                or hashlib.sha256(_canonical_json_bytes(result)).hexdigest()
                != result_sha256
            ):
                raise AgentToolError("tool event result hash differs")
        else:
            if (
                not isinstance(error_type, str)
                or not error_type
                or len(error_type) > 128
                or not isinstance(error_message, str)
                or len(error_message) > 1000
            ):
                raise AgentToolError("failed tool event error is malformed")
        event_sha256 = event.get("event_sha256")
        canonical = dict(event)
        canonical.pop("event_sha256", None)
        if (
            not isinstance(event_sha256, str)
            or len(event_sha256) != 64
            or hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()
            != event_sha256
            or event.get("previous_event_sha256") != previous_sha256
        ):
            raise AgentToolError("tool event journal hash chain is invalid")
        previous_sha256 = event_sha256
        counts["total"] += 1
        if method_charged:
            if method not in TOOL_METHODS:
                raise AgentToolError("charged tool event method is invalid")
            counts[str(method)] += 1
        if method == "finalize_submission" and status_value == "complete":
            successful_finalizations += 1

    if counts["total"] > policy.max_total_calls:
        raise AgentToolError("persisted tool calls exceed the episode budget")
    limits = {
        "search_assets": policy.max_asset_searches,
        "inspect_asset": policy.max_asset_inspections,
        "validate_submission": policy.max_submission_validations,
        "finalize_submission": policy.max_submission_validations,
    }
    for method, limit in limits.items():
        if counts[method] > limit:
            raise AgentToolError(f"persisted {method} calls exceed the budget")
    if successful_finalizations > 1:
        raise AgentToolError("tool journal has multiple successful finalizations")
    if require_finalized and successful_finalizations != 1:
        raise AgentToolError("tool journal does not have exactly one finalization")
    return {
        "schema_version": "sieve_tool_event_journal_verification_v1",
        "valid": True,
        "event_count": counts["total"],
        "charged_counts": counts,
        "successful_finalizations": successful_finalizations,
        "last_event_sha256": previous_sha256,
        "journal_size_bytes": len(raw),
        "journal_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _safe_params(method: str, params: Mapping[str, Any]) -> dict[str, Any]:
    if method == "search_assets":
        return {
            "query": str(params.get("query") or "")[:1000],
            "size_constraint": params.get("size_constraint"),
            "top_k": params.get("top_k"),
        }
    if method == "inspect_asset":
        return {"asset_id": str(params.get("asset_id") or "")[:512]}
    if method in {"validate_submission", "finalize_submission"}:
        return {"submission_path": str(params.get("submission_path") or "")[:512]}
    return {}


def _verify_safe_params(method: str, params: Mapping[str, Any]) -> None:
    expected = {
        "search_assets": {"query", "size_constraint", "top_k"},
        "inspect_asset": {"asset_id"},
        "validate_submission": {"submission_path"},
        "finalize_submission": {"submission_path"},
        "get_task": set(),
        "invalid": set(),
    }[method]
    if set(params) != expected:
        raise AgentToolError("tool event safe-parameter field set differs")
    encoded = _canonical_json_bytes(params)
    if len(encoded) > 8192:
        raise AgentToolError("tool event safe params exceed their bound")


def _public_error_message(exc: Exception) -> str:
    if isinstance(exc, (AgentToolError, ValueError)):
        return str(exc)[:1000]
    return "Agent tool operation failed"


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise AgentToolError(f"{label} keys differ from the fixed protocol")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    if not isinstance(value, dict):
        raise AgentToolError("sealed JSON artifact must be an object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, _json_file_bytes(value), mode=0o600)


def _write_bytes_exclusive(path: Path, value: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AgentToolError(f"refusing to overwrite sealed file: {path.name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    path.chmod(mode)


def _require_real_file_mode(path: Path, expected: int, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise AgentToolError(f"{label} must be a real file")
    if stat.S_IMODE(path.stat().st_mode) != expected:
        raise AgentToolError(f"{label} mode differs")


def _parse_utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise AgentToolError("tool event timestamp is invalid")
    try:
        observed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AgentToolError("tool event timestamp is invalid") from exc
    if observed.tzinfo is None or observed.utcoffset() != timezone.utc.utcoffset(observed):
        raise AgentToolError("tool event timestamp is not UTC")
    return observed


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "AgentToolError",
    "AgentToolPolicy",
    "AgentToolServer",
    "AgentToolSession",
    "AgentToolShutdownError",
    "MAX_REQUEST_BYTES",
    "MAX_TOOL_EVENT_BYTES",
    "TOOL_PROTOCOL_VERSION",
    "validate_task_submission_constraints",
    "verify_tool_event_journal",
]
