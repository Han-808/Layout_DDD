"""Budgeted local tool service for one isolated Agent episode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import socketserver
import tempfile
import threading
import time
from typing import Any, Mapping

from .contracts import ValidatedAgentSubmission, validate_agent_submission


TOOL_PROTOCOL_VERSION = "non_rectangular_agent_tool_protocol_v1"
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_SUBMISSION_BYTES = 32 * 1024 * 1024


class AgentToolError(RuntimeError):
    """Raised for an unauthorized, over-budget, or malformed tool call."""


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
        self.final_submission_path = self.workspace / "final_submission.json"
        self.finalization_path = self.workspace / "finalization.json"
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
        result: Any = None
        try:
            with self._lock:
                self._charge(name)
                charged = True
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
            self._event(
                method=name or "invalid",
                status=status,
                error_type=error_type,
                elapsed_seconds=elapsed,
                charged=charged,
                params=_safe_params(name, arguments),
                result=result,
                error_message=error_message,
            )

    def _charge(self, method: str) -> None:
        if method not in self._counts or method == "total":
            raise AgentToolError(f"unsupported Agent tool method: {method!r}")
        if self._counts["total"] >= self.policy.max_total_calls:
            raise AgentToolError("Agent episode exhausted total tool-call budget")
        limits = {
            "search_assets": self.policy.max_asset_searches,
            "inspect_asset": self.policy.max_asset_inspections,
            "validate_submission": self.policy.max_submission_validations,
            "finalize_submission": self.policy.max_submission_validations,
        }
        limit = limits.get(method)
        if limit is not None and self._counts[method] >= limit:
            raise AgentToolError(f"Agent episode exhausted {method} budget")
        self._counts["total"] += 1
        self._counts[method] += 1

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
        if self.final_submission_path.exists():
            if _load_json(self.final_submission_path) != dict(submission):
                raise AgentToolError("a different final submission is already sealed")
        else:
            _write_json_exclusive(self.final_submission_path, submission)
        expected_submission = _json_file_bytes(submission)
        if self.final_submission_path.read_bytes() != expected_submission:
            raise AgentToolError("sealed submission bytes differ from validated content")
        submission_sha256 = hashlib.sha256(expected_submission).hexdigest()
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
                "schema_version": "sieve_trusted_submission_seal_v1",
                "finalization": stable_current,
            }
            if self.seal_record_path.exists():
                if _load_json(self.seal_record_path) != trusted_record:
                    raise AgentToolError("trusted submission seal differs on resume")
            else:
                _write_json_exclusive(self.seal_record_path, trusted_record)
                self.seal_record_path.chmod(0o400)

    def _restore_counts(self) -> None:
        if not self.events_path.exists():
            return
        if not self.events_path.is_file() or self.events_path.is_symlink():
            raise AgentToolError("tool event journal must be a real file")
        previous_sha256: str | None = None
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AgentToolError("tool event journal contains invalid JSON") from exc
            if not isinstance(event, dict):
                raise AgentToolError("tool event journal entry must be an object")
            event_sha256 = event.get("event_sha256")
            canonical = dict(event)
            canonical.pop("event_sha256", None)
            if (
                not isinstance(event_sha256, str)
                or hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()
                != event_sha256
                or event.get("previous_event_sha256") != previous_sha256
            ):
                raise AgentToolError("tool event journal hash chain is invalid")
            previous_sha256 = event_sha256
            method = event.get("method")
            if event.get("charged") is not True:
                continue
            if method not in self._counts or method == "total":
                raise AgentToolError("charged tool event contains an unknown method")
            self._counts["total"] += 1
            self._counts[str(method)] += 1
        if self._counts["total"] > self.policy.max_total_calls:
            raise AgentToolError("persisted tool calls exceed the episode budget")
        self._last_event_sha256 = previous_sha256

    def _event(self, **payload: Any) -> None:
        result = payload.pop("result", None)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": "non_rectangular_agent_tool_event_v2",
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
        with self._lock:
            if self.events_path.is_symlink():
                raise AgentToolError("tool event journal must not be a symlink")
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            self._last_event_sha256 = str(event["event_sha256"])


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

    def __init__(self, session: AgentToolSession) -> None:
        self.session = session
        self.token = secrets.token_hex(32)
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
                if len(raw) > MAX_REQUEST_BYTES:
                    self._respond({"ok": False, "error_type": "request_too_large"})
                    return
                try:
                    request = json.loads(raw.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise AgentToolError("request root must be an object")
                    if request.get("protocol") != TOOL_PROTOCOL_VERSION:
                        raise AgentToolError("tool protocol version mismatch")
                    if not secrets.compare_digest(str(request.get("token") or ""), owner.token):
                        raise AgentToolError("tool authentication failed")
                    params = request.get("params")
                    if params is not None and not isinstance(params, Mapping):
                        raise AgentToolError("tool params must be an object")
                    result = owner.session.dispatch(str(request.get("method") or ""), params)
                    self._respond({"ok": True, "result": result})
                except Exception as exc:
                    self._respond(
                        {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "message": _public_error_message(exc),
                        }
                    )

            def _respond(self, value: Mapping[str, Any]) -> None:
                self.wfile.write(
                    (json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                )

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
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        if self.socket_path.exists():
            self.socket_path.unlink()
        if self._socket_root.is_dir():
            self._socket_root.rmdir()

    def __enter__(self) -> "AgentToolServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


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


def _public_error_message(exc: Exception) -> str:
    if isinstance(exc, (AgentToolError, ValueError)):
        return str(exc)[:1000]
    return "Agent tool operation failed"


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise AgentToolError(f"{label} keys differ from the fixed protocol")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AgentToolError("sealed JSON artifact must be an object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_file_bytes(value).decode("utf-8")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()


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
    "MAX_REQUEST_BYTES",
    "TOOL_PROTOCOL_VERSION",
    "validate_task_submission_constraints",
]
