"""Budgeted local tool service for one isolated Agent episode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir() or self.workspace.is_symlink():
            raise AgentToolError("Agent workspace must be a real directory")
        self.room_layout = dict(room_layout)
        self.room_program = dict(room_program)
        self.asset_catalog = asset_catalog
        self.task_payload = dict(task_payload)
        self.policy = policy
        self.events_path = self.workspace / "tool_events.jsonl"
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
        charged = False
        try:
            with self._lock:
                self._charge(name)
                charged = True
                if name == "get_task":
                    _exact_keys(arguments, set(), label="get_task params")
                    result: Any = dict(self.task_payload)
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
                    if name == "finalize_submission":
                        self._publish_final(submission, validated)
                    result = validated.public_dict()
                else:
                    raise AgentToolError(f"unsupported Agent tool method: {name!r}")
            status = "complete"
            return result
        except Exception as exc:
            error_type = type(exc).__name__
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
    ) -> None:
        if self.final_submission_path.exists():
            if _load_json(self.final_submission_path) != dict(submission):
                raise AgentToolError("a different final submission is already sealed")
        else:
            _write_json_exclusive(self.final_submission_path, submission)
        finalization = {
            **validated.public_dict(),
            "status": "sealed",
            "catalog_snapshot_id": str(self.asset_catalog.snapshot_id),
            "tool_counts": self.counts(),
        }
        if self.finalization_path.exists():
            if _load_json(self.finalization_path) != finalization:
                raise AgentToolError("finalization identity differs on resume")
        else:
            _write_json_exclusive(self.finalization_path, finalization)

    def _restore_counts(self) -> None:
        if not self.events_path.exists():
            return
        if not self.events_path.is_file() or self.events_path.is_symlink():
            raise AgentToolError("tool event journal must be a real file")
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AgentToolError("tool event journal contains invalid JSON") from exc
            method = event.get("method") if isinstance(event, dict) else None
            if not isinstance(event, dict) or event.get("charged") is not True:
                continue
            if method not in self._counts or method == "total":
                raise AgentToolError("charged tool event contains an unknown method")
            self._counts["total"] += 1
            self._counts[str(method)] += 1
        if self._counts["total"] > self.policy.max_total_calls:
            raise AgentToolError("persisted tool calls exceed the episode budget")

    def _event(self, **payload: Any) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": "non_rectangular_agent_tool_event_v1",
            **payload,
        }
        with self._lock:
            if self.events_path.is_symlink():
                raise AgentToolError("tool event journal must not be a symlink")
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()


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
    encoded = json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()


__all__ = [
    "AgentToolError",
    "AgentToolPolicy",
    "AgentToolServer",
    "AgentToolSession",
    "MAX_REQUEST_BYTES",
    "TOOL_PROTOCOL_VERSION",
]
