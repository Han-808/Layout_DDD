#!/usr/bin/env python3
"""Per-episode, route-aware model gateway with bounded safe retries.

The Agent sees only a loopback OpenAI-compatible endpoint and a short-lived
capability. Upstream endpoints and credentials stay in the trusted host
process. The gateway owns provider reasoning fields so Pi compatibility
behavior cannot silently change the registered compute treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid
from urllib.parse import urlencode, urlsplit

from api_profiles import ModelProfile, RouteProfile, RouteRuntimeBinding


MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_IDENTITY_PREFACE_BYTES = 1024 * 1024
MAX_REASONING_BRIDGE_BYTES = 4 * 1024 * 1024
MAX_REASONING_DETAIL_BLOCKS = 128
MAX_CONNECT_TIMEOUT_SECONDS = 5.0
GATEWAY_SHUTDOWN_TIMEOUT_SECONDS = 15.0
ANTHROPIC_REASONING_DETAIL_FORMAT = "sieve-anthropic-thinking-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PORTABLE_ERROR_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")


class GatewayError(RuntimeError):
    """Raised before launch when a scoped gateway cannot be made safe."""


class GatewayShutdownError(GatewayError):
    """Fatal barrier failure: a later episode must not be started."""


def completion_requires_transport_recycle(
    completion: Mapping[str, Any] | None,
) -> bool:
    """Fail safely when an accepted request lacks a proven terminal state."""

    if not isinstance(completion, Mapping):
        return False
    if completion.get("transport_recycle_required") is True:
        return True
    request_count = completion.get("request_count")
    terminal = completion.get("terminal_statuses")
    return bool(
        isinstance(request_count, int)
        and not isinstance(request_count, bool)
        and request_count > 0
        and (
            not isinstance(terminal, Mapping)
            or len(terminal) != request_count
            or completion.get("inflight") is not False
        )
    )


@dataclass(frozen=True)
class ForwardResult:
    status: str
    http_status: int | None
    downstream_started: bool
    identity_matches: bool | None
    retryable: bool
    error_type: str | None = None
    transport_recycle_required: bool = False


class SharedCooldownGate:
    """Coordinate one API-family cooldown across active model gateways."""

    def __init__(self, *, monotonic: Any = time.monotonic) -> None:
        self._monotonic = monotonic
        self._condition = threading.Condition()
        self._blocked_until = 0.0

    def wait(self, *, cancel_event: threading.Event | None = None) -> float | None:
        """Wait until the shared API family may send another attempt."""

        waited = 0.0
        with self._condition:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return None
                remaining = self._blocked_until - float(self._monotonic())
                if remaining <= 0:
                    return waited
                started = float(self._monotonic())
                self._condition.wait(timeout=min(remaining, 0.1))
                waited += max(0.0, float(self._monotonic()) - started)

    def trigger(self, seconds: float) -> float:
        """Extend, never shorten, the shared cooldown window."""

        if not math.isfinite(float(seconds)) or float(seconds) < 0:
            raise GatewayError("cooldown seconds must be finite and non-negative")
        with self._condition:
            self._blocked_until = max(
                self._blocked_until,
                float(self._monotonic()) + float(seconds),
            )
            self._condition.notify_all()
            return self._blocked_until


class ScopedModelGateway:
    """Expose exactly one registered path/model/request budget on loopback."""

    def __init__(
        self,
        *,
        route: RouteProfile,
        model: ModelProfile,
        runtime_binding: RouteRuntimeBinding,
        runtime_credential: str,
        max_requests: int,
        event_path: str | Path | None = None,
        cooldown_gate: SharedCooldownGate | None = None,
        session_id: str | None = None,
        required_system_prompt_sha256s: Sequence[str] | None = None,
        system_prompt_rewrites: Mapping[str, str] | None = None,
        managed_transport_ambiguity_probe: Callable[[bool], bool] | None = None,
    ) -> None:
        if route.route_profile_id != model.route_profile_id:
            raise GatewayError("model and route profile differ")
        if route.api_family_id != model.api_family_id:
            raise GatewayError("model and route API family differ")
        if runtime_binding.route_profile_id != route.route_profile_id:
            raise GatewayError("runtime binding and route profile differ")
        if (
            not isinstance(runtime_credential, str)
            or not runtime_credential
            or "\r" in runtime_credential
            or "\n" in runtime_credential
        ):
            raise GatewayError("runtime credential is missing or malformed")
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
            raise GatewayError("max_requests must be positive")
        upstream = urlsplit(runtime_binding.upstream_base_url.rstrip("/"))
        if upstream.scheme not in {"https", "http"} or not upstream.hostname:
            raise GatewayError("upstream base URL must be HTTP(S)")
        if upstream.scheme == "http" and not runtime_binding.allow_insecure_upstream:
            raise GatewayError("HTTP upstream lacks explicit trusted-network acknowledgement")
        if upstream.username or upstream.password or upstream.query or upstream.fragment:
            raise GatewayError("upstream base URL must not carry credentials or query data")
        try:
            upstream_port = upstream.port
        except ValueError as exc:
            raise GatewayError("upstream base URL contains an invalid port") from exc
        if upstream_port is not None and not 1 <= upstream_port <= 65535:
            raise GatewayError("upstream base URL contains an invalid port")
        if route.transport_adapter_id == "managed_tokenhub_litellm_v1":
            if (
                runtime_binding.managed_adapter_id
                != "managed_tokenhub_litellm_v1"
                or upstream.scheme != "http"
                or upstream.hostname != "127.0.0.1"
                or upstream_port is None
                or upstream.path.rstrip("/") != "/v1"
                or not runtime_binding.allow_insecure_upstream
            ):
                raise GatewayError(
                    "managed TokenHub route requires its host-owned loopback adapter"
                )
            if managed_transport_ambiguity_probe is None or not callable(
                managed_transport_ambiguity_probe
            ):
                raise GatewayError(
                    "managed TokenHub route requires its ambiguity sideband"
                )
        elif runtime_binding.managed_adapter_id is not None:
            raise GatewayError("direct route cannot carry a managed-adapter binding")
        elif managed_transport_ambiguity_probe is not None:
            raise GatewayError("direct route cannot carry a managed ambiguity sideband")
        if not 0 < model.request_timeout_seconds <= 7200:
            raise GatewayError("request timeout must be in (0, 7200]")
        self.route = route
        self.model = model
        self.runtime_binding = runtime_binding
        self.upstream = upstream
        self.runtime_credential = runtime_credential
        self.max_requests = max_requests
        self.cooldown_gate = cooldown_gate or SharedCooldownGate()
        self.managed_transport_ambiguity_probe = (
            managed_transport_ambiguity_probe
        )
        prompt_hashes = tuple(required_system_prompt_sha256s or ())
        if len(prompt_hashes) != len(set(prompt_hashes)) or any(
            not isinstance(value, str) or SHA256.fullmatch(value) is None
            for value in prompt_hashes
        ):
            raise GatewayError(
                "required system-prompt hashes must be unique lowercase SHA-256"
            )
        self.required_system_prompt_sha256s = frozenset(prompt_hashes)
        rewrites = dict(system_prompt_rewrites or {})
        if any(
            not isinstance(key, str)
            or SHA256.fullmatch(key) is None
            or not isinstance(value, str)
            or not value
            for key, value in rewrites.items()
        ):
            raise GatewayError("system-prompt rewrite map is malformed")
        if rewrites and set(rewrites) != set(self.required_system_prompt_sha256s):
            raise GatewayError("system-prompt rewrite keys differ from allowed hashes")
        self.system_prompt_rewrites = rewrites
        self.session_id = session_id or secrets.token_hex(24)
        if not self.session_id or any(character in self.session_id for character in "\r\n"):
            raise GatewayError("session identity is malformed")
        self.capability_token = secrets.token_hex(32)
        self._api2_cache_namespace = secrets.token_hex(8)
        self._api2_physical_attempt = 0
        self._api2_cache_lock = threading.Lock()
        self._request_count = 0
        self._request_lock = threading.Lock()
        self._logical_request_results: dict[int, str] = {}
        self._logical_request_tool_history: dict[int, bool] = {}
        self._logical_request_reasoning_replay: dict[int, bool | None] = {}
        self._logical_request_identity_matches: dict[int, bool | None] = {}
        self._logical_request_transport_recycle: dict[int, bool] = {}
        # Once a request crosses an ambiguous post-send/stream boundary, this
        # gateway generation is permanently poisoned.  The supervisor can
        # only recover after this gateway is fully closed and (for managed
        # transports) the nested process has been replaced.  A later Pi turn
        # must never reach the same possibly-live upstream generation.
        self._transport_poisoned = False
        self._inflight_lock = threading.Lock()
        self._lifecycle = threading.Condition()
        self._active_handlers = 0
        self._active_connections: set[http.client.HTTPConnection] = set()
        self._active_downstream_sockets: set[socket.socket] = set()
        self._closing = False
        self._closing_event = threading.Event()
        self._closed = False
        self._started = False
        self._close_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._previous_event_hash = "0" * 64
        self.event_path = self._initialize_event_path(event_path)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_POST(self) -> None:  # noqa: N802
                owner._handle_tracked(self)

            def do_GET(self) -> None:  # noqa: N802
                owner._reject(self, 405, "method_not_allowed")

            def do_PUT(self) -> None:  # noqa: N802
                owner._reject(self, 405, "method_not_allowed")

            def do_DELETE(self) -> None:  # noqa: N802
                owner._reject(self, 405, "method_not_allowed")

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # A close is not successful until every request handler and upstream
        # socket is gone.  This prevents a timed-out episode from continuing
        # to consume an API route while the next episode begins.
        self._server.daemon_threads = False
        self._server.block_on_close = False
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="sieve-scoped-model-gateway",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def endpoint_address(self) -> str:
        return f"localhost:{self.port}"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def request_count(self) -> int:
        with self._request_lock:
            return self._request_count

    def completion_report(self) -> dict[str, Any]:
        """Return a sanitized, race-safe terminal-health snapshot.

        A downstream SSE connection can end after Pi has received enough data
        to exit zero even though the registered stream contract was truncated.
        Episode acceptance therefore depends on this host-side record, never
        on the Agent process exit code alone.
        """

        with self._request_lock:
            request_count = self._request_count
            results = dict(self._logical_request_results)
            tool_history = dict(self._logical_request_tool_history)
            reasoning_replay = dict(self._logical_request_reasoning_replay)
            identities = dict(self._logical_request_identity_matches)
            recycle = dict(self._logical_request_transport_recycle)
        with self._event_lock:
            event_terminal_sha256 = self._previous_event_hash
        inflight = self._inflight_lock.locked()
        unresolved_accepted_request = bool(
            request_count > len(results) or inflight
        )
        incomplete = [
            request_id
            for request_id in range(1, request_count + 1)
            if results.get(request_id) != "complete"
        ]
        return {
            "schema_version": "sieve_model_gateway_completion_v3",
            "request_count": request_count,
            "complete_request_count": sum(
                1 for value in results.values() if value == "complete"
            ),
            "incomplete_logical_requests": incomplete,
            "terminal_statuses": {
                str(key): results[key] for key in sorted(results)
            },
            "tool_history_by_request": {
                str(key): tool_history[key] for key in sorted(tool_history)
            },
            "reasoning_replay_by_request": {
                str(key): reasoning_replay[key] for key in sorted(reasoning_replay)
            },
            "identity_matches_by_request": {
                str(key): identities[key] for key in sorted(identities)
            },
            "transport_recycle_required_by_request": {
                str(key): recycle[key] for key in sorted(recycle)
            },
            "transport_recycle_required": bool(
                any(recycle.values()) or unresolved_accepted_request
            ),
            "event_terminal_sha256": event_terminal_sha256,
            "inflight": inflight,
            "all_logical_requests_complete": (
                request_count > 0
                and not incomplete
                and len(results) == request_count
                and not inflight
            ),
        }

    def wait_for_completion_report(
        self, *, timeout_seconds: float = 5.0
    ) -> dict[str, Any]:
        """Wait briefly for the handler's final lock release, then snapshot."""

        deadline = time.monotonic() + timeout_seconds
        while True:
            report = self.completion_report()
            terminal_count = len(report["terminal_statuses"])
            if terminal_count == report["request_count"] and not report["inflight"]:
                return report
            if time.monotonic() >= deadline:
                return report
            time.sleep(0.01)

    def assert_all_logical_requests_complete(self) -> dict[str, Any]:
        report = self.wait_for_completion_report()
        if not report["all_logical_requests_complete"]:
            raise GatewayError("model_gateway_request_chain_incomplete")
        return report

    def start(self) -> "ScopedModelGateway":
        with self._close_lock:
            with self._lifecycle:
                if self._started or self._closing or self._closed:
                    raise GatewayError("model gateway lifecycle is invalid")
            try:
                self._thread.start()
            except BaseException as exc:
                # A context manager whose ``__enter__`` fails never receives
                # ``__exit__``.  Release the already-bound listener and revoke
                # every capability here.  Do not call BaseServer.shutdown:
                # serve_forever never started and shutdown would wait forever.
                failure: BaseException = exc
                with self._lifecycle:
                    self._closing = True
                    self._closing_event.set()
                try:
                    self._server.server_close()
                except BaseException as close_exc:
                    failure = close_exc
                try:
                    if self.event_path is not None and self.event_path.is_file():
                        self.event_path.chmod(0o444)
                except BaseException as seal_exc:
                    failure = seal_exc
                self.runtime_credential = ""
                self.capability_token = ""
                self.session_id = ""
                self._api2_cache_namespace = ""
                with self._lifecycle:
                    self._closed = True
                    self._lifecycle.notify_all()
                raise GatewayError("model_gateway_thread_start_failed") from failure
            with self._lifecycle:
                self._started = True
        return self

    def close(self) -> None:
        # Serialize explicit close, context-manager close, and cleanup after an
        # exception. BaseServer.shutdown itself is not safe to run twice in
        # parallel, while sequential close must remain idempotent.
        with self._close_lock:
            self._close_once()

    def _close_once(self) -> None:
        with self._lifecycle:
            if self._closed:
                return
            self._closing = True
            self._closing_event.set()
            started = self._started
            connections = tuple(self._active_connections)
            downstream_sockets = tuple(self._active_downstream_sockets)
        failure: BaseException | None = None
        try:
            if started:
                self._server.shutdown()
            if self._thread.is_alive():
                self._thread.join(timeout=5.0)
            deadline = time.monotonic() + GATEWAY_SHUTDOWN_TIMEOUT_SECONDS
            while True:
                for connection in connections:
                    self._abort_connection(connection)
                for downstream in downstream_sockets:
                    self._abort_downstream(downstream)
                with self._lifecycle:
                    if (
                        not self._active_handlers
                        and not self._active_connections
                        and not self._active_downstream_sockets
                    ):
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        failure = GatewayShutdownError(
                            "model_gateway_shutdown_incomplete"
                        )
                        break
                    self._lifecycle.wait(timeout=min(0.1, remaining))
                    connections = tuple(self._active_connections)
                    downstream_sockets = tuple(self._active_downstream_sockets)
        except BaseException as exc:
            failure = exc
        finally:
            for connection in connections:
                self._abort_connection(connection)
            for downstream in downstream_sockets:
                self._abort_downstream(downstream)
            try:
                self._server.server_close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
            try:
                if self.event_path is not None and self.event_path.is_file():
                    self.event_path.chmod(0o444)
            except BaseException as exc:
                if failure is None:
                    failure = exc
            self.runtime_credential = ""
            self.capability_token = ""
            self.session_id = ""
            self._api2_cache_namespace = ""
            with self._lifecycle:
                self._closed = True
                self._lifecycle.notify_all()
        if failure is not None:
            if isinstance(failure, GatewayShutdownError):
                raise failure
            raise GatewayShutdownError("model_gateway_shutdown_failed") from failure

    def _handle_tracked(self, handler: BaseHTTPRequestHandler) -> None:
        downstream = handler.connection
        with self._lifecycle:
            if self._closing:
                self._abort_downstream(downstream)
                return
            self._active_handlers += 1
            self._active_downstream_sockets.add(downstream)
        try:
            self._handle(handler)
        finally:
            with self._lifecycle:
                self._active_handlers -= 1
                self._active_downstream_sockets.discard(downstream)
                self._lifecycle.notify_all()

    def __enter__(self) -> "ScopedModelGateway":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def public_dict(self) -> dict[str, Any]:
        return build_gateway_public_record(
            route=self.route,
            model=self.model,
            max_requests=self.max_requests,
            required_system_prompt_sha256s=self.required_system_prompt_sha256s,
            system_prompt_rewrites=self.system_prompt_rewrites,
        )

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        authorization = handler.headers.get("Authorization", "")
        expected = f"Bearer {self.capability_token}"
        if not secrets.compare_digest(authorization, expected):
            self._reject(handler, 401, "invalid_episode_capability")
            return
        if handler.path != self.route.client_path:
            self._reject(handler, 404, "endpoint_not_allowed")
            return
        with self._request_lock:
            transport_poisoned = self._transport_poisoned
        if transport_poisoned:
            self._reject(handler, 503, "transport_recovery_required")
            return
        length_text = handler.headers.get("Content-Length", "")
        try:
            length = int(length_text)
        except ValueError:
            self._reject(handler, 411, "content_length_required")
            return
        if length < 1 or length > MAX_REQUEST_BYTES:
            self._reject(handler, 413, "request_too_large")
            return
        body = handler.rfile.read(length)
        try:
            payload = json.loads(
                body.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, ValueError):
            self._reject(handler, 400, "invalid_json")
            return
        if not isinstance(payload, dict) or payload.get("model") != self.model.client_wire_model:
            self._reject(handler, 400, "model_identity_mismatch")
            return
        if payload.get("stream") is not True:
            self._reject(handler, 400, "streaming_required")
            return
        # Preserve useful identity/stream contract errors, but reject an
        # otherwise eligible request as soon as its logical episode budget is
        # known to be exhausted.  Full payload normalization may itself reject
        # fields that are irrelevant once no further model request is allowed.
        # The second check below remains the authoritative atomic check before
        # incrementing the request count.
        with self._request_lock:
            budget_exhausted = self._request_count >= self.max_requests
        if budget_exhausted:
            self._reject(handler, 429, "episode_request_budget_exhausted")
            return
        if self.required_system_prompt_sha256s:
            observed_prompt = _system_prompt_sha256(
                payload,
                api_protocol=self.route.pi_api_protocol,
            )
            if observed_prompt not in self.required_system_prompt_sha256s:
                self._reject(handler, 400, "system_prompt_hash_mismatch")
                return
            if self.system_prompt_rewrites:
                payload = _replace_system_prompt(
                    payload,
                    api_protocol=self.route.pi_api_protocol,
                    replacement=self.system_prompt_rewrites[observed_prompt],
                )
        try:
            normalized = self._normalize_payload(payload)
        except GatewayError as exc:
            self._reject(handler, 400, str(exc))
            return
        try:
            has_tool_history, reasoning_replay = _reasoning_replay_state(
                normalized,
                api_protocol=self.route.pi_api_protocol,
                require_signed_anthropic=self._anthropic_reasoning_bridge_enabled(),
            )
        except GatewayError as exc:
            self._reject(handler, 400, str(exc))
            return
        if (
            has_tool_history
            and self.model.reasoning.preserve_across_tool_turns
            and not reasoning_replay
        ):
            self._reject(handler, 400, "required_reasoning_replay_missing")
            return
        if not self._inflight_lock.acquire(blocking=False):
            self._reject(handler, 409, "concurrent_model_request_forbidden")
            return
        try:
            with self._request_lock:
                if self._transport_poisoned:
                    self._reject(handler, 503, "transport_recovery_required")
                    return
                if self._request_count >= self.max_requests:
                    self._reject(handler, 429, "episode_request_budget_exhausted")
                    return
                self._request_count += 1
                logical_request = self._request_count
                self._logical_request_tool_history[logical_request] = has_tool_history
                self._logical_request_reasoning_replay[logical_request] = (
                    reasoning_replay if has_tool_history else None
                )
            body = json.dumps(
                normalized,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._forward_with_retry(
                handler,
                body,
                logical_request,
                has_tool_history=has_tool_history,
                reasoning_replay=(reasoning_replay if has_tool_history else None),
            )
        finally:
            self._inflight_lock.release()

    def _normalize_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        if self._anthropic_reasoning_bridge_enabled():
            normalized = _restore_anthropic_signed_reasoning(normalized)
        normalized["model"] = self.model.upstream_wire_model
        normalized["stream"] = True
        normalized.pop("reasoning_effort", None)
        normalized.pop("reasoning", None)
        normalized.pop("thinking", None)
        normalized.update(self.model.reasoning.fixed_request_fields())
        if self.model.temperature is None:
            normalized.pop("temperature", None)
        else:
            normalized["temperature"] = self.model.temperature
        maximum = self.model.pi.maximum_output_tokens
        requested_maximum = _requested_maximum_output_tokens(
            normalized,
            api_protocol=self.route.pi_api_protocol,
            ceiling=maximum,
        )
        if self.route.pi_api_protocol == "openai-completions":
            normalized.pop("max_completion_tokens", None)
            normalized.pop("max_output_tokens", None)
            normalized["max_tokens"] = requested_maximum
        elif self.route.pi_api_protocol == "openai-responses":
            # Pi 0.85 derives a prompt cache key from its internal session ID
            # even under --no-session. API2's frozen Responses adapter does
            # not own that cache identity, so never leak or depend on it.
            normalized.pop("prompt_cache_key", None)
            normalized.pop("prompt_cache_retention", None)
            normalized.pop("prompt_cache_options", None)
            normalized.pop("max_tokens", None)
            normalized.pop("max_completion_tokens", None)
            normalized["max_output_tokens"] = requested_maximum
        else:
            raise GatewayError("unsupported_route_protocol")
        return normalized

    def _anthropic_reasoning_bridge_enabled(self) -> bool:
        return bool(
            self.route.transport_adapter_id == "managed_tokenhub_litellm_v1"
            and self.route.pi_api_protocol == "openai-completions"
            and self.model.reasoning.preserve_across_tool_turns
        )

    def _forward_with_retry(
        self,
        handler: BaseHTTPRequestHandler,
        body: bytes,
        logical_request: int,
        *,
        has_tool_history: bool,
        reasoning_replay: bool | None,
    ) -> None:
        maximum_attempts = self.model.retry.maximum_attempts
        for attempt in range(1, maximum_attempts + 1):
            cooldown_wait = self.cooldown_gate.wait(
                cancel_event=self._closing_event
            )
            if cooldown_wait is None:
                self._record_event(
                    {
                        "event": "logical_request_cancelled_before_attempt",
                        "has_tool_history": has_tool_history,
                        "logical_request": logical_request,
                        "reasoning_replay_present": reasoning_replay,
                        "status": "model_gateway_closing",
                        "transport_recycle_required": False,
                    }
                )
                self._record_logical_result(
                    logical_request, "model_gateway_closing", None, False
                )
                return
            self._record_event(
                {
                    "attempt": attempt,
                    "cooldown_wait_seconds": round(cooldown_wait, 6),
                    "event": "upstream_attempt_started",
                    "has_tool_history": has_tool_history,
                    "logical_request": logical_request,
                    "reasoning_replay_present": reasoning_replay,
                }
            )
            started = time.monotonic()
            result = self._forward_once(handler, body)
            result = self._apply_managed_ambiguity_sideband(result)
            elapsed = max(0.0, time.monotonic() - started)
            retry = result.retryable and attempt < maximum_attempts
            self._record_event(
                {
                    "attempt": attempt,
                    "cooldown_wait_seconds": round(cooldown_wait, 6),
                    "downstream_started": result.downstream_started,
                    "elapsed_seconds": round(elapsed, 6),
                    "error_type": result.error_type,
                    "event": "upstream_attempt_result",
                    "http_status": result.http_status,
                    "identity_matches": result.identity_matches,
                    "logical_request": logical_request,
                    "retry_scheduled": retry,
                    "status": result.status,
                    "transport_recycle_required": (
                        result.transport_recycle_required
                    ),
                }
            )
            if result.status == "complete":
                self._record_logical_result(
                    logical_request,
                    result.status,
                    result.identity_matches,
                    result.transport_recycle_required,
                )
                return
            if result.retryable:
                # A final exhausted 429/5xx still blocks the shared API family;
                # otherwise the next scene/model would immediately hammer the
                # same unhealthy route.
                self.cooldown_gate.trigger(self.model.retry.retry_delay_seconds)
            if retry:
                continue
            if not result.downstream_started:
                status = result.http_status if result.http_status is not None else 502
                if status < 400 or status > 599:
                    status = 502
                self._reject(handler, status, result.status)
            self._record_logical_result(
                logical_request,
                result.status,
                result.identity_matches,
                result.transport_recycle_required,
            )
            return

    def _apply_managed_ambiguity_sideband(
        self, result: ForwardResult
    ) -> ForwardResult:
        """Turn a relay post-send hazard into a terminal logical request.

        LiteLLM can normalize a relay failure into an ordinary HTTP 5xx.  The
        host-owned relay therefore exposes a private in-process sideband.  It
        is checked before any retry decision so an already-sent provider call
        can never be mistaken for a safe transient response.
        """

        probe = self.managed_transport_ambiguity_probe
        if probe is None or result.transport_recycle_required:
            return result
        try:
            detected = probe(result.status == "complete")
        except Exception as exc:
            return ForwardResult(
                status="ambiguous_managed_transport_probe_failure",
                http_status=result.http_status,
                downstream_started=result.downstream_started,
                identity_matches=result.identity_matches,
                retryable=False,
                error_type=type(exc).__name__,
                transport_recycle_required=True,
            )
        if not isinstance(detected, bool):
            return ForwardResult(
                status="ambiguous_managed_transport_probe_invalid",
                http_status=result.http_status,
                downstream_started=result.downstream_started,
                identity_matches=result.identity_matches,
                retryable=False,
                transport_recycle_required=True,
            )
        if not detected:
            return result
        return ForwardResult(
            status="ambiguous_managed_transport_sideband",
            http_status=result.http_status,
            downstream_started=result.downstream_started,
            identity_matches=result.identity_matches,
            retryable=False,
            error_type=result.error_type,
            transport_recycle_required=True,
        )

    def _record_logical_result(
        self,
        logical_request: int,
        status: str,
        identity_matches: bool | None,
        transport_recycle_required: bool,
    ) -> None:
        with self._request_lock:
            prior = self._logical_request_results.get(logical_request)
            if prior is not None and prior != status:
                raise GatewayError("logical request terminal status changed")
            self._logical_request_results[logical_request] = status
            self._logical_request_identity_matches[logical_request] = identity_matches
            self._logical_request_transport_recycle[logical_request] = (
                transport_recycle_required
            )
            if transport_recycle_required:
                self._transport_poisoned = True

    def _forward_once(
        self, handler: BaseHTTPRequestHandler, body: bytes
    ) -> ForwardResult:
        try:
            connection = self._connection()
        except Exception as exc:
            return ForwardResult(
                status="upstream_connect_setup_failure",
                http_status=None,
                downstream_started=False,
                identity_matches=None,
                retryable=self.model.retry.retry_transport_failures,
                error_type=type(exc).__name__,
            )
        with self._lifecycle:
            if self._closing:
                return ForwardResult(
                    status="model_gateway_closing",
                    http_status=None,
                    downstream_started=False,
                    identity_matches=None,
                    retryable=False,
                )
            self._active_connections.add(connection)
        try:
            try:
                connection.connect()
            except Exception as exc:
                return ForwardResult(
                    status="upstream_connect_failure",
                    http_status=None,
                    downstream_started=False,
                    identity_matches=None,
                    retryable=self.model.retry.retry_transport_failures,
                    error_type=type(exc).__name__,
                )
            with self._lifecycle:
                if self._closing:
                    return ForwardResult(
                        status="model_gateway_closing",
                        http_status=None,
                        downstream_started=False,
                        identity_matches=None,
                        retryable=False,
                    )
            if connection.sock is not None:
                connection.sock.settimeout(self.model.request_timeout_seconds)
            headers = self._upstream_headers(handler)
            try:
                connection.request(
                    "POST",
                    self._upstream_path(),
                    body=body,
                    headers=headers,
                )
                response = connection.getresponse()
            except Exception as exc:
                # Request bytes may already have left the host. Retrying could
                # duplicate a live model call, so this is terminal and ambiguous.
                return ForwardResult(
                    status="ambiguous_upstream_transport",
                    http_status=None,
                    downstream_started=False,
                    identity_matches=None,
                    retryable=False,
                    error_type=type(exc).__name__,
                    transport_recycle_required=True,
                )
            if not 200 <= response.status <= 299:
                status = int(response.status)
                return ForwardResult(
                    status="upstream_http_failure",
                    http_status=status,
                    downstream_started=False,
                    identity_matches=None,
                    retryable=status in self.model.retry.retryable_http_statuses,
                )
            content_type = str(response.getheader("Content-Type") or "")
            if content_type.split(";", 1)[0].strip().lower() != "text/event-stream":
                return ForwardResult(
                    status="upstream_stream_content_type_mismatch",
                    http_status=502,
                    downstream_started=False,
                    identity_matches=None,
                    retryable=False,
                    transport_recycle_required=True,
                )
            try:
                preface, identity_matches = self._identity_preface(response)
            except GatewayError as exc:
                return ForwardResult(
                    status=str(exc),
                    http_status=502,
                    downstream_started=False,
                    identity_matches=False,
                    retryable=False,
                    transport_recycle_required=True,
                )
            except Exception as exc:
                # A successful response already exists and the upstream may
                # still be generating.  Reading its identity preface is past
                # the unambiguous retry boundary.
                return ForwardResult(
                    status="ambiguous_identity_stream",
                    http_status=int(response.status),
                    downstream_started=False,
                    identity_matches=None,
                    retryable=False,
                    error_type=type(exc).__name__,
                    transport_recycle_required=True,
                )
            validator = _SseContractValidator(self.route.response_contract)
            reasoning_bridge = (
                _AnthropicThinkingSseBridge()
                if self._anthropic_reasoning_bridge_enabled()
                else None
            )
            try:
                downstream_preface = (
                    reasoning_bridge.feed(preface)
                    if reasoning_bridge is not None
                    else preface
                )
                validator.feed(downstream_preface)
            except GatewayError as exc:
                return ForwardResult(
                    status=str(exc),
                    http_status=502,
                    downstream_started=False,
                    identity_matches=identity_matches,
                    retryable=False,
                    transport_recycle_required=True,
                )
            self._begin_downstream(handler, response)
            try:
                if downstream_preface:
                    handler.wfile.write(downstream_preface)
                    handler.wfile.flush()
                while True:
                    chunk = _read_response_chunk(response, 64 * 1024)
                    if not chunk:
                        break
                    downstream_chunk = (
                        reasoning_bridge.feed(chunk)
                        if reasoning_bridge is not None
                        else chunk
                    )
                    validator.feed(downstream_chunk)
                    if downstream_chunk:
                        handler.wfile.write(downstream_chunk)
                        handler.wfile.flush()
                if reasoning_bridge is not None:
                    downstream_tail = reasoning_bridge.finish()
                    validator.feed(downstream_tail)
                    if downstream_tail:
                        handler.wfile.write(downstream_tail)
                        handler.wfile.flush()
                validator.finish()
            except GatewayError as exc:
                return ForwardResult(
                    status=str(exc),
                    http_status=int(response.status),
                    downstream_started=True,
                    identity_matches=identity_matches,
                    retryable=False,
                    transport_recycle_required=True,
                )
            except Exception as exc:
                return ForwardResult(
                    status="ambiguous_stream_failure",
                    http_status=int(response.status),
                    downstream_started=True,
                    identity_matches=identity_matches,
                    retryable=False,
                    error_type=type(exc).__name__,
                    transport_recycle_required=True,
                )
            return ForwardResult(
                status="complete",
                http_status=int(response.status),
                downstream_started=True,
                identity_matches=identity_matches,
                retryable=False,
            )
        finally:
            with self._lifecycle:
                self._active_connections.discard(connection)
                self._lifecycle.notify_all()
            try:
                connection.close()
            except Exception:
                pass

    @staticmethod
    def _abort_connection(connection: http.client.HTTPConnection) -> None:
        """Interrupt a handler blocked in connect/read without sending data."""

        sock = getattr(connection, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            connection.close()
        except Exception:
            pass

    @staticmethod
    def _abort_downstream(downstream: socket.socket) -> None:
        """Interrupt a handler blocked while writing to the local Pi client."""

        try:
            downstream.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            downstream.close()
        except OSError:
            pass

    def _connection(self) -> http.client.HTTPConnection:
        connection_cls = (
            http.client.HTTPSConnection
            if self.upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        port = self.upstream.port or (443 if self.upstream.scheme == "https" else 80)
        return connection_cls(
            self.upstream.hostname,
            port,
            timeout=min(
                float(self.model.request_timeout_seconds),
                MAX_CONNECT_TIMEOUT_SECONDS,
            ),
        )

    def _upstream_path(self) -> str:
        return self.upstream.path.rstrip("/") + self.route.upstream_path

    def _upstream_headers(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        headers = {
            # All supported route contracts are SSE.  Do not let an Agent
            # weaken or alter the upstream protocol through a client header.
            "Accept": "text/event-stream",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "sieve-pi-agent-gateway/3.0",
        }
        if self.model.user_agent_suffix:
            headers["User-Agent"] += f" {self.model.user_agent_suffix}"
        strategy = self.route.auth_strategy
        if strategy == "standard_bearer_v1":
            headers["Authorization"] = f"Bearer {self.runtime_credential}"
        elif strategy == "api2_bearer_query_v1":
            query = dict(self.model.auth_query_parameters)
            # Match the frozen API2 adapter: each upstream attempt receives a
            # fresh cache identity, including infrastructure retries.
            query["cache_task_id"] = self._make_api2_cache_task_id()
            headers["Authorization"] = (
                f"Bearer {self.runtime_credential}?{urlencode(query)}"
            )
        elif strategy == "api3_bearer_session_v1":
            if not self.model.api3_strategy_type:
                raise GatewayError("API3 StrategyType is missing")
            headers["Authorization"] = f"Bearer {self.runtime_credential}"
            # The successful frozen API3 generation runner assigns a fresh
            # RFC-4122 identity to every physical upstream attempt.  In
            # particular, an explicit retry must not reuse the SessionID of
            # the failed attempt.  This value is routing-only and is never
            # persisted in the public/audit artifacts.
            headers["SessionID"] = str(uuid.uuid4())
            headers["StrategyType"] = self.model.api3_strategy_type
        else:
            raise GatewayError("unsupported upstream authentication strategy")
        return headers

    def _identity_preface(
        self, response: http.client.HTTPResponse
    ) -> tuple[bytes, bool | None]:
        if not self.model.response_identity_required:
            return b"", None
        buffer = bytearray()
        while len(buffer) <= MAX_IDENTITY_PREFACE_BYTES:
            identity = _extract_response_identity(bytes(buffer))
            if identity is not None:
                if identity not in self.model.accepted_response_models:
                    raise GatewayError("upstream_model_identity_mismatch")
                return bytes(buffer), True
            chunk = _read_response_chunk(response, 16 * 1024)
            if not chunk:
                break
            buffer.extend(chunk)
        raise GatewayError("upstream_model_identity_missing")

    @staticmethod
    def _begin_downstream(
        handler: BaseHTTPRequestHandler, response: http.client.HTTPResponse
    ) -> None:
        handler.send_response(int(response.status))
        for name in ("Content-Type", "Cache-Control"):
            value = response.getheader(name)
            if value:
                handler.send_header(name, value)
        handler.end_headers()

    def _make_api2_cache_task_id(self) -> str:
        if self.route.auth_strategy != "api2_bearer_query_v1":
            return ""
        with self._api2_cache_lock:
            if not self._api2_cache_namespace:
                raise GatewayError("API2 cache identity namespace is unavailable")
            self._api2_physical_attempt += 1
            if self._api2_physical_attempt > 0xFFFFFFFFFFFFFFFF:
                raise GatewayError("API2 physical-attempt counter exhausted")
            # Exactly 32 lowercase hex characters.  The first half is a
            # per-gateway random namespace and the second half is a monotonic
            # physical-attempt counter.  Unlike a clock-only hash, uniqueness
            # within the gateway is deterministic even if time_ns repeats.
            return (
                self._api2_cache_namespace
                + f"{self._api2_physical_attempt:016x}"
            )

    def _initialize_event_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise GatewayError("refusing to overwrite gateway event log")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        return path

    def _record_event(self, payload: Mapping[str, Any]) -> None:
        if self.event_path is None:
            return
        with self._event_lock:
            event = {
                "schema_version": "sieve_model_gateway_event_v2",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "api_family_id": self.model.api_family_id,
                "route_profile_id": self.route.route_profile_id,
                "model_profile_id": self.model.model_profile_id,
                **dict(payload),
                "previous_event_sha256": self._previous_event_hash,
            }
            canonical = json.dumps(
                event,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = hashlib.sha256(canonical).hexdigest()
            event["event_sha256"] = digest
            encoded = (
                json.dumps(
                    event,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            with self.event_path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._previous_event_hash = digest

    @staticmethod
    def _reject(handler: BaseHTTPRequestHandler, status: int, code: str) -> None:
        body = json.dumps(
            {"error": {"code": code}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except OSError:
            # The local Pi client may already be gone during a trusted close.
            # A disconnected downstream is itself terminal; never keep a
            # handler alive merely to report an error body.
            return


def build_gateway_public_record(
    *,
    route: RouteProfile,
    model: ModelProfile,
    max_requests: int,
    required_system_prompt_sha256s: Iterable[str],
    system_prompt_rewrites: Mapping[str, str],
) -> dict[str, Any]:
    """Build the exact secret-free gateway contract recorded per episode."""

    prompt_hashes = tuple(required_system_prompt_sha256s)
    rewrites = dict(system_prompt_rewrites)
    return {
        "schema_version": "sieve_scoped_model_gateway_public_v2",
        "route_profile_id": route.route_profile_id,
        "model_profile_id": model.model_profile_id,
        "api_family_id": model.api_family_id,
        "client_path": route.client_path,
        "request_budget": max_requests,
        "retry_policy": model.retry.public_dict(),
        "provider_reasoning": model.reasoning.public_dict(),
        "response_identity_required": model.response_identity_required,
        "system_prompt_bound": bool(prompt_hashes),
        "required_system_prompt_sha256s": sorted(prompt_hashes),
        "provider_visible_system_prompt_sha256s": sorted(
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in rewrites.values()
        ),
        "pi_dynamic_cwd_forwarded": False,
        "upstream_endpoint_recorded": False,
        "upstream_credential_recorded": False,
        "session_headers_recorded": False,
        "request_or_response_bodies_recorded": False,
    }


def verify_gateway_audit(
    event_path: str | Path,
    completion: Mapping[str, Any],
    *,
    expected_api_family_id: str | None = None,
    expected_route_profile_id: str | None = None,
    expected_model_profile_id: str | None = None,
    expected_retry_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a sealed gateway event chain against its completion report."""

    path = Path(event_path).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise GatewayError("gateway audit log must be a real file")
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise GatewayError("gateway audit log is not host-sealed")
    if completion.get("schema_version") != "sieve_model_gateway_completion_v3":
        raise GatewayError("gateway completion schema mismatch")
    if set(completion) != {
        "schema_version",
        "request_count",
        "complete_request_count",
        "incomplete_logical_requests",
        "terminal_statuses",
        "tool_history_by_request",
        "reasoning_replay_by_request",
        "identity_matches_by_request",
        "transport_recycle_required_by_request",
        "transport_recycle_required",
        "event_terminal_sha256",
        "inflight",
        "all_logical_requests_complete",
    }:
        raise GatewayError("gateway completion field set mismatch")
    request_count = completion.get("request_count")
    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count < 0
    ):
        raise GatewayError("gateway completion request count is invalid")
    expected_statuses = completion.get("terminal_statuses")
    if not isinstance(expected_statuses, dict):
        raise GatewayError("gateway completion terminal statuses are invalid")
    expected_recycle = completion.get("transport_recycle_required_by_request")
    if not isinstance(expected_recycle, dict) or not isinstance(
        completion.get("transport_recycle_required"), bool
    ):
        raise GatewayError("gateway completion recycle metadata is invalid")
    maximum_attempts: int | None = None
    retryable_http: frozenset[int] = frozenset()
    retry_transport = False
    if expected_retry_policy is not None:
        retry_count = expected_retry_policy.get("max_infrastructure_retries")
        statuses = expected_retry_policy.get("retryable_http_statuses")
        retry_transport_value = expected_retry_policy.get("retry_transport_failures")
        ambiguous_value = expected_retry_policy.get("retry_ambiguous_timeouts")
        if (
            isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or retry_count < 0
            or not isinstance(statuses, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 400
                or value > 599
                for value in statuses
            )
            or len(statuses) != len(set(statuses))
            or not isinstance(retry_transport_value, bool)
            or ambiguous_value is not False
        ):
            raise GatewayError("expected gateway retry policy is invalid")
        maximum_attempts = 1 + retry_count
        retryable_http = frozenset(statuses)
        retry_transport = retry_transport_value

    previous_hash = "0" * 64
    terminal_statuses: dict[str, str] = {}
    attempts: dict[int, int] = {}
    pending_attempts: set[tuple[int, int]] = set()
    tool_history: dict[str, bool] = {}
    reasoning_replay: dict[str, bool | None] = {}
    identity_matches: dict[str, bool | None] = {}
    transport_recycle: dict[str, bool] = {}
    cooldown_waits: dict[tuple[int, int], float] = {}
    previous_timestamp: datetime | None = None
    event_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise GatewayError("gateway audit contains an empty line")
            try:
                event = json.loads(line, parse_constant=_reject_json_constant)
            except (UnicodeError, ValueError) as exc:
                raise GatewayError(
                    f"gateway audit contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(event, dict):
                raise GatewayError("gateway audit event is not an object")
            if event.get("schema_version") != "sieve_model_gateway_event_v2":
                raise GatewayError("gateway audit event schema mismatch")
            timestamp = event.get("timestamp_utc")
            if not isinstance(timestamp, str):
                raise GatewayError("gateway audit timestamp is invalid")
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp)
            except ValueError as exc:
                raise GatewayError("gateway audit timestamp is invalid") from exc
            if (
                parsed_timestamp.tzinfo is None
                or parsed_timestamp.utcoffset() != timezone.utc.utcoffset(parsed_timestamp)
                or (
                    previous_timestamp is not None
                    and parsed_timestamp < previous_timestamp
                )
            ):
                raise GatewayError("gateway audit timestamp ordering is invalid")
            previous_timestamp = parsed_timestamp
            expected_identity = {
                "api_family_id": expected_api_family_id,
                "route_profile_id": expected_route_profile_id,
                "model_profile_id": expected_model_profile_id,
            }
            for identity_field, identity_value in expected_identity.items():
                if identity_value is not None and event.get(identity_field) != identity_value:
                    raise GatewayError(
                        f"gateway audit {identity_field.replace('_', '-')} mismatch"
                    )
            observed_hash = event.get("event_sha256")
            if not isinstance(observed_hash, str) or SHA256.fullmatch(observed_hash) is None:
                raise GatewayError("gateway audit event hash is invalid")
            unhashed = dict(event)
            del unhashed["event_sha256"]
            if unhashed.get("previous_event_sha256") != previous_hash:
                raise GatewayError("gateway audit hash chain is discontinuous")
            canonical = json.dumps(
                unhashed,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if not secrets.compare_digest(
                hashlib.sha256(canonical).hexdigest(), observed_hash
            ):
                raise GatewayError("gateway audit event hash mismatch")
            previous_hash = observed_hash
            event_count += 1

            kind = event.get("event")
            logical_request = event.get("logical_request")
            attempt = event.get("attempt")
            if kind not in {
                "upstream_attempt_started",
                "upstream_attempt_result",
                "logical_request_cancelled_before_attempt",
            }:
                raise GatewayError("gateway audit contains an unknown event")
            if (
                isinstance(logical_request, bool)
                or not isinstance(logical_request, int)
                or logical_request < 1
                or logical_request > request_count
            ):
                raise GatewayError("gateway audit request is invalid")
            if kind == "logical_request_cancelled_before_attempt":
                if set(event) != {
                    "schema_version",
                    "timestamp_utc",
                    "api_family_id",
                    "route_profile_id",
                    "model_profile_id",
                    "event",
                    "has_tool_history",
                    "logical_request",
                    "reasoning_replay_present",
                    "status",
                    "transport_recycle_required",
                    "previous_event_sha256",
                    "event_sha256",
                }:
                    raise GatewayError(
                        "gateway audit cancellation field set mismatch"
                    )
                key = str(logical_request)
                history_value = event.get("has_tool_history")
                replay_value = event.get("reasoning_replay_present")
                if (
                    key in terminal_statuses
                    or any(
                        request == logical_request
                        for request, _ in pending_attempts
                    )
                    or event.get("status") != "model_gateway_closing"
                    or event.get("transport_recycle_required") is not False
                    or not isinstance(history_value, bool)
                    or (
                        replay_value is not None
                        and not isinstance(replay_value, bool)
                    )
                ):
                    raise GatewayError(
                        "gateway audit cancellation metadata is invalid"
                    )
                if key in tool_history and (
                    tool_history[key] != history_value
                    or reasoning_replay[key] != replay_value
                ):
                    raise GatewayError(
                        "gateway audit replay metadata changed before cancellation"
                    )
                tool_history[key] = history_value
                reasoning_replay[key] = replay_value
                terminal_statuses[key] = "model_gateway_closing"
                identity_matches[key] = None
                transport_recycle[key] = False
                continue
            if (
                isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or attempt < 1
                or (maximum_attempts is not None and attempt > maximum_attempts)
            ):
                raise GatewayError("gateway audit request or attempt is invalid")
            if kind == "upstream_attempt_started":
                if set(event) != {
                    "schema_version",
                    "timestamp_utc",
                    "api_family_id",
                    "route_profile_id",
                    "model_profile_id",
                    "attempt",
                    "cooldown_wait_seconds",
                    "event",
                    "has_tool_history",
                    "logical_request",
                    "reasoning_replay_present",
                    "previous_event_sha256",
                    "event_sha256",
                }:
                    raise GatewayError("gateway audit start field set mismatch")
                if str(logical_request) in terminal_statuses:
                    raise GatewayError("gateway audit continued after a terminal result")
                if attempt != attempts.get(logical_request, 0) + 1:
                    raise GatewayError("gateway audit attempt ordering is invalid")
                attempts[logical_request] = attempt
                pending_attempts.add((logical_request, attempt))
                history_value = event.get("has_tool_history")
                replay_value = event.get("reasoning_replay_present")
                cooldown_wait = event.get("cooldown_wait_seconds")
                if not isinstance(history_value, bool) or (
                    replay_value is not None and not isinstance(replay_value, bool)
                ) or (
                    isinstance(cooldown_wait, bool)
                    or not isinstance(cooldown_wait, (int, float))
                    or not math.isfinite(float(cooldown_wait))
                    or float(cooldown_wait) < 0
                ):
                    raise GatewayError("gateway audit replay metadata is invalid")
                cooldown_waits[(logical_request, attempt)] = float(cooldown_wait)
                key = str(logical_request)
                if key in tool_history and (
                    tool_history[key] != history_value
                    or reasoning_replay[key] != replay_value
                ):
                    raise GatewayError("gateway audit replay metadata changed on retry")
                tool_history[key] = history_value
                reasoning_replay[key] = replay_value
                continue
            if (logical_request, attempt) not in pending_attempts:
                raise GatewayError("gateway audit result lacks matching start")
            pending_attempts.remove((logical_request, attempt))
            if set(event) != {
                "schema_version",
                "timestamp_utc",
                "api_family_id",
                "route_profile_id",
                "model_profile_id",
                "attempt",
                "cooldown_wait_seconds",
                "downstream_started",
                "elapsed_seconds",
                "error_type",
                "event",
                "http_status",
                "identity_matches",
                "logical_request",
                "retry_scheduled",
                "status",
                "transport_recycle_required",
                "previous_event_sha256",
                "event_sha256",
            }:
                raise GatewayError("gateway audit result field set mismatch")
            status = event.get("status")
            retry_scheduled = event.get("retry_scheduled")
            if not isinstance(status, str) or not status:
                raise GatewayError("gateway audit result status is invalid")
            if not isinstance(retry_scheduled, bool):
                raise GatewayError("gateway audit retry decision is invalid")
            http_status = event.get("http_status")
            downstream_started = event.get("downstream_started")
            elapsed_seconds = event.get("elapsed_seconds")
            cooldown_wait = event.get("cooldown_wait_seconds")
            error_type = event.get("error_type")
            recycle_required = event.get("transport_recycle_required")
            if (
                (
                    http_status is not None
                    and (
                        isinstance(http_status, bool)
                        or not isinstance(http_status, int)
                        or http_status < 100
                        or http_status > 599
                    )
                )
                or not isinstance(downstream_started, bool)
                or isinstance(elapsed_seconds, bool)
                or not isinstance(elapsed_seconds, (int, float))
                or not math.isfinite(float(elapsed_seconds))
                or float(elapsed_seconds) < 0
                or isinstance(cooldown_wait, bool)
                or not isinstance(cooldown_wait, (int, float))
                or not math.isfinite(float(cooldown_wait))
                or float(cooldown_wait) < 0
                or float(cooldown_wait)
                != cooldown_waits.pop((logical_request, attempt))
                or (
                    error_type is not None
                    and (
                        not isinstance(error_type, str)
                        or PORTABLE_ERROR_TYPE.fullmatch(error_type) is None
                    )
                )
                or not isinstance(recycle_required, bool)
            ):
                raise GatewayError("gateway audit result metadata is invalid")
            expected_recycle_required = bool(
                status.startswith("ambiguous_")
                or (downstream_started and status != "complete")
                or status
                in {
                    "upstream_stream_content_type_mismatch",
                    "upstream_model_identity_mismatch",
                    "upstream_model_identity_missing",
                }
                or status.startswith("anthropic_reasoning_bridge_")
                or status.startswith("anthropic_thinking_")
                or status.startswith("anthropic_redacted_thinking_")
            )
            if recycle_required is not expected_recycle_required:
                raise GatewayError(
                    "gateway audit transport-recycle decision mismatch"
                )
            if expected_retry_policy is not None:
                retryable = (
                    status in {
                        "upstream_connect_setup_failure",
                        "upstream_connect_failure",
                    }
                    and retry_transport
                ) or (
                    status == "upstream_http_failure"
                    and http_status in retryable_http
                )
                expected_retry = bool(
                    retryable
                    and maximum_attempts is not None
                    and attempt < maximum_attempts
                )
                if retry_scheduled is not expected_retry:
                    raise GatewayError("gateway audit retry policy mismatch")
            key = str(logical_request)
            if retry_scheduled:
                if status == "complete" or recycle_required:
                    raise GatewayError("gateway audit retries a completed attempt")
            else:
                terminal_statuses[key] = status
                identity_value = event.get("identity_matches")
                if identity_value is not None and not isinstance(identity_value, bool):
                    raise GatewayError("gateway audit identity result is invalid")
                identity_matches[key] = identity_value
                transport_recycle[key] = recycle_required

    if pending_attempts:
        raise GatewayError("gateway audit contains an unterminated attempt")
    expected_request_keys = {
        str(request_id) for request_id in range(1, request_count + 1)
    }
    if set(terminal_statuses) != expected_request_keys:
        raise GatewayError("gateway audit lacks a terminal record for a request")
    expected_hash = completion.get("event_terminal_sha256")
    if not isinstance(expected_hash, str) or SHA256.fullmatch(expected_hash) is None:
        raise GatewayError("gateway completion terminal hash is invalid")
    if not secrets.compare_digest(previous_hash, expected_hash):
        raise GatewayError("gateway audit terminal hash mismatch")
    if terminal_statuses != expected_statuses:
        raise GatewayError("gateway audit terminal statuses mismatch")
    if completion.get("tool_history_by_request") != tool_history:
        raise GatewayError("gateway audit tool-history map mismatch")
    if completion.get("reasoning_replay_by_request") != reasoning_replay:
        raise GatewayError("gateway audit reasoning-replay map mismatch")
    if completion.get("identity_matches_by_request") != identity_matches:
        raise GatewayError("gateway audit identity-result map mismatch")
    if expected_recycle != transport_recycle:
        raise GatewayError("gateway audit transport-recycle map mismatch")
    if completion.get("transport_recycle_required") is not any(
        transport_recycle.values()
    ):
        raise GatewayError("gateway audit transport-recycle aggregate mismatch")
    complete_count = sum(1 for value in terminal_statuses.values() if value == "complete")
    if completion.get("complete_request_count") != complete_count:
        raise GatewayError("gateway completion complete count mismatch")
    expected_incomplete = [
        request_id
        for request_id in range(1, request_count + 1)
        if terminal_statuses.get(str(request_id)) != "complete"
    ]
    if completion.get("incomplete_logical_requests") != expected_incomplete:
        raise GatewayError("gateway completion incomplete requests mismatch")
    all_complete = (
        request_count > 0
        and complete_count == request_count
        and len(terminal_statuses) == request_count
        and completion.get("inflight") is False
    )
    if completion.get("all_logical_requests_complete") is not all_complete:
        raise GatewayError("gateway completion aggregate mismatch")
    return {
        "schema_version": "sieve_model_gateway_audit_verification_v1",
        "event_count": event_count,
        "request_count": request_count,
        "event_terminal_sha256": previous_hash,
        "all_logical_requests_complete": all_complete,
        "transport_recycle_required": any(transport_recycle.values()),
    }


def _extract_response_identity(payload: bytes) -> str | None:
    """Extract a model identity from complete JSON or complete SSE frames."""

    if not payload:
        return None
    stripped = payload.lstrip()
    if stripped.startswith(b"{"):
        try:
            value = json.loads(stripped.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            value = None
        identity = _identity_from_json(value)
        if identity is not None:
            return identity
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            continue
        identity = _identity_from_json(value)
        if identity is not None:
            return identity
    return None


def _read_response_chunk(
    response: http.client.HTTPResponse, maximum_bytes: int
) -> bytes:
    """Read whatever is currently buffered instead of waiting to fill a block.

    HTTPResponse.read(size) may wait for `size` bytes on a long-lived SSE
    response.  `read1` preserves streaming latency while retaining a bounded
    memory read.  The fallback exists only for response-like test doubles.
    """

    read1 = getattr(response, "read1", None)
    if callable(read1):
        return read1(maximum_bytes)
    return response.read(maximum_bytes)


def _requested_maximum_output_tokens(
    payload: Mapping[str, Any], *, api_protocol: str, ceiling: int
) -> int:
    fields = (
        ("max_tokens", "max_completion_tokens", "max_output_tokens")
        if api_protocol == "openai-completions"
        else ("max_output_tokens", "max_tokens", "max_completion_tokens")
    )
    supplied = [(name, payload[name]) for name in fields if name in payload]
    if len(supplied) > 1:
        raise GatewayError("multiple_maximum_output_token_fields")
    if not supplied:
        return ceiling
    value = supplied[0][1]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GatewayError("invalid_maximum_output_tokens")
    if value > ceiling:
        raise GatewayError("maximum_output_tokens_exceeds_profile")
    return value


class _AnthropicThinkingSseBridge:
    """Preserve LiteLLM Anthropic thinking signatures through Pi 0.85.

    LiteLLM exposes signed Anthropic thinking as ``thinking_blocks`` while
    Pi's OpenAI-completions provider only persists ``reasoning_details``.
    Translate only that private replay metadata, in memory, and leave all
    visible reasoning text and ordinary stream fields untouched.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()
        self._choice_index: int | None = None
        self._thinking_parts: list[str] = []
        self._signature_parts: list[str] = []
        self._completed_details: list[dict[str, Any]] = []
        self._normal_block_started = False
        self._normal_block_finalized = False
        self._details_emitted = False
        self._last_event_template: dict[str, Any] | None = None
        self._private_bytes = 0

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        self.buffer.extend(chunk)
        if len(self.buffer) > MAX_REQUEST_BYTES:
            raise GatewayError("anthropic_reasoning_bridge_line_too_large")
        output = bytearray()
        while True:
            position = self.buffer.find(b"\n")
            if position < 0:
                break
            raw_line = bytes(self.buffer[: position + 1])
            del self.buffer[: position + 1]
            output.extend(self._line(raw_line))
        return bytes(output)

    def finish(self) -> bytes:
        output = bytearray()
        if self.buffer:
            raw_line = bytes(self.buffer)
            self.buffer.clear()
            output.extend(self._line(raw_line))
        output.extend(self._synthetic_detail_event_if_needed())
        return bytes(output)

    def _line(self, raw_line: bytes) -> bytes:
        if raw_line.endswith(b"\r\n"):
            content, ending = raw_line[:-2], b"\r\n"
        elif raw_line.endswith(b"\n"):
            content, ending = raw_line[:-1], b"\n"
        else:
            content, ending = raw_line, b""
        stripped = content.strip()
        if not stripped.startswith(b"data:"):
            return raw_line
        data = stripped[5:].strip()
        if not data:
            return raw_line
        if data == b"[DONE]":
            return self._synthetic_detail_event_if_needed(ending=ending) + raw_line
        try:
            value = json.loads(
                data.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, ValueError) as exc:
            raise GatewayError("anthropic_reasoning_bridge_invalid_json") from exc
        if not isinstance(value, dict):
            raise GatewayError("anthropic_reasoning_bridge_invalid_json_root")
        self._last_event_template = value
        changed = self._process_event(value)
        if not changed:
            return raw_line
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return b"data: " + encoded + ending

    def _process_event(self, event: dict[str, Any]) -> bool:
        """Aggregate one signed normal block before exposing replay metadata.

        Pi 0.85 merges adjacent ``reasoning.text`` details and retains only the
        first signature.  Emitting LiteLLM's partial thinking/signature deltas
        directly would therefore corrupt a fragmented signature.  Aggregate
        the fragments here and emit one complete detail at the first following
        non-thinking delta.  A second normal thinking block is rejected rather
        than silently collapsed; redacted blocks remain discrete.
        """

        choices = event.get("choices")
        if choices is None:
            return False
        if not isinstance(choices, list):
            raise GatewayError("anthropic_reasoning_bridge_choices_invalid")
        changed = False
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise GatewayError("anthropic_reasoning_bridge_choice_invalid")
            delta = choice.get("delta")
            if delta is None:
                continue
            if not isinstance(delta, dict):
                raise GatewayError("anthropic_reasoning_bridge_delta_invalid")
            candidate = _thinking_blocks_candidate(delta)
            choice_index = choice.get("index", position)
            if isinstance(choice_index, bool) or not isinstance(choice_index, int):
                raise GatewayError("anthropic_reasoning_bridge_choice_index_invalid")
            if candidate is not None:
                self._bind_choice(choice_index)
                if delta.get("reasoning_details") not in (None, []):
                    raise GatewayError("anthropic_reasoning_bridge_field_collision")
                self._consume_blocks(candidate)
                continue
            if (
                self._has_pending_details()
                and not self._details_emitted
                and choice_index == self._choice_index
            ):
                self._inject_complete_details(delta)
                changed = True
        return changed

    def _bind_choice(self, choice_index: int) -> None:
        if self._details_emitted:
            raise GatewayError("anthropic_reasoning_bridge_late_thinking_block")
        if self._choice_index is None:
            self._choice_index = choice_index
        elif self._choice_index != choice_index:
            raise GatewayError("anthropic_reasoning_bridge_multiple_choices")

    def _consume_blocks(self, value: Any) -> None:
        if not isinstance(value, list):
            raise GatewayError("anthropic_thinking_blocks_invalid")
        for block in value:
            if not isinstance(block, dict):
                raise GatewayError("anthropic_thinking_block_invalid")
            block_type = block.get("type")
            if block_type == "thinking":
                thinking = block.get("thinking")
                signature = block.get("signature")
                if not isinstance(thinking, str) or not isinstance(signature, str):
                    raise GatewayError("anthropic_thinking_block_invalid")
                if not thinking and not signature:
                    continue
                if self._normal_block_finalized or (thinking and self._signature_parts):
                    raise GatewayError(
                        "anthropic_reasoning_bridge_multiple_normal_blocks_unsupported"
                    )
                self._reserve_private_bytes(thinking, signature)
                self._normal_block_started = True
                if thinking:
                    self._thinking_parts.append(thinking)
                if signature:
                    self._signature_parts.append(signature)
            elif block_type == "redacted_thinking":
                data = block.get("data")
                if not isinstance(data, str) or not data:
                    raise GatewayError("anthropic_redacted_thinking_block_invalid")
                if self._normal_block_started and not self._normal_block_finalized:
                    self._finalize_normal_block()
                self._reserve_private_bytes(data)
                self._reserve_detail_block()
                self._completed_details.append(
                    {
                        "type": "reasoning.encrypted",
                        "data": data,
                        "format": ANTHROPIC_REASONING_DETAIL_FORMAT,
                    }
                )
            else:
                raise GatewayError("anthropic_thinking_block_type_invalid")

    def _has_pending_details(self) -> bool:
        return self._normal_block_started or bool(self._completed_details)

    def _finalize_normal_block(self) -> None:
        if not self._normal_block_started or self._normal_block_finalized:
            return
        thinking = "".join(self._thinking_parts)
        signature = "".join(self._signature_parts)
        if not thinking or not signature:
            raise GatewayError("anthropic_thinking_block_unsigned")
        self._reserve_detail_block()
        self._completed_details.append(
            {
                "type": "reasoning.text",
                "text": thinking,
                "signature": signature,
                "format": ANTHROPIC_REASONING_DETAIL_FORMAT,
            }
        )
        self._normal_block_finalized = True

    def _reserve_private_bytes(self, *values: str) -> None:
        self._private_bytes += sum(len(value.encode("utf-8")) for value in values)
        if self._private_bytes > MAX_REASONING_BRIDGE_BYTES:
            raise GatewayError("anthropic_reasoning_bridge_private_data_too_large")

    def _reserve_detail_block(self) -> None:
        if len(self._completed_details) >= MAX_REASONING_DETAIL_BLOCKS:
            raise GatewayError("anthropic_reasoning_bridge_too_many_blocks")

    def _complete_details(self) -> list[dict[str, Any]]:
        self._finalize_normal_block()
        details = list(self._completed_details)
        if not details:
            raise GatewayError("anthropic_reasoning_bridge_empty_details")
        return details

    def _inject_complete_details(self, delta: dict[str, Any]) -> None:
        if self._details_emitted:
            raise GatewayError("anthropic_reasoning_bridge_duplicate_emission")
        if delta.get("reasoning_details") not in (None, []):
            raise GatewayError("anthropic_reasoning_bridge_field_collision")
        delta["reasoning_details"] = self._complete_details()
        self._details_emitted = True

    def _synthetic_detail_event_if_needed(self, *, ending: bytes = b"\n") -> bytes:
        if not self._has_pending_details() or self._details_emitted:
            return b""
        if self._choice_index is None:
            raise GatewayError("anthropic_reasoning_bridge_choice_missing")
        event: dict[str, Any] = {}
        if self._last_event_template is not None:
            for field in ("id", "object", "created", "model", "system_fingerprint"):
                if field in self._last_event_template:
                    event[field] = self._last_event_template[field]
        delta: dict[str, Any] = {}
        self._inject_complete_details(delta)
        event["choices"] = [
            {"index": self._choice_index, "delta": delta, "finish_reason": None}
        ]
        encoded = json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        line_ending = ending or b"\n"
        return b"data: " + encoded + line_ending + line_ending


def _thinking_blocks_candidate(delta: Mapping[str, Any]) -> Any | None:
    candidates: list[Any] = []
    if delta.get("thinking_blocks") is not None:
        candidates.append(delta["thinking_blocks"])
    provider_fields = delta.get("provider_specific_fields")
    if provider_fields is not None:
        if not isinstance(provider_fields, dict):
            raise GatewayError("anthropic_reasoning_bridge_provider_fields_invalid")
        if provider_fields.get("thinking_blocks") is not None:
            candidates.append(provider_fields["thinking_blocks"])
    if not candidates:
        return None
    canonical = _canonical_json_bytes(candidates[0])
    if any(
        _canonical_json_bytes(candidate) != canonical
        for candidate in candidates[1:]
    ):
        raise GatewayError("anthropic_reasoning_bridge_duplicate_mismatch")
    return candidates[0]


def _restore_anthropic_signed_reasoning(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore Pi's opaque replay metadata to LiteLLM ``thinking_blocks``."""

    output = dict(payload)
    messages = output.get("messages")
    if not isinstance(messages, list):
        # The route-level request validator will emit the stable shape error.
        return output
    rewritten: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            rewritten.append(message)
            continue
        updated = dict(message)
        details = updated.get("reasoning_details")
        existing_blocks = updated.get("thinking_blocks")
        if updated.get("role") != "assistant":
            if details is not None or existing_blocks is not None:
                raise GatewayError("anthropic_reasoning_replay_role_invalid")
            rewritten.append(updated)
            continue
        if details is None:
            if existing_blocks is not None:
                raise GatewayError("anthropic_reasoning_replay_untrusted_blocks")
            rewritten.append(updated)
            continue
        if existing_blocks is not None:
            raise GatewayError("anthropic_reasoning_replay_field_collision")
        blocks = _reasoning_details_to_anthropic_blocks(details)
        raw_reasoning = updated.pop("reasoning_content", None)
        expected_text = "".join(
            str(block["thinking"])
            for block in blocks
            if block.get("type") == "thinking"
        )
        if raw_reasoning not in (None, "", expected_text):
            raise GatewayError("anthropic_reasoning_replay_text_mismatch")
        updated.pop("reasoning_details", None)
        updated["thinking_blocks"] = blocks
        rewritten.append(updated)
    output["messages"] = rewritten
    return output


def _reasoning_details_to_anthropic_blocks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise GatewayError("anthropic_reasoning_details_invalid")
    blocks: list[dict[str, Any]] = []
    for detail in value:
        if not isinstance(detail, dict):
            raise GatewayError("anthropic_reasoning_detail_invalid")
        detail_type = detail.get("type")
        if detail_type == "reasoning.text":
            if set(detail) != {"type", "text", "signature", "format"}:
                raise GatewayError("anthropic_reasoning_detail_fields_invalid")
            text = detail.get("text")
            signature = detail.get("signature")
            if (
                detail.get("format") != ANTHROPIC_REASONING_DETAIL_FORMAT
                or not isinstance(text, str)
                or not text
                or not isinstance(signature, str)
                or not signature
            ):
                raise GatewayError("anthropic_signed_reasoning_detail_invalid")
            blocks.append(
                {"type": "thinking", "thinking": text, "signature": signature}
            )
        elif detail_type == "reasoning.encrypted":
            if set(detail) != {"type", "data", "format"}:
                raise GatewayError("anthropic_reasoning_detail_fields_invalid")
            data = detail.get("data")
            if (
                detail.get("format") != ANTHROPIC_REASONING_DETAIL_FORMAT
                or not isinstance(data, str)
                or not data
            ):
                raise GatewayError("anthropic_redacted_reasoning_detail_invalid")
            blocks.append({"type": "redacted_thinking", "data": data})
        else:
            raise GatewayError("anthropic_reasoning_detail_type_invalid")
    return blocks


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GatewayError("anthropic_reasoning_bridge_value_invalid") from exc


class _SseContractValidator:
    """Incrementally validate the minimal terminal shape of a frozen route."""

    def __init__(self, contract: str) -> None:
        if contract not in {
            "openai_chat_stream_tool_identity_v1",
            "openai_chat_stream_tool_v1",
            "openai_responses_stream_tool_v1",
        }:
            raise GatewayError("unsupported_response_contract")
        self.contract = contract
        self.buffer = bytearray()
        self.valid_json_events = 0
        self.chat_choice_events = 0
        self.chat_done = False
        self.responses_completed = False

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.buffer.extend(chunk)
        if len(self.buffer) > MAX_REQUEST_BYTES:
            raise GatewayError("ambiguous_stream_contract_line_too_large")
        while True:
            position = self.buffer.find(b"\n")
            if position < 0:
                return
            raw_line = bytes(self.buffer[:position])
            del self.buffer[: position + 1]
            self._line(raw_line.rstrip(b"\r"))

    def finish(self) -> None:
        if self.buffer:
            self._line(bytes(self.buffer).rstrip(b"\r"))
            self.buffer.clear()
        if self.valid_json_events < 1:
            raise GatewayError("ambiguous_stream_contract_empty")
        if self.contract.startswith("openai_chat_"):
            if not self.chat_done:
                raise GatewayError("ambiguous_stream_contract_missing_done")
            if self.chat_choice_events < 1:
                raise GatewayError("ambiguous_stream_contract_missing_choices")
        if (
            self.contract == "openai_responses_stream_tool_v1"
            and not self.responses_completed
        ):
            raise GatewayError("ambiguous_stream_contract_missing_completed")

    def _line(self, line: bytes) -> None:
        stripped = line.strip()
        if not stripped or stripped.startswith(b":"):
            return
        if not stripped.startswith(b"data:"):
            if stripped.startswith((b"event:", b"id:", b"retry:")):
                return
            raise GatewayError("ambiguous_stream_contract_invalid_field")
        data = stripped[5:].strip()
        if data == b"[DONE]":
            if self.contract.startswith("openai_chat_"):
                if self.chat_done:
                    raise GatewayError("ambiguous_stream_contract_duplicate_done")
                self.chat_done = True
            return
        if not data:
            return
        try:
            value = json.loads(
                data.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, ValueError) as exc:
            raise GatewayError("ambiguous_stream_contract_invalid_json") from exc
        if not isinstance(value, dict):
            raise GatewayError("ambiguous_stream_contract_invalid_json_root")
        if self.chat_done or self.responses_completed:
            raise GatewayError("ambiguous_stream_contract_data_after_terminal")
        if value.get("error") not in (None, {}, []):
            raise GatewayError("ambiguous_stream_contract_error_event")
        self.valid_json_events += 1
        if self.contract.startswith("openai_chat_"):
            choices = value.get("choices")
            if isinstance(choices, list) and choices:
                self.chat_choice_events += 1
            elif choices is not None and not isinstance(choices, list):
                raise GatewayError("ambiguous_stream_contract_invalid_choices")
            return
        event_type = value.get("type")
        if event_type in {"error", "response.failed", "response.incomplete"}:
            raise GatewayError("ambiguous_stream_contract_failed_response")
        if event_type == "response.completed":
            response = value.get("response")
            if not isinstance(response, dict) or response.get("status") != "completed":
                raise GatewayError("ambiguous_stream_contract_invalid_completed")
            self.responses_completed = True


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _system_prompt_sha256(
    payload: Mapping[str, Any], *, api_protocol: str
) -> str | None:
    """Return the hash of the sole Pi system/developer message, if valid."""

    if api_protocol == "openai-completions":
        if payload.get("instructions") is not None:
            return None
        messages = payload.get("messages")
    elif api_protocol == "openai-responses":
        instructions = payload.get("instructions")
        messages = payload.get("input")
        if messages is not None and not isinstance(messages, list):
            return None
        embedded_prompts = _system_message_texts(messages or [])
        if embedded_prompts is None:
            return None
        if isinstance(instructions, str):
            if embedded_prompts:
                return None
            return hashlib.sha256(instructions.encode("utf-8")).hexdigest()
        if instructions is not None:
            return None
    else:
        return None
    if not isinstance(messages, list):
        return None
    prompts = _system_message_texts(messages)
    if prompts is None:
        return None
    if len(prompts) != 1:
        return None
    return hashlib.sha256(prompts[0].encode("utf-8")).hexdigest()


def _replace_system_prompt(
    payload: Mapping[str, Any], *, api_protocol: str, replacement: str
) -> dict[str, Any]:
    """Replace the one already-validated system prompt without other drift."""

    output = dict(payload)
    if api_protocol == "openai-responses" and isinstance(
        output.get("instructions"), str
    ):
        messages = output.get("input")
        if messages is not None and not isinstance(messages, list):
            raise GatewayError("system_prompt_rewrite_shape_mismatch")
        embedded_prompts = _system_message_texts(messages or [])
        if embedded_prompts is None or embedded_prompts:
            raise GatewayError("system_prompt_rewrite_shape_mismatch")
        output["instructions"] = replacement
        return output
    field = "messages" if api_protocol == "openai-completions" else "input"
    messages = output.get(field)
    if not isinstance(messages, list):
        raise GatewayError("system_prompt_rewrite_shape_mismatch")
    rewritten: list[Any] = []
    count = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") in {
            "system",
            "developer",
        }:
            updated = dict(message)
            updated["content"] = replacement
            rewritten.append(updated)
            count += 1
        else:
            rewritten.append(message)
    if count != 1:
        raise GatewayError("system_prompt_rewrite_shape_mismatch")
    output[field] = rewritten
    return output


def _system_message_texts(messages: list[Any]) -> list[str] | None:
    prompts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "developer",
        }:
            continue
        text = _message_text(message.get("content"))
        if text is None:
            return None
        prompts.append(text)
    return prompts


def _message_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    fragments: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") not in {
            "text",
            "input_text",
        }:
            return None
        text = item.get("text")
        if not isinstance(text, str):
            return None
        fragments.append(text)
    return "".join(fragments)


def _reasoning_replay_state(
    payload: Mapping[str, Any],
    *,
    api_protocol: str,
    require_signed_anthropic: bool = False,
) -> tuple[bool, bool]:
    """Validate tool-result provenance without retaining reasoning bytes.

    A reasoning block is replay evidence only for the assistant output that
    produced the exact tool call(s) whose result(s) are present.  An unrelated
    reasoning block from an older turn must never satisfy this check.
    """

    if api_protocol == "openai-completions":
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise GatewayError("chat_messages_shape_invalid")
        has_history = False
        replay_for_every_group = True
        index = 0
        while index < len(messages):
            item = messages[index]
            if not isinstance(item, dict) or item.get("role") != "tool":
                index += 1
                continue
            has_history = True
            start = index
            tool_ids: list[str] = []
            while index < len(messages):
                tool_result = messages[index]
                if not isinstance(tool_result, dict) or tool_result.get("role") != "tool":
                    break
                call_id = tool_result.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id or call_id in tool_ids:
                    raise GatewayError("chat_tool_result_identity_invalid")
                tool_ids.append(call_id)
                index += 1
            if start == 0:
                raise GatewayError("chat_tool_result_lacks_assistant_call")
            assistant = messages[start - 1]
            if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
                raise GatewayError("chat_tool_result_lacks_assistant_call")
            calls = assistant.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise GatewayError("chat_tool_result_lacks_assistant_call")
            call_ids: list[str] = []
            for call in calls:
                if not isinstance(call, dict):
                    raise GatewayError("chat_tool_call_identity_invalid")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                    raise GatewayError("chat_tool_call_identity_invalid")
                call_ids.append(call_id)
            if call_ids != tool_ids:
                raise GatewayError("chat_tool_result_call_order_mismatch")
            if require_signed_anthropic:
                replay_present = _has_valid_anthropic_thinking_blocks(
                    assistant.get("thinking_blocks")
                )
            else:
                replay_present = any(
                    assistant.get(key) not in (None, "", [], {})
                    for key in (
                        "reasoning_content",
                        "reasoning",
                        "reasoning_details",
                        "thinking",
                        "encrypted_content",
                    )
                )
            replay_for_every_group = replay_for_every_group and replay_present
        return has_history, has_history and replay_for_every_group
    if api_protocol == "openai-responses":
        items = payload.get("input")
        if not isinstance(items, list):
            raise GatewayError("responses_input_shape_invalid")
        output_types = {"function_call_output", "custom_tool_call_output"}
        has_history = False
        replay_for_every_group = True
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, dict) or item.get("type") not in output_types:
                index += 1
                continue
            has_history = True
            output_start = index
            output_ids: list[str] = []
            while index < len(items):
                result = items[index]
                if not isinstance(result, dict) or result.get("type") not in output_types:
                    break
                call_id = result.get("call_id")
                if not isinstance(call_id, str) or not call_id or call_id in output_ids:
                    raise GatewayError("responses_tool_result_identity_invalid")
                output_ids.append(call_id)
                index += 1

            call_start = output_start
            while call_start > 0:
                candidate = items[call_start - 1]
                if not isinstance(candidate, dict) or candidate.get("type") != "function_call":
                    break
                call_start -= 1
            if call_start == output_start:
                raise GatewayError("responses_tool_result_lacks_function_call")
            call_ids: list[str] = []
            for call in items[call_start:output_start]:
                if not isinstance(call, dict):
                    raise GatewayError("responses_tool_call_identity_invalid")
                call_id = call.get("call_id")
                if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                    raise GatewayError("responses_tool_call_identity_invalid")
                call_ids.append(call_id)
            if call_ids != output_ids:
                raise GatewayError("responses_tool_result_call_order_mismatch")

            # Reasoning belongs to this assistant output segment only.  Stop
            # at the previous user/system/developer item or tool result so an
            # older reasoning item cannot mask a missing current replay.
            segment_start = call_start
            while segment_start > 0:
                candidate = items[segment_start - 1]
                if not isinstance(candidate, dict):
                    break
                if candidate.get("type") in output_types or candidate.get("role") in {
                    "user",
                    "system",
                    "developer",
                    "tool",
                }:
                    break
                segment_start -= 1
            current_replay = any(
                isinstance(candidate, dict)
                and candidate.get("type") == "reasoning"
                and any(
                    candidate.get(key) not in (None, "", [], {})
                    for key in ("encrypted_content", "content", "summary")
                )
                for candidate in items[segment_start:call_start]
            )
            replay_for_every_group = replay_for_every_group and current_replay
        return has_history, has_history and replay_for_every_group
    raise GatewayError("unsupported_route_protocol")


def _has_valid_anthropic_thinking_blocks(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for block in value:
        if not isinstance(block, dict):
            return False
        if block.get("type") == "thinking":
            if (
                set(block) != {"type", "thinking", "signature"}
                or not isinstance(block.get("thinking"), str)
                or not block["thinking"]
                or not isinstance(block.get("signature"), str)
                or not block["signature"]
            ):
                return False
        elif block.get("type") == "redacted_thinking":
            if (
                set(block) != {"type", "data"}
                or not isinstance(block.get("data"), str)
                or not block["data"]
            ):
                return False
        else:
            return False
    return True


def _identity_from_json(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    direct = value.get("model")
    if isinstance(direct, str) and direct:
        return direct
    response = value.get("response")
    if isinstance(response, dict):
        nested = response.get("model")
        if isinstance(nested, str) and nested:
            return nested
    return None


__all__ = [
    "ForwardResult",
    "GatewayError",
    "GatewayShutdownError",
    "ScopedModelGateway",
    "SharedCooldownGate",
    "build_gateway_public_record",
    "completion_requires_transport_recycle",
    "verify_gateway_audit",
]
