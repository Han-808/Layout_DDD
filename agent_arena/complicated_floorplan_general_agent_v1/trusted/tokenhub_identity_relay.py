#!/usr/bin/env python3
"""Host-owned TokenHub relay that verifies raw Anthropic model identity.

LiteLLM rewrites provider streams into OpenAI-compatible chunks and can stamp
its configured alias into those chunks.  Consequently, validating only the
post-LiteLLM ``model`` field does not prove which provider model answered.  The
relay sits before LiteLLM, keeps the provider credential in trusted host
memory, and releases a successful stream only after the raw Anthropic
``message_start.message.model`` matches the frozen contract.

No request or response body, endpoint, credential, request identifier,
reasoning, or model content is persisted by this module.
"""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import secrets
import socket
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit


LOOPBACK_HOST = "127.0.0.1"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_IDENTITY_PREFACE_BYTES = 1024 * 1024
CONNECT_TIMEOUT_SECONDS = 5.0
DRAIN_TIMEOUT_SECONDS = 15.0


class TokenHubIdentityRelayError(RuntimeError):
    """The raw-provider identity boundary could not be proven safe."""


class TokenHubIdentityRelay:
    """Forward one Anthropic route while enforcing its raw response model."""

    def __init__(
        self,
        *,
        provider_base_url: str,
        provider_credential: str,
        expected_request_model: str,
        accepted_response_models: tuple[str, ...],
        request_timeout_seconds: float,
        allow_insecure_provider: bool = False,
    ) -> None:
        parsed = urlsplit(provider_base_url.rstrip("/"))
        if (
            parsed.scheme not in ({"https", "http"} if allow_insecure_provider else {"https"})
            or not parsed.hostname
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise TokenHubIdentityRelayError("provider base contract is invalid")
        if (
            not isinstance(provider_credential, str)
            or not provider_credential
            or "\r" in provider_credential
            or "\n" in provider_credential
        ):
            raise TokenHubIdentityRelayError("provider credential is malformed")
        if not isinstance(expected_request_model, str) or not expected_request_model:
            raise TokenHubIdentityRelayError("expected request model is malformed")
        if (
            not accepted_response_models
            or len(set(accepted_response_models)) != len(accepted_response_models)
            or any(not isinstance(value, str) or not value for value in accepted_response_models)
        ):
            raise TokenHubIdentityRelayError("accepted response models are malformed")
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, (int, float))
            or not math.isfinite(float(request_timeout_seconds))
            or not 0 < float(request_timeout_seconds) <= 7200
        ):
            raise TokenHubIdentityRelayError("request timeout is invalid")

        self._provider = parsed
        self._provider_credential = provider_credential
        self._expected_request_model = expected_request_model
        self._accepted_response_models = frozenset(accepted_response_models)
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._capability = "sieve-tokenhub-relay-" + secrets.token_hex(32)
        self._condition = threading.Condition()
        self._active_handlers = 0
        self._active_connections: set[http.client.HTTPConnection] = set()
        self._active_downstreams: set[socket.socket] = set()
        self._started = False
        self._closing = False
        self._closed = False
        self._verified_response_count = 0
        self._rejected_identity_count = 0
        self._provider_2xx_response_count = 0
        self._provider_2xx_pending_outer_result = 0
        self._provider_2xx_acknowledged_count = 0
        self._post_send_ambiguity_count = 0
        self._post_send_poisoned = False
        self._drain_count = 0
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

        self._server = ThreadingHTTPServer((LOOPBACK_HOST, 0), Handler)
        self._server.daemon_threads = False
        self._server.block_on_close = False
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="sieve-tokenhub-provider-identity-relay",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.port}"

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def capability(self) -> str:
        if not self._capability:
            raise TokenHubIdentityRelayError("relay capability is unavailable")
        return self._capability

    def start(self) -> None:
        with self._condition:
            if self._started or self._closing or self._closed:
                raise TokenHubIdentityRelayError("relay lifecycle is invalid")
        try:
            self._thread.start()
        except BaseException as exc:
            # ``BaseServer.shutdown`` waits for ``serve_forever`` to publish
            # its shutdown event.  If ``Thread.start`` itself failed, that
            # event can never be published, so close the listener directly
            # and permanently revoke the in-memory capabilities instead.
            failure: BaseException = exc
            try:
                self._server.server_close()
            except BaseException as close_exc:
                failure = close_exc
            self._provider_credential = ""
            self._capability = ""
            with self._condition:
                self._closing = True
                self._closed = True
                self._condition.notify_all()
            raise TokenHubIdentityRelayError(
                "identity relay thread failed to start"
            ) from failure
        with self._condition:
            self._started = True
        try:
            self.ensure_healthy()
        except BaseException:
            self.close()
            raise

    def ensure_healthy(self) -> None:
        with self._condition:
            if (
                not self._started
                or self._closing
                or self._closed
                or not self._thread.is_alive()
                or not self._provider_credential
                or not self._capability
            ):
                raise TokenHubIdentityRelayError("identity relay is not healthy")

    def post_send_ambiguity_detected(self) -> bool:
        """Return the private sideband consumed by the outer model gateway."""

        with self._condition:
            return self._post_send_poisoned

    def classify_outer_result(self, outer_request_complete: bool) -> bool:
        """Atomically reconcile one LiteLLM result with its raw-provider call.

        A provider 2xx may be completely consumed and then fail inside
        LiteLLM before LiteLLM starts its OpenAI SSE response.  Such a local
        5xx is not an explicit provider 5xx and must never be retried.  Every
        raw provider 2xx therefore remains pending until the outer gateway
        proves a complete response.  Explicit provider non-2xx responses do
        not create pending success state and retain the registered retry
        policy.
        """

        if not isinstance(outer_request_complete, bool):
            raise TokenHubIdentityRelayError(
                "outer request completion signal is malformed"
            )
        with self._condition:
            if self._post_send_poisoned:
                return True
            pending = self._provider_2xx_pending_outer_result
            if outer_request_complete and pending == 1:
                self._provider_2xx_pending_outer_result = 0
                self._provider_2xx_acknowledged_count += 1
                return False
            if not outer_request_complete and pending == 0:
                # A pre-connect/local failure or explicit provider non-2xx is
                # safe to classify using the ordinary gateway retry policy.
                return False
            # A successful LiteLLM response without exactly one raw provider
            # 2xx, multiple provider 2xx calls for one physical attempt, or a
            # local failure after any provider 2xx is conservatively terminal.
            self._mark_post_send_ambiguity_locked()
            return True

    def drain_ambiguous(self) -> None:
        """Abort every raw-provider request before a new LiteLLM generation."""

        with self._condition:
            if not self._started or self._closed:
                raise TokenHubIdentityRelayError("relay drain lifecycle is invalid")
            connections = tuple(self._active_connections)
            downstreams = tuple(self._active_downstreams)
        deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
        while True:
            for connection in connections:
                self._abort_connection(connection)
            for downstream in downstreams:
                self._abort_downstream(downstream)
            with self._condition:
                if not self._active_handlers and not self._active_connections:
                    self._drain_count += 1
                    self._post_send_poisoned = False
                    self._provider_2xx_pending_outer_result = 0
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TokenHubIdentityRelayError(
                        "identity relay upstream drain did not complete"
                    )
                self._condition.wait(timeout=min(0.1, remaining))
                connections = tuple(self._active_connections)
                downstreams = tuple(self._active_downstreams)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closing = True
        failure: BaseException | None = None
        try:
            if self._started:
                self._server.shutdown()
                self.drain_ambiguous()
                if self._thread.is_alive():
                    self._thread.join(timeout=5.0)
                    if self._thread.is_alive():
                        raise TokenHubIdentityRelayError("relay thread did not stop")
        except BaseException as exc:
            failure = exc
        finally:
            try:
                self._server.server_close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
            self._provider_credential = ""
            self._capability = ""
            with self._condition:
                self._closed = True
                self._condition.notify_all()
        if failure is not None:
            raise TokenHubIdentityRelayError("identity relay close failed") from failure

    def public_record(self) -> dict[str, Any]:
        with self._condition:
            return {
                "schema_version": "sieve_tokenhub_provider_identity_relay_v1",
                "raw_response_identity_field": "message_start.message.model",
                "accepted_response_models": sorted(self._accepted_response_models),
                "expected_request_model": self._expected_request_model,
                "verified_response_count": self._verified_response_count,
                "rejected_identity_count": self._rejected_identity_count,
                "provider_2xx_response_count": self._provider_2xx_response_count,
                "provider_2xx_acknowledged_count": (
                    self._provider_2xx_acknowledged_count
                ),
                "provider_2xx_pending_outer_result": (
                    self._provider_2xx_pending_outer_result
                ),
                "post_send_ambiguity_count": self._post_send_ambiguity_count,
                "post_send_poisoned": self._post_send_poisoned,
                "drain_count": self._drain_count,
                "loopback_only": True,
                "provider_credential_forwarded_to_litellm": False,
                "provider_endpoint_credential_request_response_or_reasoning_recorded": False,
            }

    def _handle_tracked(self, handler: BaseHTTPRequestHandler) -> None:
        downstream = handler.connection
        with self._condition:
            if self._closing:
                self._abort_downstream(downstream)
                return
            self._active_handlers += 1
            self._active_downstreams.add(downstream)
        try:
            self._handle(handler)
        finally:
            with self._condition:
                self._active_handlers -= 1
                self._active_downstreams.discard(downstream)
                self._condition.notify_all()

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        supplied = handler.headers.get("x-api-key", "")
        if not secrets.compare_digest(supplied, self._capability):
            self._reject(handler, 401, "invalid_identity_relay_capability")
            return
        if handler.path != "/v1/messages":
            self._reject(handler, 404, "endpoint_not_allowed")
            return
        with self._condition:
            if self._post_send_poisoned:
                self._reject(handler, 503, "provider_transport_recovery_required")
                return
        try:
            length = int(handler.headers.get("Content-Length", ""))
        except ValueError:
            self._reject(handler, 411, "content_length_required")
            return
        if length < 1 or length > MAX_REQUEST_BYTES:
            self._reject(handler, 413, "request_too_large")
            return
        body = handler.rfile.read(length)
        if len(body) != length:
            self._reject(handler, 400, "request_body_incomplete")
            return
        try:
            payload = json.loads(body.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeError, ValueError):
            self._reject(handler, 400, "invalid_json")
            return
        if (
            not isinstance(payload, dict)
            or payload.get("model") != self._expected_request_model
            or payload.get("stream") is not True
        ):
            self._reject(handler, 400, "provider_request_contract_mismatch")
            return

        connection = self._provider_connection()
        with self._condition:
            if self._closing:
                self._reject(handler, 503, "identity_relay_closing")
                return
            self._active_connections.add(connection)
        downstream_started = False
        provider_send_started = False
        provider_terminal_response = False
        try:
            headers = {
                "Accept": "text/event-stream",
                "Accept-Encoding": "identity",
                "Content-Type": "application/json",
                "User-Agent": "sieve-tokenhub-identity-relay/1.0",
                "x-api-key": self._provider_credential,
            }
            anthropic_version = handler.headers.get("anthropic-version")
            if anthropic_version:
                headers["anthropic-version"] = anthropic_version
            anthropic_beta = handler.headers.get("anthropic-beta")
            if anthropic_beta:
                headers["anthropic-beta"] = anthropic_beta
            # Keep the connect deadline short, then apply the frozen model
            # request timeout to response headers and all stream reads.  The
            # HTTPConnection constructor has only one timeout field, so an
            # explicit connect is required to separate these two boundaries.
            connection.connect()
            if connection.sock is None:
                raise TokenHubIdentityRelayError(
                    "provider connection did not publish a socket"
                )
            connection.sock.settimeout(self._request_timeout_seconds)
            provider_send_started = True
            connection.request("POST", "/v1/messages", body=body, headers=headers)
            response = connection.getresponse()
            if not 200 <= response.status <= 299:
                # An explicit provider HTTP response is a terminal attempt;
                # forwarding its optional body cannot make the generation
                # state ambiguous again.
                provider_terminal_response = True
                downstream_started = True
                self._begin_downstream(handler, response)
                self._copy_response(response, handler)
                return
            with self._condition:
                self._provider_2xx_response_count += 1
                self._provider_2xx_pending_outer_result += 1
            content_type = str(response.getheader("Content-Type") or "")
            if content_type.split(";", 1)[0].strip().lower() != "text/event-stream":
                self._mark_post_send_ambiguity()
                self._reject(handler, 502, "provider_stream_content_type_mismatch")
                return
            preface = bytearray()
            identity: str | None = None
            while len(preface) <= MAX_IDENTITY_PREFACE_BYTES:
                identity = _anthropic_message_start_model(bytes(preface))
                if identity is not None:
                    break
                chunk = _read_chunk(response, 16 * 1024)
                if not chunk:
                    break
                preface.extend(chunk)
            if identity is None:
                with self._condition:
                    self._rejected_identity_count += 1
                self._mark_post_send_ambiguity()
                self._reject(handler, 502, "provider_model_identity_missing")
                return
            if identity not in self._accepted_response_models:
                with self._condition:
                    self._rejected_identity_count += 1
                self._mark_post_send_ambiguity()
                self._reject(handler, 502, "provider_model_identity_mismatch")
                return
            with self._condition:
                self._verified_response_count += 1
            downstream_started = True
            self._begin_downstream(handler, response)
            terminal = _AnthropicTerminalTracker()
            terminal.feed(bytes(preface))
            provider_terminal_response = terminal.complete
            if preface:
                handler.wfile.write(preface)
                handler.wfile.flush()
            while True:
                chunk = _read_chunk(response, 64 * 1024)
                if not chunk:
                    break
                terminal.feed(chunk)
                provider_terminal_response = terminal.complete
                handler.wfile.write(chunk)
                handler.wfile.flush()
            terminal.finish()
            provider_terminal_response = True
        except Exception:
            if provider_send_started and not provider_terminal_response:
                self._mark_post_send_ambiguity()
            if not downstream_started:
                self._reject(handler, 502, "provider_transport_failure")
        finally:
            with self._condition:
                self._active_connections.discard(connection)
                self._condition.notify_all()
            self._abort_connection(connection)

    def _provider_connection(self) -> http.client.HTTPConnection:
        cls = (
            http.client.HTTPSConnection
            if self._provider.scheme == "https"
            else http.client.HTTPConnection
        )
        return cls(
            self._provider.hostname,
            self._provider.port or (443 if self._provider.scheme == "https" else 80),
            timeout=min(self._request_timeout_seconds, CONNECT_TIMEOUT_SECONDS),
        )

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

    @staticmethod
    def _copy_response(
        response: http.client.HTTPResponse, handler: BaseHTTPRequestHandler
    ) -> None:
        while True:
            chunk = _read_chunk(response, 64 * 1024)
            if not chunk:
                return
            handler.wfile.write(chunk)
            handler.wfile.flush()

    @staticmethod
    def _reject(handler: BaseHTTPRequestHandler, status: int, code: str) -> None:
        body = json.dumps(
            {"type": "error", "error": {"type": code, "message": code}},
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
            return

    @staticmethod
    def _abort_connection(connection: http.client.HTTPConnection) -> None:
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
        try:
            downstream.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            downstream.close()
        except OSError:
            pass

    def _mark_post_send_ambiguity(self) -> None:
        with self._condition:
            self._mark_post_send_ambiguity_locked()

    def _mark_post_send_ambiguity_locked(self) -> None:
        if not self._post_send_poisoned:
            self._post_send_ambiguity_count += 1
        self._post_send_poisoned = True
        self._condition.notify_all()


class _AnthropicTerminalTracker:
    """Boundedly require a raw Anthropic ``message_stop`` SSE event."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._complete = False

    def feed(self, chunk: bytes) -> None:
        self._buffer.extend(chunk)
        while True:
            boundary = _first_sse_boundary(self._buffer)
            if boundary is None:
                if len(self._buffer) > MAX_IDENTITY_PREFACE_BYTES:
                    raise TokenHubIdentityRelayError(
                        "provider stream frame exceeds the relay bound"
                    )
                return
            index, width = boundary
            frame = bytes(self._buffer[:index]).replace(b"\r\n", b"\n")
            del self._buffer[: index + width]
            data_lines = []
            for line in frame.split(b"\n"):
                if line.startswith(b"data:"):
                    value = line[5:]
                    if value.startswith(b" "):
                        value = value[1:]
                    data_lines.append(value)
            if not data_lines:
                continue
            try:
                event = json.loads(b"\n".join(data_lines).decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise TokenHubIdentityRelayError(
                    "provider stream contains malformed JSON"
                ) from exc
            if isinstance(event, Mapping) and event.get("type") == "message_stop":
                self._complete = True

    @property
    def complete(self) -> bool:
        return self._complete

    def finish(self) -> None:
        if not self._complete:
            raise TokenHubIdentityRelayError(
                "provider stream ended without message_stop"
            )


def _first_sse_boundary(buffer: bytearray) -> tuple[int, int] | None:
    candidates = []
    for marker in (b"\n\n", b"\r\n\r\n"):
        index = buffer.find(marker)
        if index >= 0:
            candidates.append((index, len(marker)))
    return min(candidates) if candidates else None


def _anthropic_message_start_model(buffer: bytes) -> str | None:
    """Return the raw Anthropic model from a complete message-start frame."""

    normalized = buffer.replace(b"\r\n", b"\n")
    for frame in normalized.split(b"\n\n")[:-1]:
        data_lines = []
        for line in frame.split(b"\n"):
            if line.startswith(b"data:"):
                value = line[5:]
                if value.startswith(b" "):
                    value = value[1:]
                data_lines.append(value)
        if not data_lines:
            continue
        try:
            event = json.loads(b"\n".join(data_lines).decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise TokenHubIdentityRelayError(
                "provider stream contains malformed JSON"
            ) from exc
        if not isinstance(event, Mapping) or event.get("type") != "message_start":
            continue
        message = event.get("message")
        model = message.get("model") if isinstance(message, Mapping) else None
        if not isinstance(model, str) or not model:
            raise TokenHubIdentityRelayError(
                "provider message-start identity is malformed"
            )
        return model
    return None


def _read_chunk(response: http.client.HTTPResponse, size: int) -> bytes:
    reader = getattr(response, "read1", None)
    if callable(reader):
        return bytes(reader(size))
    return bytes(response.read(size))


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
