#!/usr/bin/env python3
"""Credential-free real-Pi/real-LiteLLM TokenHub compatibility gate.

The fake Anthropic endpoint is loopback-only and holds request bodies in the
request handler stack only.  It persists no headers, prompts, model output,
reasoning, or signatures.  The returned record contains stable booleans and
counts only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import http.client
import json
from pathlib import Path
import secrets
import shutil
import socket
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit

from api_profiles import ExperimentProfile, ProfileRegistry, RuntimeBindings
from managed_transport import (
    LOOPBACK_HOST,
    TOKENHUB_ADAPTER,
    ManagedTransportError,
    _TokenHubLiteLLM,
)
from model_gateway import (
    ScopedModelGateway,
    SharedCooldownGate,
    verify_gateway_audit,
)
from pi_harness import (
    ROUTE_PREFLIGHT_FIXTURE,
    ROUTE_PREFLIGHT_SYSTEM_PROMPT,
    ROUTE_PREFLIGHT_TASK_PROMPT,
)
from preflight_pi_route import run_pi_route_preflight


FIXTURE_THINKING = "SIEVE private signed-thinking replay fixture"
FIXTURE_SIGNATURE = "sieve_opaque_signature_fixture_v1"
FIXTURE_SIGNATURE_PARTS = ("sieve_opaque_signature_", "fixture_v1")
FIXTURE_TOOL_CALL_ID = "call_sieve_fixture"
FIXTURE_TOOL_NAME = "read"
FIXTURE_TOOL_ARGUMENTS = {"path": "preflight_fixture.txt"}
FIXTURE_PROVIDER_KEY = "sieve-synthetic-local-fixture-key"
EXPECTED_TOOL_NAMES = frozenset({"read", "write", "edit", "bash"})
# Filled from, and then frozen against, the exact pinned Pi 0.85 -> pinned
# LiteLLM 1.83.14 Anthropic tool serialization.  This is not a self-consistency
# check between turn one and turn two.
EXPECTED_ANTHROPIC_FOUR_TOOL_SHA256 = (
    "bbf1a2b2496a207c2fedafefa7dd9f5eb2526094426eabb8ab6939ba5bc625e1"
)
EXPECTED_ANTHROPIC_REQUEST_FIELDS = frozenset(
    {"model", "messages", "max_tokens", "stream", "thinking", "tools", "system"}
)
EXPECTED_ABORT_PROBE_REQUEST_FIELDS = frozenset(
    {"model", "messages", "max_tokens", "stream", "thinking"}
)
MAX_FIXTURE_REQUEST_BYTES = 16 * 1024 * 1024
FIXTURE_READ_TIMEOUT_SECONDS = 5.0
ABORT_PROBE_GATEWAY_TIMEOUT_SECONDS = 0.35
ABORT_PROBE_DRAIN_TIMEOUT_SECONDS = 5.0
MAX_PUBLIC_AUDIT_FILES = 2_048
MAX_PUBLIC_AUDIT_BYTES = 64 * 1024 * 1024
FORBIDDEN_RAW_ARTIFACT_NAMES = frozenset(
    {
        "api-response.body",
        "request-headers.json",
        "request.json",
        "response.body",
        "response.json",
    }
)
BOUNDARY_REQUEST_MARKER = "SIEVE boundary fixture"
BOUNDARY_RESPONSE_MARKERS = (
    "wrong-provider-model",
    "malformed-fixture-usage",
    '"unexpected":"non-stream"',
    '"type":"rate_limit_error"',
)


@dataclass
class _FixtureState:
    requests: int = 0
    first_request_contract_exact: bool = False
    second_request_contract_exact: bool = False
    signed_thinking_exact: bool = False
    tool_use_identity_exact: bool = False
    tool_result_identity_exact: bool = False
    provider_auth_contract_exact: bool = False
    four_tool_contract_exact: bool = False
    system_prompt_contract_exact: bool = False
    task_prompt_contract_exact: bool = False
    abort_probe_upstream_disconnected: bool = False
    abort_probe_followup_complete: bool = False
    abort_probe_active_requests: int = field(default=0, repr=False)
    abort_probe_max_active_requests: int = field(default=0, repr=False)
    abort_probe_disconnect_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    validation_errors: list[str] = field(default_factory=list)
    forbidden_proxy_key: str = field(default="", repr=False)
    first_tools_sha256: str = field(default="", repr=False)
    first_system_sha256: str = field(default="", repr=False)
    first_user_sha256: str = field(default="", repr=False)
    auth_exact_requests: int = field(default=0, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def next_request(self) -> int:
        with self.lock:
            self.requests += 1
            return self.requests

    def fail(self, code: str) -> None:
        with self.lock:
            if code not in self.validation_errors:
                self.validation_errors.append(code)

    def bind_proxy_key(self, value: str) -> None:
        with self.lock:
            if self.forbidden_proxy_key or not value:
                raise ManagedTransportError("fixture proxy-key binding is invalid")
            self.forbidden_proxy_key = value

    def mark_auth_exact(self) -> None:
        with self.lock:
            self.auth_exact_requests += 1
            self.provider_auth_contract_exact = self.auth_exact_requests == 4

    def begin_abort_probe_request(self) -> None:
        with self.lock:
            self.abort_probe_active_requests += 1
            self.abort_probe_max_active_requests = max(
                self.abort_probe_max_active_requests,
                self.abort_probe_active_requests,
            )

    def end_abort_probe_request(self) -> None:
        with self.lock:
            self.abort_probe_active_requests -= 1

    def mark_abort_probe_disconnected(self) -> None:
        with self.lock:
            self.abort_probe_upstream_disconnected = True
        self.abort_probe_disconnect_event.set()

    def mark_abort_probe_followup_complete(self) -> None:
        with self.lock:
            self.abort_probe_followup_complete = True

    def bind_or_check_tools(self, tools: Any, *, first: bool) -> None:
        digest = _validate_four_tools_and_hash(tools)
        with self.lock:
            if first:
                if self.first_tools_sha256:
                    raise _FixtureContractError("first_tools_repeated")
                self.first_tools_sha256 = digest
            elif not self.first_tools_sha256 or not secrets.compare_digest(
                self.first_tools_sha256, digest
            ):
                raise _FixtureContractError("second_tools_contract_mismatch")
            self.four_tool_contract_exact = bool(
                self.first_tools_sha256 and not first
            )

    def bind_or_check_prompts(self, payload: Mapping[str, Any], *, first: bool) -> None:
        system_text, system_digest = _exact_anthropic_text_and_hash(
            payload.get("system"), "provider_system_prompt"
        )
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or not isinstance(
            messages[0], dict
        ):
            raise _FixtureContractError("provider_task_prompt_missing")
        user_text, user_digest = _exact_anthropic_text_and_hash(
            messages[0].get("content"), "provider_task_prompt"
        )
        if system_text != ROUTE_PREFLIGHT_SYSTEM_PROMPT:
            raise _FixtureContractError("provider_system_prompt_mismatch")
        # Pi's stdin prompt is line-delimited, and its CLI removes that one
        # transport newline before constructing the user message.
        if user_text != ROUTE_PREFLIGHT_TASK_PROMPT.rstrip("\n"):
            raise _FixtureContractError("provider_task_prompt_mismatch")
        with self.lock:
            if first:
                if self.first_system_sha256 or self.first_user_sha256:
                    raise _FixtureContractError("provider_prompt_contract_repeated")
                self.first_system_sha256 = system_digest
                self.first_user_sha256 = user_digest
            elif (
                not self.first_system_sha256
                or not self.first_user_sha256
                or not secrets.compare_digest(self.first_system_sha256, system_digest)
                or not secrets.compare_digest(self.first_user_sha256, user_digest)
            ):
                raise _FixtureContractError("provider_prompt_contract_drifted")
            self.system_prompt_contract_exact = bool(
                self.first_system_sha256 and not first
            )
            self.task_prompt_contract_exact = bool(
                self.first_user_sha256 and not first
            )

    def safe_record(self) -> dict[str, Any]:
        with self.lock:
            return {
                "upstream_requests": self.requests,
                "first_request_contract_exact": self.first_request_contract_exact,
                "second_request_contract_exact": self.second_request_contract_exact,
                "signed_thinking_text_and_signature_exact": self.signed_thinking_exact,
                "tool_use_identity_and_order_exact": self.tool_use_identity_exact,
                "tool_result_identity_exact": self.tool_result_identity_exact,
                "provider_auth_contract_exact": self.provider_auth_contract_exact,
                "four_tool_contract_exact": self.four_tool_contract_exact,
                "system_prompt_contract_exact": self.system_prompt_contract_exact,
                "task_prompt_contract_exact": self.task_prompt_contract_exact,
                "observed_anthropic_four_tool_sha256": self.first_tools_sha256,
                "abort_probe_upstream_disconnected": (
                    self.abort_probe_upstream_disconnected
                ),
                "abort_probe_no_request_overlap": (
                    self.abort_probe_max_active_requests == 1
                    and self.abort_probe_active_requests == 0
                ),
                "abort_probe_followup_complete": self.abort_probe_followup_complete,
                "validation_error_codes": list(self.validation_errors),
            }


class _LocalFixtureTokenHubLiteLLM(_TokenHubLiteLLM):
    """Use the production manager while keeping its HTTPS provider gate intact."""

    def __init__(self, *, local_provider_base: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        parsed = urlsplit(local_provider_base)
        if (
            parsed.scheme != "http"
            or parsed.hostname != LOOPBACK_HOST
            or parsed.port is None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ManagedTransportError(
                "local TokenHub release fixture must be a loopback HTTP root"
            )
        self._local_provider_base = local_provider_base.rstrip("/")

    def _provider_base(self) -> str:
        return self._local_provider_base

    def _allow_insecure_provider_for_release_gate(self) -> bool:
        return True


class _FixtureServer:
    def __init__(self) -> None:
        self.state = _FixtureState()
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_POST(self) -> None:  # noqa: N802
                self.connection.settimeout(FIXTURE_READ_TIMEOUT_SECONDS)
                request_number = state.next_request()
                try:
                    self._validate_headers()
                    payload = self._read_payload()
                    if self.path != "/v1/messages":
                        raise _FixtureContractError("anthropic_path_mismatch")
                    if request_number == 1:
                        _validate_first_request(payload, state)
                        body = _first_anthropic_stream()
                    elif request_number == 2:
                        _validate_second_request(payload, state)
                        body = _second_anthropic_stream()
                    elif request_number == 3:
                        _validate_abort_probe_request(payload)
                        self._serve_abort_probe()
                        return
                    elif request_number == 4:
                        _validate_abort_probe_request(payload)
                        self._serve_abort_probe_followup(
                            _abort_probe_complete_stream()
                        )
                        return
                    else:
                        raise _FixtureContractError("unexpected_upstream_request")
                except _FixtureContractError as exc:
                    state.fail(str(exc))
                    body = b'{"type":"error","error":{"type":"invalid_request_error","message":"fixture contract"}}'
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                # Exercise the complete real transport with boundaries that do
                # not align to Anthropic events or LiteLLM output chunks.
                for offset in range(0, len(body), 17):
                    self.wfile.write(body[offset : offset + 17])
                    self.wfile.flush()

            def _serve_abort_probe(self) -> None:
                state.begin_abort_probe_request()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(_abort_probe_start_stream())
                    self.wfile.flush()
                    deadline = time.monotonic() + ABORT_PROBE_DRAIN_TIMEOUT_SECONDS
                    while time.monotonic() < deadline:
                        try:
                            self.wfile.write(b": sieve-abort-probe-keepalive\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            state.mark_abort_probe_disconnected()
                            return
                        time.sleep(0.02)
                    state.fail("abort_probe_upstream_not_disconnected")
                finally:
                    state.end_abort_probe_request()

            def _serve_abort_probe_followup(self, body: bytes) -> None:
                state.begin_abort_probe_request()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    for offset in range(0, len(body), 17):
                        self.wfile.write(body[offset : offset + 17])
                        self.wfile.flush()
                    state.mark_abort_probe_followup_complete()
                finally:
                    state.end_abort_probe_request()

            def _validate_headers(self) -> None:
                if len(self.headers.get_all("x-api-key", [])) != 1:
                    raise _FixtureContractError("anthropic_provider_auth_count_mismatch")
                provider_key = self.headers.get("x-api-key", "")
                if not secrets.compare_digest(provider_key, FIXTURE_PROVIDER_KEY):
                    raise _FixtureContractError("anthropic_provider_auth_mismatch")
                if self.headers.get("Authorization") not in (None, ""):
                    raise _FixtureContractError("anthropic_authorization_unexpected")
                if self.headers.get("anthropic-version") != "2023-06-01":
                    raise _FixtureContractError("anthropic_version_header_mismatch")
                content_type = self.headers.get("Content-Type", "")
                if content_type.split(";", 1)[0].strip().lower() != "application/json":
                    raise _FixtureContractError("anthropic_content_type_mismatch")
                with state.lock:
                    forbidden = state.forbidden_proxy_key
                if not forbidden:
                    raise _FixtureContractError("fixture_proxy_key_not_bound")
                for value in self.headers.values():
                    if forbidden in value:
                        raise _FixtureContractError("local_proxy_key_leaked_upstream")
                state.mark_auth_exact()

            def _read_payload(self) -> dict[str, Any]:
                raw_length = self.headers.get("Content-Length", "")
                try:
                    length = int(raw_length)
                except ValueError as exc:
                    raise _FixtureContractError(
                        "anthropic_content_length_invalid"
                    ) from exc
                if length < 1 or length > MAX_FIXTURE_REQUEST_BYTES:
                    raise _FixtureContractError("anthropic_request_size_invalid")
                try:
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise _FixtureContractError(
                            "anthropic_request_body_truncated"
                        )
                    with state.lock:
                        forbidden = state.forbidden_proxy_key
                    if forbidden and forbidden.encode("utf-8") in body:
                        raise _FixtureContractError(
                            "local_proxy_key_leaked_in_upstream_body"
                        )
                    value = json.loads(
                        body.decode("utf-8"),
                        parse_constant=_reject_json_constant,
                    )
                except _FixtureContractError:
                    raise
                except (TimeoutError, socket.timeout, OSError) as exc:
                    raise _FixtureContractError(
                        "anthropic_request_body_read_failed"
                    ) from exc
                except (UnicodeError, ValueError) as exc:
                    raise _FixtureContractError("anthropic_request_json_invalid") from exc
                if not isinstance(value, dict):
                    raise _FixtureContractError("anthropic_request_root_invalid")
                return value

            def log_message(self, format: str, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer((LOOPBACK_HOST, 0), Handler)
        self.server.daemon_threads = False
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="sieve-tokenhub-fake-anthropic",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.server.server_address[1]}"

    def __enter__(self) -> "_FixtureServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise ManagedTransportError("local Anthropic fixture did not stop")


class _FixtureContractError(RuntimeError):
    pass


class _TransportBoundaryFixture:
    """Exercise raw identity failures and an explicit retryable response."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_POST(self) -> None:  # noqa: N802
                self.connection.settimeout(FIXTURE_READ_TIMEOUT_SECONDS)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.close_connection = True
                    return
                if length < 1 or length > MAX_FIXTURE_REQUEST_BYTES:
                    self.close_connection = True
                    return
                try:
                    body = self.rfile.read(length)
                except OSError:
                    self.close_connection = True
                    return
                if len(body) != length:
                    self.close_connection = True
                    return
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeError, ValueError):
                    self._send_json(400, b'{"error":"invalid_fixture_json"}')
                    return
                if (
                    self.path != "/v1/messages"
                    or self.headers.get("x-api-key") != FIXTURE_PROVIDER_KEY
                    or not isinstance(payload, dict)
                    or payload.get("model") != "hy4-preview"
                    or payload.get("stream") is not True
                ):
                    self._send_json(400, b'{"error":"fixture_contract"}')
                    return
                with owner._lock:
                    owner.requests += 1
                    request_number = owner.requests
                if request_number == 1:
                    stream = _abort_probe_complete_stream().replace(
                        b'"model":"hy4-preview"',
                        b'"model":"wrong-provider-model"',
                        1,
                    )
                    self._send_stream(stream)
                elif request_number == 2:
                    self._send_stream(
                        _anthropic_sse(
                            [("message_stop", {"type": "message_stop"})]
                        )
                    )
                elif request_number == 3:
                    self._send_json(200, b'{"unexpected":"non-stream"}')
                elif request_number == 4:
                    self._send_stream(
                        b'event: message_start\ndata: {"type":"message_start"'
                    )
                elif request_number == 5:
                    # Raw identity and terminal framing are valid, but the
                    # usage shape makes pinned LiteLLM fail while transforming
                    # a provider-success stream.  The relay's pending-2xx
                    # handshake must prevent that local 5xx from being retried.
                    self._send_stream(
                        _anthropic_sse(
                            [
                                (
                                    "message_start",
                                    {
                                        "type": "message_start",
                                        "message": {
                                            "model": "hy4-preview",
                                            "usage": "malformed-fixture-usage",
                                        },
                                    },
                                ),
                                ("message_stop", {"type": "message_stop"}),
                            ]
                        )
                    )
                elif request_number == 6:
                    self._send_json(
                        429,
                        b'{"type":"error","error":{"type":"rate_limit_error"}}',
                    )
                elif request_number == 7:
                    self._send_stream(_abort_probe_complete_stream())
                else:
                    self._send_json(400, b'{"error":"unexpected_request"}')

            def _send_stream(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer((LOOPBACK_HOST, 0), Handler)
        self.server.daemon_threads = False
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="sieve-tokenhub-transport-boundary-fixture",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.server.server_address[1]}"

    def __enter__(self) -> "_TransportBoundaryFixture":
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise ManagedTransportError("transport boundary fixture did not stop")


def _verify_public_gate_artifacts(
    *, roots: tuple[Path, ...], forbidden_values: tuple[str, ...]
) -> dict[str, Any]:
    """Fail closed if this gate persisted known private/raw transport material."""

    needles = tuple(
        sorted(
            {value.encode("utf-8") for value in forbidden_values if value},
            key=len,
            reverse=True,
        )
    )
    raw_header_markers = (
        b"authorization:",
        b"x-api-key:",
        b'"authorization"',
        b'"x-api-key"',
    )
    file_count = 0
    symlink_count = 0
    byte_count = 0

    def reject_private_material(payload: bytes) -> None:
        if any(needle in payload for needle in needles):
            raise ManagedTransportError(
                "public gate audit contains known private transport material"
            )
        lowered = payload.lower()
        if any(marker in lowered for marker in raw_header_markers):
            raise ManagedTransportError(
                "public gate audit contains a raw credential header"
            )

    for root in roots:
        if not root.is_dir() or root.is_symlink():
            raise ManagedTransportError("public gate audit root is invalid")
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            reject_private_material(
                path.relative_to(root).as_posix().encode("utf-8")
            )
            if path.name.casefold() in FORBIDDEN_RAW_ARTIFACT_NAMES:
                raise ManagedTransportError(
                    "public gate audit contains a forbidden raw artifact"
                )
            if path.is_symlink():
                symlink_count += 1
                payload = path.readlink().as_posix().encode("utf-8")
            elif path.is_dir():
                continue
            elif path.is_file():
                file_count += 1
                if file_count > MAX_PUBLIC_AUDIT_FILES:
                    raise ManagedTransportError("public gate audit file bound exceeded")
                size = path.stat().st_size
                if size < 0 or byte_count + size > MAX_PUBLIC_AUDIT_BYTES:
                    raise ManagedTransportError("public gate audit byte bound exceeded")
                payload = path.read_bytes()
                if len(payload) != size:
                    raise ManagedTransportError("public gate audit changed during scan")
                byte_count += size
            else:
                raise ManagedTransportError("public gate audit entry type is invalid")
            reject_private_material(payload)
    return {
        "schema_version": "sieve_public_gate_artifact_scan_v1",
        "scan_complete": True,
        "roots_scanned": len(roots),
        "regular_files_scanned": file_count,
        "symlinks_scanned": symlink_count,
        "bytes_scanned": byte_count,
        "known_private_values_absent": True,
        "raw_credential_headers_absent": True,
        "forbidden_raw_artifact_names_absent": True,
    }


def verify_tokenhub_pi_litellm_roundtrip(
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    bindings: RuntimeBindings,
    pi_runtime_root: Path,
    tokenhub_runtime_root: Path,
) -> dict[str, Any]:
    """Prove signed-thinking replay through both pinned runtimes, without a key."""

    if experiment.api_family_id != "tokenhub" or experiment.model_profile_ids != (
        "tokenhub-hy4-preview-agent-v1",
    ):
        raise ManagedTransportError("TokenHub release gate experiment differs")
    model = registry.model(experiment.model_profile_ids[0])
    route = registry.route(model.route_profile_id)
    if route.transport_adapter_id != TOKENHUB_ADAPTER:
        raise ManagedTransportError("TokenHub release gate adapter differs")

    audit_root = (
        Path(__file__).resolve().parents[1]
        / "episodes"
        / "route-preflight"
        / model.model_profile_id
        / f"run-release-{secrets.token_hex(12)}"
    )
    report = None
    abort_probe = None
    manager: _LocalFixtureTokenHubLiteLLM | None = None
    manager_closed = False
    manager_listener_released = False
    identity_relay_listener_released = False
    identity_relay_record: Mapping[str, Any] | None = None
    public_artifact_scan: Mapping[str, Any] | None = None
    with _FixtureServer() as fixture:
        forbidden_public_values = [
            FIXTURE_THINKING,
            FIXTURE_SIGNATURE,
            FIXTURE_SIGNATURE_PARTS[0],
            FIXTURE_PROVIDER_KEY,
            fixture.base_url,
        ]
        with _temporary_directory("sieve-tokenhub-pi-litellm-gate-") as temporary:
            invocation_root = temporary / "transport"
            invocation_root.mkdir(mode=0o700)
            manager = _LocalFixtureTokenHubLiteLLM(
                registry=registry,
                experiment=experiment,
                bindings=bindings,
                provider_credential=FIXTURE_PROVIDER_KEY,
                invocation_root=invocation_root,
                runtime_root=tokenhub_runtime_root,
                local_provider_base=fixture.base_url,
            )
            try:
                manager.start()
                transport = manager.transport()
                forbidden_public_values.extend(
                    (
                        transport.credential,
                        transport.bindings.for_route(
                            route.route_profile_id
                        ).upstream_base_url,
                    )
                )
                fixture.state.bind_proxy_key(transport.credential)
                report = run_pi_route_preflight(
                    runtime_root=pi_runtime_root,
                    audit_root=audit_root,
                    experiment_id="tokenhub-signed-replay-release-gate-v1",
                    experiment_sha256=hashlib.sha256(
                        b"tokenhub-signed-replay-release-gate-v1"
                    ).hexdigest(),
                    profile_registry_sha256=registry.content_sha256,
                    route=route,
                    model=model,
                    runtime_binding=transport.bindings.for_route(
                        route.route_profile_id
                    ),
                    runtime_credential=transport.credential,
                    cooldown_gate=SharedCooldownGate(),
                    wall_clock_seconds=120,
                    managed_transport_ambiguity_probe=(
                        transport.classify_outer_result
                    ),
                )
                abort_probe = _verify_gateway_abort_drains_litellm(
                    fixture=fixture,
                    transport=transport,
                    model=model,
                    route=route,
                )
                transport.ensure_healthy()
            finally:
                try:
                    manager.close()
                    manager_closed = True
                    completion_path = (
                        invocation_root / "transport_completion.json"
                    )
                    completion = json.loads(
                        completion_path.read_text(encoding="utf-8")
                    )
                    manager_listener_released = bool(
                        completion.get("ended") is True
                        and completion.get("listener_released") is True
                    )
                    identity_relay_listener_released = bool(
                        completion.get(
                            "provider_identity_relay_listener_released"
                        )
                        is True
                    )
                    raw_relay_record = completion.get(
                        "provider_identity_relay"
                    )
                    if isinstance(raw_relay_record, Mapping):
                        identity_relay_record = raw_relay_record
                finally:
                    try:
                        public_artifact_scan = _verify_public_gate_artifacts(
                            roots=(audit_root, invocation_root),
                            forbidden_values=tuple(forbidden_public_values),
                        )
                    finally:
                        if audit_root.exists() and not audit_root.is_symlink():
                            shutil.rmtree(audit_root)

        state = fixture.state.safe_record()

    if report is None or not report.ok:
        fixture_codes = state.get("validation_error_codes")
        if isinstance(fixture_codes, list) and fixture_codes:
            safe_suffix = ":" + ",".join(str(code) for code in fixture_codes)
        else:
            failure_code = report.failure_code if report is not None else "missing"
            safe_suffix = f":pi_{failure_code}"
        raise ManagedTransportError(
            "real Pi/LiteLLM signed-replay preflight failed" + safe_suffix
        )
    if state != {
        "upstream_requests": 4,
        "first_request_contract_exact": True,
        "second_request_contract_exact": True,
        "signed_thinking_text_and_signature_exact": True,
        "tool_use_identity_and_order_exact": True,
        "tool_result_identity_exact": True,
        "provider_auth_contract_exact": True,
        "four_tool_contract_exact": True,
        "system_prompt_contract_exact": True,
        "task_prompt_contract_exact": True,
        "observed_anthropic_four_tool_sha256": (
            EXPECTED_ANTHROPIC_FOUR_TOOL_SHA256
        ),
        "abort_probe_upstream_disconnected": True,
        "abort_probe_no_request_overlap": True,
        "abort_probe_followup_complete": True,
        "validation_error_codes": [],
    }:
        raise ManagedTransportError("Anthropic-boundary signed replay differs")
    if (
        identity_relay_record is None
        or identity_relay_record.get("verified_response_count") != 4
        or identity_relay_record.get("rejected_identity_count") != 0
        or identity_relay_record.get("provider_2xx_response_count") != 4
        or identity_relay_record.get("provider_2xx_acknowledged_count") != 3
        or identity_relay_record.get("provider_2xx_pending_outer_result") != 0
        or identity_relay_record.get("post_send_ambiguity_count") != 1
        or identity_relay_record.get("provider_credential_forwarded_to_litellm")
        is not False
        or identity_relay_listener_released is not True
    ):
        raise ManagedTransportError("raw-provider identity relay proof differs")
    if public_artifact_scan is None or public_artifact_scan.get("scan_complete") is not True:
        raise ManagedTransportError("public gate artifact scan is missing")
    transport_boundary_probe = _verify_real_litellm_transport_boundaries(
        registry=registry,
        experiment=experiment,
        bindings=bindings,
        tokenhub_runtime_root=tokenhub_runtime_root,
    )
    return {
        "schema_version": "sieve_tokenhub_pi_litellm_release_gate_v2",
        "ok": True,
        "real_pinned_pi_used": True,
        "real_pinned_litellm_used": True,
        "loopback_fake_anthropic_used": True,
        "production_provider_credential_used": False,
        "synthetic_loopback_fixture_credential_observed": state[
            "provider_auth_contract_exact"
        ],
        "configured_provider_target_loopback_only": True,
        "logical_model_requests": report.logical_requests,
        "exactly_one_read_tool_roundtrip": report.exactly_one_read_call,
        "final_marker_exact": report.final_marker_exact,
        "gateway_stream_contract_complete": report.gateway_stream_contract_complete,
        "response_identity_matches": report.response_identity_matches,
        "signed_thinking_text_and_signature_exact": state[
            "signed_thinking_text_and_signature_exact"
        ],
        "tool_use_identity_and_order_exact": state[
            "tool_use_identity_and_order_exact"
        ],
        "tool_result_identity_exact": state["tool_result_identity_exact"],
        "four_tool_contract_exact": state["four_tool_contract_exact"],
        "provider_auth_contract_exact": state["provider_auth_contract_exact"],
        "system_prompt_contract_exact": state["system_prompt_contract_exact"],
        "task_prompt_contract_exact": state["task_prompt_contract_exact"],
        "anthropic_four_tool_contract_sha256": state[
            "observed_anthropic_four_tool_sha256"
        ],
        "ambiguous_timeout_disconnect_barrier": abort_probe,
        "managed_adapter_ended": manager_closed,
        "managed_adapter_listener_released": manager_listener_released,
        "raw_provider_identity_verified_before_litellm": True,
        "raw_provider_identity_verified_response_count": 4,
        "raw_provider_identity_rejected_response_count": 0,
        "provider_identity_relay_listener_released": (
            identity_relay_listener_released
        ),
        "real_litellm_transport_boundary_probe": transport_boundary_probe,
        "public_artifact_safety_scan": dict(public_artifact_scan),
        "raw_request_response_reasoning_or_signature_recorded": not bool(
            public_artifact_scan.get("known_private_values_absent")
            and public_artifact_scan.get("raw_credential_headers_absent")
            and public_artifact_scan.get("forbidden_raw_artifact_names_absent")
        ),
    }


def _verify_real_litellm_transport_boundaries(
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    bindings: RuntimeBindings,
    tokenhub_runtime_root: Path,
) -> dict[str, Any]:
    """Prove relay sideband beats LiteLLM 5xx normalization before retry."""

    base_model = registry.model(experiment.model_profile_ids[0])
    route = registry.route(base_model.route_profile_id)
    probe_model = replace(
        base_model,
        request_timeout_seconds=30,
        retry=replace(
            base_model.retry,
            max_infrastructure_retries=1,
            retry_delay_seconds=0,
        ),
    )
    payload = {
        "model": probe_model.client_wire_model,
        "messages": [{"role": "user", "content": "SIEVE boundary fixture"}],
        "stream": True,
        "max_tokens": 65536,
    }
    manager_closed = False
    completion: Mapping[str, Any] | None = None
    relay_record: Mapping[str, Any] | None = None
    public_artifact_scan: Mapping[str, Any] | None = None
    with _TransportBoundaryFixture() as fixture:
        forbidden_public_values = [
            FIXTURE_PROVIDER_KEY,
            fixture.base_url,
            BOUNDARY_REQUEST_MARKER,
            *BOUNDARY_RESPONSE_MARKERS,
        ]
        with _temporary_directory("sieve-tokenhub-boundary-gate-") as temporary:
            invocation_root = temporary / "transport"
            invocation_root.mkdir(mode=0o700)
            manager = _LocalFixtureTokenHubLiteLLM(
                registry=registry,
                experiment=experiment,
                bindings=bindings,
                provider_credential=FIXTURE_PROVIDER_KEY,
                invocation_root=invocation_root,
                runtime_root=tokenhub_runtime_root,
                local_provider_base=fixture.base_url,
            )
            try:
                manager.start()
                transport = manager.transport()
                forbidden_public_values.extend(
                    (
                        transport.credential,
                        transport.bindings.for_route(
                            route.route_profile_id
                        ).upstream_base_url,
                    )
                )
                ambiguous_scenarios = (
                    "wrong_raw_model_identity",
                    "missing_raw_model_identity",
                    "wrong_success_content_type",
                    "truncated_identity_preface",
                    "post_identity_provider_success_transform_failure",
                )
                for index, scenario in enumerate(ambiguous_scenarios, start=1):
                    before = fixture.requests
                    event_path = (
                        invocation_root / f"boundary_ambiguous_{index:02d}.jsonl"
                    )
                    with ScopedModelGateway(
                        route=route,
                        model=probe_model,
                        runtime_binding=transport.bindings.for_route(
                            route.route_profile_id
                        ),
                        runtime_credential=transport.credential,
                        max_requests=1,
                        cooldown_gate=SharedCooldownGate(),
                        managed_transport_ambiguity_probe=(
                            transport.classify_outer_result
                        ),
                        event_path=event_path,
                    ) as gateway:
                        status = _post_gateway_probe(
                            gateway, route.client_path, payload
                        )
                        report = gateway.wait_for_completion_report()
                    audit = verify_gateway_audit(
                        event_path,
                        report,
                        expected_api_family_id=probe_model.api_family_id,
                        expected_route_profile_id=route.route_profile_id,
                        expected_model_profile_id=probe_model.model_profile_id,
                        expected_retry_policy=probe_model.retry.public_dict(),
                    )
                    if (
                        not 400 <= status <= 599
                        or fixture.requests != before + 1
                        or report.get("terminal_statuses")
                        != {"1": "ambiguous_managed_transport_sideband"}
                        or report.get("transport_recycle_required") is not True
                        or audit.get("event_count") != 2
                    ):
                        raise ManagedTransportError(
                            f"{scenario} crossed the safe retry boundary"
                        )
                    transport.recycle_after_ambiguous()
                    receipt = invocation_root / f"transport_recycle_{index:04d}.json"
                    if not receipt.is_file() or receipt.is_symlink():
                        raise ManagedTransportError(
                            f"{scenario} lacks a recycle receipt"
                        )

                before_retryable = fixture.requests
                retry_event_path = invocation_root / "boundary_retryable_429.jsonl"
                with ScopedModelGateway(
                    route=route,
                    model=probe_model,
                    runtime_binding=transport.bindings.for_route(
                        route.route_profile_id
                    ),
                    runtime_credential=transport.credential,
                    max_requests=1,
                    cooldown_gate=SharedCooldownGate(),
                    managed_transport_ambiguity_probe=(
                        transport.classify_outer_result
                    ),
                    event_path=retry_event_path,
                ) as gateway:
                    final_status = _post_gateway_probe(
                        gateway, route.client_path, payload
                    )
                    final_report = gateway.wait_for_completion_report()
                retry_audit = verify_gateway_audit(
                    retry_event_path,
                    final_report,
                    expected_api_family_id=probe_model.api_family_id,
                    expected_route_profile_id=route.route_profile_id,
                    expected_model_profile_id=probe_model.model_profile_id,
                    expected_retry_policy=probe_model.retry.public_dict(),
                )
                if (
                    final_status != 200
                    or fixture.requests != before_retryable + 2
                    or final_report.get("terminal_statuses") != {"1": "complete"}
                    or final_report.get("transport_recycle_required") is not False
                    or transport.post_send_ambiguity_detected() is not False
                    or retry_audit.get("event_count") != 4
                ):
                    raise ManagedTransportError(
                        "explicit provider 429 did not retain bounded retry semantics"
                    )
                transport.ensure_healthy()
            finally:
                try:
                    manager.close()
                    manager_closed = True
                    completion = json.loads(
                        (invocation_root / "transport_completion.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    raw_relay_record = completion.get("provider_identity_relay")
                    if isinstance(raw_relay_record, Mapping):
                        relay_record = raw_relay_record
                finally:
                    public_artifact_scan = _verify_public_gate_artifacts(
                        roots=(invocation_root,),
                        forbidden_values=tuple(forbidden_public_values),
                    )

        request_count = fixture.requests

    if (
        not manager_closed
        or completion is None
        or completion.get("ended") is not True
        or completion.get("listener_released") is not True
        or completion.get("provider_identity_relay_listener_released") is not True
        or completion.get("recycle_count") != 5
        or completion.get("runtime_reverification_count") != 5
        or relay_record is None
        or relay_record.get("verified_response_count") != 2
        or relay_record.get("rejected_identity_count") != 3
        or relay_record.get("provider_2xx_response_count") != 6
        or relay_record.get("provider_2xx_acknowledged_count") != 1
        or relay_record.get("provider_2xx_pending_outer_result") != 0
        or relay_record.get("post_send_ambiguity_count") != 5
        or relay_record.get("post_send_poisoned") is not False
        or request_count != 7
        or public_artifact_scan is None
        or public_artifact_scan.get("scan_complete") is not True
    ):
        raise ManagedTransportError("real LiteLLM transport-boundary proof differs")
    return {
        "wrong_raw_identity_terminal_without_retry": True,
        "missing_raw_identity_terminal_without_retry": True,
        "wrong_success_content_type_terminal_without_retry": True,
        "truncated_identity_preface_terminal_without_retry": True,
        "provider_2xx_then_litellm_transform_failure_terminal_without_retry": True,
        "explicit_provider_429_retried_once_then_completed": True,
        "ambiguous_scenarios": 5,
        "provider_requests": 7,
        "managed_transport_recycles": 5,
        "public_artifact_safety_scan": dict(public_artifact_scan),
        "provider_request_response_or_reasoning_recorded": not bool(
            public_artifact_scan.get("known_private_values_absent")
            and public_artifact_scan.get("raw_credential_headers_absent")
            and public_artifact_scan.get("forbidden_raw_artifact_names_absent")
        ),
    }


def _validate_first_request(payload: Mapping[str, Any], state: _FixtureState) -> None:
    _require_exact_keys(
        payload,
        EXPECTED_ANTHROPIC_REQUEST_FIELDS,
        "first_request_fields_mismatch",
    )
    if payload.get("model") != "hy4-preview" or payload.get("stream") is not True:
        raise _FixtureContractError("first_model_or_stream_mismatch")
    if payload.get("thinking") != {"type": "enabled", "budget_tokens": 4096}:
        raise _FixtureContractError("first_thinking_config_mismatch")
    if payload.get("max_tokens") != 65536:
        raise _FixtureContractError("first_max_tokens_mismatch")
    if "temperature" in payload:
        raise _FixtureContractError("first_temperature_must_be_absent")
    state.bind_or_check_tools(payload.get("tools"), first=True)
    state.bind_or_check_prompts(payload, first=True)
    messages = payload.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or set(messages[0]) != {"role", "content"}
        or messages[0].get("role") != "user"
        or messages[0].get("content") in (None, "", [], {})
    ):
        raise _FixtureContractError("first_messages_contract_mismatch")
    with state.lock:
        state.first_request_contract_exact = True


def _validate_second_request(payload: Mapping[str, Any], state: _FixtureState) -> None:
    _require_exact_keys(
        payload,
        EXPECTED_ANTHROPIC_REQUEST_FIELDS,
        "second_request_fields_mismatch",
    )
    if payload.get("model") != "hy4-preview" or payload.get("stream") is not True:
        raise _FixtureContractError("second_model_or_stream_mismatch")
    if payload.get("thinking") != {"type": "enabled", "budget_tokens": 4096}:
        raise _FixtureContractError("second_thinking_config_mismatch")
    if payload.get("max_tokens") != 65536:
        raise _FixtureContractError("second_max_tokens_mismatch")
    if "temperature" in payload:
        raise _FixtureContractError("second_temperature_must_be_absent")
    state.bind_or_check_tools(payload.get("tools"), first=False)
    state.bind_or_check_prompts(payload, first=False)
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise _FixtureContractError("second_messages_count_mismatch")
    first_user, assistant, result_user = messages
    if (
        not isinstance(first_user, dict)
        or set(first_user) != {"role", "content"}
        or first_user.get("role") != "user"
        or first_user.get("content") in (None, "", [], {})
        or not isinstance(assistant, dict)
        or set(assistant) != {"role", "content"}
        or assistant.get("role") != "assistant"
        or not isinstance(result_user, dict)
        or set(result_user) != {"role", "content"}
        or result_user.get("role") != "user"
    ):
        raise _FixtureContractError("second_messages_order_mismatch")
    assistant_content = assistant.get("content")
    if not isinstance(assistant_content, list):
        raise _FixtureContractError("second_assistant_content_missing")
    if len(assistant_content) != 2:
        raise _FixtureContractError("second_assistant_content_order_mismatch")
    thinking, tool_use = assistant_content
    signed_exact = thinking == {
        "type": "thinking",
        "thinking": FIXTURE_THINKING,
        "signature": FIXTURE_SIGNATURE,
    }
    if not signed_exact:
        raise _FixtureContractError("second_signed_thinking_mismatch")
    tool_exact = bool(
        isinstance(tool_use, dict)
        and set(tool_use) == {"type", "id", "name", "input"}
        and tool_use.get("type") == "tool_use"
        and tool_use.get("id") == FIXTURE_TOOL_CALL_ID
        and tool_use.get("name") == FIXTURE_TOOL_NAME
        and tool_use.get("input") == FIXTURE_TOOL_ARGUMENTS
    )
    if not tool_exact:
        raise _FixtureContractError("second_tool_use_identity_mismatch")
    all_tool_uses = [
        item
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "tool_use"
    ]
    if len(all_tool_uses) != 1:
        raise _FixtureContractError("second_global_tool_use_count_mismatch")
    user_content = result_user.get("content")
    if not isinstance(user_content, list) or len(user_content) != 1:
        raise _FixtureContractError("second_tool_result_content_mismatch")
    all_tool_results = [
        item
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "tool_result"
    ]
    if len(all_tool_results) != 1 or user_content[0] is not all_tool_results[0]:
        raise _FixtureContractError("second_global_tool_result_count_mismatch")
    tool_result = all_tool_results[0]
    result_exact = bool(
        set(tool_result) == {"type", "tool_use_id", "content"}
        and tool_result.get("tool_use_id") == FIXTURE_TOOL_CALL_ID
        and _tool_result_text(tool_result.get("content"))
        == ROUTE_PREFLIGHT_FIXTURE
    )
    if not result_exact:
        raise _FixtureContractError("second_tool_result_identity_mismatch")
    with state.lock:
        state.second_request_contract_exact = True
        state.signed_thinking_exact = signed_exact
        state.tool_use_identity_exact = tool_exact
        state.tool_result_identity_exact = result_exact


def _validate_abort_probe_request(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(
        payload,
        EXPECTED_ABORT_PROBE_REQUEST_FIELDS,
        "abort_probe_request_fields_mismatch",
    )
    if payload.get("model") != "hy4-preview" or payload.get("stream") is not True:
        raise _FixtureContractError("abort_probe_model_or_stream_mismatch")
    if payload.get("thinking") != {"type": "enabled", "budget_tokens": 4096}:
        raise _FixtureContractError("abort_probe_thinking_config_mismatch")
    if payload.get("max_tokens") != 65536:
        raise _FixtureContractError("abort_probe_max_tokens_mismatch")
    messages = payload.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or set(messages[0]) != {"role", "content"}
        or messages[0].get("role") != "user"
    ):
        raise _FixtureContractError("abort_probe_messages_mismatch")
    text, _ = _exact_anthropic_text_and_hash(
        messages[0].get("content"), "abort_probe_user_prompt"
    )
    if text != "SIEVE local ambiguous-timeout cancellation probe":
        raise _FixtureContractError("abort_probe_user_prompt_mismatch")


def _verify_gateway_abort_drains_litellm(
    *, fixture: _FixtureServer, transport: Any, model: Any, route: Any
) -> dict[str, Any]:
    manager = getattr(transport, "_manager", None)
    if not isinstance(manager, _TokenHubLiteLLM):
        raise ManagedTransportError("abort probe lacks the managed adapter")
    if manager._process is None or manager._tracker is None:
        raise ManagedTransportError("abort probe managed adapter is not running")
    old_process_identity = (
        manager._process.pid,
        manager._tracker.leader_birth,
    )
    old_port = manager._port
    old_proxy_key = manager._proxy_key
    probe_model = replace(
        model,
        request_timeout_seconds=ABORT_PROBE_GATEWAY_TIMEOUT_SECONDS,
        retry=replace(model.retry, max_infrastructure_retries=0),
    )
    binding = transport.bindings.for_route(route.route_profile_id)
    payload = {
        "model": probe_model.client_wire_model,
        "messages": [
            {
                "role": "user",
                "content": "SIEVE local ambiguous-timeout cancellation probe",
            }
        ],
        "stream": True,
        "max_tokens": 65536,
    }
    with ScopedModelGateway(
        route=route,
        model=probe_model,
        runtime_binding=binding,
        runtime_credential=transport.credential,
        max_requests=2,
        cooldown_gate=SharedCooldownGate(),
        managed_transport_ambiguity_probe=(
            transport.classify_outer_result
        ),
    ) as first_gateway:
        first_status = _post_gateway_probe(
            first_gateway, route.client_path, payload
        )
        first_report = first_gateway.wait_for_completion_report()
        if (
            first_status != 200
            or first_report.get("terminal_statuses")
            != {"1": "ambiguous_stream_failure"}
            or first_report.get("transport_recycle_required") is not True
        ):
            raise ManagedTransportError(
                "ambiguous-timeout cancellation probe did not reach stream boundary"
            )
        upstream_before_poisoned_followup = fixture.state.safe_record()[
            "upstream_requests"
        ]
        poisoned_status = _post_gateway_probe(
            first_gateway, route.client_path, payload
        )
        poisoned_report = first_gateway.wait_for_completion_report()
        if (
            poisoned_status != 503
            or poisoned_report.get("request_count") != 1
            or poisoned_report.get("terminal_statuses")
            != {"1": "ambiguous_stream_failure"}
            or fixture.state.safe_record()["upstream_requests"]
            != upstream_before_poisoned_followup
        ):
            raise ManagedTransportError(
                "poisoned gateway accepted a later logical request"
            )
    # Production uses the same lifecycle boundary: the poisoned episode
    # gateway is fully closed first, then the nested transport is replaced,
    # then a fresh gateway generation may send the next logical unit.
    transport.recycle_after_ambiguous()
    if manager._process is None or manager._tracker is None:
        raise ManagedTransportError("recycled managed adapter is not running")
    new_process_identity = (
        manager._process.pid,
        manager._tracker.leader_birth,
    )
    if new_process_identity == old_process_identity:
        raise ManagedTransportError("managed adapter process was not replaced")
    if manager._port != old_port or manager._proxy_key != old_proxy_key:
        raise ManagedTransportError("managed adapter local binding changed")
    receipt_path = manager.invocation_root / "transport_recycle_0001.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ManagedTransportError("managed adapter recycle receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt != {
        "schema_version": "sieve_runtime_transport_recycle_v1",
        "adapter_id": TOKENHUB_ADAPTER,
        "reason": "ambiguous_post_send_or_stream_state",
        "recycle_index": 1,
        "old_process_tree_terminated": True,
        "old_listener_released_before_restart": True,
        "provider_identity_relay_upstreams_drained": True,
        "runtime_reverified_before_restart": True,
        "same_loopback_binding_and_capability_reused": True,
        "new_process_ready": True,
        "provider_credential_endpoint_request_response_or_hidden_reasoning_recorded": False,
    }:
        raise ManagedTransportError("managed adapter recycle receipt differs")
    if not fixture.state.abort_probe_disconnect_event.wait(
        ABORT_PROBE_DRAIN_TIMEOUT_SECONDS
    ):
        raise ManagedTransportError(
            "LiteLLM did not close the cancelled upstream stream"
        )
    deadline = time.monotonic() + 1.0
    while True:
        safe = fixture.state.safe_record()
        if safe["abort_probe_no_request_overlap"]:
            break
        if time.monotonic() >= deadline:
            raise ManagedTransportError(
                "cancelled upstream stream did not reach the drain barrier"
            )
        time.sleep(0.01)
    with ScopedModelGateway(
        route=route,
        model=probe_model,
        runtime_binding=binding,
        runtime_credential=transport.credential,
        max_requests=1,
        cooldown_gate=SharedCooldownGate(),
        managed_transport_ambiguity_probe=(
            transport.classify_outer_result
        ),
    ) as followup_gateway:
        second_status = _post_gateway_probe(
            followup_gateway, route.client_path, payload
        )
        final_report = followup_gateway.wait_for_completion_report()
        if (
            second_status != 200
            or final_report.get("terminal_statuses") != {"1": "complete"}
            or final_report.get("transport_recycle_required") is not False
        ):
            raise ManagedTransportError(
                "post-cancellation follow-up request did not complete"
            )
    safe = fixture.state.safe_record()
    if not (
        safe["abort_probe_upstream_disconnected"]
        and safe["abort_probe_no_request_overlap"]
        and safe["abort_probe_followup_complete"]
    ):
        raise ManagedTransportError("ambiguous-timeout drain invariant differs")
    return {
        "outer_gateway_status": "ambiguous_stream_failure",
        "poisoned_gateway_rejected_later_request": True,
        "old_process_identity_replaced": True,
        "litellm_upstream_connection_closed": True,
        "provider_identity_relay_upstreams_drained": True,
        "old_listener_released_before_restart": True,
        "runtime_reverified_before_restart": True,
        "same_loopback_binding_and_capability_reused": True,
        "followup_started_after_drain": True,
        "request_overlap_observed": False,
        "followup_stream_complete": True,
    }


def _post_gateway_probe(
    gateway: ScopedModelGateway, path: str, payload: Mapping[str, Any]
) -> int:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection = http.client.HTTPConnection(
        LOOPBACK_HOST,
        gateway.port,
        timeout=ABORT_PROBE_DRAIN_TIMEOUT_SECONDS + 2.0,
    )
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Authorization": f"Bearer {gateway.capability_token}",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        status = int(response.status)
        response.read()
        return status
    finally:
        connection.close()


def _validate_four_tools_and_hash(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 4:
        raise _FixtureContractError("four_tool_contract_count_mismatch")
    names: list[str] = []
    for tool in value:
        if not isinstance(tool, dict):
            raise _FixtureContractError("four_tool_contract_shape_mismatch")
        name = tool.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(tool.get("description"), str)
            or not tool["description"]
            or not isinstance(tool.get("input_schema"), dict)
        ):
            raise _FixtureContractError("four_tool_contract_shape_mismatch")
        names.append(name)
    if frozenset(names) != EXPECTED_TOOL_NAMES:
        raise _FixtureContractError("four_tool_contract_names_mismatch")
    digest = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        EXPECTED_ANTHROPIC_FOUR_TOOL_SHA256
        and not secrets.compare_digest(
            digest, EXPECTED_ANTHROPIC_FOUR_TOOL_SHA256
        )
    ):
        raise _FixtureContractError("four_tool_contract_frozen_hash_mismatch")
    return digest


def _exact_anthropic_text_and_hash(value: Any, label: str) -> tuple[str, str]:
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
        and value[0].get("type") == "text"
        and isinstance(value[0].get("text"), str)
        and set(value[0]) == {"type", "text"}
    ):
        text = str(value[0]["text"])
    else:
        raise _FixtureContractError(f"{label}_shape_mismatch")
    if not text:
        raise _FixtureContractError(f"{label}_empty")
    digest = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return text, digest


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str] | set[str], code: str
) -> None:
    if set(value) != set(expected):
        raise _FixtureContractError(code)


def _tool_result_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
        and value[0].get("type") == "text"
        and isinstance(value[0].get("text"), str)
    ):
        return str(value[0]["text"])
    return None


def _first_anthropic_stream() -> bytes:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_sieve_fixture_first",
                    "type": "message",
                    "role": "assistant",
                    "model": "hy4-preview",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": FIXTURE_THINKING},
            },
        ),
        *[
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "signature_delta",
                        "signature": part,
                    },
                },
            )
            for part in FIXTURE_SIGNATURE_PARTS
        ],
        (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": FIXTURE_TOOL_CALL_ID,
                    "name": FIXTURE_TOOL_NAME,
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(
                        FIXTURE_TOOL_ARGUMENTS, separators=(",", ":")
                    ),
                },
            },
        ),
        (
            "content_block_stop",
            {"type": "content_block_stop", "index": 1},
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 20},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return _anthropic_sse(events)


def _second_anthropic_stream() -> bytes:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_sieve_fixture_second",
                    "type": "message",
                    "role": "assistant",
                    "model": "hy4-preview",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "SIEVE_PREFLIGHT_OK"},
            },
        ),
        (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return _anthropic_sse(events)


def _abort_probe_start_stream() -> bytes:
    return _anthropic_sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_sieve_abort_probe",
                        "type": "message",
                        "role": "assistant",
                        "model": "hy4-preview",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "partial"},
                },
            ),
        ]
    )


def _abort_probe_complete_stream() -> bytes:
    return _anthropic_sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_sieve_abort_probe_followup",
                        "type": "message",
                        "role": "assistant",
                        "model": "hy4-preview",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "complete"},
                },
            ),
            (
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


def _anthropic_sse(events: list[tuple[str, Mapping[str, Any]]]) -> bytes:
    return (
        "".join(
            "event: "
            + event
            + "\ndata: "
            + json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            + "\n\n"
            for event, value in events
        )
    ).encode("utf-8")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


class _temporary_directory:
    def __init__(self, prefix: str) -> None:
        import tempfile

        self.path = Path(tempfile.mkdtemp(prefix=prefix))

    def __enter__(self) -> Path:
        self.path.chmod(0o700)
        return self.path

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.path.exists() and not self.path.is_symlink():
            shutil.rmtree(self.path)


__all__ = ["verify_tokenhub_pi_litellm_roundtrip"]
