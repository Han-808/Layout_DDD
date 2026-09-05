"""One-attempt HTTP transport with explicit delivery-ambiguity semantics."""

from __future__ import annotations

import http.client
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit


TransportStatus = Literal["response", "transport_failure", "transport_ambiguous"]


@dataclass(frozen=True)
class TransportResult:
    status: TransportStatus
    elapsed_seconds: float
    stage: str
    http_status: int | None = None
    http_reason: str | None = None
    response_headers: tuple[tuple[str, str], ...] | None = None
    response_body: bytes | None = None
    error_type: str | None = None
    error_message: str | None = None


class EndpointError(ValueError):
    pass


def _connection(endpoint: str, connect_timeout: float):
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise EndpointError("endpoint scheme must be http or https")
    if not parsed.hostname:
        raise EndpointError("endpoint must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise EndpointError("credentials in endpoint URLs are not allowed")
    if parsed.fragment:
        raise EndpointError("endpoint fragments are not allowed")

    port = parsed.port
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(
            parsed.hostname,
            port=port,
            timeout=connect_timeout,
            context=ssl.create_default_context(),
        )
    else:
        conn = http.client.HTTPConnection(
            parsed.hostname,
            port=port,
            timeout=connect_timeout,
        )

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return conn, path


def _network_error_result(
    *,
    started: float,
    status: Literal["transport_failure", "transport_ambiguous"],
    stage: str,
    exc: BaseException,
) -> TransportResult:
    return TransportResult(
        status=status,
        elapsed_seconds=time.monotonic() - started,
        stage=stage,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def post_once(
    endpoint: str,
    request_body: bytes,
    *,
    connect_timeout: float,
    read_timeout: float,
    request_headers: dict[str, str] | None = None,
) -> TransportResult:
    """POST exactly once; never retry and never resend after ambiguity."""
    conn, path = _connection(endpoint, connect_timeout)
    started = time.monotonic()

    try:
        try:
            conn.connect()
        except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            # No HTTP request bytes have been deliberately sent at this point.
            return _network_error_result(
                started=started,
                status="transport_failure",
                stage="connect",
                exc=exc,
            )

        if conn.sock is None:
            return _network_error_result(
                started=started,
                status="transport_failure",
                stage="connect",
                exc=ConnectionError("connection established without a socket"),
            )
        conn.sock.settimeout(read_timeout)

        try:
            # From this line onward, partial or complete request delivery is possible.
            headers = request_headers or {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "User-Agent": "sceneeval-hy4-online-capture-runner/0.5.0",
            }
            conn.request(
                "POST",
                path,
                body=request_body,
                headers=headers,
            )
        except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            return _network_error_result(
                started=started,
                status="transport_ambiguous",
                stage="send_request",
                exc=exc,
            )

        try:
            response = conn.getresponse()
        except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            return _network_error_result(
                started=started,
                status="transport_ambiguous",
                stage="wait_response",
                exc=exc,
            )

        response_headers = tuple(response.getheaders())
        try:
            response_body = response.read()
        except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            return TransportResult(
                status="transport_ambiguous",
                elapsed_seconds=time.monotonic() - started,
                stage="read_response",
                http_status=response.status,
                http_reason=response.reason,
                response_headers=response_headers,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        return TransportResult(
            status="response",
            elapsed_seconds=time.monotonic() - started,
            stage="complete",
            http_status=response.status,
            http_reason=response.reason,
            response_headers=response_headers,
            response_body=response_body,
        )
    finally:
        conn.close()
