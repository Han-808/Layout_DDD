"""One-request HTTP transport with explicit delivery ambiguity."""

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
    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    kwargs = {"host": parsed.hostname, "port": parsed.port, "timeout": connect_timeout}
    if parsed.scheme == "https":
        kwargs["context"] = ssl.create_default_context()
    connection = connection_cls(**kwargs)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return connection, path


def _network_error(
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
    request_headers: dict[str, str],
) -> TransportResult:
    connection, path = _connection(endpoint, connect_timeout)
    started = time.monotonic()
    try:
        try:
            connection.connect()
        except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            return _network_error(
                started=started,
                status="transport_failure",
                stage="connect",
                exc=exc,
            )
        if connection.sock is None:
            return _network_error(
                started=started,
                status="transport_failure",
                stage="connect",
                exc=ConnectionError("connection established without a socket"),
            )
        connection.sock.settimeout(read_timeout)
        try:
            connection.request("POST", path, body=request_body, headers=request_headers)
        except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            return _network_error(
                started=started,
                status="transport_ambiguous",
                stage="send_request",
                exc=exc,
            )
        try:
            response = connection.getresponse()
        except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            return _network_error(
                started=started,
                status="transport_ambiguous",
                stage="wait_response",
                exc=exc,
            )
        headers = tuple(response.getheaders())
        try:
            body = response.read()
        except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            return TransportResult(
                status="transport_ambiguous",
                elapsed_seconds=time.monotonic() - started,
                stage="read_response",
                http_status=response.status,
                http_reason=response.reason,
                response_headers=headers,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        return TransportResult(
            status="response",
            elapsed_seconds=time.monotonic() - started,
            stage="complete",
            http_status=response.status,
            http_reason=response.reason,
            response_headers=headers,
            response_body=body,
        )
    finally:
        connection.close()

