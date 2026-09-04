#!/usr/bin/env python3
"""A per-episode loopback proxy that keeps upstream credentials host-side."""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import secrets
import threading
from typing import Any
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 16 * 1024 * 1024


class GatewayError(RuntimeError):
    pass


class ScopedModelGateway:
    """Expose one path/model/request budget through a loopback capability."""

    def __init__(
        self,
        *,
        upstream_base_url: str,
        upstream_secret: str,
        fixed_model: str,
        endpoint: str,
        max_requests: int,
        upstream_auth_header: str = "Authorization",
        upstream_auth_prefix: str = "Bearer ",
        upstream_timeout_seconds: float = 600.0,
        allow_insecure_loopback_upstream: bool = False,
    ) -> None:
        upstream = urlsplit(upstream_base_url.rstrip("/"))
        if upstream.scheme not in {"https", "http"} or not upstream.hostname:
            raise GatewayError("upstream_base_url must be HTTP(S)")
        if upstream.scheme != "https" and not (
            allow_insecure_loopback_upstream
            and upstream.hostname in {"127.0.0.1", "localhost"}
        ):
            raise GatewayError("production upstream must use HTTPS")
        if not isinstance(upstream_secret, str) or not upstream_secret:
            raise GatewayError("upstream secret is missing")
        if not isinstance(fixed_model, str) or not fixed_model:
            raise GatewayError("fixed model is missing")
        if not endpoint.startswith("/") or "?" in endpoint or ".." in endpoint:
            raise GatewayError("gateway endpoint is invalid")
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
            raise GatewayError("max_requests must be positive")
        if (
            isinstance(upstream_timeout_seconds, bool)
            or not isinstance(upstream_timeout_seconds, (int, float))
            or not math.isfinite(float(upstream_timeout_seconds))
            or not 0 < float(upstream_timeout_seconds) <= 7200
        ):
            raise GatewayError("upstream_timeout_seconds must be in (0, 7200]")
        self.upstream = upstream
        self.upstream_secret = upstream_secret
        self.fixed_model = fixed_model
        self.endpoint = endpoint
        self.max_requests = max_requests
        self.upstream_auth_header = upstream_auth_header
        self.upstream_auth_prefix = upstream_auth_prefix
        self.upstream_timeout_seconds = float(upstream_timeout_seconds)
        self.capability_token = secrets.token_hex(32)
        self._request_count = 0
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_POST(self) -> None:  # noqa: N802
                owner._handle(self)

            def do_GET(self) -> None:  # noqa: N802
                owner._reject(self, 405, "method_not_allowed")

            def do_PUT(self) -> None:  # noqa: N802
                owner._reject(self, 405, "method_not_allowed")

            def do_DELETE(self) -> None:  # noqa: N802
                owner._reject(self, 405, "method_not_allowed")

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
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
        # Seatbelt's remote-tcp filter requires the special host spelling
        # `localhost`; the HTTP listener itself remains bound to 127.0.0.1.
        return f"localhost:{self.port}"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    def start(self) -> "ScopedModelGateway":
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        self.upstream_secret = ""
        self.capability_token = ""

    def __enter__(self) -> "ScopedModelGateway":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        authorization = handler.headers.get("Authorization", "")
        expected = f"Bearer {self.capability_token}"
        if not secrets.compare_digest(authorization, expected):
            self._reject(handler, 401, "invalid_episode_capability")
            return
        if handler.path != self.endpoint:
            self._reject(handler, 404, "endpoint_not_allowed")
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
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._reject(handler, 400, "invalid_json")
            return
        if not isinstance(payload, dict) or payload.get("model") != self.fixed_model:
            self._reject(handler, 400, "model_identity_mismatch")
            return
        with self._lock:
            if self._request_count >= self.max_requests:
                self._reject(handler, 429, "episode_request_budget_exhausted")
                return
            self._request_count += 1
        try:
            self._forward(handler, body)
        except Exception:
            self._reject(handler, 502, "upstream_transport_failure")

    def _forward(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        connection_cls = (
            http.client.HTTPSConnection
            if self.upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        port = self.upstream.port or (443 if self.upstream.scheme == "https" else 80)
        connection = connection_cls(
            self.upstream.hostname,
            port,
            timeout=self.upstream_timeout_seconds,
        )
        base_path = self.upstream.path.rstrip("/")
        upstream_path = base_path + self.endpoint
        headers = {
            "Accept": handler.headers.get("Accept", "text/event-stream"),
            "Content-Type": "application/json",
            self.upstream_auth_header: self.upstream_auth_prefix + self.upstream_secret,
        }
        connection.request("POST", upstream_path, body=body, headers=headers)
        response = connection.getresponse()
        handler.send_response(response.status)
        for name in ("Content-Type", "Cache-Control", "Content-Encoding"):
            value = response.getheader(name)
            if value:
                handler.send_header(name, value)
        handler.end_headers()
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)
            handler.wfile.flush()
        connection.close()

    @staticmethod
    def _reject(handler: BaseHTTPRequestHandler, status: int, code: str) -> None:
        body = json.dumps({"error": {"code": code}}, sort_keys=True).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)


__all__ = ["GatewayError", "ScopedModelGateway"]
