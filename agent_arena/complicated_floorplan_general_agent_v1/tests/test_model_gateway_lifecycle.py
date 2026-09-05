from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
if str(TRUSTED) not in sys.path:
    sys.path.insert(0, str(TRUSTED))

from api_profiles import ProfileRegistry, RouteRuntimeBinding  # noqa: E402
from model_gateway import (  # noqa: E402
    GatewayError,
    ScopedModelGateway,
    SharedCooldownGate,
)


class _Upstream:
    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_Upstream":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"


class ModelGatewayLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProfileRegistry.load()
        cls.base_model = cls.registry.model("tokenhub-hy4-preview-agent-v1")
        cls.route = cls.registry.route(cls.base_model.route_profile_id)

    def test_close_cancels_shared_cooldown_and_scrubs_secrets(self) -> None:
        gate = SharedCooldownGate()
        gate.trigger(60)
        with tempfile.TemporaryDirectory(prefix="gateway-close-cooldown-") as temporary:
            events = Path(temporary) / "events.jsonl"
            gateway = self._gateway(
                upstream_base_url="http://127.0.0.1:9/v1",
                event_path=events,
                cooldown_gate=gate,
            ).start()
            errors: list[BaseException] = []
            client = threading.Thread(
                target=lambda: self._request_catching(gateway, errors), daemon=True
            )
            client.start()
            self._wait_until(lambda: gateway.request_count == 1)
            started = time.monotonic()
            gateway.close()
            self.assertLess(time.monotonic() - started, 3.0)
            client.join(timeout=3)
            self.assertFalse(client.is_alive())
            self.assertEqual(gateway.runtime_credential, "")
            self.assertEqual(gateway.capability_token, "")
            self.assertEqual(gateway.session_id, "")
            self.assertEqual(events.stat().st_mode & 0o777, 0o444)
            gateway.close()

    def test_thread_start_failure_closes_listener_and_revokes_capabilities(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gateway-start-failure-") as temporary:
            events = Path(temporary) / "events.jsonl"
            gateway = self._gateway(
                upstream_base_url="http://127.0.0.1:9/v1",
                event_path=events,
            )
            port = gateway.port
            started = time.monotonic()
            with mock.patch.object(
                gateway._thread,
                "start",
                side_effect=RuntimeError("fixture start failure"),
            ):
                with self.assertRaisesRegex(
                    GatewayError, "model_gateway_thread_start_failed"
                ) as caught:
                    gateway.start()
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertNotIn(
                "fixture-secret-that-must-be-scrubbed", str(caught.exception)
            )
            self.assertEqual(gateway.runtime_credential, "")
            self.assertEqual(gateway.capability_token, "")
            self.assertEqual(gateway.session_id, "")
            self.assertEqual(events.stat().st_mode & 0o777, 0o444)
            gateway.close()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(0.25)
                self.assertNotEqual(client.connect_ex(("127.0.0.1", port)), 0)

    def test_close_interrupts_post_send_getresponse_wait(self) -> None:
        request_arrived = threading.Event()
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                self.rfile.read(int(self.headers["Content-Length"]))
                request_arrived.set()
                release.wait(timeout=10)

            def log_message(self, format: str, *args: object) -> None:
                return

        with _Upstream(Handler) as upstream:
            gateway = self._gateway(upstream_base_url=upstream.base_url).start()
            errors: list[BaseException] = []
            client = threading.Thread(
                target=lambda: self._request_catching(gateway, errors), daemon=True
            )
            client.start()
            self.assertTrue(request_arrived.wait(timeout=3))
            started = time.monotonic()
            try:
                gateway.close()
            finally:
                release.set()
            self.assertLess(time.monotonic() - started, 3.0)
            client.join(timeout=3)
            self.assertFalse(client.is_alive())

    def test_close_interrupts_downstream_client_that_does_not_read(self) -> None:
        streaming = threading.Event()
        upstream_stopped = threading.Event()
        event = (
            'data: {"model":"hy4-preview","choices":[{"delta":{"content":"'
            + ("x" * 32768)
            + '"}}]}\n\n'
        ).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for index in range(4096):
                        self.wfile.write(event)
                        self.wfile.flush()
                        if index == 3:
                            streaming.set()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    upstream_stopped.set()

            def log_message(self, format: str, *args: object) -> None:
                return

        with _Upstream(Handler) as upstream:
            gateway = self._gateway(upstream_base_url=upstream.base_url).start()
            client_socket = socket.create_connection(("127.0.0.1", gateway.port), 3)
            client_socket.settimeout(3)
            payload = json.dumps(self._payload()).encode("utf-8")
            request = (
                f"POST {self.route.client_path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{gateway.port}\r\n"
                f"Authorization: Bearer {gateway.capability_token}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii") + payload
            client_socket.sendall(request)
            self.assertTrue(streaming.wait(timeout=3))
            time.sleep(0.15)
            started = time.monotonic()
            try:
                gateway.close()
            finally:
                client_socket.close()
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertTrue(upstream_stopped.wait(timeout=3))

    def test_close_before_start_is_safe_and_start_after_close_is_forbidden(self) -> None:
        gateway = self._gateway(upstream_base_url="http://127.0.0.1:9/v1")
        gateway.close()
        gateway.close()
        self.assertEqual(gateway.runtime_credential, "")
        with self.assertRaisesRegex(GatewayError, "lifecycle"):
            gateway.start()

    def _gateway(
        self,
        *,
        upstream_base_url: str,
        event_path: Path | None = None,
        cooldown_gate: SharedCooldownGate | None = None,
    ) -> ScopedModelGateway:
        model = replace(
            self.base_model,
            request_timeout_seconds=60,
            retry=replace(
                self.base_model.retry,
                max_infrastructure_retries=0,
                retry_delay_seconds=60,
            ),
        )
        return ScopedModelGateway(
            route=self.route,
            model=model,
            runtime_binding=RouteRuntimeBinding(
                route_profile_id=self.route.route_profile_id,
                binding_profile_id="fixture-lifecycle-binding-v1",
                upstream_base_url=upstream_base_url,
                allow_insecure_upstream=True,
                managed_adapter_id="managed_tokenhub_litellm_v1",
            ),
            runtime_credential="fixture-secret-that-must-be-scrubbed",
            max_requests=1,
            event_path=event_path,
            cooldown_gate=cooldown_gate,
            managed_transport_ambiguity_probe=lambda _complete: False,
        )

    def _request_catching(
        self, gateway: ScopedModelGateway, errors: list[BaseException]
    ) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", gateway.port, timeout=5)
        try:
            connection.request(
                "POST",
                self.route.client_path,
                body=json.dumps(self._payload()),
                headers={
                    "Authorization": f"Bearer {gateway.capability_token}",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            response.read()
        except BaseException as exc:
            errors.append(exc)
        finally:
            connection.close()

    def _payload(self) -> dict[str, object]:
        return {
            "model": self.base_model.client_wire_model,
            "stream": True,
            "messages": [{"role": "user", "content": "fixture"}],
        }

    @staticmethod
    def _wait_until(predicate: object, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():  # type: ignore[operator]
                return
            time.sleep(0.01)
        raise AssertionError("condition did not become true")


if __name__ == "__main__":
    unittest.main()
