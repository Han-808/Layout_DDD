from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from unittest import mock


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
if str(TRUSTED) not in sys.path:
    sys.path.insert(0, str(TRUSTED))

from tokenhub_identity_relay import (  # noqa: E402
    TokenHubIdentityRelay,
    TokenHubIdentityRelayError,
)


class _RawAnthropicFixture:
    def __init__(self) -> None:
        self.response_model = "hy4-preview"
        self.seen_provider_credential = False
        self.seen_relay_capability = False
        self.request_count = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_POST(self) -> None:  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                payload = json.loads(body.decode("utf-8"))
                owner.request_count += 1
                owner.seen_provider_credential = (
                    self.headers.get("x-api-key") == "fixture-provider-secret"
                )
                owner.seen_relay_capability = any(
                    "sieve-tokenhub-relay-" in value
                    for value in self.headers.values()
                ) or b"sieve-tokenhub-relay-" in body
                if (
                    self.path != "/v1/messages"
                    or payload.get("model") != "hy4-preview"
                    or payload.get("stream") is not True
                ):
                    self.send_response(400)
                    self.end_headers()
                    return
                stream = (
                    "event: message_start\n"
                    "data: "
                    + json.dumps(
                        {
                            "type": "message_start",
                            "message": {"model": owner.response_model},
                        },
                        separators=(",", ":"),
                    )
                    + "\n\n"
                    "event: message_stop\n"
                    'data: {"type":"message_stop"}\n\n'
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(stream)))
                self.end_headers()
                self.wfile.write(stream)

            def log_message(self, format: str, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> "_RawAnthropicFixture":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class TokenHubIdentityRelayTests(unittest.TestCase):
    def test_thread_start_failure_revokes_and_closes_without_shutdown_hang(self) -> None:
        relay = TokenHubIdentityRelay(
            provider_base_url="http://127.0.0.1:9",
            provider_credential="fixture-provider-secret",
            expected_request_model="hy4-preview",
            accepted_response_models=("hy4-preview",),
            request_timeout_seconds=30,
            allow_insecure_provider=True,
        )
        port = relay.port
        started = time.monotonic()
        with mock.patch.object(
            relay._thread, "start", side_effect=RuntimeError("fixture failure")
        ):
            with self.assertRaises(TokenHubIdentityRelayError) as caught:
                relay.start()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertNotIn("fixture-provider-secret", str(caught.exception))
        relay.close()
        with self.assertRaises(TokenHubIdentityRelayError):
            _ = relay.capability
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(0.25)
            self.assertNotEqual(client.connect_ex(("127.0.0.1", port)), 0)

    def test_raw_provider_identity_mismatch_fails_before_success_stream(self) -> None:
        with _RawAnthropicFixture() as fixture:
            relay = TokenHubIdentityRelay(
                provider_base_url=fixture.base_url,
                provider_credential="fixture-provider-secret",
                expected_request_model="hy4-preview",
                accepted_response_models=("hy4-preview",),
                request_timeout_seconds=30,
                allow_insecure_provider=True,
            )
            port = relay.port
            relay.start()
            try:
                fixture.response_model = "wrong-provider-model"
                status, mismatch_body = self._request(relay)
                self.assertEqual(status, 502)
                self.assertIn(b"provider_model_identity_mismatch", mismatch_body)

                fixture.response_model = "hy4-preview"
                status, poisoned_body = self._request(relay)
                self.assertEqual(status, 503)
                self.assertIn(
                    b"provider_transport_recovery_required", poisoned_body
                )
                self.assertEqual(fixture.request_count, 1)
                relay.drain_ambiguous()
                status, valid_body = self._request(relay)
                self.assertEqual(status, 200)
                self.assertIn(b'"model":"hy4-preview"', valid_body)
                self.assertFalse(relay.classify_outer_result(True))
                record = relay.public_record()
                self.assertEqual(record["verified_response_count"], 1)
                self.assertEqual(record["rejected_identity_count"], 1)
                self.assertEqual(record["provider_2xx_response_count"], 2)
                self.assertEqual(record["provider_2xx_acknowledged_count"], 1)
                self.assertEqual(record["provider_2xx_pending_outer_result"], 0)
                self.assertFalse(record["provider_credential_forwarded_to_litellm"])
                self.assertNotIn(
                    "fixture-provider-secret", json.dumps(record, sort_keys=True)
                )
                self.assertTrue(fixture.seen_provider_credential)
                self.assertFalse(fixture.seen_relay_capability)
            finally:
                relay.close()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(0.25)
            self.assertNotEqual(client.connect_ex(("127.0.0.1", port)), 0)

    def test_provider_2xx_then_outer_failure_is_terminal_and_poisoned(self) -> None:
        with _RawAnthropicFixture() as fixture:
            relay = TokenHubIdentityRelay(
                provider_base_url=fixture.base_url,
                provider_credential="fixture-provider-secret",
                expected_request_model="hy4-preview",
                accepted_response_models=("hy4-preview",),
                request_timeout_seconds=30,
                allow_insecure_provider=True,
            )
            relay.start()
            try:
                status, _ = self._request(relay)
                self.assertEqual(status, 200)
                self.assertTrue(relay.classify_outer_result(False))
                self.assertTrue(relay.post_send_ambiguity_detected())
                status, _ = self._request(relay)
                self.assertEqual(status, 503)
                self.assertEqual(fixture.request_count, 1)
                relay.drain_ambiguous()
                status, _ = self._request(relay)
                self.assertEqual(status, 200)
                self.assertFalse(relay.classify_outer_result(True))
                record = relay.public_record()
                self.assertEqual(record["provider_2xx_response_count"], 2)
                self.assertEqual(record["provider_2xx_acknowledged_count"], 1)
                self.assertEqual(record["post_send_ambiguity_count"], 1)
            finally:
                relay.close()

    def test_outer_success_without_raw_provider_2xx_fails_closed(self) -> None:
        relay = TokenHubIdentityRelay(
            provider_base_url="http://127.0.0.1:9",
            provider_credential="fixture-provider-secret",
            expected_request_model="hy4-preview",
            accepted_response_models=("hy4-preview",),
            request_timeout_seconds=30,
            allow_insecure_provider=True,
        )
        relay.start()
        try:
            self.assertTrue(relay.classify_outer_result(True))
            self.assertTrue(relay.post_send_ambiguity_detected())
        finally:
            relay.close()

    @staticmethod
    def _request(relay: TokenHubIdentityRelay) -> tuple[int, bytes]:
        payload = json.dumps(
            {
                "model": "hy4-preview",
                "messages": [{"role": "user", "content": "fixture"}],
                "stream": True,
                "max_tokens": 128,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=5)
        try:
            connection.request(
                "POST",
                "/v1/messages",
                body=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": relay.capability,
                },
            )
            response = connection.getresponse()
            return int(response.status), response.read()
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
