from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import uuid
from urllib.parse import parse_qs


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
if str(TRUSTED) not in sys.path:
    sys.path.insert(0, str(TRUSTED))

from api_profiles import ProfileRegistry, RouteRuntimeBinding  # noqa: E402
from model_gateway import (  # noqa: E402
    ANTHROPIC_REASONING_DETAIL_FORMAT,
    GatewayError,
    ScopedModelGateway,
    SharedCooldownGate,
    _AnthropicThinkingSseBridge,
    _reasoning_replay_state,
    _replace_system_prompt,
    _restore_anthropic_signed_reasoning,
    _system_prompt_sha256,
    verify_gateway_audit,
)
from preflight_agent_route import run_tool_call_preflight  # noqa: E402


def _chat_sse(*events: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(item, separators=(',', ':'))}" for item in events]
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


def _responses_sse(model: str, *events: dict[str, object]) -> bytes:
    values = list(events)
    values.append(
        {
            "type": "response.completed",
            "response": {"model": model, "status": "completed"},
        }
    )
    return (
        "\n\n".join(
            f"data: {json.dumps(item, separators=(',', ':'))}" for item in values
        )
        + "\n\n"
    ).encode("utf-8")


class _Server:
    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"


class GatewayAndPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProfileRegistry.load()

    def test_chat_tool_roundtrip_replays_reasoning_without_recording_it(self) -> None:
        observed: list[dict[str, object]] = []
        headers: list[str | None] = []
        private_reasoning = "fixture-private-reasoning-never-persist"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers["Content-Length"])
                observed.append(json.loads(self.rfile.read(length)))
                headers.append(self.headers.get("Authorization"))
                if len(observed) == 1:
                    body = _chat_sse(
                        {
                            "model": "hy4-preview",
                            "choices": [
                                {
                                    "delta": {
                                        "reasoning_content": private_reasoning,
                                        "thinking_blocks": [
                                            {
                                                "type": "thinking",
                                                "thinking": private_reasoning,
                                                "signature": "fixture-private-signature",
                                            }
                                        ],
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_fixture",
                                                "type": "function",
                                                "function": {
                                                    "name": "sieve_preflight_echo",
                                                    "arguments": '{"nonce":"sieve-preflight-v1"}',
                                                },
                                            }
                                        ],
                                    }
                                }
                            ],
                        }
                    )
                else:
                    body = _chat_sse(
                        {
                            "model": "hy4-preview",
                            "choices": [{"delta": {"content": "Tool result accepted."}}],
                        }
                    )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        model = self.registry.model("tokenhub-hy4-preview-agent-v1")
        route = self.registry.route(model.route_profile_id)
        with tempfile.TemporaryDirectory(prefix="preflight-chat-") as temporary:
            events = Path(temporary) / "events.jsonl"
            with _Server(Handler) as upstream:
                binding = RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-tokenhub-binding-v1",
                    upstream_base_url=f"{upstream.base_url}/v1",
                    allow_insecure_upstream=True,
                    managed_adapter_id="managed_tokenhub_litellm_v1",
                )
                with ScopedModelGateway(
                    route=route,
                    model=model,
                    runtime_binding=binding,
                    runtime_credential="fixture-tokenhub-key",
                    max_requests=2,
                    event_path=events,
                    managed_transport_ambiguity_probe=lambda _complete: False,
                ) as gateway:
                    report = run_tool_call_preflight(
                        gateway=gateway, route=route, model=model
                    )
                    completion = gateway.wait_for_completion_report()
            verification = verify_gateway_audit(events, completion)
            self.assertTrue(report.ok, report.public_dict())
            self.assertTrue(verification["all_logical_requests_complete"])
            self.assertEqual(events.stat().st_mode & 0o777, 0o444)
            self.assertTrue(report.reasoning_replayed)
            self.assertTrue(report.reasoning_signal_present)
            self.assertTrue(report.response_identity_matches)
            self.assertEqual(len(observed), 2)
            self.assertEqual(headers, ["Bearer fixture-tokenhub-key"] * 2)
            self.assertEqual(observed[0]["reasoning_effort"], "high")
            assistant = observed[1]["messages"][1]
            self.assertEqual(
                assistant["thinking_blocks"],
                [
                    {
                        "type": "thinking",
                        "thinking": private_reasoning,
                        "signature": "fixture-private-signature",
                    }
                ],
            )
            self.assertNotIn("reasoning_content", assistant)
            self.assertNotIn("reasoning_details", assistant)
            public = json.dumps(report.public_dict(), sort_keys=True)
            event_text = events.read_text(encoding="utf-8")
            self.assertNotIn(private_reasoning, public)
            self.assertNotIn(private_reasoning, event_text)
            self.assertNotIn("fixture-tokenhub-key", public + event_text)

    def test_anthropic_signed_thinking_bridge_is_fragment_safe_and_roundtrips(self) -> None:
        private_thinking = "fixture thinking that must stay private"
        signature_part_1 = "opaque-fixture-"
        signature_part_2 = "signature"
        private_signature = signature_part_1 + signature_part_2
        events = _chat_sse(
            {
                "model": "hy4-preview",
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": private_thinking,
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": private_thinking,
                                    "signature": "",
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "model": "hy4-preview",
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "",
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": signature_part_1,
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "model": "hy4-preview",
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "",
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": signature_part_2,
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "model": "hy4-preview",
                "choices": [{"delta": {"tool_calls": []}}],
            },
            {
                "model": "hy4-preview",
                "choices": [{"delta": {"tool_calls": []}}],
            },
        )
        bridge = _AnthropicThinkingSseBridge()
        translated = bytearray()
        for offset in range(0, len(events), 7):
            translated.extend(bridge.feed(events[offset : offset + 7]))
        translated.extend(bridge.finish())
        details: list[dict[str, object]] = []
        for raw_line in bytes(translated).splitlines():
            if not raw_line.startswith(b"data:") or raw_line.endswith(b"[DONE]"):
                continue
            event = json.loads(raw_line[5:])
            delta = event["choices"][0]["delta"]
            details.extend(delta.get("reasoning_details", []))
        self.assertEqual(
            details,
            [
                {
                    "type": "reasoning.text",
                    "text": private_thinking,
                    "signature": private_signature,
                    "format": ANTHROPIC_REASONING_DETAIL_FORMAT,
                },
            ],
        )
        replay = _restore_anthropic_signed_reasoning(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call_fixture"}],
                        "reasoning_content": "",
                        "reasoning_details": [
                            {
                                "type": "reasoning.text",
                                "text": private_thinking,
                                "signature": private_signature,
                                "format": ANTHROPIC_REASONING_DETAIL_FORMAT,
                            }
                        ],
                    }
                ]
            }
        )
        assistant = replay["messages"][0]
        self.assertEqual(
            assistant["thinking_blocks"],
            [
                {
                    "type": "thinking",
                    "thinking": private_thinking,
                    "signature": private_signature,
                }
            ],
        )
        self.assertNotIn("reasoning_details", assistant)
        self.assertNotIn("reasoning_content", assistant)

    def test_anthropic_reasoning_bridge_rejects_a_second_normal_block(self) -> None:
        events = _chat_sse(
            {
                "choices": [
                    {
                        "delta": {
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "first",
                                    "signature": "signature-one",
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "second",
                                    "signature": "signature-two",
                                }
                            ]
                        }
                    }
                ]
            },
        )
        bridge = _AnthropicThinkingSseBridge()
        with self.assertRaisesRegex(GatewayError, "multiple_normal_blocks"):
            bridge.feed(events)

    def test_anthropic_reasoning_bridge_bounds_private_replay_data(self) -> None:
        events = _chat_sse(
            {
                "choices": [
                    {
                        "delta": {
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "private-too-large",
                                    "signature": "signature",
                                }
                            ]
                        }
                    }
                ]
            }
        )
        with mock.patch("model_gateway.MAX_REASONING_BRIDGE_BYTES", 8):
            with self.assertRaisesRegex(GatewayError, "private_data_too_large"):
                _AnthropicThinkingSseBridge().feed(events)

    def test_anthropic_reasoning_bridge_rejects_unsigned_or_forged_replay(self) -> None:
        base = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_fixture"}],
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "private fixture",
                            "signature": "",
                            "format": ANTHROPIC_REASONING_DETAIL_FORMAT,
                        }
                    ],
                }
            ]
        }
        with self.assertRaisesRegex(GatewayError, "signed_reasoning"):
            _restore_anthropic_signed_reasoning(base)
        forged = json.loads(json.dumps(base))
        forged["messages"][0]["reasoning_details"][0]["signature"] = "opaque"
        forged["messages"][0]["reasoning_details"][0]["format"] = "other"
        with self.assertRaisesRegex(GatewayError, "signed_reasoning"):
            _restore_anthropic_signed_reasoning(forged)

    def test_responses_tool_roundtrip_replays_reasoning_item(self) -> None:
        observed: list[dict[str, object]] = []
        private_encrypted = "fixture-encrypted-reasoning"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers["Content-Length"])
                observed.append(json.loads(self.rfile.read(length)))
                if len(observed) == 1:
                    body = _responses_sse(
                        "glm-5.3",
                        {
                            "type": "response.created",
                            "response": {"model": "glm-5.3"},
                        },
                        {
                            "type": "response.output_item.added",
                            "item": {
                                "type": "reasoning",
                                "id": "rs_fixture",
                                "encrypted_content": private_encrypted,
                                "summary": [],
                            },
                        },
                        {
                            "type": "response.output_item.added",
                            "item": {
                                "type": "function_call",
                                "call_id": "call_fixture",
                                "name": "sieve_preflight_echo",
                                "arguments": '{"nonce":"sieve-preflight-v1"}',
                            },
                        },
                    )
                else:
                    body = _responses_sse(
                        "glm-5.3",
                        {
                            "type": "response.created",
                            "response": {"model": "glm-5.3"},
                        },
                        {
                            "type": "response.output_text.delta",
                            "delta": "Tool result accepted.",
                        },
                    )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        model = self.registry.model("api2-glm-5-3-agent-v1")
        route = self.registry.route(model.route_profile_id)
        with _Server(Handler) as upstream:
            binding = RouteRuntimeBinding(
                route_profile_id=route.route_profile_id,
                binding_profile_id="fixture-api2-responses-binding-v1",
                upstream_base_url=f"{upstream.base_url}/openapi/v2",
                allow_insecure_upstream=True,
            )
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=binding,
                runtime_credential="fixture-app:fixture-key",
                max_requests=2,
            ) as gateway:
                report = run_tool_call_preflight(
                    gateway=gateway, route=route, model=model
                )
        self.assertTrue(report.ok, report.public_dict())
        self.assertTrue(report.reasoning_replayed)
        self.assertEqual(observed[0]["reasoning"], {"effort": "max"})
        for request in observed:
            self.assertNotIn("prompt_cache_key", request)
            self.assertNotIn("prompt_cache_retention", request)
            self.assertNotIn("prompt_cache_options", request)
        reasoning_items = [
            item for item in observed[1]["input"] if item.get("type") == "reasoning"
        ]
        self.assertEqual(reasoning_items[0]["encrypted_content"], private_encrypted)
        self.assertNotIn(private_encrypted, json.dumps(report.public_dict()))

    def test_explicit_429_is_retried_but_is_one_logical_request(self) -> None:
        calls = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                nonlocal calls
                calls += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                if calls == 1:
                    body = b"discarded upstream error body"
                    self.send_response(429)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = _chat_sse(
                    {
                        "model": "api_azure_openai_gpt-5.6-sol",
                        "choices": [{"delta": {"content": "ok"}}],
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        original = self.registry.model("api2-gpt-5-6-sol-agent-v1")
        model = replace(
            original,
            retry=replace(
                original.retry,
                max_infrastructure_retries=1,
                retry_delay_seconds=0,
            ),
        )
        route = self.registry.route(model.route_profile_id)
        with _Server(Handler) as upstream:
            binding = RouteRuntimeBinding(
                route_profile_id=route.route_profile_id,
                binding_profile_id="fixture-api2-chat-binding-v1",
                upstream_base_url=f"{upstream.base_url}/v1",
                allow_insecure_upstream=True,
            )
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=binding,
                runtime_credential="fixture-app:fixture-key",
                max_requests=1,
            ) as gateway:
                status, body = _gateway_request(
                    gateway,
                    route.client_path,
                    _minimal_chat_payload(model),
                )
                self.assertEqual(gateway.request_count, 1)
        self.assertEqual(status, 200)
        self.assertIn("data:", body)
        self.assertEqual(calls, 2)

    def test_post_send_disconnect_is_ambiguous_and_not_retried(self) -> None:
        calls = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                nonlocal calls
                calls += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                self.close_connection = True

            def log_message(self, format: str, *args: object) -> None:
                return

        original = self.registry.model("api2-gpt-5-6-sol-agent-v1")
        model = replace(
            original,
            retry=replace(original.retry, retry_delay_seconds=0),
        )
        route = self.registry.route(model.route_profile_id)
        with tempfile.TemporaryDirectory(prefix="gateway-ambiguous-") as temporary:
            events = Path(temporary) / "events.jsonl"
            with _Server(Handler) as upstream:
                binding = RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-api2-chat-binding-v1",
                    upstream_base_url=f"{upstream.base_url}/v1",
                    allow_insecure_upstream=True,
                )
                with ScopedModelGateway(
                    route=route,
                    model=model,
                    runtime_binding=binding,
                    runtime_credential="fixture-app:fixture-key",
                    max_requests=2,
                    event_path=events,
                ) as gateway:
                    status, _ = _gateway_request(
                        gateway,
                        route.client_path,
                        _minimal_chat_payload(model),
                    )
                    poisoned_status, poisoned_body = _gateway_request(
                        gateway,
                        route.client_path,
                        _minimal_chat_payload(model),
                    )
                    completion = gateway.wait_for_completion_report()
            records = [json.loads(line) for line in events.read_text().splitlines()]
        self.assertEqual(status, 502)
        self.assertEqual(poisoned_status, 503)
        self.assertIn("transport_recovery_required", poisoned_body)
        self.assertEqual(calls, 1)
        self.assertEqual(completion["request_count"], 1)
        self.assertTrue(completion["transport_recycle_required"])
        self.assertEqual(records[-1]["status"], "ambiguous_upstream_transport")
        self.assertFalse(records[-1]["retry_scheduled"])

    def test_managed_provider_2xx_sideband_blocks_normalized_local_500_retry(self) -> None:
        calls = 0
        classifications: list[bool] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                nonlocal calls
                calls += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                body = b'{"error":"normalized_after_provider_success"}'
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        def classify(outer_complete: bool) -> bool:
            classifications.append(outer_complete)
            return True

        original = self.registry.model("tokenhub-hy4-preview-agent-v1")
        model = replace(
            original,
            retry=replace(
                original.retry,
                max_infrastructure_retries=1,
                retry_delay_seconds=0,
            ),
        )
        route = self.registry.route(model.route_profile_id)
        with tempfile.TemporaryDirectory(prefix="gateway-managed-sideband-") as temporary:
            events = Path(temporary) / "events.jsonl"
            with _Server(Handler) as upstream:
                binding = RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-tokenhub-binding-v1",
                    upstream_base_url=f"{upstream.base_url}/v1",
                    allow_insecure_upstream=True,
                    managed_adapter_id="managed_tokenhub_litellm_v1",
                )
                with ScopedModelGateway(
                    route=route,
                    model=model,
                    runtime_binding=binding,
                    runtime_credential="fixture-tokenhub-key",
                    max_requests=1,
                    event_path=events,
                    managed_transport_ambiguity_probe=classify,
                ) as gateway:
                    status, _ = _gateway_request(
                        gateway,
                        route.client_path,
                        _minimal_chat_payload(model),
                    )
                    completion = gateway.wait_for_completion_report()
            records = [json.loads(line) for line in events.read_text().splitlines()]
        self.assertEqual(status, 500)
        self.assertEqual(calls, 1)
        self.assertEqual(classifications, [False])
        self.assertEqual(
            completion["terminal_statuses"],
            {"1": "ambiguous_managed_transport_sideband"},
        )
        self.assertTrue(completion["transport_recycle_required"])
        self.assertEqual(len(records), 2)
        self.assertFalse(records[-1]["retry_scheduled"])

    def test_truncated_chat_stream_fails_preflight_closed(self) -> None:
        calls = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                nonlocal calls
                calls += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                if calls == 1:
                    body = (
                        "data: "
                        + json.dumps(
                            {
                                "model": "hy4-preview",
                                "choices": [
                                    {
                                        "delta": {
                                            "reasoning_content": "private",
                                            "thinking_blocks": [
                                                {
                                                    "type": "thinking",
                                                    "thinking": "private",
                                                    "signature": "fixture-signature",
                                                }
                                            ],
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "id": "call_fixture",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "sieve_preflight_echo",
                                                        "arguments": '{"nonce":"sieve-preflight-v1"}',
                                                    },
                                                }
                                            ],
                                        }
                                    }
                                ],
                            },
                            separators=(",", ":"),
                        )
                        + "\n\n"
                    ).encode("utf-8")
                else:
                    body = _chat_sse(
                        {
                            "model": "hy4-preview",
                            "choices": [{"delta": {"content": "accepted"}}],
                        }
                    )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        model = self.registry.model("tokenhub-hy4-preview-agent-v1")
        route = self.registry.route(model.route_profile_id)
        with _Server(Handler) as upstream:
            binding = RouteRuntimeBinding(
                route_profile_id=route.route_profile_id,
                binding_profile_id="fixture-tokenhub-binding-v1",
                upstream_base_url=f"{upstream.base_url}/v1",
                allow_insecure_upstream=True,
                managed_adapter_id="managed_tokenhub_litellm_v1",
            )
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=binding,
                runtime_credential="fixture-tokenhub-key",
                max_requests=2,
                managed_transport_ambiguity_probe=lambda _complete: False,
            ) as gateway:
                report = run_tool_call_preflight(
                    gateway=gateway, route=route, model=model
                )
        self.assertFalse(report.ok)
        self.assertEqual(report.failure_code, "second_http_503")
        self.assertEqual(calls, 1)
        self.assertFalse(report.gateway_stream_contract_complete)

    def test_api2_cache_identity_is_fresh_for_each_upstream_attempt(self) -> None:
        authorizations: list[str] = []
        calls = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                nonlocal calls
                calls += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                authorizations.append(str(self.headers.get("Authorization")))
                if calls == 1:
                    body = b'{"error":"fixture_retry"}'
                    self.send_response(429)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = _chat_sse(
                    {
                        "model": "kimi-k3",
                        "choices": [{"delta": {"content": "ok"}}],
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        original = self.registry.model("api2-kimi-k3-agent-v1")
        model = replace(
            original,
            retry=replace(
                original.retry,
                max_infrastructure_retries=1,
                retry_delay_seconds=0,
            ),
        )
        route = self.registry.route(model.route_profile_id)
        with _Server(Handler) as upstream:
            binding = RouteRuntimeBinding(
                route_profile_id=route.route_profile_id,
                binding_profile_id="fixture-api2-chat-binding-v1",
                upstream_base_url=f"{upstream.base_url}/openapi/v2",
                allow_insecure_upstream=True,
            )
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=binding,
                runtime_credential="fixture-app:fixture-key",
                max_requests=1,
            ) as gateway:
                with mock.patch("model_gateway.time.time_ns", return_value=1):
                    status, _ = _gateway_request(
                        gateway,
                        route.client_path,
                        _minimal_chat_payload(model),
                    )
                self.assertEqual(status, 200)
        cache_ids = [
            parse_qs(value.split("?", 1)[1])["cache_task_id"][0]
            for value in authorizations
        ]
        self.assertEqual(len(cache_ids), 2)
        self.assertTrue(all(len(value) == 32 for value in cache_ids))
        self.assertTrue(all(int(value, 16) >= 0 for value in cache_ids))
        self.assertNotEqual(cache_ids[0], cache_ids[1])

    def test_api3_session_id_is_fresh_uuid_for_every_physical_attempt(self) -> None:
        session_ids: list[str] = []
        calls = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                nonlocal calls
                calls += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                session_ids.append(str(self.headers.get("SessionID")))
                self.assert_header_contract()
                if calls in {1, 3}:
                    self.send_response(429)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = _chat_sse(
                    {
                        "model": "claude-opus-5",
                        "choices": [{"delta": {"content": "ok"}}],
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def assert_header_contract(self) -> None:
                if self.headers.get("StrategyType") != "ConsistentHash":
                    raise AssertionError("API3 StrategyType differs")

            def log_message(self, format: str, *args: object) -> None:
                return

        original = self.registry.model("api3-claude-opus-5-agent-v1")
        model = replace(
            original,
            retry=replace(
                original.retry,
                max_infrastructure_retries=1,
                retry_delay_seconds=0,
            ),
        )
        route = self.registry.route(model.route_profile_id)
        with _Server(Handler) as upstream:
            binding = RouteRuntimeBinding(
                route_profile_id=route.route_profile_id,
                binding_profile_id="fixture-api3-binding-v1",
                upstream_base_url=f"{upstream.base_url}/v1",
                allow_insecure_upstream=True,
            )
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=binding,
                runtime_credential="fixture-api3-key",
                max_requests=2,
                session_id="not-an-api3-session-id",
            ) as gateway:
                for _ in range(2):
                    status, _ = _gateway_request(
                        gateway,
                        route.client_path,
                        _minimal_chat_payload(model),
                    )
                    self.assertEqual(status, 200)
        self.assertEqual(len(session_ids), 4)
        self.assertEqual(len(set(session_ids)), 4)
        for value in session_ids:
            self.assertEqual(str(uuid.UUID(value)), value)

    def test_final_retryable_failure_starts_shared_family_cooldown(self) -> None:
        calls = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                nonlocal calls
                calls += 1
                self.rfile.read(int(self.headers["Content-Length"]))
                if calls == 1:
                    self.send_response(429)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = _chat_sse(
                    {
                        "model": "kimi-k3",
                        "choices": [{"delta": {"content": "ok"}}],
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        original = self.registry.model("api2-kimi-k3-agent-v1")
        model = replace(
            original,
            retry=replace(
                original.retry,
                max_infrastructure_retries=0,
                retry_delay_seconds=0.08,
            ),
        )
        route = self.registry.route(model.route_profile_id)
        gate = SharedCooldownGate()
        with _Server(Handler) as upstream:
            binding = RouteRuntimeBinding(
                route_profile_id=route.route_profile_id,
                binding_profile_id="fixture-api2-chat-binding-v1",
                upstream_base_url=f"{upstream.base_url}/openapi/v2",
                allow_insecure_upstream=True,
            )
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=binding,
                runtime_credential="fixture-app:fixture-key",
                max_requests=1,
                cooldown_gate=gate,
            ) as first:
                status, _ = _gateway_request(
                    first,
                    route.client_path,
                    _minimal_chat_payload(model),
                )
                self.assertEqual(status, 429)
            started = time.monotonic()
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=binding,
                runtime_credential="fixture-app:fixture-key",
                max_requests=1,
                cooldown_gate=gate,
            ) as second:
                status, _ = _gateway_request(
                    second,
                    route.client_path,
                    _minimal_chat_payload(model),
                )
            self.assertEqual(status, 200)
            self.assertGreaterEqual(time.monotonic() - started, 0.05)

    def test_close_during_shared_cooldown_has_auditable_terminal_record(self) -> None:
        original = self.registry.model("api2-kimi-k3-agent-v1")
        model = replace(
            original,
            retry=replace(original.retry, retry_delay_seconds=30),
        )
        route = self.registry.route(model.route_profile_id)
        gate = SharedCooldownGate()
        gate.trigger(30)
        binding = RouteRuntimeBinding(
            route_profile_id=route.route_profile_id,
            binding_profile_id="fixture-api2-chat-binding-v1",
            upstream_base_url="http://127.0.0.1:1/openapi/v2",
            allow_insecure_upstream=True,
        )
        with tempfile.TemporaryDirectory(prefix="gateway-cooldown-close-") as temporary:
            events = Path(temporary) / "events.jsonl"
            gateway = ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=binding,
                runtime_credential="fixture-app:fixture-key",
                max_requests=1,
                event_path=events,
                cooldown_gate=gate,
            ).start()
            client_errors: list[BaseException] = []

            def request() -> None:
                try:
                    _gateway_request(
                        gateway,
                        route.client_path,
                        _minimal_chat_payload(model),
                    )
                except BaseException as exc:  # downstream is intentionally aborted
                    client_errors.append(exc)

            client = threading.Thread(target=request, daemon=True)
            client.start()
            deadline = time.monotonic() + 2.0
            while gateway.request_count != 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(gateway.request_count, 1)
            gateway.close()
            client.join(timeout=2.0)
            self.assertFalse(client.is_alive())
            completion = gateway.wait_for_completion_report()
            verification = verify_gateway_audit(
                events,
                completion,
                expected_api_family_id=model.api_family_id,
                expected_route_profile_id=route.route_profile_id,
                expected_model_profile_id=model.model_profile_id,
                expected_retry_policy=model.retry.public_dict(),
            )
        self.assertEqual(
            completion["terminal_statuses"], {"1": "model_gateway_closing"}
        )
        self.assertFalse(completion["transport_recycle_required"])
        self.assertFalse(verification["transport_recycle_required"])

    def test_provider_visible_system_prompt_is_canonical_across_episode_paths(self) -> None:
        observed_prompts: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed_prompts.append(payload["messages"][0]["content"])
                body = _chat_sse(
                    {
                        "model": "api_azure_openai_gpt-5.6-sol",
                        "choices": [{"delta": {"content": "ok"}}],
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        model = self.registry.model("api2-gpt-5-6-sol-agent-v1")
        route = self.registry.route(model.route_profile_id)
        canonical = "fixed benchmark system prompt"
        local_prompts = (
            canonical + "\nCurrent working directory: /episodes/model-a/run-a/workspace\n",
            canonical + "\nCurrent working directory: /episodes/model-b/run-b/workspace\n",
        )
        with _Server(Handler) as upstream:
            binding = RouteRuntimeBinding(
                route_profile_id=route.route_profile_id,
                binding_profile_id="fixture-api2-sol-binding-v1",
                upstream_base_url=f"{upstream.base_url}/v1",
                allow_insecure_upstream=True,
            )
            for prompt in local_prompts:
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                with ScopedModelGateway(
                    route=route,
                    model=model,
                    runtime_binding=binding,
                    runtime_credential="fixture-token",
                    max_requests=1,
                    required_system_prompt_sha256s=(digest,),
                    system_prompt_rewrites={digest: canonical},
                ) as gateway:
                    status, _ = _gateway_request(
                        gateway,
                        route.client_path,
                        {
                            "model": model.client_wire_model,
                            "stream": True,
                            "messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": "fixture"},
                            ],
                        },
                    )
                    self.assertEqual(status, 200)
        self.assertEqual(observed_prompts, [canonical, canonical])
        self.assertNotIn("model-a", json.dumps(observed_prompts))

    def test_chat_reasoning_must_belong_to_exact_tool_call_group(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "reasoning_content": "stale-reasoning",
                    "tool_calls": [{"id": "old-call"}],
                },
                {"role": "tool", "tool_call_id": "old-call", "content": "old"},
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "new-a"}, {"id": "new-b"}],
                },
                {"role": "tool", "tool_call_id": "new-a", "content": "a"},
                {"role": "tool", "tool_call_id": "new-b", "content": "b"},
            ]
        }
        self.assertEqual(
            _reasoning_replay_state(payload, api_protocol="openai-completions"),
            (True, False),
        )
        payload["messages"][3]["reasoning_content"] = "current-reasoning"
        self.assertEqual(
            _reasoning_replay_state(payload, api_protocol="openai-completions"),
            (True, True),
        )
        payload["messages"][4], payload["messages"][5] = (
            payload["messages"][5],
            payload["messages"][4],
        )
        with self.assertRaisesRegex(GatewayError, "call_order_mismatch"):
            _reasoning_replay_state(payload, api_protocol="openai-completions")

    def test_tokenhub_replay_requires_a_complete_signed_anthropic_block(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "reasoning_content": "raw text alone is insufficient",
                    "tool_calls": [{"id": "call-fixture"}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-fixture",
                    "content": "fixture",
                },
            ]
        }
        self.assertEqual(
            _reasoning_replay_state(
                payload,
                api_protocol="openai-completions",
                require_signed_anthropic=True,
            ),
            (True, False),
        )
        payload["messages"][0]["thinking_blocks"] = [
            {
                "type": "thinking",
                "thinking": "raw text alone is insufficient",
                "signature": "opaque-signature",
            }
        ]
        self.assertEqual(
            _reasoning_replay_state(
                payload,
                api_protocol="openai-completions",
                require_signed_anthropic=True,
            ),
            (True, True),
        )

    def test_responses_reasoning_must_belong_to_exact_parallel_call_group(self) -> None:
        payload = {
            "input": [
                {"type": "reasoning", "encrypted_content": "stale"},
                {"type": "function_call", "call_id": "old-call"},
                {"type": "function_call_output", "call_id": "old-call"},
                {"role": "user", "content": "continue"},
                {"type": "function_call", "call_id": "new-a"},
                {"type": "function_call", "call_id": "new-b"},
                {"type": "function_call_output", "call_id": "new-a"},
                {"type": "function_call_output", "call_id": "new-b"},
            ]
        }
        self.assertEqual(
            _reasoning_replay_state(payload, api_protocol="openai-responses"),
            (True, False),
        )
        payload["input"].insert(
            4, {"type": "reasoning", "encrypted_content": "current"}
        )
        self.assertEqual(
            _reasoning_replay_state(payload, api_protocol="openai-responses"),
            (True, True),
        )
        payload["input"][-2], payload["input"][-1] = (
            payload["input"][-1],
            payload["input"][-2],
        )
        with self.assertRaisesRegex(GatewayError, "call_order_mismatch"):
            _reasoning_replay_state(payload, api_protocol="openai-responses")

    def test_responses_system_prompt_has_exactly_one_channel(self) -> None:
        prompt = "frozen prompt"
        expected = hashlib.sha256(prompt.encode()).hexdigest()
        instructions_only = {
            "instructions": prompt,
            "input": [{"role": "user", "content": "task"}],
        }
        self.assertEqual(
            _system_prompt_sha256(
                instructions_only, api_protocol="openai-responses"
            ),
            expected,
        )
        mixed = {
            "instructions": prompt,
            "input": [
                {"role": "developer", "content": "injected"},
                {"role": "user", "content": "task"},
            ],
        }
        self.assertIsNone(
            _system_prompt_sha256(mixed, api_protocol="openai-responses")
        )
        with self.assertRaisesRegex(GatewayError, "rewrite_shape_mismatch"):
            _replace_system_prompt(
                mixed,
                api_protocol="openai-responses",
                replacement="canonical",
            )
        multiple = {
            "input": [
                {"role": "system", "content": prompt},
                {"role": "developer", "content": prompt},
                {"role": "user", "content": "task"},
            ]
        }
        self.assertIsNone(
            _system_prompt_sha256(multiple, api_protocol="openai-responses")
        )


def _minimal_chat_payload(model: object) -> dict[str, object]:
    return {
        "model": getattr(model, "client_wire_model"),
        "stream": True,
        "messages": [{"role": "user", "content": "fixture"}],
    }


def _gateway_request(
    gateway: ScopedModelGateway, path: str, payload: dict[str, object]
) -> tuple[int, str]:
    connection = http.client.HTTPConnection("127.0.0.1", gateway.port, timeout=5)
    connection.request(
        "POST",
        path,
        body=json.dumps(payload),
        headers={
            "Authorization": f"Bearer {gateway.capability_token}",
            "Content-Type": "application/json",
        },
    )
    response = connection.getresponse()
    body = response.read().decode("utf-8")
    status = response.status
    connection.close()
    return status, body


if __name__ == "__main__":
    unittest.main()
