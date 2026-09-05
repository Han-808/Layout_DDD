from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
import sys
import threading
import time
import unittest


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
if str(TRUSTED) not in sys.path:
    sys.path.insert(0, str(TRUSTED))

from api_profiles import ProfileRegistry, RouteRuntimeBinding  # noqa: E402
from model_gateway import SharedCooldownGate  # noqa: E402
from preflight_pi_route import run_pi_route_preflight  # noqa: E402


RUNTIME_ROOT = (ARENA_ROOT / "../runtime_bundles/pi-0.85.0").resolve()


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


def _chat_sse(model: str, *, tool_call: bool) -> bytes:
    identifier = "chatcmpl-sieve-fixture"
    created = int(time.time())
    if tool_call:
        deltas = [
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "private-fixture-reasoning",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_sieve_fixture",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": '{"path":"preflight_fixture.txt"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                ],
            },
        ]
    else:
        deltas = [
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "SIEVE_PREFLIGHT_OK",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            },
        ]
    deltas.append(
        {
            "id": identifier,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )
    payload = "\n\n".join(
        f"data: {json.dumps(item, separators=(',', ':'))}" for item in deltas
    )
    return (payload + "\n\ndata: [DONE]\n\n").encode("utf-8")


def _tokenhub_chat_sse(*, tool_call: bool) -> bytes:
    """Mimic the pinned LiteLLM Anthropic streaming projection."""

    identifier = "chatcmpl-sieve-tokenhub-fixture"
    created = int(time.time())
    if tool_call:
        deltas = [
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "hy4-preview",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "private-tokenhub-thinking",
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "private-tokenhub-thinking",
                                    "signature": "",
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "hy4-preview",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "",
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": "opaque-tokenhub-signature",
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "hy4-preview",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_sieve_fixture",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": '{"path":"preflight_fixture.txt"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "hy4-preview",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                ],
            },
        ]
    else:
        deltas = [
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "hy4-preview",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "SIEVE_PREFLIGHT_OK",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "hy4-preview",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            },
        ]
    deltas.append(
        {
            "id": identifier,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "hy4-preview",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )
    payload = "\n\n".join(
        f"data: {json.dumps(item, separators=(',', ':'))}" for item in deltas
    )
    return (payload + "\n\ndata: [DONE]\n\n").encode("utf-8")


def _responses_sse(model: str, *, tool_call: bool) -> bytes:
    response_id = "resp_sieve_fixture"
    usage = {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 1 if tool_call else 0},
        "total_tokens": 15,
    }
    if tool_call:
        reasoning = {
            "type": "reasoning",
            "id": "rs_sieve_fixture",
            "summary": [],
            "encrypted_content": "private-encrypted-fixture",
        }
        function_call = {
            "type": "function_call",
            "id": "fc_sieve_fixture",
            "call_id": "call_sieve_fixture",
            "name": "read",
            "arguments": '{"path":"preflight_fixture.txt"}',
            "status": "completed",
        }
        output = [reasoning, function_call]
        events = [
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "status": "in_progress",
                    "model": model,
                    "output": [],
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": reasoning,
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": reasoning,
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {**function_call, "arguments": "", "status": "in_progress"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_sieve_fixture",
                "output_index": 1,
                "delta": '{"path":"preflight_fixture.txt"}',
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_sieve_fixture",
                "output_index": 1,
                "arguments": '{"path":"preflight_fixture.txt"}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": function_call,
            },
        ]
    else:
        message = {
            "type": "message",
            "id": "msg_sieve_fixture",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "SIEVE_PREFLIGHT_OK",
                    "annotations": [],
                }
            ],
        }
        output = [message]
        events = [
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "status": "in_progress",
                    "model": model,
                    "output": [],
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**message, "content": [], "status": "in_progress"},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_sieve_fixture",
                "output_index": 0,
                "content_index": 0,
                "delta": "SIEVE_PREFLIGHT_OK",
            },
            {
                "type": "response.output_text.done",
                "item_id": "msg_sieve_fixture",
                "output_index": 0,
                "content_index": 0,
                "text": "SIEVE_PREFLIGHT_OK",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": message,
            },
        ]
    events.append(
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": model,
                "output": output,
                "usage": usage,
            },
        }
    )
    return (
        "\n\n".join(
            f"event: {item['type']}\ndata: {json.dumps(item, separators=(',', ':'))}"
            for item in events
        )
        + "\n\n"
    ).encode("utf-8")


class PinnedPiRouteCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProfileRegistry.load()

    def test_real_pi_chat_tool_roundtrip_preserves_reasoning(self) -> None:
        observed: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                observed.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                body = _chat_sse("kimi-k3", tool_call=len(observed) == 1)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        model = self.registry.model("api2-kimi-k3-agent-v1")
        route = self.registry.route(model.route_profile_id)
        audit_root = self._audit_root(model.model_profile_id)
        self.addCleanup(shutil.rmtree, audit_root, True)
        with _Server(Handler) as upstream:
            report = run_pi_route_preflight(
                runtime_root=RUNTIME_ROOT,
                audit_root=audit_root,
                experiment_id="fixture-api2-pi-preflight-v1",
                experiment_sha256="1" * 64,
                profile_registry_sha256=self.registry.content_sha256,
                route=route,
                model=model,
                runtime_binding=RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-api2-chat-binding-v1",
                    upstream_base_url=f"{upstream.base_url}/openapi/v2",
                    allow_insecure_upstream=True,
                ),
                runtime_credential="fixture-app:fixture-key",
                cooldown_gate=SharedCooldownGate(),
                wall_clock_seconds=60,
            )
        self.assertTrue(report.ok, report.public_dict())
        self.assertTrue(report.exactly_one_read_call)
        self.assertTrue(report.final_marker_exact)
        self.assertTrue(report.reasoning_replayed_on_tool_followup)
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0]["reasoning_effort"], "max")
        self.assertTrue(observed[0]["stream"])
        self.assertEqual(observed[0]["max_tokens"], 65536)
        tool_names = sorted(
            item["function"]["name"] for item in observed[0]["tools"]
        )
        self.assertEqual(tool_names, ["bash", "edit", "read", "write"])
        assistant_messages = [
            item
            for item in observed[1]["messages"]
            if item.get("role") == "assistant"
        ]
        self.assertEqual(
            assistant_messages[-1]["reasoning_content"],
            "private-fixture-reasoning",
        )
        public_artifacts = "".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in (audit_root / "host").iterdir()
            if path.is_file()
        )
        self.assertNotIn("private-fixture-reasoning", public_artifacts)
        self.assertNotIn("fixture-app:fixture-key", public_artifacts)

    def test_real_pi_tokenhub_roundtrip_restores_signed_anthropic_thinking(self) -> None:
        observed: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                observed.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                body = _tokenhub_chat_sse(tool_call=len(observed) == 1)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        model = self.registry.model("tokenhub-hy4-preview-agent-v1")
        route = self.registry.route(model.route_profile_id)
        audit_root = self._audit_root(model.model_profile_id)
        self.addCleanup(shutil.rmtree, audit_root, True)
        with _Server(Handler) as upstream:
            report = run_pi_route_preflight(
                runtime_root=RUNTIME_ROOT,
                audit_root=audit_root,
                experiment_id="fixture-tokenhub-pi-preflight-v1",
                experiment_sha256="3" * 64,
                profile_registry_sha256=self.registry.content_sha256,
                route=route,
                model=model,
                runtime_binding=RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-tokenhub-binding-v1",
                    upstream_base_url=f"{upstream.base_url}/v1",
                    allow_insecure_upstream=True,
                    managed_adapter_id="managed_tokenhub_litellm_v1",
                ),
                runtime_credential="fixture-local-proxy-key",
                cooldown_gate=SharedCooldownGate(),
                wall_clock_seconds=60,
                # This fixture emulates already-transformed managed output;
                # relay ambiguity behavior is exercised by the real
                # Pi/LiteLLM release gate in test_managed_transport.py.
                managed_transport_ambiguity_probe=lambda _complete: False,
            )
        self.assertTrue(report.ok, report.public_dict())
        self.assertTrue(report.exactly_one_read_call)
        self.assertTrue(report.final_marker_exact)
        self.assertTrue(report.reasoning_replayed_on_tool_followup)
        self.assertEqual(len(observed), 2)
        assistant_messages = [
            item
            for item in observed[1]["messages"]
            if item.get("role") == "assistant"
        ]
        self.assertEqual(
            assistant_messages[-1]["thinking_blocks"],
            [
                {
                    "type": "thinking",
                    "thinking": "private-tokenhub-thinking",
                    "signature": "opaque-tokenhub-signature",
                }
            ],
        )
        self.assertNotIn("reasoning_details", assistant_messages[-1])
        self.assertNotIn("reasoning_content", assistant_messages[-1])
        public_artifacts = "".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in (audit_root / "host").iterdir()
            if path.is_file()
        )
        self.assertNotIn("private-tokenhub-thinking", public_artifacts)
        self.assertNotIn("opaque-tokenhub-signature", public_artifacts)
        self.assertNotIn("fixture-local-proxy-key", public_artifacts)

    def test_real_pi_responses_tool_roundtrip_replays_reasoning_item(self) -> None:
        observed: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                observed.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                body = _responses_sse("glm-5.3", tool_call=len(observed) == 1)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        model = self.registry.model("api2-glm-5-3-agent-v1")
        route = self.registry.route(model.route_profile_id)
        audit_root = self._audit_root(model.model_profile_id)
        self.addCleanup(shutil.rmtree, audit_root, True)
        with _Server(Handler) as upstream:
            report = run_pi_route_preflight(
                runtime_root=RUNTIME_ROOT,
                audit_root=audit_root,
                experiment_id="fixture-api2-responses-pi-preflight-v1",
                experiment_sha256="2" * 64,
                profile_registry_sha256=self.registry.content_sha256,
                route=route,
                model=model,
                runtime_binding=RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-api2-responses-binding-v1",
                    upstream_base_url=f"{upstream.base_url}/openapi/v2",
                    allow_insecure_upstream=True,
                ),
                runtime_credential="fixture-app:fixture-key",
                cooldown_gate=SharedCooldownGate(),
                wall_clock_seconds=60,
            )
        self.assertTrue(report.ok, report.public_dict())
        self.assertTrue(report.exactly_one_read_call)
        self.assertTrue(report.final_marker_exact)
        self.assertTrue(report.reasoning_replayed_on_tool_followup)
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0]["reasoning"], {"effort": "max"})
        self.assertFalse(observed[0]["store"])
        self.assertEqual(observed[0]["max_output_tokens"], 65536)
        for request in observed:
            self.assertNotIn("prompt_cache_key", request)
            self.assertNotIn("prompt_cache_retention", request)
            self.assertNotIn("prompt_cache_options", request)
        self.assertEqual(
            sorted(item["name"] for item in observed[0]["tools"]),
            ["bash", "edit", "read", "write"],
        )
        replayed_reasoning = [
            item for item in observed[1]["input"] if item.get("type") == "reasoning"
        ]
        self.assertEqual(
            replayed_reasoning[-1]["encrypted_content"],
            "private-encrypted-fixture",
        )
        self.assertTrue(
            any(
                item.get("type") == "function_call_output"
                for item in observed[1]["input"]
            )
        )
        public_artifacts = "".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in (audit_root / "host").iterdir()
            if path.is_file()
        )
        self.assertNotIn("private-encrypted-fixture", public_artifacts)
        self.assertNotIn("fixture-app:fixture-key", public_artifacts)

    @staticmethod
    def _audit_root(model_profile_id: str) -> Path:
        run_id = f"run-test-{secrets.token_hex(8)}"
        return (
            ARENA_ROOT
            / "episodes"
            / "route-preflight"
            / model_profile_id
            / run_id
        )


if __name__ == "__main__":
    unittest.main()
