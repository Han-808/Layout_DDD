from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
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
from isolated_exec import IsolationResult, run_isolated  # noqa: E402
from model_gateway import ScopedModelGateway, SharedCooldownGate  # noqa: E402
from pi_harness import (  # noqa: E402
    PiEpisodeConfig,
    ROUTE_PREFLIGHT_SYSTEM_PROMPT,
    pi_system_prompt_binding,
    prepare_route_preflight,
)
from preflight_pi_route import (  # noqa: E402
    _initialize_workspace,
    _unused_tool_socket,
)


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


def _chat_sse(model: str, *, command: str | None) -> bytes:
    identifier = "chatcmpl-sieve-bash-containment"
    created = int(time.time())
    if command is not None:
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
                            "reasoning_content": "private-containment-fixture",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_sieve_bash_fixture",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": json.dumps(
                                            {"command": command},
                                            separators=(",", ":"),
                                        ),
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
                        "delta": {"role": "assistant", "content": "DONE"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
    payload = "\n\n".join(
        f"data: {json.dumps(item, separators=(',', ':'))}" for item in deltas
    )
    return (payload + "\n\ndata: [DONE]\n\n").encode("utf-8")


class PinnedPiBashContainmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProfileRegistry.load()

    def test_real_pi_background_job_is_terminal_and_removed(self) -> None:
        result, workspace = self._run(
            "/bin/sleep 60 & echo $! > background.pid",
            timeout_seconds=20,
        )
        self.assertEqual(result.status, "resource_limit_exceeded")
        self.assertIn(
            result.resource_limit_kind,
            {"residual_bash_process", "residual_process_group"},
        )
        self._assert_recorded_pid_is_gone(workspace / "background.pid")

    def test_real_pi_outer_timeout_removes_foreground_bash(self) -> None:
        result, workspace = self._run(
            "echo $$ > foreground.pid; /bin/sleep 60",
            timeout_seconds=2,
        )
        self.assertEqual(result.status, "ambiguous_timeout")
        self.assertTrue(result.timed_out)
        self._assert_recorded_pid_is_gone(workspace / "foreground.pid")

    def test_real_pi_outer_timeout_removes_setsid_descendant(self) -> None:
        result, workspace = self._run(
            "/usr/bin/ruby -e 'Process.setsid; File.write(\"setsid.pid\", Process.pid.to_s); sleep 60'",
            timeout_seconds=2,
        )
        self.assertEqual(result.status, "ambiguous_timeout")
        self.assertTrue(result.timed_out)
        self._assert_recorded_pid_is_gone(workspace / "setsid.pid")

    def _run(self, command: str, *, timeout_seconds: int) -> tuple[IsolationResult, Path]:
        model = self.registry.model("api2-kimi-k3-agent-v1")
        route = self.registry.route(model.route_profile_id)
        run_root = (
            ARENA_ROOT
            / "episodes"
            / "pi-bash-containment"
            / model.model_profile_id
            / f"run-test-{secrets.token_hex(8)}"
        )
        host = run_root / "host"
        workspace = run_root / "workspace"
        host.mkdir(parents=True, mode=0o700)
        workspace.mkdir(mode=0o700)
        _initialize_workspace(workspace)
        self.addCleanup(shutil.rmtree, run_root, True)
        observed = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                nonlocal observed
                self.rfile.read(int(self.headers["Content-Length"]))
                observed += 1
                body = _chat_sse(
                    model.client_wire_model,
                    command=command if observed == 1 else None,
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        prompt_binding = pi_system_prompt_binding(
            workspace,
            ROUTE_PREFLIGHT_SYSTEM_PROMPT,
        )
        with _Server(Handler) as upstream, _unused_tool_socket() as (
            tool_socket,
            tool_token,
        ):
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-pi-bash-containment-v1",
                    upstream_base_url=f"{upstream.base_url}/openapi/v2",
                    allow_insecure_upstream=True,
                ),
                runtime_credential="fixture-app:fixture-key",
                max_requests=2,
                event_path=host / "model_gateway_events.jsonl",
                cooldown_gate=SharedCooldownGate(),
                session_id=secrets.token_hex(32),
                required_system_prompt_sha256s=tuple(prompt_binding),
                system_prompt_rewrites=prompt_binding,
            ) as gateway:
                material = prepare_route_preflight(
                    PiEpisodeConfig(
                        runtime_root=RUNTIME_ROOT,
                        workspace=workspace,
                        gateway_base_url=gateway.base_url,
                        model_profile=model,
                        experiment_id="fixture-pi-bash-containment-v1",
                        experiment_sha256="3" * 64,
                        profile_registry_sha256=self.registry.content_sha256,
                        max_model_requests=2,
                        wall_clock_seconds=max(timeout_seconds, 1),
                    )
                )
                result = run_isolated(
                    workspace=workspace,
                    runtime_root=RUNTIME_ROOT,
                    command=material["command"],
                    tool_socket=tool_socket,
                    tool_token=tool_token,
                    stdout_path=host / "agent.stdout.jsonl",
                    stderr_path=host / "agent.stderr.log",
                    stdin_text=material["stdin_text"],
                    timeout_seconds=timeout_seconds,
                    model_gateway=gateway.endpoint_address,
                    model_gateway_token=gateway.capability_token,
                    extra_environment={
                        "ARENA_AGENT_ID": "pi-bash-containment",
                        "ARENA_MODEL_ID": model.model_profile_id,
                        "ARENA_RUN_ID": run_root.name,
                    },
                    harness_extension=material["harness_extension_path"],
                )
        return result, workspace

    def _assert_recorded_pid_is_gone(self, path: Path) -> None:
        self.assertTrue(path.is_file(), path)
        pid = int(path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while _pid_exists(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(_pid_exists(pid), pid)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
