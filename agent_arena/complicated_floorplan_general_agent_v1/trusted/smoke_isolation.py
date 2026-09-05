#!/usr/bin/env python3
"""Prove the local Seatbelt boundary without starting a model or generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import shutil
import socketserver
import tempfile
import threading

from arena import ARENA_ROOT
from isolated_exec import run_isolated


TOKEN = secrets.token_hex(32)
HOST_SECRET_NAME = "SIEVE_SMOKE_HOST_SECRET"


class ToolHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(1024 * 1024)
        request = json.loads(raw.decode("utf-8"))
        ok = (
            request.get("protocol") == "non_rectangular_agent_tool_protocol_v1"
            and secrets.compare_digest(str(request.get("token") or ""), TOKEN)
            and request.get("method") == "get_task"
        )
        response = (
            {"ok": True, "result": {"smoke": "database_socket_reachable"}}
            if ok
            else {"ok": False, "error_type": "unauthorized"}
        )
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class ToolServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main() -> int:
    episodes = ARENA_ROOT / "episodes"
    smoke_id = secrets.token_hex(8)
    agent_root = episodes / f"isolation-smoke-{smoke_id}"
    temporary_root = agent_root / "fixture-scene" / "fixture-run"
    temporary_root.mkdir(parents=True, mode=0o700)
    socket_root = Path(tempfile.mkdtemp(prefix="sieve-smoke-tool-"))
    try:
        workspace = temporary_root / "workspace"
        host = temporary_root / "host"
        workspace.mkdir(mode=0o700)
        host.mkdir(mode=0o700)
        (workspace / ".home").mkdir(mode=0o700)
        (workspace / ".tmp").mkdir(mode=0o700)
        shutil.copy2(ARENA_ROOT / "public/sieve-agent-tool", workspace / "sieve-agent-tool")
        (workspace / "sieve-agent-tool").chmod(0o555)
        outside = ARENA_ROOT / "README.md"
        worker = workspace / "worker.rb"
        worker.write_text(
            "\n".join(
                [
                    "require 'json'",
                    "require 'socket'",
                    f"outside = {str(outside)!r}",
                    f"exit(45) if ENV.key?({HOST_SECRET_NAME!r})",
                    "puts ENV.fetch('LAYOUT_DDD_AGENT_TOOL_TOKEN')",
                    "File.write('workspace-write.txt', 'allowed')",
                    "begin",
                    "  File.read(outside)",
                    "  exit(41)",
                    "rescue Errno::EPERM, Errno::EACCES",
                    "  filesystem = 'denied'",
                    "end",
                    "tool_output = IO.popen(['./sieve-agent-tool', 'get-task'], &:read)",
                    "exit(42) unless $?.success?",
                    "payload = JSON.parse(tool_output)",
                    "exit(43) unless payload.dig('result', 'smoke') == 'database_socket_reachable'",
                    "begin",
                    "  connection = TCPSocket.new('127.0.0.1', 9)",
                    "  connection.close",
                    "  exit(44)",
                    "rescue SystemCallError",
                    "  tcp = 'denied'",
                    "end",
                    "File.write('smoke-result.json', JSON.generate({'outside_read' => filesystem, 'database_socket' => 'allowed', 'arbitrary_tcp' => tcp}))",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        worker.chmod(0o555)
        socket_path = socket_root / "tool.sock"
        server = ToolServer(str(socket_path), ToolHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        os.environ[HOST_SECRET_NAME] = "must-not-enter-agent-environment"
        try:
            result = run_isolated(
                workspace=workspace,
                runtime_root="/usr/bin",
                command=["/usr/bin/ruby", "worker.rb"],
                tool_socket=socket_path,
                tool_token=TOKEN,
                stdout_path=host / "stdout.log",
                stderr_path=host / "stderr.log",
                stdin_text="",
                timeout_seconds=30.0,
            )
        finally:
            os.environ.pop(HOST_SECRET_NAME, None)
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)
        if result.returncode != 0:
            stdout = (host / "stdout.log").read_text(encoding="utf-8", errors="replace")
            stderr = (host / "stderr.log").read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                "isolation smoke process failed: "
                + json.dumps(result.public_dict(), sort_keys=True)
                + f" stdout={stdout[-1000:]!r} stderr={stderr[-1000:]!r}"
            )
        observed = json.loads((workspace / "smoke-result.json").read_text(encoding="utf-8"))
        expected = {
            "outside_read": "denied",
            "database_socket": "allowed",
            "arbitrary_tcp": "denied",
        }
        if observed != expected:
            raise RuntimeError("isolation smoke assertions differ")
        sanitized_stdout = (host / "stdout.log").read_text(encoding="utf-8")
        if TOKEN in sanitized_stdout or "[REDACTED_EPISODE_CAPABILITY]" not in sanitized_stdout:
            raise RuntimeError("episode capability was not redacted from Agent logs")
        print(
            json.dumps(
                {
                    "status": "valid",
                    "model_or_generation_started": False,
                    "filesystem": "workspace_only",
                    "database_socket": "allowed",
                    "arbitrary_tcp": "denied",
                    "host_environment_inherited": False,
                    "capability_log_redaction": "valid",
                    "process": result.public_dict(),
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        shutil.rmtree(agent_root, ignore_errors=True)
        shutil.rmtree(socket_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
