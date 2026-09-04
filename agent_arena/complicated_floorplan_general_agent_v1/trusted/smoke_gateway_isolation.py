#!/usr/bin/env python3
"""Prove the exact loopback-gateway allowlist without any external network."""

from __future__ import annotations

import json
from pathlib import Path
import secrets
import shutil
import socketserver
import tempfile
import threading

from arena import ARENA_ROOT
from isolated_exec import run_isolated


class UnixHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.rfile.readline(1024 * 1024)
        self.wfile.write(b'{"ok":true,"result":{}}\n')


class UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class TCPHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.wfile.write(b"ok")
        self.wfile.flush()


class TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _start(server: socketserver.BaseServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def main() -> int:
    episodes = ARENA_ROOT / "episodes"
    temporary_root = Path(tempfile.mkdtemp(prefix="gateway-smoke-", dir=episodes))
    socket_root = Path(tempfile.mkdtemp(prefix="sieve-gateway-tool-"))
    tool_server = UnixServer(str(socket_root / "tool.sock"), UnixHandler)
    allowed_server = TCPServer(("127.0.0.1", 0), TCPHandler)
    blocked_server = TCPServer(("127.0.0.1", 0), TCPHandler)
    threads = [_start(tool_server), _start(allowed_server), _start(blocked_server)]
    try:
        workspace = temporary_root / "workspace"
        host = temporary_root / "host"
        workspace.mkdir(mode=0o700)
        host.mkdir(mode=0o700)
        (workspace / ".home").mkdir(mode=0o700)
        (workspace / ".tmp").mkdir(mode=0o700)
        allowed_port = int(allowed_server.server_address[1])
        blocked_port = int(blocked_server.server_address[1])
        worker = workspace / "worker.rb"
        worker.write_text(
            "\n".join(
                [
                    "require 'json'",
                    "require 'socket'",
                    f"allowed = TCPSocket.new('127.0.0.1', {allowed_port})",
                    "reply = allowed.read(2)",
                    "allowed.close",
                    "exit(51) unless reply == 'ok'",
                    "begin",
                    f"  blocked = TCPSocket.new('127.0.0.1', {blocked_port})",
                    "  blocked.close",
                    "  exit(52)",
                    "rescue SystemCallError",
                    "  denied = true",
                    "end",
                    "File.write('gateway-smoke.json', JSON.generate({'exact_gateway' => 'allowed', 'other_tcp' => (denied ? 'denied' : 'unexpected')}))",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        worker.chmod(0o555)
        result = run_isolated(
            workspace=workspace,
            runtime_root="/usr/bin",
            command=["/usr/bin/ruby", "worker.rb"],
            tool_socket=socket_root / "tool.sock",
            tool_token=secrets.token_hex(32),
            stdout_path=host / "stdout.log",
            stderr_path=host / "stderr.log",
            stdin_text="",
            timeout_seconds=30.0,
            model_gateway=f"localhost:{allowed_port}",
            model_gateway_token=secrets.token_hex(32),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "gateway isolation smoke failed: "
                + json.dumps(result.public_dict(), sort_keys=True)
                + " stderr="
                + (host / "stderr.log").read_text(
                    encoding="utf-8", errors="replace"
                )[-1000:]
            )
        observed = json.loads(
            (workspace / "gateway-smoke.json").read_text(encoding="utf-8")
        )
        if observed != {"exact_gateway": "allowed", "other_tcp": "denied"}:
            raise RuntimeError("gateway isolation smoke assertions differ")
        print(
            json.dumps(
                {
                    "status": "valid",
                    "model_or_generation_started": False,
                    "exact_gateway": "allowed",
                    "other_tcp": "denied",
                    "process": result.public_dict(),
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        for server in (tool_server, allowed_server, blocked_server):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5.0)
        shutil.rmtree(temporary_root, ignore_errors=True)
        shutil.rmtree(socket_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
