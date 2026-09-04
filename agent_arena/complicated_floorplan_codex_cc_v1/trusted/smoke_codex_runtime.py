#!/usr/bin/env python3
"""Verify the installed Codex binary starts inside Seatbelt without a model call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import shutil
import socketserver
import tempfile
import threading

from arena import ARENA_ROOT, sha256_file
from isolated_exec import run_isolated


class EmptyHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        return


class UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-executable", default=shutil.which("codex"))
    args = parser.parse_args()
    if not args.codex_executable:
        raise RuntimeError("Codex executable is unavailable")
    executable = Path(args.codex_executable).expanduser().resolve(strict=True)
    runtime_root = executable.parent.parent
    temporary_root = Path(
        tempfile.mkdtemp(prefix="codex-runtime-smoke-", dir=ARENA_ROOT / "episodes")
    )
    socket_root = Path(tempfile.mkdtemp(prefix="sieve-codex-tool-"))
    server = UnixServer(str(socket_root / "tool.sock"), EmptyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        workspace = temporary_root / "workspace"
        host = temporary_root / "host"
        workspace.mkdir(mode=0o700)
        host.mkdir(mode=0o700)
        (workspace / ".home").mkdir(mode=0o700)
        (workspace / ".tmp").mkdir(mode=0o700)
        result = run_isolated(
            workspace=workspace,
            runtime_root=runtime_root,
            command=[str(executable), "--version"],
            tool_socket=socket_root / "tool.sock",
            tool_token=secrets.token_hex(32),
            stdout_path=host / "stdout.log",
            stderr_path=host / "stderr.log",
            stdin_text="",
            timeout_seconds=30.0,
        )
        stdout = (host / "stdout.log").read_text(encoding="utf-8", errors="replace").strip()
        if result.returncode != 0 or not stdout.startswith("codex-cli "):
            stderr = (host / "stderr.log").read_text(
                encoding="utf-8", errors="replace"
            )
            raise RuntimeError(
                "Codex runtime smoke failed: "
                + json.dumps(result.public_dict(), sort_keys=True)
                + f" stdout={stdout[-500:]!r} stderr={stderr[-500:]!r}"
            )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "codex_version": stdout,
                    "codex_executable_sha256": sha256_file(executable),
                    "outer_isolation": "macos_seatbelt_sandbox_exec_v1",
                    "model_or_generation_started": False,
                    "network_mode": "database_socket_only",
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        shutil.rmtree(temporary_root, ignore_errors=True)
        shutil.rmtree(socket_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
