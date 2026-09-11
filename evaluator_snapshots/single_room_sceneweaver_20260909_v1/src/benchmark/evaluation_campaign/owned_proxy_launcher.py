#!/usr/bin/env python3
"""Reviewed owned-process launcher for a local LiteLLM evaluation adapter.

This file intentionally does no routing policy of its own.  It consumes and
validates the exact config/host/port contract, then replaces itself with the
LiteLLM child so process ownership and signal handling remain unambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", choices=("127.0.0.1",), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ownership-token", required=True)
    parser.add_argument("--verify-contract-only", action="store_true")
    args = parser.parse_args(argv)

    config = args.config.expanduser().resolve()
    if not config.is_file() or config.is_symlink():
        raise SystemExit("adapter config must be a regular file")
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be in 1..65535")
    if len(args.ownership_token) < 32:
        raise SystemExit("ownership token is too short")
    digest = _sha256(config)
    if args.verify_contract_only:
        print(
            json.dumps(
                {
                    "schema_version": "owned_proxy_launcher_contract_v1",
                    "config_sha256": digest,
                    "host": args.host,
                    "port": args.port,
                    "ownership_token_present": True,
                },
                sort_keys=True,
            )
        )
        return

    backend = [
        sys.executable,
        "-m",
        "litellm",
        "--config",
        str(config),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    child = subprocess.Popen(backend)

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=8)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        raise SystemExit(child.wait())
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
