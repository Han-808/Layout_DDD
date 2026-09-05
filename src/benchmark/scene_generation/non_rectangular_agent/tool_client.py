"""Small CLI client for the local Agent-track tool service."""

from __future__ import annotations

import argparse
import json
import os
import socket
from typing import Any, Iterable, Mapping

from .tool_server import MAX_REQUEST_BYTES, TOOL_PROTOCOL_VERSION


SOCKET_ENV = "LAYOUT_DDD_AGENT_TOOL_SOCKET"
TOKEN_ENV = "LAYOUT_DDD_AGENT_TOOL_TOKEN"


def call_tool(
    method: str,
    params: Mapping[str, Any],
    *,
    socket_path: str | None = None,
    token: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    resolved_socket = socket_path or os.environ.get(SOCKET_ENV, "")
    resolved_token = token or os.environ.get(TOKEN_ENV, "")
    if not resolved_socket or not resolved_token:
        raise RuntimeError("Agent tool connection environment is unavailable")
    request = {
        "protocol": TOOL_PROTOCOL_VERSION,
        "token": resolved_token,
        "method": method,
        "params": dict(params),
    }
    encoded = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        raise RuntimeError("Agent tool request exceeds size limit")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout_seconds)
        connection.connect(resolved_socket)
        connection.sendall(encoded)
        stream = connection.makefile("rb")
        raw = stream.readline(MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise RuntimeError("Agent tool response is missing or oversized")
    response = json.loads(raw.decode("utf-8"))
    if not isinstance(response, dict):
        raise RuntimeError("Agent tool response root must be an object")
    return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("get-task")
    search = commands.add_parser("search-assets")
    search.add_argument("--query", required=True)
    search.add_argument("--size", nargs=3, type=float)
    search.add_argument("--top-k", type=int, default=8)
    inspect = commands.add_parser("inspect-asset")
    inspect.add_argument("asset_id")
    validate = commands.add_parser("validate-submission")
    validate.add_argument("submission_path")
    finalize = commands.add_parser("finalize-submission")
    finalize.add_argument("submission_path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "get-task":
        method, params = "get_task", {}
    elif args.command == "search-assets":
        method, params = "search_assets", {
            "query": args.query,
            "size_constraint": args.size,
            "top_k": args.top_k,
        }
    elif args.command == "inspect-asset":
        method, params = "inspect_asset", {"asset_id": args.asset_id}
    elif args.command == "validate-submission":
        method, params = "validate_submission", {
            "submission_path": args.submission_path
        }
    else:
        method, params = "finalize_submission", {
            "submission_path": args.submission_path
        }
    try:
        response = call_tool(method, params)
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    return 0 if response.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SOCKET_ENV", "TOKEN_ENV", "call_tool", "main"]
