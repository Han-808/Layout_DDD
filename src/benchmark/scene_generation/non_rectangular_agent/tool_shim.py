#!/usr/bin/env python3
"""Standalone workspace client copied beside each Agent task."""

from __future__ import annotations

import argparse
import json
import os
import socket


PROTOCOL = "non_rectangular_agent_tool_protocol_v1"
MAX_BYTES = 4 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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


def _request(method: str, params: dict) -> dict:
    socket_path = os.environ.get("LAYOUT_DDD_AGENT_TOOL_SOCKET", "")
    token = os.environ.get("LAYOUT_DDD_AGENT_TOOL_TOKEN", "")
    if not socket_path or not token:
        raise RuntimeError("Agent tool connection is unavailable")
    payload = {
        "protocol": PROTOCOL,
        "token": token,
        "method": method,
        "params": params,
    }
    encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise RuntimeError("Agent tool request is oversized")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(120.0)
        connection.connect(socket_path)
        connection.sendall(encoded)
        raw = connection.makefile("rb").readline(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise RuntimeError("Agent tool response is missing or oversized")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Agent tool response must be an object")
    return value


def main() -> int:
    args = _parser().parse_args()
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
        response = _request(method, params)
    except Exception as exc:
        response = {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    return 0 if response.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
