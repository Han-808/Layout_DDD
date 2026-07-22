#!/usr/bin/env python3
"""Probe an internal model gateway without leaking credentials or cookies.

The default checks distinguish transport-level HTTP success from an
application-level error such as Amazon Coral's UnknownOperationException.  If
the gateway owner later provides an ``X-Amz-Target`` and request payload, the
same script can send that exact operation with ``--target`` and
``--payload-file``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


DEFAULT_BASE_URL = "http://trpc-gpt-eval.production.polaris:8080"
DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_BODY_CHARS = 2_000


def main() -> None:
    args = _parse_args()
    api_key = _resolve_api_key(args.api_key_env)
    base_url = args.base_url.rstrip("/")
    checks: list[dict[str, Any]] = []

    checks.append(
        _request(
            name="health",
            method="POST",
            url=_url(base_url, "/health"),
            api_key=api_key,
            auth_header=args.auth_header,
            auth_prefix=args.auth_prefix,
            payload={},
            timeout=args.timeout_seconds,
        )
    )

    for path in ("/models", "/v1/models"):
        checks.append(
            _request(
                name=f"models:{path}",
                method="GET",
                url=_url(base_url, path),
                api_key=api_key,
                auth_header=args.auth_header,
                auth_prefix=args.auth_prefix,
                timeout=args.timeout_seconds,
            )
        )

    chat_payload: dict[str, Any] = {}
    if args.model:
        chat_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8,
            "temperature": 0,
        }
    for path in ("/chat/completions", "/v1/chat/completions"):
        checks.append(
            _request(
                name=f"chat:{path}",
                method="POST",
                url=_url(base_url, path),
                api_key=api_key,
                auth_header=args.auth_header,
                auth_prefix=args.auth_prefix,
                payload=chat_payload,
                timeout=args.timeout_seconds,
            )
        )

    if args.target:
        payload = _read_payload(args.payload_file)
        checks.append(
            _request(
                name=f"coral_target:{args.target}",
                method="POST",
                url=_url(base_url, args.target_path),
                api_key=api_key,
                auth_header=args.auth_header,
                auth_prefix=args.auth_prefix,
                payload=payload,
                timeout=args.timeout_seconds,
                content_type="application/x-amz-json-1.0",
                extra_headers={"X-Amz-Target": args.target},
            )
        )

    report = {
        "base_url": base_url,
        "credential_source": f"environment:{args.api_key_env}",
        "classification": _classify(checks),
        "openai_compatible": any(check.get("openai_chat_response") for check in checks),
        "models": _model_ids(checks),
        "checks": checks,
        "next_step": _next_step(checks, target_supplied=bool(args.target)),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    if args.out:
        output_path = Path(args.out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely probe an internal OpenAI-like or Amazon Coral model gateway."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--api-key-env",
        default="BENCHMARK_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument("--auth-header", default="Authorization")
    parser.add_argument("--auth-prefix", default="Bearer ")
    parser.add_argument("--model", default=None, help="Optional known model alias for chat probes.")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--target", default=None, help="Optional X-Amz-Target supplied by the API owner.")
    parser.add_argument("--target-path", default="/")
    parser.add_argument("--payload-file", default=None, help="JSON payload for --target.")
    parser.add_argument("--out", default=None, help="Optional path for the sanitized JSON report.")
    return parser.parse_args()


def _resolve_api_key(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    raise SystemExit(f"No API key configured. Export {env_name} and rerun the probe.")


def _url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _request(
    *,
    name: str,
    method: str,
    url: str,
    api_key: str,
    auth_header: str,
    auth_prefix: str,
    timeout: float,
    payload: dict[str, Any] | list[Any] | None = None,
    content_type: str = "application/json",
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        auth_header: f"{auth_prefix}{api_key}",
    }
    if payload is not None:
        headers["Content-Type"] = content_type
    headers.update(extra_headers or {})
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            response_body = response.read().decode("utf-8", errors="replace")
            response_headers = response.headers
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        response_body = exc.read().decode("utf-8", errors="replace")
        response_headers = exc.headers
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return {
            "name": name,
            "method": method,
            "url": url,
            "transport_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    parsed = _parse_json(response_body)
    application_error = _application_error(parsed)
    return {
        "name": name,
        "method": method,
        "url": url,
        "http_status": status,
        "transport_ok": 200 <= status < 300,
        "application_ok": 200 <= status < 300 and application_error is None,
        "application_error": application_error,
        "openai_chat_response": _is_openai_chat_response(parsed),
        "content_type": response_headers.get("Content-Type"),
        "request_id": response_headers.get("X-Request-Id"),
        # Deliberately exclude Set-Cookie, Authorization, and proxy session data.
        "body": _safe_body(response_body, api_key),
    }


def _read_payload(path: str | None) -> dict[str, Any] | list[Any]:
    if path is None:
        return {}
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, (dict, list)):
        raise ValueError("payload JSON must be an object or array")
    return value


def _parse_json(body: str) -> Any:
    try:
        return json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return None


def _application_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    output = payload.get("Output")
    if isinstance(output, dict) and output.get("__type"):
        return str(output["__type"])
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("type") or error.get("code") or error.get("message") or "error")
    return None


def _is_openai_chat_response(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("choices"), list)


def _model_ids(checks: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for check in checks:
        if not str(check.get("name") or "").startswith("models:"):
            continue
        payload = _parse_json(str(check.get("body") or ""))
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            continue
        for item in payload["data"]:
            if isinstance(item, dict) and item.get("id"):
                result.append(str(item["id"]))
    return sorted(set(result))


def _classify(checks: list[dict[str, Any]]) -> str:
    if any(check.get("openai_chat_response") for check in checks):
        return "openai_compatible_chat"
    errors = {str(check.get("application_error") or "") for check in checks}
    if any("UnknownOperationException" in error for error in errors):
        return "reachable_custom_amazon_coral_or_rpc"
    if any(check.get("transport_ok") for check in checks):
        return "reachable_protocol_unknown"
    return "unreachable"


def _next_step(checks: list[dict[str, Any]], *, target_supplied: bool) -> str:
    if any(check.get("openai_chat_response") for check in checks):
        return "The gateway accepted an OpenAI-compatible chat response."
    if target_supplied:
        return "Inspect the coral_target result and align the payload with the provider contract."
    if any(
        "UnknownOperationException" in str(check.get("application_error") or "")
        for check in checks
    ):
        return (
            "Obtain the provider's X-Amz-Target (or RPC method), content type, payload schema, "
            "authentication header, and model aliases; then rerun with --target."
        )
    return "Obtain a provider request example before implementing a benchmark adapter."


def _safe_body(body: str, api_key: str) -> str:
    safe = body.replace(api_key, "<redacted>") if api_key else body
    return safe[:MAX_BODY_CHARS]


if __name__ == "__main__":
    main()
