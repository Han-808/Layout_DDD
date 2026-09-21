#!/usr/bin/env python3
"""Diagnose API3 alias and request-field compatibility without exposing content."""

from __future__ import annotations

import http.client
import json
import re
import ssl
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from generation_runner import _load_model_config, _request_headers
from transport import post_once


RUNNER_ROOT = Path(__file__).resolve().parent
MODEL_KEY = "api3-claude-opus-4-8"
EXPECTED_ALIASES = (
    "claude-opus-4-8-aihub",
    "claude-sonnet-5-aihub",
    "claude-opus-5-aihub",
    "claude-fable-5-aihub",
)


def _safe_error(body: bytes | None) -> dict[str, object]:
    if not body:
        return {}
    try:
        value = json.loads(body.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError):
        return {"error_envelope": "non_json"}
    error = value.get("error") if isinstance(value, dict) else None
    if not isinstance(error, dict):
        return {"error_envelope": "missing_error_object"}
    message = error.get("message")
    if isinstance(message, str):
        message = re.sub(r"Bearer\s+\S+", "Bearer <redacted>", message)
        message = " ".join(message.split())[:500]
    return {
        "error_type": error.get("type"),
        "error_code": error.get("code"),
        "error_message": message,
    }


def _discover(endpoint: str, api_key: str) -> tuple[int, set[str]]:
    parsed = urlsplit(endpoint)
    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    kwargs: dict[str, object] = {"host": parsed.hostname, "port": parsed.port, "timeout": 30.0}
    if parsed.scheme == "https":
        kwargs["context"] = ssl.create_default_context()
    connection = connection_cls(**kwargs)
    try:
        connection.request(
            "GET",
            "/v1/models",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        )
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    aliases: set[str] = set()
    if response.status == 200:
        value = json.loads(body.decode("utf-8", errors="strict"))
        rows = value.get("data") if isinstance(value, dict) else None
        if isinstance(rows, list):
            aliases = {
                str(row["id"])
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("id"), str)
            }
    return response.status, aliases


def _probe(model, name: str, body: dict[str, object]) -> bool:
    result = post_once(
        model.endpoint,
        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        connect_timeout=30.0,
        read_timeout=300.0,
        request_headers=_request_headers(model, str(uuid.uuid4())),
    )
    report: dict[str, object] = {
        "probe": name,
        "transport_status": result.status,
        "http_status": result.http_status,
        "http_reason": result.http_reason,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
    }
    ok = result.status == "response" and result.http_status is not None and 200 <= result.http_status < 300
    if not ok:
        report.update(_safe_error(result.response_body))
    elif result.response_body:
        try:
            envelope = json.loads(result.response_body.decode("utf-8", errors="strict"))
            choices = envelope.get("choices") if isinstance(envelope, dict) else None
            message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
            usage = envelope.get("usage") if isinstance(envelope, dict) else None
            details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
            reasoning_value = message.get("reasoning_content") if isinstance(message, dict) else None
            if not isinstance(reasoning_value, str) and isinstance(message, dict):
                reasoning_value = message.get("reasoning")
            report["reasoning_nonempty"] = isinstance(reasoning_value, str) and bool(reasoning_value.strip())
            report["reasoning_tokens"] = details.get("reasoning_tokens") if isinstance(details, dict) else None
        except (UnicodeError, json.JSONDecodeError):
            report["success_envelope_parse"] = False
    report["ok"] = ok
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return ok


def main() -> int:
    model = _load_model_config(RUNNER_ROOT / "models.pod.json", MODEL_KEY)
    discovery_status, aliases = _discover(model.endpoint, model.api_key)
    print(json.dumps({
        "probe": "model_discovery",
        "http_status": discovery_status,
        "expected_aliases_present": {alias: alias in aliases for alias in EXPECTED_ALIASES},
    }, sort_keys=True))
    if discovery_status != 200 or model.wire_model not in aliases:
        return 2

    base: dict[str, object] = {
        "model": model.wire_model,
        "messages": [
            {"role": "system", "content": "Reply with exactly OK."},
            {"role": "user", "content": "OK"},
        ],
        "max_tokens": 8,
        "stream": False,
    }
    if not _probe(model, "minimal", dict(base)):
        return 2

    variants = (
        ("max_tokens_65536", {**base, "max_tokens": model.max_tokens}),
        ("temperature_only", {**base, "temperature": model.temperature}),
        ("top_k", {**base, "top_k": model.top_k}),
        ("reasoning_effort_top_level", {**base, "reasoning_effort": model.reasoning_effort}),
        ("thinking_enabled_4096", {**base, "thinking": {"type": "enabled", "budget_tokens": 4096}}),
        ("thinking_adaptive", {**base, "thinking": {"type": "adaptive"}}),
        ("output_config_effort_high", {**base, "output_config": {"effort": "high"}}),
        ("candidate_reasoning_effort", {
            **base,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_k": model.top_k,
            "reasoning_effort": model.reasoning_effort,
        }),
    )
    all_ok = True
    for name, body in variants:
        all_ok = _probe(model, name, body) and all_ok
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
