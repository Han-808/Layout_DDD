#!/usr/bin/env python3
"""Sanitized API3 Opus 4.8 preflight with reasoning-signal diagnostics."""

from __future__ import annotations

import argparse
import json
import re
import uuid

import generation_runner as adapter


PREFLIGHT_MAX_TOKENS = 4096


def normalized(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def safe_error(body: bytes | None) -> dict[str, object]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-reasoning-signal",
        action="store_true",
        help=(
            "Treat a missing visible reasoning artifact/token count as a failure. "
            "By default it is diagnostic because adaptive high may omit thinking "
            "and proxy-normalized usage may not expose hidden thinking tokens."
        ),
    )
    args = parser.parse_args()
    runner = adapter.configure_core()
    model = runner._load_model_config(adapter.MODELS_PATH, adapter.MODEL_KEY)
    request = {
        "model": model.wire_model,
        "messages": [
            {
                "role": "system",
                "content": "Solve the task privately and return only the final integer.",
            },
            {
                "role": "user",
                "content": "Compute the remainder of 987654321987654321 divided by 104729.",
            },
        ],
        "max_tokens": PREFLIGHT_MAX_TOKENS,
        "stream": False,
        "thinking": {"type": "adaptive"},
        "reasoning_effort": "high",
    }
    result = runner.post_once(
        model.endpoint,
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        ),
        connect_timeout=30.0,
        read_timeout=min(model.timeout_seconds, 600.0),
        request_headers=runner._request_headers(model, str(uuid.uuid4())),
    )
    report: dict[str, object] = {
        "transport_status": result.status,
        "http_status": result.http_status,
        "http_reason": result.http_reason,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "thinking_type": "adaptive",
        "reasoning_effort": "high",
    }
    if result.status != "response" or result.http_status is None or not 200 <= result.http_status < 300:
        report.update(safe_error(result.response_body))
        report["ok"] = False
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 2
    assert result.response_body is not None
    try:
        envelope = json.loads(result.response_body.decode("utf-8", errors="strict"))
        content, reasoning, reasoning_content, usage = runner._extract_api_message(
            result.response_body
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        report.update(
            {
                "ok": False,
                "status": "invalid_api_response",
                "error_type": type(exc).__name__,
            }
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 2
    returned_model = envelope.get("model") if isinstance(envelope, dict) else None
    requested_core = normalized(model.wire_model.removesuffix("-aihub"))
    model_matches = (
        isinstance(returned_model, str)
        and requested_core in normalized(returned_model)
    )
    reasoning_nonempty = bool(
        (reasoning is not None and reasoning.strip())
        or (reasoning_content is not None and reasoning_content.strip())
    )
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage, dict)
        else None
    )
    reasoning_tokens = (
        details.get("reasoning_tokens") if isinstance(details, dict) else None
    )
    reasoning_signal = reasoning_nonempty or (
        isinstance(reasoning_tokens, int) and reasoning_tokens > 0
    )
    response_ok = bool(content.strip()) and model_matches
    ok = response_ok and (reasoning_signal or not args.require_reasoning_signal)
    if not response_ok:
        status = "response_validation_failed"
    elif reasoning_signal:
        status = "passed"
    elif args.require_reasoning_signal:
        status = "reasoning_signal_missing"
    else:
        status = "passed_without_visible_reasoning_signal"
    report.update(
        {
            "ok": ok,
            "status": status,
            "model_identity_matches": model_matches,
            "content_nonempty": bool(content.strip()),
            "reasoning_nonempty": reasoning_nonempty,
            "reasoning_tokens": reasoning_tokens,
            "reasoning_signal_required": args.require_reasoning_signal,
        }
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
