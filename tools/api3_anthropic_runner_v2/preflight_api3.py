#!/usr/bin/env python3
"""Perform one sanitized real API3 request using the formal request shape."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from generation_runner import (
    _extract_api_message,
    _load_model_config,
    _request_headers,
    _request_value,
)
from transport import post_once


RUNNER_ROOT = Path(__file__).resolve().parent
MODEL_KEYS = (
    "api3-claude-opus-4-8",
    "api3-claude-sonnet-5",
    "api3-claude-opus-5",
    "api3-claude-fable-5",
)


def _normalized(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    args = parser.parse_args()

    model = _load_model_config(RUNNER_ROOT / "models.pod.json", args.model)
    request = _request_value(
        model=model,
        system_prompt="Reply with exactly OK and no other text.",
        user_value={"preflight": "OK"},
    )
    result = post_once(
        model.endpoint,
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        connect_timeout=30.0,
        read_timeout=min(model.timeout_seconds, 300.0),
        request_headers=_request_headers(model, str(uuid.uuid4())),
    )
    if result.status != "response":
        print(json.dumps({
            "model_key": model.key,
            "ok": False,
            "status": result.status,
            "stage": result.stage,
            "error_type": result.error_type,
        }, sort_keys=True))
        return 2
    if result.http_status is None or not 200 <= result.http_status < 300:
        print(json.dumps({
            "model_key": model.key,
            "ok": False,
            "status": "http_error",
            "http_status": result.http_status,
            "http_reason": result.http_reason,
        }, sort_keys=True))
        return 2
    assert result.response_body is not None
    try:
        envelope = json.loads(result.response_body.decode("utf-8", errors="strict"))
        content, _, _, _ = _extract_api_message(result.response_body)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "model_key": model.key,
            "ok": False,
            "status": "invalid_api_response",
            "error_type": type(exc).__name__,
        }, sort_keys=True))
        return 2
    returned_model = envelope.get("model") if isinstance(envelope, dict) else None
    requested_core = _normalized(model.wire_model.removesuffix("-aihub"))
    model_matches = isinstance(returned_model, str) and requested_core in _normalized(returned_model)
    ok = bool(content.strip()) and model_matches
    print(json.dumps({
        "model_key": model.key,
        "ok": ok,
        "status": "passed" if ok else "model_identity_mismatch_or_empty_content",
        "http_status": result.http_status,
        "model_identity_matches": model_matches,
        "content_nonempty": bool(content.strip()),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
    }, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

