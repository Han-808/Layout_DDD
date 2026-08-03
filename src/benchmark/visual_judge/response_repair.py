from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from benchmark.models import parse_json_object
from benchmark.visual_judge.contracts import ResponseSchemaRepairError


_RAW_RESPONSE_LIMIT = 20_000
_BINARY_SCHEMA_REPAIR_PROMPT = """Your previous response violated the required
JSON response schema. Use exactly the same visual evidence and preserve the same
semantic decision and explanation. Correct only the JSON structure. Return one
object with status valid, invalid, or need_more_evidence; confidence in [0,1];
a non-empty reason; defects exactly []; and evidence_request null for valid or
invalid, or the required structured evidence_request for need_more_evidence.
Return JSON only."""


def repair_binary_response_schema_once(
    *,
    model: Any,
    messages: list[dict[str, Any]],
    response_format_json: bool,
    call_type: str,
    judge_label: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate once, then permit one same-evidence schema-only repair."""

    raw = model.chat_messages(
        messages,
        response_format_json=response_format_json,
        call_type=call_type,
    )
    first_metadata = dict(model.last_request_metadata)
    try:
        result = validator(parse_json_object(raw))
    except (TypeError, ValueError, KeyError) as first_error:
        first_attempt = {
            "attempt": 1,
            "call_type": call_type,
            "raw_response": _bounded_raw_response(raw),
            "validation_error_type": type(first_error).__name__,
            "validation_error": str(first_error),
            "request_metadata": first_metadata,
        }
    else:
        return result, {
            "policy": "single_schema_repair_retry_v1",
            "attempt_count": 1,
            "repair_retry_count": 0,
            "recovered": False,
            "attempts": [
                {
                    "attempt": 1,
                    "call_type": call_type,
                    "validation_error": None,
                }
            ],
        }

    repair_call_type = f"{call_type}.schema_repair"
    repair_messages = [
        *deepcopy(messages),
        {"role": "assistant", "content": raw},
        {"role": "user", "content": _BINARY_SCHEMA_REPAIR_PROMPT},
    ]
    try:
        repaired_raw = model.chat_messages(
            repair_messages,
            response_format_json=response_format_json,
            call_type=repair_call_type,
        )
    except Exception as repair_transport_error:
        audit = {
            "policy": "single_schema_repair_retry_v1",
            "attempt_count": 2,
            "repair_retry_count": 1,
            "recovered": False,
            "attempts": [
                first_attempt,
                {
                    "attempt": 2,
                    "call_type": repair_call_type,
                    "raw_response": None,
                    "validation_error_type": type(
                        repair_transport_error
                    ).__name__,
                    "validation_error": str(repair_transport_error),
                    "failure_kind": "transport",
                },
            ],
        }
        raise ResponseSchemaRepairError(
            f"{judge_label} schema repair request failed: "
            f"{repair_transport_error}",
            schema_audit=audit,
        ) from repair_transport_error
    second_metadata = dict(model.last_request_metadata)
    try:
        repaired = validator(parse_json_object(repaired_raw))
    except (TypeError, ValueError, KeyError) as second_error:
        audit = {
            "policy": "single_schema_repair_retry_v1",
            "attempt_count": 2,
            "repair_retry_count": 1,
            "recovered": False,
            "attempts": [
                first_attempt,
                {
                    "attempt": 2,
                    "call_type": repair_call_type,
                    "raw_response": _bounded_raw_response(repaired_raw),
                    "validation_error_type": type(second_error).__name__,
                    "validation_error": str(second_error),
                    "request_metadata": second_metadata,
                },
            ],
        }
        raise ResponseSchemaRepairError(
            f"{judge_label} response schema remained invalid after one "
            f"repair retry: {second_error}",
            schema_audit=audit,
        ) from second_error
    return repaired, {
        "policy": "single_schema_repair_retry_v1",
        "attempt_count": 2,
        "repair_retry_count": 1,
        "recovered": True,
        "attempts": [
            first_attempt,
            {
                "attempt": 2,
                "call_type": repair_call_type,
                "raw_response": _bounded_raw_response(repaired_raw),
                "validation_error": None,
                "request_metadata": second_metadata,
            },
        ],
    }


def _bounded_raw_response(value: Any) -> str:
    raw = str(value)
    if len(raw) <= _RAW_RESPONSE_LIMIT:
        return raw
    return raw[:_RAW_RESPONSE_LIMIT] + "\n...[truncated]"
