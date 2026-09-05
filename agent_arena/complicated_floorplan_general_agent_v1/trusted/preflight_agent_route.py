#!/usr/bin/env python3
"""Streaming two-turn tool-call preflight for one registered model route.

The preflight deliberately exercises the same scoped gateway that Pi will use:
one forced function call, one function result, and one non-empty follow-up turn.
Provider reasoning needed for a stateless tool turn is retained only in memory
and replayed when the registered profile requires it.  Raw prompts, response
text, reasoning, headers, endpoints, credentials, and provider identifiers are
never written to the report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from api_profiles import ModelProfile, RouteProfile
from model_gateway import ScopedModelGateway


PREFLIGHT_TOOL = "sieve_preflight_echo"
PREFLIGHT_NONCE = "sieve-preflight-v1"
MAX_PREFLIGHT_RESPONSE_BYTES = 8 * 1024 * 1024


class PreflightError(RuntimeError):
    """A sanitized, stable preflight failure code."""


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    model_profile_id: str
    api_family_id: str
    route_profile_id: str
    response_contract: str
    protocol: str
    logical_requests: int
    first_http_status: int | None
    second_http_status: int | None
    tool_call_detected: bool
    tool_name_matches: bool
    tool_arguments_match: bool
    reasoning_replay_required: bool
    reasoning_signal_present: bool | None
    reasoning_replayed: bool
    followup_nonempty: bool
    response_identity_required: bool
    response_identity_matches: bool | None
    gateway_stream_contract_complete: bool
    failure_category: str | None
    failure_code: str | None
    elapsed_seconds: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sieve_agent_tool_call_preflight_report_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **asdict(self),
            "raw_request_recorded": False,
            "raw_response_recorded": False,
            "hidden_reasoning_recorded": False,
            "credential_or_endpoint_recorded": False,
        }


@dataclass(frozen=True)
class _Turn:
    http_status: int
    model_identity: str | None
    text_nonempty: bool
    call_id: str | None
    call_name: str | None
    call_arguments: str | None
    chat_reasoning_fields: Mapping[str, Any]
    responses_reasoning_items: tuple[Mapping[str, Any], ...]

    @property
    def reasoning_signal_present(self) -> bool:
        return bool(self.chat_reasoning_fields or self.responses_reasoning_items)


def run_tool_call_preflight(
    *,
    gateway: ScopedModelGateway,
    route: RouteProfile,
    model: ModelProfile,
) -> PreflightReport:
    """Run the two logical requests and return only sanitized observables."""

    if gateway.route != route or gateway.model != model:
        raise PreflightError("gateway_profile_mismatch")
    started = time.monotonic()
    first_status: int | None = None
    second_status: int | None = None
    tool_detected = False
    tool_name_matches = False
    tool_arguments_match = False
    reasoning_present: bool | None = None
    reasoning_replayed = False
    followup_nonempty = False
    identity_matches: bool | None = None
    gateway_stream_contract_complete = False
    category: str | None = None
    code: str | None = None
    logical_requests = 0
    try:
        first_payload = _first_payload(route, model)
        first = _request(gateway, route, first_payload)
        logical_requests += 1
        first_status = first.http_status
        if not 200 <= first.http_status <= 299:
            raise PreflightError(f"first_http_{first.http_status}")
        tool_detected = first.call_id is not None
        tool_name_matches = first.call_name == PREFLIGHT_TOOL
        tool_arguments_match = _arguments_match(first.call_arguments)
        if not tool_detected:
            raise PreflightError("tool_call_missing")
        if not tool_name_matches:
            raise PreflightError("tool_name_mismatch")
        if not tool_arguments_match:
            raise PreflightError("tool_arguments_mismatch")
        reasoning_present = first.reasoning_signal_present
        if model.reasoning.preserve_across_tool_turns and not reasoning_present:
            raise PreflightError("required_reasoning_replay_signal_missing")
        second_payload, reasoning_replayed = _second_payload(
            route,
            model,
            first,
        )
        second = _request(gateway, route, second_payload)
        logical_requests += 1
        second_status = second.http_status
        if not 200 <= second.http_status <= 299:
            raise PreflightError(f"second_http_{second.http_status}")
        followup_nonempty = second.text_nonempty
        if not followup_nonempty:
            raise PreflightError("followup_content_empty")
        observed_identities = tuple(
            value
            for value in (first.model_identity, second.model_identity)
            if value is not None
        )
        if model.response_identity_required:
            identity_matches = bool(observed_identities) and all(
                value in model.accepted_response_models
                for value in observed_identities
            )
            if not identity_matches:
                raise PreflightError("response_model_identity_mismatch")
        else:
            identity_matches = None
        gateway_completion = gateway.wait_for_completion_report()
        gateway_stream_contract_complete = bool(
            gateway_completion["all_logical_requests_complete"]
        )
        if not gateway_stream_contract_complete:
            raise PreflightError("gateway_stream_contract_incomplete")
    except PreflightError as exc:
        code = str(exc)
        category = _failure_category(code)
    except (OSError, http.client.HTTPException, TimeoutError):
        code = "preflight_transport_failure"
        category = "transport"
    ok = code is None
    return PreflightReport(
        ok=ok,
        model_profile_id=model.model_profile_id,
        api_family_id=model.api_family_id,
        route_profile_id=route.route_profile_id,
        response_contract=route.response_contract,
        protocol=route.pi_api_protocol,
        logical_requests=logical_requests,
        first_http_status=first_status,
        second_http_status=second_status,
        tool_call_detected=tool_detected,
        tool_name_matches=tool_name_matches,
        tool_arguments_match=tool_arguments_match,
        reasoning_replay_required=model.reasoning.preserve_across_tool_turns,
        reasoning_signal_present=reasoning_present,
        reasoning_replayed=reasoning_replayed,
        followup_nonempty=followup_nonempty,
        response_identity_required=model.response_identity_required,
        response_identity_matches=identity_matches,
        gateway_stream_contract_complete=gateway_stream_contract_complete,
        failure_category=category,
        failure_code=code,
        elapsed_seconds=round(max(0.0, time.monotonic() - started), 6),
    )


def write_preflight_report(path: str | Path, report: PreflightReport) -> None:
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PreflightError("preflight_report_already_exists")
    encoded = (
        json.dumps(
            report.public_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _first_payload(route: RouteProfile, model: ModelProfile) -> dict[str, Any]:
    instruction = (
        "Call sieve_preflight_echo exactly once with nonce sieve-preflight-v1. "
        "Do not answer in prose before calling the tool."
    )
    tool = {
        "name": PREFLIGHT_TOOL,
        "description": "Fixed route preflight echo. Use only the supplied nonce.",
        "parameters": {
            "type": "object",
            "properties": {"nonce": {"type": "string"}},
            "required": ["nonce"],
            "additionalProperties": False,
        },
    }
    if route.pi_api_protocol == "openai-completions":
        return {
            "model": model.client_wire_model,
            "messages": [{"role": "user", "content": instruction}],
            "tools": [{"type": "function", "function": tool}],
            "tool_choice": {
                "type": "function",
                "function": {"name": PREFLIGHT_TOOL},
            },
            "parallel_tool_calls": False,
            "stream": True,
        }
    if route.pi_api_protocol == "openai-responses":
        return {
            "model": model.client_wire_model,
            "input": [{"role": "user", "content": instruction}],
            "tools": [{"type": "function", **tool}],
            "tool_choice": {"type": "function", "name": PREFLIGHT_TOOL},
            "parallel_tool_calls": False,
            "stream": True,
        }
    raise PreflightError("unsupported_preflight_protocol")


def _second_payload(
    route: RouteProfile,
    model: ModelProfile,
    first: _Turn,
) -> tuple[dict[str, Any], bool]:
    if not first.call_id or not first.call_name or first.call_arguments is None:
        raise PreflightError("tool_call_incomplete")
    tool_result = json.dumps(
        {"nonce": PREFLIGHT_NONCE, "ok": True},
        sort_keys=True,
        separators=(",", ":"),
    )
    final_instruction = "Acknowledge the successful tool result in one short sentence."
    if route.pi_api_protocol == "openai-completions":
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": first.call_id,
                    "type": "function",
                    "function": {
                        "name": first.call_name,
                        "arguments": first.call_arguments,
                    },
                }
            ],
        }
        replayed = False
        if model.reasoning.preserve_across_tool_turns:
            for key, value in first.chat_reasoning_fields.items():
                assistant[key] = value
            replayed = bool(first.chat_reasoning_fields)
        return (
            {
                "model": model.client_wire_model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call sieve_preflight_echo exactly once with nonce "
                            "sieve-preflight-v1."
                        ),
                    },
                    assistant,
                    {
                        "role": "tool",
                        "tool_call_id": first.call_id,
                        "name": first.call_name,
                        "content": tool_result,
                    },
                    {"role": "user", "content": final_instruction},
                ],
                "stream": True,
            },
            replayed,
        )
    if route.pi_api_protocol == "openai-responses":
        input_items: list[Mapping[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Call sieve_preflight_echo exactly once with nonce "
                    "sieve-preflight-v1."
                ),
            }
        ]
        replayed = False
        if model.reasoning.preserve_across_tool_turns:
            input_items.extend(first.responses_reasoning_items)
            replayed = bool(first.responses_reasoning_items)
        input_items.extend(
            [
                {
                    "type": "function_call",
                    "call_id": first.call_id,
                    "name": first.call_name,
                    "arguments": first.call_arguments,
                },
                {
                    "type": "function_call_output",
                    "call_id": first.call_id,
                    "output": tool_result,
                },
                {"role": "user", "content": final_instruction},
            ]
        )
        return (
            {
                "model": model.client_wire_model,
                "input": input_items,
                "stream": True,
            },
            replayed,
        )
    raise PreflightError("unsupported_preflight_protocol")


def _request(
    gateway: ScopedModelGateway,
    route: RouteProfile,
    payload: Mapping[str, Any],
) -> _Turn:
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        gateway.port,
        timeout=min(7200.0, gateway.model.request_timeout_seconds + 60.0),
    )
    body = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        connection.request(
            "POST",
            route.client_path,
            body=body,
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {gateway.capability_token}",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        raw = response.read(MAX_PREFLIGHT_RESPONSE_BYTES + 1)
        status = int(response.status)
    finally:
        connection.close()
    if len(raw) > MAX_PREFLIGHT_RESPONSE_BYTES:
        raise PreflightError("preflight_response_too_large")
    if not 200 <= status <= 299:
        return _Turn(status, None, False, None, None, None, {}, ())
    events = tuple(_event_values(raw))
    if route.pi_api_protocol == "openai-completions":
        return _parse_chat_turn(status, events)
    if route.pi_api_protocol == "openai-responses":
        return _parse_responses_turn(status, events)
    raise PreflightError("unsupported_preflight_protocol")


def _event_values(raw: bytes) -> Iterable[Mapping[str, Any]]:
    stripped = raw.lstrip()
    if stripped.startswith(b"{"):
        try:
            value = json.loads(stripped.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PreflightError("invalid_preflight_json") from exc
        if isinstance(value, dict):
            yield value
            return
        raise PreflightError("invalid_preflight_json_root")
    seen = False
    for line in raw.splitlines():
        current = line.strip()
        if not current.startswith(b"data:"):
            continue
        data = current[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PreflightError("invalid_preflight_sse") from exc
        if isinstance(value, dict):
            seen = True
            yield value
    if not seen:
        raise PreflightError("empty_preflight_stream")


def _parse_chat_turn(status: int, events: Iterable[Mapping[str, Any]]) -> _Turn:
    identity: str | None = None
    text_parts: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    reasoning_text: dict[str, list[str]] = {}
    reasoning_other: dict[str, Any] = {}
    for event in events:
        if isinstance(event.get("model"), str):
            identity = str(event["model"])
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                message = choice.get("message")
                delta = message if isinstance(message, dict) else {}
            content = delta.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            for key in ("reasoning_content", "reasoning", "thinking"):
                value = delta.get(key)
                if isinstance(value, str) and value:
                    reasoning_text.setdefault(key, []).append(value)
            for key in ("reasoning_details", "encrypted_content"):
                value = delta.get(key)
                if value not in (None, "", [], {}):
                    reasoning_other[key] = value
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for position, raw_call in enumerate(tool_calls):
                if not isinstance(raw_call, dict):
                    continue
                index = raw_call.get("index", position)
                if isinstance(index, bool) or not isinstance(index, int):
                    continue
                call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if isinstance(raw_call.get("id"), str):
                    call["id"] += raw_call["id"]
                function = raw_call.get("function")
                if isinstance(function, dict):
                    if isinstance(function.get("name"), str):
                        call["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        call["arguments"] += function["arguments"]
    first = calls[min(calls)] if calls else {}
    reasoning_fields: dict[str, Any] = {
        key: "".join(parts) for key, parts in reasoning_text.items()
    }
    reasoning_fields.update(reasoning_other)
    return _Turn(
        http_status=status,
        model_identity=identity,
        text_nonempty=bool("".join(text_parts).strip()),
        call_id=first.get("id") or None,
        call_name=first.get("name") or None,
        call_arguments=first.get("arguments") if first else None,
        chat_reasoning_fields=reasoning_fields,
        responses_reasoning_items=(),
    )


def _parse_responses_turn(
    status: int, events: Iterable[Mapping[str, Any]]
) -> _Turn:
    identity: str | None = None
    text_parts: list[str] = []
    call: dict[str, str] = {"id": "", "name": "", "arguments": ""}
    reasoning_items: dict[str, Mapping[str, Any]] = {}
    for event in events:
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("model"), str):
            identity = str(response["model"])
        if isinstance(event.get("model"), str):
            identity = str(event["model"])
        event_type = event.get("type")
        if event_type == "response.output_text.delta" and isinstance(
            event.get("delta"), str
        ):
            text_parts.append(str(event["delta"]))
        if event_type == "response.function_call_arguments.delta" and isinstance(
            event.get("delta"), str
        ):
            call["arguments"] += str(event["delta"])
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            if isinstance(item.get("call_id"), str):
                call["id"] = str(item["call_id"])
            if isinstance(item.get("name"), str):
                call["name"] = str(item["name"])
            if isinstance(item.get("arguments"), str) and item["arguments"]:
                call["arguments"] = str(item["arguments"])
        elif item_type == "reasoning":
            item_id = str(item.get("id") or len(reasoning_items))
            reasoning_items[item_id] = dict(item)
    return _Turn(
        http_status=status,
        model_identity=identity,
        text_nonempty=bool("".join(text_parts).strip()),
        call_id=call["id"] or None,
        call_name=call["name"] or None,
        call_arguments=call["arguments"] if call["id"] else None,
        chat_reasoning_fields={},
        responses_reasoning_items=tuple(reasoning_items.values()),
    )


def _arguments_match(value: str | None) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return parsed == {"nonce": PREFLIGHT_NONCE}


def _failure_category(code: str) -> str:
    if code.startswith("first_http_") or code.startswith("second_http_"):
        try:
            status = int(code.rsplit("_", 1)[1])
        except ValueError:
            return "transport"
        if status in {401, 403}:
            return "authentication_or_entitlement"
        if status == 429:
            return "rate_limit_or_capacity"
        if status >= 500 or status in {408, 409, 425}:
            return "transport"
        return "route_contract"
    if "identity" in code:
        return "model_identity"
    if "reasoning" in code:
        return "reasoning_replay"
    if "tool" in code or "followup" in code or "stream" in code:
        return "tool_call_contract"
    return "route_contract"


__all__ = [
    "PREFLIGHT_NONCE",
    "PREFLIGHT_TOOL",
    "PreflightError",
    "PreflightReport",
    "run_tool_call_preflight",
    "write_preflight_report",
]
