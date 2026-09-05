#!/usr/bin/env python3
"""Project Pi's sanitized JSONL stream into a complete tool audit chain."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from isolated_exec import sanitize_json_log_value


ALLOWED_TOOLS = frozenset({"read", "write", "edit", "bash"})


class PiToolTranscriptError(RuntimeError):
    """Raised when an accepted Pi tool transcript cannot be proven complete."""


def project_pi_tool_transcript(
    *,
    source_path: str | Path,
    output_path: str | Path,
    require_complete: bool,
) -> dict[str, Any]:
    source = _real_file(source_path, "Pi JSONL process log")
    output = Path(output_path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise PiToolTranscriptError("refusing to overwrite Pi tool transcript")

    descriptor = _open_private_exclusive(output)
    previous_hash = "0" * 64
    event_count = 0
    started_count = 0
    ended_count = 0
    update_count = 0
    active: dict[str, tuple[str, str]] = {}
    completed_ids: set[str] = set()
    sequential = True
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with source.open("r", encoding="utf-8") as source_handle:
                for line_number, raw_line_with_newline in enumerate(
                    source_handle, start=1
                ):
                    raw_line = raw_line_with_newline.rstrip("\r\n")
                    if not raw_line:
                        continue
                    try:
                        record = json.loads(raw_line, parse_constant=_reject_constant)
                    except (UnicodeError, ValueError) as exc:
                        raise PiToolTranscriptError(
                            f"Pi JSONL contains invalid JSON at line {line_number}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise PiToolTranscriptError(
                            f"Pi JSONL root is not an object at line {line_number}"
                        )
                    kind = record.get("type")
                    if kind not in {
                        "tool_execution_start",
                        "tool_execution_update",
                        "tool_execution_end",
                    }:
                        continue
                    call_id = _text(record.get("toolCallId"), "toolCallId")
                    tool_name = _text(record.get("toolName"), "toolName")
                    if tool_name not in ALLOWED_TOOLS:
                        raise PiToolTranscriptError("Pi invoked an unregistered tool")
                    public_call_id = hashlib.sha256(
                        f"pi-tool-call-v1:{call_id}".encode("utf-8")
                    ).hexdigest()
                    event: dict[str, Any] = {
                        "schema_version": "sieve_pi_tool_event_v1",
                        "event": kind,
                        "source_line": line_number,
                        "tool_call_sha256": public_call_id,
                        "tool_name": tool_name,
                    }
                    if kind == "tool_execution_start":
                        if call_id in active or call_id in completed_ids:
                            raise PiToolTranscriptError("duplicate Pi tool-call start")
                        if active:
                            sequential = False
                        arguments = sanitize_json_log_value(record.get("args"))
                        arguments_hash = _canonical_hash(arguments)
                        active[call_id] = (tool_name, arguments_hash)
                        event["arguments"] = arguments
                        event["arguments_sha256"] = arguments_hash
                        started_count += 1
                    elif kind == "tool_execution_update":
                        if call_id not in active or active[call_id][0] != tool_name:
                            raise PiToolTranscriptError("orphan Pi tool-call update")
                        arguments = sanitize_json_log_value(record.get("args"))
                        if _canonical_hash(arguments) != active[call_id][1]:
                            raise PiToolTranscriptError("Pi tool arguments changed in flight")
                        partial = sanitize_json_log_value(record.get("partialResult"))
                        event["partial_result_sha256"] = _canonical_hash(partial)
                        event["partial_result_json_bytes"] = len(_canonical_bytes(partial))
                        update_count += 1
                    else:
                        if call_id not in active or active[call_id][0] != tool_name:
                            raise PiToolTranscriptError("orphan Pi tool-call end")
                        is_error = record.get("isError")
                        if not isinstance(is_error, bool):
                            raise PiToolTranscriptError("Pi tool result lacks boolean isError")
                        result = sanitize_json_log_value(record.get("result"))
                        event["is_error"] = is_error
                        event["result"] = result
                        event["result_sha256"] = _canonical_hash(result)
                        del active[call_id]
                        completed_ids.add(call_id)
                        ended_count += 1
                    previous_hash = _write_chained_event(handle, event, previous_hash)
                    event_count += 1

            complete = (
                sequential
                and not active
                and started_count == ended_count
                and started_count > 0
            )
            final_event = {
                "schema_version": "sieve_pi_tool_event_v1",
                "event": "transcript_terminal",
                "complete": complete,
                "sequential": sequential,
                "started_tool_calls": started_count,
                "ended_tool_calls": ended_count,
                "update_events": update_count,
                "open_tool_calls": len(active),
            }
            previous_hash = _write_chained_event(handle, final_event, previous_hash)
            event_count += 1
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.chmod(0o600)
        raise
    output.chmod(0o444)

    summary = {
        "schema_version": "sieve_pi_tool_transcript_summary_v1",
        "complete": complete,
        "sequential": sequential,
        "started_tool_calls": started_count,
        "ended_tool_calls": ended_count,
        "update_events": update_count,
        "transcript_event_count": event_count,
        "transcript_terminal_sha256": previous_hash,
        "source_process_log_sha256": _sha256_file(source),
        "raw_tool_call_ids_recorded": False,
        "hidden_reasoning_recorded": False,
    }
    if require_complete and not complete:
        raise PiToolTranscriptError("Pi tool transcript is incomplete or non-sequential")
    return summary


def verify_pi_tool_transcript(
    *,
    source_path: str | Path,
    transcript_path: str | Path,
    summary: Mapping[str, Any],
    require_complete: bool,
) -> dict[str, Any]:
    """Re-verify a projected transcript and its source-log binding."""

    source = _real_file(source_path, "Pi JSONL process log")
    transcript = _real_file(transcript_path, "Pi tool transcript")
    if summary.get("schema_version") != "sieve_pi_tool_transcript_summary_v1":
        raise PiToolTranscriptError("Pi tool transcript summary schema mismatch")
    if summary.get("source_process_log_sha256") != _sha256_file(source):
        raise PiToolTranscriptError("Pi tool transcript source hash mismatch")

    previous_hash = "0" * 64
    event_count = 0
    started_count = 0
    ended_count = 0
    update_count = 0
    active: dict[str, tuple[str, str]] = {}
    completed: set[str] = set()
    sequential = True
    terminal: dict[str, Any] | None = None
    saw_line = False
    with transcript.open("r", encoding="utf-8") as transcript_handle:
        for line_number, raw_line in enumerate(transcript_handle, start=1):
            saw_line = True
            line = raw_line.rstrip("\r\n")
            if not line:
                raise PiToolTranscriptError("Pi tool transcript contains an empty line")
            try:
                event = json.loads(line, parse_constant=_reject_constant)
            except (UnicodeError, ValueError) as exc:
                raise PiToolTranscriptError(
                    f"Pi tool transcript contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(event, dict):
                raise PiToolTranscriptError("Pi tool transcript event is not an object")
            observed_hash = event.get("event_sha256")
            if not isinstance(observed_hash, str) or len(observed_hash) != 64:
                raise PiToolTranscriptError("Pi tool transcript event hash is invalid")
            unhashed = dict(event)
            del unhashed["event_sha256"]
            if unhashed.get("previous_event_sha256") != previous_hash:
                raise PiToolTranscriptError("Pi tool transcript hash chain is discontinuous")
            if hashlib.sha256(_canonical_bytes(unhashed)).hexdigest() != observed_hash:
                raise PiToolTranscriptError("Pi tool transcript event hash mismatch")
            previous_hash = observed_hash
            event_count += 1

            kind = event.get("event")
            if kind == "transcript_terminal":
                if terminal is not None:
                    raise PiToolTranscriptError("Pi transcript terminal event is misplaced")
                terminal = event
                continue
            if terminal is not None or kind not in {
                "tool_execution_start",
                "tool_execution_update",
                "tool_execution_end",
            }:
                raise PiToolTranscriptError("Pi tool transcript event kind is invalid")
            call_id = _text(event.get("tool_call_sha256"), "tool_call_sha256")
            tool_name = _text(event.get("tool_name"), "tool_name")
            if tool_name not in ALLOWED_TOOLS:
                raise PiToolTranscriptError("Pi transcript contains an unregistered tool")
            if kind == "tool_execution_start":
                if call_id in active or call_id in completed:
                    raise PiToolTranscriptError("duplicate Pi transcript tool-call start")
                if active:
                    sequential = False
                arguments = event.get("arguments")
                arguments_hash = event.get("arguments_sha256")
                if arguments_hash != _canonical_hash(arguments):
                    raise PiToolTranscriptError("Pi transcript argument hash mismatch")
                active[call_id] = (tool_name, str(arguments_hash))
                started_count += 1
            elif kind == "tool_execution_update":
                if call_id not in active or active[call_id][0] != tool_name:
                    raise PiToolTranscriptError("orphan Pi transcript tool-call update")
                partial_hash = event.get("partial_result_sha256")
                partial_size = event.get("partial_result_json_bytes")
                if (
                    not isinstance(partial_hash, str)
                    or len(partial_hash) != 64
                    or isinstance(partial_size, bool)
                    or not isinstance(partial_size, int)
                    or partial_size < 0
                ):
                    raise PiToolTranscriptError("Pi transcript update metadata is invalid")
                update_count += 1
            else:
                if call_id not in active or active[call_id][0] != tool_name:
                    raise PiToolTranscriptError("orphan Pi transcript tool-call end")
                if not isinstance(event.get("is_error"), bool):
                    raise PiToolTranscriptError("Pi transcript result error flag is invalid")
                if event.get("result_sha256") != _canonical_hash(event.get("result")):
                    raise PiToolTranscriptError("Pi transcript result hash mismatch")
                del active[call_id]
                completed.add(call_id)
                ended_count += 1

    if not saw_line:
        raise PiToolTranscriptError("Pi tool transcript is empty")

    if terminal is None:
        raise PiToolTranscriptError("Pi tool transcript lacks a terminal event")
    complete = sequential and not active and started_count == ended_count and started_count > 0
    expected_terminal = {
        "complete": complete,
        "sequential": sequential,
        "started_tool_calls": started_count,
        "ended_tool_calls": ended_count,
        "update_events": update_count,
        "open_tool_calls": len(active),
    }
    for key, expected in expected_terminal.items():
        if terminal.get(key) != expected:
            raise PiToolTranscriptError("Pi transcript terminal aggregate mismatch")
    expected_summary = {
        "complete": complete,
        "sequential": sequential,
        "started_tool_calls": started_count,
        "ended_tool_calls": ended_count,
        "update_events": update_count,
        "transcript_event_count": event_count,
        "transcript_terminal_sha256": previous_hash,
        "raw_tool_call_ids_recorded": False,
        "hidden_reasoning_recorded": False,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise PiToolTranscriptError("Pi tool transcript summary mismatch")
    if require_complete and not complete:
        raise PiToolTranscriptError("Pi tool transcript is incomplete or non-sequential")
    return {
        "schema_version": "sieve_pi_tool_transcript_verification_v1",
        "complete": complete,
        "event_count": event_count,
        "transcript_terminal_sha256": previous_hash,
    }


def _write_chained_event(handle: Any, event: Mapping[str, Any], previous: str) -> str:
    chained = {**dict(event), "previous_event_sha256": previous}
    digest = hashlib.sha256(_canonical_bytes(chained)).hexdigest()
    chained["event_sha256"] = digest
    handle.write(_canonical_bytes(chained) + b"\n")
    return digest


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_private_exclusive(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    return descriptor


def _real_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise PiToolTranscriptError(f"{label} must be a real file")
    return path.resolve(strict=True)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise PiToolTranscriptError(f"invalid {label}")
    return value


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


__all__ = [
    "PiToolTranscriptError",
    "project_pi_tool_transcript",
    "verify_pi_tool_transcript",
]
