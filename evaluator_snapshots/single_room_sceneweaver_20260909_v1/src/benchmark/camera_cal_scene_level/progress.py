"""Progress JSONL formatting and persistence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
from typing import Any, Callable

from benchmark.camera_cal_scene_level.io import utc_now


PROGRESS_SCHEMA_VERSION = "camera_cal_scene_level_progress_v1"


def progress_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return str(value).replace("\n", " ")


def format_progress_record(record: dict[str, Any]) -> str:
    timestamp = str(record.get("timestamp") or "")
    clock = timestamp[11:19] if len(timestamp) >= 19 else timestamp
    case_id = str(record.get("case_id") or "run")
    event = str(record.get("event") or "progress")
    details = record.get("details")
    details = details if isinstance(details, dict) else {}
    preferred = (
        "phase",
        "metric",
        "group_id",
        "role",
        "call_type",
        "status",
        "api_call_number",
        "cumulative_api_calls",
        "duration_seconds",
        "evidence_count",
        "error_type",
    )
    fragments: list[str] = []
    for key in preferred:
        value = details.get(key)
        if value is not None:
            fragments.append(f"{key}={progress_value(value)}")
    if isinstance(details.get("tokens_usage"), dict):
        fragments.append("tokens=" + progress_value(details["tokens_usage"]))
    suffix = " " + " ".join(fragments) if fragments else ""
    return f"[{clock}] [{case_id}] {event}{suffix}"


class ProgressReporter:
    """Persist concise progress events and optionally mirror them to stdout."""

    def __init__(
        self,
        path: Path,
        *,
        terminal: bool = True,
        clock: Callable[[], str] | None = None,
        formatter: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.terminal = bool(terminal)
        self._clock = clock or utc_now
        self._formatter = formatter or format_progress_record
        self._lock = threading.Lock()

    def emit(
        self,
        event: str,
        *,
        case_id: str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        record = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "timestamp": self._clock(),
            "event": str(event),
            "case_id": str(case_id) if case_id else None,
            "details": deepcopy(details),
        }
        encoded = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
            if self.terminal:
                print(self._formatter(record), flush=True)
        return record
