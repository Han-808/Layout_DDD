"""Small, dependency-free safety primitives; usable in offline tests."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Iterator

from . import VERSION


class ExecutionGuardError(RuntimeError):
    """An execution guard stopped work, not a scientific invalid verdict."""


class IntegrityError(ExecutionGuardError):
    pass


class StorageGuardError(ExecutionGuardError):
    pass


class AdmissionTimeout(ExecutionGuardError):
    pass


@dataclass(frozen=True)
class Policy:
    minimum_free_gib: float = 30.0
    room_slot_timeout_seconds: float = 3600.0
    room_timeout_seconds: float = 43200.0
    heartbeat_timeout_seconds: float = 180.0
    worker_timeout_seconds: float = 259200.0
    terminate_grace_seconds: float = 30.0
    checkpoint_enabled: bool = True
    max_request_attempts_per_room: int = 1024
    max_same_request_attempts: int = 18
    max_infra_failures_per_room: int = 12
    max_repeated_failure_attempts: int = 3
    circuit_failure_threshold: int = 3
    circuit_probe_limit: int = 3
    circuit_cooldown_seconds: float = 30.0
    circuit_max_cooldown_seconds: float = 300.0
    circuit_outage_timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if key == "checkpoint_enabled":
                if not isinstance(value, bool):
                    raise ValueError("checkpoint_enabled must be boolean")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            if key.startswith("max_") or key in {"circuit_failure_threshold", "circuit_probe_limit"}:
                if type(value) is not int:
                    raise ValueError(f"{key} must be an integer")
        if self.circuit_max_cooldown_seconds < self.circuit_cooldown_seconds:
            raise ValueError("maximum circuit cooldown cannot be shorter than initial cooldown")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError("JSON artifact must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError("JSON artifact must be an object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    """Private, crash-consistent replacement; always close/delete our temp."""
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def check_storage(path: Path, minimum_free_gib: float) -> None:
    candidate = path.resolve()
    while not candidate.exists():
        candidate = candidate.parent
    if shutil.disk_usage(candidate).free < minimum_free_gib * (1024 ** 3):
        raise StorageGuardError("free disk space below execution admission threshold")


def secret_values() -> tuple[str, ...]:
    return tuple(value for key, value in os.environ.items()
                 if len(value) >= 6 and re.search(
                     r"(?:CREDENTIAL|API_KEY|APP_KEY|MASTER_KEY|ACCESS_TOKEN|AUTH_TOKEN)$", key))


def redact(text: str) -> str:
    for secret in sorted(secret_values(), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+[^\s,;\"']+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"(https?://[^\s?\"']+)\?[^\s\"']+", r"\1?[REDACTED]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|app[_-]?key|authorization|credential|access_token)"
        r"[\"']?\s*[:=]\s*[\"']?)[^\s,}\"']+", r"\1[REDACTED]", text)
    return text


_DROP_KEYS = {"raw_response", "raw_request", "messages", "headers", "authorization",
              "api_key", "credential", "request_metadata", "raw_text", "prompt"}


def diagnostic_projection(value: Any, depth: int = 0) -> Any:
    """Preserve structural error/scope/field information, never raw exchanges."""
    if depth > 24:
        return "[DEPTH_LIMIT]"
    if isinstance(value, dict):
        return {str(k): diagnostic_projection(v, depth + 1)
                for k, v in value.items() if str(k).lower() not in _DROP_KEYS
                and not re.search(r"(?i)(secret|credential|api_key|access_token)", str(k))}
    if isinstance(value, (tuple, list)):
        return [diagnostic_projection(item, depth + 1) for item in value]
    if isinstance(value, str):
        return redact(value)[:2048]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[NONFINITE]"
    return f"[{type(value).__name__}]"


def exception_record(exc: BaseException) -> dict[str, Any]:
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen and len(chain) < 8:
        seen.add(id(exc))
        frames, tb = [], exc.__traceback__
        while tb is not None:
            frames.append({"file": tb.tb_frame.f_code.co_filename,
                           "line": tb.tb_lineno, "function": tb.tb_frame.f_code.co_name})
            tb = tb.tb_next
        message = str(exc)
        if type(exc).__name__.startswith("Endpoint"):
            match = re.search(r"HTTP\s+(\d{3})", message)
            message = f"HTTP {match[1]}" if match else "endpoint request failed; body omitted"
        record = {"error_type": type(exc).__name__, "error": redact(message)[:2048],
                  "frames": frames[-12:]}
        for name in ("errno", "returncode"):
            if type(getattr(exc, name, None)) is int:
                record[name] = getattr(exc, name)
        chain.append(record)
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return {"exception_chain": chain}


def best_effort_record(path: Path, value: Any) -> bool:
    try:
        atomic_json(path, diagnostic_projection(value))
        return True
    except Exception:
        # Do not leak the original payload/exception when diagnostics also fail.
        try:
            os.write(2, b"execution_diagnostic_write_failed\n")
        except OSError:
            pass
        return False


@contextmanager
def room_slot(root: Path, *, slots: int, timeout: float,
              work: dict[str, Any], cancelled: Callable[[], bool] = lambda: False,
              write: Callable = atomic_json) -> Iterator[None]:
    """flock is the authority. State JSON failure can never leak a slot."""
    if slots < 1 or timeout <= 0:
        raise ValueError("positive admission bounds required")
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    held = None
    while held is None:
        if cancelled():
            raise ExecutionGuardError("admission cancelled")
        for index in range(slots):
            fd = os.open(root / f"slot_{index}.lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            except BaseException:
                os.close(fd)
                raise
            held = index, fd
            break
        if held is None:
            if time.monotonic() >= deadline:
                raise AdmissionTimeout("global room admission deadline exceeded")
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    index, fd = held
    try:
        write(root / f"slot_{index}.json", {"status": "active", "pid": os.getpid(),
                                          "started_at": stamp(), **work})
        yield
    finally:
        # Release the kernel lock BEFORE the fallible observational write.
        os.close(fd)
        try:
            write(root / f"slot_{index}.json", {"status": "released", **work,
                                              "released_at": stamp()})
        except Exception:
            try:
                os.write(2, b"room_slot_release_record_failed; lock released\n")
            except OSError:
                pass


class Heartbeat:
    def __init__(self, path: Path, run_id: str, interval: float = 5.0) -> None:
        self.path, self.run_id, self.interval = path, run_id, interval
        self.lock = threading.Lock()
        self.rooms: dict[str, dict[str, Any]] = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self.emit()
        self.thread.start()
        return self

    def update(self, key: str, **fields: Any) -> None:
        with self.lock:
            self.rooms.setdefault(key, {}).update(fields, updated_at=time.time())

    def remove(self, key: str) -> None:
        with self.lock:
            self.rooms.pop(key, None)

    def emit(self) -> None:
        with self.lock:
            value = {"run_id": self.run_id, "pid": os.getpid(), "time": time.time(),
                     "rooms": dict(self.rooms)}
        best_effort_record(self.path, value)

    def _loop(self) -> None:
        while not self.stop.wait(self.interval):
            self.emit()

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=self.interval + 1)
        self.emit()


def worker_outcome(evaluation: Path, returncode: int, *, expected_model: str,
                   expected_rooms: int, expected_run_identity: str | None = None) -> dict[str, Any]:
    """A nonzero code alone NEVER authorizes launching the next dataset."""
    result = {"status": "fatal", "continue_lane": False, "returncode": returncode}
    try:
        run = read_json(evaluation / "run_manifest.json")
        if identity(run["identity"]) != run["identity_sha256"]:
            raise IntegrityError("campaign identity hash mismatch")
        if expected_run_identity and run["identity_sha256"] != expected_run_identity:
            raise IntegrityError("campaign identity changed during execution")
        if run["identity"]["model_order"] != [expected_model]:
            raise IntegrityError("unexpected worker model identity")
        terminal = read_json(evaluation / "terminal_manifest.json")
        if terminal.get("schema_version") != "non_rectangular_resilient_terminal_manifest_v1":
            raise IntegrityError("unsupported terminal manifest")
        if terminal.get("room_count") != expected_rooms:
            raise IntegrityError("terminal room count differs from selected work")
        if any(type(terminal.get(k)) is not int or terminal[k] < 0
               for k in ("room_count", "complete_room_count", "failed_room_count")):
            raise IntegrityError("terminal counters must be non-negative integers")
        if terminal.get("nonretryable_scene_failure_count", 0) or terminal.get("excluded_incomplete_scene_count", 0):
            raise IntegrityError("input preflight rejection is not a room-only partial result")
        rooms = list((evaluation / "models" / expected_model / "scenes").glob("*/rooms/*/summary.json"))
        if len(rooms) != expected_rooms:
            raise IntegrityError("missing room terminal summaries")
        counts = {"complete": 0, "failed": 0}
        for path in rooms:
            summary = read_json(path)
            status = summary.get("status")
            if status == "complete":
                pointer = read_json(path.parent / "room_report_selected.json")
                report_path = Path(pointer["room_report_path"])
                if report_path.is_symlink() or digest(report_path) != pointer["room_report_sha256"]:
                    raise IntegrityError("successful report hash mismatch")
                report = read_json(report_path)
                required = {"collision", "oob", "support", "scale_consistency", "style_consistency",
                            "object_pairing_consistency", "functional_consistency", "semantic_placement_consistency"}
                metrics = report.get("metrics")
                if (report.get("status") != "complete"
                        or report.get("schema_version") != "non_rectangular_complete_room_report_v1"
                        or report.get("room_id") != path.parent.name
                        or not isinstance(metrics, dict) or set(metrics) != required):
                    raise IntegrityError("selected report is not a complete matching room report")
                for name, metric in metrics.items():
                    score = metric.get("score") if isinstance(metric, dict) else None
                    if (not isinstance(metric, dict) or metric.get("metric") != name
                            or metric.get("status") != "complete" or isinstance(score, bool)
                            or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
                        raise IntegrityError("selected room metric contract invalid")
                counts["complete"] += 1
            elif status in {"failed_nonretryable", "failed_retry_exhausted"}:
                failure = summary.get("latest_failure") or {}
                if failure.get("stage") not in {"evaluation", "materialization"}:
                    raise IntegrityError("not a terminal room execution failure")
                if any(word in str(failure.get("category", "")) for word in
                       ("identity_drift", "configuration", "interrupted", "unclassified", "channel_unavailable")):
                    raise IntegrityError("fatal/unknown failure cannot be treated as normal partial")
                counts["failed"] += 1
            else:
                raise IntegrityError("pending/interrupted rooms do not prove terminal partial completion")
        if counts["complete"] != terminal.get("complete_room_count") or counts["failed"] != terminal.get("failed_room_count"):
            raise IntegrityError("terminal counters do not reconcile")
        if returncode == 0 and terminal.get("status") == "complete" and not counts["failed"]:
            return {**result, "status": "complete", "continue_lane": True, **counts}
        if returncode == 2 and terminal.get("status") == "failed" and counts["failed"]:
            return {**result, "status": "partial_failure", "continue_lane": True, **counts}
        raise IntegrityError("process return code and terminal manifest disagree")
    except Exception as exc:
        return {**result, "diagnostic": exception_record(exc)}
