#!/usr/bin/env python3
"""Run one Agent command under the arena's outer macOS Seatbelt boundary."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import tempfile
import threading
import time
from typing import Mapping, Sequence


TRUSTED_ROOT = Path(__file__).resolve().parent
ARENA_ROOT = TRUSTED_ROOT.parent
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
TOOL_SOCKET_ENV = "LAYOUT_DDD_AGENT_TOOL_SOCKET"
TOOL_TOKEN_ENV = "LAYOUT_DDD_AGENT_TOOL_TOKEN"
GATEWAY_TOKEN_ENV = "ARENA_MODEL_GATEWAY_TOKEN"
BASH_REGISTRY_FD_ENV = "SIEVE_BASH_REGISTRY_FD"
BASH_REGISTRY_ACK_FD_ENV = "SIEVE_BASH_REGISTRY_ACK_FD"
BASH_REGISTRY_NONCE_ENV = "SIEVE_BASH_REGISTRY_NONCE"
LOOPBACK_GATEWAY = re.compile(r"^localhost:([1-9][0-9]{0,4})$")
SYSTEM_EXEC_ROOTS = tuple(Path(item) for item in ("/bin", "/sbin", "/usr/bin", "/usr/sbin"))
HARNESS_EXTENSION = TRUSTED_ROOT / "pi_harness/sequential_tools_extension.ts"
PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_LOG_LINE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_LOG_BYTES = 64 * 1024 * 1024
MAX_WORKSPACE_BYTES = 2 * 1024 * 1024 * 1024
MAX_WORKSPACE_ENTRIES = 100_000
MAX_PROCESS_GROUP_MEMBERS = 128
RESOURCE_POLL_SECONDS = 0.10
WORKSPACE_POLL_SECONDS = 1.0
BASH_EXIT_GRACE_SECONDS = 0.50
BASH_EVENT_BUFFER_LIMIT = 64 * 1024
PROCESS_OPEN_FILES_LIMIT = 256
PROCESS_FILE_BLOCK_LIMIT = 2_097_152  # 1 GiB in POSIX 512-byte blocks.
PROCESS_USER_HEADROOM = MAX_PROCESS_GROUP_MEMBERS + 16
PROC_ALL_PIDS = 1
PROC_PIDTBSDINFO = 3
PROC_LIST_CAPACITY = 65_536
PROC_STATUS_ZOMBIE = 5


class _ProcBsdInfo(ctypes.Structure):
    """Darwin ``proc_bsdinfo``; start time provides a microsecond PID epoch."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


class IsolationError(RuntimeError):
    """Raised when isolation cannot be proven before process launch."""


@dataclass(frozen=True)
class IsolationResult:
    status: str
    returncode: int | None
    timed_out: bool
    elapsed_seconds: float
    stdout_bytes: int
    stderr_bytes: int
    isolation_backend: str = "macos_seatbelt_sandbox_exec_v1"
    filesystem_scope: str = "single_episode_workspace"
    network_mode: str = "database_socket_only"
    host_environment_inherited: bool = False
    logs_sanitized: bool = True
    hidden_reasoning_recorded: bool = False
    resource_limit_exceeded: bool = False
    resource_limit_kind: str | None = None
    workspace_bytes_observed: int = 0
    workspace_entries_observed: int = 0

    def public_dict(self) -> dict[str, object]:
        return {
            "schema_version": "sieve_isolated_agent_process_result_v2",
            **asdict(self),
        }


class _OutputQuota:
    """Atomically reserve a bounded amount of sanitized output."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.written = 0
        self.exceeded = threading.Event()
        self._lock = threading.Lock()

    def write(self, destination: object, data: bytes) -> None:
        with self._lock:
            if self.exceeded.is_set():
                return
            if self.written + len(data) > self.limit:
                self.exceeded.set()
                return
            destination.write(data)  # type: ignore[attr-defined]
            self.written += len(data)


@dataclass(frozen=True)
class _ProcessInfo:
    ppid: int
    pgid: int
    state: str
    birth: str

    @property
    def active(self) -> bool:
        return self.state != "Z"


@dataclass
class _BashInvocation:
    root_pid: int
    root_birth: str
    owned_pids: dict[int, str]
    exited_at: float | None = None


class _OwnedProcessTracker:
    """Track the Pi tree, including a descendant that changes PGID/session."""

    def __init__(
        self,
        leader_pid: int,
        initial_snapshot: Mapping[int, _ProcessInfo] | None = None,
    ) -> None:
        snapshot = _process_snapshot() if initial_snapshot is None else initial_snapshot
        leader = snapshot.get(leader_pid)
        if leader is None or not leader.active:
            raise IsolationError("isolated leader has no stable process identity")
        self.leader_pid = leader_pid
        self.leader_pgid = leader.pgid
        self.leader_birth = leader.birth
        # Bind ownership to (PID, birth time), never to a bare recyclable PID.
        self.owned_pids: dict[int, str] = {leader_pid: leader.birth}
        self.bash: dict[tuple[int, str], _BashInvocation] = {}
        self.event_error: str | None = None

    def register(
        self,
        event: str,
        pid: int,
        now: float,
        snapshot: Mapping[int, _ProcessInfo] | None = None,
    ) -> bool:
        if pid <= 1:
            self.event_error = "bash_registry_invalid_pid"
            return False
        if event == "start":
            if snapshot is None:
                self.event_error = "bash_registry_missing_start_snapshot"
                return False
            self.refresh(snapshot)
            info = snapshot.get(pid)
            if (
                info is None
                or not info.active
                or self.owned_pids.get(pid) != info.birth
            ):
                self.event_error = "bash_registry_unowned_start"
                return False
            identity = (pid, info.birth)
            if identity in self.bash or any(
                invocation.root_pid == pid and invocation.exited_at is None
                for invocation in self.bash.values()
            ):
                self.event_error = "bash_registry_duplicate_start"
                return False
            self.bash[identity] = _BashInvocation(
                pid,
                info.birth,
                {pid: info.birth},
            )
            return True
        unresolved = [
            invocation
            for invocation in self.bash.values()
            if invocation.root_pid == pid and invocation.exited_at is None
        ]
        if event != "exit" or len(unresolved) != 1:
            self.event_error = "bash_registry_unmatched_exit"
            return False
        invocation = unresolved[0]
        invocation.exited_at = now
        return True

    @staticmethod
    def _matches(
        identities: Mapping[int, str],
        pid: int,
        snapshot: Mapping[int, _ProcessInfo],
    ) -> bool:
        info = snapshot.get(pid)
        return (
            info is not None
            and info.active
            and identities.get(pid) == info.birth
        )

    @staticmethod
    def _claim(
        identities: dict[int, str], pid: int, info: _ProcessInfo
    ) -> bool:
        if identities.get(pid) == info.birth:
            return False
        identities[pid] = info.birth
        return True

    def refresh(self, snapshot: Mapping[int, _ProcessInfo]) -> None:
        # Preserve every birth-bound identity once attributed. A recycled PID
        # is not owned unless it is observed again as a descendant of a live,
        # birth-matching owned process.
        changed = True
        while changed:
            changed = False
            live_owned = {
                pid
                for pid in self.owned_pids
                if self._matches(self.owned_pids, pid, snapshot)
            }
            for pid, info in snapshot.items():
                if info.ppid in live_owned:
                    changed = self._claim(self.owned_pids, pid, info) or changed

        for invocation in self.bash.values():
            changed = True
            while changed:
                changed = False
                live_invocation = {
                    pid
                    for pid in invocation.owned_pids
                    if self._matches(invocation.owned_pids, pid, snapshot)
                }
                for pid, info in snapshot.items():
                    if info.ppid in live_invocation:
                        changed = (
                            self._claim(invocation.owned_pids, pid, info) or changed
                        )
                        self._claim(self.owned_pids, pid, info)

        # ``start_new_session=True`` gives the episode a process group whose
        # numeric identity is the birth-bound leader PID.  A short-lived
        # launcher can spawn a background child and exit between two polling
        # snapshots; ancestry alone then cannot discover the reparented child.
        # While that original group still exists, every active member belongs
        # to this episode.  If the leader PID is ever observed with another
        # birth time, the numeric PGID has been recycled and must not be
        # claimed (or signalled).
        observed_leader = snapshot.get(self.leader_pid)
        leader_identity_recycled = (
            observed_leader is not None
            and observed_leader.birth != self.leader_birth
        )
        if self.leader_pgid > 1 and not leader_identity_recycled:
            for pid, info in snapshot.items():
                if info.active and info.pgid == self.leader_pgid:
                    self._claim(self.owned_pids, pid, info)

        # A normal background child can outlive its Bash parent but cannot leave
        # the episode group under the benchmark-owned backend. Attribute every
        # remaining member of a currently live owned PGID even after reparenting.
        live_groups = {
            info.pgid
            for pid, info in snapshot.items()
            if self._matches(self.owned_pids, pid, snapshot) and info.pgid > 1
        }
        for pid, info in snapshot.items():
            if info.pgid in live_groups:
                self._claim(self.owned_pids, pid, info)

    def live_pids(self, snapshot: Mapping[int, _ProcessInfo]) -> set[int]:
        return {
            pid
            for pid in self.owned_pids
            if self._matches(self.owned_pids, pid, snapshot)
        }

    def live_count(self, snapshot: Mapping[int, _ProcessInfo]) -> int:
        return len(self.live_pids(snapshot))

    def residual_bash(self, snapshot: Mapping[int, _ProcessInfo], now: float) -> bool:
        for invocation in self.bash.values():
            if (
                invocation.exited_at is None
                or now - invocation.exited_at < BASH_EXIT_GRACE_SECONDS
            ):
                continue
            if any(
                (info := snapshot.get(pid)) is not None and info.active
                and invocation.owned_pids.get(pid) == info.birth
                for pid in invocation.owned_pids
                if pid != invocation.root_pid
            ):
                return True
        return False

    def live_groups(self, snapshot: Mapping[int, _ProcessInfo]) -> set[int]:
        groups: set[int] = set()
        for pid in self.live_pids(snapshot):
            info = snapshot.get(pid)
            if info is not None and info.pgid > 1:
                groups.add(info.pgid)
        return groups


class _BashEventStream:
    """Read atomic records emitted only by the pinned Pi extension."""

    def __init__(self, descriptor: int, ack_descriptor: int, nonce: str) -> None:
        self.descriptor = descriptor
        self.ack_descriptor = ack_descriptor
        self.nonce = nonce
        self.buffer = bytearray()
        self.eof = False
        os.set_blocking(descriptor, False)

    def drain(self, tracker: _OwnedProcessTracker) -> None:
        if self.eof or tracker.event_error is not None:
            return
        while True:
            try:
                chunk = os.read(self.descriptor, 4096)
            except BlockingIOError:
                break
            except OSError:
                tracker.event_error = "bash_registry_read_failure"
                break
            if not chunk:
                self.eof = True
                break
            self.buffer.extend(chunk)
            if len(self.buffer) > BASH_EVENT_BUFFER_LIMIT:
                tracker.event_error = "bash_registry_buffer_limit"
                break
            while b"\n" in self.buffer:
                raw, _, remainder = self.buffer.partition(b"\n")
                self.buffer = bytearray(remainder)
                self._record(raw, tracker)

    def _record(self, raw: bytes, tracker: _OwnedProcessTracker) -> None:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError):
            tracker.event_error = "bash_registry_malformed_event"
            return
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "nonce", "event", "pid"}
            or value.get("schema_version") != "sieve_bash_process_event_v1"
            or value.get("nonce") != self.nonce
            or value.get("event") not in {"start", "exit"}
            or isinstance(value.get("pid"), bool)
            or not isinstance(value.get("pid"), int)
        ):
            tracker.event_error = "bash_registry_invalid_event"
            return
        event = str(value["event"])
        pid = int(value["pid"])
        snapshot: Mapping[int, _ProcessInfo] | None = None
        if event == "start":
            try:
                snapshot = _process_snapshot()
            except IsolationError:
                tracker.event_error = "bash_registry_start_snapshot_failure"
                return
        if not tracker.register(
            event,
            pid,
            time.monotonic(),
            snapshot,
        ):
            return
        if event == "start":
            acknowledgement = (
                json.dumps(
                    {
                        "schema_version": "sieve_bash_process_ack_v1",
                        "nonce": self.nonce,
                        "pid": pid,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            try:
                os.write(self.ack_descriptor, acknowledgement)
            except OSError:
                tracker.event_error = "bash_registry_ack_failure"

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError:
            pass
        try:
            os.close(self.ack_descriptor)
        except OSError:
            pass


def run_isolated(
    *,
    workspace: str | Path,
    runtime_root: str | Path,
    command: Sequence[str],
    tool_socket: str | Path,
    tool_token: str,
    stdout_path: str | Path,
    stderr_path: str | Path,
    stdin_text: str,
    timeout_seconds: float,
    model_gateway: str | None = None,
    model_gateway_token: str | None = None,
    extra_environment: Mapping[str, str] | None = None,
    harness_extension: str | Path = HARNESS_EXTENSION,
) -> IsolationResult:
    if not SANDBOX_EXEC.is_file():
        raise IsolationError("macOS sandbox-exec is unavailable; refusing host execution")
    workspace_path = _episode_workspace(workspace)
    runtime_path = _real_directory(runtime_root, "runtime_root")
    raw_socket_path = Path(tool_socket).expanduser().absolute()
    if not raw_socket_path.exists() or raw_socket_path.is_symlink():
        raise IsolationError("tool socket is missing or linked")
    socket_path = raw_socket_path.resolve(strict=True)
    if not stat.S_ISSOCK(socket_path.stat().st_mode):
        raise IsolationError("tool socket is not a Unix-domain socket")
    _capability(tool_token, "tool capability token")
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise IsolationError("Agent command must be a non-empty argv array")
    if not isinstance(stdin_text, str):
        raise IsolationError("Agent stdin must be text")
    executable = Path(command[0]).expanduser()
    if not executable.is_absolute():
        raise IsolationError("Agent executable must be an absolute path")
    executable = executable.resolve(strict=True)
    if not executable.is_file():
        raise IsolationError("Agent executable is not a file")
    _validate_runtime_scope(runtime_path, executable, workspace_path)
    if not _executable_is_allowed(executable, runtime_path, workspace_path):
        raise IsolationError("Agent executable is outside pinned execution roots")
    argv = [str(executable), *command[1:]]

    gateway = _validate_gateway(model_gateway)
    if (gateway is None) != (model_gateway_token is None):
        raise IsolationError("gateway endpoint and capability token must be supplied together")
    if model_gateway_token is not None:
        _capability(model_gateway_token, "model-gateway capability token")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise IsolationError("timeout_seconds must be finite and positive")
    extension_path = _real_file(harness_extension, "harness extension")
    expected_extension = _real_file(HARNESS_EXTENSION, "pinned harness extension")
    if extension_path != expected_extension:
        raise IsolationError("only the pinned harness extension may be exposed")
    policy = TRUSTED_ROOT / "policies" / (
        "scoped_gateway.sb" if gateway is not None else "offline.sb"
    )
    if not policy.is_file() or policy.is_symlink():
        raise IsolationError("Seatbelt policy is missing or linked")

    home = workspace_path / ".home"
    temporary = workspace_path / ".tmp"
    _workspace_child_directory(home, workspace_path, ".home")
    _workspace_child_directory(temporary, workspace_path, ".tmp")
    if any(temporary.iterdir()):
        raise IsolationError("episode .tmp must be empty before launch")
    environment = _clean_environment(
        workspace=workspace_path,
        home=home,
        temporary=temporary,
        tool_socket=socket_path,
        tool_token=tool_token,
        gateway_token=model_gateway_token,
        extra=extra_environment,
    )

    bash_registry_read: int | None = None
    bash_registry_write: int | None = None
    bash_registry_ack_read: int | None = None
    bash_registry_ack_write: int | None = None
    bash_registry_nonce: str | None = None
    if _uses_pinned_pi_extension(argv, runtime_path, extension_path):
        bash_registry_read, bash_registry_write = os.pipe()
        bash_registry_ack_read, bash_registry_ack_write = os.pipe()
        bash_registry_nonce = secrets.token_hex(32)
        environment[BASH_REGISTRY_FD_ENV] = str(bash_registry_write)
        environment[BASH_REGISTRY_ACK_FD_ENV] = str(bash_registry_ack_read)
        environment[BASH_REGISTRY_NONCE_ENV] = bash_registry_nonce

    sandbox_command = [
        str(SANDBOX_EXEC),
        "-f",
        str(policy),
        "-D",
        f"WORKSPACE={workspace_path}",
        "-D",
        f"RUNTIME_ROOT={runtime_path}",
        "-D",
        f"TOOL_SOCKET={socket_path}",
        "-D",
        f"HARNESS_EXTENSION={extension_path}",
    ]
    if gateway is not None:
        sandbox_command.extend(["-D", f"MODEL_GATEWAY={gateway}"])
    sandbox_command.extend(argv)
    # A dedicated wrapper applies inherited POSIX resource ceilings without a
    # Python preexec_fn (unsafe here because the DB and gateway own threads).
    # RLIMIT_NPROC is per host user on macOS, not per process tree.  Bind its
    # ceiling to the observed host baseline plus narrowly bounded episode
    # headroom; the independent PGID monitor below enforces the exact per-run
    # maximum.  A fixed low value would make ordinary child tools fail merely
    # because unrelated applications owned by the same user are running.
    user_process_ceiling = _user_process_count() + PROCESS_USER_HEADROOM
    resource_command = [
        "/bin/zsh",
        "-fc",
        (
            "set -e; "
            f"ulimit -n {PROCESS_OPEN_FILES_LIMIT}; "
            f"ulimit -u {user_process_ceiling}; "
            "ulimit -c 0; "
            f"ulimit -f {PROCESS_FILE_BLOCK_LIMIT}; "
            'exec "$@"'
        ),
        "sieve-resource-wrapper",
        *sandbox_command,
    ]

    stdout = _exclusive_log(stdout_path)
    stderr = _exclusive_log(stderr_path)
    secrets_to_remove = tuple(
        item
        for item in (tool_token, model_gateway_token)
        if isinstance(item, str) and item
    )
    started = time.monotonic()
    timed_out = False
    resource_limit_kind: str | None = None
    returncode: int | None = None
    reader_errors: list[BaseException] = []
    reader_failure = threading.Event()
    log_quota = _OutputQuota(MAX_TOTAL_LOG_BYTES)
    workspace_bytes = 0
    workspace_entries = 0
    event_stream: _BashEventStream | None = None
    tracker: _OwnedProcessTracker | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        with stdout, stderr:
            try:
                process = subprocess.Popen(
                    resource_command,
                    cwd=workspace_path,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    close_fds=True,
                    pass_fds=(bash_registry_write, bash_registry_ack_read)
                    if bash_registry_write is not None
                    and bash_registry_ack_read is not None
                    else (),
                )
            finally:
                if bash_registry_write is not None:
                    os.close(bash_registry_write)
                    bash_registry_write = None
                if bash_registry_ack_read is not None:
                    os.close(bash_registry_ack_read)
                    bash_registry_ack_read = None
            if process.stdout is None or process.stderr is None or process.stdin is None:
                raise IsolationError("isolated process pipes were not created")
            initial_snapshot = _process_snapshot()
            tracker = _OwnedProcessTracker(process.pid, initial_snapshot)
            if (
                bash_registry_read is not None
                and bash_registry_ack_write is not None
                and bash_registry_nonce is not None
            ):
                event_stream = _BashEventStream(
                    bash_registry_read,
                    bash_registry_ack_write,
                    bash_registry_nonce,
                )
                bash_registry_read = None
                bash_registry_ack_write = None
            readers = [
                threading.Thread(
                    target=_stream_sanitized_log,
                    args=(
                        process.stdout,
                        stdout,
                        secrets_to_remove,
                        reader_errors,
                        reader_failure,
                        log_quota,
                    ),
                    name="sieve-agent-stdout-sanitizer",
                    daemon=True,
                ),
                threading.Thread(
                    target=_stream_sanitized_log,
                    args=(
                        process.stderr,
                        stderr,
                        secrets_to_remove,
                        reader_errors,
                        reader_failure,
                        log_quota,
                    ),
                    name="sieve-agent-stderr-sanitizer",
                    daemon=True,
                ),
            ]
            for reader in readers:
                reader.start()
            try:
                try:
                    process.stdin.write(stdin_text.encode("utf-8"))
                    process.stdin.flush()
                except BrokenPipeError:
                    pass
                finally:
                    process.stdin.close()
                deadline = started + float(timeout_seconds)
                next_workspace_check = started
                next_process_check = started
                while process.poll() is None:
                    now = time.monotonic()
                    if event_stream is not None:
                        event_stream.drain(tracker)
                    if tracker.event_error is not None:
                        resource_limit_kind = tracker.event_error
                        _terminate_owned_processes(process, tracker)
                        break
                    if now >= deadline:
                        timed_out = True
                        _terminate_owned_processes(process, tracker)
                        break
                    if log_quota.exceeded.is_set():
                        resource_limit_kind = "sanitized_log_bytes"
                        _terminate_owned_processes(process, tracker)
                        break
                    if reader_failure.is_set():
                        resource_limit_kind = "log_sanitizer_failure"
                        _terminate_owned_processes(process, tracker)
                        break
                    if now >= next_process_check:
                        snapshot = _process_snapshot()
                        tracker.refresh(snapshot)
                        if tracker.live_count(snapshot) > MAX_PROCESS_GROUP_MEMBERS:
                            resource_limit_kind = "process_group_members"
                            _terminate_owned_processes(process, tracker)
                            break
                        if tracker.residual_bash(snapshot, now):
                            resource_limit_kind = "residual_bash_process"
                            _terminate_owned_processes(process, tracker)
                            break
                        next_process_check = now + RESOURCE_POLL_SECONDS
                    if now >= next_workspace_check:
                        workspace_bytes, workspace_entries = _workspace_usage(
                            workspace_path
                        )
                        if workspace_bytes > MAX_WORKSPACE_BYTES:
                            resource_limit_kind = "workspace_bytes"
                            _terminate_owned_processes(process, tracker)
                            break
                        if workspace_entries > MAX_WORKSPACE_ENTRIES:
                            resource_limit_kind = "workspace_entries"
                            _terminate_owned_processes(process, tracker)
                            break
                        next_workspace_check = now + WORKSPACE_POLL_SECONDS
                    time.sleep(RESOURCE_POLL_SECONDS)
                if process.poll() is None:
                    process.wait(timeout=5.0)
                # Drain records written immediately before Pi crossed its exit
                # boundary, then take a fresh ownership snapshot. A zero-exit
                # leader is not success while any owned descendant can still
                # race normalization or retain the database capability.
                if event_stream is not None:
                    drain_deadline = time.monotonic() + BASH_EXIT_GRACE_SECONDS
                    while time.monotonic() < drain_deadline and not event_stream.eof:
                        event_stream.drain(tracker)
                        if tracker.event_error is not None:
                            break
                        time.sleep(0.02)
                snapshot = _process_snapshot()
                tracker.refresh(snapshot)
                if tracker.event_error is not None:
                    if not timed_out and resource_limit_kind is None:
                        resource_limit_kind = tracker.event_error
                elif (
                    process.returncode == 0
                    and any(item.exited_at is None for item in tracker.bash.values())
                ):
                    if not timed_out and resource_limit_kind is None:
                        resource_limit_kind = "bash_registry_incomplete"
                if tracker.live_pids(snapshot):
                    if not timed_out and resource_limit_kind is None:
                        resource_limit_kind = "residual_process_group"
                    _terminate_owned_processes(process, tracker)
            finally:
                if tracker is not None:
                    snapshot = _process_snapshot()
                    tracker.refresh(snapshot)
                    if process.poll() is None or tracker.live_pids(snapshot):
                        _terminate_owned_processes(process, tracker)
            for reader in readers:
                reader.join(timeout=10.0)
            if any(reader.is_alive() for reader in readers):
                raise IsolationError("log sanitizer did not reach end-of-stream")
            if reader_errors:
                raise IsolationError(
                    f"streaming log sanitizer failed: {type(reader_errors[0]).__name__}"
                )
            # The leader can emit a final burst and create files after the last
            # polling tick but before wait() observes exit. Recompute both
            # quotas only after the sanitizer readers reached EOF; stale
            # pre-exit counters must never turn an over-limit run into success.
            if log_quota.exceeded.is_set() and resource_limit_kind is None:
                resource_limit_kind = "sanitized_log_bytes"
            workspace_bytes, workspace_entries = _workspace_usage(workspace_path)
            if workspace_bytes > MAX_WORKSPACE_BYTES and resource_limit_kind is None:
                resource_limit_kind = "workspace_bytes"
            if (
                workspace_entries > MAX_WORKSPACE_ENTRIES
                and resource_limit_kind is None
            ):
                resource_limit_kind = "workspace_entries"
            returncode = process.returncode
    except OSError as exc:
        raise IsolationError(f"isolated process launch failed: {type(exc).__name__}") from exc
    finally:
        cleanup_error: IsolationError | None = None
        if process is not None:
            try:
                if tracker is None:
                    _terminate_untracked_process_group(process)
                else:
                    snapshot = _process_snapshot()
                    tracker.refresh(snapshot)
                    if process.poll() is None or tracker.live_pids(snapshot):
                        _terminate_owned_processes(process, tracker)
            except IsolationError as exc:
                cleanup_error = exc
        if event_stream is not None:
            event_stream.close()
        if bash_registry_read is not None:
            os.close(bash_registry_read)
        if bash_registry_write is not None:
            os.close(bash_registry_write)
        if bash_registry_ack_read is not None:
            os.close(bash_registry_ack_read)
        if bash_registry_ack_write is not None:
            os.close(bash_registry_ack_write)
        if cleanup_error is not None:
            raise cleanup_error
    elapsed = max(0.0, time.monotonic() - started)
    out_path = Path(stdout_path)
    err_path = Path(stderr_path)
    # A second pass is defense in depth.  Raw process bytes were never written
    # to disk: both pipes were sanitized line-by-line before the first write.
    sanitize_process_logs((out_path, err_path), secrets_to_remove)
    if timed_out:
        status = "ambiguous_timeout"
    elif resource_limit_kind is not None:
        status = "resource_limit_exceeded"
    elif returncode == 0:
        status = "exited_zero"
    else:
        status = "process_failure"
    return IsolationResult(
        status=status,
        returncode=returncode,
        timed_out=timed_out,
        elapsed_seconds=elapsed,
        stdout_bytes=out_path.stat().st_size,
        stderr_bytes=err_path.stat().st_size,
        resource_limit_exceeded=resource_limit_kind is not None,
        resource_limit_kind=resource_limit_kind,
        workspace_bytes_observed=workspace_bytes,
        workspace_entries_observed=workspace_entries,
        network_mode="scoped_loopback_model_gateway"
        if gateway is not None
        else "database_socket_only",
    )


def _clean_environment(
    *,
    workspace: Path,
    home: Path,
    temporary: Path,
    tool_socket: Path,
    tool_token: str,
    gateway_token: str | None,
    extra: Mapping[str, str] | None,
) -> dict[str, str]:
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local/share"),
        "AGENT_CONFIG_HOME": str(home / ".agent-config"),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "RUBYOPT": "--disable-gems",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "PI_CODING_AGENT_DIR": str(home / ".pi/agent"),
        "PI_OFFLINE": "1",
        "PI_TELEMETRY": "0",
        "ARENA_WORKSPACE": str(workspace),
        TOOL_SOCKET_ENV: str(tool_socket),
        TOOL_TOKEN_ENV: tool_token,
    }
    if gateway_token is not None:
        environment[GATEWAY_TOKEN_ENV] = gateway_token
    allowed_extra = {
        "TERM",
        "COLORTERM",
        "ARENA_AGENT_ID",
        "ARENA_MODEL_ID",
        "ARENA_RUN_ID",
    }
    for name, value in dict(extra or {}).items():
        if name not in allowed_extra or not isinstance(value, str):
            raise IsolationError(f"extra environment is not allowlisted: {name}")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise IsolationError(f"extra environment contains control data: {name}")
        environment[name] = value
    return environment


def _uses_pinned_pi_extension(
    argv: Sequence[str], runtime: Path, extension: Path
) -> bool:
    """Only the pinned Pi process receives the trusted Bash event descriptor."""

    try:
        extension_index = argv.index("--extension")
        tools_index = argv.index("--tools")
    except ValueError:
        return False
    if extension_index + 1 >= len(argv) or tools_index + 1 >= len(argv):
        return False
    if argv[extension_index + 1] != str(extension):
        return False
    if argv[tools_index + 1] != "read,write,edit,bash":
        return False
    if "--no-extensions" not in argv or "--no-approve" not in argv:
        return False
    executable = Path(argv[0]).resolve()
    expected_node = (runtime / "bin/node").resolve()
    return executable == expected_node


def _workspace_usage(workspace: Path) -> tuple[int, int]:
    """Measure one workspace without following links or reading file bodies."""

    total_bytes = 0
    entries = 0
    pending = [workspace]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    entries += 1
                    total_bytes += max(0, int(info.st_size))
                    if (
                        total_bytes > MAX_WORKSPACE_BYTES
                        or entries > MAX_WORKSPACE_ENTRIES
                    ):
                        return total_bytes, entries
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
    except OSError as exc:
        raise IsolationError(
            f"workspace resource accounting failed: {type(exc).__name__}"
        ) from exc
    return total_bytes, entries


def _process_snapshot() -> dict[int, _ProcessInfo]:
    """Return a high-resolution, birth-bound Darwin process snapshot.

    ``ps lstart`` is only second-granular, which is insufficient when a long
    Agent episode can observe PID reuse. ``proc_bsdinfo`` supplies the process
    start microsecond without reading argv or environments.
    """

    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        libproc.proc_listpids.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        libproc.proc_listpids.restype = ctypes.c_int
        libproc.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        libproc.proc_pidinfo.restype = ctypes.c_int
        pid_array_type = ctypes.c_int * PROC_LIST_CAPACITY
        pids = pid_array_type()
        returned_bytes = libproc.proc_listpids(
            PROC_ALL_PIDS,
            0,
            pids,
            ctypes.sizeof(pids),
        )
    except (OSError, ValueError, TypeError) as exc:
        raise IsolationError(
            f"process-group accounting failed: {type(exc).__name__}"
        ) from exc
    if returned_bytes <= 0 or returned_bytes >= ctypes.sizeof(pids):
        raise IsolationError("process-group accounting exceeded its fixed capacity")
    snapshot: dict[int, _ProcessInfo] = {}
    for pid in pids[: returned_bytes // ctypes.sizeof(ctypes.c_int)]:
        if pid <= 0:
            continue
        info = _ProcBsdInfo()
        observed = libproc.proc_pidinfo(
            pid,
            PROC_PIDTBSDINFO,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if observed != ctypes.sizeof(info) or int(info.pbi_pid) != pid:
            continue
        snapshot[pid] = _ProcessInfo(
            ppid=int(info.pbi_ppid),
            pgid=int(info.pbi_pgid),
            state="Z" if int(info.pbi_status) == PROC_STATUS_ZOMBIE else "?",
            birth=f"{int(info.pbi_start_tvsec)}:{int(info.pbi_start_tvusec):06d}",
        )
    if not snapshot:
        raise IsolationError("process-group accounting returned no processes")
    return snapshot


def _process_group_members(process_group: int) -> int:
    """Count active members of a process group using one sanitized snapshot."""

    return sum(
        1
        for info in _process_snapshot().values()
        if info.pgid == process_group and info.active
    )


def _user_process_count() -> int:
    """Count the current macOS user's processes before applying RLIMIT_NPROC."""

    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "uid="],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IsolationError(
            f"user-process accounting failed: {type(exc).__name__}"
        ) from exc
    uid = os.getuid()
    count = 0
    for line in result.stdout.splitlines():
        try:
            observed = int(line.strip())
        except ValueError:
            continue
        if observed == uid:
            count += 1
    if count < 1:
        raise IsolationError("user-process accounting returned no owner process")
    return count


def _validate_gateway(value: str | None) -> str | None:
    if value is None:
        return None
    match = LOOPBACK_GATEWAY.fullmatch(value)
    if match is None or int(match.group(1)) > 65535:
        raise IsolationError("model gateway policy address must be localhost:PORT")
    return value


def _real_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise IsolationError(f"{label} must be a real directory")
    return path.resolve()


def _real_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise IsolationError(f"{label} must be a real file")
    return path.resolve(strict=True)


def _episode_workspace(value: str | Path) -> Path:
    path = _real_directory(value, "workspace")
    episodes = (ARENA_ROOT / "episodes").resolve(strict=True)
    try:
        relative = path.relative_to(episodes)
    except ValueError as exc:
        raise IsolationError("workspace is outside the arena episodes root") from exc
    if len(relative.parts) != 4 or relative.parts[-1] != "workspace":
        raise IsolationError(
            "workspace must be episodes/<agent>/<scene>/<run>/workspace"
        )
    for identity in relative.parts[:3]:
        if PORTABLE_ID.fullmatch(identity) is None:
            raise IsolationError("episode path contains a non-portable identity")
    cursor = episodes
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink() or not cursor.is_dir():
            raise IsolationError("episode workspace path contains a linked component")
    return path


def _workspace_child_directory(path: Path, workspace: Path, label: str) -> None:
    if path.parent != workspace or not path.is_dir() or path.is_symlink():
        raise IsolationError(f"episode {label} must be a real direct child directory")


def _capability(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or not 32 <= len(value) <= 4096
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise IsolationError(f"{label} is invalid")


def _executable_is_allowed(executable: Path, runtime: Path, workspace: Path) -> bool:
    for root in (*SYSTEM_EXEC_ROOTS, runtime, workspace):
        try:
            executable.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _validate_runtime_scope(runtime: Path, executable: Path, workspace: Path) -> None:
    try:
        executable.relative_to(workspace)
    except ValueError:
        pass
    else:
        if runtime not in SYSTEM_EXEC_ROOTS:
            _reject_broad_runtime(runtime)
        return
    for system_root in SYSTEM_EXEC_ROOTS:
        try:
            executable.relative_to(system_root)
        except ValueError:
            continue
        if runtime != system_root:
            raise IsolationError("system executable runtime_root must be its exact bin root")
        return
    try:
        relative = executable.relative_to(runtime)
    except ValueError as exc:
        raise IsolationError("pinned Agent executable is outside runtime_root") from exc
    if len(relative.parts) > 4:
        raise IsolationError("runtime_root is broader than the pinned Agent bundle")
    _reject_broad_runtime(runtime)


def _reject_broad_runtime(runtime: Path) -> None:
    try:
        ARENA_ROOT.resolve().relative_to(runtime)
    except ValueError:
        pass
    else:
        raise IsolationError("runtime_root must not contain the arena or repository")
    home = Path.home().resolve()
    if runtime in {Path("/").resolve(), home, home.parent}:
        raise IsolationError("runtime_root must not expose a broad host directory")


def _exclusive_log(value: str | Path):
    path = Path(value).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise IsolationError(f"refusing to overwrite process log: {path.name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise IsolationError(
            f"cannot create private process log: {type(exc).__name__}"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise


def _stream_sanitized_log(
    source: object,
    destination: object,
    secrets: Sequence[str],
    errors: list[BaseException],
    failure: threading.Event,
    quota: _OutputQuota,
) -> None:
    """Stream bounded lines from a process pipe through the sanitizer."""

    buffer = bytearray()
    discarding_oversize = False
    try:
        while True:
            chunk = source.read(64 * 1024)  # type: ignore[attr-defined]
            if not chunk:
                break
            for byte in chunk:
                if discarding_oversize:
                    if byte == 10:
                        quota.write(destination, b"[REDACTED_OVERSIZE_LOG_LINE]\n")
                        discarding_oversize = False
                    continue
                buffer.append(byte)
                if byte == 10:
                    _write_sanitized_line(
                        destination, bytes(buffer), secrets, quota=quota
                    )
                    buffer.clear()
                elif len(buffer) > MAX_LOG_LINE_BYTES:
                    buffer.clear()
                    discarding_oversize = True
        if discarding_oversize:
            quota.write(destination, b"[REDACTED_OVERSIZE_LOG_LINE]\n")
        elif buffer:
            _write_sanitized_line(destination, bytes(buffer), secrets, quota=quota)
        destination.flush()  # type: ignore[attr-defined]
        os.fsync(destination.fileno())  # type: ignore[attr-defined]
    except BaseException as exc:  # recorded and surfaced by the supervisor
        errors.append(exc)
        failure.set()
    finally:
        try:
            source.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def _write_sanitized_line(
    destination: object,
    raw_line: bytes,
    secrets: Sequence[str],
    *,
    quota: _OutputQuota | None = None,
) -> None:
    text = raw_line.decode("utf-8", errors="replace")
    had_newline = text.endswith("\n")
    line = text[:-1] if had_newline else text
    if line.endswith("\r"):
        line = line[:-1]
    sanitized = _sanitize_log_line(line, secrets)
    encoded = (sanitized + ("\n" if had_newline else "")).encode("utf-8")
    if quota is None:
        destination.write(encoded)  # type: ignore[attr-defined]
    else:
        quota.write(destination, encoded)


def sanitize_process_logs(paths: Sequence[Path], secrets: Sequence[str]) -> None:
    """Remove capabilities, hidden reasoning, routing data, and request IDs.

    Pi keeps provider reasoning in its in-memory turn state when a route needs
    it for tool-call replay.  Its JSON-mode process stream is not that state and
    is a public-ish host artifact, so reasoning blocks and routing identifiers
    are irreversibly removed before the run result is published.
    """

    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise IsolationError("process log disappeared before sealing")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.sanitize-",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as destination:
                with path.open("rb") as source:
                    for raw_line in source:
                        _write_sanitized_line(destination, raw_line, secrets)
                    destination.flush()
                    os.fsync(destination.fileno())
            os.replace(temporary, path)
            path.chmod(0o600)
        finally:
            if temporary.exists():
                temporary.unlink()


_SENSITIVE_EXACT_KEYS = frozenset(
    {
        "access_token",
        "api_token",
        "bearer_token",
        "capability_token",
        "token",
    }
)


_SENSITIVE_KEY_FRAGMENTS = (
    "authorization",
    "api_key",
    "apikey",
    "credential",
    "encrypted_content",
    "endpoint",
    "reasoning",
    "request_id",
    "requestid",
    "response_id",
    "responseid",
    "session_id",
    "sessionid",
    "signature",
    "thinking",
    "thought",
    "url",
)


_REASONING_COUNT_KEYS = frozenset(
    {
        "reasoning_token_count",
        "reasoning_tokens",
        "reasoningtokencount",
        "reasoningtokens",
    }
)


def _sanitize_log_line(line: str, secrets: Sequence[str]) -> str:
    value = line
    for secret in secrets:
        value = value.replace(secret, "[REDACTED_EPISODE_CAPABILITY]")
    try:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    except (UnicodeError, ValueError):
        return _sanitize_plain_log_line(value)
    return json.dumps(
        _sanitize_json_log_value(parsed),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _sanitize_json_log_value(
    value: object, *, parent_key: str | None = None
) -> object:
    if isinstance(value, list):
        return [
            _sanitize_json_log_value(item, parent_key=parent_key) for item in value
        ]
    if isinstance(value, str):
        return _sanitize_plain_log_line(value)
    if not isinstance(value, dict):
        return value
    event_type = value.get("type")
    if isinstance(event_type, str) and any(
        fragment in event_type.lower()
        for fragment in ("reasoning", "thinking", "thought")
    ):
        return {
            "type": event_type,
            "redacted": "hidden_reasoning_event",
        }
    output: dict[str, object] = {}
    for raw_key, child in value.items():
        key = str(raw_key)
        normalized = key.lower().replace("-", "_")
        if (
            normalized == "reasoning"
            and parent_key == "usage"
            and isinstance(child, int)
            and not isinstance(child, bool)
            and child >= 0
        ):
            output[key] = child
            continue
        if normalized in _REASONING_COUNT_KEYS and (
            isinstance(child, int) and not isinstance(child, bool) and child >= 0
        ):
            output[key] = child
            continue
        if normalized in _SENSITIVE_EXACT_KEYS or any(
            fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS
        ):
            output[key] = "[REDACTED_SENSITIVE_FIELD]"
        else:
            output[key] = _sanitize_json_log_value(
                child,
                parent_key=normalized,
            )
    return output


def sanitize_json_log_value(value: object) -> object:
    """Public wrapper used by trusted artifact projectors."""

    return _sanitize_json_log_value(value)


def _sanitize_plain_log_line(line: str) -> str:
    lowered = line.lower()
    if any(
        marker in lowered
        for marker in (
            "reasoning_content",
            "reasoning_details",
            "thinking_delta",
            "thinking_start",
            "thinking_end",
            "encrypted_content",
        )
    ):
        return "[REDACTED_HIDDEN_REASONING_EVENT]"
    value = re.sub(r"https?://[^\s\"']+", "[REDACTED_URL]", line)
    value = re.sub(
        r"(?i)\b(?:req(?:uest)?|resp(?:onse)?|session)[_-][A-Za-z0-9._~-]{4,}",
        "[REDACTED_REQUEST_OR_SESSION_ID]",
        value,
    )
    value = re.sub(
        r"(?i)(authorization|api[_-]?key|credential|request[_-]?id|session[_-]?id)"
        r"\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        value,
    )
    return value


def _terminate_owned_processes(
    process: subprocess.Popen[bytes], tracker: _OwnedProcessTracker
) -> None:
    """Terminate and prove empty every observed episode PID/process group."""

    _signal_owned(process, tracker, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        snapshot = _process_snapshot()
        tracker.refresh(snapshot)
        if not tracker.live_pids(snapshot):
            break
        if process.poll() is None:
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(0.05)

    snapshot = _process_snapshot()
    tracker.refresh(snapshot)
    if tracker.live_pids(snapshot):
        _signal_owned(process, tracker, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            snapshot = _process_snapshot()
            tracker.refresh(snapshot)
            if not tracker.live_pids(snapshot):
                break
            if process.poll() is None:
                try:
                    process.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    pass
            else:
                time.sleep(0.05)

    if process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            raise IsolationError("isolated leader did not terminate") from exc
    snapshot = _process_snapshot()
    tracker.refresh(snapshot)
    if tracker.live_pids(snapshot):
        raise IsolationError("isolated owned process tree did not terminate")


def _terminate_untracked_process_group(process: subprocess.Popen[bytes]) -> None:
    """Boundedly terminate a just-spawned new session before tracking exists.

    This path is used only when ownership snapshot/tracker construction fails
    immediately after ``Popen(start_new_session=True)``.  No Agent stdin has
    been delivered yet.  The process PID is therefore the PGID we just
    created; the bounded group kill prevents that exceptional window from
    leaking a capability-bearing Pi process.
    """

    group = process.pid
    for requested_signal, timeout in (
        (signal.SIGTERM, 5.0),
        (signal.SIGKILL, 5.0),
    ):
        try:
            os.killpg(group, requested_signal)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise IsolationError(
                "cannot signal newly spawned isolated process group"
            ) from exc
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        try:
            if _process_group_members(group) == 0:
                return
        except IsolationError:
            # One transient accounting failure is why this cleanup path may
            # have been entered. Continue to SIGKILL before surfacing it.
            if requested_signal == signal.SIGTERM:
                continue
            raise
    try:
        members = _process_group_members(group)
    except IsolationError as exc:
        raise IsolationError(
            "new isolated process group cleanup could not be proven"
        ) from exc
    if process.poll() is None or members:
        raise IsolationError("new isolated process group did not terminate")


def _signal_owned(
    process: subprocess.Popen[bytes],
    tracker: _OwnedProcessTracker,
    requested_signal: signal.Signals,
) -> None:
    snapshot = _process_snapshot()
    tracker.refresh(snapshot)
    supervisor_group = os.getpgrp()
    for group in sorted(tracker.live_groups(snapshot), reverse=True):
        if group <= 1 or group == supervisor_group:
            continue
        # Revalidate at the last possible point. A bare PGID is recyclable;
        # signal it only while it still contains a birth-matching owned PID.
        current = _process_snapshot()
        tracker.refresh(current)
        if group not in tracker.live_groups(current):
            continue
        try:
            os.killpg(group, requested_signal)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise IsolationError("cannot signal an owned process group") from exc
    current = _process_snapshot()
    tracker.refresh(current)
    for pid in sorted(tracker.live_pids(current), reverse=True):
        if pid in {os.getpid(), process.pid}:
            continue
        latest = _process_snapshot()
        if pid not in tracker.live_pids(latest):
            continue
        try:
            os.kill(pid, requested_signal)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise IsolationError("cannot signal an owned process") from exc


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    """Compatibility wrapper for callers that own only the leader PGID."""

    _terminate_owned_processes(process, _OwnedProcessTracker(process.pid))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--tool-socket", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--model-gateway")
    parser.add_argument("command", nargs="+")
    args = parser.parse_args()
    token = os.environ.get(TOOL_TOKEN_ENV, "")
    gateway_token = os.environ.get(GATEWAY_TOKEN_ENV)
    result = run_isolated(
        workspace=args.workspace,
        runtime_root=args.runtime_root,
        command=args.command,
        tool_socket=args.tool_socket,
        tool_token=token,
        stdout_path=args.stdout,
        stderr_path=args.stderr,
        stdin_text="",
        timeout_seconds=args.timeout,
        model_gateway=args.model_gateway,
        model_gateway_token=gateway_token,
    )
    print(json.dumps(result.public_dict(), sort_keys=True))
    return 0 if result.returncode == 0 and not result.timed_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
