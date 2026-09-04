#!/usr/bin/env python3
"""Run one Agent command under the arena's outer macOS Seatbelt boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
from typing import Mapping, Sequence


TRUSTED_ROOT = Path(__file__).resolve().parent
ARENA_ROOT = TRUSTED_ROOT.parent
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
TOOL_SOCKET_ENV = "LAYOUT_DDD_AGENT_TOOL_SOCKET"
TOOL_TOKEN_ENV = "LAYOUT_DDD_AGENT_TOOL_TOKEN"
GATEWAY_TOKEN_ENV = "ARENA_MODEL_GATEWAY_TOKEN"
LOOPBACK_GATEWAY = re.compile(r"^localhost:([1-9][0-9]{0,4})$")
SYSTEM_EXEC_ROOTS = tuple(Path(item) for item in ("/bin", "/sbin", "/usr/bin", "/usr/sbin"))


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

    def public_dict(self) -> dict[str, object]:
        return {
            "schema_version": "sieve_isolated_agent_process_result_v1",
            **asdict(self),
        }


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
) -> IsolationResult:
    if not SANDBOX_EXEC.is_file():
        raise IsolationError("macOS sandbox-exec is unavailable; refusing host execution")
    workspace_path = _real_directory(workspace, "workspace")
    try:
        workspace_path.relative_to((ARENA_ROOT / "episodes").resolve())
    except ValueError as exc:
        raise IsolationError("workspace is outside the arena episodes root") from exc
    runtime_path = _real_directory(runtime_root, "runtime_root")
    socket_path = Path(tool_socket).expanduser().resolve(strict=True)
    if not socket_path.exists() or socket_path.is_symlink():
        raise IsolationError("tool socket is missing or linked")
    if not stat.S_ISSOCK(socket_path.stat().st_mode):
        raise IsolationError("tool socket is not a Unix-domain socket")
    if not isinstance(tool_token, str) or len(tool_token) < 32:
        raise IsolationError("tool capability token is invalid")
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise IsolationError("Agent command must be a non-empty argv array")
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
    if model_gateway_token is not None and len(model_gateway_token) < 32:
        raise IsolationError("model-gateway capability token is invalid")
    policy = TRUSTED_ROOT / "policies" / (
        "scoped_gateway.sb" if gateway is not None else "offline.sb"
    )
    if not policy.is_file() or policy.is_symlink():
        raise IsolationError("Seatbelt policy is missing or linked")

    home = workspace_path / ".home"
    temporary = workspace_path / ".tmp"
    home.mkdir(mode=0o700, exist_ok=True)
    temporary.mkdir(mode=0o700, exist_ok=True)
    environment = _clean_environment(
        workspace=workspace_path,
        home=home,
        temporary=temporary,
        tool_socket=socket_path,
        tool_token=tool_token,
        gateway_token=model_gateway_token,
        extra=extra_environment,
    )

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
    ]
    if gateway is not None:
        sandbox_command.extend(["-D", f"MODEL_GATEWAY={gateway}"])
    sandbox_command.extend(argv)

    stdout = _exclusive_log(stdout_path)
    stderr = _exclusive_log(stderr_path)
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    try:
        with stdout, stderr:
            process = subprocess.Popen(
                sandbox_command,
                cwd=workspace_path,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
            try:
                process.communicate(
                    input=stdin_text.encode("utf-8"), timeout=float(timeout_seconds)
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_group(process)
            returncode = process.returncode
    except OSError as exc:
        raise IsolationError(f"isolated process launch failed: {type(exc).__name__}") from exc
    elapsed = max(0.0, time.monotonic() - started)
    out_path = Path(stdout_path)
    err_path = Path(stderr_path)
    _redact_capabilities(
        (out_path, err_path),
        tuple(
            item
            for item in (tool_token, model_gateway_token)
            if isinstance(item, str) and item
        ),
    )
    if timed_out:
        status = "ambiguous_timeout"
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
        "CODEX_HOME": str(home / ".codex"),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "RUBYOPT": "--disable-gems",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
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
        environment[name] = value
    return environment


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
    return path.open("xb")


def _redact_capabilities(paths: Sequence[Path], secrets: Sequence[str]) -> None:
    encoded = [item.encode("utf-8") for item in secrets]
    for path in paths:
        payload = path.read_bytes()
        for secret in encoded:
            payload = payload.replace(secret, b"[REDACTED_EPISODE_CAPABILITY]")
        path.write_bytes(payload)
        path.chmod(0o600)


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.0)


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
