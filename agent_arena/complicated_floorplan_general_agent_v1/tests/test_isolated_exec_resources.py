from __future__ import annotations

import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import sys
import tempfile
import time
import unittest
from unittest import mock


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
if str(TRUSTED) not in sys.path:
    sys.path.insert(0, str(TRUSTED))

import isolated_exec  # noqa: E402
from isolated_exec import run_isolated  # noqa: E402


class IsolatedExecResourceTests(unittest.TestCase):
    def test_recycled_pid_is_not_owned_or_signaled(self) -> None:
        leader_pid = 98761
        initial = {
            leader_pid: isolated_exec._ProcessInfo(
                ppid=1,
                pgid=leader_pid,
                state="S",
                birth="original-birth",
            )
        }
        tracker = isolated_exec._OwnedProcessTracker(leader_pid, initial)
        recycled = {
            leader_pid: isolated_exec._ProcessInfo(
                ppid=1,
                pgid=leader_pid,
                state="S",
                birth="recycled-birth",
            ),
            leader_pid + 1: isolated_exec._ProcessInfo(
                ppid=leader_pid,
                pgid=leader_pid,
                state="S",
                birth="unrelated-child",
            ),
        }
        tracker.refresh(recycled)
        self.assertEqual(tracker.live_pids(recycled), set())
        process = mock.Mock(pid=leader_pid)
        with (
            mock.patch.object(isolated_exec, "_process_snapshot", return_value=recycled),
            mock.patch.object(isolated_exec.os, "killpg") as killpg,
            mock.patch.object(isolated_exec.os, "kill") as kill_pid,
        ):
            isolated_exec._signal_owned(process, tracker, signal.SIGTERM)
        killpg.assert_not_called()
        kill_pid.assert_not_called()

    def test_background_descendant_makes_zero_exit_fail_closed(self) -> None:
        run_root = (
            ARENA_ROOT
            / "episodes"
            / "isolation-test"
            / "background-child"
            / f"run-{secrets.token_hex(8)}"
        )
        workspace = run_root / "workspace"
        host = run_root / "host"
        workspace.mkdir(parents=True, mode=0o700)
        host.mkdir(mode=0o700)
        (workspace / ".home").mkdir(mode=0o700)
        (workspace / ".tmp").mkdir(mode=0o700)
        worker = workspace / "worker.rb"
        worker.write_text(
            "\n".join(
                [
                    "child = Process.spawn('/usr/bin/ruby', '-e', 'sleep 60', in: 'worker.rb', out: 'background.stdout', err: 'background.stderr')",
                    "File.write('background.pid', child.to_s)",
                    "exit 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        worker.chmod(0o555)
        socket_root = Path(tempfile.mkdtemp(prefix="sieve-isolation-test-"))
        socket_path = socket_root / "tool.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(1)
        child_pid: int | None = None
        try:
            result = run_isolated(
                workspace=workspace,
                runtime_root="/usr/bin",
                command=["/usr/bin/ruby", "worker.rb"],
                tool_socket=socket_path,
                tool_token="f" * 64,
                stdout_path=host / "stdout.log",
                stderr_path=host / "stderr.log",
                stdin_text="",
                timeout_seconds=30,
            )
            self.assertTrue(
                (workspace / "background.pid").is_file(),
                {
                    "result": result.public_dict(),
                    "stdout": (host / "stdout.log").read_text(
                        encoding="utf-8", errors="replace"
                    ),
                    "stderr": (host / "stderr.log").read_text(
                        encoding="utf-8", errors="replace"
                    ),
                },
            )
            child_pid = int((workspace / "background.pid").read_text())
            self.assertEqual(result.status, "resource_limit_exceeded")
            self.assertTrue(result.resource_limit_exceeded)
            self.assertEqual(result.resource_limit_kind, "residual_process_group")
            deadline = time.monotonic() + 2.0
            while _pid_exists(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(_pid_exists(child_pid))
        finally:
            if child_pid is not None and _pid_exists(child_pid):
                try:
                    os.kill(child_pid, 9)
                except ProcessLookupError:
                    pass
            server.close()
            if socket_path.exists():
                socket_path.unlink()
            shutil.rmtree(socket_root, ignore_errors=True)
            shutil.rmtree(run_root, ignore_errors=True)

    def test_exit_boundary_log_burst_is_not_accepted(self) -> None:
        with mock.patch.object(isolated_exec, "MAX_TOTAL_LOG_BYTES", 1024):
            result = self._run_ruby(
                case="exit-log-burst",
                source="STDOUT.write('x' * 4096); exit 0\n",
            )
        self.assertEqual(result.status, "resource_limit_exceeded")
        self.assertEqual(result.resource_limit_kind, "sanitized_log_bytes")

    def test_exit_boundary_workspace_burst_is_recounted(self) -> None:
        with (
            mock.patch.object(isolated_exec, "MAX_WORKSPACE_BYTES", 1024),
            mock.patch.object(isolated_exec, "WORKSPACE_POLL_SECONDS", 3600.0),
        ):
            result = self._run_ruby(
                case="exit-workspace-burst",
                source="File.binwrite('burst.bin', 'x' * 4096); exit 0\n",
            )
        self.assertEqual(result.status, "resource_limit_exceeded")
        self.assertEqual(result.resource_limit_kind, "workspace_bytes")
        self.assertGreater(result.workspace_bytes_observed, 1024)

    def _run_ruby(self, *, case: str, source: str):
        run_root = (
            ARENA_ROOT
            / "episodes"
            / "isolation-test"
            / case
            / f"run-{secrets.token_hex(8)}"
        )
        workspace = run_root / "workspace"
        host = run_root / "host"
        workspace.mkdir(parents=True, mode=0o700)
        host.mkdir(mode=0o700)
        (workspace / ".home").mkdir(mode=0o700)
        (workspace / ".tmp").mkdir(mode=0o700)
        worker = workspace / "worker.rb"
        worker.write_text(source, encoding="utf-8")
        worker.chmod(0o555)
        socket_root = Path(tempfile.mkdtemp(prefix="sieve-isolation-burst-"))
        socket_path = socket_root / "tool.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(1)
        try:
            return run_isolated(
                workspace=workspace,
                runtime_root="/usr/bin",
                command=["/usr/bin/ruby", "worker.rb"],
                tool_socket=socket_path,
                tool_token="e" * 64,
                stdout_path=host / "stdout.log",
                stderr_path=host / "stderr.log",
                stdin_text="",
                timeout_seconds=30,
            )
        finally:
            server.close()
            if socket_path.exists():
                socket_path.unlink()
            shutil.rmtree(socket_root, ignore_errors=True)
            shutil.rmtree(run_root, ignore_errors=True)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
