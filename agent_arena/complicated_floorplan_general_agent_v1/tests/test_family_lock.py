from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
if str(TRUSTED) not in sys.path:
    sys.path.insert(0, str(TRUSTED))

from family_lock import ApiFamilyInvocationLock, ApiFamilyLockError  # noqa: E402


CHILD = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from family_lock import ApiFamilyInvocationLock
with ApiFamilyInvocationLock(
    lock_root=Path(sys.argv[2]),
    api_family_id=sys.argv[3],
    experiment_id="fixture-experiment",
    invocation_nonce=sys.argv[4],
):
    print("acquired", flush=True)
    sys.stdin.readline()
"""


class ApiFamilyInvocationLockTests(unittest.TestCase):
    def test_same_family_fails_closed_and_different_family_can_coexist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sieve-family-lock-") as temporary:
            root = Path(temporary) / "locks"
            process = self._start_child(root, "api2", "child-owner")
            try:
                self.assertEqual(process.stdout.readline().strip(), "acquired")
                with self.assertRaisesRegex(
                    ApiFamilyLockError, "already owns this API family"
                ):
                    with ApiFamilyInvocationLock(
                        lock_root=root,
                        api_family_id="api2",
                        experiment_id="fixture-experiment",
                        invocation_nonce="parent-contender",
                    ):
                        self.fail("same-family contender unexpectedly acquired lock")
                with ApiFamilyInvocationLock(
                    lock_root=root,
                    api_family_id="api3",
                    experiment_id="fixture-experiment",
                    invocation_nonce="parent-other-family",
                ):
                    record = json.loads((root / "api3.lock").read_text())
                    self.assertEqual(record["api_family_id"], "api3")
                    self.assertFalse(record["credential_endpoint_or_request_recorded"])
            finally:
                if process.stdin:
                    process.stdin.write("release\n")
                    process.stdin.flush()
                process.wait(timeout=5)

    def test_crashed_owner_releases_kernel_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sieve-family-crash-") as temporary:
            root = Path(temporary) / "locks"
            process = self._start_child(root, "tokenhub", "crashing-owner")
            self.assertEqual(process.stdout.readline().strip(), "acquired")
            process.kill()
            process.wait(timeout=5)
            with ApiFamilyInvocationLock(
                lock_root=root,
                api_family_id="tokenhub",
                experiment_id="fixture-experiment",
                invocation_nonce="recovery-owner",
            ):
                record = json.loads((root / "tokenhub.lock").read_text())
                self.assertEqual(record["invocation_nonce"], "recovery-owner")

    def test_symlink_lock_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sieve-family-link-") as temporary:
            root = Path(temporary) / "locks"
            root.mkdir()
            target = Path(temporary) / "target"
            target.write_text("fixture", encoding="utf-8")
            (root / "api2.lock").symlink_to(target)
            with self.assertRaisesRegex(ApiFamilyLockError, "unavailable"):
                with ApiFamilyInvocationLock(
                    lock_root=root,
                    api_family_id="api2",
                    experiment_id="fixture-experiment",
                    invocation_nonce="linked-owner",
                ):
                    self.fail("symlink lock unexpectedly accepted")

    @staticmethod
    def _start_child(root: Path, family: str, nonce: str) -> subprocess.Popen[str]:
        environment = {"PATH": os.environ.get("PATH", "")}
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                CHILD,
                str(TRUSTED),
                str(root),
                family,
                nonce,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )


if __name__ == "__main__":
    unittest.main()
