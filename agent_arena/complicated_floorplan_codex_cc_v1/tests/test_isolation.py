from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import unittest


ARENA_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(Path("/usr/bin/sandbox-exec").is_file(), "macOS Seatbelt unavailable")
class IsolationIntegrationTests(unittest.TestCase):
    def test_outer_boundary_smoke(self) -> None:
        result = subprocess.run(
            ["/usr/bin/python3", str(ARENA_ROOT / "trusted/smoke_isolation.py")],
            cwd=ARENA_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["filesystem"], "workspace_only")
        self.assertEqual(report["database_socket"], "allowed")
        self.assertEqual(report["arbitrary_tcp"], "denied")
        self.assertEqual(report["capability_log_redaction"], "valid")
        self.assertFalse(report["host_environment_inherited"])
        self.assertFalse(report["model_or_generation_started"])

    @unittest.skipUnless(shutil.which("codex"), "Codex CLI unavailable")
    def test_installed_codex_starts_inside_outer_boundary(self) -> None:
        result = subprocess.run(
            [
                "/usr/bin/python3",
                str(ARENA_ROOT / "trusted/smoke_codex_runtime.py"),
            ],
            cwd=ARENA_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "valid")
        self.assertTrue(report["codex_version"].startswith("codex-cli "))
        self.assertFalse(report["model_or_generation_started"])

    def test_exact_gateway_is_the_only_tcp_destination(self) -> None:
        result = subprocess.run(
            [
                "/usr/bin/python3",
                str(ARENA_ROOT / "trusted/smoke_gateway_isolation.py"),
            ],
            cwd=ARENA_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["exact_gateway"], "allowed")
        self.assertEqual(report["other_tcp"], "denied")
        self.assertFalse(report["model_or_generation_started"])


if __name__ == "__main__":
    unittest.main()
