from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
if str(TRUSTED) not in sys.path:
    sys.path.insert(0, str(TRUSTED))

from pi_harness import (  # noqa: E402
    PiEpisodeConfig,
    PiHarnessError,
    prepare_episode,
    verify_runtime,
)


class PiHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = (ARENA_ROOT / "../runtime_bundles/pi-0.85.0").resolve()

    def test_pinned_runtime_is_present(self) -> None:
        observed = verify_runtime(self.runtime)
        self.assertEqual(observed["pi_version"], "0.85.0")
        self.assertTrue(Path(observed["node"]).is_file())
        self.assertTrue(Path(observed["cli"]).is_file())

    def test_episode_configuration_is_secret_free_and_fixed(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pi-harness-test-"))
        try:
            workspace = root / "workspace"
            (workspace / ".home").mkdir(parents=True)
            (workspace / "TODO.md").write_text("perform fixture task\n", encoding="utf-8")
            material = prepare_episode(
                PiEpisodeConfig(
                    runtime_root=self.runtime,
                    workspace=workspace,
                    gateway_base_url="http://localhost:43123",
                    wire_model="fixture-model",
                    api="openai-completions",
                    thinking="high",
                    context_window=131072,
                    max_tokens=65536,
                )
            )
            models = json.loads(Path(material["models_path"]).read_text(encoding="utf-8"))
            provider = models["providers"]["sieve-gateway"]
            self.assertEqual(provider["baseUrl"], "http://127.0.0.1:43123/v1")
            self.assertEqual(provider["apiKey"], "$ARENA_MODEL_GATEWAY_TOKEN")
            self.assertNotIn("secret", json.dumps(models))
            command = material["command"]
            self.assertIn("--no-context-files", command)
            self.assertIn("--no-extensions", command)
            self.assertIn("--no-skills", command)
            self.assertIn("--no-session", command)
            self.assertIn("--offline", command)
            self.assertIn("--print", command)
            self.assertEqual(material["stdin_text"], "perform fixture task\n")
        finally:
            shutil.rmtree(root)

    def test_non_loopback_gateway_is_rejected(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pi-harness-test-"))
        try:
            workspace = root / "workspace"
            (workspace / ".home").mkdir(parents=True)
            (workspace / "TODO.md").write_text("fixture\n", encoding="utf-8")
            with self.assertRaisesRegex(PiHarnessError, "loopback"):
                prepare_episode(
                    PiEpisodeConfig(
                        runtime_root=self.runtime,
                        workspace=workspace,
                        gateway_base_url="https://example.com/v1",
                        wire_model="fixture-model",
                        api="openai-completions",
                        thinking="off",
                        context_window=1000,
                        max_tokens=500,
                    )
                )
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
