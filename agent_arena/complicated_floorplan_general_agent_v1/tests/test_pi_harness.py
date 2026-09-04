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
            workspace = _fixture_workspace(root, "perform fixture task\n")
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
                    max_model_requests=120,
                    wall_clock_seconds=7200,
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
            record = material["launch_record"]
            self.assertEqual(record["harness_id"], "sieve-pi-common-harness-v2")
            self.assertEqual(record["limits"]["maximum_model_turns"], 120)
            self.assertEqual(record["limits"]["wall_clock_seconds"], 7200)
            self.assertEqual(record["limits"]["maximum_concurrent_tool_calls"], 1)
            self.assertEqual(len(record["prompts"]["system_prompt_sha256"]), 64)
            self.assertEqual(len(record["prompts"]["task_prompt_sha256"]), 64)
            self.assertEqual(len(record["starting_workspace_sha256"]), 64)
        finally:
            shutil.rmtree(root)

    def test_non_loopback_gateway_is_rejected(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pi-harness-test-"))
        try:
            workspace = _fixture_workspace(root, "fixture\n")
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
                        max_model_requests=10,
                        wall_clock_seconds=60,
                    )
                )
        finally:
            shutil.rmtree(root)


def _fixture_workspace(root: Path, prompt: str) -> Path:
    workspace = root / "workspace"
    (workspace / ".home").mkdir(parents=True)
    (workspace / "TODO.md").write_text(prompt, encoding="utf-8")
    (workspace / "task.json").write_text(
        json.dumps(
            {
                "tool_policy": {"max_total_calls": 160, "max_top_k": 12},
                "asset_database": {"snapshot_id": "fixture-db"},
                "public_validation_policy": {"version": "fixture-v1"},
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "database-interface.json",
        "floorplan.json",
        "room_program.json",
        "submission.schema.json",
    ):
        (workspace / name).write_text("{}\n", encoding="utf-8")
    (workspace / "sieve-agent-tool").write_text("fixture\n", encoding="utf-8")
    return workspace


if __name__ == "__main__":
    unittest.main()
