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
    DATABASE_EVENT_SCHEMA_VERSION,
    PiEpisodeConfig,
    PiHarnessError,
    prepare_episode,
    verify_existing_episode_launch,
    verify_prepared_episode_material,
    verify_runtime,
)
from api_profiles import ProfileRegistry  # noqa: E402


class PiHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = (ARENA_ROOT / "../runtime_bundles/pi-0.85.0").resolve()
        cls.registry = ProfileRegistry.load()
        cls.model = cls.registry.model("api2-gpt-5-6-sol-agent-v1")

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
                    model_profile=self.model,
                    experiment_id="fixture-experiment-v1",
                    experiment_sha256="1" * 64,
                    profile_registry_sha256=self.registry.content_sha256,
                    max_model_requests=120,
                    wall_clock_seconds=7200,
                )
            )
            models = json.loads(Path(material["models_path"]).read_text(encoding="utf-8"))
            provider = models["providers"]["sieve-gateway"]
            self.assertEqual(provider["baseUrl"], "http://127.0.0.1:43123/v1")
            self.assertEqual(provider["apiKey"], "$ARENA_MODEL_GATEWAY_TOKEN")
            self.assertNotIn("secret", json.dumps(models))
            self.assertEqual(provider["models"][0]["id"], self.model.client_wire_model)
            settings = json.loads(
                Path(material["settings_path"]).read_text(encoding="utf-8")
            )
            self.assertFalse(settings["retry"]["enabled"])
            self.assertEqual(settings["retry"]["maxRetries"], 0)
            self.assertEqual(settings["retry"]["provider"]["maxRetries"], 0)
            self.assertEqual(settings["transport"], "sse")
            command = material["command"]
            self.assertIn("--no-context-files", command)
            self.assertIn("--no-extensions", command)
            self.assertIn("--no-skills", command)
            self.assertIn("--no-session", command)
            self.assertIn("--offline", command)
            self.assertIn("--print", command)
            self.assertEqual(material["stdin_text"], "perform fixture task\n")
            record = material["launch_record"]
            self.assertEqual(record["harness_id"], "sieve-pi-common-harness-v4")
            self.assertEqual(
                record["tool_transcript"]["database_event_schema"],
                DATABASE_EVENT_SCHEMA_VERSION,
            )
            self.assertEqual(record["limits"]["maximum_model_turns"], 120)
            self.assertEqual(record["limits"]["wall_clock_seconds"], 7200)
            self.assertEqual(record["limits"]["maximum_concurrent_tool_calls"], 1)
            self.assertTrue(
                record["retry_ownership"]["trusted_gateway_is_only_retry_owner"]
            )
            self.assertEqual(
                record["retry_ownership"]["pi_provider_sdk_max_retries"], 0
            )
            self.assertEqual(
                len(record["prompts"]["source_system_prompt_sha256"]), 64
            )
            self.assertEqual(
                record["prompts"]["source_system_prompt_sha256"],
                record["prompts"]["provider_visible_episode_system_prompt_sha256"],
            )
            self.assertFalse(
                record["prompts"]["pi_cwd_suffix_forwarded_to_provider"]
            )
            self.assertEqual(len(record["prompts"]["task_prompt_sha256"]), 64)
            self.assertEqual(len(record["starting_workspace_sha256"]), 64)
            self.assertEqual(
                record["tooling"]["all_tools_execution_mode"], "sequential"
            )
            self.assertFalse(
                record["tooling"]["bash_model_gateway_capability_inherited"]
            )
            verify_prepared_episode_material(material)
            verification = verify_existing_episode_launch(
                launch_record=record,
                workspace=workspace,
                runtime_root=self.runtime,
                model_profile=self.model,
                experiment_id="fixture-experiment-v1",
                experiment_sha256="1" * 64,
                profile_registry_sha256=self.registry.content_sha256,
                max_model_requests=120,
                wall_clock_seconds=7200,
            )
            self.assertTrue(verification["valid"])
        finally:
            shutil.rmtree(root)

    def test_database_event_schema_matches_public_and_harness_contracts(self) -> None:
        public_interface = json.loads(
            (ARENA_ROOT / "public/database-interface.json").read_text(
                encoding="utf-8"
            )
        )
        harness = json.loads(
            (TRUSTED / "pi_harness/harness.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            DATABASE_EVENT_SCHEMA_VERSION,
            "non_rectangular_agent_tool_event_v3",
        )
        self.assertEqual(
            public_interface["transcript_policy"]["event_schema"],
            DATABASE_EVENT_SCHEMA_VERSION,
        )
        self.assertEqual(
            harness["fairness_and_audit"]["tool_transcript"][
                "database_event_schema"
            ],
            DATABASE_EVENT_SCHEMA_VERSION,
        )

    def test_agent_visible_pi_configuration_tampering_is_rejected(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pi-harness-tamper-"))
        try:
            workspace = _fixture_workspace(root, "perform fixture task\n")
            material = prepare_episode(
                PiEpisodeConfig(
                    runtime_root=self.runtime,
                    workspace=workspace,
                    gateway_base_url="http://localhost:43123",
                    model_profile=self.model,
                    experiment_id="fixture-experiment-v1",
                    experiment_sha256="1" * 64,
                    profile_registry_sha256=self.registry.content_sha256,
                    max_model_requests=120,
                    wall_clock_seconds=7200,
                )
            )
            models_path = Path(material["models_path"])
            models_path.chmod(0o600)
            models_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(PiHarnessError, "models.json changed"):
                verify_prepared_episode_material(material)
        finally:
            shutil.rmtree(root)

    def test_provider_visible_system_prompt_hash_is_path_independent(self) -> None:
        roots = [Path(tempfile.mkdtemp(prefix=f"pi-path-{index}-")) for index in (1, 2)]
        try:
            records = []
            for root in roots:
                workspace = _fixture_workspace(root, "same task\n")
                material = prepare_episode(
                    PiEpisodeConfig(
                        runtime_root=self.runtime,
                        workspace=workspace,
                        gateway_base_url="http://localhost:43123",
                        model_profile=self.model,
                        experiment_id="fixture-experiment-v1",
                        experiment_sha256="1" * 64,
                        profile_registry_sha256=self.registry.content_sha256,
                        max_model_requests=120,
                        wall_clock_seconds=7200,
                    )
                )
                records.append(material["launch_record"]["prompts"])
            self.assertNotEqual(
                records[0]["effective_pi_system_prompt_sha256"],
                records[1]["effective_pi_system_prompt_sha256"],
            )
            self.assertEqual(
                records[0]["provider_visible_episode_system_prompt_sha256"],
                records[1]["provider_visible_episode_system_prompt_sha256"],
            )
        finally:
            for root in roots:
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
                        model_profile=self.model,
                        experiment_id="fixture-experiment-v1",
                        experiment_sha256="1" * 64,
                        profile_registry_sha256=self.registry.content_sha256,
                        max_model_requests=10,
                        wall_clock_seconds=60,
                    )
                )
        finally:
            shutil.rmtree(root)


def _fixture_workspace(root: Path, prompt: str) -> Path:
    workspace = root / "workspace"
    (workspace / ".home").mkdir(parents=True)
    (workspace / ".tmp").mkdir(parents=True)
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
