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

from api_profiles import (  # noqa: E402
    ProfileError,
    ProfileRegistry,
    load_experiment,
    load_runtime_bindings,
    normalize_runtime_credential,
)


class ApiProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProfileRegistry.load()

    def test_registered_families_routes_and_models_are_exact(self) -> None:
        self.assertEqual(set(self.registry.api_families), {"api2", "api3", "tokenhub"})
        self.assertEqual(len(self.registry.routes), 5)
        self.assertEqual(len(self.registry.models), 8)
        for model in self.registry.models.values():
            self.assertEqual(model.retry.retry_delay_seconds, 30)
            self.assertEqual(model.retry.max_infrastructure_retries, 5)
            self.assertFalse(model.retry.retry_ambiguous_timeouts)
            self.assertEqual(
                set(model.retry.retryable_http_statuses),
                {408, 409, 425, 429, 500, 502, 503, 504},
            )
            self.assertEqual(model.pi.context_window, 131072)
            self.assertEqual(model.pi.maximum_output_tokens, 65536)

    def test_api_specific_auth_and_reasoning_profiles_are_frozen(self) -> None:
        api3 = self.registry.model("api3-claude-sonnet-5-agent-v1")
        api3_route = self.registry.route(api3.route_profile_id)
        self.assertEqual(api3_route.auth_strategy, "api3_bearer_session_v1")
        self.assertEqual(api3.api3_strategy_type, "ConsistentHash")
        self.assertTrue(api3.response_identity_required)

        tokenhub = self.registry.model("tokenhub-hy4-preview-agent-v1")
        tokenhub_route = self.registry.route(tokenhub.route_profile_id)
        self.assertEqual(tokenhub_route.auth_strategy, "standard_bearer_v1")
        self.assertEqual(tokenhub.reasoning.effort, "high")
        self.assertEqual(tokenhub.pi.thinking_level, "high")
        self.assertTrue(tokenhub.reasoning.preserve_across_tool_turns)

        glm = self.registry.model("api2-glm-5-3-agent-v1")
        self.assertEqual(glm.pi.api_protocol, "openai-responses")
        self.assertEqual(glm.reasoning.style, "responses_reasoning")

        kimi = self.registry.model("api2-kimi-k3-agent-v1")
        self.assertEqual(kimi.auth_query_parameters["timeout"], "600")
        self.assertEqual(kimi.request_timeout_seconds, 660)

        api2_hy4 = self.registry.model("api2-hy4-preview-agent-v1")
        self.assertTrue(api2_hy4.response_identity_required)
        self.assertEqual(api2_hy4.accepted_response_models, ("hy4-preview",))

    def test_each_example_is_one_api_family_many_models(self) -> None:
        expected_counts = {"api2": 4, "api3": 3, "tokenhub": 1}
        for path in sorted((TRUSTED / "experiments").glob("*.example.json")):
            experiment = load_experiment(path, self.registry)
            self.assertEqual(
                len(experiment.model_profile_ids), expected_counts[experiment.api_family_id]
            )
            self.assertEqual(len(experiment.scene_ids), 10)
            self.assertTrue(experiment.require_tool_call_preflight)
            self.assertEqual(experiment.maximum_concurrent_episodes, 1)
            for model_id in experiment.model_profile_ids:
                self.assertEqual(
                    self.registry.model(model_id).api_family_id,
                    experiment.api_family_id,
                )

    def test_mixed_api_experiment_is_rejected(self) -> None:
        source = TRUSTED / "experiments/api2-all-models.example.json"
        value = json.loads(source.read_text(encoding="utf-8"))
        value["model_profile_ids"].append("api3-claude-opus-5-agent-v1")
        root = Path(tempfile.mkdtemp(prefix="mixed-api-profile-"))
        try:
            path = root / "mixed.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "cannot mix API families"):
                load_experiment(path, self.registry)
        finally:
            shutil.rmtree(root)

    def test_runtime_bindings_contain_no_credential_fields(self) -> None:
        raw = (TRUSTED / "profiles/runtime_bindings.example.json").read_text(
            encoding="utf-8"
        )
        lowered = raw.lower()
        for forbidden in ("api_key", "apikey", "authorization", "credential"):
            self.assertNotIn(forbidden, lowered)
        experiment = load_experiment(
            TRUSTED / "experiments/api2-all-models.example.json", self.registry
        )
        bindings = load_runtime_bindings(
            TRUSTED / "profiles/runtime_bindings.example.json",
            registry=self.registry,
            experiment=experiment,
        )
        self.assertEqual(bindings.api_family_id, "api2")
        self.assertEqual(len(bindings.routes), 3)

    def test_credentials_are_normalized_but_never_part_of_profiles(self) -> None:
        api2 = self.registry.api_family("api2")
        self.assertEqual(
            normalize_runtime_credential(api2, "fixture-app:fixture-key?ignored=1"),
            "fixture-app:fixture-key",
        )
        with self.assertRaisesRegex(ProfileError, "APP_ID:APP_KEY"):
            normalize_runtime_credential(api2, "malformed")
        tokenhub = self.registry.api_family("tokenhub")
        self.assertEqual(
            normalize_runtime_credential(tokenhub, "fixture-bearer"),
            "fixture-bearer",
        )

    def test_protocol_path_and_stream_contract_cannot_drift(self) -> None:
        root = self._copy_profile_root("profile-route-drift-")
        try:
            path = root / "route_profiles.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["routes"][0]["client_path"] = "/v1/responses"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "protocol/path/stream"):
                ProfileRegistry.load(root)
        finally:
            shutil.rmtree(root)

    def test_api_family_auth_strategy_cannot_drift(self) -> None:
        root = self._copy_profile_root("profile-auth-drift-")
        try:
            path = root / "route_profiles.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            api3 = next(
                row for row in value["routes"] if row["api_family_id"] == "api3"
            )
            api3["auth_strategy"] = "standard_bearer_v1"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "family/auth strategy"):
                ProfileRegistry.load(root)
        finally:
            shutil.rmtree(root)

    def test_api2_query_model_must_equal_upstream_wire_model(self) -> None:
        root = self._copy_profile_root("profile-api2-query-drift-")
        try:
            path = root / "model_profiles.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            kimi = next(
                row
                for row in value["models"]
                if row["model_profile_id"] == "api2-kimi-k3-agent-v1"
            )
            kimi["auth_query_parameters"]["model"] = "wrong-model"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "query model differs"):
                ProfileRegistry.load(root)
        finally:
            shutil.rmtree(root)

    def test_api2_platform_timeout_must_leave_a_local_ambiguity_grace(self) -> None:
        root = self._copy_profile_root("profile-api2-timeout-drift-")
        try:
            path = root / "model_profiles.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            kimi = next(
                row
                for row in value["models"]
                if row["model_profile_id"] == "api2-kimi-k3-agent-v1"
            )
            kimi["request_timeout_seconds"] = 600
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "platform timeout"):
                ProfileRegistry.load(root)
        finally:
            shutil.rmtree(root)

    def test_runtime_binding_requires_opaque_revision_identity(self) -> None:
        experiment = load_experiment(
            TRUSTED / "experiments/tokenhub-hy4.example.json",
            self.registry,
        )
        source = TRUSTED / "profiles/runtime_bindings.example.json"
        value = json.loads(source.read_text(encoding="utf-8"))
        del value["api_families"]["tokenhub"]["routes"][
            "tokenhub-chat-reasoning-agent-v1"
        ]["binding_profile_id"]
        root = Path(tempfile.mkdtemp(prefix="binding-revision-"))
        try:
            path = root / "bindings.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "binding_profile_id"):
                load_runtime_bindings(
                    path,
                    registry=self.registry,
                    experiment=experiment,
                )
        finally:
            shutil.rmtree(root)

    def test_official_experiment_cannot_retry_whole_episode(self) -> None:
        source = TRUSTED / "experiments/api2-all-models.example.json"
        value = json.loads(source.read_text(encoding="utf-8"))
        value["limits"]["episode_attempts"] = 2
        root = Path(tempfile.mkdtemp(prefix="episode-retry-profile-"))
        try:
            path = root / "experiment.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "episode_attempts=1"):
                load_experiment(path, self.registry)
        finally:
            shutil.rmtree(root)

    @staticmethod
    def _copy_profile_root(prefix: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix=prefix))
        for name in ("api_families.json", "route_profiles.json", "model_profiles.json"):
            shutil.copy2(TRUSTED / "profiles" / name, root / name)
        return root


if __name__ == "__main__":
    unittest.main()
