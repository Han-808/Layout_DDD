from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
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

from api_profiles import (  # noqa: E402
    ProfileRegistry,
    RouteRuntimeBinding,
    load_experiment,
    load_runtime_bindings,
)
from isolated_exec import IsolationError, _ProcessInfo  # noqa: E402
import managed_transport as subject  # noqa: E402
import isolated_exec  # noqa: E402
from model_gateway import GatewayError, ScopedModelGateway  # noqa: E402
import tokenhub_release_gate as release_gate  # noqa: E402


DEFAULT_TOKENHUB_RUNTIME = Path(
    os.environ.get(
        "SIEVE_TOKENHUB_LITELLM_RUNTIME",
        str(
            ARENA_ROOT.parents[1]
            / "Support/third_party/uni_llm_hhr_cursor_snapshot"
        ),
    )
)
PINNED_PI_RUNTIME = (ARENA_ROOT / "../runtime_bundles/pi-0.85.0").resolve()


class _FakeProcess:
    def __init__(self, pid: int = 424242) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0


class ManagedTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProfileRegistry.load()
        cls.experiment = load_experiment(
            TRUSTED / "experiments/tokenhub-hy4.example.json",
            cls.registry,
        )
        cls.bindings = load_runtime_bindings(
            TRUSTED / "profiles/runtime_bindings.example.json",
            registry=cls.registry,
            experiment=cls.experiment,
        )

    def test_manifest_config_freezes_thinking_and_timeout_layering(self) -> None:
        manifest = subject._read_manifest(subject.TOKENHUB_MANIFEST)
        subject._validate_tokenhub_experiment_contract(
            self.registry,
            self.experiment,
            manifest,
        )
        config = subject._render_tokenhub_config(
            manifest, "https://tokenhub-fixture.example.invalid"
        ).decode("utf-8")
        self.assertIn("model: \"anthropic/hy4-preview\"", config)
        self.assertIn("api_key: os.environ/TOKENHUB_API_KEY", config)
        self.assertIn("budget_tokens: 4096", config)
        self.assertIn("num_retries: 0", config)
        self.assertIn("request_timeout: 1860", config)
        self.assertNotIn("fixture-provider-secret", config)

        drifted = deepcopy(manifest)
        drifted["proxy"]["request_timeout_seconds"] = 1800
        with self.assertRaisesRegex(
            subject.ManagedTransportError, "ambiguous timeout"
        ):
            subject._validate_tokenhub_experiment_contract(
                self.registry,
                self.experiment,
                drifted,
            )

    def test_managed_contract_rejects_profile_or_route_drift(self) -> None:
        manifest = subject._read_manifest(subject.TOKENHUB_MANIFEST)
        model_id = self.experiment.model_profile_ids[0]
        model = self.registry.model(model_id)
        route = self.registry.route(model.route_profile_id)
        mutations = {
            "model identity": replace(model, model_profile_id="drifted-profile"),
            "wire model": replace(model, upstream_wire_model="other-model"),
            "reasoning": replace(
                model, reasoning=replace(model.reasoning, effort="low")
            ),
            "Pi thinking": replace(
                model, pi=replace(model.pi, thinking_level="low")
            ),
            "Pi compatibility": replace(
                model,
                pi=replace(
                    model.pi,
                    compatibility={**model.pi.compatibility, "supportsStore": True},
                ),
            ),
            "retry": replace(
                model,
                retry=replace(model.retry, max_infrastructure_retries=4),
            ),
            "response identity": replace(
                model, response_identity_required=False
            ),
        }
        for label, drifted_model in mutations.items():
            with self.subTest(label=label):
                registry = replace(
                    self.registry,
                    models={**self.registry.models, model_id: drifted_model},
                )
                with self.assertRaisesRegex(
                    subject.ManagedTransportError, "model/route contract differs"
                ):
                    subject._validate_tokenhub_experiment_contract(
                        registry, self.experiment, manifest
                    )
        drifted_route = replace(route, option_style="legacy_core_v1")
        registry = replace(
            self.registry,
            routes={**self.registry.routes, route.route_profile_id: drifted_route},
        )
        with self.assertRaisesRegex(
            subject.ManagedTransportError, "model/route contract differs"
        ):
            subject._validate_tokenhub_experiment_contract(
                registry, self.experiment, manifest
            )

    def test_transport_contract_is_endpoint_and_credential_free(self) -> None:
        contract = subject.transport_contracts_for_experiment(
            self.registry, self.experiment
        )
        encoded = json.dumps(contract, sort_keys=True)
        self.assertEqual(len(contract), 1)
        self.assertEqual(contract[0]["adapter_id"], subject.TOKENHUB_ADAPTER)
        self.assertRegex(contract[0]["manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("tokenhub.tencentmaas.com", encoded)
        self.assertNotIn("api_key", encoded.lower())
        self.assertNotIn("authorization", encoded.lower())

    def test_runtime_gate_is_credential_free_and_requires_tokenhub_pin(self) -> None:
        api3 = load_experiment(
            TRUSTED / "experiments/api3-all-models.example.json",
            self.registry,
        )
        direct = subject.verify_transport_runtime_compatibility(
            registry=self.registry,
            experiment=api3,
            tokenhub_runtime_root=None,
        )
        self.assertTrue(direct["ok"])
        self.assertFalse(direct["managed_process_required"])
        self.assertFalse(
            direct["production_provider_credential_or_route_used"]
        )
        with self.assertRaisesRegex(
            subject.ManagedTransportError, "requires --tokenhub-litellm-runtime"
        ):
            subject.verify_transport_runtime_compatibility(
                registry=self.registry,
                experiment=self.experiment,
                tokenhub_runtime_root=None,
            )
        with self.assertRaisesRegex(
            subject.ManagedTransportError, "requires runtime bindings"
        ):
            subject.verify_transport_runtime_compatibility(
                registry=self.registry,
                experiment=self.experiment,
                tokenhub_runtime_root=Path("/not-inspected-before-binding-check"),
                verify_managed_lifecycle=True,
            )
        with self.assertRaisesRegex(
            subject.ManagedTransportError, "requires the pinned Pi runtime"
        ):
            subject.verify_transport_runtime_compatibility(
                registry=self.registry,
                experiment=self.experiment,
                tokenhub_runtime_root=Path("/not-inspected-before-pi-check"),
                bindings=self.bindings,
                verify_managed_lifecycle=True,
            )

    def test_raw_tokenhub_binding_cannot_bypass_host_adapter(self) -> None:
        model = self.registry.model("tokenhub-hy4-preview-agent-v1")
        route = self.registry.route(model.route_profile_id)
        with self.assertRaisesRegex(GatewayError, "host-owned loopback adapter"):
            ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-raw-tokenhub-v1",
                    upstream_base_url="https://tokenhub-fixture.example.invalid",
                    allow_insecure_upstream=False,
                ),
                runtime_credential="fixture-provider-secret",
                max_requests=1,
            )

    def test_managed_start_returns_only_loopback_capability(self) -> None:
        process = _FakeProcess()
        snapshot = {
            process.pid: _ProcessInfo(
                ppid=1,
                pgid=process.pid,
                state="S",
                birth="fixture-birth",
            )
        }
        captured_environment: dict[str, str] = {}

        def spawn(*_args: object, **kwargs: object) -> _FakeProcess:
            captured_environment.update(dict(kwargs["env"]))  # type: ignore[arg-type]
            return process

        with tempfile.TemporaryDirectory(prefix="sieve-managed-transport-") as tmp:
            invocation = Path(tmp)
            manager = subject._TokenHubLiteLLM(
                registry=self.registry,
                experiment=self.experiment,
                bindings=self.bindings,
                provider_credential="fixture-provider-secret",
                invocation_root=invocation,
                runtime_root=Path("/fixture/runtime"),
            )

            def terminate(fake: _FakeProcess, _tracker: object) -> None:
                fake.returncode = 0

            with (
                mock.patch.object(
                    subject,
                    "verify_tokenhub_runtime",
                    return_value={"verified": True},
                ),
                mock.patch.object(subject, "_select_loopback_port", return_value=43210),
                mock.patch.object(subject.subprocess, "Popen", side_effect=spawn),
                mock.patch.object(subject, "_process_snapshot", return_value=snapshot),
                mock.patch.object(manager, "_wait_until_ready"),
                mock.patch.object(
                    subject,
                    "_read_model_aliases",
                    return_value={"hy4-preview"},
                ) as inventory,
                mock.patch.object(
                    subject,
                    "_terminate_owned_processes",
                    side_effect=terminate,
                ),
                mock.patch.object(
                    subject, "_loopback_listener_exists", return_value=False
                ),
            ):
                manager.start()
                transport = manager.transport()
                self.assertNotEqual(transport.credential, "fixture-provider-secret")
                binding = transport.bindings.for_route(
                    "tokenhub-chat-reasoning-agent-v1"
                )
                self.assertEqual(
                    binding.upstream_base_url, "http://127.0.0.1:43210/v1"
                )
                self.assertEqual(
                    binding.managed_adapter_id, subject.TOKENHUB_ADAPTER
                )
                public = json.dumps(transport.public_record, sort_keys=True)
                self.assertNotIn("fixture-provider-secret", public)
                self.assertNotIn("tokenhub.tencentmaas.com", public)
                inventory.assert_called_once_with(43210, transport.credential)
                self.assertEqual(
                    captured_environment["TOKENHUB_API_KEY"],
                    manager._identity_relay.capability,
                )
                self.assertNotEqual(
                    captured_environment["TOKENHUB_API_KEY"],
                    "fixture-provider-secret",
                )
                manager.close()

            self.assertEqual(manager._provider_credential, "")
            self.assertEqual(manager._proxy_key, "")
            self.assertTrue((invocation / "transport_start.json").is_file())
            self.assertTrue((invocation / "transport_completion.json").is_file())
            self.assertFalse(manager._temporary_root.exists())

    def test_ambiguous_recycle_replaces_process_and_preserves_binding(self) -> None:
        old_process = _FakeProcess(pid=424201)
        new_process = _FakeProcess(pid=424202)
        snapshots = {
            old_process.pid: _ProcessInfo(
                ppid=1,
                pgid=old_process.pid,
                state="S",
                birth="fixture-old-birth",
            ),
            new_process.pid: _ProcessInfo(
                ppid=1,
                pgid=new_process.pid,
                state="S",
                birth="fixture-new-birth",
            ),
        }
        captured_environments: list[dict[str, str]] = []

        def spawn(*_args: object, **kwargs: object) -> _FakeProcess:
            captured_environments.append(dict(kwargs["env"]))  # type: ignore[arg-type]
            return [old_process, new_process][len(captured_environments) - 1]

        def terminate(fake: _FakeProcess, _tracker: object) -> None:
            fake.returncode = 0

        with tempfile.TemporaryDirectory(prefix="sieve-managed-recycle-") as tmp:
            invocation = Path(tmp)
            manager = subject._TokenHubLiteLLM(
                registry=self.registry,
                experiment=self.experiment,
                bindings=self.bindings,
                provider_credential="fixture-provider-secret",
                invocation_root=invocation,
                runtime_root=Path("/fixture/runtime"),
            )
            with (
                mock.patch.object(
                    subject,
                    "verify_tokenhub_runtime",
                    return_value={"verified": True},
                ),
                mock.patch.object(subject, "_select_loopback_port", return_value=43219),
                mock.patch.object(subject.subprocess, "Popen", side_effect=spawn),
                mock.patch.object(subject, "_process_snapshot", return_value=snapshots),
                mock.patch.object(manager, "_wait_until_ready"),
                mock.patch.object(
                    subject, "_read_model_aliases", return_value={"hy4-preview"}
                ),
                mock.patch.object(
                    subject, "_terminate_owned_processes", side_effect=terminate
                ),
                mock.patch.object(
                    subject, "_loopback_listener_exists", return_value=False
                ),
            ):
                manager.start()
                before = manager.transport()
                before_binding = before.bindings.for_route(
                    subject.TOKENHUB_ROUTE_PROFILE_ID
                ).upstream_base_url
                before_key = before.credential
                before_identity = (
                    manager._process.pid,
                    manager._tracker.leader_birth,
                )
                manager.recycle_after_ambiguous()
                after = manager.transport()
                after_identity = (
                    manager._process.pid,
                    manager._tracker.leader_birth,
                )
                self.assertNotEqual(before_identity, after_identity)
                self.assertEqual(after.credential, before_key)
                self.assertEqual(
                    after.bindings.for_route(
                        subject.TOKENHUB_ROUTE_PROFILE_ID
                    ).upstream_base_url,
                    before_binding,
                )
                self.assertEqual(len(captured_environments), 2)
                self.assertTrue(all(
                    item["TOKENHUB_API_KEY"]
                    == manager._identity_relay.capability
                    for item in captured_environments
                ))
                self.assertTrue(all(
                    item["TOKENHUB_API_KEY"] != "fixture-provider-secret"
                    for item in captured_environments
                ))
                receipt = json.loads(
                    (invocation / "transport_recycle_0001.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(receipt["recycle_index"], 1)
                self.assertTrue(receipt["old_process_tree_terminated"])
                self.assertTrue(receipt["old_listener_released_before_restart"])
                self.assertTrue(
                    receipt["provider_identity_relay_upstreams_drained"]
                )
                self.assertTrue(receipt["runtime_reverified_before_restart"])
                manager.close()
            completion = json.loads(
                (invocation / "transport_completion.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                completion["schema_version"],
                "sieve_runtime_transport_completion_v2",
            )
            self.assertEqual(
                set(completion),
                {
                    "schema_version",
                    "adapter_id",
                    "started",
                    "ready",
                    "ended",
                    "listener_released",
                    "recycle_count",
                    "runtime_reverification_count",
                    "provider_identity_relay",
                    "provider_identity_relay_listener_released",
                    "provider_credential_endpoint_request_response_or_hidden_reasoning_recorded",
                },
            )
            self.assertEqual(completion["recycle_count"], 1)
            self.assertEqual(completion["runtime_reverification_count"], 1)

    def test_recycle_reverifies_runtime_before_replacement_spawn(self) -> None:
        process = _FakeProcess(pid=424211)
        snapshot = {
            process.pid: _ProcessInfo(
                ppid=1,
                pgid=process.pid,
                state="S",
                birth="fixture-runtime-drift-birth",
            )
        }

        def terminate(fake: _FakeProcess, _tracker: object) -> None:
            fake.returncode = 0

        with tempfile.TemporaryDirectory(prefix="sieve-managed-drift-") as tmp:
            invocation = Path(tmp)
            manager = subject._TokenHubLiteLLM(
                registry=self.registry,
                experiment=self.experiment,
                bindings=self.bindings,
                provider_credential="fixture-provider-secret",
                invocation_root=invocation,
                runtime_root=Path("/fixture/runtime"),
            )
            with (
                mock.patch.object(
                    subject,
                    "verify_tokenhub_runtime",
                    side_effect=[
                        {"verified": True, "pin": "initial"},
                        {"verified": True, "pin": "drifted"},
                    ],
                ),
                mock.patch.object(subject, "_select_loopback_port", return_value=43229),
                mock.patch.object(
                    subject.subprocess, "Popen", return_value=process
                ) as popen,
                mock.patch.object(subject, "_process_snapshot", return_value=snapshot),
                mock.patch.object(manager, "_wait_until_ready"),
                mock.patch.object(
                    subject, "_terminate_owned_processes", side_effect=terminate
                ),
                mock.patch.object(
                    subject, "_loopback_listener_exists", return_value=False
                ),
            ):
                manager.start()
                with self.assertRaisesRegex(
                    subject.ManagedTransportError,
                    "runtime changed before restart",
                ):
                    manager.recycle_after_ambiguous()
                self.assertEqual(popen.call_count, 1)
                self.assertFalse(
                    (invocation / "transport_recycle_0001.json").exists()
                )
                manager.close()

    def test_snapshot_failure_terminates_just_spawned_adapter(self) -> None:
        process = _FakeProcess(pid=434343)
        with tempfile.TemporaryDirectory(prefix="sieve-managed-fault-") as tmp:
            manager = subject._TokenHubLiteLLM(
                registry=self.registry,
                experiment=self.experiment,
                bindings=self.bindings,
                provider_credential="fixture-provider-secret",
                invocation_root=Path(tmp),
                runtime_root=Path("/fixture/runtime"),
            )
            with (
                mock.patch.object(
                    subject,
                    "verify_tokenhub_runtime",
                    return_value={"verified": True},
                ),
                mock.patch.object(subject, "_select_loopback_port", return_value=43211),
                mock.patch.object(subject.subprocess, "Popen", return_value=process),
                mock.patch.object(
                    subject,
                    "_process_snapshot",
                    side_effect=IsolationError("fixture snapshot failure"),
                ),
                mock.patch.object(subject.os, "killpg") as killpg,
                mock.patch.object(
                    subject, "_loopback_listener_exists", return_value=False
                ),
            ):
                with self.assertRaisesRegex(
                    subject.ManagedTransportError, "process_start_failed"
                ):
                    manager.start()
                manager.close()
            self.assertEqual(
                killpg.call_args_list,
                [
                    mock.call(process.pid, signal.SIGTERM),
                ],
            )
            self.assertEqual(process.returncode, 0)
            self.assertEqual(manager._provider_credential, "")

    def test_untracked_cleanup_proves_group_empty_after_leader_already_exited(self) -> None:
        process = _FakeProcess(pid=444444)
        process.returncode = 0
        manager = subject._TokenHubLiteLLM(
            registry=self.registry,
            experiment=self.experiment,
            bindings=self.bindings,
            provider_credential="fixture-provider-secret",
            invocation_root=Path("/unused"),
            runtime_root=Path("/fixture/runtime"),
        )
        manager._process = process
        with (
            mock.patch.object(isolated_exec.os, "killpg") as killpg,
            mock.patch.object(
                isolated_exec,
                "_process_group_members",
                side_effect=[1, 0],
            ),
        ):
            manager._terminate_untracked_spawn()
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(process.pid, signal.SIGTERM),
                mock.call(process.pid, signal.SIGKILL),
            ],
        )

    def test_close_snapshot_failure_never_signals_a_recyclable_bare_pgid(self) -> None:
        process = _FakeProcess(pid=454545)
        with tempfile.TemporaryDirectory(prefix="sieve-managed-close-fault-") as tmp:
            root = Path(tmp)
            private = root / "private"
            private.mkdir()
            manager = subject._TokenHubLiteLLM(
                registry=self.registry,
                experiment=self.experiment,
                bindings=self.bindings,
                provider_credential="fixture-provider-secret",
                invocation_root=root,
                runtime_root=Path("/fixture/runtime"),
            )
            manager._process = process
            manager._tracker = object()  # type: ignore[assignment]
            manager._temporary_root = private
            manager._port = 43212
            manager._proxy_key = "fixture-local-key"
            manager._ready = True

            with (
                mock.patch.object(
                    subject,
                    "_process_snapshot",
                    side_effect=IsolationError("fixture close snapshot failure"),
                ),
                mock.patch.object(
                    subject,
                    "_terminate_untracked_process_group",
                ) as terminate_group,
                mock.patch.object(
                    subject, "_loopback_listener_exists", return_value=False
                ),
            ):
                with self.assertRaisesRegex(
                    subject.ManagedTransportError, "close_barrier_failed"
                ):
                    manager.close()
            terminate_group.assert_not_called()
            self.assertIsNone(process.returncode)
            self.assertFalse(private.exists())
            completion = json.loads(
                (root / "transport_completion.json").read_text(encoding="utf-8")
            )
            self.assertFalse(completion["ended"])

    def test_direct_adapter_records_completion_even_when_body_raises(self) -> None:
        experiment = load_experiment(
            TRUSTED / "experiments/api3-all-models.example.json",
            self.registry,
        )
        bindings = load_runtime_bindings(
            TRUSTED / "profiles/runtime_bindings.example.json",
            registry=self.registry,
            experiment=experiment,
        )
        with tempfile.TemporaryDirectory(prefix="sieve-direct-transport-") as tmp:
            invocation = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "fixture body failure"):
                with subject.prepare_runtime_transport(
                    registry=self.registry,
                    experiment=experiment,
                    bindings=bindings,
                    provider_credential="fixture-api3-secret",
                    invocation_root=invocation,
                    tokenhub_runtime_root=None,
                ):
                    raise RuntimeError("fixture body failure")
            completion = json.loads(
                (invocation / "transport_completion.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(completion["ended"])
            self.assertEqual(
                set(completion),
                {
                    "schema_version",
                    "adapter_id",
                    "started",
                    "ready",
                    "ended",
                    "listener_released",
                    "recycle_count",
                    "provider_credential_endpoint_request_response_or_hidden_reasoning_recorded",
                },
            )
            public = "".join(
                path.read_text(encoding="utf-8")
                for path in invocation.iterdir()
                if path.is_file()
            )
            self.assertNotIn("fixture-api3-secret", public)

    @unittest.skipUnless(
        DEFAULT_TOKENHUB_RUNTIME.is_dir(),
        "pinned TokenHub LiteLLM runtime is not installed",
    )
    def test_installed_tokenhub_runtime_release_gate_uses_only_loopback_upstream(self) -> None:
        gate = subject.verify_transport_runtime_compatibility(
            registry=self.registry,
            experiment=self.experiment,
            tokenhub_runtime_root=DEFAULT_TOKENHUB_RUNTIME,
            pi_runtime_root=PINNED_PI_RUNTIME,
            bindings=self.bindings,
            verify_managed_lifecycle=True,
        )
        self.assertEqual(
            set(gate),
            {
                "schema_version",
                "ok",
                "adapter_id",
                "managed_process_required",
                "adapter_manifest_sha256",
                "runtime",
                "managed_lifecycle",
                "signed_reasoning_e2e",
                "production_provider_credential_or_route_used",
            },
        )
        self.assertEqual(
            gate["schema_version"], "sieve_transport_runtime_compatibility_v1"
        )
        self.assertTrue(gate["ok"])
        self.assertFalse(
            gate["production_provider_credential_or_route_used"]
        )
        record = gate["runtime"]
        self.assertEqual(record["litellm_version"], "1.83.14")
        self.assertEqual(
            record["reasoning_codec"],
            {
                "input_reasoning_effort": "high",
                "anthropic_thinking": {
                    "type": "enabled",
                    "budget_tokens": 4096,
                },
                "openai_reasoning_effort_forwarded": False,
                "maximum_output_tokens": 65536,
            },
        )
        self.assertEqual(
            gate["managed_lifecycle"],
            {
                "started": True,
                "authenticated_model_inventory_exact": True,
                "raw_provider_identity_verified_before_litellm": True,
                "ended": True,
                "listener_released": True,
                "provider_identity_relay_listener_released": True,
                "upstream_model_request_sent": True,
                "provider_route_kind": "loopback_fixture_only",
            },
        )
        e2e = gate["signed_reasoning_e2e"]
        self.assertTrue(e2e["ok"])
        self.assertTrue(e2e["real_pinned_pi_used"])
        self.assertTrue(e2e["real_pinned_litellm_used"])
        self.assertTrue(e2e["signed_thinking_text_and_signature_exact"])
        self.assertTrue(e2e["tool_use_identity_and_order_exact"])
        self.assertTrue(e2e["tool_result_identity_exact"])
        self.assertTrue(e2e["four_tool_contract_exact"])
        self.assertTrue(e2e["provider_auth_contract_exact"])
        self.assertTrue(e2e["system_prompt_contract_exact"])
        self.assertTrue(e2e["task_prompt_contract_exact"])
        self.assertEqual(
            e2e["anthropic_four_tool_contract_sha256"],
            release_gate.EXPECTED_ANTHROPIC_FOUR_TOOL_SHA256,
        )
        self.assertFalse(e2e["production_provider_credential_used"])
        self.assertTrue(
            e2e["synthetic_loopback_fixture_credential_observed"]
        )
        self.assertTrue(e2e["configured_provider_target_loopback_only"])
        self.assertTrue(e2e["managed_adapter_ended"])
        self.assertTrue(e2e["managed_adapter_listener_released"])
        self.assertTrue(e2e["raw_provider_identity_verified_before_litellm"])
        self.assertEqual(
            e2e["raw_provider_identity_verified_response_count"], 4
        )
        self.assertEqual(
            e2e["raw_provider_identity_rejected_response_count"], 0
        )
        self.assertTrue(e2e["provider_identity_relay_listener_released"])
        public_scan = e2e["public_artifact_safety_scan"]
        self.assertEqual(
            public_scan["schema_version"], "sieve_public_gate_artifact_scan_v1"
        )
        self.assertTrue(public_scan["scan_complete"])
        self.assertEqual(public_scan["roots_scanned"], 2)
        self.assertTrue(public_scan["known_private_values_absent"])
        self.assertTrue(public_scan["raw_credential_headers_absent"])
        self.assertTrue(public_scan["forbidden_raw_artifact_names_absent"])
        boundary = e2e["real_litellm_transport_boundary_probe"]
        self.assertTrue(boundary["wrong_raw_identity_terminal_without_retry"])
        self.assertTrue(boundary["missing_raw_identity_terminal_without_retry"])
        self.assertTrue(
            boundary["wrong_success_content_type_terminal_without_retry"]
        )
        self.assertTrue(
            boundary["truncated_identity_preface_terminal_without_retry"]
        )
        self.assertTrue(
            boundary[
                "provider_2xx_then_litellm_transform_failure_terminal_without_retry"
            ]
        )
        self.assertTrue(
            boundary["explicit_provider_429_retried_once_then_completed"]
        )
        self.assertEqual(boundary["ambiguous_scenarios"], 5)
        self.assertEqual(boundary["provider_requests"], 7)
        self.assertEqual(boundary["managed_transport_recycles"], 5)
        boundary_scan = boundary["public_artifact_safety_scan"]
        self.assertEqual(
            boundary_scan["schema_version"],
            "sieve_public_gate_artifact_scan_v1",
        )
        self.assertTrue(boundary_scan["scan_complete"])
        self.assertEqual(boundary_scan["roots_scanned"], 1)
        self.assertTrue(boundary_scan["known_private_values_absent"])
        self.assertFalse(boundary["provider_request_response_or_reasoning_recorded"])
        self.assertTrue(
            e2e["ambiguous_timeout_disconnect_barrier"][
                "poisoned_gateway_rejected_later_request"
            ]
        )
        self.assertFalse(
            e2e["raw_request_response_reasoning_or_signature_recorded"]
        )

    def test_public_gate_artifact_scan_is_evidence_based_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sieve-public-gate-scan-") as tmp:
            root = Path(tmp)
            (root / "safe.json").write_text('{"safe":true}\n', encoding="utf-8")
            record = release_gate._verify_public_gate_artifacts(
                roots=(root,), forbidden_values=("fixture-private-value",)
            )
            self.assertTrue(record["scan_complete"])
            self.assertEqual(record["regular_files_scanned"], 1)
            self.assertTrue(record["known_private_values_absent"])

            (root / "unsafe.log").write_text(
                "fixture-private-value", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                subject.ManagedTransportError, "known private transport material"
            ):
                release_gate._verify_public_gate_artifacts(
                    roots=(root,), forbidden_values=("fixture-private-value",)
                )
            (root / "unsafe.log").unlink()

            private_name = root / "fixture-private-value"
            private_name.write_bytes(b"")
            with self.assertRaisesRegex(
                subject.ManagedTransportError, "known private transport material"
            ):
                release_gate._verify_public_gate_artifacts(
                    roots=(root,), forbidden_values=("fixture-private-value",)
                )
            private_name.unlink()

            raw_link = root / "request.json"
            raw_link.symlink_to("safe.json")
            with self.assertRaisesRegex(
                subject.ManagedTransportError, "forbidden raw artifact"
            ):
                release_gate._verify_public_gate_artifacts(
                    roots=(root,), forbidden_values=("fixture-private-value",)
                )
            raw_link.unlink()

            (root / "response.json").mkdir()
            with self.assertRaisesRegex(
                subject.ManagedTransportError, "forbidden raw artifact"
            ):
                release_gate._verify_public_gate_artifacts(
                    roots=(root,), forbidden_values=("fixture-private-value",)
                )

    def test_tokenhub_release_fixture_bounds_a_truncated_request(self) -> None:
        with release_gate._FixtureServer() as fixture:
            fixture.state.bind_proxy_key("fixture-local-proxy-capability")
            started = time.monotonic()
            with socket.create_connection(
                ("127.0.0.1", fixture.server.server_address[1]), timeout=2.0
            ) as client:
                client.settimeout(release_gate.FIXTURE_READ_TIMEOUT_SECONDS + 3.0)
                request = (
                    "POST /v1/messages HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n"
                    f"x-api-key: {release_gate.FIXTURE_PROVIDER_KEY}\r\n"
                    "anthropic-version: 2023-06-01\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: 100\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                    "{"
                ).encode("ascii")
                client.sendall(request)
                response = client.recv(4096)
            elapsed = time.monotonic() - started
            self.assertIn(b" 400 ", response)
            self.assertLess(
                elapsed, release_gate.FIXTURE_READ_TIMEOUT_SECONDS + 3.0
            )
            self.assertEqual(
                fixture.state.safe_record()["validation_error_codes"],
                ["anthropic_request_body_read_failed"],
            )

    def test_transport_boundary_fixture_bounds_a_slow_request_body(self) -> None:
        started = time.monotonic()
        with mock.patch.object(
            release_gate, "FIXTURE_READ_TIMEOUT_SECONDS", 0.1
        ):
            with release_gate._TransportBoundaryFixture() as fixture:
                with socket.create_connection(
                    ("127.0.0.1", fixture.server.server_address[1]),
                    timeout=1.0,
                ) as client:
                    client.sendall(
                        b"POST /v1/messages HTTP/1.0\r\n"
                        b"Host: 127.0.0.1\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Content-Length: 128\r\n\r\n"
                        b"{"
                    )
                    time.sleep(0.2)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_tokenhub_release_fixture_rejects_embedded_proxy_capability(self) -> None:
        capability = "fixture-local-proxy-capability"
        body = json.dumps({"unexpected": f"prefix-{capability}-suffix"}).encode(
            "utf-8"
        )
        with release_gate._FixtureServer() as fixture:
            fixture.state.bind_proxy_key(capability)
            with socket.create_connection(
                ("127.0.0.1", fixture.server.server_address[1]), timeout=2.0
            ) as client:
                client.settimeout(3.0)
                request = (
                    "POST /v1/messages HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n"
                    f"x-api-key: {release_gate.FIXTURE_PROVIDER_KEY}\r\n"
                    "anthropic-version: 2023-06-01\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                ).encode("ascii") + body
                client.sendall(request)
                response = client.recv(4096)
            self.assertIn(b" 400 ", response)
            self.assertEqual(
                fixture.state.safe_record()["validation_error_codes"],
                ["local_proxy_key_leaked_in_upstream_body"],
            )

    def test_tokenhub_release_gate_rejects_extra_top_level_request_fields(self) -> None:
        payload = {
            name: None for name in release_gate.EXPECTED_ANTHROPIC_REQUEST_FIELDS
        }
        payload["unexpected"] = True
        with self.assertRaisesRegex(
            release_gate._FixtureContractError, "first_request_fields_mismatch"
        ):
            release_gate._validate_first_request(
                payload, release_gate._FixtureState()
            )


if __name__ == "__main__":
    unittest.main()
