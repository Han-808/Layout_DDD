from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
if str(TRUSTED) not in sys.path:
    sys.path.insert(0, str(TRUSTED))

from api_profiles import (  # noqa: E402
    ProfileRegistry,
    load_experiment,
    load_runtime_bindings,
)
from model_gateway import SharedCooldownGate  # noqa: E402
from preflight_pi_route import PiRoutePreflightReport  # noqa: E402
from run_pi_episode import EpisodeOutcome  # noqa: E402
import run_pi_experiment as subject  # noqa: E402


class ExperimentResumePlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProfileRegistry.load()
        source = load_experiment(
            TRUSTED / "experiments/api2-all-models.example.json",
            cls.registry,
        )
        cls.model_id = source.model_profile_ids[0]
        cls.scene_ids = source.scene_ids[:2]
        cls.experiment = replace(
            source,
            model_profile_ids=(cls.model_id,),
            scene_ids=cls.scene_ids,
        )
        cls.bindings = load_runtime_bindings(
            TRUSTED / "profiles/runtime_bindings.example.json",
            registry=cls.registry,
            experiment=cls.experiment,
        )

    def test_fully_sealed_model_resumes_without_any_live_preflight(self) -> None:
        outcomes = {
            scene_id: self._complete(scene_id) for scene_id in self.scene_ids
        }
        with tempfile.TemporaryDirectory(prefix="sieve-resume-full-") as temporary:
            invocation = Path(temporary)
            with (
                patch.object(
                    subject,
                    "inspect_episode_if_present",
                    side_effect=lambda _spec, scene_id: outcomes[scene_id],
                ) as inspect,
                patch.object(subject, "run_preflights") as preflight,
                patch.object(subject, "run_one_episode") as run,
            ):
                summary = self._execute(invocation)

        self.assertEqual(inspect.call_count, 2)
        preflight.assert_not_called()
        run.assert_not_called()
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["complete"], 2)
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["preflight"], {self.model_id: None})
        self.assertEqual(summary["live_preflight_models"], [])
        self.assertEqual(summary["preflight_not_required_models"], [self.model_id])
        self.assertEqual(
            summary["preflight_not_required_reason"],
            "listed_models_have_all_exact_episode_paths_present",
        )
        self.assertTrue(all(item["resumed"] for item in summary["outcomes"]))
        self.assertEqual(
            set(summary),
            {
                "schema_version",
                "experiment_id",
                "api_family_id",
                "status",
                "expected_episodes",
                "terminal_records",
                "complete",
                "failed",
                "skipped",
                "missing",
                "stopped_early",
                "transport_adapter_id",
                "transport_started",
                "transport_recycle_count",
                "transport_recovery_barrier_count",
                "preflight",
                "live_preflight_models",
                "preflight_not_required_models",
                "preflight_not_required_reason",
                "outcomes",
                "credential_endpoint_request_response_or_hidden_reasoning_recorded",
            },
        )
        self.assertEqual(summary["schema_version"], "sieve_pi_experiment_summary_v3")
        self.assertEqual(
            set(summary["outcomes"][0]),
            {
                "schema_version",
                "status",
                "experiment_id",
                "model_profile_id",
                "scene_id",
                "attempt",
                "run_id",
                "episode_relative_path",
                "resumed",
                "failure_category",
                "failure_code",
                "submission_sha256",
                "elapsed_seconds",
                "transport_recycle_required",
                "credential_endpoint_or_hidden_reasoning_recorded",
            },
        )
        self.assertEqual(
            summary["outcomes"][0]["schema_version"],
            "sieve_pi_episode_outcome_v2",
        )

    def test_mixed_model_resume_reason_applies_only_to_listed_models(self) -> None:
        source = load_experiment(
            TRUSTED / "experiments/api2-all-models.example.json",
            self.registry,
        )
        resumed_model, live_model = source.model_profile_ids[:2]
        scene_id = source.scene_ids[0]
        experiment = replace(
            source,
            model_profile_ids=(resumed_model, live_model),
            scene_ids=(scene_id,),
        )
        bindings = load_runtime_bindings(
            TRUSTED / "profiles/runtime_bindings.example.json",
            registry=self.registry,
            experiment=experiment,
        )
        resumed = EpisodeOutcome(
            status="complete",
            experiment_id=experiment.experiment_id,
            model_profile_id=resumed_model,
            scene_id=scene_id,
            attempt=1,
            run_id="run-resumed",
            episode_relative_path="episodes/resumed",
            resumed=True,
            failure_category=None,
            failure_code=None,
            submission_sha256="c" * 64,
            elapsed_seconds=1.0,
        )
        completed_live = replace(
            resumed,
            model_profile_id=live_model,
            run_id="run-live",
            episode_relative_path="episodes/live",
            resumed=False,
        )
        existing = {
            (resumed_model, scene_id): resumed,
            (live_model, scene_id): None,
        }

        with tempfile.TemporaryDirectory(prefix="sieve-resume-mixed-model-") as temporary:
            with (
                patch.object(
                    subject,
                    "run_preflights",
                    return_value={live_model: True},
                ) as preflight,
                patch.object(
                    subject,
                    "run_one_episode",
                    return_value=completed_live,
                ) as run,
                redirect_stdout(io.StringIO()),
            ):
                summary = subject.execute_experiment(
                    registry=self.registry,
                    experiment=experiment,
                    bindings=bindings,
                    credential="fixture-app:fixture-key",
                    runtime_root=ARENA_ROOT / "../runtime_bundles/pi-0.85.0",
                    resource_bindings=Path("/fixture/resource-bindings.json"),
                    invocation_root=Path(temporary),
                    existing=existing,
                )

        self.assertEqual(
            preflight.call_args.kwargs["model_profile_ids"], (live_model,)
        )
        run.assert_called_once()
        self.assertEqual(summary["live_preflight_models"], [live_model])
        self.assertEqual(
            summary["preflight_not_required_models"], [resumed_model]
        )
        self.assertEqual(
            summary["preflight_not_required_reason"],
            "listed_models_have_all_exact_episode_paths_present",
        )

    def test_preflight_and_preflight_summary_public_schemas_are_exact(self) -> None:
        report = PiRoutePreflightReport(
            ok=True,
            model_profile_id=self.model_id,
            api_family_id=self.experiment.api_family_id,
            route_profile_id=self.registry.model(self.model_id).route_profile_id,
            protocol="openai-completions",
            audit_relative_path="episodes/route-preflight/fixture",
            process_status="exited_zero",
            process_returncode_zero=True,
            logical_requests=2,
            gateway_stream_contract_complete=True,
            pinned_pi_tool_roundtrip_complete=True,
            tool_calls_started=1,
            tool_calls_ended=1,
            exactly_one_read_call=True,
            final_marker_exact=True,
            reasoning_replay_required=True,
            reasoning_replayed_on_tool_followup=True,
            response_identity_required=True,
            response_identity_matches=True,
            routing_identity_assurance="request_and_response_identity_gated",
            transport_recycle_required=False,
            failure_category=None,
            failure_code=None,
            elapsed_seconds=1.0,
        ).public_dict()
        self.assertEqual(report["schema_version"], "sieve_pi_route_preflight_report_v2")
        self.assertEqual(
            set(report),
            {
                "schema_version",
                "created_at_utc",
                "ok",
                "model_profile_id",
                "api_family_id",
                "route_profile_id",
                "protocol",
                "audit_relative_path",
                "process_status",
                "process_returncode_zero",
                "logical_requests",
                "gateway_stream_contract_complete",
                "pinned_pi_tool_roundtrip_complete",
                "tool_calls_started",
                "tool_calls_ended",
                "exactly_one_read_call",
                "final_marker_exact",
                "reasoning_replay_required",
                "reasoning_replayed_on_tool_followup",
                "response_identity_required",
                "response_identity_matches",
                "routing_identity_assurance",
                "transport_recycle_required",
                "failure_category",
                "failure_code",
                "elapsed_seconds",
                "real_pinned_pi_process_used",
                "handwritten_provider_payload_used",
                "raw_request_recorded",
                "raw_response_recorded",
                "hidden_reasoning_recorded",
                "credential_headers_or_endpoint_recorded",
            },
        )

        with tempfile.TemporaryDirectory(prefix="sieve-preflight-summary-") as temporary:
            with patch.object(
                subject, "run_preflights", return_value={self.model_id: True}
            ):
                summary = subject.preflight_only(
                    registry=self.registry,
                    experiment=self.experiment,
                    bindings=self.bindings,
                    credential="fixture-app:fixture-key",
                    invocation_root=Path(temporary),
                    runtime_root=ARENA_ROOT / "../runtime_bundles/pi-0.85.0",
                    transport_adapter_id="direct-scoped-gateway-v1",
                )
        self.assertEqual(
            summary["schema_version"], "sieve_pi_experiment_preflight_summary_v2"
        )
        self.assertEqual(
            set(summary),
            {
                "schema_version",
                "experiment_id",
                "api_family_id",
                "ok",
                "model_results",
                "generation_started",
                "transport_adapter_id",
                "transport_started",
                "transport_recycle_count",
                "transport_recovery_barrier_count",
                "credential_endpoint_request_response_or_hidden_reasoning_recorded",
            },
        )

    def test_partial_resume_preflight_failure_skips_only_missing_scene(self) -> None:
        existing_scene, missing_scene = self.scene_ids

        def inspect(_spec: object, scene_id: str) -> EpisodeOutcome | None:
            return self._complete(scene_id) if scene_id == existing_scene else None

        with tempfile.TemporaryDirectory(prefix="sieve-resume-partial-") as temporary:
            invocation = Path(temporary)
            with (
                patch.object(
                    subject,
                    "inspect_episode_if_present",
                    side_effect=inspect,
                ),
                patch.object(
                    subject,
                    "run_preflights",
                    return_value={self.model_id: False},
                ) as preflight,
                patch.object(subject, "run_one_episode") as run,
            ):
                summary = self._execute(invocation)

        run.assert_not_called()
        self.assertEqual(
            preflight.call_args.kwargs["model_profile_ids"], (self.model_id,)
        )
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["complete"], 1)
        self.assertEqual(summary["skipped"], 1)
        by_scene = {row["scene_id"]: row for row in summary["outcomes"]}
        self.assertTrue(by_scene[existing_scene]["resumed"])
        self.assertEqual(by_scene[missing_scene]["status"], "skipped")
        self.assertEqual(
            by_scene[missing_scene]["failure_code"], "preflight_failed"
        )

    def test_main_full_resume_does_not_acquire_key_or_start_transport(self) -> None:
        tokenhub = load_experiment(
            TRUSTED / "experiments/tokenhub-hy4.example.json",
            self.registry,
        )
        existing = {
            (model_id, scene_id): EpisodeOutcome(
                status="complete",
                experiment_id=tokenhub.experiment_id,
                model_profile_id=model_id,
                scene_id=scene_id,
                attempt=1,
                run_id=f"run-{scene_id}",
                episode_relative_path=f"episodes/{scene_id}",
                resumed=True,
                failure_category=None,
                failure_code=None,
                submission_sha256="b" * 64,
                elapsed_seconds=1.0,
            )
            for model_id in tokenhub.model_profile_ids
            for scene_id in tokenhub.scene_ids
        }
        with tempfile.TemporaryDirectory(prefix="sieve-main-full-resume-") as temporary:
            root = Path(temporary)
            resource_bindings = root / "resources.json"
            resource_bindings.write_text("{}\n", encoding="utf-8")
            invocation = root / "invocation"
            invocation.mkdir()
            acquire = Mock(side_effect=AssertionError("credential must not be read"))
            transport = Mock(side_effect=AssertionError("transport must not start"))
            runtime_gate = Mock(
                side_effect=AssertionError("runtime gate must not start on full resume")
            )
            plan = {
                "experiment": {"experiment_id": tokenhub.experiment_id},
                "execution_fingerprint_sha256": "c" * 64,
            }
            with (
                patch.object(subject.ProfileRegistry, "load", return_value=self.registry),
                patch.object(subject, "verify_runtime", return_value={}),
                patch.object(subject, "verify_lock", return_value={}),
                patch.object(subject, "build_plan", return_value=plan),
                patch.object(subject, "_prepare_invocation", return_value=invocation),
                patch.object(subject, "ApiFamilyInvocationLock", side_effect=lambda **_: nullcontext()),
                patch.object(subject, "inspect_existing_matrix", return_value=existing),
                patch.object(subject, "_acquire_credential", acquire),
                patch.object(subject, "prepare_runtime_transport", transport),
                patch.object(
                    subject, "verify_transport_runtime_compatibility", runtime_gate
                ),
                redirect_stdout(io.StringIO()),
            ):
                status = subject.main(
                    [
                        "--experiment",
                        str(TRUSTED / "experiments/tokenhub-hy4.example.json"),
                        "--runtime-bindings",
                        str(TRUSTED / "profiles/runtime_bindings.example.json"),
                        "--runtime-root",
                        str(ARENA_ROOT / "../runtime_bundles/pi-0.85.0"),
                        "--resource-bindings",
                        str(resource_bindings),
                        "--execute",
                    ]
                )
            self.assertEqual(status, 0)
            acquire.assert_not_called()
            transport.assert_not_called()
            runtime_gate.assert_not_called()
            marker = subject._read_json(invocation / "transport_not_started.json")
            self.assertFalse(marker["credential_acquired"])
            self.assertFalse(marker["live_preflight_required"])

    def test_main_partial_resume_acquires_once_and_starts_one_family_transport(self) -> None:
        tokenhub = load_experiment(
            TRUSTED / "experiments/tokenhub-hy4.example.json",
            self.registry,
        )
        tokenhub_bindings = load_runtime_bindings(
            TRUSTED / "profiles/runtime_bindings.example.json",
            registry=self.registry,
            experiment=tokenhub,
        )
        existing = {
            (model_id, scene_id): None
            for model_id in tokenhub.model_profile_ids
            for scene_id in tokenhub.scene_ids
        }
        with tempfile.TemporaryDirectory(prefix="sieve-main-partial-resume-") as temporary:
            root = Path(temporary)
            resource_bindings = root / "resources.json"
            resource_bindings.write_text("{}\n", encoding="utf-8")
            adapter_runtime = root / "adapter-runtime"
            adapter_runtime.mkdir()
            invocation = root / "invocation"
            invocation.mkdir()
            plan = {
                "experiment": {"experiment_id": tokenhub.experiment_id},
                "execution_fingerprint_sha256": "d" * 64,
            }
            health = Mock()
            recycle = Mock()
            runtime_transport = SimpleNamespace(
                bindings=tokenhub_bindings,
                credential="fixture-local-proxy-capability",
                adapter_id="managed_tokenhub_litellm_v1",
                ensure_healthy=health,
                recycle_after_ambiguous=recycle,
                post_send_ambiguity_detected=lambda: False,
                classify_outer_result=lambda _complete: False,
            )
            acquire = Mock(return_value="fixture-provider-credential")
            prepare = Mock(return_value=nullcontext(runtime_transport))
            runtime_gate = Mock(
                return_value={
                    "schema_version": "sieve_transport_runtime_compatibility_v1",
                    "ok": True,
                }
            )
            execute = Mock(return_value={"status": "complete"})
            with (
                patch.object(subject.ProfileRegistry, "load", return_value=self.registry),
                patch.object(subject, "verify_runtime", return_value={}),
                patch.object(subject, "verify_lock", return_value={}),
                patch.object(subject, "build_plan", return_value=plan),
                patch.object(subject, "_prepare_invocation", return_value=invocation),
                patch.object(subject, "ApiFamilyInvocationLock", side_effect=lambda **_: nullcontext()),
                patch.object(subject, "inspect_existing_matrix", return_value=existing),
                patch.object(subject, "_acquire_credential", acquire),
                patch.object(subject, "prepare_runtime_transport", prepare),
                patch.object(
                    subject,
                    "verify_transport_runtime_compatibility",
                    runtime_gate,
                ),
                patch.object(subject, "execute_experiment", execute),
                redirect_stdout(io.StringIO()),
            ):
                status = subject.main(
                    [
                        "--experiment",
                        str(TRUSTED / "experiments/tokenhub-hy4.example.json"),
                        "--runtime-bindings",
                        str(TRUSTED / "profiles/runtime_bindings.example.json"),
                        "--runtime-root",
                        str(ARENA_ROOT / "../runtime_bundles/pi-0.85.0"),
                        "--tokenhub-litellm-runtime",
                        str(adapter_runtime),
                        "--resource-bindings",
                        str(resource_bindings),
                        "--execute",
                    ]
                )
            self.assertEqual(status, 0)
            runtime_gate.assert_called_once()
            acquire.assert_called_once()
            prepare.assert_called_once()
            self.assertEqual(
                prepare.call_args.kwargs["provider_credential"],
                "fixture-provider-credential",
            )
            self.assertEqual(
                prepare.call_args.kwargs["tokenhub_runtime_root"],
                adapter_runtime.resolve(),
            )
            execute.assert_called_once()
            receipt = subject._read_json(
                invocation / "runtime_compatibility_gate.json"
            )
            self.assertTrue(receipt["transport"]["ok"])
            self.assertFalse(
                receipt["production_provider_credential_or_route_used"]
            )
            self.assertEqual(
                execute.call_args.kwargs["credential"],
                "fixture-local-proxy-capability",
            )
            self.assertIs(
                execute.call_args.kwargs["transport_health_check"], health
            )
            self.assertIs(
                execute.call_args.kwargs[
                    "transport_recycle_after_ambiguous"
                ],
                recycle,
            )

    def test_main_direct_transport_passes_no_managed_sideband(self) -> None:
        direct = load_experiment(
            TRUSTED / "experiments/api2-all-models.example.json",
            self.registry,
        )
        direct_bindings = load_runtime_bindings(
            TRUSTED / "profiles/runtime_bindings.example.json",
            registry=self.registry,
            experiment=direct,
        )
        existing = {
            (model_id, scene_id): None
            for model_id in direct.model_profile_ids
            for scene_id in direct.scene_ids
        }
        with tempfile.TemporaryDirectory(prefix="sieve-main-direct-") as temporary:
            root = Path(temporary)
            resource_bindings = root / "resources.json"
            resource_bindings.write_text("{}\n", encoding="utf-8")
            invocation = root / "invocation"
            invocation.mkdir()
            classifier = Mock(side_effect=AssertionError("direct route used sideband"))
            runtime_transport = SimpleNamespace(
                bindings=direct_bindings,
                credential="fixture-app:fixture-key",
                adapter_id="direct_http_v1",
                ensure_healthy=Mock(),
                recycle_after_ambiguous=Mock(return_value=False),
                classify_outer_result=classifier,
            )
            execute = Mock(return_value={"status": "complete"})
            with (
                patch.object(subject.ProfileRegistry, "load", return_value=self.registry),
                patch.object(subject, "verify_runtime", return_value={}),
                patch.object(subject, "verify_lock", return_value={}),
                patch.object(
                    subject,
                    "build_plan",
                    return_value={
                        "experiment": {"experiment_id": direct.experiment_id},
                        "execution_fingerprint_sha256": "e" * 64,
                    },
                ),
                patch.object(subject, "_prepare_invocation", return_value=invocation),
                patch.object(
                    subject,
                    "ApiFamilyInvocationLock",
                    side_effect=lambda **_: nullcontext(),
                ),
                patch.object(subject, "inspect_existing_matrix", return_value=existing),
                patch.object(subject, "_acquire_credential", return_value="fixture-app:fixture-key"),
                patch.object(
                    subject,
                    "verify_transport_runtime_compatibility",
                    return_value={
                        "schema_version": "sieve_transport_runtime_compatibility_v1",
                        "ok": True,
                    },
                ),
                patch.object(
                    subject,
                    "prepare_runtime_transport",
                    return_value=nullcontext(runtime_transport),
                ),
                patch.object(subject, "execute_experiment", execute),
                redirect_stdout(io.StringIO()),
            ):
                status = subject.main(
                    [
                        "--experiment",
                        str(TRUSTED / "experiments/api2-all-models.example.json"),
                        "--runtime-bindings",
                        str(TRUSTED / "profiles/runtime_bindings.example.json"),
                        "--runtime-root",
                        str(ARENA_ROOT / "../runtime_bundles/pi-0.85.0"),
                        "--resource-bindings",
                        str(resource_bindings),
                        "--execute",
                    ]
                )
        self.assertEqual(status, 0)
        execute.assert_called_once()
        self.assertIsNone(
            execute.call_args.kwargs["managed_transport_ambiguity_probe"]
        )
        classifier.assert_not_called()

    def test_preflight_recycles_before_returning_after_ambiguous_state(self) -> None:
        report = SimpleNamespace(
            ok=False,
            failure_category="transport",
            failure_code="ambiguous_stream_failure",
            transport_recycle_required=True,
        )
        recycle = Mock(return_value=True)
        with tempfile.TemporaryDirectory(prefix="sieve-preflight-recycle-") as temporary:
            with (
                patch.object(subject, "run_pi_route_preflight", return_value=report),
                patch.object(subject, "write_pi_route_preflight_report"),
                redirect_stdout(io.StringIO()),
            ):
                results = subject.run_preflights(
                    registry=self.registry,
                    experiment=self.experiment,
                    bindings=self.bindings,
                    credential="fixture-app:fixture-key",
                    cooldown_gate=SharedCooldownGate(),
                    invocation_root=Path(temporary),
                    runtime_root=ARENA_ROOT / "../runtime_bundles/pi-0.85.0",
                    transport_recycle_after_ambiguous=recycle,
                )
        self.assertEqual(results, {self.model_id: False})
        recycle.assert_called_once_with()

    def test_preflight_rejects_non_boolean_recycle_result(self) -> None:
        report = SimpleNamespace(
            ok=False,
            failure_category="transport",
            failure_code="ambiguous_stream_failure",
            transport_recycle_required=True,
        )
        with tempfile.TemporaryDirectory(prefix="sieve-preflight-bad-recycle-") as temporary:
            with (
                patch.object(subject, "run_pi_route_preflight", return_value=report),
                patch.object(subject, "write_pi_route_preflight_report"),
                self.assertRaisesRegex(
                    subject.ExperimentRunError, "non-boolean"
                ),
                redirect_stdout(io.StringIO()),
            ):
                subject.run_preflights(
                    registry=self.registry,
                    experiment=self.experiment,
                    bindings=self.bindings,
                    credential="fixture-app:fixture-key",
                    cooldown_gate=SharedCooldownGate(),
                    invocation_root=Path(temporary),
                    runtime_root=ARENA_ROOT / "../runtime_bundles/pi-0.85.0",
                    transport_recycle_after_ambiguous=lambda: None,  # type: ignore[arg-type,return-value]
                )

    def test_episode_recovery_barrier_runs_before_next_scene(self) -> None:
        first_scene, second_scene = self.scene_ids
        order: list[str] = []
        outcomes = {
            first_scene: EpisodeOutcome(
                status="failed",
                experiment_id=self.experiment.experiment_id,
                model_profile_id=self.model_id,
                scene_id=first_scene,
                attempt=1,
                run_id="run-first",
                episode_relative_path="episodes/first",
                resumed=False,
                failure_category="transport",
                failure_code="ambiguous_stream_failure",
                submission_sha256=None,
                elapsed_seconds=1.0,
                transport_recycle_required=True,
            ),
            second_scene: self._complete(second_scene),
        }

        def run(_spec: object, scene_id: str) -> EpisodeOutcome:
            order.append(f"run:{scene_id}")
            return outcomes[scene_id]

        def recycle() -> bool:
            order.append("recycle")
            return True

        existing = {
            (self.model_id, scene_id): None for scene_id in self.scene_ids
        }
        with tempfile.TemporaryDirectory(prefix="sieve-episode-recycle-") as temporary:
            with (
                patch.object(
                    subject,
                    "run_preflights",
                    return_value={self.model_id: True},
                ),
                patch.object(subject, "run_one_episode", side_effect=run),
                redirect_stdout(io.StringIO()),
            ):
                summary = subject.execute_experiment(
                    registry=self.registry,
                    experiment=self.experiment,
                    bindings=self.bindings,
                    credential="fixture-app:fixture-key",
                    runtime_root=ARENA_ROOT / "../runtime_bundles/pi-0.85.0",
                    resource_bindings=Path("/fixture/resource-bindings.json"),
                    invocation_root=Path(temporary),
                    existing=existing,
                    transport_recycle_after_ambiguous=recycle,
                )
        self.assertEqual(
            order,
            [f"run:{first_scene}", "recycle", f"run:{second_scene}"],
        )
        self.assertEqual(summary["transport_recovery_barrier_count"], 1)
        self.assertEqual(summary["transport_recycle_count"], 1)

    def test_direct_ambiguity_counts_barrier_but_not_managed_recycle(self) -> None:
        scene = self.scene_ids[0]
        experiment = replace(self.experiment, scene_ids=(scene,))
        outcome = EpisodeOutcome(
            status="failed",
            experiment_id=experiment.experiment_id,
            model_profile_id=self.model_id,
            scene_id=scene,
            attempt=1,
            run_id="run-direct-ambiguity",
            episode_relative_path="episodes/direct-ambiguity",
            resumed=False,
            failure_category="transport",
            failure_code="ambiguous_upstream_transport",
            submission_sha256=None,
            elapsed_seconds=1.0,
            transport_recycle_required=True,
        )
        recycle = Mock(return_value=False)
        with tempfile.TemporaryDirectory(prefix="sieve-direct-barrier-") as temporary:
            with (
                patch.object(
                    subject,
                    "run_preflights",
                    return_value={self.model_id: True},
                ),
                patch.object(subject, "run_one_episode", return_value=outcome),
                redirect_stdout(io.StringIO()),
            ):
                summary = subject.execute_experiment(
                    registry=self.registry,
                    experiment=experiment,
                    bindings=self.bindings,
                    credential="fixture-app:fixture-key",
                    runtime_root=ARENA_ROOT / "../runtime_bundles/pi-0.85.0",
                    resource_bindings=Path("/fixture/resource-bindings.json"),
                    invocation_root=Path(temporary),
                    existing={(self.model_id, scene): None},
                    transport_recycle_after_ambiguous=recycle,
                )
        recycle.assert_called_once_with()
        self.assertEqual(summary["transport_recovery_barrier_count"], 1)
        self.assertEqual(summary["transport_recycle_count"], 0)

    def _execute(self, invocation: Path) -> dict[str, object]:
        return subject.execute_experiment(
            registry=self.registry,
            experiment=self.experiment,
            bindings=self.bindings,
            credential="fixture-app:fixture-key",
            runtime_root=ARENA_ROOT / "../runtime_bundles/pi-0.85.0",
            resource_bindings=Path("/fixture/resource-bindings.json"),
            invocation_root=invocation,
        )

    def _complete(self, scene_id: str) -> EpisodeOutcome:
        return EpisodeOutcome(
            status="complete",
            experiment_id=self.experiment.experiment_id,
            model_profile_id=self.model_id,
            scene_id=scene_id,
            attempt=1,
            run_id=f"run-{scene_id}",
            episode_relative_path=f"episodes/{scene_id}",
            resumed=True,
            failure_category=None,
            failure_code=None,
            submission_sha256="a" * 64,
            elapsed_seconds=1.0,
        )


if __name__ == "__main__":
    unittest.main()
