#!/usr/bin/env python3
"""Run one pristine, sealed SIEVE FloorPlan episode through pinned Pi.

This module is the trusted episode boundary.  It never acquires credentials;
the experiment supervisor passes one already-normalized in-memory credential.
Every episode receives its own database capability, model-gateway capability,
workspace, event log, stdout/stderr logs, and immutable run identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping

from api_profiles import (
    ExperimentProfile,
    ModelProfile,
    ProfileRegistry,
    RouteProfile,
    RouteRuntimeBinding,
)
from arena import (
    ARENA_ROOT,
    ArenaError,
    Episode,
    create_episode,
    fixed_case,
    read_json,
    verify_episode_inputs,
)
from database_host import (
    EpisodeDatabase,
    arena_tool_policy,
    collect_and_normalize,
    verify_normalized_artifact_manifest,
)
from benchmark.scene_generation.non_rectangular_agent.tool_server import (  # noqa: E402
    AgentToolShutdownError,
    verify_tool_event_journal,
)
from isolated_exec import (
    MAX_TOTAL_LOG_BYTES,
    MAX_WORKSPACE_BYTES,
    MAX_WORKSPACE_ENTRIES,
    IsolationError,
    run_isolated,
)
from managed_transport import transport_contract_for_route
from model_gateway import (
    GatewayError,
    GatewayShutdownError,
    ScopedModelGateway,
    SharedCooldownGate,
    build_gateway_public_record,
    completion_requires_transport_recycle,
    verify_gateway_audit,
)
from pi_harness import (
    PiEpisodeConfig,
    PiHarnessError,
    allowed_pi_system_prompt_sha256s,
    prepare_episode,
    system_prompt_rewrite_map,
    verify_existing_episode_launch,
    verify_prepared_episode_material,
    verify_runtime,
)
from pi_tool_transcript import (
    PiToolTranscriptError,
    project_pi_tool_transcript,
    verify_pi_tool_transcript,
)
from verify_arena import ArenaLockError, verify_lock


class EpisodeRunError(RuntimeError):
    """Raised for invalid trusted orchestration input."""


@dataclass(frozen=True)
class EpisodeRunSpec:
    experiment: ExperimentProfile
    registry: ProfileRegistry
    model: ModelProfile
    route: RouteProfile
    runtime_binding: RouteRuntimeBinding
    runtime_credential: str
    runtime_root: Path
    resource_bindings: Path
    cooldown_gate: SharedCooldownGate
    attempt: int
    managed_transport_ambiguity_probe: Callable[[bool], bool] | None = None


@dataclass(frozen=True)
class EpisodeOutcome:
    status: str
    experiment_id: str
    model_profile_id: str
    scene_id: str
    attempt: int
    run_id: str
    episode_relative_path: str
    resumed: bool
    failure_category: str | None
    failure_code: str | None
    submission_sha256: str | None
    elapsed_seconds: float | None
    transport_recycle_required: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sieve_pi_episode_outcome_v2",
            **asdict(self),
            "credential_endpoint_or_hidden_reasoning_recorded": False,
        }


def build_run_identity(spec: EpisodeRunSpec, scene_id: str) -> dict[str, Any]:
    if spec.model.api_family_id != spec.experiment.api_family_id:
        raise EpisodeRunError("experiment_model_api_family_mismatch")
    if spec.route.route_profile_id != spec.model.route_profile_id:
        raise EpisodeRunError("route_model_profile_mismatch")
    if spec.runtime_binding.route_profile_id != spec.route.route_profile_id:
        raise EpisodeRunError("route_runtime_binding_mismatch")
    if spec.attempt < 1 or spec.attempt > spec.experiment.episode_attempts:
        raise EpisodeRunError("episode_attempt_out_of_range")
    runtime = verify_runtime(spec.runtime_root)
    arena_lock = verify_lock()
    model_hash = _canonical_hash(spec.model.public_dict())
    route_hash = _canonical_hash(spec.route.public_dict())
    identity = {
        "schema_version": "sieve_pi_episode_run_identity_v1",
        "experiment_id": spec.experiment.experiment_id,
        "experiment_sha256": spec.experiment.source_sha256,
        "profile_registry_sha256": spec.registry.content_sha256,
        "api_family_id": spec.experiment.api_family_id,
        "model_profile_id": spec.model.model_profile_id,
        "model_profile_sha256": model_hash,
        "route_profile_id": spec.route.route_profile_id,
        "route_profile_sha256": route_hash,
        "transport_contract": transport_contract_for_route(
            spec.route.transport_adapter_id
        ),
        "runtime_binding": {
            "binding_profile_id": spec.runtime_binding.binding_profile_id,
            "endpoint": "operator_private_unrecorded",
        },
        "scene_id": scene_id,
        "harness_id": spec.experiment.harness_id,
        "runtime_content_root_sha256": runtime["content_root_sha256"],
        "runtime_manifest_sha256": runtime["runtime_manifest_sha256"],
        "arena_content_root_sha256": arena_lock["content_root_sha256"],
        "database_snapshot_id": read_json(
            ARENA_ROOT / "fixed_suite/shared_database_contract.json"
        )["snapshot_id"],
        "limits": {
            "maximum_model_requests": spec.experiment.maximum_model_requests,
            "wall_clock_seconds": spec.experiment.wall_clock_seconds,
            "maximum_concurrent_tool_calls": 1,
        },
        "resume_policy": spec.experiment.resume_policy,
        "attempt": spec.attempt,
        "credential_recorded_or_hashed": False,
        "endpoint_recorded_or_hashed": False,
    }
    identity["run_fingerprint_sha256"] = _canonical_hash(identity)
    return identity


def run_one_episode(spec: EpisodeRunSpec, scene_id: str) -> EpisodeOutcome:
    identity = build_run_identity(spec, scene_id)
    run_id = f"run-{identity['run_fingerprint_sha256'][:24]}"
    agent_id = _agent_id(spec.experiment.agent_id_prefix, spec.model.model_profile_id)
    root = ARENA_ROOT / "episodes" / agent_id / scene_id / run_id
    if root.exists() or root.is_symlink():
        existing = inspect_episode_if_present(spec, scene_id)
        if existing is None:  # pragma: no cover - root was checked above
            raise EpisodeRunError("existing_episode_disappeared")
        return existing

    episode: Episode | None = None
    process_elapsed: float | None = None
    gateway_completion: dict[str, Any] | None = None
    gateway_started = False
    transport_recycle_required = False
    try:
        episode = create_episode(agent_id=agent_id, scene_id=scene_id, run_id=run_id)
        _write_json_exclusive(episode.host / "run_identity.json", identity, 0o444)
        allowed_system_prompts = allowed_pi_system_prompt_sha256s(
            episode.workspace
        )
        prompt_rewrites = system_prompt_rewrite_map(episode.workspace)
        database = EpisodeDatabase(
            episode=episode,
            resource_bindings=spec.resource_bindings,
        )
        with database:
            with ScopedModelGateway(
                route=spec.route,
                model=spec.model,
                runtime_binding=spec.runtime_binding,
                runtime_credential=spec.runtime_credential,
                max_requests=spec.experiment.maximum_model_requests,
                event_path=episode.host / "model_gateway_events.jsonl",
                cooldown_gate=spec.cooldown_gate,
                session_id=identity["run_fingerprint_sha256"],
                required_system_prompt_sha256s=allowed_system_prompts,
                system_prompt_rewrites=prompt_rewrites,
                managed_transport_ambiguity_probe=(
                    spec.managed_transport_ambiguity_probe
                ),
            ) as gateway:
                gateway_started = True
                material = prepare_episode(
                    PiEpisodeConfig(
                        runtime_root=spec.runtime_root,
                        workspace=episode.workspace,
                        gateway_base_url=gateway.base_url,
                        model_profile=spec.model,
                        experiment_id=spec.experiment.experiment_id,
                        experiment_sha256=spec.experiment.source_sha256,
                        profile_registry_sha256=spec.registry.content_sha256,
                        max_model_requests=spec.experiment.maximum_model_requests,
                        wall_clock_seconds=spec.experiment.wall_clock_seconds,
                    )
                )
                _write_json_exclusive(
                    episode.host / "launch_record.json",
                    material["launch_record"],
                    0o444,
                )
                _write_json_exclusive(
                    episode.host / "model_gateway.json",
                    gateway.public_dict(),
                    0o444,
                )
                process = run_isolated(
                    workspace=episode.workspace,
                    runtime_root=spec.runtime_root,
                    command=material["command"],
                    tool_socket=database.socket_path,
                    tool_token=database.capability_token,
                    stdout_path=episode.host / "agent.stdout.jsonl",
                    stderr_path=episode.host / "agent.stderr.log",
                    stdin_text=material["stdin_text"],
                    timeout_seconds=spec.experiment.wall_clock_seconds,
                    model_gateway=gateway.endpoint_address,
                    model_gateway_token=gateway.capability_token,
                    extra_environment={
                        "ARENA_AGENT_ID": agent_id,
                        "ARENA_MODEL_ID": spec.model.model_profile_id,
                        "ARENA_RUN_ID": run_id,
                    },
                    harness_extension=material["harness_extension_path"],
                )
                process_elapsed = process.elapsed_seconds
                _write_json_exclusive(
                    episode.host / "process_result.json",
                    process.public_dict(),
                    0o444,
                )
                pi_tool_transcript = project_pi_tool_transcript(
                    source_path=episode.host / "agent.stdout.jsonl",
                    output_path=episode.host / "pi_tool_transcript.jsonl",
                    require_complete=(
                        process.status == "exited_zero"
                    ),
                )
                _write_json_exclusive(
                    episode.host / "pi_tool_transcript_summary.json",
                    pi_tool_transcript,
                    0o444,
                )
                if process.status == "exited_zero":
                    verify_prepared_episode_material(material)
            gateway_completion = gateway.wait_for_completion_report()
            transport_recycle_required = completion_requires_transport_recycle(
                gateway_completion
            )
            _write_json_exclusive(
                episode.host / "model_gateway_completion.json",
                gateway_completion,
                0o444,
            )
            verify_gateway_audit(
                episode.host / "model_gateway_events.jsonl",
                gateway_completion,
                expected_api_family_id=spec.model.api_family_id,
                expected_route_profile_id=spec.route.route_profile_id,
                expected_model_profile_id=spec.model.model_profile_id,
                expected_retry_policy=spec.model.retry.public_dict(),
            )
            if (
                process.status == "exited_zero"
                and not gateway_completion["all_logical_requests_complete"]
            ):
                raise GatewayError("model_gateway_request_chain_incomplete")
            if process.status != "exited_zero":
                if process.timed_out:
                    code = "ambiguous_agent_wall_clock_timeout"
                elif process.resource_limit_exceeded:
                    code = f"agent_resource_limit_{process.resource_limit_kind}"
                else:
                    code = "agent_process_nonzero"
                return _failed_outcome(
                    episode,
                    spec,
                    scene_id,
                    run_id,
                    process_elapsed,
                    "agent_runtime",
                    code,
                    transport_recycle_required=transport_recycle_required,
                )
        # Only trusted, host-sealed bytes are normalized, and only after the
        # per-episode DB socket is closed and every handler has drained.
        summary = collect_and_normalize(episode, database)
        outcome = EpisodeOutcome(
            status="complete",
            experiment_id=spec.experiment.experiment_id,
            model_profile_id=spec.model.model_profile_id,
            scene_id=scene_id,
            attempt=spec.attempt,
            run_id=run_id,
            episode_relative_path=episode.root.relative_to(ARENA_ROOT).as_posix(),
            resumed=False,
            failure_category=None,
            failure_code=None,
            submission_sha256=str(summary["submission_sha256"]),
            elapsed_seconds=process_elapsed,
            transport_recycle_required=False,
        )
        _write_json_exclusive(
            episode.host / "episode_outcome.json", outcome.public_dict(), 0o444
        )
        return outcome
    except (GatewayShutdownError, AgentToolShutdownError):
        # A handler may still own an upstream request.  This is an experiment-
        # wide safety barrier, not an ordinary per-scene failure: never begin
        # another scene/model until the operator has inspected the host.
        raise
    except Exception as exc:
        if episode is None:
            raise
        if gateway_started and gateway_completion is None:
            transport_recycle_required = True
        category, code = _sanitize_failure(exc)
        return _failed_outcome(
            episode,
            spec,
            scene_id,
            run_id,
            process_elapsed,
            category,
            code,
            transport_recycle_required=transport_recycle_required,
        )
    finally:
        if episode is not None:
            _seal_process_logs_if_present(episode.host)


def inspect_episode_if_present(
    spec: EpisodeRunSpec, scene_id: str
) -> EpisodeOutcome | None:
    """Inspect an exact sealed episode without starting Pi or touching an API.

    This entry point lets the experiment supervisor resume completed work
    before deciding whether a live route preflight is needed for that model.
    A present but invalid path is returned as ``failed_existing_episode`` and
    is never overwritten.
    """

    identity = build_run_identity(spec, scene_id)
    run_id = f"run-{identity['run_fingerprint_sha256'][:24]}"
    agent_id = _agent_id(spec.experiment.agent_id_prefix, spec.model.model_profile_id)
    root = ARENA_ROOT / "episodes" / agent_id / scene_id / run_id
    if not root.exists() and not root.is_symlink():
        return None
    return inspect_existing_episode(
        root=root,
        expected_identity=identity,
        experiment=spec.experiment,
        registry=spec.registry,
        model=spec.model,
        route=spec.route,
        runtime_binding=spec.runtime_binding,
        runtime_root=spec.runtime_root,
        scene_id=scene_id,
        attempt=spec.attempt,
        run_id=run_id,
    )


def inspect_existing_episode(
    *,
    root: Path,
    expected_identity: Mapping[str, Any],
    experiment: ExperimentProfile,
    registry: ProfileRegistry,
    model: ModelProfile,
    route: RouteProfile,
    runtime_binding: RouteRuntimeBinding,
    runtime_root: Path,
    scene_id: str,
    attempt: int,
    run_id: str,
) -> EpisodeOutcome:
    """Accept only an exactly matching, host-sealed normalized episode."""

    relative = root.relative_to(ARENA_ROOT).as_posix()
    host = root / "host"
    workspace = root / "workspace"
    try:
        expected_agent_id = _agent_id(
            experiment.agent_id_prefix, model.model_profile_id
        )
        expected_root = (
            ARENA_ROOT / "episodes" / expected_agent_id / scene_id / run_id
        )
        if root.expanduser().absolute() != expected_root:
            raise EpisodeRunError("existing_episode_path_identity_mismatch")
        if (
            not root.is_dir()
            or not host.is_dir()
            or not workspace.is_dir()
            or root.is_symlink()
            or host.is_symlink()
            or workspace.is_symlink()
        ):
            raise EpisodeRunError("existing_episode_is_linked")
        if model.api_family_id != experiment.api_family_id:
            raise EpisodeRunError("existing_experiment_model_family_mismatch")
        if route.route_profile_id != model.route_profile_id:
            raise EpisodeRunError("existing_route_model_mismatch")
        if runtime_binding.route_profile_id != route.route_profile_id:
            raise EpisodeRunError("existing_runtime_binding_route_mismatch")

        case = fixed_case(scene_id)
        episode = Episode(root=root, workspace=workspace, host=host, case=case)
        verify_episode_inputs(episode)

        observed_identity = _read_json_file(
            host / "run_identity.json", expected_mode=0o444
        )
        if observed_identity != dict(expected_identity):
            raise EpisodeRunError("existing_run_identity_mismatch")
        launch_record = _read_json_file(
            host / "launch_record.json", expected_mode=0o444
        )
        verify_existing_episode_launch(
            launch_record=launch_record,
            workspace=workspace,
            runtime_root=runtime_root,
            model_profile=model,
            experiment_id=experiment.experiment_id,
            experiment_sha256=experiment.source_sha256,
            profile_registry_sha256=registry.content_sha256,
            max_model_requests=experiment.maximum_model_requests,
            wall_clock_seconds=experiment.wall_clock_seconds,
        )
        allowed_system_prompts = allowed_pi_system_prompt_sha256s(workspace)
        prompt_rewrites = system_prompt_rewrite_map(workspace)
        expected_gateway = build_gateway_public_record(
            route=route,
            model=model,
            max_requests=experiment.maximum_model_requests,
            required_system_prompt_sha256s=allowed_system_prompts,
            system_prompt_rewrites=prompt_rewrites,
        )
        gateway_record = _read_json_file(
            host / "model_gateway.json", expected_mode=0o444
        )
        if gateway_record != expected_gateway:
            raise EpisodeRunError("existing_model_gateway_contract_mismatch")

        summary = _read_json_file(host / "summary.json", expected_mode=0o444)
        expected_summary_keys = {
            "schema_version",
            "status",
            "scene_id",
            "database_snapshot_id",
            "planned_instance_count",
            "room_count",
            "submission_sha256",
            "tool_event_journal_sha256",
            "official_evaluation_connected",
            "normalized_artifacts",
            "normalized_artifact_manifest",
        }
        if (
            set(summary) != expected_summary_keys
            or
            summary.get("schema_version")
            != "sieve_isolated_agent_episode_summary_v3"
            or summary.get("status") != "complete"
            or summary.get("scene_id") != scene_id
            or summary.get("database_snapshot_id")
            != expected_identity.get("database_snapshot_id")
            or summary.get("room_count") != case.room_count
            or summary.get("official_evaluation_connected") is not False
        ):
            raise EpisodeRunError("existing_summary_not_complete")
        planned = summary.get("planned_instance_count")
        if (
            isinstance(planned, bool)
            or not isinstance(planned, int)
            or not case.target_min <= planned <= case.target_max
        ):
            raise EpisodeRunError("existing_summary_instance_count_invalid")
        submission = workspace / "final_submission.json"
        if not submission.is_file() or submission.is_symlink():
            raise EpisodeRunError("existing_sealed_submission_missing")
        sealed_submission = _require_real_file_mode(
            host / "sealed_submission.json", 0o400, "host-sealed submission"
        )
        sealed_bytes = sealed_submission.read_bytes()
        digest = hashlib.sha256(sealed_bytes).hexdigest()
        if summary.get("submission_sha256") != digest:
            raise EpisodeRunError("existing_submission_hash_mismatch")
        if submission.read_bytes() != sealed_bytes:
            raise EpisodeRunError("existing_workspace_submission_differs_from_host")
        submission_payload = _read_json_file(
            sealed_submission, expected_mode=0o400
        )
        received_submission = _read_json_file(
            host / "received_submission.json", expected_mode=0o444
        )
        if received_submission != submission_payload:
            raise EpisodeRunError("existing_received_submission_mismatch")
        seal = _read_json_file(host / "submission_seal.json", expected_mode=0o400)
        if set(seal) != {
            "schema_version",
            "finalization",
            "sealed_submission",
        } or seal.get(
            "schema_version"
        ) != "sieve_trusted_submission_seal_v2":
            raise EpisodeRunError("existing_trusted_seal_schema_mismatch")
        finalization = seal.get("finalization")
        if not isinstance(finalization, dict) or finalization.get(
            "submission_sha256"
        ) != digest:
            raise EpisodeRunError("existing_trusted_seal_mismatch")
        if seal.get("sealed_submission") != {
            "path": "sealed_submission.json",
            "sha256": digest,
            "size_bytes": len(sealed_bytes),
            "mode": "0o400",
        }:
            raise EpisodeRunError("existing_host_sealed_submission_record_mismatch")
        workspace_finalization = _read_json_file(workspace / "finalization.json")
        workspace_finalization.pop("tool_counts", None)
        if workspace_finalization != finalization:
            raise EpisodeRunError("existing_workspace_finalization_mismatch")
        stored_tool_audit = _read_json_file(
            host / "tool_audit_verification.json", expected_mode=0o444
        )
        observed_tool_audit = verify_tool_event_journal(
            host / "tool_events.jsonl",
            policy=arena_tool_policy(),
            require_finalized=True,
            expected_mode=0o444,
        )
        if stored_tool_audit != observed_tool_audit:
            raise EpisodeRunError("existing_tool_audit_verification_mismatch")
        if summary.get("tool_event_journal_sha256") != observed_tool_audit.get(
            "journal_sha256"
        ):
            raise EpisodeRunError("existing_summary_tool_audit_hash_mismatch")
        verify_normalized_artifact_manifest(host, summary)
        process = _read_json_file(
            host / "process_result.json", expected_mode=0o444
        )
        _verify_process_result(
            host=host,
            process=process,
            wall_clock_seconds=experiment.wall_clock_seconds,
        )
        gateway_completion = _read_json_file(
            host / "model_gateway_completion.json", expected_mode=0o444
        )
        verification = verify_gateway_audit(
            host / "model_gateway_events.jsonl",
            gateway_completion,
            expected_api_family_id=model.api_family_id,
            expected_route_profile_id=route.route_profile_id,
            expected_model_profile_id=model.model_profile_id,
            expected_retry_policy=model.retry.public_dict(),
        )
        if (
            not verification["all_logical_requests_complete"]
            or verification["request_count"] > experiment.maximum_model_requests
        ):
            raise EpisodeRunError("existing_gateway_request_chain_incomplete")
        identity_matches = gateway_completion.get("identity_matches_by_request")
        if model.response_identity_required and (
            not isinstance(identity_matches, dict)
            or not identity_matches
            or any(value is not True for value in identity_matches.values())
        ):
            raise EpisodeRunError("existing_gateway_model_identity_unverified")
        transcript_summary = _read_json_file(
            host / "pi_tool_transcript_summary.json", expected_mode=0o444
        )
        _require_real_file_mode(
            host / "pi_tool_transcript.jsonl", 0o444, "Pi tool transcript"
        )
        verify_pi_tool_transcript(
            source_path=host / "agent.stdout.jsonl",
            transcript_path=host / "pi_tool_transcript.jsonl",
            summary=transcript_summary,
            require_complete=True,
        )
        stored_outcome = _read_json_file(
            host / "episode_outcome.json", expected_mode=0o444
        )
        elapsed = process.get("elapsed_seconds")
        expected_outcome_fields = {
            "schema_version": "sieve_pi_episode_outcome_v2",
            "status": "complete",
            "experiment_id": experiment.experiment_id,
            "model_profile_id": model.model_profile_id,
            "scene_id": scene_id,
            "attempt": attempt,
            "run_id": run_id,
            "episode_relative_path": relative,
            "resumed": False,
            "failure_category": None,
            "failure_code": None,
            "submission_sha256": digest,
            "elapsed_seconds": elapsed,
            "transport_recycle_required": False,
            "credential_endpoint_or_hidden_reasoning_recorded": False,
        }
        if stored_outcome != expected_outcome_fields:
            raise EpisodeRunError("existing_episode_outcome_mismatch")
        return EpisodeOutcome(
            status="complete",
            experiment_id=experiment.experiment_id,
            model_profile_id=model.model_profile_id,
            scene_id=scene_id,
            attempt=attempt,
            run_id=run_id,
            episode_relative_path=relative,
            resumed=True,
            failure_category=None,
            failure_code=None,
            submission_sha256=digest,
            elapsed_seconds=float(elapsed),
            transport_recycle_required=False,
        )
    except Exception as exc:
        category, code = _sanitize_failure(exc)
        return EpisodeOutcome(
            status="failed_existing_episode",
            experiment_id=experiment.experiment_id,
            model_profile_id=model.model_profile_id,
            scene_id=scene_id,
            attempt=attempt,
            run_id=run_id,
            episode_relative_path=relative,
            resumed=False,
            failure_category=category,
            failure_code=code,
            submission_sha256=None,
            elapsed_seconds=None,
            transport_recycle_required=False,
        )


def _failed_outcome(
    episode: Episode,
    spec: EpisodeRunSpec,
    scene_id: str,
    run_id: str,
    elapsed: float | None,
    category: str,
    code: str,
    *,
    transport_recycle_required: bool = False,
) -> EpisodeOutcome:
    outcome = EpisodeOutcome(
        status="failed",
        experiment_id=spec.experiment.experiment_id,
        model_profile_id=spec.model.model_profile_id,
        scene_id=scene_id,
        attempt=spec.attempt,
        run_id=run_id,
        episode_relative_path=episode.root.relative_to(ARENA_ROOT).as_posix(),
        resumed=False,
        failure_category=category,
        failure_code=code,
        submission_sha256=None,
        elapsed_seconds=elapsed,
        transport_recycle_required=transport_recycle_required,
    )
    path = episode.host / "episode_outcome.json"
    if not path.exists() and not path.is_symlink():
        _write_json_exclusive(path, outcome.public_dict(), 0o444)
    return outcome


def _sanitize_failure(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, (GatewayError, PiHarnessError, PiToolTranscriptError)):
        return "route_or_harness", type(exc).__name__
    if isinstance(exc, IsolationError):
        return "agent_runtime", type(exc).__name__
    if isinstance(exc, (ArenaError, ArenaLockError, EpisodeRunError)):
        return "contract_or_validation", str(exc) if isinstance(exc, EpisodeRunError) else type(exc).__name__
    return "workflow_infrastructure", type(exc).__name__


def _agent_id(prefix: str, model_profile_id: str) -> str:
    value = f"{prefix}-{model_profile_id}"
    if len(value) <= 127:
        return value
    return f"{prefix[:60]}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_process_result(
    *,
    host: Path,
    process: Mapping[str, Any],
    wall_clock_seconds: int,
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "returncode",
        "timed_out",
        "elapsed_seconds",
        "stdout_bytes",
        "stderr_bytes",
        "isolation_backend",
        "filesystem_scope",
        "network_mode",
        "host_environment_inherited",
        "logs_sanitized",
        "hidden_reasoning_recorded",
        "resource_limit_exceeded",
        "resource_limit_kind",
        "workspace_bytes_observed",
        "workspace_entries_observed",
    }
    if set(process) != expected_keys:
        raise EpisodeRunError("existing_process_result_field_set_mismatch")
    returncode = process.get("returncode")
    elapsed = process.get("elapsed_seconds")
    stdout_bytes = process.get("stdout_bytes")
    stderr_bytes = process.get("stderr_bytes")
    workspace_bytes = process.get("workspace_bytes_observed")
    workspace_entries = process.get("workspace_entries_observed")
    if (
        process.get("schema_version") != "sieve_isolated_agent_process_result_v2"
        or process.get("status") != "exited_zero"
        or isinstance(returncode, bool)
        or returncode != 0
        or process.get("timed_out") is not False
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
        or float(elapsed) > float(wall_clock_seconds) + 60.0
        or process.get("isolation_backend")
        != "macos_seatbelt_sandbox_exec_v1"
        or process.get("filesystem_scope") != "single_episode_workspace"
        or process.get("network_mode") != "scoped_loopback_model_gateway"
        or process.get("host_environment_inherited") is not False
        or process.get("logs_sanitized") is not True
        or process.get("hidden_reasoning_recorded") is not False
        or process.get("resource_limit_exceeded") is not False
        or process.get("resource_limit_kind") is not None
    ):
        raise EpisodeRunError("existing_process_result_not_successful")
    for value, label in (
        (stdout_bytes, "stdout"),
        (stderr_bytes, "stderr"),
        (workspace_bytes, "workspace_bytes"),
        (workspace_entries, "workspace_entries"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EpisodeRunError(f"existing_process_result_{label}_invalid")
    if stdout_bytes + stderr_bytes > MAX_TOTAL_LOG_BYTES:
        raise EpisodeRunError("existing_process_result_log_limit_exceeded")
    if workspace_bytes > MAX_WORKSPACE_BYTES:
        raise EpisodeRunError("existing_process_result_workspace_limit_exceeded")
    if workspace_entries > MAX_WORKSPACE_ENTRIES:
        raise EpisodeRunError("existing_process_result_entry_limit_exceeded")
    stdout = _require_real_file_mode(
        host / "agent.stdout.jsonl", 0o444, "agent stdout"
    )
    stderr = _require_real_file_mode(
        host / "agent.stderr.log", 0o444, "agent stderr"
    )
    if stdout.stat().st_size != stdout_bytes or stderr.stat().st_size != stderr_bytes:
        raise EpisodeRunError("existing_process_log_size_mismatch")


def _read_json_file(
    path: Path, *, expected_mode: int | None = None
) -> dict[str, Any]:
    if expected_mode is None:
        if not path.is_file() or path.is_symlink():
            raise EpisodeRunError(f"required_existing_file_missing_{path.name}")
    else:
        _require_real_file_mode(path, expected_mode, path.name)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise EpisodeRunError(f"existing_json_invalid_{path.name}") from exc
    if not isinstance(value, dict):
        raise EpisodeRunError(f"existing_json_root_invalid_{path.name}")
    return value


def _require_real_file_mode(path: Path, expected: int, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise EpisodeRunError(f"required_existing_file_missing_{path.name}")
    observed = stat.S_IMODE(path.stat().st_mode)
    if observed != expected:
        raise EpisodeRunError(
            f"existing_file_mode_mismatch_{label.replace(' ', '_')}"
        )
    return path


def _seal_process_logs_if_present(host: Path) -> None:
    for name in ("agent.stdout.jsonl", "agent.stderr.log"):
        path = host / name
        if not path.exists() and not path.is_symlink():
            continue
        if not path.is_file() or path.is_symlink():
            raise EpisodeRunError(f"process_log_not_real_{name}")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _write_json_exclusive(path: Path, value: Mapping[str, Any], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise EpisodeRunError(f"refusing_to_overwrite_{path.name}")
    encoded = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)


__all__ = [
    "EpisodeOutcome",
    "EpisodeRunError",
    "EpisodeRunSpec",
    "build_run_identity",
    "inspect_episode_if_present",
    "inspect_existing_episode",
    "run_one_episode",
]
