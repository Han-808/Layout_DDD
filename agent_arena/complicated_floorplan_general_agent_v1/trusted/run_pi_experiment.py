#!/usr/bin/env python3
"""Plan, preflight, or execute one one-API/many-model Pi experiment.

Credentials are acquired exactly once for the selected API family and remain
only in this supervisor process.  The command line never accepts a credential
value.  A real run requires the explicit ``--execute`` switch; ``--dry-run``
performs no network access and creates no episode.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
from typing import Any, Callable, Mapping

from api_profiles import (
    ApiFamilyProfile,
    ExperimentProfile,
    ModelProfile,
    ProfileRegistry,
    RuntimeBindings,
    load_experiment,
    load_runtime_bindings,
    normalize_runtime_credential,
)
from arena import ARENA_ROOT, verify_fixed_suite
from family_lock import ApiFamilyInvocationLock
from managed_transport import (
    TOKENHUB_ADAPTER,
    prepare_runtime_transport,
    transport_contracts_for_experiment,
    verify_transport_runtime_compatibility,
)
from model_gateway import SharedCooldownGate
from pi_harness import verify_runtime
from preflight_pi_route import (
    run_pi_route_preflight,
    write_pi_route_preflight_report,
)
from run_pi_episode import (
    EpisodeOutcome,
    EpisodeRunSpec,
    inspect_episode_if_present,
    run_one_episode,
)
from verify_arena import verify_lock


DEFAULT_RUNTIME_ROOT = (ARENA_ROOT / "../runtime_bundles/pi-0.85.0").resolve()
DEFAULT_RUNTIME_BINDINGS = (
    ARENA_ROOT / "trusted/profiles/runtime_bindings.local.json"
)
EXAMPLE_RUNTIME_BINDINGS = (
    ARENA_ROOT / "trusted/profiles/runtime_bindings.example.json"
)
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


class ExperimentRunError(RuntimeError):
    """Raised for fail-closed experiment configuration errors."""


def build_plan(
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    bindings: RuntimeBindings,
    runtime: Mapping[str, Any],
    arena_lock: Mapping[str, Any],
) -> dict[str, Any]:
    fixed = verify_fixed_suite()
    available = set(fixed["cases"])
    unknown = sorted(set(experiment.scene_ids) - available)
    if unknown:
        raise ExperimentRunError(f"experiment contains unknown fixed scenes: {unknown}")
    models: list[dict[str, Any]] = []
    bound_route_ids: list[str] = []
    binding_profile_ids: dict[str, str] = {}
    for model_id in experiment.model_profile_ids:
        model = registry.model(model_id)
        route = registry.route(model.route_profile_id)
        binding = bindings.for_route(route.route_profile_id)
        if binding.route_profile_id not in bound_route_ids:
            bound_route_ids.append(binding.route_profile_id)
        binding_profile_ids[binding.route_profile_id] = binding.binding_profile_id
        models.append(
            {
                "model_profile_id": model.model_profile_id,
                "model_profile_sha256": _canonical_hash(model.public_dict()),
                "route_profile_id": route.route_profile_id,
                "route_profile_sha256": _canonical_hash(route.public_dict()),
                "pi_api_protocol": route.pi_api_protocol,
                "pi_thinking_level": model.pi.thinking_level,
                "provider_reasoning": model.reasoning.public_dict(),
                "request_timeout_seconds": model.request_timeout_seconds,
                "retry_policy": model.retry.public_dict(),
                "response_identity_required": model.response_identity_required,
            }
        )
    plan = {
        "schema_version": "sieve_pi_experiment_plan_v1",
        "experiment": experiment.public_dict(),
        "profile_registry_sha256": registry.content_sha256,
        "api_family": registry.api_family(experiment.api_family_id).public_dict(),
        "models": models,
        "scene_ids": list(experiment.scene_ids),
        "planned_episode_count": len(models) * len(experiment.scene_ids),
        "operator_private_runtime_bindings": {
            "bound_route_profile_ids": sorted(bound_route_ids),
            "binding_profile_ids": {
                key: binding_profile_ids[key] for key in sorted(binding_profile_ids)
            },
            "endpoint_recorded_or_hashed": False,
        },
        "runtime": {
            "pi_version": runtime["pi_version"],
            "content_root_sha256": runtime["content_root_sha256"],
            "runtime_manifest_sha256": runtime["runtime_manifest_sha256"],
        },
        "arena_content_root_sha256": arena_lock["content_root_sha256"],
        "transport_contracts": transport_contracts_for_experiment(
            registry, experiment
        ),
        "credential_acquisition": "once_per_experiment_host_memory_only",
        "one_api_family_enforced": True,
        "credential_endpoint_request_response_or_hidden_reasoning_recorded": False,
    }
    plan["execution_fingerprint_sha256"] = _canonical_hash(plan)
    return plan


def run_preflights(
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    bindings: RuntimeBindings,
    credential: str,
    cooldown_gate: SharedCooldownGate,
    invocation_root: Path,
    runtime_root: Path,
    model_profile_ids: tuple[str, ...] | None = None,
    transport_health_check: Callable[[], None] | None = None,
    transport_recycle_after_ambiguous: Callable[[], bool] | None = None,
    managed_transport_ambiguity_probe: Callable[[bool], bool] | None = None,
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    selected_models = (
        experiment.model_profile_ids
        if model_profile_ids is None
        else model_profile_ids
    )
    if any(model_id not in experiment.model_profile_ids for model_id in selected_models):
        raise ExperimentRunError("preflight model is outside the experiment")
    for model_id in selected_models:
        if transport_health_check is not None:
            transport_health_check()
        model = registry.model(model_id)
        route = registry.route(model.route_profile_id)
        binding = bindings.for_route(route.route_profile_id)
        model_root = invocation_root / "preflight" / model_id
        model_root.mkdir(parents=True, exist_ok=False)
        audit_run_id = "run-" + hashlib.sha256(
            (
                f"{invocation_root.name}:{experiment.source_sha256}:"
                f"{model.model_profile_id}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        audit_root = (
            ARENA_ROOT
            / "episodes"
            / "route-preflight"
            / model.model_profile_id
            / audit_run_id
        )
        report = run_pi_route_preflight(
            runtime_root=runtime_root,
            audit_root=audit_root,
            experiment_id=experiment.experiment_id,
            experiment_sha256=experiment.source_sha256,
            profile_registry_sha256=registry.content_sha256,
            route=route,
            model=model,
            runtime_binding=binding,
            runtime_credential=credential,
            cooldown_gate=cooldown_gate,
            wall_clock_seconds=experiment.wall_clock_seconds,
            managed_transport_ambiguity_probe=(
                managed_transport_ambiguity_probe
            ),
        )
        write_pi_route_preflight_report(model_root / "report.json", report)
        results[model_id] = report.ok
        _print_event(
            {
                "event": "model_preflight",
                "model_profile_id": model_id,
                "ok": report.ok,
                "failure_category": report.failure_category,
                "failure_code": report.failure_code,
            }
        )
        if report.transport_recycle_required:
            if transport_recycle_after_ambiguous is None:
                raise ExperimentRunError(
                    "ambiguous route state lacks a transport recycle barrier"
                )
            managed_recycled = transport_recycle_after_ambiguous()
            if not isinstance(managed_recycled, bool):
                raise ExperimentRunError(
                    "transport recycle callback returned a non-boolean result"
                )
            _print_event(
                {
                    "event": "transport_recovery_barrier_after_ambiguous_preflight",
                    "managed_transport_recycled": managed_recycled,
                    "model_profile_id": model_id,
                }
            )
    return results


def execute_experiment(
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    bindings: RuntimeBindings,
    credential: str,
    runtime_root: Path,
    resource_bindings: Path,
    invocation_root: Path,
    existing: Mapping[tuple[str, str], EpisodeOutcome | None] | None = None,
    transport_health_check: Callable[[], None] | None = None,
    transport_recycle_after_ambiguous: Callable[[], bool] | None = None,
    managed_transport_ambiguity_probe: Callable[[bool], bool] | None = None,
    transport_adapter_id: str | None = None,
    transport_started: bool = True,
) -> dict[str, Any]:
    cooldown_gate = SharedCooldownGate()
    transport_recycle_count = 0
    transport_recovery_barrier_count = 0

    def recycle_transport() -> bool:
        nonlocal transport_recycle_count, transport_recovery_barrier_count
        if transport_recycle_after_ambiguous is None:
            raise ExperimentRunError(
                "ambiguous route state lacks a transport recycle barrier"
            )
        managed_recycled = transport_recycle_after_ambiguous()
        if not isinstance(managed_recycled, bool):
            raise ExperimentRunError(
                "transport recycle callback returned a non-boolean result"
            )
        transport_recovery_barrier_count += 1
        if managed_recycled is True:
            transport_recycle_count += 1
        return managed_recycled
    observed_existing: dict[tuple[str, str], EpisodeOutcome | None] = (
        dict(existing) if existing is not None else {}
    )
    models_needing_preflight: list[str] = []
    for model_id in experiment.model_profile_ids:
        model = registry.model(model_id)
        route = registry.route(model.route_profile_id)
        binding = bindings.for_route(route.route_profile_id)
        missing = False
        for scene_id in experiment.scene_ids:
            key = (model_id, scene_id)
            if key in observed_existing:
                observed = observed_existing[key]
            else:
                observed = inspect_episode_if_present(
                    EpisodeRunSpec(
                        experiment=experiment,
                        registry=registry,
                        model=model,
                        route=route,
                        runtime_binding=binding,
                        runtime_credential=credential,
                        runtime_root=runtime_root,
                        resource_bindings=resource_bindings,
                        cooldown_gate=cooldown_gate,
                        attempt=1,
                        managed_transport_ambiguity_probe=(
                            managed_transport_ambiguity_probe
                        ),
                    ),
                    scene_id,
                )
                observed_existing[key] = observed
            if observed is None:
                missing = True
        if missing:
            models_needing_preflight.append(model_id)
    live_preflight = (
        run_preflights(
            registry=registry,
            experiment=experiment,
            bindings=bindings,
            credential=credential,
            cooldown_gate=cooldown_gate,
            invocation_root=invocation_root,
            runtime_root=runtime_root,
            model_profile_ids=tuple(models_needing_preflight),
            transport_health_check=transport_health_check,
            transport_recycle_after_ambiguous=recycle_transport,
            managed_transport_ambiguity_probe=(
                managed_transport_ambiguity_probe
            ),
        )
        if models_needing_preflight
        else {}
    )
    preflight: dict[str, bool | None] = {
        model_id: live_preflight.get(model_id)
        if model_id in models_needing_preflight
        else None
        for model_id in experiment.model_profile_ids
    }
    outcomes: list[EpisodeOutcome] = []
    stopped_early = False
    for model_id in experiment.model_profile_ids:
        model = registry.model(model_id)
        route = registry.route(model.route_profile_id)
        binding = bindings.for_route(route.route_profile_id)
        for scene_id in experiment.scene_ids:
            prior = observed_existing[(model_id, scene_id)]
            if prior is not None:
                outcomes.append(prior)
                _print_event(
                    {
                        "event": "episode_resume_inspected",
                        "model_profile_id": model_id,
                        "scene_id": scene_id,
                        "status": prior.status,
                        "failure_category": prior.failure_category,
                        "failure_code": prior.failure_code,
                    }
                )
                if (
                    prior.status != "complete"
                    and not experiment.continue_after_episode_failure
                ):
                    stopped_early = True
                    break
                continue
            if preflight.get(model_id) is not True:
                outcomes.append(
                    _skipped_outcome(experiment, model, scene_id, "preflight_failed")
                )
                if not experiment.continue_after_episode_failure:
                    stopped_early = True
                    break
                continue
            if transport_health_check is not None:
                transport_health_check()
            final: EpisodeOutcome | None = None
            for attempt in range(1, experiment.episode_attempts + 1):
                final = run_one_episode(
                    EpisodeRunSpec(
                        experiment=experiment,
                        registry=registry,
                        model=model,
                        route=route,
                        runtime_binding=binding,
                        runtime_credential=credential,
                        runtime_root=runtime_root,
                        resource_bindings=resource_bindings,
                        cooldown_gate=cooldown_gate,
                        attempt=attempt,
                        managed_transport_ambiguity_probe=(
                            managed_transport_ambiguity_probe
                        ),
                    ),
                    scene_id,
                )
                _print_event(
                    {
                        "event": "episode_terminal",
                        "model_profile_id": model_id,
                        "scene_id": scene_id,
                        "attempt": attempt,
                        "status": final.status,
                        "failure_category": final.failure_category,
                        "failure_code": final.failure_code,
                    }
                )
                if final.transport_recycle_required:
                    managed_recycled = recycle_transport()
                    _print_event(
                        {
                            "event": "transport_recovery_barrier_after_ambiguous_episode",
                            "managed_transport_recycled": managed_recycled,
                            "model_profile_id": model_id,
                            "scene_id": scene_id,
                            "attempt": attempt,
                        }
                    )
                if final.status == "complete":
                    break
            if final is None:
                raise ExperimentRunError("episode loop produced no outcome")
            outcomes.append(final)
            if final.status != "complete" and not experiment.continue_after_episode_failure:
                stopped_early = True
                break
        if stopped_early:
            break
    expected = len(experiment.model_profile_ids) * len(experiment.scene_ids)
    complete = sum(item.status == "complete" for item in outcomes)
    skipped = sum(item.status == "skipped" for item in outcomes)
    failed = len(outcomes) - complete - skipped
    missing = max(0, expected - len(outcomes))
    status = "complete" if complete == expected else "incomplete"
    summary = {
        "schema_version": "sieve_pi_experiment_summary_v3",
        "experiment_id": experiment.experiment_id,
        "api_family_id": experiment.api_family_id,
        "status": status,
        "expected_episodes": expected,
        "terminal_records": len(outcomes),
        "complete": complete,
        "failed": failed,
        "skipped": skipped,
        "missing": missing,
        "stopped_early": stopped_early,
        "transport_adapter_id": (
            transport_adapter_id
            if transport_adapter_id is not None
            else _single_transport_adapter_id(registry, experiment)
        ),
        "transport_started": transport_started,
        "transport_recycle_count": transport_recycle_count,
        "transport_recovery_barrier_count": transport_recovery_barrier_count,
        "preflight": preflight,
        "live_preflight_models": models_needing_preflight,
        "preflight_not_required_models": [
            model_id
            for model_id in experiment.model_profile_ids
            if model_id not in models_needing_preflight
        ],
        "preflight_not_required_reason": (
            "listed_models_have_all_exact_episode_paths_present"
            if len(models_needing_preflight) < len(experiment.model_profile_ids)
            else None
        ),
        "outcomes": [item.public_dict() for item in outcomes],
        "credential_endpoint_request_response_or_hidden_reasoning_recorded": False,
    }
    _write_json_exclusive(invocation_root / "summary.json", summary, 0o444)
    return summary


def preflight_only(
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    bindings: RuntimeBindings,
    credential: str,
    invocation_root: Path,
    runtime_root: Path,
    transport_health_check: Callable[[], None] | None = None,
    transport_recycle_after_ambiguous: Callable[[], bool] | None = None,
    managed_transport_ambiguity_probe: Callable[[bool], bool] | None = None,
    transport_adapter_id: str | None = None,
) -> dict[str, Any]:
    transport_recycle_count = 0
    transport_recovery_barrier_count = 0

    def recycle_transport() -> bool:
        nonlocal transport_recycle_count, transport_recovery_barrier_count
        if transport_recycle_after_ambiguous is None:
            raise ExperimentRunError(
                "ambiguous route state lacks a transport recycle barrier"
            )
        managed_recycled = transport_recycle_after_ambiguous()
        if not isinstance(managed_recycled, bool):
            raise ExperimentRunError(
                "transport recycle callback returned a non-boolean result"
            )
        transport_recovery_barrier_count += 1
        if managed_recycled is True:
            transport_recycle_count += 1
        return managed_recycled

    results = run_preflights(
        registry=registry,
        experiment=experiment,
        bindings=bindings,
        credential=credential,
        cooldown_gate=SharedCooldownGate(),
        invocation_root=invocation_root,
        runtime_root=runtime_root,
        transport_health_check=transport_health_check,
        transport_recycle_after_ambiguous=recycle_transport,
        managed_transport_ambiguity_probe=(
            managed_transport_ambiguity_probe
        ),
    )
    summary = {
        "schema_version": "sieve_pi_experiment_preflight_summary_v2",
        "experiment_id": experiment.experiment_id,
        "api_family_id": experiment.api_family_id,
        "ok": all(results.values()),
        "model_results": results,
        "generation_started": False,
        "transport_adapter_id": (
            transport_adapter_id
            if transport_adapter_id is not None
            else _single_transport_adapter_id(registry, experiment)
        ),
        "transport_started": True,
        "transport_recycle_count": transport_recycle_count,
        "transport_recovery_barrier_count": transport_recovery_barrier_count,
        "credential_endpoint_request_response_or_hidden_reasoning_recorded": False,
    }
    _write_json_exclusive(invocation_root / "summary.json", summary, 0o444)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--runtime-bindings",
        default=str(DEFAULT_RUNTIME_BINDINGS),
        help="Secret-free local endpoint binding JSON (never copied to episodes)",
    )
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument(
        "--tokenhub-litellm-runtime",
        help=(
            "Pinned LiteLLM snapshot root required for TokenHub; ignored by no "
            "other API family"
        ),
    )
    parser.add_argument("--resource-bindings")
    parser.add_argument(
        "--credential-env",
        help="Read one credential from this environment variable, then remove it",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-runtime-only", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    registry = ProfileRegistry.load()
    experiment = load_experiment(args.experiment, registry)
    runtime_root = Path(args.runtime_root).expanduser().resolve(strict=True)
    runtime = verify_runtime(runtime_root)
    arena_lock = verify_lock()
    binding_path = Path(args.runtime_bindings).expanduser()
    if args.dry_run and not binding_path.exists() and binding_path == DEFAULT_RUNTIME_BINDINGS:
        binding_path = EXAMPLE_RUNTIME_BINDINGS
    bindings = load_runtime_bindings(
        binding_path,
        registry=registry,
        experiment=experiment,
    )
    tokenhub_runtime_root: Path | None = None
    if args.tokenhub_litellm_runtime:
        tokenhub_runtime_root = (
            Path(args.tokenhub_litellm_runtime).expanduser().resolve(strict=True)
        )
        if experiment.api_family_id != "tokenhub":
            raise ExperimentRunError(
                "--tokenhub-litellm-runtime is only valid for TokenHub"
            )
    plan = build_plan(
        registry=registry,
        experiment=experiment,
        bindings=bindings,
        runtime=runtime,
        arena_lock=arena_lock,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if args.verify_runtime_only:
        verification = verify_transport_runtime_compatibility(
            registry=registry,
            experiment=experiment,
            tokenhub_runtime_root=tokenhub_runtime_root,
            pi_runtime_root=runtime_root,
            bindings=bindings,
            verify_managed_lifecycle=True,
        )
        print(
            json.dumps(
                {
                    "schema_version": "sieve_pi_experiment_runtime_gate_v2",
                    "experiment_id": experiment.experiment_id,
                    "execution_fingerprint_sha256": plan[
                        "execution_fingerprint_sha256"
                    ],
                    "transport": verification,
                    "production_provider_credential_or_route_used": False,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 0

    resource_bindings: Path | None = None
    if args.execute:
        if not args.resource_bindings:
            parser.error("--execute requires --resource-bindings")
        resource_bindings = Path(args.resource_bindings).expanduser().resolve(strict=True)
        if not resource_bindings.is_file() or resource_bindings.is_symlink():
            raise ExperimentRunError("resource bindings must be a real file")

    family = registry.api_family(experiment.api_family_id)
    invocation_nonce = secrets.token_hex(16)
    with ApiFamilyInvocationLock(
        lock_root=ARENA_ROOT / "episodes/_family_locks",
        api_family_id=family.api_family_id,
        experiment_id=experiment.experiment_id,
        invocation_nonce=invocation_nonce,
    ):
        raw_credential = ""
        provider_credential = ""
        runtime_credential = ""
        try:
            invocation_root = _prepare_invocation(plan)
            if resource_bindings is None:
                if args.execute:
                    raise ExperimentRunError("resource bindings were not resolved")

            existing: dict[tuple[str, str], EpisodeOutcome | None] | None = None
            if args.execute:
                if resource_bindings is None:  # pragma: no cover - guarded above
                    raise ExperimentRunError("resource bindings were not resolved")
                existing = inspect_existing_matrix(
                    registry=registry,
                    experiment=experiment,
                    bindings=bindings,
                    runtime_root=runtime_root,
                    resource_bindings=resource_bindings,
                )
                if all(value is not None for value in existing.values()):
                    _write_json_exclusive(
                        invocation_root / "transport_not_started.json",
                        {
                            "schema_version": "sieve_runtime_transport_not_started_v1",
                            "adapter_id": _single_transport_adapter_id(
                                registry, experiment
                            ),
                            "reason": "all_exact_episode_paths_already_present",
                            "credential_acquired": False,
                            "live_preflight_required": False,
                        },
                        0o444,
                    )
                    summary = execute_experiment(
                        registry=registry,
                        experiment=experiment,
                        bindings=bindings,
                        credential="sieve-resume-inspection-no-provider-credential",
                        runtime_root=runtime_root,
                        resource_bindings=resource_bindings,
                        invocation_root=invocation_root,
                        existing=existing,
                        transport_started=False,
                    )
                    print(
                        json.dumps(
                            summary, indent=2, sort_keys=True, ensure_ascii=False
                        )
                    )
                    return 0 if summary["status"] == "complete" else 3

            # Any path that can issue a new model request must pass the
            # credential-free runtime gate before a provider credential is
            # acquired.  A fully sealed exact resume returns above and needs
            # neither the gate nor a live transport.
            runtime_gate = verify_transport_runtime_compatibility(
                registry=registry,
                experiment=experiment,
                tokenhub_runtime_root=tokenhub_runtime_root,
                pi_runtime_root=runtime_root,
                bindings=bindings,
                verify_managed_lifecycle=True,
            )
            _write_json_exclusive(
                invocation_root / "runtime_compatibility_gate.json",
                {
                    "schema_version": "sieve_runtime_compatibility_gate_receipt_v2",
                    "experiment_id": experiment.experiment_id,
                    "execution_fingerprint_sha256": plan[
                        "execution_fingerprint_sha256"
                    ],
                    "transport": runtime_gate,
                    "production_provider_credential_or_route_used": False,
                },
                0o444,
            )
            raw_credential = _acquire_credential(family, args.credential_env)
            provider_credential = normalize_runtime_credential(
                family, raw_credential
            )
            with prepare_runtime_transport(
                registry=registry,
                experiment=experiment,
                bindings=bindings,
                provider_credential=provider_credential,
                invocation_root=invocation_root,
                tokenhub_runtime_root=tokenhub_runtime_root,
            ) as transport:
                runtime_credential = transport.credential
                if args.preflight_only:
                    summary = preflight_only(
                        registry=registry,
                        experiment=experiment,
                        bindings=transport.bindings,
                        credential=runtime_credential,
                        invocation_root=invocation_root,
                        runtime_root=runtime_root,
                        transport_health_check=transport.ensure_healthy,
                        transport_recycle_after_ambiguous=(
                            transport.recycle_after_ambiguous
                        ),
                        managed_transport_ambiguity_probe=(
                            transport.classify_outer_result
                            if transport.adapter_id == TOKENHUB_ADAPTER
                            else None
                        ),
                        transport_adapter_id=transport.adapter_id,
                    )
                    print(
                        json.dumps(
                            summary, indent=2, sort_keys=True, ensure_ascii=False
                        )
                    )
                    return 0 if summary["ok"] else 2
                if resource_bindings is None:  # pragma: no cover - guarded above
                    raise ExperimentRunError("resource bindings were not resolved")
                summary = execute_experiment(
                    registry=registry,
                    experiment=experiment,
                    bindings=transport.bindings,
                    credential=runtime_credential,
                    runtime_root=runtime_root,
                    resource_bindings=resource_bindings,
                    invocation_root=invocation_root,
                    existing=existing,
                    transport_health_check=transport.ensure_healthy,
                    transport_recycle_after_ambiguous=(
                        transport.recycle_after_ambiguous
                    ),
                    managed_transport_ambiguity_probe=(
                        transport.classify_outer_result
                        if transport.adapter_id == TOKENHUB_ADAPTER
                        else None
                    ),
                    transport_adapter_id=transport.adapter_id,
                    transport_started=True,
                )
                print(
                    json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
                )
                return 0 if summary["status"] == "complete" else 3
        finally:
            raw_credential = ""
            provider_credential = ""
            runtime_credential = ""


def inspect_existing_matrix(
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    bindings: RuntimeBindings,
    runtime_root: Path,
    resource_bindings: Path,
) -> dict[tuple[str, str], EpisodeOutcome | None]:
    """Inspect exact sealed paths before acquiring a provider credential."""

    cooldown_gate = SharedCooldownGate()
    observed: dict[tuple[str, str], EpisodeOutcome | None] = {}
    for model_id in experiment.model_profile_ids:
        model = registry.model(model_id)
        route = registry.route(model.route_profile_id)
        binding = bindings.for_route(route.route_profile_id)
        for scene_id in experiment.scene_ids:
            observed[(model_id, scene_id)] = inspect_episode_if_present(
                EpisodeRunSpec(
                    experiment=experiment,
                    registry=registry,
                    model=model,
                    route=route,
                    runtime_binding=binding,
                    runtime_credential=(
                        "sieve-resume-inspection-no-provider-credential"
                    ),
                    runtime_root=runtime_root,
                    resource_bindings=resource_bindings,
                    cooldown_gate=cooldown_gate,
                    attempt=1,
                ),
                scene_id,
            )
    return observed


def _single_transport_adapter_id(
    registry: ProfileRegistry, experiment: ExperimentProfile
) -> str:
    adapters = {
        registry.route(registry.model(model_id).route_profile_id).transport_adapter_id
        for model_id in experiment.model_profile_ids
    }
    if len(adapters) != 1:
        raise ExperimentRunError("experiment mixes transport adapters")
    return next(iter(adapters))


def _prepare_invocation(plan: Mapping[str, Any]) -> Path:
    experiment_id = str(plan["experiment"]["experiment_id"])
    fingerprint = str(plan["execution_fingerprint_sha256"])
    root = ARENA_ROOT / "episodes/_experiments" / experiment_id / fingerprint[:24]
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        if _read_json(plan_path) != dict(plan):
            raise ExperimentRunError("existing experiment plan differs")
    else:
        _write_json_exclusive(plan_path, plan, 0o444)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    invocation = root / "invocations" / f"{timestamp}-{secrets.token_hex(4)}"
    invocation.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        invocation / "invocation.json",
        {
            "schema_version": "sieve_pi_experiment_invocation_v1",
            "execution_fingerprint_sha256": fingerprint,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "credential_endpoint_request_response_or_hidden_reasoning_recorded": False,
        },
        0o444,
    )
    return invocation


def _acquire_credential(family: ApiFamilyProfile, env_name: str | None) -> str:
    if env_name is not None:
        if ENV_NAME.fullmatch(env_name) is None:
            raise ExperimentRunError("credential environment name is invalid")
        value = os.environ.pop(env_name, "")
        if not value:
            raise ExperimentRunError("credential environment variable is empty")
        return value
    if not sys.stdin.isatty():
        raise ExperimentRunError(
            "interactive credential prompt requires a TTY; use --credential-env"
        )
    return getpass.getpass(f"{family.display_label} credential (hidden): ")


def _skipped_outcome(
    experiment: ExperimentProfile,
    model: ModelProfile,
    scene_id: str,
    code: str,
) -> EpisodeOutcome:
    return EpisodeOutcome(
        status="skipped",
        experiment_id=experiment.experiment_id,
        model_profile_id=model.model_profile_id,
        scene_id=scene_id,
        attempt=0,
        run_id="not_started",
        episode_relative_path="not_started",
        resumed=False,
        failure_category="preflight",
        failure_code=code,
        submission_sha256=None,
        elapsed_seconds=None,
    )


def _print_event(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True, ensure_ascii=False), flush=True)


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ExperimentRunError(f"required JSON is missing or linked: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentRunError(f"JSON root must be an object: {path.name}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ExperimentRunError(f"refusing to overwrite {path.name}")
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


if __name__ == "__main__":
    raise SystemExit(main())
