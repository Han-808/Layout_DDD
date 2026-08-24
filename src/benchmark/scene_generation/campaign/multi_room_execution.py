"""Canonical campaign lifecycle implementation for the additive multi-room mode."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from benchmark.scene_generation.campaign.execution import (
    DEFAULT_PROFILE_RELATIVE,
    DEFAULT_RETRIEVAL_CATALOG_RELATIVE,
    GatedRetrieverAdapter,
    _CORE_BUNDLES,
    _manifest_sha,
    _model_identity_matches,
    _post_gate_private_runtime,
    _preflight_report,
    _sha256,
    gate_resources,
    repository_root,
)
from benchmark.scene_generation.campaign.bindings import (
    LocalRouteBindings,
    PrivateRouteBinding,
    select_binding_path as select_route_binding_path,
)
from benchmark.scene_generation.campaign.loader import load_campaign_profile_bundle
from benchmark.scene_generation.campaign.profiles import (
    CampaignProfileBundle,
    ExecutionPolicyContract,
    ModelProfile,
    PreflightContract,
    RouteProfile,
)
from benchmark.scene_generation.campaign.runtime import (
    RuntimeProviderModel,
    build_provider_route,
)
from benchmark.scene_generation.frozen_two_stage.compatibility.loader import (
    inspect_core_metadata,
    load_frozen_core,
)
from benchmark.scene_generation.frozen_two_stage.retry_policy import RetryPolicy
from benchmark.scene_generation.frozen_two_stage.trust import TrustInventory
from benchmark.scene_generation.multi_room.assembly import (
    ASSEMBLY_MANIFEST_SCHEMA_VERSION,
    COMPILED_ARCHITECTURE_SCHEMA_VERSION,
    MULTI_ROOM_SCENE_SCHEMA_VERSION,
    ROOM_EVALUATION_INDEX_SCHEMA_VERSION,
)
from benchmark.scene_generation.multi_room.artifacts import (
    MultiRoomArtifactLayout,
    gate_artifact_start,
)
from benchmark.scene_generation.multi_room.floor_plan import (
    LoadedFloorPlan,
    load_floor_plan,
)
from benchmark.scene_generation.campaign.multi_room_profiles import (
    MultiRoomArtifactContract,
    MultiRoomCampaignProfile,
    MultiRoomProfileRegistry,
    MultiRoomWorkflowContract,
    load_multi_room_profile_registry,
)
from benchmark.scene_generation.multi_room.provenance import (
    compatibility_source_manifest,
    run_input_fingerprint,
)
from benchmark.scene_generation.multi_room.runtime import (
    DEFAULT_STAGE_A_PROMPT,
    DEFAULT_STAGE_C_PROMPT,
    expected_room_resume_identities,
    run_multi_room_campaign,
)
from benchmark.scene_generation.retrieval import RetrievalCatalog, build_runtime


ROOM_RESULT_SCHEMA_VERSION = "multi_room_room_result_v1"
RUN_MANIFEST_SCHEMA_VERSION = "multi_room_generation_run_manifest_v1"
SUMMARY_SCHEMA_VERSION = "multi_room_generation_summary_v1"


@dataclass(frozen=True, slots=True)
class PreparedMultiRoomCampaign:
    repo_root: Path
    profile_root: Path
    retrieval_catalog_path: Path
    bundle: CampaignProfileBundle
    additive_registry: MultiRoomProfileRegistry
    campaign: MultiRoomCampaignProfile
    model: ModelProfile
    route: RouteProfile
    workflow: MultiRoomWorkflowContract
    execution: ExecutionPolicyContract
    artifact: MultiRoomArtifactContract
    preflight: PreflightContract
    core_root: Path
    models_public_path: Path
    provider_route: Any
    retry_policy: RetryPolicy
    floor_plan: LoadedFloorPlan
    trust_manifest_path: Path
    shared_trust_report: Mapping[str, Any]
    mode_trust_report: Mapping[str, Any]
    source_manifest: Mapping[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "generation_campaign_resolution_v3",
            "campaign": self.campaign.public_dict(),
            "model": self.model.to_public_dict(),
            "route": self.route.to_public_dict(),
            "contracts": {
                "workflow": self.workflow.public_dict(),
                "execution_policy_id": self.execution.execution_policy_id,
                "artifact": self.artifact.public_dict(),
                "preflight_contract_id": self.preflight.preflight_contract_id,
            },
            "floor_plan": self.floor_plan.public_dict(),
            "retry_policy": self.retry_policy.to_public_dict(),
            "trust": {
                "shared_campaign": dict(self.shared_trust_report),
                "multi_room_mode": dict(self.mode_trust_report),
            },
            "source_manifest_sha256": self.source_manifest["manifest_sha256"],
            "credential_loaded": False,
            "network_used": False,
        }


def _retry(contract: ExecutionPolicyContract) -> RetryPolicy:
    return RetryPolicy(
        max_infrastructure_retries=contract.max_infrastructure_retries,
        retryable_transport_statuses=frozenset(contract.retryable_transport_statuses),
        retryable_http_statuses=frozenset(contract.retryable_http_statuses),
        retry_delay_seconds=contract.retry_delay_seconds,
        retry_ambiguous_timeouts=contract.retry_ambiguous_timeouts,
        # Multi-room scientific semantics always terminalize the current room
        # and continue to the next declared room. No failed room is regenerated.
        continue_after_case_failure=True,
    )


def _verify_workflow_content(
    prepared_root: Path,
    workflow: MultiRoomWorkflowContract,
    core_root: Path,
) -> Mapping[str, Any]:
    static_core = inspect_core_metadata(core_root)
    if static_core.runner_version != workflow.core_runner_version:
        raise ValueError("multi-room core runner version mismatch")
    if static_core.runner_sha256 != workflow.core_runner_source_sha256:
        raise ValueError("multi-room core runner source mismatch")
    runtime_manifest = compatibility_source_manifest()
    if runtime_manifest["runtime_version"] != workflow.compatibility_runtime_version:
        raise ValueError("multi-room compatibility runtime version mismatch")
    if runtime_manifest["manifest_sha256"] != workflow.compatibility_source_manifest_sha256:
        raise ValueError("multi-room compatibility source manifest mismatch")
    checks = {
        "stage_a_prompt_sha256": _sha256(DEFAULT_STAGE_A_PROMPT),
        "stage_c_prompt_sha256": _sha256(DEFAULT_STAGE_C_PROMPT),
        "floor_plan_schema_sha256": _sha256(
            prepared_root
            / "src/benchmark/_resources/schemas/multi_room/floor_plan_v1.schema.json"
        ),
        "object_plan_schema_sha256": _sha256(
            prepared_root
            / "src/benchmark/_resources/schemas/multi_room/object_plan_v1.schema.json"
        ),
        "multi_room_scene_schema_sha256": _sha256(
            prepared_root
            / "src/benchmark/_resources/schemas/multi_room/scene_v1.schema.json"
        ),
        "compiled_architecture_schema_sha256": _sha256(
            prepared_root
            / "src/benchmark/_resources/schemas/multi_room/compiled_architecture_v1.schema.json"
        ),
        "room_evaluation_index_schema_sha256": _sha256(
            prepared_root
            / "src/benchmark/_resources/schemas/multi_room/room_evaluation_index_v1.schema.json"
        ),
        "assembly_manifest_schema_sha256": _sha256(
            prepared_root
            / "src/benchmark/_resources/schemas/multi_room/assembly_manifest_v1.schema.json"
        ),
    }
    for field_name, actual in checks.items():
        if actual != getattr(workflow, field_name):
            raise ValueError(f"multi-room pinned content mismatch: {field_name}")
    return runtime_manifest


def _verify_artifact_contract(artifact: MultiRoomArtifactContract) -> None:
    expected = {
        "run_manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "room_result_schema_version": ROOM_RESULT_SCHEMA_VERSION,
        "global_scene_schema_version": MULTI_ROOM_SCENE_SCHEMA_VERSION,
        "compiled_architecture_schema_version": COMPILED_ARCHITECTURE_SCHEMA_VERSION,
        "room_evaluation_index_schema_version": ROOM_EVALUATION_INDEX_SCHEMA_VERSION,
        "assembly_manifest_schema_version": ASSEMBLY_MANIFEST_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
    }
    for name, value in expected.items():
        if getattr(artifact, name) != value:
            raise ValueError(f"multi-room artifact contract mismatch: {name}")


def prepare_multi_room_campaign(
    campaign_id: str,
    *,
    floor_plan_path: str | Path,
    profile_root: str | Path | None = None,
    retrieval_catalog_path: str | Path | None = None,
    trust_manifest: str | Path | None = None,
) -> PreparedMultiRoomCampaign:
    """Validate all static contracts and floor-plan semantics without bindings."""

    root = repository_root()
    profiles = (
        root / DEFAULT_PROFILE_RELATIVE
        if profile_root is None
        else Path(profile_root).expanduser().resolve()
    )
    catalog_path = (
        root / DEFAULT_RETRIEVAL_CATALOG_RELATIVE
        if retrieval_catalog_path is None
        else Path(retrieval_catalog_path).expanduser().resolve()
    )
    bundle = load_campaign_profile_bundle(profiles)
    additive = load_multi_room_profile_registry(root, bundle)
    resolved = additive.resolve(campaign_id)
    plan = load_floor_plan(floor_plan_path)
    try:
        core_relative = _CORE_BUNDLES[resolved.workflow.core_bundle_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown multi-room core bundle: {resolved.workflow.core_bundle_id!r}"
        ) from exc
    core_root = (root / core_relative).resolve()
    runtime_manifest = _verify_workflow_content(root, resolved.workflow, core_root)
    _verify_artifact_contract(resolved.artifact)
    RetrievalCatalog.load(catalog_path).compose(
        resolved.campaign.retrieval_profile_id
    )
    inventory = TrustInventory.load(trust_manifest)
    shared_trust = inventory.verify_campaign_inputs(
        core_root=core_root,
        campaign_runtime_root=Path(repository_root.__code__.co_filename).resolve().parent,
        campaign_profile_path=profiles / "campaigns_v2.json",
        retrieval_runtime_root=Path(build_runtime.__code__.co_filename).resolve().parent,
        retrieval_catalog_path=catalog_path,
    )
    mode_trust = inventory.verify_campaign_inputs(
        core_root=core_root,
        campaign_runtime_root=Path(
            run_multi_room_campaign.__code__.co_filename
        ).resolve().parent,
        campaign_profile_path=additive.manifest_path,
        retrieval_runtime_root=Path(build_runtime.__code__.co_filename).resolve().parent,
        retrieval_catalog_path=catalog_path,
    )
    provider_route = build_provider_route(resolved.route, resolved.model)
    preflight = bundle.contracts.preflight_by_id[
        resolved.model.preflight_contract_id
    ]
    payload = {
        "schema_version": "generation_campaign_source_manifest_v3",
        "campaign_id": resolved.campaign.campaign_id,
        "model_profile_id": resolved.model.model_profile_id,
        "route_profile_id": resolved.route.route_profile_id,
        "retrieval_profile_id": resolved.campaign.retrieval_profile_id,
        "workflow_profile_id": resolved.workflow.workflow_profile_id,
        "generation_mode": resolved.workflow.generation_mode,
        "execution_policy_id": resolved.execution.execution_policy_id,
        "artifact_contract_id": resolved.artifact.artifact_contract_id,
        "preflight_contract_id": preflight.preflight_contract_id,
        "public_registry_sha256": {
            "routes": _sha256(profiles / "route_profiles_v2.json"),
            "models": _sha256(profiles / "model_profiles_v2.json"),
            "base_contracts": _sha256(profiles / "contracts_v2.json"),
            "retrieval_catalog": _sha256(catalog_path),
            "additive_manifest": additive.manifest_sha256,
            "additive_fragments": dict(sorted(additive.fragment_hashes.items())),
        },
        "frozen_content_sha256": {
            "generation_runner": resolved.workflow.core_runner_source_sha256,
            "multi_room_compatibility": runtime_manifest["manifest_sha256"],
            "stage_a_template": resolved.workflow.stage_a_prompt_sha256,
            "stage_c_template": resolved.workflow.stage_c_prompt_sha256,
            "floor_plan_schema": resolved.workflow.floor_plan_schema_sha256,
            "object_plan_schema": resolved.workflow.object_plan_schema_sha256,
            "global_scene_schema": resolved.workflow.multi_room_scene_schema_sha256,
            "compiled_architecture_schema": resolved.workflow.compiled_architecture_schema_sha256,
            "room_evaluation_index_schema": resolved.workflow.room_evaluation_index_schema_sha256,
            "assembly_manifest_schema": resolved.workflow.assembly_manifest_schema_sha256,
        },
        "provider_route": provider_route.source_manifest(),
        "transport_binding": "private-redacted-v1",
        "trust": {
            "shared_campaign": shared_trust,
            "multi_room_mode": mode_trust,
        },
    }
    source_manifest = {**payload, "manifest_sha256": _manifest_sha(payload)}
    return PreparedMultiRoomCampaign(
        repo_root=root,
        profile_root=profiles,
        retrieval_catalog_path=catalog_path,
        bundle=bundle,
        additive_registry=additive,
        campaign=resolved.campaign,
        model=resolved.model,
        route=resolved.route,
        workflow=resolved.workflow,
        execution=resolved.execution,
        artifact=resolved.artifact,
        preflight=preflight,
        core_root=core_root,
        models_public_path=profiles / "model_profiles_v2.json",
        provider_route=provider_route,
        retry_policy=_retry(resolved.execution),
        floor_plan=plan,
        trust_manifest_path=inventory.manifest_path,
        shared_trust_report=shared_trust,
        mode_trust_report=mode_trust,
        source_manifest=source_manifest,
    )


def _reverify_trust(prepared: PreparedMultiRoomCampaign) -> None:
    inventory = TrustInventory.load(prepared.trust_manifest_path)
    actual_shared = inventory.verify_campaign_inputs(
        core_root=prepared.core_root,
        campaign_runtime_root=Path(repository_root.__code__.co_filename).resolve().parent,
        campaign_profile_path=prepared.profile_root / "campaigns_v2.json",
        retrieval_runtime_root=Path(build_runtime.__code__.co_filename).resolve().parent,
        retrieval_catalog_path=prepared.retrieval_catalog_path,
    )
    actual_mode = inventory.verify_campaign_inputs(
        core_root=prepared.core_root,
        campaign_runtime_root=Path(
            run_multi_room_campaign.__code__.co_filename
        ).resolve().parent,
        campaign_profile_path=prepared.additive_registry.manifest_path,
        retrieval_runtime_root=Path(build_runtime.__code__.co_filename).resolve().parent,
        retrieval_catalog_path=prepared.retrieval_catalog_path,
    )
    if actual_shared != dict(prepared.shared_trust_report):
        raise ValueError("shared generation trust identity changed after resolution")
    if actual_mode != dict(prepared.mode_trust_report):
        raise ValueError("multi-room trust identity changed after resolution")
    actual_runtime = compatibility_source_manifest()
    if actual_runtime["manifest_sha256"] != prepared.workflow.compatibility_source_manifest_sha256:
        raise ValueError("multi-room runtime identity changed after resolution")
    stored_source = dict(prepared.source_manifest)
    stored_digest = stored_source.pop("manifest_sha256", None)
    if stored_digest != _manifest_sha(stored_source):
        raise ValueError("multi-room source manifest is internally inconsistent")
    frozen = stored_source.get("frozen_content_sha256")
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("multi_room_compatibility")
        != actual_runtime["manifest_sha256"]
    ):
        raise ValueError("multi-room source manifest runtime identity mismatch")


def _resolve_route_binding_without_credential(
    prepared: PreparedMultiRoomCampaign,
    *,
    generation_bindings_path: str | Path | None,
    environ: Mapping[str, str] | None,
) -> PrivateRouteBinding:
    bindings = LocalRouteBindings.load(
        select_route_binding_path(
            repo_root=prepared.repo_root,
            explicit_path=generation_bindings_path,
            environ=environ,
        )
    )
    return bindings.require(prepared.route.route_profile_id)


def _revalidate_prepared_floor_plan(
    prepared: PreparedMultiRoomCampaign,
) -> LoadedFloorPlan:
    """Close file and mutable-value TOCTOU before any external construction."""

    current = load_floor_plan(prepared.floor_plan.path)
    if (
        current.source_sha256 != prepared.floor_plan.source_sha256
        or current.canonical_sha256 != prepared.floor_plan.canonical_sha256
        or current.value != prepared.floor_plan.value
        or current.validation_report != prepared.floor_plan.validation_report
    ):
        raise ValueError("floor-plan identity changed after campaign preparation")
    return current


def _run_fingerprint(
    prepared: PreparedMultiRoomCampaign,
    plan: LoadedFloorPlan,
) -> Mapping[str, Any]:
    return run_input_fingerprint(
        campaign_id=prepared.campaign.campaign_id,
        workflow_profile_id=prepared.workflow.workflow_profile_id,
        model_profile_id=prepared.model.model_profile_id,
        route_profile_id=prepared.route.route_profile_id,
        retrieval_profile_id=prepared.campaign.retrieval_profile_id,
        execution_policy_id=prepared.execution.execution_policy_id,
        artifact_contract_id=prepared.artifact.artifact_contract_id,
        floor_plan_source_sha256=plan.source_sha256,
        floor_plan_canonical_sha256=plan.canonical_sha256,
        generation_order=plan.generation_order,
        source_manifest_sha256=prepared.source_manifest["manifest_sha256"],
        additive_manifest_sha256=prepared.additive_registry.manifest_sha256,
        fragment_hashes=prepared.additive_registry.fragment_hashes,
    )


def preflight_multi_room_campaign(
    prepared: PreparedMultiRoomCampaign,
    *,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    runtime_factory: Callable[..., Any] = build_runtime,
    transport: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], GatedRetrieverAdapter | None]:
    retriever, gate = gate_resources(
        prepared,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
        runtime_factory=runtime_factory,
    )
    if retriever is None:
        return {
            "schema_version": "generation_campaign_preflight_report_v2",
            "campaign_id": prepared.campaign.campaign_id,
            "ok": False,
            "failure_category": "retrieval_resource_gate",
            "resource_gate_status": gate.get("status"),
        }, None
    _reverify_trust(prepared)
    core = load_frozen_core(prepared.core_root)
    _reverify_trust(prepared)
    model, _ = _post_gate_private_runtime(
        prepared,
        generation_bindings_path=generation_bindings_path,
        environ=environ,
    )
    return _preflight_report(
        prepared=prepared,
        core=core,
        model=model,
        transport=transport,
    ), retriever


def run_prepared_multi_room_campaign(
    prepared: PreparedMultiRoomCampaign,
    *,
    output_root: str | Path,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    runtime_factory: Callable[..., Any] = build_runtime,
    transport: Callable[..., Any] | None = None,
    resume: bool = False,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    plan = _revalidate_prepared_floor_plan(prepared)
    fingerprint = _run_fingerprint(prepared, plan)
    room_resume_identities = expected_room_resume_identities(
        plan,
        campaign_id=prepared.campaign.campaign_id,
        workflow_profile_id=prepared.workflow.workflow_profile_id,
        model_key=prepared.model.model_profile_id,
        model_label=prepared.model.display_label,
        input_fingerprint_sha256=fingerprint["fingerprint_sha256"],
        source_manifest_sha256=prepared.source_manifest["manifest_sha256"],
    )
    layout = MultiRoomArtifactLayout(Path(output_root), plan.layout_id)
    terminal_room_count = gate_artifact_start(
        layout,
        resume=resume,
        expected_run_schema=prepared.artifact.run_manifest_schema_version,
        expected_fingerprint=fingerprint,
        floor_plan_source_sha256=plan.source_sha256,
        floor_plan_validation=plan.validation_report,
        sha256_file=_sha256,
        expected_room_ids=plan.generation_order,
        room_result_schema=prepared.artifact.room_result_schema_version,
        expected_room_results=room_resume_identities,
    )
    if resume and terminal_room_count == plan.room_count:
        _reverify_trust(prepared)
        core = load_frozen_core(prepared.core_root)
        _reverify_trust(prepared)
        execution_policy = json.loads(
            layout.execution_policy_path.read_text(encoding="utf-8")
        )
        provenance = execution_policy.get("run_provenance")
        preflight = (
            provenance.get("preflight")
            if isinstance(provenance, Mapping)
            else None
        )
        if not isinstance(preflight, Mapping) or preflight.get("ok") is not True:
            raise ValueError("resume execution policy lacks a valid original preflight")
        model = RuntimeProviderModel.from_profile(
            prepared.model,
            endpoint="https://resume-local.invalid/v1",
            api_key="local-finalization-only",
        )
        run_spec = {
            "campaign_id": prepared.campaign.campaign_id,
            "workflow_profile_id": prepared.workflow.workflow_profile_id,
            "model_profile_id": prepared.model.model_profile_id,
            "wire_model": prepared.model.wire_model,
            "route_profile_id": prepared.route.route_profile_id,
            "retrieval_profile_id": prepared.campaign.retrieval_profile_id,
            "retry_policy": prepared.retry_policy,
            "execution_policy": execution_policy,
            "source_manifest": prepared.source_manifest,
            "input_fingerprint": fingerprint,
        }
        summary, stopped = run_multi_room_campaign(
            core=core,
            provider_route=prepared.provider_route,
            model=model,
            retriever=None,
            plan=plan,
            output_root=output_root,
            artifact=prepared.artifact,
            run_spec=run_spec,
            resume=True,
            progress=progress,
        )
        return summary, stopped, dict(preflight)
    binding = _resolve_route_binding_without_credential(
        prepared,
        generation_bindings_path=generation_bindings_path,
        environ=environ,
    )
    gate_artifact_start(
        layout,
        resume=resume,
        expected_run_schema=prepared.artifact.run_manifest_schema_version,
        expected_fingerprint=fingerprint,
        floor_plan_source_sha256=plan.source_sha256,
        floor_plan_validation=plan.validation_report,
        sha256_file=_sha256,
        expected_room_ids=plan.generation_order,
        room_result_schema=prepared.artifact.room_result_schema_version,
        expected_route_binding=binding.public_dict(),
        expected_room_results=room_resume_identities,
    )
    retriever, gate = gate_resources(
        prepared,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
        runtime_factory=runtime_factory,
    )
    if retriever is None:
        raise RuntimeError(f"retrieval resource gate failed: status={gate.get('status')}")
    _reverify_trust(prepared)
    core = load_frozen_core(prepared.core_root)
    _reverify_trust(prepared)
    model = RuntimeProviderModel.from_profile(
        prepared.model,
        endpoint=binding.endpoint,
        api_key=binding.credential(environ),
    )
    preflight = _preflight_report(
        prepared=prepared,
        core=core,
        model=model,
        transport=transport,
    )
    if not preflight["ok"]:
        raise RuntimeError(
            f"campaign preflight failed: category={preflight.get('failure_category')}"
        )
    execution_policy = {
        "schema_version": "multi_room_generation_execution_policy_v1",
        "campaign_id": prepared.campaign.campaign_id,
        "workflow_profile_id": prepared.workflow.workflow_profile_id,
        "generation_mode": prepared.workflow.generation_mode,
        "execution_policy_id": prepared.execution.execution_policy_id,
        "artifact_contract_id": prepared.artifact.artifact_contract_id,
        "expected_room_ids": list(plan.generation_order),
        "room_execution": "sequential_isolated_v1",
        "continue_after_terminal_room": True,
        "semantic_retry_allowed": False,
        "run_provenance": {
            "schema_version": "generation_campaign_run_provenance_v1",
            "static_source_manifest_sha256": prepared.source_manifest[
                "manifest_sha256"
            ],
            "route_binding": binding.public_dict(),
            "preflight": dict(preflight),
        },
        **prepared.retry_policy.to_public_dict(),
    }
    run_spec = {
        "campaign_id": prepared.campaign.campaign_id,
        "workflow_profile_id": prepared.workflow.workflow_profile_id,
        "model_profile_id": prepared.model.model_profile_id,
        "wire_model": prepared.model.wire_model,
        "route_profile_id": prepared.route.route_profile_id,
        "retrieval_profile_id": prepared.campaign.retrieval_profile_id,
        "retry_policy": prepared.retry_policy,
        "execution_policy": execution_policy,
        "source_manifest": prepared.source_manifest,
        "input_fingerprint": fingerprint,
    }
    summary, stopped = run_multi_room_campaign(
        core=core,
        provider_route=prepared.provider_route,
        model=model,
        retriever=retriever,
        plan=plan,
        output_root=output_root,
        artifact=prepared.artifact,
        run_spec=run_spec,
        resume=resume,
        progress=progress,
    )
    return summary, stopped, preflight


__all__ = [
    "PreparedMultiRoomCampaign",
    "preflight_multi_room_campaign",
    "prepare_multi_room_campaign",
    "run_prepared_multi_room_campaign",
]
