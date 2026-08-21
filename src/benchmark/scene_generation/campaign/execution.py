"""Executable campaign v2 adapter around the frozen generation kernel."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping
import uuid

from benchmark.scene_generation.campaign.bindings import (
    LocalRouteBindings,
    PrivateRouteBinding,
    select_binding_path as select_route_binding_path,
)
from benchmark.scene_generation.campaign.loader import load_campaign_profile_bundle
from benchmark.scene_generation.campaign.profiles import (
    ArtifactContract,
    BriefSetContract,
    CampaignProfile,
    CampaignProfileBundle,
    ExecutionPolicyContract,
    ModelProfile,
    PreflightContract,
    RouteProfile,
    WorkflowContract,
)
from benchmark.scene_generation.campaign.runtime import (
    RuntimeProviderModel,
    build_provider_route,
)
from benchmark.scene_generation.frozen_two_stage.compatibility.loader import (
    inspect_brief_ids,
    inspect_core_metadata,
    load_frozen_core,
    load_selected_briefs,
)
from benchmark.scene_generation.frozen_two_stage.orchestrator import (
    FrozenTwoStageOrchestrator,
    SafeProgress,
)
from benchmark.scene_generation.frozen_two_stage.retry_policy import RetryPolicy
from benchmark.scene_generation.frozen_two_stage.spec import GenerationRunSpec
from benchmark.scene_generation.frozen_two_stage.trust import TrustInventory
from benchmark.scene_generation.retrieval import RetrievalCatalog, build_runtime
from benchmark.scene_generation.retrieval.bindings import (
    LocalResourceBindings,
    select_binding_path as select_resource_binding_path,
)


DEFAULT_PROFILE_RELATIVE = Path("configs/generation/campaign_v2")
DEFAULT_RETRIEVAL_CATALOG_RELATIVE = Path("configs/retrieval/profiles_v2.json")
_CORE_BUNDLES = {
    "api3-anthropic-runner-core-v2": Path("tools/api3_anthropic_runner_v2")
}


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / DEFAULT_PROFILE_RELATIVE).is_dir()
            and (candidate / "tools/api3_anthropic_runner_v2/generation_runner.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        "generation campaign execution requires a Layout_DDD source checkout; "
        "the frozen core is intentionally not duplicated into the library wheel"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _manifest_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _literal_string_assignment(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            matches.append(node.value.value)
    if len(matches) != 1:
        raise ValueError(f"frozen core must declare exactly one {name}")
    return matches[0]


def _validate_static_artifact_contract(
    core_root: Path, artifact: ArtifactContract
) -> None:
    runner = core_root / "generation_runner.py"
    actual_run = _literal_string_assignment(runner, "RUN_MANIFEST_SCHEMA_VERSION")
    actual_case = _literal_string_assignment(runner, "CASE_RESULT_SCHEMA_VERSION")
    if actual_run != artifact.run_manifest_schema_version:
        raise ValueError("artifact contract run-manifest schema differs from frozen core")
    if actual_case != artifact.case_result_schema_version:
        raise ValueError("artifact contract case-result schema differs from frozen core")


def _validate_loaded_artifact_contract(core: Any, artifact: ArtifactContract) -> None:
    if getattr(core, "RUN_MANIFEST_SCHEMA_VERSION", None) != (
        artifact.run_manifest_schema_version
    ):
        raise ValueError("loaded core run-manifest schema differs from artifact contract")
    if getattr(core, "CASE_RESULT_SCHEMA_VERSION", None) != (
        artifact.case_result_schema_version
    ):
        raise ValueError("loaded core case-result schema differs from artifact contract")


@dataclass(frozen=True, slots=True)
class PreparedCampaign:
    repo_root: Path
    profile_root: Path
    retrieval_catalog_path: Path
    bundle: CampaignProfileBundle
    campaign: CampaignProfile
    model: ModelProfile
    route: RouteProfile
    workflow: WorkflowContract
    brief_set: BriefSetContract
    execution: ExecutionPolicyContract
    artifact: ArtifactContract
    preflight: PreflightContract
    core_root: Path
    briefs_path: Path
    models_public_path: Path
    provider_route: Any
    retry_policy: RetryPolicy
    trust_manifest_path: Path
    trust_report: Mapping[str, Any]
    source_manifest: Mapping[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "generation_campaign_resolution_v2",
            "campaign": self.campaign.to_public_dict(),
            "model": self.model.to_public_dict(),
            "route": self.route.to_public_dict(),
            "contracts": {
                "workflow_profile_id": self.workflow.workflow_profile_id,
                "brief_set_id": self.brief_set.brief_set_id,
                "execution_policy_id": self.execution.execution_policy_id,
                "artifact_contract_id": self.artifact.artifact_contract_id,
                "preflight_contract_id": self.preflight.preflight_contract_id,
            },
            "retry_policy": self.retry_policy.to_public_dict(),
            "trust": dict(self.trust_report),
            "source_manifest_sha256": self.source_manifest["manifest_sha256"],
            "credential_loaded": False,
            "network_used": False,
        }


class GatedRetrieverAdapter:
    """Expose a once-gated shared runtime through the frozen-core interface."""

    def __init__(self, runtime: Any, gate_report: Mapping[str, Any]) -> None:
        self.runtime = runtime
        self.gate_report = dict(gate_report)
        self.embedding_model_name = runtime.embedding_model_name
        self.retrieval_profile_id = runtime.profile_id
        self.public_provenance = runtime.public_provenance()
        profile = getattr(getattr(runtime, "composed", None), "profile", None)
        policy = getattr(profile, "policy", None)
        if policy is not None:
            self.retrieval_policy = {
                "category_argument": policy.category_argument,
                "size_tolerance": policy.size_tolerance,
                "top_k": policy.top_k,
                "min_score": policy.min_score,
                "tie_order": policy.tie_order,
            }

    def retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.retrieve_batch(request)


def _compile_retry(contract: ExecutionPolicyContract) -> RetryPolicy:
    return RetryPolicy(
        max_infrastructure_retries=contract.max_infrastructure_retries,
        retryable_transport_statuses=frozenset(contract.retryable_transport_statuses),
        retryable_http_statuses=frozenset(contract.retryable_http_statuses),
        retry_delay_seconds=contract.retry_delay_seconds,
        retry_ambiguous_timeouts=contract.retry_ambiguous_timeouts,
        continue_after_case_failure=contract.continue_after_case_failure,
    )


def prepare_campaign(
    campaign_id: str,
    *,
    profile_root: str | Path | None = None,
    retrieval_catalog_path: str | Path | None = None,
    trust_manifest: str | Path | None = None,
) -> PreparedCampaign:
    """Resolve and hash every public contract without bindings or credentials."""

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
    inventory = TrustInventory.load(trust_manifest)
    trust_report = inventory.verify_campaign_inputs(
        core_root=root / _CORE_BUNDLES["api3-anthropic-runner-core-v2"],
        campaign_runtime_root=Path(__file__).resolve().parent,
        campaign_profile_path=profiles / "campaigns_v2.json",
        retrieval_runtime_root=Path(build_runtime.__code__.co_filename).resolve().parent,
        retrieval_catalog_path=catalog_path,
    )
    bundle = load_campaign_profile_bundle(profiles)
    try:
        campaign, model, route = bundle.resolve_campaign(campaign_id)
    except KeyError as exc:
        raise ValueError(f"unknown campaign ID: {campaign_id!r}") from exc
    workflow = bundle.contracts.workflow_by_id[campaign.workflow_profile_id]
    brief_set = bundle.contracts.brief_set_by_id[campaign.brief_set_id]
    execution = bundle.contracts.execution_by_id[campaign.execution_policy_id]
    artifact = bundle.contracts.artifact_by_id[campaign.artifact_contract_id]
    preflight = bundle.contracts.preflight_by_id[model.preflight_contract_id]

    try:
        core_relative = _CORE_BUNDLES[workflow.core_bundle_id]
    except KeyError as exc:
        raise ValueError(f"unknown runtime core bundle: {workflow.core_bundle_id!r}") from exc
    core_root = (root / core_relative).resolve()
    briefs_path = core_root / "briefs.json"
    models_public_path = profiles / "model_profiles_v2.json"
    static_core = inspect_core_metadata(core_root)
    if static_core.runner_version != workflow.runner_version:
        raise ValueError("frozen core runner version differs from workflow contract")
    if static_core.runner_sha256 != workflow.runner_source_sha256:
        raise ValueError("frozen core runner hash differs from workflow contract")
    if _sha256(core_root / "stage_a_prompt.txt") != workflow.stage_a_prompt_sha256:
        raise ValueError("Stage A prompt hash differs from workflow contract")
    if _sha256(core_root / "stage_c_prompt.txt") != workflow.stage_c_prompt_sha256:
        raise ValueError("Stage C prompt hash differs from workflow contract")
    if _sha256(briefs_path) != brief_set.content_sha256:
        raise ValueError("brief content hash differs from brief-set contract")
    if inspect_brief_ids(briefs_path) != brief_set.ordered_brief_ids:
        raise ValueError("brief identities differ from brief-set contract")
    if route.runner_version != workflow.runner_version:
        raise ValueError("route runner version differs from workflow contract")
    _validate_static_artifact_contract(core_root, artifact)
    RetrievalCatalog.load(catalog_path).compose(campaign.retrieval_profile_id)
    retry_policy = _compile_retry(execution)
    provider_route = build_provider_route(route, model)

    payload = {
        "schema_version": "generation_campaign_source_manifest_v3",
        "campaign_id": campaign.campaign_id,
        "model_profile_id": model.model_profile_id,
        "route_profile_id": route.route_profile_id,
        "retrieval_profile_id": campaign.retrieval_profile_id,
        "workflow_profile_id": workflow.workflow_profile_id,
        "brief_set_id": brief_set.brief_set_id,
        "execution_policy_id": execution.execution_policy_id,
        "artifact_contract_id": artifact.artifact_contract_id,
        "preflight_contract_id": preflight.preflight_contract_id,
        "public_registry_sha256": {
            "routes": _sha256(profiles / "route_profiles_v2.json"),
            "models": _sha256(models_public_path),
            "campaigns": _sha256(profiles / "campaigns_v2.json"),
            "contracts": _sha256(profiles / "contracts_v2.json"),
            "retrieval_catalog": _sha256(catalog_path),
        },
        "frozen_content_sha256": {
            "generation_runner": workflow.runner_source_sha256,
            "briefs": brief_set.content_sha256,
            "stage_a_template": workflow.stage_a_prompt_sha256,
            "stage_c_template": workflow.stage_c_prompt_sha256,
        },
        "provider_route": provider_route.source_manifest(),
        "transport_binding": "private-redacted-v1",
        "trust": trust_report,
    }
    source_manifest = {**payload, "manifest_sha256": _manifest_sha(payload)}
    return PreparedCampaign(
        repo_root=root,
        profile_root=profiles,
        retrieval_catalog_path=catalog_path,
        bundle=bundle,
        campaign=campaign,
        model=model,
        route=route,
        workflow=workflow,
        brief_set=brief_set,
        execution=execution,
        artifact=artifact,
        preflight=preflight,
        core_root=core_root,
        briefs_path=briefs_path,
        models_public_path=models_public_path,
        provider_route=provider_route,
        retry_policy=retry_policy,
        trust_manifest_path=inventory.manifest_path,
        trust_report=trust_report,
        source_manifest=source_manifest,
    )


def _reverify_trust(prepared: PreparedCampaign) -> None:
    """Fail on any post-resolution source/config drift before credentials."""

    actual = TrustInventory.load(prepared.trust_manifest_path).verify_campaign_inputs(
        core_root=prepared.core_root,
        campaign_runtime_root=Path(__file__).resolve().parent,
        campaign_profile_path=prepared.profile_root / "campaigns_v2.json",
        retrieval_runtime_root=Path(build_runtime.__code__.co_filename).resolve().parent,
        retrieval_catalog_path=prepared.retrieval_catalog_path,
    )
    if actual != dict(prepared.trust_report):
        raise ValueError("generation campaign trust identity changed after resolution")


def resolve_bindings(
    prepared: PreparedCampaign,
    *,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[PrivateRouteBinding, LocalResourceBindings, dict[str, Any]]:
    """Resolve private binding files without reading a credential value."""

    route_bindings = LocalRouteBindings.load(
        select_route_binding_path(
            repo_root=prepared.repo_root,
            explicit_path=generation_bindings_path,
            environ=environ,
        )
    )
    route_binding = route_bindings.require(prepared.route.route_profile_id)
    resource_binding = LocalResourceBindings.load(
        select_resource_binding_path(
            catalog_path=prepared.retrieval_catalog_path,
            explicit_path=resource_bindings_path,
            environ=environ,
        )
    )
    composed = RetrievalCatalog.load(prepared.retrieval_catalog_path).compose(
        prepared.campaign.retrieval_profile_id
    )
    required_resources = {
        composed.index.metadata_file.resource_id,
        composed.index.matrix_file.resource_id,
        composed.encoder.model_resource_id,
    }
    actual_resources = set(resource_binding.paths)
    missing_resources = required_resources - actual_resources
    if missing_resources:
        raise ValueError(
            "selected retrieval profile resources are not fully bound: "
            f"missing={sorted(missing_resources)}"
        )
    public = {
        "schema_version": "generation_campaign_binding_resolution_v2",
        "campaign_id": prepared.campaign.campaign_id,
        "route_binding": route_binding.public_dict(),
        "resource_binding": {
            "schema_version": "generation_resource_bindings_v2",
            "bound_resource_ids": sorted(required_resources),
        },
        "credential_loaded": False,
        "network_used": False,
    }
    return route_binding, resource_binding, public


def gate_resources(
    prepared: PreparedCampaign,
    *,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    runtime_factory: Callable[..., Any] = build_runtime,
) -> tuple[GatedRetrieverAdapter | None, dict[str, Any]]:
    """Run the complete strict golden gate before credential or network access."""

    runtime = runtime_factory(
        catalog_path=prepared.retrieval_catalog_path,
        retrieval_profile_id=prepared.campaign.retrieval_profile_id,
        resource_bindings_path=resource_bindings_path,
        environ=os.environ if environ is None else environ,
    )
    report = runtime.gate(strict=None, run_golden=True)
    if report.get("status") == "failed":
        return None, dict(report)
    return GatedRetrieverAdapter(runtime, report), dict(report)


def _normalize_model(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _preflight_report(
    *,
    prepared: PreparedCampaign,
    core: Any,
    model: RuntimeProviderModel,
    transport: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    route = prepared.provider_route
    request = route.request_value(
        model=model,
        system_prompt='Return only the JSON object {"ok":true}.',
        user_value={"preflight": "validate the configured response contract"},
        canonical_json_bytes=core.canonical_json_bytes,
    )
    request_body = core.canonical_json_bytes(request)
    sender = core.post_once if transport is None else transport
    result = sender(
        model.endpoint,
        request_body,
        connect_timeout=min(model.timeout_seconds, 30.0),
        read_timeout=min(model.timeout_seconds, 600.0),
        request_headers=route.request_headers(model, str(uuid.uuid4())),
    )
    report: dict[str, Any] = {
        "schema_version": "generation_campaign_preflight_report_v2",
        "campaign_id": prepared.campaign.campaign_id,
        "preflight_contract_id": prepared.preflight.preflight_contract_id,
        "transport_status": result.status,
        "http_status": result.http_status,
        "response_contract_valid": False,
        "model_identity_matches": None,
        "content_nonempty": False,
        "reasoning_signal": None,
        "ok": False,
    }
    if (
        result.status != "response"
        or result.http_status is None
        or not 200 <= result.http_status < 300
        or result.response_body is None
    ):
        report["failure_category"] = "transport_or_http"
        report["transport_stage"] = result.stage
        report["error_type"] = result.error_type
        return report
    try:
        envelope = json.loads(result.response_body.decode("utf-8", errors="strict"))
        normalized = route.extract_api_message(result.response_body)
        content = normalized.content.strip()
        if prepared.preflight.content_validation == "json_object":
            parsed_content = json.loads(content.decode("utf-8", errors="strict"))
            if not isinstance(parsed_content, dict):
                raise ValueError("preflight content must be a JSON object")
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        report["failure_category"] = "response_contract"
        report["error_type"] = type(exc).__name__
        return report
    report["response_contract_valid"] = True
    report["content_nonempty"] = bool(content)
    returned_model = envelope.get("model") if isinstance(envelope, dict) else None
    if prepared.preflight.require_model_identity:
        expected_identity = prepared.model.response_model_identity
        if expected_identity is None:
            raise ValueError("model identity contract was not resolved")
        requested = _normalize_model(expected_identity)
        model_matches = isinstance(returned_model, str) and requested in _normalize_model(returned_model)
    else:
        model_matches = True
    report["model_identity_matches"] = model_matches
    reasoning_nonempty = bool(
        (normalized.reasoning is not None and normalized.reasoning.strip())
        or (
            normalized.reasoning_content is not None
            and normalized.reasoning_content.strip()
        )
    )
    details = (
        normalized.usage.get("completion_tokens_details")
        if isinstance(normalized.usage, Mapping)
        else None
    )
    tokens = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    reasoning_signal = reasoning_nonempty or (
        isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0
    )
    if prepared.preflight.record_reasoning_diagnostic:
        report["reasoning_signal"] = reasoning_signal
        report["reasoning_tokens"] = tokens if isinstance(tokens, int) else None
    ok = bool(content) and model_matches and (
        reasoning_signal or not prepared.preflight.require_reasoning_signal
    )
    report["ok"] = ok
    if not ok:
        report["failure_category"] = "preflight_assertion"
    return report


def _post_gate_private_runtime(
    prepared: PreparedCampaign,
    *,
    generation_bindings_path: str | Path | None,
    environ: Mapping[str, str] | None,
) -> tuple[RuntimeProviderModel, PrivateRouteBinding]:
    route_bindings = LocalRouteBindings.load(
        select_route_binding_path(
            repo_root=prepared.repo_root,
            explicit_path=generation_bindings_path,
            environ=environ,
        )
    )
    binding = route_bindings.require(prepared.route.route_profile_id)
    credential = binding.credential(environ)
    return (
        RuntimeProviderModel.from_profile(
            prepared.model,
            endpoint=binding.endpoint,
            api_key=credential,
        ),
        binding,
    )


def preflight_campaign(
    prepared: PreparedCampaign,
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
    # The complete source/config inventory is reverified after the resource
    # gate.  The frozen core is imported while no credential has been read,
    # then verified once more before the private credential is resolved.
    _reverify_trust(prepared)
    core = load_frozen_core(prepared.core_root)
    _reverify_trust(prepared)
    _validate_loaded_artifact_contract(core, prepared.artifact)
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


def _bound_source_manifest(
    prepared: PreparedCampaign,
    binding: PrivateRouteBinding,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(prepared.source_manifest)
    value.pop("manifest_sha256", None)
    value["route_binding"] = binding.public_dict()
    value["preflight"] = dict(preflight)
    return {**value, "manifest_sha256": _manifest_sha(value)}


def _generation_spec(
    prepared: PreparedCampaign,
    output_root: Path,
    binding: PrivateRouteBinding,
    preflight: Mapping[str, Any],
) -> GenerationRunSpec:
    source_manifest = _bound_source_manifest(prepared, binding, preflight)
    execution_policy = {
        "schema_version": "generation_campaign_execution_policy_v3",
        "campaign_id": prepared.campaign.campaign_id,
        "workflow_profile_id": prepared.workflow.workflow_profile_id,
        "execution_policy_id": prepared.execution.execution_policy_id,
        "artifact_contract_id": prepared.artifact.artifact_contract_id,
        "expected_brief_ids": list(prepared.campaign.ordered_brief_ids),
        **prepared.retry_policy.to_public_dict(),
    }
    return GenerationRunSpec(
        provider_key=prepared.route.route_profile_id,
        model_key=prepared.model.model_profile_id,
        wire_model=prepared.model.wire_model,
        ordered_brief_ids=prepared.campaign.ordered_brief_ids,
        briefs_path=prepared.briefs_path,
        # The core hashes but does not parse this path during adapter-driven
        # initialization. It is the public registry, never a private model pod.
        models_path=prepared.models_public_path,
        output_root=output_root,
        retry_policy=prepared.retry_policy,
        execution_policy=execution_policy,
        summary_schema_version="generation_campaign_run_summary_v3",
        summary_extra={"campaign_id": prepared.campaign.campaign_id},
        generation_parameters={
            "model_profile_id": prepared.model.model_profile_id,
            "route_profile_id": prepared.route.route_profile_id,
            "retrieval_profile_id": prepared.campaign.retrieval_profile_id,
        },
        artifact_schema_versions={
            "artifact_contract_id": prepared.artifact.artifact_contract_id,
            "run_manifest": prepared.artifact.run_manifest_schema_version,
            "case_result": prepared.artifact.case_result_schema_version,
        },
        provenance_hashes={
            "source_manifest_sha256": source_manifest["manifest_sha256"]
        },
        source_manifest=source_manifest,
    )


def _verify_campaign_artifacts(
    prepared: PreparedCampaign,
    output_root: Path,
) -> None:
    manifest_path = output_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != prepared.artifact.run_manifest_schema_version:
        raise ValueError("written run-manifest schema differs from artifact contract")
    public_model = manifest.get("model")
    if not isinstance(public_model, dict):
        raise ValueError("written run manifest has no public model object")
    forbidden = {"endpoint", "api_key", "credential", "credential_env"}
    if forbidden & public_model.keys():
        raise ValueError("written run manifest leaked private transport fields")
    for result_path in sorted(output_root.glob("brief_*/case.result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("schema_version") != prepared.artifact.case_result_schema_version:
            raise ValueError("written case-result schema differs from artifact contract")


def run_campaign(
    prepared: PreparedCampaign,
    *,
    output_root: str | Path,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    progress: SafeProgress | None = None,
    runtime_factory: Callable[..., Any] = build_runtime,
    transport: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
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
    _validate_loaded_artifact_contract(core, prepared.artifact)
    model, binding = _post_gate_private_runtime(
        prepared,
        generation_bindings_path=generation_bindings_path,
        environ=environ,
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
    briefs = load_selected_briefs(
        core, prepared.briefs_path, prepared.campaign.ordered_brief_ids
    )
    summary, stopped = FrozenTwoStageOrchestrator(
        core, prepared.provider_route
    ).run(
        spec=_generation_spec(prepared, destination, binding, preflight),
        model=model,
        briefs=briefs,
        retriever=retriever,
        progress=progress,
    )
    _verify_campaign_artifacts(prepared, destination)
    return summary, stopped, preflight
