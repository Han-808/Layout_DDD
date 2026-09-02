"""Standalone campaign lifecycle for the global polygon generation mode."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from benchmark.non_rectangular import validate_room_layout, validate_room_program
from benchmark.scene_generation.campaign.execution import (
    DEFAULT_PROFILE_RELATIVE,
    GatedRetrieverAdapter,
    _preflight_report,
    gate_resources,
    repository_root,
    resolve_bindings,
)
from benchmark.scene_generation.campaign.loader import (
    load_campaign_profile_bundle,
    load_model_profile_registry,
)
from benchmark.scene_generation.non_rectangular_multi_room.profiles import (
    WORKFLOW_PROFILE_ID,
    WORKFLOW_PROFILE_V2_ID,
    GENERATION_MODE,
    GENERATION_MODE_V2,
    NonRectangularCampaignSpec,
    SUPPLEMENTAL_EXECUTION_PROFILES_RELATIVE,
    SUPPLEMENTAL_MODEL_PROFILES_RELATIVE,
    load_non_rectangular_campaign_registry,
    load_non_rectangular_campaign_registry_v2,
    load_non_rectangular_execution_profiles,
)
from benchmark.scene_generation.campaign.runtime import (
    RuntimeProviderModel,
    build_provider_route,
)
from benchmark.scene_generation.frozen_two_stage.compatibility.loader import (
    inspect_core_metadata,
    load_frozen_core,
)
from benchmark.scene_generation.frozen_two_stage.providers.base import ProviderRoute
from benchmark.scene_generation.frozen_two_stage.providers.gateways.api2 import (
    API2GatewayPolicy,
)
from benchmark.scene_generation.frozen_two_stage.retry_policy import RetryPolicy
from benchmark.scene_generation.non_rectangular_multi_room.artifacts import (
    NonRectangularGenerationArtifacts,
)
from benchmark.scene_generation.non_rectangular_multi_room.provenance import (
    compatibility_source_manifest,
    sha256_mapping,
)
from benchmark.scene_generation.non_rectangular_multi_room.runtime import (
    run_non_rectangular_generation,
    run_non_rectangular_generation_v2,
)
from benchmark.scene_generation.retrieval import RetrievalCatalog, build_runtime


_CORE_RELATIVE = Path("tools/api3_anthropic_runner_v2")


@dataclass(frozen=True, slots=True)
class PreparedNonRectangularCampaign:
    repo_root: Path
    profile_root: Path
    retrieval_catalog_path: Path
    campaign: NonRectangularCampaignSpec
    model: Any
    route: Any
    execution: Any
    preflight: Any
    provider_route: Any
    retry_policy: RetryPolicy
    core_root: Path
    room_layout_path: Path
    room_program_path: Path
    room_layout: Mapping[str, Any]
    room_program: Mapping[str, Any]
    room_layout_source_sha256: str
    room_program_source_sha256: str
    room_layout_canonical_sha256: str
    room_program_canonical_sha256: str
    profile_bundle_sha256: str
    retrieval_catalog_sha256: str
    source_manifest: Mapping[str, Any]
    contract_version: str
    workflow_profile_id: str
    generation_mode: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (
                "prepared_non_rectangular_campaign_v2"
                if self.contract_version == "v2"
                else "prepared_non_rectangular_campaign_v1"
            ),
            "campaign_id": self.campaign.campaign_id,
            "contract_version": self.contract_version,
            "workflow_profile_id": self.workflow_profile_id,
            "generation_mode": self.generation_mode,
            "model_profile_id": self.model.model_profile_id,
            "model_label": self.model.display_label,
            "route_profile_id": self.route.route_profile_id,
            "retrieval_profile_id": self.campaign.retrieval_profile_id,
            "execution_policy_id": self.execution.execution_policy_id,
            "layout_id": self.room_layout["layout_id"],
            "room_layout_path": str(self.room_layout_path),
            "room_program_path": str(self.room_program_path),
            "room_layout_source_sha256": self.room_layout_source_sha256,
            "room_program_source_sha256": self.room_program_source_sha256,
            "room_layout_canonical_sha256": self.room_layout_canonical_sha256,
            "room_program_canonical_sha256": self.room_program_canonical_sha256,
            "profile_bundle_sha256": self.profile_bundle_sha256,
            "retrieval_catalog_sha256": self.retrieval_catalog_sha256,
            "source_manifest_sha256": self.source_manifest["manifest_sha256"],
            "credential_loaded": False,
            "network_used": False,
        }


@dataclass(frozen=True, slots=True)
class ActivatedNonRectangularCampaign:
    """Private, model-scoped runtime reusable across a checked scene cohort."""

    campaign_id: str
    model_profile_id: str
    route_profile_id: str
    profile_bundle_sha256: str
    retrieval_catalog_sha256: str
    source_manifest_sha256: str
    route_binding_sha256: str
    core: Any
    model: Any
    provider_route: Any
    retriever: GatedRetrieverAdapter
    preflight: Mapping[str, Any]


def prepare_non_rectangular_campaign(
    campaign_id: str,
    *,
    room_layout_path: str | Path,
    room_program_path: str | Path,
    profile_root: str | Path | None = None,
    retrieval_catalog_path: str | Path | None = None,
    contract_version: str = "v1",
) -> PreparedNonRectangularCampaign:
    """Resolve all public contracts without bindings, credentials, or network."""

    root = repository_root()
    profiles = (
        root / DEFAULT_PROFILE_RELATIVE
        if profile_root is None
        else Path(profile_root).expanduser().resolve()
    )
    bundle = load_campaign_profile_bundle(profiles)
    if contract_version == "v1":
        registry = load_non_rectangular_campaign_registry(root)
        workflow_profile_id = WORKFLOW_PROFILE_ID
        generation_mode = GENERATION_MODE
    elif contract_version == "v2":
        registry = load_non_rectangular_campaign_registry_v2(root)
        workflow_profile_id = WORKFLOW_PROFILE_V2_ID
        generation_mode = GENERATION_MODE_V2
    else:
        raise ValueError(
            f"unsupported non-rectangular contract_version {contract_version!r}"
        )
    try:
        campaign = registry[campaign_id]
        model = bundle.models.by_id.get(campaign.model_profile_id)
        if model is None:
            supplemental_models = load_model_profile_registry(
                root / SUPPLEMENTAL_MODEL_PROFILES_RELATIVE,
                bundle.routes,
            )
            model = supplemental_models.by_id[campaign.model_profile_id]
        route = bundle.routes.by_id[model.route_profile_id]
        execution = bundle.contracts.execution_by_id.get(
            campaign.execution_policy_id
        )
        if execution is None:
            execution = load_non_rectangular_execution_profiles(root)[
                campaign.execution_policy_id
            ]
        preflight = bundle.contracts.preflight_by_id[model.preflight_contract_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown or incomplete non-rectangular campaign {campaign_id!r}"
        ) from exc
    model.validate_for(route)
    catalog = (
        root / "configs/retrieval/profiles_v2.json"
        if retrieval_catalog_path is None
        else Path(retrieval_catalog_path).expanduser().resolve()
    )
    RetrievalCatalog.load(catalog).compose(campaign.retrieval_profile_id)
    layout_path = Path(room_layout_path).expanduser().resolve()
    program_path = Path(room_program_path).expanduser().resolve()
    layout = _load_json(layout_path, label="room layout")
    program = _load_json(program_path, label="room program")
    layout_report = validate_room_layout(layout)
    program_report = validate_room_program(program)
    if layout_report["layout_id"] != program_report["layout_id"]:
        raise ValueError("room layout/program layout_id mismatch")
    if layout_report["room_count"] != program_report["program_count"]:
        raise ValueError("room layout/program cardinality mismatch")
    core_root = root / _CORE_RELATIVE
    metadata = inspect_core_metadata(core_root)
    if metadata.runner_version != "2.0.0":
        raise ValueError("unsupported frozen core runner version")
    retry = RetryPolicy(
        max_infrastructure_retries=execution.max_infrastructure_retries,
        retryable_transport_statuses=frozenset(
            execution.retryable_transport_statuses
        ),
        retryable_http_statuses=frozenset(execution.retryable_http_statuses),
        retry_delay_seconds=execution.retry_delay_seconds,
        retry_ambiguous_timeouts=execution.retry_ambiguous_timeouts,
        continue_after_case_failure=execution.continue_after_case_failure,
    )
    return PreparedNonRectangularCampaign(
        repo_root=root,
        profile_root=profiles,
        retrieval_catalog_path=catalog,
        campaign=campaign,
        model=model,
        route=route,
        execution=execution,
        preflight=preflight,
        provider_route=build_provider_route(route, model),
        retry_policy=retry,
        core_root=core_root,
        room_layout_path=layout_path,
        room_program_path=program_path,
        room_layout=layout,
        room_program=program,
        room_layout_source_sha256=_sha256_file(layout_path),
        room_program_source_sha256=_sha256_file(program_path),
        room_layout_canonical_sha256=sha256_mapping(layout),
        room_program_canonical_sha256=sha256_mapping(program),
        profile_bundle_sha256=_profile_identity_sha256(root, profiles),
        retrieval_catalog_sha256=_sha256_file(catalog),
        source_manifest=compatibility_source_manifest(),
        contract_version=contract_version,
        workflow_profile_id=workflow_profile_id,
        generation_mode=generation_mode,
    )


def rebind_prepared_non_rectangular_campaign_inputs(
    prepared: PreparedNonRectangularCampaign,
    *,
    room_layout_path: str | Path,
    room_program_path: str | Path,
) -> PreparedNonRectangularCampaign:
    """Bind another checked scene without re-resolving static model profiles."""

    layout_path = Path(room_layout_path).expanduser().resolve()
    program_path = Path(room_program_path).expanduser().resolve()
    layout = _load_json(layout_path, label="room layout")
    program = _load_json(program_path, label="room program")
    layout_report = validate_room_layout(layout)
    program_report = validate_room_program(program)
    if layout_report["layout_id"] != program_report["layout_id"]:
        raise ValueError("room layout/program layout_id mismatch")
    if layout_report["room_count"] != program_report["program_count"]:
        raise ValueError("room layout/program cardinality mismatch")
    return replace(
        prepared,
        room_layout_path=layout_path,
        room_program_path=program_path,
        room_layout=layout,
        room_program=program,
        room_layout_source_sha256=_sha256_file(layout_path),
        room_program_source_sha256=_sha256_file(program_path),
        room_layout_canonical_sha256=sha256_mapping(layout),
        room_program_canonical_sha256=sha256_mapping(program),
    )


def preflight_non_rectangular_campaign(
    prepared: PreparedNonRectangularCampaign,
    *,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    runtime_factory: Callable[..., Any] = build_runtime,
    transport: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], GatedRetrieverAdapter | None]:
    report, activated = activate_non_rectangular_campaign(
        prepared,
        generation_bindings_path=generation_bindings_path,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
        runtime_factory=runtime_factory,
        transport=transport,
    )
    return report, None if activated is None else activated.retriever


def activate_non_rectangular_campaign(
    prepared: PreparedNonRectangularCampaign,
    *,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    runtime_factory: Callable[..., Any] = build_runtime,
    transport: Callable[..., Any] | None = None,
    stage_c_timeout_seconds: float | None = None,
) -> tuple[dict[str, Any], ActivatedNonRectangularCampaign | None]:
    """Gate and preflight one model once for a sequential scene cohort."""

    retriever, gate = gate_resources(
        prepared,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
        runtime_factory=runtime_factory,
    )
    if retriever is None:
        return {
            "schema_version": "non_rectangular_preflight_v1",
            "ok": False,
            "failure_category": "retrieval_resource_gate",
            "resource_gate": gate,
        }, None
    binding, _, _ = resolve_bindings(
        prepared,
        generation_bindings_path=generation_bindings_path,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
    )
    core = load_frozen_core(prepared.core_root)
    model: Any = RuntimeProviderModel.from_profile(
        prepared.model,
        endpoint=binding.endpoint,
        api_key=binding.credential(environ),
    )
    if stage_c_timeout_seconds is not None:
        model = _StageTimeoutProviderModel(
            base=model,
            stage_c_timeout_seconds=stage_c_timeout_seconds,
        )
    provider_route = _stage_timeout_provider_route(prepared.provider_route)
    report = _preflight_report(
        prepared=prepared,
        core=core,
        model=model,
        transport=transport,
    )
    public_report = {
        "schema_version": "non_rectangular_preflight_v1",
        **report,
        "resource_gate": gate,
    }
    if public_report.get("ok") is not True:
        return public_report, None
    return public_report, ActivatedNonRectangularCampaign(
        campaign_id=prepared.campaign.campaign_id,
        model_profile_id=prepared.model.model_profile_id,
        route_profile_id=prepared.route.route_profile_id,
        profile_bundle_sha256=prepared.profile_bundle_sha256,
        retrieval_catalog_sha256=prepared.retrieval_catalog_sha256,
        source_manifest_sha256=prepared.source_manifest["manifest_sha256"],
        route_binding_sha256=binding.binding_sha256,
        core=core,
        model=model,
        provider_route=provider_route,
        retriever=retriever,
        preflight=public_report,
    )


def run_activated_non_rectangular_campaign(
    prepared: PreparedNonRectangularCampaign,
    activated: ActivatedNonRectangularCampaign,
    *,
    output_root: str | Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    resume: bool = False,
    configuration_identity_extra: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one checked scene with a model-scoped preflight/runtime."""

    layout, program = _revalidate_inputs(prepared)
    expected = (
        prepared.campaign.campaign_id,
        prepared.model.model_profile_id,
        prepared.route.route_profile_id,
        prepared.profile_bundle_sha256,
        prepared.retrieval_catalog_sha256,
        prepared.source_manifest["manifest_sha256"],
    )
    actual = (
        activated.campaign_id,
        activated.model_profile_id,
        activated.route_profile_id,
        activated.profile_bundle_sha256,
        activated.retrieval_catalog_sha256,
        activated.source_manifest_sha256,
    )
    if actual != expected or activated.preflight.get("ok") is not True:
        raise ValueError("activated non-rectangular campaign identity mismatch")
    runner = (
        run_non_rectangular_generation_v2
        if prepared.contract_version == "v2"
        else run_non_rectangular_generation
    )
    identity = _configuration_identity(prepared)
    if configuration_identity_extra is not None:
        identity["route_binding_sha256"] = activated.route_binding_sha256
    for key, value in sorted((configuration_identity_extra or {}).items()):
        if key in identity:
            raise ValueError(f"duplicate configuration identity key: {key}")
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise ValueError("extra configuration identity must be non-empty text")
        identity[key] = value
    return runner(
        core=activated.core,
        provider_route=activated.provider_route,
        model=activated.model,
        retriever=activated.retriever,
        retry_policy=prepared.retry_policy,
        room_layout=layout,
        room_program=program,
        output_root=Path(output_root),
        campaign_id=prepared.campaign.campaign_id,
        workflow_profile_id=prepared.workflow_profile_id,
        retrieval_profile_id=prepared.campaign.retrieval_profile_id,
        configuration_identity=identity,
        resume=resume,
        progress=progress,
    )


def run_prepared_non_rectangular_campaign(
    prepared: PreparedNonRectangularCampaign,
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
    layout, program = _revalidate_inputs(prepared)
    output = Path(output_root)
    runner = (
        run_non_rectangular_generation_v2
        if prepared.contract_version == "v2"
        else run_non_rectangular_generation
    )
    if resume and NonRectangularGenerationArtifacts(output).summary.is_file():
        core = load_frozen_core(prepared.core_root)
        model = RuntimeProviderModel.from_profile(
            prepared.model,
            endpoint="https://resume-local.invalid/v1",
            api_key="local-terminal-resume",
        )
        summary = runner(
            core=core,
            provider_route=prepared.provider_route,
            model=model,
            retriever=None,
            retry_policy=prepared.retry_policy,
            room_layout=layout,
            room_program=program,
            output_root=output,
            campaign_id=prepared.campaign.campaign_id,
            workflow_profile_id=prepared.workflow_profile_id,
            retrieval_profile_id=prepared.campaign.retrieval_profile_id,
            configuration_identity={
                "profile_bundle_sha256": prepared.profile_bundle_sha256,
                "retrieval_catalog_sha256": prepared.retrieval_catalog_sha256,
                "room_layout_source_sha256": (
                    prepared.room_layout_source_sha256
                ),
                "room_program_source_sha256": (
                    prepared.room_program_source_sha256
                ),
            },
            resume=True,
            progress=progress,
        )
        return summary, False, {
            "schema_version": "non_rectangular_preflight_v1",
            "ok": True,
            "mode": "local_terminal_resume",
        }
    preflight, retriever = preflight_non_rectangular_campaign(
        prepared,
        generation_bindings_path=generation_bindings_path,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
        runtime_factory=runtime_factory,
        transport=transport,
    )
    if preflight.get("ok") is not True or retriever is None:
        raise RuntimeError(
            "non-rectangular campaign preflight failed: "
            f"category={preflight.get('failure_category')}"
        )
    binding, _, _ = resolve_bindings(
        prepared,
        generation_bindings_path=generation_bindings_path,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
    )
    core = load_frozen_core(prepared.core_root)
    model = RuntimeProviderModel.from_profile(
        prepared.model,
        endpoint=binding.endpoint,
        api_key=binding.credential(environ),
    )
    summary = runner(
        core=core,
        provider_route=prepared.provider_route,
        model=model,
        retriever=retriever,
        retry_policy=prepared.retry_policy,
        room_layout=layout,
        room_program=program,
        output_root=output,
        campaign_id=prepared.campaign.campaign_id,
        workflow_profile_id=prepared.workflow_profile_id,
        retrieval_profile_id=prepared.campaign.retrieval_profile_id,
        configuration_identity=_configuration_identity(prepared),
        progress=progress,
        resume=resume,
    )
    return summary, False, preflight


def _configuration_identity(
    prepared: PreparedNonRectangularCampaign,
) -> dict[str, str]:
    return {
        "profile_bundle_sha256": prepared.profile_bundle_sha256,
        "retrieval_catalog_sha256": prepared.retrieval_catalog_sha256,
        "room_layout_source_sha256": prepared.room_layout_source_sha256,
        "room_program_source_sha256": prepared.room_program_source_sha256,
    }


def _revalidate_inputs(
    prepared: PreparedNonRectangularCampaign,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout = _load_json(prepared.room_layout_path, label="room layout")
    program = _load_json(prepared.room_program_path, label="room program")
    if (
        _sha256_file(prepared.room_layout_path)
        != prepared.room_layout_source_sha256
        or sha256_mapping(layout) != prepared.room_layout_canonical_sha256
        or layout != dict(prepared.room_layout)
    ):
        raise ValueError("room-layout identity changed after preparation")
    if (
        _sha256_file(prepared.room_program_path)
        != prepared.room_program_source_sha256
        or sha256_mapping(program) != prepared.room_program_canonical_sha256
        or program != dict(prepared.room_program)
    ):
        raise ValueError("room-program identity changed after preparation")
    if compatibility_source_manifest() != dict(prepared.source_manifest):
        raise ValueError("non-rectangular generation source identity changed")
    if _profile_identity_sha256(prepared.repo_root, prepared.profile_root) != (
        prepared.profile_bundle_sha256
    ):
        raise ValueError("generation profile identity changed after preparation")
    if _sha256_file(prepared.retrieval_catalog_path) != (
        prepared.retrieval_catalog_sha256
    ):
        raise ValueError("retrieval catalog identity changed after preparation")
    return layout, program


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_manifest_sha256(root: Path) -> str:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _profile_identity_sha256(repo_root: Path, profile_root: Path) -> str:
    payload = {
        "shared_profile_directory_sha256": _directory_manifest_sha256(profile_root),
        "supplemental_model_profiles_sha256": _sha256_file(
            repo_root / SUPPLEMENTAL_MODEL_PROFILES_RELATIVE
        ),
        "supplemental_execution_profiles_sha256": _sha256_file(
            repo_root / SUPPLEMENTAL_EXECUTION_PROFILES_RELATIVE
        ),
    }
    return sha256_mapping(payload)


@dataclass(frozen=True, slots=True)
class _StageTimeoutProviderModel:
    base: RuntimeProviderModel
    stage_c_timeout_seconds: float

    def __post_init__(self) -> None:
        value = self.stage_c_timeout_seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Stage C timeout must be positive")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def gateway_timeout_override_seconds(self) -> float:
        return float(self.base.timeout_seconds)

    def for_stage(self, stage: str) -> Any:
        if stage != "stage_c":
            return self
        return _StageTimeoutProviderModelView(
            base=self.base,
            timeout_seconds=float(self.stage_c_timeout_seconds),
            gateway_timeout_override_seconds=float(self.stage_c_timeout_seconds),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            **self.base.public_dict(),
            "stage_c_timeout_seconds": float(self.stage_c_timeout_seconds),
        }


@dataclass(frozen=True, slots=True)
class _StageTimeoutProviderModelView:
    base: RuntimeProviderModel
    timeout_seconds: float
    gateway_timeout_override_seconds: float

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def public_dict(self) -> dict[str, Any]:
        return {
            **self.base.public_dict(),
            "stage_c_timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class _StageAwareAPI2Gateway:
    """Apply model-owned request timeouts only for this additive route."""

    base: API2GatewayPolicy

    def request_headers(self, model: Any, session_id: str) -> dict[str, str]:
        headers = self.base.request_headers(model, session_id)
        timeout = getattr(model, "gateway_timeout_override_seconds", None)
        if timeout is None:
            return headers
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or float(timeout) <= 0
        ):
            raise ValueError("non-rectangular API2 timeout must be positive")
        authorization = headers.get("Authorization")
        if not isinstance(authorization, str):
            raise ValueError("API2 authorization header is unavailable")
        updated, count = re.subn(
            r"([?&]timeout=)[0-9]+",
            rf"\g<1>{max(1, int(float(timeout)))}",
            authorization,
            count=1,
        )
        if count != 1:
            raise ValueError("API2 authorization timeout is unavailable")
        return {**headers, "Authorization": updated}

    def public_dict(self) -> dict[str, Any]:
        return {
            **self.base.public_dict(),
            "timeout_policy": "nonrect_stage_owned_v1",
        }


def _stage_timeout_provider_route(provider_route: Any) -> Any:
    if not isinstance(provider_route, ProviderRoute) or not isinstance(
        provider_route.gateway, API2GatewayPolicy
    ):
        return provider_route
    return ProviderRoute(
        codec=provider_route.codec,
        gateway=_StageAwareAPI2Gateway(provider_route.gateway),
        route_key=provider_route.route_key,
        provenance_modules=provider_route.provenance_modules + (__name__,),
    )


__all__ = [
    "ActivatedNonRectangularCampaign",
    "PreparedNonRectangularCampaign",
    "activate_non_rectangular_campaign",
    "preflight_non_rectangular_campaign",
    "prepare_non_rectangular_campaign",
    "rebind_prepared_non_rectangular_campaign_inputs",
    "run_prepared_non_rectangular_campaign",
    "run_activated_non_rectangular_campaign",
]
