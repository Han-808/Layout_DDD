"""Read-only characterization of legacy run configs against v2 profiles.

This is a migration aid, not the current campaign loader.  Endpoint and
credential-environment fields present in legacy model pods are deliberately
discarded from the projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from benchmark.scene_generation.campaign.profiles import (
    CampaignProfile,
    CampaignProfileBundle,
    GatewayOptions,
    ModelProfile,
    RequestOptions,
    RouteProfile,
    TransportPolicy,
)
from benchmark.scene_generation.frozen_two_stage.compatibility.loader import (
    inspect_model_metadata,
)
from benchmark.scene_generation.frozen_two_stage.config import load_run_config


@dataclass(frozen=True, slots=True)
class LegacyV1Projection:
    """Deployment-free semantic projection of one v1 run configuration."""

    legacy_route_kind: str
    option_contract_id: str
    runner_version: str
    model_profile_id: str
    display_label: str
    configured_model: str
    wire_model: str
    gateway_options: GatewayOptions
    request_options: RequestOptions
    transport_policy: TransportPolicy
    preflight_contract_id: str
    workflow_profile_id: str
    retrieval_profile_id: str
    brief_set_id: str
    execution_policy_id: str
    artifact_contract_id: str
    ordered_brief_ids: tuple[str, ...]


_LEGACY_BRIEF_SETS = {
    "c016ed9b926309c2a02278936dbc0e777a46b0bd30a71bcc1f563919418dcb55": (
        "hy34-paired-briefs-v1"
    )
}
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lookup_content_id(
    path: Path,
    table: dict[str, str],
    *,
    label: str,
) -> str:
    digest = _sha256(path)
    try:
        return table[digest]
    except KeyError as exc:
        raise ValueError(
            f"legacy {label} content has no reviewed v2 mapping: sha256={digest}"
        ) from exc


def _execution_policy_id(config: object, *, retries: int, delay: float) -> str:
    route = getattr(config, "route")
    retry = getattr(config, "retry")
    ordered_brief_ids = tuple(getattr(config, "ordered_brief_ids"))
    policy = getattr(config, "execution_policy")
    gateway_family = "api2" if route.kind.startswith("api2_") else "api3"
    signature = (
        gateway_family,
        retries,
        float(delay),
        tuple(retry.retryable_transport_statuses),
        tuple(retry.retryable_http_statuses),
        retry.retry_ambiguous_timeouts,
        retry.continue_after_case_failure,
    )
    common_statuses = (408, 409, 425, 429, 500, 502, 503, 504)
    expected = {
        (
            "api2",
            3,
            30.0,
            ("transport_failure",),
            common_statuses,
            False,
            True,
        ): "api2-scene10-retry3-continue-v1",
        (
            "api3",
            2,
            30.0,
            ("transport_failure",),
            common_statuses,
            False,
            True,
        ): "api3-scene10-retry2-continue-v1",
    }
    try:
        execution_policy_id = expected[signature]
    except KeyError as exc:
        raise ValueError(
            "legacy retry behavior has no reviewed v2 execution contract"
        ) from exc
    if policy.get("case_failure_policy") != "record_and_continue_next_brief":
        raise ValueError("legacy case-failure policy differs from v2 contract")
    if tuple(policy.get("expected_brief_ids", ())) != ordered_brief_ids:
        raise ValueError("legacy execution-policy brief order differs")
    if policy.get("maximum_infrastructure_retries") != retries:
        raise ValueError("legacy execution-policy retry count differs")
    required_statuses = tuple(policy.get("required_http_retry_statuses", ()))
    if any(status not in retry.retryable_http_statuses for status in required_statuses):
        raise ValueError("legacy execution-policy retry statuses are inconsistent")
    return execution_policy_id


def _option_contract_id(route_kind: str, chat_option_style: str | None) -> str:
    if route_kind == "api2_chat" and chat_option_style == "top_level_reasoning":
        return "chat_top_level_reasoning_v1"
    if route_kind == "api2_responses":
        return "responses_reasoning_effort_v1"
    if route_kind == "api3_chat" and chat_option_style == "adaptive_thinking":
        return "chat_adaptive_thinking_v1"
    if route_kind == "api3_chat" and chat_option_style == "legacy_core":
        return "chat_legacy_core_v1"
    raise ValueError(
        "legacy run config has no reviewed v2 option contract: "
        f"route_kind={route_kind!r}, chat_option_style={chat_option_style!r}"
    )


def _preflight_contract_id(config: object, option_contract_id: str) -> str:
    if option_contract_id == "chat_top_level_reasoning_v1":
        return "api2-chat-json-content-v1"
    if option_contract_id == "responses_reasoning_effort_v1":
        return "api2-responses-completed-json-v1"
    if option_contract_id == "chat_legacy_core_v1":
        return "api3-chat-content-model-identity-v1"
    if option_contract_id == "chat_adaptive_thinking_v1":
        policy = getattr(config, "execution_policy")
        if policy.get("preflight_requires_reasoning_signal") is not False:
            raise ValueError("legacy adaptive preflight failure rule differs")
        if policy.get("preflight_records_reasoning_signal_diagnostic") is not True:
            raise ValueError("legacy adaptive preflight diagnostic rule differs")
        return "api3-chat-adaptive-thinking-diagnostic-v1"
    raise ValueError(f"unsupported option contract: {option_contract_id!r}")


def project_legacy_v1(path: str | Path) -> LegacyV1Projection:
    """Project a trusted legacy config without retaining its local binding."""

    config = load_run_config(path)
    model = inspect_model_metadata(config.models_path, config.model_key)
    option_contract_id = _option_contract_id(
        config.route.kind, config.route.chat_option_style
    )
    if config.route.kind.startswith("api2_"):
        gateway_options = GatewayOptions(
            provider=config.route.provider,
            gateway_model=config.route.gateway_model,
            user_agent_suffix=config.route.user_agent_suffix,
            auth_timeout_seconds=config.route.timeout_seconds,
            strategy_type=model.strategy_type,
        )
    else:
        gateway_options = GatewayOptions(strategy_type=model.strategy_type)
    thinking_type = (
        "adaptive" if option_contract_id == "chat_adaptive_thinking_v1" else None
    )
    brief_set_id = _lookup_content_id(
        config.briefs_path,
        _LEGACY_BRIEF_SETS,
        label="brief set",
    )
    # Phase A's v1 loader resolves the historical implicit retriever to the
    # content-addressed v2 catalog/profile before this projection is built.
    retrieval_profile_id = config.retrieval_profile_id
    execution_policy_id = _execution_policy_id(
        config,
        retries=model.max_infrastructure_retries,
        delay=model.retry_delay_seconds,
    )
    preflight_contract_id = _preflight_contract_id(config, option_contract_id)
    return LegacyV1Projection(
        legacy_route_kind=config.route.kind,
        option_contract_id=option_contract_id,
        runner_version=config.route.runner_version,
        model_profile_id=model.key,
        display_label=model.label,
        configured_model=model.configured_model,
        wire_model=model.wire_model,
        gateway_options=gateway_options,
        request_options=RequestOptions(
            request_timeout_seconds=model.timeout_seconds,
            max_tokens=model.max_tokens,
            temperature=model.temperature,
            top_p=model.top_p,
            top_k=model.top_k,
            repetition_penalty=model.repetition_penalty,
            reasoning_effort=model.reasoning_effort,
            thinking_type=thinking_type,
            preserved_thinking=model.preserved_thinking,
        ),
        transport_policy=TransportPolicy(
            max_infrastructure_retries=model.max_infrastructure_retries,
            retry_delay_seconds=model.retry_delay_seconds,
        ),
        preflight_contract_id=preflight_contract_id,
        workflow_profile_id="frozen-two-stage-generation-v2",
        retrieval_profile_id=retrieval_profile_id,
        brief_set_id=brief_set_id,
        execution_policy_id=execution_policy_id,
        artifact_contract_id="hy34-two-stage-artifacts-v3",
        ordered_brief_ids=config.ordered_brief_ids,
    )


def _matches_route(projection: LegacyV1Projection, route: RouteProfile) -> bool:
    return (
        route.grammar.legacy_route_kind == projection.legacy_route_kind
        and route.option_contract_id == projection.option_contract_id
        and route.runner_version == projection.runner_version
    )


def _matches_model(
    projection: LegacyV1Projection,
    model: ModelProfile,
    route: RouteProfile,
) -> bool:
    return (
        _matches_route(projection, route)
        and model.model_profile_id == projection.model_profile_id
        and model.display_label == projection.display_label
        and model.configured_model == projection.configured_model
        and model.wire_model == projection.wire_model
        and model.gateway_options == projection.gateway_options
        and model.request_options == projection.request_options
        and model.transport_policy == projection.transport_policy
        and model.preflight_contract_id == projection.preflight_contract_id
    )


def map_legacy_v1_to_campaign(
    path: str | Path,
    bundle: CampaignProfileBundle,
) -> tuple[LegacyV1Projection, CampaignProfile, ModelProfile, RouteProfile]:
    """Return the unique v2 characterization or fail closed on ambiguity."""

    projection = project_legacy_v1(path)
    matching_models: list[tuple[ModelProfile, RouteProfile]] = []
    for model in bundle.models.models:
        route = bundle.routes.by_id[model.route_profile_id]
        if _matches_model(projection, model, route):
            matching_models.append((model, route))
    if len(matching_models) != 1:
        raise ValueError(
            "legacy model projection must match exactly one v2 profile; "
            f"matched={len(matching_models)}"
        )
    model, route = matching_models[0]
    campaigns = [
        campaign
        for campaign in bundle.campaigns.campaigns
        if campaign.model_profile_id == model.model_profile_id
        and campaign.ordered_brief_ids == projection.ordered_brief_ids
        and campaign.workflow_profile_id == projection.workflow_profile_id
        and campaign.retrieval_profile_id == projection.retrieval_profile_id
        and campaign.brief_set_id == projection.brief_set_id
        and campaign.execution_policy_id == projection.execution_policy_id
        and campaign.artifact_contract_id == projection.artifact_contract_id
    ]
    if len(campaigns) != 1:
        raise ValueError(
            "legacy run projection must match exactly one v2 campaign; "
            f"matched={len(campaigns)}"
        )
    return projection, campaigns[0], model, route
