"""Immutable, deployment-free profiles for generation campaigns.

The profile layer deliberately classifies protocol grammar rather than model
names.  A model alias is ordinary data.  Only request/response codecs, gateway
contracts, and option grammars are allowlisted because those require reviewed
Python implementations.

Runtime endpoints, credential environment variables, filesystem paths, and
retrieval resource locations are outside this module and must be supplied by a
separate local binding layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping


ROUTE_REGISTRY_SCHEMA_VERSION = "generation_route_profile_registry_v2"
MODEL_REGISTRY_SCHEMA_VERSION = "generation_model_profile_registry_v2"
CAMPAIGN_REGISTRY_SCHEMA_VERSION = "generation_campaign_registry_v2"
CONTRACT_REGISTRY_SCHEMA_VERSION = "generation_campaign_contract_registry_v2"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


@dataclass(frozen=True, slots=True)
class ProtocolGrammar:
    """One reviewed codec + gateway + option + response composition."""

    codec_id: str
    gateway_id: str
    option_contract_id: str
    response_contract_id: str
    legacy_route_kind: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.codec_id,
            self.gateway_id,
            self.option_contract_id,
            self.response_contract_id,
        )


_GRAMMARS = (
    ProtocolGrammar(
        codec_id="openai_chat_completions_v1",
        gateway_id="api2_bearer_query_v1",
        option_contract_id="chat_top_level_reasoning_v1",
        response_contract_id="openai_chat_single_choice_v1",
        legacy_route_kind="api2_chat",
    ),
    ProtocolGrammar(
        codec_id="openai_responses_v1",
        gateway_id="api2_bearer_query_v1",
        option_contract_id="responses_reasoning_effort_v1",
        response_contract_id="openai_responses_completed_output_text_v1",
        legacy_route_kind="api2_responses",
    ),
    ProtocolGrammar(
        codec_id="openai_chat_completions_v1",
        gateway_id="api3_bearer_session_v1",
        option_contract_id="chat_adaptive_thinking_v1",
        response_contract_id="openai_chat_single_choice_v1",
        legacy_route_kind="api3_chat",
    ),
    ProtocolGrammar(
        codec_id="openai_chat_completions_v1",
        gateway_id="api3_bearer_session_v1",
        option_contract_id="chat_legacy_core_v1",
        response_contract_id="openai_chat_single_choice_v1",
        legacy_route_kind="api3_chat",
    ),
)
PROTOCOL_GRAMMARS: Mapping[tuple[str, str, str, str], ProtocolGrammar] = (
    MappingProxyType({grammar.key: grammar for grammar in _GRAMMARS})
)


def _nonempty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _optional_string(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, field_name=field_name)


def _identifier(value: str, *, field_name: str) -> str:
    text = _nonempty(value, field_name=field_name)
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError(
            f"{field_name} must contain only portable identifier characters"
        )
    return text


def _finite_number(
    value: int | float | None,
    *,
    field_name: str,
    allow_none: bool = True,
) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must be numeric")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric or null")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be finite")
    return result


def _positive_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class RouteProfile:
    """Public protocol composition with no endpoint or credential binding."""

    route_profile_id: str
    codec_id: str
    gateway_id: str
    option_contract_id: str
    response_contract_id: str
    runner_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "route_profile_id",
            "codec_id",
            "gateway_id",
            "option_contract_id",
            "response_contract_id",
            "runner_version",
        ):
            _identifier(getattr(self, field_name), field_name=field_name)
        if self.grammar_key not in PROTOCOL_GRAMMARS:
            raise ValueError(
                "unsupported protocol grammar: "
                f"codec={self.codec_id!r}, gateway={self.gateway_id!r}, "
                f"options={self.option_contract_id!r}, "
                f"response={self.response_contract_id!r}"
            )

    @property
    def grammar_key(self) -> tuple[str, str, str, str]:
        return (
            self.codec_id,
            self.gateway_id,
            self.option_contract_id,
            self.response_contract_id,
        )

    @property
    def grammar(self) -> ProtocolGrammar:
        return PROTOCOL_GRAMMARS[self.grammar_key]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "route_profile_id": self.route_profile_id,
            "codec_id": self.codec_id,
            "gateway_id": self.gateway_id,
            "option_contract_id": self.option_contract_id,
            "response_contract_id": self.response_contract_id,
            "runner_version": self.runner_version,
        }


@dataclass(frozen=True, slots=True)
class GatewayOptions:
    """Wire-affecting public gateway values supplied by a model profile."""

    provider: str | None = None
    gateway_model: str | None = None
    user_agent_suffix: str | None = None
    auth_timeout_seconds: int | None = None
    strategy_type: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "provider",
            "gateway_model",
            "user_agent_suffix",
            "strategy_type",
        ):
            _optional_string(getattr(self, field_name), field_name=field_name)
        if self.user_agent_suffix is not None and (
            len(self.user_agent_suffix) > 160
            or any(character in self.user_agent_suffix for character in "\r\n")
        ):
            raise ValueError("user_agent_suffix is not a safe header fragment")
        if self.auth_timeout_seconds is not None:
            _positive_int(
                self.auth_timeout_seconds, field_name="auth_timeout_seconds"
            )

    def validate_for(self, route: RouteProfile) -> None:
        if route.gateway_id == "api2_bearer_query_v1":
            required = {
                "provider": self.provider,
                "gateway_model": self.gateway_model,
                "user_agent_suffix": self.user_agent_suffix,
                "auth_timeout_seconds": self.auth_timeout_seconds,
                "strategy_type": self.strategy_type,
            }
            missing = sorted(key for key, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    f"API2 gateway options are missing required fields: {missing}"
                )
            return
        if route.gateway_id == "api3_bearer_session_v1":
            if self.strategy_type is None:
                raise ValueError("API3 gateway options require strategy_type")
            unexpected = {
                "provider": self.provider,
                "gateway_model": self.gateway_model,
                "user_agent_suffix": self.user_agent_suffix,
                "auth_timeout_seconds": self.auth_timeout_seconds,
            }
            present = sorted(key for key, value in unexpected.items() if value is not None)
            if present:
                raise ValueError(
                    f"API3 gateway options contain unsupported fields: {present}"
                )
            return
        raise ValueError(f"unsupported gateway contract: {route.gateway_id!r}")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "provider": self.provider,
                "gateway_model": self.gateway_model,
                "user_agent_suffix": self.user_agent_suffix,
                "auth_timeout_seconds": self.auth_timeout_seconds,
                "strategy_type": self.strategy_type,
            }.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class RequestOptions:
    """Provider-neutral model values consumed by the reviewed option grammar."""

    request_timeout_seconds: float
    max_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    reasoning_effort: str | None
    thinking_type: str | None
    preserved_thinking: bool | None

    def __post_init__(self) -> None:
        timeout = _finite_number(
            self.request_timeout_seconds,
            field_name="request_timeout_seconds",
            allow_none=False,
        )
        assert timeout is not None
        if timeout <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        object.__setattr__(self, "request_timeout_seconds", timeout)
        _positive_int(self.max_tokens, field_name="max_tokens")
        for field_name in ("temperature", "top_p", "repetition_penalty"):
            value = _finite_number(getattr(self, field_name), field_name=field_name)
            object.__setattr__(self, field_name, value)
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int)
        ):
            raise ValueError("top_k must be an integer or null")
        _optional_string(self.reasoning_effort, field_name="reasoning_effort")
        _optional_string(self.thinking_type, field_name="thinking_type")
        if self.preserved_thinking is not None and not isinstance(
            self.preserved_thinking, bool
        ):
            raise ValueError("preserved_thinking must be a boolean or null")

    def validate_for(self, route: RouteProfile) -> None:
        option_id = route.option_contract_id
        if option_id in {
            "chat_top_level_reasoning_v1",
            "responses_reasoning_effort_v1",
        }:
            if self.reasoning_effort is None:
                raise ValueError(f"{option_id} requires reasoning_effort")
            if self.thinking_type is not None or self.preserved_thinking is not None:
                raise ValueError(f"{option_id} does not accept thinking fields")
            ignored = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
            }
            present = sorted(key for key, value in ignored.items() if value is not None)
            if present:
                raise ValueError(
                    f"{option_id} contains fields its codec cannot emit: {present}"
                )
            return
        if option_id == "chat_adaptive_thinking_v1":
            if self.reasoning_effort is None:
                raise ValueError("adaptive thinking requires reasoning_effort")
            if self.thinking_type != "adaptive":
                raise ValueError("adaptive thinking requires thinking_type='adaptive'")
            if self.preserved_thinking is not True:
                raise ValueError("adaptive thinking requires preserved_thinking=true")
            ignored = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
            }
            present = sorted(key for key, value in ignored.items() if value is not None)
            if present:
                raise ValueError(
                    "adaptive thinking contains fields its codec cannot emit: "
                    f"{present}"
                )
            return
        if option_id == "chat_legacy_core_v1":
            if self.thinking_type is not None:
                raise ValueError("legacy-core options do not accept thinking_type")
            if (self.reasoning_effort is None) != (
                self.preserved_thinking is None
            ):
                raise ValueError(
                    "legacy-core reasoning_effort and preserved_thinking must be "
                    "both set or both null"
                )
            return
        raise ValueError(f"unsupported option contract: {option_id!r}")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "reasoning_effort": self.reasoning_effort,
            "thinking_type": self.thinking_type,
            "preserved_thinking": self.preserved_thinking,
        }


@dataclass(frozen=True, slots=True)
class TransportPolicy:
    """Request timeout retry values currently owned by legacy model pods."""

    max_infrastructure_retries: int
    retry_delay_seconds: float

    def __post_init__(self) -> None:
        _nonnegative_int(
            self.max_infrastructure_retries,
            field_name="max_infrastructure_retries",
        )
        delay = _finite_number(
            self.retry_delay_seconds,
            field_name="retry_delay_seconds",
            allow_none=False,
        )
        assert delay is not None
        if delay < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        object.__setattr__(self, "retry_delay_seconds", delay)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
        }


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One model alias on an explicit protocol grammar.

    The profile is not a support whitelist.  Any alias can be represented when
    it conforms to a reviewed route grammar and passes the external preflight.
    """

    model_profile_id: str
    display_label: str
    configured_model: str
    wire_model: str
    route_profile_id: str
    gateway_options: GatewayOptions
    request_options: RequestOptions
    transport_policy: TransportPolicy
    preflight_contract_id: str
    response_model_identity: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "model_profile_id",
            "route_profile_id",
            "preflight_contract_id",
        ):
            _identifier(getattr(self, field_name), field_name=field_name)
        for field_name in ("display_label", "configured_model", "wire_model"):
            _nonempty(getattr(self, field_name), field_name=field_name)
        _optional_string(
            self.response_model_identity,
            field_name="response_model_identity",
        )
        if not isinstance(self.gateway_options, GatewayOptions):
            raise TypeError("gateway_options must be GatewayOptions")
        if not isinstance(self.request_options, RequestOptions):
            raise TypeError("request_options must be RequestOptions")
        if not isinstance(self.transport_policy, TransportPolicy):
            raise TypeError("transport_policy must be TransportPolicy")

    def validate_for(self, route: RouteProfile) -> None:
        if self.route_profile_id != route.route_profile_id:
            raise ValueError(
                "model route_profile_id does not match resolved route: "
                f"model={self.route_profile_id!r}, route={route.route_profile_id!r}"
            )
        self.gateway_options.validate_for(route)
        self.request_options.validate_for(route)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "model_profile_id": self.model_profile_id,
            "display_label": self.display_label,
            "configured_model": self.configured_model,
            "wire_model": self.wire_model,
            "route_profile_id": self.route_profile_id,
            "gateway_options": self.gateway_options.to_public_dict(),
            "request_options": self.request_options.to_public_dict(),
            "transport_policy": self.transport_policy.to_public_dict(),
            "preflight_contract_id": self.preflight_contract_id,
            "response_model_identity": self.response_model_identity,
        }


@dataclass(frozen=True, slots=True)
class CampaignProfile:
    """Portable campaign declaration containing IDs, never resource details."""

    campaign_id: str
    workflow_profile_id: str
    model_profile_id: str
    retrieval_profile_id: str
    brief_set_id: str
    ordered_brief_ids: tuple[str, ...]
    execution_policy_id: str
    artifact_contract_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "campaign_id",
            "workflow_profile_id",
            "model_profile_id",
            "retrieval_profile_id",
            "brief_set_id",
            "execution_policy_id",
            "artifact_contract_id",
        ):
            _identifier(getattr(self, field_name), field_name=field_name)
        values = tuple(self.ordered_brief_ids)
        if not values:
            raise ValueError("ordered_brief_ids must not be empty")
        for value in values:
            _identifier(value, field_name="ordered_brief_ids item")
        if len(values) != len(set(values)):
            raise ValueError("ordered_brief_ids must not contain duplicates")
        object.__setattr__(self, "ordered_brief_ids", values)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "workflow_profile_id": self.workflow_profile_id,
            "model_profile_id": self.model_profile_id,
            "retrieval_profile_id": self.retrieval_profile_id,
            "brief_set_id": self.brief_set_id,
            "ordered_brief_ids": list(self.ordered_brief_ids),
            "execution_policy_id": self.execution_policy_id,
            "artifact_contract_id": self.artifact_contract_id,
        }


@dataclass(frozen=True, slots=True)
class WorkflowContract:
    workflow_profile_id: str
    core_bundle_id: str
    runner_version: str
    runner_source_sha256: str
    stage_a_prompt_sha256: str
    stage_c_prompt_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("workflow_profile_id", "core_bundle_id", "runner_version"):
            _identifier(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "runner_source_sha256",
            "stage_a_prompt_sha256",
            "stage_c_prompt_sha256",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field_name} must be a lowercase sha256")


@dataclass(frozen=True, slots=True)
class BriefSetContract:
    brief_set_id: str
    core_bundle_id: str
    content_sha256: str
    ordered_brief_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.brief_set_id, field_name="brief_set_id")
        _identifier(self.core_bundle_id, field_name="core_bundle_id")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase sha256")
        values = tuple(self.ordered_brief_ids)
        if not values or len(values) != len(set(values)):
            raise ValueError("brief-set ordered_brief_ids must be non-empty and unique")
        for value in values:
            _identifier(value, field_name="brief-set ordered_brief_ids item")
        object.__setattr__(self, "ordered_brief_ids", values)


@dataclass(frozen=True, slots=True)
class ExecutionPolicyContract:
    execution_policy_id: str
    max_infrastructure_retries: int
    retry_delay_seconds: float
    retryable_transport_statuses: tuple[str, ...]
    retryable_http_statuses: tuple[int, ...]
    retry_ambiguous_timeouts: bool
    continue_after_case_failure: bool

    def __post_init__(self) -> None:
        _identifier(self.execution_policy_id, field_name="execution_policy_id")
        _nonnegative_int(
            self.max_infrastructure_retries,
            field_name="max_infrastructure_retries",
        )
        delay = _finite_number(
            self.retry_delay_seconds,
            field_name="retry_delay_seconds",
            allow_none=False,
        )
        assert delay is not None
        if delay < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        object.__setattr__(self, "retry_delay_seconds", delay)
        statuses = tuple(self.retryable_transport_statuses)
        if not statuses or len(statuses) != len(set(statuses)):
            raise ValueError("retryable_transport_statuses must be non-empty and unique")
        for status in statuses:
            _identifier(status, field_name="retryable transport status")
        http = tuple(self.retryable_http_statuses)
        if len(http) != len(set(http)) or any(
            isinstance(status, bool)
            or not isinstance(status, int)
            or not 100 <= status <= 599
            for status in http
        ):
            raise ValueError("retryable_http_statuses must be unique HTTP status codes")
        if not isinstance(self.retry_ambiguous_timeouts, bool):
            raise TypeError("retry_ambiguous_timeouts must be boolean")
        if not isinstance(self.continue_after_case_failure, bool):
            raise TypeError("continue_after_case_failure must be boolean")


@dataclass(frozen=True, slots=True)
class ArtifactContract:
    artifact_contract_id: str
    run_manifest_schema_version: str
    case_result_schema_version: str
    public_model_contract: str

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_contract_id",
            "run_manifest_schema_version",
            "case_result_schema_version",
            "public_model_contract",
        ):
            _identifier(getattr(self, field_name), field_name=field_name)


@dataclass(frozen=True, slots=True)
class PreflightContract:
    preflight_contract_id: str
    response_contract_id: str
    content_validation: str
    require_model_identity: bool
    require_reasoning_signal: bool
    record_reasoning_diagnostic: bool

    def __post_init__(self) -> None:
        _identifier(self.preflight_contract_id, field_name="preflight_contract_id")
        _identifier(self.response_contract_id, field_name="response_contract_id")
        if self.content_validation not in {"json_object", "nonempty_text"}:
            raise ValueError(
                "content_validation must be 'json_object' or 'nonempty_text'"
            )
        for field_name in (
            "require_model_identity",
            "require_reasoning_signal",
            "record_reasoning_diagnostic",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean")


@dataclass(frozen=True, slots=True)
class CampaignContractRegistry:
    workflows: tuple[WorkflowContract, ...]
    brief_sets: tuple[BriefSetContract, ...]
    execution_policies: tuple[ExecutionPolicyContract, ...]
    artifact_contracts: tuple[ArtifactContract, ...]
    preflight_contracts: tuple[PreflightContract, ...]

    @property
    def workflow_by_id(self) -> Mapping[str, WorkflowContract]:
        return _unique_by_id(
            self.workflows, id_field="workflow_profile_id", label="workflow contract"
        )

    @property
    def brief_set_by_id(self) -> Mapping[str, BriefSetContract]:
        return _unique_by_id(self.brief_sets, id_field="brief_set_id", label="brief-set contract")

    @property
    def execution_by_id(self) -> Mapping[str, ExecutionPolicyContract]:
        return _unique_by_id(
            self.execution_policies,
            id_field="execution_policy_id",
            label="execution-policy contract",
        )

    @property
    def artifact_by_id(self) -> Mapping[str, ArtifactContract]:
        return _unique_by_id(
            self.artifact_contracts,
            id_field="artifact_contract_id",
            label="artifact contract",
        )

    @property
    def preflight_by_id(self) -> Mapping[str, PreflightContract]:
        return _unique_by_id(
            self.preflight_contracts,
            id_field="preflight_contract_id",
            label="preflight contract",
        )


def _unique_by_id(
    values: Iterable[Any],
    *,
    id_field: str,
    label: str,
) -> Mapping[str, Any]:
    by_id: dict[str, Any] = {}
    for value in values:
        item_id = getattr(value, id_field)
        if item_id in by_id:
            raise ValueError(f"duplicate {label} ID: {item_id!r}")
        by_id[item_id] = value
    if not by_id:
        raise ValueError(f"{label} registry must not be empty")
    return MappingProxyType(by_id)


@dataclass(frozen=True, slots=True)
class RouteProfileRegistry:
    routes: tuple[RouteProfile, ...]

    def __post_init__(self) -> None:
        values = tuple(self.routes)
        _unique_by_id(values, id_field="route_profile_id", label="route profile")
        object.__setattr__(self, "routes", values)

    @property
    def by_id(self) -> Mapping[str, RouteProfile]:
        return _unique_by_id(
            self.routes, id_field="route_profile_id", label="route profile"
        )


@dataclass(frozen=True, slots=True)
class ModelProfileRegistry:
    models: tuple[ModelProfile, ...]

    def __post_init__(self) -> None:
        values = tuple(self.models)
        _unique_by_id(values, id_field="model_profile_id", label="model profile")
        object.__setattr__(self, "models", values)

    @property
    def by_id(self) -> Mapping[str, ModelProfile]:
        return _unique_by_id(
            self.models, id_field="model_profile_id", label="model profile"
        )


@dataclass(frozen=True, slots=True)
class CampaignProfileRegistry:
    campaigns: tuple[CampaignProfile, ...]

    def __post_init__(self) -> None:
        values = tuple(self.campaigns)
        _unique_by_id(values, id_field="campaign_id", label="campaign")
        object.__setattr__(self, "campaigns", values)

    @property
    def by_id(self) -> Mapping[str, CampaignProfile]:
        return _unique_by_id(self.campaigns, id_field="campaign_id", label="campaign")


@dataclass(frozen=True, slots=True)
class CampaignProfileBundle:
    """Cross-validated route, model, and campaign registries."""

    routes: RouteProfileRegistry
    models: ModelProfileRegistry
    campaigns: CampaignProfileRegistry
    contracts: CampaignContractRegistry

    def __post_init__(self) -> None:
        route_by_id = self.routes.by_id
        for model in self.models.models:
            try:
                route = route_by_id[model.route_profile_id]
            except KeyError as exc:
                raise ValueError(
                    f"model {model.model_profile_id!r} references unknown route "
                    f"{model.route_profile_id!r}"
                ) from exc
            model.validate_for(route)
        model_by_id = self.models.by_id
        for campaign in self.campaigns.campaigns:
            if campaign.model_profile_id not in model_by_id:
                raise ValueError(
                    f"campaign {campaign.campaign_id!r} references unknown model "
                    f"{campaign.model_profile_id!r}"
                )
            model = model_by_id[campaign.model_profile_id]
            route = route_by_id[model.route_profile_id]
            try:
                workflow = self.contracts.workflow_by_id[campaign.workflow_profile_id]
                brief_set = self.contracts.brief_set_by_id[campaign.brief_set_id]
                execution = self.contracts.execution_by_id[campaign.execution_policy_id]
                self.contracts.artifact_by_id[campaign.artifact_contract_id]
                preflight = self.contracts.preflight_by_id[model.preflight_contract_id]
            except KeyError as exc:
                raise ValueError(
                    f"campaign {campaign.campaign_id!r} contains an unknown contract reference: {exc}"
                ) from exc
            if workflow.core_bundle_id != brief_set.core_bundle_id:
                raise ValueError("workflow and brief set reference different core bundles")
            if campaign.ordered_brief_ids != brief_set.ordered_brief_ids:
                raise ValueError(
                    f"campaign {campaign.campaign_id!r} brief order differs from its brief-set contract"
                )
            if execution.max_infrastructure_retries != model.transport_policy.max_infrastructure_retries:
                raise ValueError("execution retry count differs from model transport policy")
            if execution.retry_delay_seconds != model.transport_policy.retry_delay_seconds:
                raise ValueError("execution retry delay differs from model transport policy")
            if preflight.response_contract_id != route.response_contract_id:
                raise ValueError("preflight response contract differs from the route response contract")
            if preflight.require_model_identity:
                if model.response_model_identity is None:
                    raise ValueError(
                        f"model {model.model_profile_id!r} requires an explicit "
                        "response_model_identity"
                    )
            elif model.response_model_identity is not None:
                raise ValueError(
                    f"model {model.model_profile_id!r} declares an unused "
                    "response_model_identity"
                )

    def resolve_campaign(
        self, campaign_id: str
    ) -> tuple[CampaignProfile, ModelProfile, RouteProfile]:
        campaign = self.campaigns.by_id[campaign_id]
        model = self.models.by_id[campaign.model_profile_id]
        route = self.routes.by_id[model.route_profile_id]
        return campaign, model, route
