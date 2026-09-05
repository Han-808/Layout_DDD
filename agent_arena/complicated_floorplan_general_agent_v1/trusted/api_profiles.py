#!/usr/bin/env python3
"""Strict, secret-free profiles for one-API/many-model Pi experiments.

The profile registry is controlled benchmark input.  Runtime endpoints are
loaded from a separate local binding file and credentials are supplied only to
the host supervisor.  Neither is copied into an Agent workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit


TRUSTED_ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = TRUSTED_ROOT / "profiles"
API_FAMILIES_PATH = PROFILE_ROOT / "api_families.json"
ROUTES_PATH = PROFILE_ROOT / "route_profiles.json"
MODELS_PATH = PROFILE_ROOT / "model_profiles.json"
PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_PI_APIS = frozenset({"openai-completions", "openai-responses"})
SUPPORTED_THINKING = frozenset({"off", "minimal", "low", "medium", "high"})
SUPPORTED_AUTH = frozenset(
    {
        "standard_bearer_v1",
        "api2_bearer_query_v1",
        "api3_bearer_session_v1",
    }
)
SUPPORTED_TRANSPORT_ADAPTERS = frozenset(
    {"direct_http_v1", "managed_tokenhub_litellm_v1"}
)
SUPPORTED_ATTEMPT_IDENTITIES = frozenset(
    {
        "none",
        "fresh_api2_cache_task_id_per_physical_attempt_v1",
        "fresh_api3_uuid_session_id_per_physical_attempt_v1",
    }
)
SUPPORTED_OPTION_STYLES = frozenset(
    {
        "legacy_core_v1",
        "top_level_reasoning_v1",
        "responses_reasoning_v1",
        "adaptive_thinking_v1",
    }
)
SUPPORTED_REASONING_STYLES = frozenset(
    {"none", "top_level_reasoning", "responses_reasoning", "adaptive_thinking"}
)
SUPPORTED_REASONING_EFFORTS = frozenset(
    {"minimal", "low", "medium", "high", "max"}
)
SUPPORTED_RESPONSE_CONTRACTS = frozenset(
    {
        "openai_chat_stream_tool_identity_v1",
        "openai_chat_stream_tool_v1",
        "openai_responses_stream_tool_v1",
    }
)
RETRYABLE_HTTP_DEFAULT = (408, 409, 425, 429, 500, 502, 503, 504)


class ProfileError(RuntimeError):
    """Raised when a controlled profile or private binding is inconsistent."""


@dataclass(frozen=True)
class ApiFamilyProfile:
    api_family_id: str
    display_label: str
    credential_kind: str
    prompt_once_per_experiment: bool
    route_profile_ids: tuple[str, ...]
    shared_cooldown_seconds: float
    maximum_concurrent_episodes: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "api_family_id": self.api_family_id,
            "display_label": self.display_label,
            "credential_kind": self.credential_kind,
            "prompt_once_per_experiment": self.prompt_once_per_experiment,
            "route_profile_ids": list(self.route_profile_ids),
            "shared_cooldown_seconds": self.shared_cooldown_seconds,
            "maximum_concurrent_episodes": self.maximum_concurrent_episodes,
        }


@dataclass(frozen=True)
class RouteProfile:
    route_profile_id: str
    api_family_id: str
    pi_api_protocol: str
    client_path: str
    upstream_path: str
    auth_strategy: str
    transport_adapter_id: str
    physical_attempt_identity_strategy: str
    option_style: str
    response_contract: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "route_profile_id": self.route_profile_id,
            "api_family_id": self.api_family_id,
            "pi_api_protocol": self.pi_api_protocol,
            "client_path": self.client_path,
            "upstream_path": self.upstream_path,
            "auth_strategy": self.auth_strategy,
            "transport_adapter_id": self.transport_adapter_id,
            "physical_attempt_identity_strategy": self.physical_attempt_identity_strategy,
            "option_style": self.option_style,
            "response_contract": self.response_contract,
        }


@dataclass(frozen=True)
class RetryPolicy:
    max_infrastructure_retries: int
    retry_delay_seconds: float
    retryable_http_statuses: tuple[int, ...]
    retry_transport_failures: bool
    retry_ambiguous_timeouts: bool

    @property
    def maximum_attempts(self) -> int:
        return 1 + self.max_infrastructure_retries

    def public_dict(self) -> dict[str, Any]:
        return {
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "retryable_http_statuses": list(self.retryable_http_statuses),
            "retry_transport_failures": self.retry_transport_failures,
            "retry_ambiguous_timeouts": self.retry_ambiguous_timeouts,
        }


@dataclass(frozen=True)
class ReasoningProfile:
    style: str
    effort: str | None
    thinking_type: str | None
    preserve_across_tool_turns: bool

    def fixed_request_fields(self) -> dict[str, Any]:
        if self.style == "none":
            return {}
        if self.style == "top_level_reasoning":
            return {"reasoning_effort": self.effort}
        if self.style == "responses_reasoning":
            return {"reasoning": {"effort": self.effort}}
        if self.style == "adaptive_thinking":
            return {
                "thinking": {"type": self.thinking_type},
                "reasoning_effort": self.effort,
            }
        raise ProfileError(f"unsupported reasoning style: {self.style}")

    def public_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "effort": self.effort,
            "thinking_type": self.thinking_type,
            "preserve_across_tool_turns": self.preserve_across_tool_turns,
        }


@dataclass(frozen=True)
class PiModelProfile:
    api_protocol: str
    thinking_level: str
    context_window: int
    maximum_output_tokens: int
    compatibility: Mapping[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "api_protocol": self.api_protocol,
            "thinking_level": self.thinking_level,
            "context_window": self.context_window,
            "maximum_output_tokens": self.maximum_output_tokens,
            "compatibility": dict(self.compatibility),
        }


@dataclass(frozen=True)
class ModelProfile:
    model_profile_id: str
    display_label: str
    api_family_id: str
    route_profile_id: str
    client_wire_model: str
    upstream_wire_model: str
    request_timeout_seconds: float
    temperature: float | None
    reasoning: ReasoningProfile
    pi: PiModelProfile
    retry: RetryPolicy
    auth_query_parameters: Mapping[str, str]
    api3_strategy_type: str | None
    user_agent_suffix: str | None
    response_identity_required: bool
    accepted_response_models: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "model_profile_id": self.model_profile_id,
            "display_label": self.display_label,
            "api_family_id": self.api_family_id,
            "route_profile_id": self.route_profile_id,
            "client_wire_model": self.client_wire_model,
            "upstream_wire_model": self.upstream_wire_model,
            "request_timeout_seconds": self.request_timeout_seconds,
            "temperature": self.temperature,
            "reasoning": self.reasoning.public_dict(),
            "pi": self.pi.public_dict(),
            "retry": self.retry.public_dict(),
            "auth_query_parameters": dict(self.auth_query_parameters),
            "api3_strategy_type": self.api3_strategy_type,
            "user_agent_suffix": self.user_agent_suffix,
            "response_identity": {
                "required": self.response_identity_required,
                "accepted_models": list(self.accepted_response_models),
            },
        }


@dataclass(frozen=True)
class ExperimentProfile:
    experiment_id: str
    agent_id_prefix: str
    api_family_id: str
    model_profile_ids: tuple[str, ...]
    scene_ids: tuple[str, ...]
    harness_id: str
    maximum_model_requests: int
    wall_clock_seconds: int
    maximum_concurrent_episodes: int
    continue_after_episode_failure: bool
    require_tool_call_preflight: bool
    episode_attempts: int
    resume_policy: str
    source_path: Path
    source_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "agent_id_prefix": self.agent_id_prefix,
            "api_family_id": self.api_family_id,
            "model_profile_ids": list(self.model_profile_ids),
            "scene_ids": list(self.scene_ids),
            "harness_id": self.harness_id,
            "limits": {
                "maximum_model_requests": self.maximum_model_requests,
                "wall_clock_seconds": self.wall_clock_seconds,
                "maximum_concurrent_episodes": self.maximum_concurrent_episodes,
                "episode_attempts": self.episode_attempts,
            },
            "execution": {
                "continue_after_episode_failure": self.continue_after_episode_failure,
                "require_tool_call_preflight": self.require_tool_call_preflight,
                "resume_policy": self.resume_policy,
            },
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class RouteRuntimeBinding:
    route_profile_id: str
    binding_profile_id: str
    upstream_base_url: str
    allow_insecure_upstream: bool
    managed_adapter_id: str | None = None


@dataclass(frozen=True)
class RuntimeBindings:
    api_family_id: str
    routes: Mapping[str, RouteRuntimeBinding]
    source_path: Path
    source_sha256: str

    def for_route(self, route_profile_id: str) -> RouteRuntimeBinding:
        try:
            return self.routes[route_profile_id]
        except KeyError as exc:
            raise ProfileError(
                f"runtime binding is missing route: {route_profile_id}"
            ) from exc


@dataclass(frozen=True)
class ProfileRegistry:
    api_families: Mapping[str, ApiFamilyProfile]
    routes: Mapping[str, RouteProfile]
    models: Mapping[str, ModelProfile]
    content_sha256: str

    @classmethod
    def load(cls, profile_root: str | Path = PROFILE_ROOT) -> "ProfileRegistry":
        root_input = Path(profile_root).expanduser().absolute()
        if not root_input.is_dir() or root_input.is_symlink():
            raise ProfileError("profile root must be a real directory")
        root = root_input.resolve(strict=True)
        family_path = root / API_FAMILIES_PATH.name
        route_path = root / ROUTES_PATH.name
        model_path = root / MODELS_PATH.name
        families_raw = _read_json(family_path)
        routes_raw = _read_json(route_path)
        models_raw = _read_json(model_path)
        if families_raw.get("schema_version") != "sieve_agent_api_families_v1":
            raise ProfileError("unsupported API-family registry schema")
        if routes_raw.get("schema_version") != "sieve_agent_route_profiles_v1":
            raise ProfileError("unsupported route registry schema")
        if models_raw.get("schema_version") != "sieve_agent_model_profiles_v1":
            raise ProfileError("unsupported model registry schema")
        families = _load_families(families_raw.get("api_families"))
        routes = _load_routes(routes_raw.get("routes"))
        models = _load_models(models_raw.get("models"))
        _cross_validate(families, routes, models)
        hashes = {
            path.name: _sha256_file(path)
            for path in (family_path, route_path, model_path)
        }
        content_sha256 = hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(families, routes, models, content_sha256)

    def api_family(self, api_family_id: str) -> ApiFamilyProfile:
        try:
            return self.api_families[api_family_id]
        except KeyError as exc:
            raise ProfileError(f"unknown API family: {api_family_id}") from exc

    def route(self, route_profile_id: str) -> RouteProfile:
        try:
            return self.routes[route_profile_id]
        except KeyError as exc:
            raise ProfileError(f"unknown route profile: {route_profile_id}") from exc

    def model(self, model_profile_id: str) -> ModelProfile:
        try:
            return self.models[model_profile_id]
        except KeyError as exc:
            raise ProfileError(f"unknown model profile: {model_profile_id}") from exc


def load_experiment(
    path: str | Path, registry: ProfileRegistry
) -> ExperimentProfile:
    source = _real_input_file(path, "experiment")
    value = _read_json(source)
    _keys(
        value,
        required={
            "schema_version",
            "experiment_id",
            "agent_id_prefix",
            "api_family_id",
            "model_profile_ids",
            "scene_ids",
            "harness_id",
            "limits",
            "execution",
        },
        label="experiment",
    )
    if value["schema_version"] != "sieve_pi_experiment_v1":
        raise ProfileError("unsupported experiment schema")
    experiment_id = _portable_id(value["experiment_id"], "experiment_id")
    agent_id_prefix = _portable_id(value["agent_id_prefix"], "agent_id_prefix")
    api_family_id = _portable_id(value["api_family_id"], "api_family_id")
    family = registry.api_family(api_family_id)
    model_ids = _string_tuple(value["model_profile_ids"], "model_profile_ids")
    scene_ids = _string_tuple(value["scene_ids"], "scene_ids")
    for scene_id in scene_ids:
        _portable_id(scene_id, "scene_id")
    for model_id in model_ids:
        model = registry.model(model_id)
        if model.api_family_id != api_family_id:
            raise ProfileError(
                "one experiment cannot mix API families: "
                f"{model_id} belongs to {model.api_family_id}, not {api_family_id}"
            )
    harness_id = _text(value["harness_id"], "harness_id")
    if harness_id != "sieve-pi-common-harness-v4":
        raise ProfileError("experiment must use the frozen Pi harness")
    limits = _object(value["limits"], "limits")
    _keys(
        limits,
        required={
            "maximum_model_requests",
            "wall_clock_seconds",
            "maximum_concurrent_episodes",
            "episode_attempts",
        },
        label="limits",
    )
    concurrency = _positive_int(
        limits["maximum_concurrent_episodes"], "maximum_concurrent_episodes"
    )
    if concurrency > family.maximum_concurrent_episodes:
        raise ProfileError("experiment concurrency exceeds the API-family ceiling")
    attempts = _positive_int(limits["episode_attempts"], "episode_attempts")
    # Request-level retries are owned by the scoped gateway.  Retrying an
    # entire Agent episode cannot prove that no earlier model request reached
    # the provider, so official runs deliberately have one pristine attempt.
    if attempts != 1:
        raise ProfileError("official experiments require episode_attempts=1")
    execution = _object(value["execution"], "execution")
    _keys(
        execution,
        required={
            "continue_after_episode_failure",
            "require_tool_call_preflight",
            "resume_policy",
        },
        label="execution",
    )
    resume_policy = _text(execution["resume_policy"], "resume_policy")
    if resume_policy != "sealed_hash_match_only_v1":
        raise ProfileError("unsupported resume policy")
    require_preflight = _boolean(
        execution["require_tool_call_preflight"],
        "require_tool_call_preflight",
    )
    if not require_preflight:
        raise ProfileError("official experiments require a streaming tool-call preflight")
    return ExperimentProfile(
        experiment_id=experiment_id,
        agent_id_prefix=agent_id_prefix,
        api_family_id=api_family_id,
        model_profile_ids=model_ids,
        scene_ids=scene_ids,
        harness_id=harness_id,
        maximum_model_requests=_positive_int(
            limits["maximum_model_requests"], "maximum_model_requests"
        ),
        wall_clock_seconds=_positive_int(
            limits["wall_clock_seconds"], "wall_clock_seconds"
        ),
        maximum_concurrent_episodes=concurrency,
        continue_after_episode_failure=_boolean(
            execution["continue_after_episode_failure"],
            "continue_after_episode_failure",
        ),
        require_tool_call_preflight=require_preflight,
        episode_attempts=attempts,
        resume_policy=resume_policy,
        source_path=source,
        source_sha256=_sha256_file(source),
    )


def load_runtime_bindings(
    path: str | Path,
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
) -> RuntimeBindings:
    source = _real_input_file(path, "runtime bindings")
    value = _read_json(source)
    _keys(value, required={"schema_version", "api_families"}, label="bindings")
    if value["schema_version"] != "sieve_agent_runtime_bindings_v2":
        raise ProfileError("unsupported runtime-binding schema")
    families = _object(value["api_families"], "api_families")
    raw_family = _object(
        families.get(experiment.api_family_id),
        f"api_families.{experiment.api_family_id}",
    )
    _keys(raw_family, required={"routes"}, label="runtime API family")
    raw_routes = _object(raw_family["routes"], "runtime routes")
    required_routes = {
        registry.model(model_id).route_profile_id
        for model_id in experiment.model_profile_ids
    }
    routes: dict[str, RouteRuntimeBinding] = {}
    for route_id in sorted(required_routes):
        item = _object(raw_routes.get(route_id), f"runtime route {route_id}")
        _keys(
            item,
            required={
                "binding_profile_id",
                "upstream_base_url",
                "allow_insecure_upstream",
            },
            label=f"runtime route {route_id}",
        )
        url = _upstream_base_url(item["upstream_base_url"])
        allow_insecure = _boolean(
            item["allow_insecure_upstream"], "allow_insecure_upstream"
        )
        parsed = urlsplit(url)
        if parsed.scheme == "http" and not allow_insecure:
            raise ProfileError(
                f"HTTP runtime route requires explicit local acknowledgement: {route_id}"
            )
        route = registry.route(route_id)
        if route.transport_adapter_id == "managed_tokenhub_litellm_v1" and (
            parsed.scheme != "https"
            or allow_insecure
            or parsed.path not in {"", "/"}
            or parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ProfileError(
                "managed TokenHub binding must be an HTTPS provider root; "
                "the host adapter, not the operator binding, owns /v1"
            )
        routes[route_id] = RouteRuntimeBinding(
            route_profile_id=route_id,
            binding_profile_id=_portable_id(
                item["binding_profile_id"], "binding_profile_id"
            ),
            upstream_base_url=url,
            allow_insecure_upstream=allow_insecure,
            managed_adapter_id=None,
        )
    return RuntimeBindings(
        api_family_id=experiment.api_family_id,
        routes=routes,
        source_path=source,
        source_sha256=_sha256_file(source),
    )


def normalize_runtime_credential(
    family: ApiFamilyProfile, raw_value: str
) -> str:
    if not isinstance(raw_value, str) or not raw_value or "\n" in raw_value or "\r" in raw_value:
        raise ProfileError("runtime credential is empty or malformed")
    if family.credential_kind == "api2_app_id_app_key_v1":
        base = raw_value.split("?", 1)[0]
        if base.count(":") != 1 or any(not part for part in base.split(":", 1)):
            raise ProfileError("API2 credential must have APP_ID:APP_KEY form")
        return base
    if family.credential_kind == "bearer_token_v1":
        return raw_value
    raise ProfileError(f"unsupported credential kind: {family.credential_kind}")


def _load_families(value: Any) -> dict[str, ApiFamilyProfile]:
    rows = _array(value, "api_families")
    output: dict[str, ApiFamilyProfile] = {}
    for index, raw in enumerate(rows):
        item = _object(raw, f"api_families[{index}]")
        _keys(
            item,
            required={
                "api_family_id",
                "display_label",
                "credential_kind",
                "prompt_once_per_experiment",
                "route_profile_ids",
                "shared_cooldown_seconds",
                "maximum_concurrent_episodes",
            },
            label=f"api_families[{index}]",
        )
        identity = _portable_id(item["api_family_id"], "api_family_id")
        if identity in output:
            raise ProfileError(f"duplicate API family: {identity}")
        credential_kind = _text(item["credential_kind"], "credential_kind")
        if credential_kind not in {"api2_app_id_app_key_v1", "bearer_token_v1"}:
            raise ProfileError(f"unsupported credential kind: {credential_kind}")
        prompt_once = _boolean(
            item["prompt_once_per_experiment"], "prompt_once_per_experiment"
        )
        if not prompt_once:
            raise ProfileError("credentials must be acquired once per experiment")
        output[identity] = ApiFamilyProfile(
            api_family_id=identity,
            display_label=_text(item["display_label"], "display_label"),
            credential_kind=credential_kind,
            prompt_once_per_experiment=prompt_once,
            route_profile_ids=_string_tuple(
                item["route_profile_ids"], "route_profile_ids"
            ),
            shared_cooldown_seconds=_nonnegative_number(
                item["shared_cooldown_seconds"], "shared_cooldown_seconds"
            ),
            maximum_concurrent_episodes=_positive_int(
                item["maximum_concurrent_episodes"],
                "maximum_concurrent_episodes",
            ),
        )
    if not output:
        raise ProfileError("API-family registry is empty")
    return output


def _load_routes(value: Any) -> dict[str, RouteProfile]:
    rows = _array(value, "routes")
    output: dict[str, RouteProfile] = {}
    for index, raw in enumerate(rows):
        item = _object(raw, f"routes[{index}]")
        _keys(
            item,
            required={
                "route_profile_id",
                "api_family_id",
                "pi_api_protocol",
                "client_path",
                "upstream_path",
                "auth_strategy",
                "transport_adapter_id",
                "physical_attempt_identity_strategy",
                "option_style",
                "response_contract",
            },
            label=f"routes[{index}]",
        )
        identity = _portable_id(item["route_profile_id"], "route_profile_id")
        if identity in output:
            raise ProfileError(f"duplicate route profile: {identity}")
        protocol = _text(item["pi_api_protocol"], "pi_api_protocol")
        if protocol not in SUPPORTED_PI_APIS:
            raise ProfileError(f"unsupported Pi API protocol: {protocol}")
        auth = _text(item["auth_strategy"], "auth_strategy")
        if auth not in SUPPORTED_AUTH:
            raise ProfileError(f"unsupported auth strategy: {auth}")
        adapter = _text(item["transport_adapter_id"], "transport_adapter_id")
        if adapter not in SUPPORTED_TRANSPORT_ADAPTERS:
            raise ProfileError(f"unsupported transport adapter: {adapter}")
        attempt_identity = _text(
            item["physical_attempt_identity_strategy"],
            "physical_attempt_identity_strategy",
        )
        if attempt_identity not in SUPPORTED_ATTEMPT_IDENTITIES:
            raise ProfileError(
                f"unsupported physical-attempt identity: {attempt_identity}"
            )
        style = _text(item["option_style"], "option_style")
        if style not in SUPPORTED_OPTION_STYLES:
            raise ProfileError(f"unsupported option style: {style}")
        response_contract = _text(item["response_contract"], "response_contract")
        if response_contract not in SUPPORTED_RESPONSE_CONTRACTS:
            raise ProfileError(
                f"unsupported streaming response contract: {response_contract}"
            )
        output[identity] = RouteProfile(
            route_profile_id=identity,
            api_family_id=_portable_id(item["api_family_id"], "api_family_id"),
            pi_api_protocol=protocol,
            client_path=_http_path(item["client_path"], "client_path"),
            upstream_path=_http_path(item["upstream_path"], "upstream_path"),
            auth_strategy=auth,
            transport_adapter_id=adapter,
            physical_attempt_identity_strategy=attempt_identity,
            option_style=style,
            response_contract=response_contract,
        )
    if not output:
        raise ProfileError("route registry is empty")
    return output


def _load_models(value: Any) -> dict[str, ModelProfile]:
    rows = _array(value, "models")
    output: dict[str, ModelProfile] = {}
    for index, raw in enumerate(rows):
        item = _object(raw, f"models[{index}]")
        _keys(
            item,
            required={
                "model_profile_id",
                "display_label",
                "api_family_id",
                "route_profile_id",
                "client_wire_model",
                "upstream_wire_model",
                "request_timeout_seconds",
                "temperature",
                "reasoning",
                "pi",
                "retry",
                "auth_query_parameters",
                "api3_strategy_type",
                "user_agent_suffix",
                "response_identity",
            },
            label=f"models[{index}]",
        )
        identity = _portable_id(item["model_profile_id"], "model_profile_id")
        if identity in output:
            raise ProfileError(f"duplicate model profile: {identity}")
        reasoning = _load_reasoning(item["reasoning"], identity)
        pi = _load_pi(item["pi"], identity)
        retry = _load_retry(item["retry"], identity)
        auth_query = _object(item["auth_query_parameters"], "auth_query_parameters")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(child, str)
            or not child
            for key, child in auth_query.items()
        ):
            raise ProfileError("auth query parameters must be non-empty strings")
        response = _object(item["response_identity"], "response_identity")
        _keys(
            response,
            required={"required", "accepted_models"},
            label="response_identity",
        )
        required = _boolean(response["required"], "response identity required")
        accepted = _string_tuple(
            response["accepted_models"],
            "accepted_models",
            allow_empty=not required,
        )
        if required and not accepted:
            raise ProfileError("required response identity needs accepted models")
        temperature = item["temperature"]
        if temperature is not None:
            temperature = _nonnegative_number(temperature, "temperature")
        suffix = item["user_agent_suffix"]
        if suffix is not None:
            suffix = _portable_id(suffix, "user_agent_suffix")
        strategy_type = item["api3_strategy_type"]
        if strategy_type is not None:
            strategy_type = _portable_id(strategy_type, "api3_strategy_type")
        output[identity] = ModelProfile(
            model_profile_id=identity,
            display_label=_text(item["display_label"], "display_label"),
            api_family_id=_portable_id(item["api_family_id"], "api_family_id"),
            route_profile_id=_portable_id(
                item["route_profile_id"], "route_profile_id"
            ),
            client_wire_model=_model_id(
                item["client_wire_model"], "client_wire_model"
            ),
            upstream_wire_model=_model_id(
                item["upstream_wire_model"], "upstream_wire_model"
            ),
            request_timeout_seconds=_positive_number(
                item["request_timeout_seconds"], "request_timeout_seconds"
            ),
            temperature=temperature,
            reasoning=reasoning,
            pi=pi,
            retry=retry,
            auth_query_parameters=dict(auth_query),
            api3_strategy_type=strategy_type,
            user_agent_suffix=suffix,
            response_identity_required=required,
            accepted_response_models=accepted,
        )
    if not output:
        raise ProfileError("model registry is empty")
    return output


def _load_reasoning(value: Any, model_id: str) -> ReasoningProfile:
    item = _object(value, f"{model_id}.reasoning")
    _keys(
        item,
        required={"style", "effort", "thinking_type", "preserve_across_tool_turns"},
        label=f"{model_id}.reasoning",
    )
    style = _text(item["style"], "reasoning style")
    if style not in SUPPORTED_REASONING_STYLES:
        raise ProfileError(f"unsupported reasoning style: {style}")
    effort = item["effort"]
    if effort is not None:
        effort = _text(effort, "reasoning effort")
        if effort not in SUPPORTED_REASONING_EFFORTS:
            raise ProfileError(f"unsupported reasoning effort: {effort}")
    thinking_type = item["thinking_type"]
    if thinking_type is not None:
        thinking_type = _text(thinking_type, "thinking_type")
    preserve = _boolean(
        item["preserve_across_tool_turns"], "preserve_across_tool_turns"
    )
    if style == "none" and (effort is not None or thinking_type is not None or preserve):
        raise ProfileError("reasoning style none cannot carry reasoning settings")
    if style != "none" and effort is None:
        raise ProfileError("reasoning-enabled model requires an effort")
    if style == "adaptive_thinking" and thinking_type != "adaptive":
        raise ProfileError("adaptive reasoning requires thinking_type=adaptive")
    if style != "adaptive_thinking" and thinking_type is not None:
        raise ProfileError("thinking_type is only valid for adaptive reasoning")
    return ReasoningProfile(style, effort, thinking_type, preserve)


def _load_pi(value: Any, model_id: str) -> PiModelProfile:
    item = _object(value, f"{model_id}.pi")
    _keys(
        item,
        required={
            "api_protocol",
            "thinking_level",
            "context_window",
            "maximum_output_tokens",
            "compatibility",
        },
        label=f"{model_id}.pi",
    )
    api = _text(item["api_protocol"], "Pi API protocol")
    if api not in SUPPORTED_PI_APIS:
        raise ProfileError(f"unsupported Pi API protocol: {api}")
    thinking = _text(item["thinking_level"], "Pi thinking level")
    if thinking not in SUPPORTED_THINKING:
        raise ProfileError(f"unsupported Pi thinking level: {thinking}")
    context = _positive_int(item["context_window"], "context_window")
    output = _positive_int(item["maximum_output_tokens"], "maximum_output_tokens")
    if output > context:
        raise ProfileError("maximum output tokens exceed context window")
    compatibility = _object(item["compatibility"], "Pi compatibility")
    _validate_pi_compatibility(compatibility, api_protocol=api)
    return PiModelProfile(api, thinking, context, output, dict(compatibility))


def _load_retry(value: Any, model_id: str) -> RetryPolicy:
    item = _object(value, f"{model_id}.retry")
    _keys(
        item,
        required={
            "max_infrastructure_retries",
            "retry_delay_seconds",
            "retryable_http_statuses",
            "retry_transport_failures",
            "retry_ambiguous_timeouts",
        },
        label=f"{model_id}.retry",
    )
    maximum = _nonnegative_int(
        item["max_infrastructure_retries"], "max_infrastructure_retries"
    )
    if maximum > 10:
        raise ProfileError("infrastructure retry count exceeds hard ceiling 10")
    statuses_raw = _array(item["retryable_http_statuses"], "retryable_http_statuses")
    statuses: list[int] = []
    for child in statuses_raw:
        if isinstance(child, bool) or not isinstance(child, int) or not 400 <= child <= 599:
            raise ProfileError("retryable HTTP statuses must be 4xx/5xx integers")
        statuses.append(child)
    if len(statuses) != len(set(statuses)):
        raise ProfileError("retryable HTTP statuses contain duplicates")
    ambiguous = _boolean(
        item["retry_ambiguous_timeouts"], "retry_ambiguous_timeouts"
    )
    if ambiguous:
        raise ProfileError("official Agent routes cannot retry ambiguous timeouts")
    return RetryPolicy(
        max_infrastructure_retries=maximum,
        retry_delay_seconds=_nonnegative_number(
            item["retry_delay_seconds"], "retry_delay_seconds"
        ),
        retryable_http_statuses=tuple(statuses),
        retry_transport_failures=_boolean(
            item["retry_transport_failures"], "retry_transport_failures"
        ),
        retry_ambiguous_timeouts=ambiguous,
    )


def _cross_validate(
    families: Mapping[str, ApiFamilyProfile],
    routes: Mapping[str, RouteProfile],
    models: Mapping[str, ModelProfile],
) -> None:
    declared_routes: set[str] = set()
    for family in families.values():
        for route_id in family.route_profile_ids:
            if route_id in declared_routes:
                raise ProfileError(f"route belongs to multiple API families: {route_id}")
            route = routes.get(route_id)
            if route is None or route.api_family_id != family.api_family_id:
                raise ProfileError(f"API family references an incompatible route: {route_id}")
            declared_routes.add(route_id)
    if declared_routes != set(routes):
        raise ProfileError("one or more routes are not owned by an API family")
    protocol_contract = {
        "openai-completions": {
            "client_path": "/v1/chat/completions",
            "upstream_path": "/chat/completions",
            "response_contracts": {
                "openai_chat_stream_tool_v1",
                "openai_chat_stream_tool_identity_v1",
            },
        },
        "openai-responses": {
            "client_path": "/v1/responses",
            "upstream_path": "/responses",
            "response_contracts": {"openai_responses_stream_tool_v1"},
        },
    }
    family_auth = {
        "api2": {"standard_bearer_v1", "api2_bearer_query_v1"},
        "api3": {"api3_bearer_session_v1"},
        "tokenhub": {"standard_bearer_v1"},
    }
    family_adapter = {
        "api2": "direct_http_v1",
        "api3": "direct_http_v1",
        "tokenhub": "managed_tokenhub_litellm_v1",
    }
    auth_attempt_identity = {
        "standard_bearer_v1": "none",
        "api2_bearer_query_v1": (
            "fresh_api2_cache_task_id_per_physical_attempt_v1"
        ),
        "api3_bearer_session_v1": (
            "fresh_api3_uuid_session_id_per_physical_attempt_v1"
        ),
    }
    for route in routes.values():
        contract = protocol_contract[route.pi_api_protocol]
        if (
            route.client_path != contract["client_path"]
            or route.upstream_path != contract["upstream_path"]
            or route.response_contract not in contract["response_contracts"]
        ):
            raise ProfileError(
                f"route protocol/path/stream contract disagree: {route.route_profile_id}"
            )
        if route.auth_strategy not in family_auth[route.api_family_id]:
            raise ProfileError(
                f"route family/auth strategy disagree: {route.route_profile_id}"
            )
        if route.transport_adapter_id != family_adapter[route.api_family_id]:
            raise ProfileError(
                f"route family/transport adapter disagree: {route.route_profile_id}"
            )
        if (
            route.physical_attempt_identity_strategy
            != auth_attempt_identity[route.auth_strategy]
        ):
            raise ProfileError(
                f"route auth/attempt identity disagree: {route.route_profile_id}"
            )
    expected_reasoning = {
        "legacy_core_v1": "none",
        "top_level_reasoning_v1": "top_level_reasoning",
        "responses_reasoning_v1": "responses_reasoning",
        "adaptive_thinking_v1": "adaptive_thinking",
    }
    for model in models.values():
        family = families.get(model.api_family_id)
        route = routes.get(model.route_profile_id)
        if family is None or route is None:
            raise ProfileError(f"model references an unknown API family or route: {model.model_profile_id}")
        if route.api_family_id != model.api_family_id:
            raise ProfileError(f"model route belongs to a different API family: {model.model_profile_id}")
        if model.pi.api_protocol != route.pi_api_protocol:
            raise ProfileError(f"Pi and route protocols disagree: {model.model_profile_id}")
        if model.reasoning.style != expected_reasoning[route.option_style]:
            raise ProfileError(f"route and reasoning styles disagree: {model.model_profile_id}")
        if model.retry.retry_delay_seconds != family.shared_cooldown_seconds:
            raise ProfileError(
                "model retry delay must equal its API-family shared cooldown: "
                f"{model.model_profile_id}"
            )
        if route.auth_strategy == "api2_bearer_query_v1":
            if set(model.auth_query_parameters) != {"provider", "model", "timeout"}:
                raise ProfileError(
                    f"API2 query auth parameters differ: {model.model_profile_id}"
                )
            if model.auth_query_parameters["model"] != model.upstream_wire_model:
                raise ProfileError(
                    f"API2 query model differs from wire model: {model.model_profile_id}"
                )
            try:
                query_timeout = float(model.auth_query_parameters["timeout"])
            except ValueError as exc:
                raise ProfileError(
                    f"API2 query timeout is invalid: {model.model_profile_id}"
                ) from exc
            if not math.isfinite(query_timeout) or query_timeout <= 0:
                raise ProfileError(
                    f"API2 query timeout is invalid: {model.model_profile_id}"
                )
            if query_timeout >= float(model.request_timeout_seconds):
                raise ProfileError(
                    "API2 platform timeout must be below the local request timeout: "
                    f"{model.model_profile_id}"
                )
        if route.auth_strategy != "api2_bearer_query_v1" and model.auth_query_parameters:
            raise ProfileError(f"non-API2 route carries query auth parameters: {model.model_profile_id}")
        if route.auth_strategy == "api3_bearer_session_v1" and not model.api3_strategy_type:
            raise ProfileError(f"API3 route lacks StrategyType: {model.model_profile_id}")
        if route.auth_strategy != "api3_bearer_session_v1" and model.api3_strategy_type:
            raise ProfileError(f"non-API3 route carries StrategyType: {model.model_profile_id}")


def _validate_pi_compatibility(
    value: Mapping[str, Any], *, api_protocol: str
) -> None:
    completion_boolean_fields = {
        "supportsStore",
        "supportsDeveloperRole",
        "supportsReasoningEffort",
        "supportsUsageInStreaming",
        "supportsFinishReason",
        "requiresToolResultName",
        "requiresAssistantAfterToolResult",
        "requiresThinkingAsText",
        "requiresReasoningContentOnAssistantMessages",
        "sendSessionAffinityHeaders",
        "supportsStrictMode",
        "supportsOpenAIGrammarTools",
        "supportsLongCacheRetention",
    }
    completion_string_options = {
        "maxTokensField": {"max_tokens", "max_completion_tokens"},
        "sessionAffinityFormat": {"openai", "openai-nosession", "openrouter"},
        "thinkingFormat": {
            "openai",
            "openrouter",
            "deepseek",
            "together",
            "baseten",
            "zai",
            "qwen",
            "chat-template",
            "qwen-chat-template",
            "string-thinking",
            "ant-ling",
        },
        "deferredToolsMode": {"kimi"},
    }
    responses_boolean_fields = {
        "supportsDeveloperRole",
        "supportsLongCacheRetention",
        "supportsStrictMode",
        "supportsOpenAIGrammarTools",
        "supportsAdditionalTools",
        "supportsToolSearch",
        "supportsMaxOutputTokens",
    }
    responses_string_options = {
        "sessionAffinityFormat": {"openai", "openai-nosession", "openrouter"},
    }
    if api_protocol == "openai-completions":
        boolean_fields = completion_boolean_fields
        string_options = completion_string_options
    elif api_protocol == "openai-responses":
        boolean_fields = responses_boolean_fields
        string_options = responses_string_options
    else:  # guarded by the caller; retain a fail-closed branch.
        raise ProfileError(f"unsupported Pi API protocol: {api_protocol}")
    allowed = boolean_fields | set(string_options)
    unknown = set(value) - allowed
    if unknown:
        raise ProfileError(f"unsupported Pi compatibility fields: {sorted(unknown)}")
    for name, child in value.items():
        if name in boolean_fields and not isinstance(child, bool):
            raise ProfileError(f"Pi compatibility {name} must be boolean")
        if name in string_options and child not in string_options[name]:
            raise ProfileError(f"Pi compatibility {name} has an invalid value")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProfileError(f"required JSON is missing or linked: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ProfileError(f"cannot read JSON {path.name}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ProfileError(f"JSON root must be an object: {path.name}")
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _real_input_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise ProfileError(f"{label} must be a real file")
    return path.resolve(strict=True)


def _keys(
    value: Mapping[str, Any], *, required: set[str], label: str
) -> None:
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise ProfileError(f"{label} keys differ; missing={missing}, extra={extra}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileError(f"{label} must be an object")
    return dict(value)


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProfileError(f"{label} must be an array")
    return value


def _string_tuple(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    items = _array(value, label)
    if not allow_empty and not items:
        raise ProfileError(f"{label} must not be empty")
    if any(not isinstance(item, str) or not item for item in items):
        raise ProfileError(f"{label} must contain non-empty strings")
    if len(items) != len(set(items)):
        raise ProfileError(f"{label} must not contain duplicates")
    return tuple(items)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{label} must be a non-empty string")
    return value


def _portable_id(value: Any, label: str) -> str:
    text = _text(value, label)
    if PORTABLE_ID.fullmatch(text) is None:
        raise ProfileError(f"{label} must be a portable identifier")
    return text


def _model_id(value: Any, label: str) -> str:
    text = _text(value, label)
    if MODEL_ID.fullmatch(text) is None:
        raise ProfileError(f"{label} must be a portable model identity")
    return text


def _http_path(value: Any, label: str) -> str:
    text = _text(value, label)
    if not text.startswith("/") or "?" in text or ".." in text or "#" in text:
        raise ProfileError(f"{label} must be an absolute, query-free HTTP path")
    return text


def _upstream_base_url(value: Any) -> str:
    text = _text(value, "upstream_base_url").rstrip("/")
    parsed = urlsplit(text)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ProfileError("upstream base URL contains an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ProfileError("upstream base URL must be uncredentialed HTTP(S)")
    return text


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ProfileError(f"{label} must be boolean")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProfileError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfileError(f"{label} must be a non-negative integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ProfileError(f"{label} must be a positive number")
    return float(value)


def _nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ProfileError(f"{label} must be a non-negative number")
    return float(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ApiFamilyProfile",
    "ExperimentProfile",
    "ModelProfile",
    "PiModelProfile",
    "ProfileError",
    "ProfileRegistry",
    "RETRYABLE_HTTP_DEFAULT",
    "ReasoningProfile",
    "RetryPolicy",
    "RouteProfile",
    "RouteRuntimeBinding",
    "RuntimeBindings",
    "load_experiment",
    "load_runtime_bindings",
    "normalize_runtime_credential",
]
