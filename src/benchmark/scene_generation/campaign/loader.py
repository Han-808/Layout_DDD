"""Strict JSON loaders for deployment-free generation campaign profiles."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn

from benchmark.scene_generation.campaign.profiles import (
    CAMPAIGN_REGISTRY_SCHEMA_VERSION,
    CONTRACT_REGISTRY_SCHEMA_VERSION,
    MODEL_REGISTRY_SCHEMA_VERSION,
    ROUTE_REGISTRY_SCHEMA_VERSION,
    ArtifactContract,
    BriefSetContract,
    CampaignProfile,
    CampaignProfileBundle,
    CampaignContractRegistry,
    CampaignProfileRegistry,
    ExecutionPolicyContract,
    GatewayOptions,
    ModelProfile,
    ModelProfileRegistry,
    PreflightContract,
    RequestOptions,
    RouteProfile,
    RouteProfileRegistry,
    TransportPolicy,
    WorkflowContract,
)


_MAX_JSON_BYTES = 1_000_000
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]")
_URI = re.compile(r"(?i)(?:^|\s)[a-z][a-z0-9+.-]*://")
_HOST_PORT = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9_-])(?:localhost|[a-z0-9.-]+|\[[0-9a-f:]+\]):[0-9]{2,5}(?:$|[^0-9])"
)
_FORBIDDEN_EXACT_FIELDS = frozenset(
    {
        "access_token",
        "api_base",
        "api_key",
        "api_key_env",
        "apikey",
        "auth",
        "authorization",
        "base_url",
        "bearer_token",
        "client_secret",
        "credential",
        "credential_env",
        "credentials",
        "directory",
        "endpoint",
        "endpoint_env",
        "index_stem",
        "model_path",
        "password",
        "path",
        "private_key",
        "proxy_secret",
        "refresh_token",
        "root",
        "secret",
        "secret_env",
        "session_token",
        "token",
        "url",
    }
)


def _normalized_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _forbidden_field(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized in _FORBIDDEN_EXACT_FIELDS:
        return True
    return any(
        normalized.endswith(suffix)
        for suffix in (
            "_api_key",
            "_credential",
            "_env",
            "_password",
            "_path",
            "_secret",
            "_token",
            "_url",
        )
    )


def _looks_like_local_binding(value: str) -> bool:
    lowered = value.lower()
    return (
        value.startswith(("/", "~/", "./", "../", "\\\\", "//"))
        or value.startswith("$")
        or bool(_WINDOWS_ABSOLUTE_PATH.search(value))
        or bool(_URI.search(value))
        or bool(_HOST_PORT.search(value))
        or "/users/" in lowered
        or "/home/" in lowered
        or "\\users\\" in lowered
        or "${" in value
    )


def _scan_public_json(value: Any, *, location: str) -> None:
    """Reject deployment, secret, and local-path material before parsing."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} keys must be strings")
            if _forbidden_field(key):
                raise ValueError(f"{location} contains forbidden field: {key}")
            _scan_public_json(child, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_public_json(child, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and _looks_like_local_binding(value):
        raise ValueError(
            f"{location} contains an endpoint, environment expansion, or local path"
        )


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _load_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    data = candidate.read_bytes()
    if len(data) > _MAX_JSON_BYTES:
        raise ValueError(f"profile registry exceeds {_MAX_JSON_BYTES} bytes")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid profile registry JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("profile registry must be a JSON object")
    _scan_public_json(value, location="registry")
    return value


def _object(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    field_name: str,
    required: frozenset[str],
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required
    if missing:
        raise ValueError(f"{field_name} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {sorted(unknown)}")


def _string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    return value


def _nullable_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name=field_name)


def _integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be finite")
    return result


def _nullable_number(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _number(value, field_name=field_name)


def _nullable_integer(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field_name=field_name)


def _nullable_boolean(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean or null")
    return value


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _array(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    return value


def load_route_profile_registry(path: str | Path) -> RouteProfileRegistry:
    value = _load_json(path)
    _exact_keys(
        value,
        field_name="route registry",
        required=frozenset({"schema_version", "routes"}),
    )
    if value["schema_version"] != ROUTE_REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"unsupported route registry schema: {value['schema_version']!r}")
    routes: list[RouteProfile] = []
    required = frozenset(
        {
            "route_profile_id",
            "codec_id",
            "gateway_id",
            "option_contract_id",
            "response_contract_id",
            "runner_version",
        }
    )
    for index, raw in enumerate(_array(value["routes"], field_name="routes")):
        item = _object(raw, field_name=f"routes[{index}]")
        _exact_keys(item, field_name=f"routes[{index}]", required=required)
        routes.append(
            RouteProfile(
                route_profile_id=_string(
                    item["route_profile_id"],
                    field_name=f"routes[{index}].route_profile_id",
                ),
                codec_id=_string(item["codec_id"], field_name="codec_id"),
                gateway_id=_string(item["gateway_id"], field_name="gateway_id"),
                option_contract_id=_string(
                    item["option_contract_id"], field_name="option_contract_id"
                ),
                response_contract_id=_string(
                    item["response_contract_id"], field_name="response_contract_id"
                ),
                runner_version=_string(
                    item["runner_version"], field_name="runner_version"
                ),
            )
        )
    return RouteProfileRegistry(tuple(routes))


def _gateway_options(
    value: Any,
    *,
    field_name: str,
    route: RouteProfile,
) -> GatewayOptions:
    item = _object(value, field_name=field_name)
    if route.gateway_id == "api2_bearer_query_v1":
        _exact_keys(
            item,
            field_name=field_name,
            required=frozenset(
                {
                    "provider",
                    "gateway_model",
                    "user_agent_suffix",
                    "auth_timeout_seconds",
                    "strategy_type",
                }
            ),
        )
        return GatewayOptions(
            provider=_string(item["provider"], field_name=f"{field_name}.provider"),
            gateway_model=_string(
                item["gateway_model"], field_name=f"{field_name}.gateway_model"
            ),
            user_agent_suffix=_string(
                item["user_agent_suffix"],
                field_name=f"{field_name}.user_agent_suffix",
            ),
            auth_timeout_seconds=_integer(
                item["auth_timeout_seconds"],
                field_name=f"{field_name}.auth_timeout_seconds",
            ),
            strategy_type=_string(
                item["strategy_type"], field_name=f"{field_name}.strategy_type"
            ),
        )
    if route.gateway_id == "api3_bearer_session_v1":
        _exact_keys(
            item,
            field_name=field_name,
            required=frozenset({"strategy_type"}),
        )
        return GatewayOptions(
            strategy_type=_string(
                item["strategy_type"], field_name=f"{field_name}.strategy_type"
            )
        )
    raise ValueError(f"unsupported gateway contract: {route.gateway_id!r}")


def _request_options(value: Any, *, field_name: str) -> RequestOptions:
    item = _object(value, field_name=field_name)
    required = frozenset(
        {
            "request_timeout_seconds",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "reasoning_effort",
            "thinking_type",
            "preserved_thinking",
        }
    )
    _exact_keys(item, field_name=field_name, required=required)
    return RequestOptions(
        request_timeout_seconds=_number(
            item["request_timeout_seconds"],
            field_name=f"{field_name}.request_timeout_seconds",
        ),
        max_tokens=_integer(
            item["max_tokens"], field_name=f"{field_name}.max_tokens"
        ),
        temperature=_nullable_number(
            item["temperature"], field_name=f"{field_name}.temperature"
        ),
        top_p=_nullable_number(item["top_p"], field_name=f"{field_name}.top_p"),
        top_k=_nullable_integer(item["top_k"], field_name=f"{field_name}.top_k"),
        repetition_penalty=_nullable_number(
            item["repetition_penalty"],
            field_name=f"{field_name}.repetition_penalty",
        ),
        reasoning_effort=_nullable_string(
            item["reasoning_effort"],
            field_name=f"{field_name}.reasoning_effort",
        ),
        thinking_type=_nullable_string(
            item["thinking_type"], field_name=f"{field_name}.thinking_type"
        ),
        preserved_thinking=_nullable_boolean(
            item["preserved_thinking"],
            field_name=f"{field_name}.preserved_thinking",
        ),
    )


def _transport_policy(value: Any, *, field_name: str) -> TransportPolicy:
    item = _object(value, field_name=field_name)
    _exact_keys(
        item,
        field_name=field_name,
        required=frozenset(
            {"max_infrastructure_retries", "retry_delay_seconds"}
        ),
    )
    return TransportPolicy(
        max_infrastructure_retries=_integer(
            item["max_infrastructure_retries"],
            field_name=f"{field_name}.max_infrastructure_retries",
        ),
        retry_delay_seconds=_number(
            item["retry_delay_seconds"],
            field_name=f"{field_name}.retry_delay_seconds",
        ),
    )


def load_model_profile_registry(
    path: str | Path,
    routes: RouteProfileRegistry,
) -> ModelProfileRegistry:
    value = _load_json(path)
    _exact_keys(
        value,
        field_name="model registry",
        required=frozenset({"schema_version", "models"}),
    )
    if value["schema_version"] != MODEL_REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"unsupported model registry schema: {value['schema_version']!r}")
    required = frozenset(
        {
            "model_profile_id",
            "display_label",
            "configured_model",
            "wire_model",
            "route_profile_id",
            "gateway_options",
            "request_options",
            "transport_policy",
            "preflight_contract_id",
            "response_model_identity",
        }
    )
    models: list[ModelProfile] = []
    route_by_id = routes.by_id
    for index, raw in enumerate(_array(value["models"], field_name="models")):
        item = _object(raw, field_name=f"models[{index}]")
        _exact_keys(item, field_name=f"models[{index}]", required=required)
        route_profile_id = _string(
            item["route_profile_id"], field_name=f"models[{index}].route_profile_id"
        )
        try:
            route = route_by_id[route_profile_id]
        except KeyError as exc:
            raise ValueError(
                f"models[{index}] references unknown route {route_profile_id!r}"
            ) from exc
        model = ModelProfile(
            model_profile_id=_string(
                item["model_profile_id"],
                field_name=f"models[{index}].model_profile_id",
            ),
            display_label=_string(
                item["display_label"], field_name=f"models[{index}].display_label"
            ),
            configured_model=_string(
                item["configured_model"],
                field_name=f"models[{index}].configured_model",
            ),
            wire_model=_string(
                item["wire_model"], field_name=f"models[{index}].wire_model"
            ),
            route_profile_id=route_profile_id,
            gateway_options=_gateway_options(
                item["gateway_options"],
                field_name=f"models[{index}].gateway_options",
                route=route,
            ),
            request_options=_request_options(
                item["request_options"],
                field_name=f"models[{index}].request_options",
            ),
            transport_policy=_transport_policy(
                item["transport_policy"],
                field_name=f"models[{index}].transport_policy",
            ),
            preflight_contract_id=_string(
                item["preflight_contract_id"],
                field_name=f"models[{index}].preflight_contract_id",
            ),
            response_model_identity=_nullable_string(
                item["response_model_identity"],
                field_name=f"models[{index}].response_model_identity",
            ),
        )
        model.validate_for(route)
        models.append(model)
    return ModelProfileRegistry(tuple(models))


def load_campaign_profile_registry(
    path: str | Path,
    models: ModelProfileRegistry,
) -> CampaignProfileRegistry:
    value = _load_json(path)
    _exact_keys(
        value,
        field_name="campaign registry",
        required=frozenset({"schema_version", "campaigns"}),
    )
    if value["schema_version"] != CAMPAIGN_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported campaign registry schema: {value['schema_version']!r}"
        )
    required = frozenset(
        {
            "campaign_id",
            "workflow_profile_id",
            "model_profile_id",
            "retrieval_profile_id",
            "brief_set_id",
            "ordered_brief_ids",
            "execution_policy_id",
            "artifact_contract_id",
        }
    )
    campaigns: list[CampaignProfile] = []
    model_ids = models.by_id
    for index, raw in enumerate(_array(value["campaigns"], field_name="campaigns")):
        item = _object(raw, field_name=f"campaigns[{index}]")
        _exact_keys(item, field_name=f"campaigns[{index}]", required=required)
        # Campaigns intentionally have no nested retrieval declaration.  This
        # explicit guard gives a clear error if such fields are added later.
        forbidden_retrieval_fields = {
            "dataset",
            "dataset_id",
            "encoder",
            "encoder_id",
            "index",
            "index_id",
            "resource_binding",
        }.intersection(item)
        if forbidden_retrieval_fields:
            raise ValueError(
                "campaign must reference only retrieval_profile_id; forbidden "
                f"fields: {sorted(forbidden_retrieval_fields)}"
            )
        model_profile_id = _string(
            item["model_profile_id"],
            field_name=f"campaigns[{index}].model_profile_id",
        )
        if model_profile_id not in model_ids:
            raise ValueError(
                f"campaigns[{index}] references unknown model {model_profile_id!r}"
            )
        brief_values = _array(
            item["ordered_brief_ids"],
            field_name=f"campaigns[{index}].ordered_brief_ids",
        )
        campaigns.append(
            CampaignProfile(
                campaign_id=_string(
                    item["campaign_id"], field_name=f"campaigns[{index}].campaign_id"
                ),
                workflow_profile_id=_string(
                    item["workflow_profile_id"],
                    field_name=f"campaigns[{index}].workflow_profile_id",
                ),
                model_profile_id=model_profile_id,
                retrieval_profile_id=_string(
                    item["retrieval_profile_id"],
                    field_name=f"campaigns[{index}].retrieval_profile_id",
                ),
                brief_set_id=_string(
                    item["brief_set_id"],
                    field_name=f"campaigns[{index}].brief_set_id",
                ),
                ordered_brief_ids=tuple(
                    _string(
                        brief_id,
                        field_name=f"campaigns[{index}].ordered_brief_ids item",
                    )
                    for brief_id in brief_values
                ),
                execution_policy_id=_string(
                    item["execution_policy_id"],
                    field_name=f"campaigns[{index}].execution_policy_id",
                ),
                artifact_contract_id=_string(
                    item["artifact_contract_id"],
                    field_name=f"campaigns[{index}].artifact_contract_id",
                ),
            )
        )
    return CampaignProfileRegistry(tuple(campaigns))


def load_campaign_contract_registry(path: str | Path) -> CampaignContractRegistry:
    value = _load_json(path)
    _exact_keys(
        value,
        field_name="contract registry",
        required=frozenset(
            {
                "schema_version",
                "workflows",
                "brief_sets",
                "execution_policies",
                "artifact_contracts",
                "preflight_contracts",
            }
        ),
    )
    if value["schema_version"] != CONTRACT_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported contract registry schema: {value['schema_version']!r}"
        )

    workflows: list[WorkflowContract] = []
    workflow_keys = frozenset(
        {
            "workflow_profile_id",
            "core_bundle_id",
            "runner_version",
            "runner_source_sha256",
            "stage_a_prompt_sha256",
            "stage_c_prompt_sha256",
        }
    )
    for index, raw in enumerate(_array(value["workflows"], field_name="workflows")):
        item = _object(raw, field_name=f"workflows[{index}]")
        _exact_keys(item, field_name=f"workflows[{index}]", required=workflow_keys)
        workflows.append(
            WorkflowContract(
                workflow_profile_id=_string(item["workflow_profile_id"], field_name="workflow_profile_id"),
                core_bundle_id=_string(item["core_bundle_id"], field_name="core_bundle_id"),
                runner_version=_string(item["runner_version"], field_name="runner_version"),
                runner_source_sha256=_string(item["runner_source_sha256"], field_name="runner_source_sha256"),
                stage_a_prompt_sha256=_string(item["stage_a_prompt_sha256"], field_name="stage_a_prompt_sha256"),
                stage_c_prompt_sha256=_string(item["stage_c_prompt_sha256"], field_name="stage_c_prompt_sha256"),
            )
        )

    brief_sets: list[BriefSetContract] = []
    brief_keys = frozenset({"brief_set_id", "core_bundle_id", "content_sha256", "ordered_brief_ids"})
    for index, raw in enumerate(_array(value["brief_sets"], field_name="brief_sets")):
        item = _object(raw, field_name=f"brief_sets[{index}]")
        _exact_keys(item, field_name=f"brief_sets[{index}]", required=brief_keys)
        brief_sets.append(
            BriefSetContract(
                brief_set_id=_string(item["brief_set_id"], field_name="brief_set_id"),
                core_bundle_id=_string(item["core_bundle_id"], field_name="core_bundle_id"),
                content_sha256=_string(item["content_sha256"], field_name="content_sha256"),
                ordered_brief_ids=tuple(
                    _string(child, field_name="ordered_brief_ids item")
                    for child in _array(item["ordered_brief_ids"], field_name="ordered_brief_ids")
                ),
            )
        )

    executions: list[ExecutionPolicyContract] = []
    execution_keys = frozenset(
        {
            "execution_policy_id",
            "max_infrastructure_retries",
            "retry_delay_seconds",
            "retryable_transport_statuses",
            "retryable_http_statuses",
            "retry_ambiguous_timeouts",
            "continue_after_case_failure",
        }
    )
    for index, raw in enumerate(
        _array(value["execution_policies"], field_name="execution_policies")
    ):
        item = _object(raw, field_name=f"execution_policies[{index}]")
        _exact_keys(item, field_name=f"execution_policies[{index}]", required=execution_keys)
        executions.append(
            ExecutionPolicyContract(
                execution_policy_id=_string(item["execution_policy_id"], field_name="execution_policy_id"),
                max_infrastructure_retries=_integer(item["max_infrastructure_retries"], field_name="max_infrastructure_retries"),
                retry_delay_seconds=_number(item["retry_delay_seconds"], field_name="retry_delay_seconds"),
                retryable_transport_statuses=tuple(
                    _string(child, field_name="retryable_transport_statuses item")
                    for child in _array(item["retryable_transport_statuses"], field_name="retryable_transport_statuses")
                ),
                retryable_http_statuses=tuple(
                    _integer(child, field_name="retryable_http_statuses item")
                    for child in _array(item["retryable_http_statuses"], field_name="retryable_http_statuses")
                ),
                retry_ambiguous_timeouts=_boolean(item["retry_ambiguous_timeouts"], field_name="retry_ambiguous_timeouts"),
                continue_after_case_failure=_boolean(item["continue_after_case_failure"], field_name="continue_after_case_failure"),
            )
        )

    artifacts: list[ArtifactContract] = []
    artifact_keys = frozenset(
        {"artifact_contract_id", "run_manifest_schema_version", "case_result_schema_version", "public_model_contract"}
    )
    for index, raw in enumerate(_array(value["artifact_contracts"], field_name="artifact_contracts")):
        item = _object(raw, field_name=f"artifact_contracts[{index}]")
        _exact_keys(item, field_name=f"artifact_contracts[{index}]", required=artifact_keys)
        artifacts.append(
            ArtifactContract(
                artifact_contract_id=_string(item["artifact_contract_id"], field_name="artifact_contract_id"),
                run_manifest_schema_version=_string(item["run_manifest_schema_version"], field_name="run_manifest_schema_version"),
                case_result_schema_version=_string(item["case_result_schema_version"], field_name="case_result_schema_version"),
                public_model_contract=_string(item["public_model_contract"], field_name="public_model_contract"),
            )
        )

    preflights: list[PreflightContract] = []
    preflight_keys = frozenset(
        {
            "preflight_contract_id",
            "response_contract_id",
            "content_validation",
            "require_model_identity",
            "require_reasoning_signal",
            "record_reasoning_diagnostic",
        }
    )
    for index, raw in enumerate(_array(value["preflight_contracts"], field_name="preflight_contracts")):
        item = _object(raw, field_name=f"preflight_contracts[{index}]")
        _exact_keys(item, field_name=f"preflight_contracts[{index}]", required=preflight_keys)
        preflights.append(
            PreflightContract(
                preflight_contract_id=_string(item["preflight_contract_id"], field_name="preflight_contract_id"),
                response_contract_id=_string(item["response_contract_id"], field_name="response_contract_id"),
                content_validation=_string(
                    item["content_validation"], field_name="content_validation"
                ),
                require_model_identity=_boolean(item["require_model_identity"], field_name="require_model_identity"),
                require_reasoning_signal=_boolean(item["require_reasoning_signal"], field_name="require_reasoning_signal"),
                record_reasoning_diagnostic=_boolean(item["record_reasoning_diagnostic"], field_name="record_reasoning_diagnostic"),
            )
        )
    return CampaignContractRegistry(
        workflows=tuple(workflows),
        brief_sets=tuple(brief_sets),
        execution_policies=tuple(executions),
        artifact_contracts=tuple(artifacts),
        preflight_contracts=tuple(preflights),
    )


def load_campaign_profile_bundle(root: str | Path) -> CampaignProfileBundle:
    """Load the canonical three registries from one directory."""

    directory = Path(root).expanduser().resolve()
    routes = load_route_profile_registry(directory / "route_profiles_v2.json")
    models = load_model_profile_registry(
        directory / "model_profiles_v2.json", routes
    )
    campaigns = load_campaign_profile_registry(
        directory / "campaigns_v2.json", models
    )
    contracts = load_campaign_contract_registry(directory / "contracts_v2.json")
    return CampaignProfileBundle(
        routes=routes, models=models, campaigns=campaigns, contracts=contracts
    )
