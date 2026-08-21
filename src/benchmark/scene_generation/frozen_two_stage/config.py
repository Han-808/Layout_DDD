"""Allowlisted JSON configuration for config-only model onboarding.

See ``docs/generation_transport_compatibility.md``.  This loader accepts only
the codec/gateway combinations already implemented by the compatibility layer;
it is intentionally not a request-template language and it rejects credential
or raw-prompt fields recursively.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from benchmark.scene_generation.frozen_two_stage.providers.base import ProviderRoute
from benchmark.scene_generation.frozen_two_stage.providers.codecs.openai_chat import (
    ChatOptionPolicy,
)
from benchmark.scene_generation.frozen_two_stage.providers.routes import (
    make_api2_chat_route,
    make_api2_responses_route,
    make_api3_chat_route,
)
from benchmark.scene_generation.frozen_two_stage.retry_policy import RetryPolicy


RUN_CONFIG_SCHEMA_VERSION = "frozen_two_stage_run_config_v1"
_MAX_CONFIG_BYTES = 1_000_000
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_CREDENTIAL_KEYS = frozenset(
    {
        "access_token",
        "api_key_env",
        "auth",
        "api_key",
        "api_key_env",
        "apikey",
        "app_key",
        "auth",
        "authentication",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credential_env",
        "credential_env",
        "credentials",
        "password",
        "private_key",
        "proxy_secret",
        "refresh_token",
        "secret",
        "secret_env",
        "session_token",
        "token",
        "token",
    }
)
_RAW_TEMPLATE_KEYS = frozenset(
    {
        "headers",
        "messages",
        "prompt",
        "prompt_text",
        "raw_model_content",
        "reasoning_content",
        "request_body",
        "request_template",
        "response_body",
        "stage_a_prompt",
        "stage_c_prompt",
        "system_prompt",
        "user_prompt",
    }
)
_SUMMARY_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "model_key",
        "model_label",
        "requested_briefs",
        "processed_briefs",
        "complete",
        "failed",
        "eligible",
        "stopped_early",
        "results",
        "completed_at",
    }
)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _forbidden_key(key: str) -> bool:
    normalized = _normalize_key(key)
    if normalized in _CREDENTIAL_KEYS or normalized in _RAW_TEMPLATE_KEYS:
        return True
    return any(
        normalized.endswith(f"_{suffix}")
        for suffix in (
            "access_token",
            "api_key",
            "app_key",
            "credential",
            "password",
            "refresh_token",
            "secret",
            "session_token",
            "token",
        )
    )


def _freeze_public_json(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if _forbidden_key(key):
                raise ValueError(f"{path} contains forbidden field: {key}")
            frozen[key] = _freeze_public_json(child, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_public_json(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) > _MAX_CONFIG_BYTES:
        raise ValueError(f"run config exceeds {_MAX_CONFIG_BYTES} bytes")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run-config JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("run config must be a JSON object")
    _freeze_public_json(value, path="run_config")
    return value


def _object(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    field_name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{field_name} missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {sorted(unknown)}")


def _string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    return value


def _slug(value: Any, *, field_name: str) -> str:
    text = _string(value, field_name=field_name)
    if not _SLUG_RE.fullmatch(text):
        raise ValueError(f"{field_name} must contain only safe slug characters")
    return text


def _header_fragment(value: Any, *, field_name: str) -> str:
    text = _string(value, field_name=field_name)
    if len(text) > 160 or any(character in text for character in "\r\n"):
        raise ValueError(f"{field_name} is not a safe header fragment")
    return text


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _resolve_path(config_dir: Path, value: Any, *, field_name: str) -> Path:
    text = _string(value, field_name=field_name)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class RouteConfig:
    """One allowlisted provider-route composition."""

    kind: str
    key: str
    runner_version: str
    provider: str | None = None
    gateway_model: str | None = None
    user_agent_suffix: str | None = None
    timeout_seconds: int | None = None
    chat_option_style: str | None = None
    reasoning_effort: str | None = None

    def build_route(self) -> ProviderRoute:
        """Compose existing codec/gateway classes; never infer from model name."""

        if self.kind == "api2_chat":
            return make_api2_chat_route(
                provider=self._required(self.provider, "provider"),
                gateway_model=self._required(self.gateway_model, "gateway_model"),
                user_agent_suffix=self._required(
                    self.user_agent_suffix, "user_agent_suffix"
                ),
                option_policy=ChatOptionPolicy.top_level_reasoning(
                    default_reasoning_effort=self._required(
                        self.reasoning_effort, "reasoning_effort"
                    )
                ),
                route_key=self.key,
                runner_version=self.runner_version,
                timeout_seconds=self.timeout_seconds or 600,
            )
        if self.kind == "api2_responses":
            return make_api2_responses_route(
                provider=self._required(self.provider, "provider"),
                gateway_model=self._required(self.gateway_model, "gateway_model"),
                user_agent_suffix=self._required(
                    self.user_agent_suffix, "user_agent_suffix"
                ),
                default_reasoning_effort=self._required(
                    self.reasoning_effort, "reasoning_effort"
                ),
                route_key=self.key,
                runner_version=self.runner_version,
                timeout_seconds=self.timeout_seconds or 600,
            )
        if self.kind == "api3_chat":
            if self.chat_option_style == "legacy_core":
                option_policy = ChatOptionPolicy.legacy_core()
            elif self.chat_option_style == "adaptive_thinking":
                option_policy = ChatOptionPolicy.adaptive_thinking(
                    reasoning_effort=self._required(
                        self.reasoning_effort, "reasoning_effort"
                    )
                )
            else:  # guarded by the loader
                raise ValueError(
                    f"unsupported API3 Chat option style: {self.chat_option_style!r}"
                )
            return make_api3_chat_route(
                option_policy=option_policy,
                route_key=self.key,
                runner_version=self.runner_version,
            )
        raise ValueError(f"unsupported route kind: {self.kind!r}")

    @staticmethod
    def _required(value: str | None, field_name: str) -> str:
        if value is None:
            raise ValueError(f"route {field_name} is required")
        return value

    def to_public_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "kind": self.kind,
                "key": self.key,
                "runner_version": self.runner_version,
                "provider": self.provider,
                "gateway_model": self.gateway_model,
                "user_agent_suffix": self.user_agent_suffix,
                "timeout_seconds": self.timeout_seconds,
                "chat_option_style": self.chat_option_style,
                "reasoning_effort": self.reasoning_effort,
            }.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Declarative retry classifications; counts/delay come from model config."""

    retryable_transport_statuses: tuple[str, ...]
    retryable_http_statuses: tuple[int, ...]
    retry_ambiguous_timeouts: bool
    continue_after_case_failure: bool

    def build_policy(
        self,
        *,
        max_infrastructure_retries: int,
        retry_delay_seconds: float,
    ) -> RetryPolicy:
        return RetryPolicy(
            max_infrastructure_retries=max_infrastructure_retries,
            retryable_transport_statuses=frozenset(
                self.retryable_transport_statuses
            ),
            retryable_http_statuses=frozenset(self.retryable_http_statuses),
            retry_delay_seconds=retry_delay_seconds,
            retry_ambiguous_timeouts=self.retry_ambiguous_timeouts,
            continue_after_case_failure=self.continue_after_case_failure,
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "retryable_transport_statuses": list(
                self.retryable_transport_statuses
            ),
            "retryable_http_statuses": list(self.retryable_http_statuses),
            "retry_ambiguous_timeouts": self.retry_ambiguous_timeouts,
            "continue_after_case_failure": self.continue_after_case_failure,
        }


@dataclass(frozen=True, slots=True)
class FrozenTwoStageRunConfig:
    """Validated, credential-free configuration for the generic runner."""

    path: Path
    sha256: str
    core_root: Path
    briefs_path: Path
    models_path: Path
    retriever_root: Path
    model_key: str
    ordered_brief_ids: tuple[str, ...]
    route: RouteConfig
    retry: RetryConfig
    execution_policy: Mapping[str, Any]
    summary_schema_version: str
    summary_extra: Mapping[str, Any]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_CONFIG_SCHEMA_VERSION,
            "path": str(self.path),
            "sha256": self.sha256,
            "core_root": str(self.core_root),
            "briefs_path": str(self.briefs_path),
            "models_path": str(self.models_path),
            "retriever_root": str(self.retriever_root),
            "model_key": self.model_key,
            "ordered_brief_ids": list(self.ordered_brief_ids),
            "route": self.route.to_public_dict(),
            "retry": self.retry.to_public_dict(),
            "execution_policy": _thaw(self.execution_policy),
            "summary": {
                "schema_version": self.summary_schema_version,
                "extra": _thaw(self.summary_extra),
            },
        }


def _parse_chat_options(value: Any, *, route_kind: str) -> tuple[str, str | None]:
    options = _object(value, field_name="route.chat_options")
    style = _string(options.get("style"), field_name="route.chat_options.style")
    if style == "legacy_core":
        _keys(
            options,
            field_name="route.chat_options",
            required=frozenset({"style"}),
        )
        if route_kind != "api3_chat":
            raise ValueError("legacy_core Chat options are allowlisted only for API3")
        return style, None
    if style == "top_level_reasoning":
        _keys(
            options,
            field_name="route.chat_options",
            required=frozenset({"style", "default_reasoning_effort"}),
        )
        if route_kind != "api2_chat":
            raise ValueError(
                "top_level_reasoning Chat options are allowlisted only for API2"
            )
        effort = _slug(
            options["default_reasoning_effort"],
            field_name="route.chat_options.default_reasoning_effort",
        )
        return style, effort
    if style == "adaptive_thinking":
        _keys(
            options,
            field_name="route.chat_options",
            required=frozenset({"style", "reasoning_effort"}),
        )
        if route_kind != "api3_chat":
            raise ValueError(
                "adaptive_thinking Chat options are allowlisted only for API3"
            )
        effort = _slug(
            options["reasoning_effort"],
            field_name="route.chat_options.reasoning_effort",
        )
        return style, effort
    raise ValueError(f"unsupported route.chat_options.style: {style!r}")


def _parse_route(value: Any) -> RouteConfig:
    route = _object(value, field_name="route")
    kind = _string(route.get("kind"), field_name="route.kind")
    common_required = frozenset({"kind", "key"})
    common_optional = frozenset({"runner_version"})
    runner_version = _slug(
        route.get("runner_version", "2.0.0"), field_name="route.runner_version"
    )
    key = _slug(route.get("key"), field_name="route.key")
    if kind in {"api2_chat", "api2_responses"}:
        required = common_required | frozenset(
            {"provider", "gateway_model", "user_agent_suffix"}
        )
        optional = common_optional | frozenset({"timeout_seconds"})
        if kind == "api2_chat":
            required |= frozenset({"chat_options"})
        else:
            required |= frozenset({"default_reasoning_effort"})
        _keys(route, field_name="route", required=required, optional=optional)
        style: str | None = None
        effort: str | None
        if kind == "api2_chat":
            style, effort = _parse_chat_options(route["chat_options"], route_kind=kind)
        else:
            effort = _slug(
                route["default_reasoning_effort"],
                field_name="route.default_reasoning_effort",
            )
        return RouteConfig(
            kind=kind,
            key=key,
            runner_version=runner_version,
            provider=_slug(route["provider"], field_name="route.provider"),
            gateway_model=_slug(
                route["gateway_model"], field_name="route.gateway_model"
            ),
            user_agent_suffix=_header_fragment(
                route["user_agent_suffix"], field_name="route.user_agent_suffix"
            ),
            timeout_seconds=_positive_int(
                route.get("timeout_seconds", 600),
                field_name="route.timeout_seconds",
            ),
            chat_option_style=style,
            reasoning_effort=effort,
        )
    if kind == "api3_chat":
        _keys(
            route,
            field_name="route",
            required=common_required | frozenset({"chat_options"}),
            optional=common_optional,
        )
        style, effort = _parse_chat_options(route["chat_options"], route_kind=kind)
        return RouteConfig(
            kind=kind,
            key=key,
            runner_version=runner_version,
            chat_option_style=style,
            reasoning_effort=effort,
        )
    raise ValueError(f"unsupported route.kind: {kind!r}")


def _parse_retry(value: Any) -> RetryConfig:
    retry = _object(value, field_name="retry")
    _keys(
        retry,
        field_name="retry",
        required=frozenset(
            {
                "retryable_transport_statuses",
                "retryable_http_statuses",
                "retry_ambiguous_timeouts",
                "continue_after_case_failure",
            }
        ),
    )
    transports_value = retry["retryable_transport_statuses"]
    if not isinstance(transports_value, list) or not transports_value:
        raise ValueError("retryable_transport_statuses must be a non-empty array")
    transports = tuple(
        _slug(item, field_name="retryable_transport_statuses item")
        for item in transports_value
    )
    if len(transports) != len(set(transports)):
        raise ValueError("retryable_transport_statuses contains duplicates")
    http_value = retry["retryable_http_statuses"]
    if not isinstance(http_value, list):
        raise ValueError("retryable_http_statuses must be an array")
    statuses = tuple(http_value)
    if len(statuses) != len(set(statuses)):
        raise ValueError("retryable_http_statuses contains duplicates")
    for status in statuses:
        if isinstance(status, bool) or not isinstance(status, int):
            raise ValueError("retryable_http_statuses must contain integers")
    ambiguous = retry["retry_ambiguous_timeouts"]
    continue_after = retry["continue_after_case_failure"]
    if not isinstance(ambiguous, bool) or not isinstance(continue_after, bool):
        raise ValueError("retry boolean fields must be booleans")
    # Validate classification consistency now; count/delay are derived later.
    policy = RetryPolicy(
        max_infrastructure_retries=0,
        retryable_transport_statuses=frozenset(transports),
        retryable_http_statuses=frozenset(statuses),
        retry_delay_seconds=0.0,
        retry_ambiguous_timeouts=ambiguous,
        continue_after_case_failure=continue_after,
    )
    return RetryConfig(
        retryable_transport_statuses=tuple(
            sorted(policy.retryable_transport_statuses)
        ),
        retryable_http_statuses=tuple(sorted(policy.retryable_http_statuses)),
        retry_ambiguous_timeouts=ambiguous,
        continue_after_case_failure=continue_after,
    )


def load_run_config(path: str | Path) -> FrozenTwoStageRunConfig:
    """Load one strict, credential-free run config with relative path resolution."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"run config does not exist: {config_path}")
    value = _load_json_object(config_path)
    _keys(
        value,
        field_name="run_config",
        required=frozenset(
            {
                "schema_version",
                "core_root",
                "model_key",
                "ordered_brief_ids",
                "route",
                "retry",
                "execution_policy",
            }
        ),
        optional=frozenset(
            {"briefs_path", "models_path", "retriever_root", "summary"}
        ),
    )
    if value["schema_version"] != RUN_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported run-config schema: {value['schema_version']!r}"
        )
    config_dir = config_path.parent
    core_root = _resolve_path(
        config_dir, value["core_root"], field_name="core_root"
    )
    briefs_path = _resolve_path(
        config_dir,
        value.get("briefs_path", str(core_root / "briefs.json")),
        field_name="briefs_path",
    )
    models_path = _resolve_path(
        config_dir,
        value.get("models_path", str(core_root / "models.pod.json")),
        field_name="models_path",
    )
    retriever_root = _resolve_path(
        config_dir,
        value.get("retriever_root", str(core_root)),
        field_name="retriever_root",
    )
    if not core_root.is_dir():
        raise FileNotFoundError(f"core_root is not a directory: {core_root}")
    for field_name, candidate in (
        ("briefs_path", briefs_path),
        ("models_path", models_path),
    ):
        if not candidate.is_file():
            raise FileNotFoundError(f"{field_name} is not a file: {candidate}")
    if not retriever_root.is_dir():
        raise FileNotFoundError(
            f"retriever_root is not a directory: {retriever_root}"
        )
    if retriever_root != core_root:
        raise ValueError(
            "retriever_root must equal core_root for the frozen two-stage "
            "workflow; alternate retriever implementations require a separately "
            "versioned workflow"
        )

    brief_ids_value = value["ordered_brief_ids"]
    if not isinstance(brief_ids_value, list) or not brief_ids_value:
        raise ValueError("ordered_brief_ids must be a non-empty array")
    brief_ids = tuple(
        _slug(item, field_name="ordered_brief_ids item")
        for item in brief_ids_value
    )
    if len(brief_ids) != len(set(brief_ids)):
        raise ValueError("ordered_brief_ids contains duplicates")

    execution_policy_value = _object(
        value["execution_policy"], field_name="execution_policy"
    )
    execution_policy = _freeze_public_json(
        execution_policy_value, path="execution_policy"
    )
    execution_schema = execution_policy.get("schema_version")
    if not isinstance(execution_schema, str) or not execution_schema.strip():
        raise ValueError("execution_policy must contain a string schema_version")

    summary_value = _object(value.get("summary", {}), field_name="summary")
    _keys(
        summary_value,
        field_name="summary",
        required=frozenset(),
        optional=frozenset({"schema_version", "extra"}),
    )
    summary_schema = _slug(
        summary_value.get("schema_version", "hy34_two_stage_run_summary_v2"),
        field_name="summary.schema_version",
    )
    summary_extra_value = _object(
        summary_value.get("extra", {}), field_name="summary.extra"
    )
    summary_extra = _freeze_public_json(summary_extra_value, path="summary.extra")
    summary_conflicts = _SUMMARY_RESERVED_KEYS.intersection(summary_extra)
    if summary_conflicts:
        raise ValueError(
            "summary.extra must not replace canonical fields: "
            f"{sorted(summary_conflicts)}"
        )

    return FrozenTwoStageRunConfig(
        path=config_path,
        sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        core_root=core_root,
        briefs_path=briefs_path,
        models_path=models_path,
        retriever_root=retriever_root,
        model_key=_slug(value["model_key"], field_name="model_key"),
        ordered_brief_ids=brief_ids,
        route=_parse_route(value["route"]),
        retry=_parse_retry(value["retry"]),
        execution_policy=execution_policy,
        summary_schema_version=summary_schema,
        summary_extra=summary_extra,
    )
