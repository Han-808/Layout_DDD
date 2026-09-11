from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from benchmark.grouping.interfaces import (
    GroupingAlgorithm,
    GroupingRequest,
    GroupingResult,
)
from benchmark.visual_judge.roles import DecisionContract, VLMRole


DEFAULT_GROUPING_FALLBACK_CONFIG: dict[str, Any] = {
    "enabled": True,
    "backend": "topology",
}
GROUPING_FALLBACK_BACKENDS = ("topology",)
GROUPING_FALLBACK_AUDIT_VERSION = "grouping_fallback_audit_v2"
_SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_REQUEST_METADATA_FIELDS = frozenset(
    {
        "api_key_env",
        "authorization_configured",
        "call_type",
        "content_chars",
        "endpoint",
        "finish_reason",
        "image_count",
        "max_completion_tokens",
        "max_tokens",
        "max_tokens_field",
        "message_count",
        "model",
        "prompt_budget_report",
        "prompt_budget_exceeded",
        "prompt_budget_warning",
        "prompt_chars",
        "response_format_json",
        "send_temperature",
        "temperature",
        "timeout_seconds",
        "url",
        "usage",
    }
)
_SAFE_NESTED_METADATA_FIELDS = frozenset(
    {
        "accepted_prediction_tokens",
        "audio_tokens",
        "cached_tokens",
        "call_type",
        "case_id",
        "chars",
        "compaction_level",
        "completion_tokens",
        "completion_tokens_details",
        "context_length",
        "estimated_prompt_tokens",
        "estimated_tokens",
        "estimated_total_tokens",
        "fits_context",
        "input_mode",
        "item_count",
        "iteration",
        "largest_sections",
        "max_tokens",
        "max_tokens_source",
        "name",
        "object_count",
        "omitted_count",
        "over_budget_tokens",
        "prompt_budget",
        "prompt_chars",
        "prompt_tokens",
        "prompt_tokens_details",
        "reasoning_tokens",
        "rejected_prediction_tokens",
        "safety_margin_tokens",
        "scene_id",
        "sections",
        "total_tokens",
        "warning",
    }
)


class VLMPrimaryGroupingAlgorithm:
    """Run VLM grouping first and use one explicit deterministic fallback.

    The wrapper is created only by the grouping factory. Passing an explicit
    ``algorithm=`` to :func:`group_scene` bypasses it, so custom algorithms do
    not acquire an implicit recovery policy.
    """

    backend = "vlm"

    def __init__(
        self,
        *,
        primary: GroupingAlgorithm | None,
        fallback: GroupingAlgorithm | None,
        primary_policy_id: str,
        fallback_config: dict[str, Any],
        primary_unavailable_error: BaseException | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.policy_id = str(primary_policy_id)
        self.fallback_config = deepcopy(fallback_config)
        self.primary_unavailable_error = primary_unavailable_error

    def group(self, request: GroupingRequest) -> GroupingResult:
        request = GroupingRequest.from_value(request)
        primary_error = self.primary_unavailable_error
        if primary_error is None:
            if self.primary is None:
                raise RuntimeError(
                    "VLM grouping primary is unavailable without a recorded "
                    "failure"
                )
            try:
                result = self.primary.group(request)
            except Exception as exc:
                primary_error = exc
            else:
                return _annotate_result(
                    result,
                    fallback_config=self.fallback_config,
                    primary_policy_id=self.policy_id,
                    primary_outcome="complete",
                    primary=self.primary,
                    fallback=self.fallback,
                    fallback_used=False,
                )

        if primary_error is None:
            raise RuntimeError(
                "VLM grouping did not complete but no primary failure was "
                "recorded"
            )
        if not self.fallback_config["enabled"]:
            raise primary_error
        if self.fallback is None:
            raise RuntimeError(
                "grouping fallback is enabled but no fallback algorithm is "
                "configured"
            ) from primary_error

        try:
            result = self.fallback.group(request)
        except Exception as fallback_error:
            raise GroupingFallbackError(
                primary_error=primary_error,
                fallback_error=fallback_error,
                fallback_backend=str(
                    self.fallback_config.get("backend") or "unknown"
                ),
            ) from fallback_error
        return _annotate_result(
            result,
            fallback_config=self.fallback_config,
            primary_policy_id=self.policy_id,
            primary_outcome="failed",
            primary_error=primary_error,
            primary=self.primary,
            fallback=self.fallback,
            fallback_used=True,
        )


class GroupingFallbackError(RuntimeError):
    """Both the VLM grouping primary and deterministic fallback failed."""

    def __init__(
        self,
        *,
        primary_error: BaseException,
        fallback_error: BaseException,
        fallback_backend: str,
    ) -> None:
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        self.fallback_backend = fallback_backend
        super().__init__(
            "VLM grouping failed and deterministic fallback "
            f"{fallback_backend!r} also failed: "
            f"{type(fallback_error).__name__}: {fallback_error}"
        )


def resolve_grouping_fallback_config(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve and strictly validate the small fallback policy."""

    if config is not None and not isinstance(config, dict):
        raise TypeError("grouping config must be a JSON object")
    root = config or {}
    section = root.get("grouping", root)
    if not isinstance(section, dict):
        raise TypeError("grouping config section must be a JSON object")
    raw = section.get("fallback")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("grouping fallback config must be a JSON object")
    unknown = sorted(set(raw) - {"enabled", "backend"})
    if unknown:
        raise ValueError(
            "grouping fallback config contains unknown fields: "
            f"{unknown}"
        )

    enabled = raw.get(
        "enabled",
        DEFAULT_GROUPING_FALLBACK_CONFIG["enabled"],
    )
    if not isinstance(enabled, bool):
        raise TypeError("grouping fallback enabled must be a boolean")
    backend = raw.get(
        "backend",
        DEFAULT_GROUPING_FALLBACK_CONFIG["backend"],
    )
    if not isinstance(backend, str) or not backend.strip():
        raise TypeError(
            "grouping fallback backend must be a non-empty string"
        )
    backend = backend.strip()
    if backend not in GROUPING_FALLBACK_BACKENDS:
        raise ValueError(
            "grouping fallback backend must be one of "
            f"{list(GROUPING_FALLBACK_BACKENDS)}"
        )
    return {
        "enabled": enabled,
        "backend": backend,
        "value_sources": {
            "enabled": (
                "config" if "enabled" in raw else "default"
            ),
            "backend": (
                "config" if "backend" in raw else "default"
            ),
        },
    }


def grouping_fallback_route(
    result: GroupingResult | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(result, GroupingResult):
        provenance = result.provenance
    elif isinstance(result, dict):
        provenance = result.get("provenance")
    else:
        return {}
    if not isinstance(provenance, dict):
        return {}
    route = provenance.get("grouping_fallback")
    return deepcopy(route) if isinstance(route, dict) else {}


def _annotate_result(
    result: GroupingResult,
    *,
    fallback_config: dict[str, Any],
    primary_policy_id: str,
    primary_outcome: str,
    primary: GroupingAlgorithm | None,
    fallback: GroupingAlgorithm | None,
    fallback_used: bool,
    primary_error: BaseException | None = None,
) -> GroupingResult:
    if not isinstance(result, GroupingResult):
        raise TypeError(
            "grouping algorithm must return a GroupingResult"
        )
    fallback_backend = str(
        fallback_config.get("backend") or "unknown"
    )
    fallback_policy_id = (
        str(getattr(fallback, "policy_id", "unknown"))
        if fallback is not None
        else None
    )
    route: dict[str, Any] = {
        "audit_version": GROUPING_FALLBACK_AUDIT_VERSION,
        "vlm_role": VLMRole.VLM_GROUPING.value,
        "decision_contract": DecisionContract.GROUPING_PARTITION.value,
        "primary_backend": "vlm",
        "primary_policy_id": primary_policy_id,
        "primary_outcome": primary_outcome,
        "fallback_enabled": bool(fallback_config["enabled"]),
        "fallback_used": fallback_used,
        "fallback_backend": fallback_backend,
        "fallback_policy_id": fallback_policy_id,
        "effective_values": {
            "enabled": bool(fallback_config["enabled"]),
            "backend": fallback_backend,
        },
        "value_sources": deepcopy(
            fallback_config.get("value_sources") or {}
        ),
    }
    route.update(_primary_model_audit(primary))
    if primary_error is not None:
        route["primary_failure"] = _error_record(primary_error)

    provenance = deepcopy(result.provenance)
    provenance["grouping_fallback"] = route
    resolved_config = deepcopy(result.resolved_grouping_config)
    resolved_config["fallback"] = {
        "enabled": bool(fallback_config["enabled"]),
        "backend": fallback_backend,
        "value_sources": deepcopy(
            fallback_config.get("value_sources") or {}
        ),
    }
    return replace(
        result,
        provenance=provenance,
        resolved_grouping_config=resolved_config,
    )


def _error_record(error: BaseException) -> dict[str, str]:
    message = str(error)
    if len(message) > 1000:
        message = message[:997] + "..."
    return {
        "error_type": type(error).__name__,
        "message": message,
    }


def _primary_model_audit(
    primary: GroupingAlgorithm | None,
) -> dict[str, Any]:
    """Return a bounded, credential-free audit snapshot of the VLM primary."""

    if primary is None:
        return {}
    model = getattr(primary, "model", None)
    if model is None:
        return {}

    audit: dict[str, Any] = {}
    model_id = _safe_short_string(getattr(model, "model_id", None))
    if model_id is not None:
        audit["model"] = model_id
    endpoint = _safe_http_url(getattr(model, "endpoint", None))
    if endpoint is not None:
        audit["endpoint"] = endpoint
    request_metadata = _safe_request_metadata(
        getattr(model, "last_request_metadata", None)
    )
    if request_metadata:
        audit["last_request_metadata"] = request_metadata
    return audit


def _safe_request_metadata(value: Any) -> dict[str, Any]:
    """Copy only transport/accounting fields, never request bodies or media."""

    if not isinstance(value, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in sorted(_SAFE_REQUEST_METADATA_FIELDS.intersection(value)):
        if key in {"endpoint", "url"}:
            item = _safe_http_url(value[key])
        elif key == "api_key_env":
            raw = value[key]
            item = (
                raw
                if isinstance(raw, str) and _SAFE_ENV_NAME.fullmatch(raw)
                else None
            )
        else:
            item = _safe_metadata_value(value[key])
        if item is not None and item is not _UNSAFE:
            safe[key] = item
    return safe


_UNSAFE = object()


def _safe_metadata_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return _UNSAFE
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNSAFE
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, list):
        result = []
        for item in value[:50]:
            safe_item = _safe_metadata_value(item, depth=depth + 1)
            if safe_item is not _UNSAFE:
                result.append(safe_item)
        return result
    if isinstance(value, dict):
        result = {}
        for key in sorted(_SAFE_NESTED_METADATA_FIELDS.intersection(value)):
            safe_item = _safe_metadata_value(
                value[key],
                depth=depth + 1,
            )
            if safe_item is not _UNSAFE:
                result[key] = safe_item
        return result
    return _UNSAFE


def _safe_short_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:500] if value else None


def _safe_http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        # Drop userinfo, query and fragment so credentials cannot be retained.
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return None
