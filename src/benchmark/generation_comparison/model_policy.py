"""Same-backing-model controls for harness comparison runs."""

from __future__ import annotations

import hashlib
import ipaddress
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


MODEL_POLICY_SCHEMA_VERSION = "generation_comparison_model_policy_v1"
SAME_BACKING_MODEL = "same_backing_model"


def normalize_model_policy(value: Any) -> dict[str, Any] | None:
    """Normalize the optional cross-harness model policy.

    The policy controls only model identity.  Call counts, tools, renderers, and
    deterministic optimizers remain method-native and are reported separately.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("generation.model_policy must be an object")
    policy = str(value.get("policy") or "").strip()
    if policy != SAME_BACKING_MODEL:
        raise ArtifactValidationError(
            f"generation.model_policy.policy must be {SAME_BACKING_MODEL!r}"
        )
    comparison_group = _method_list(
        value.get("comparison_group"),
        "generation.model_policy.comparison_group",
        required=True,
    )
    excluded = _method_list(
        value.get("excluded_baselines") or [],
        "generation.model_policy.excluded_baselines",
        required=False,
    )
    overlap = sorted(set(comparison_group) & set(excluded))
    if overlap:
        raise ArtifactValidationError(
            "generation.model_policy comparison_group and excluded_baselines "
            f"must be disjoint; overlap={overlap}"
        )
    identity = normalize_model_identity(
        value.get("required_identity"),
        path="generation.model_policy.required_identity",
    )
    result = {
        "schema_version": MODEL_POLICY_SCHEMA_VERSION,
        "policy": SAME_BACKING_MODEL,
        "comparison_group": comparison_group,
        "excluded_baselines": excluded,
        "required_identity": identity,
        "required_identity_sha256": canonical_json_sha256(identity),
        "workflow_budget_policy": str(
            value.get("workflow_budget_policy") or "method_native_recorded"
        ),
    }
    if value.get("required_deployment_id") is not None:
        result["required_deployment_id"] = _required_text(
            value.get("required_deployment_id"),
            "generation.model_policy.required_deployment_id",
        )
    if value.get("required_api_base_sha256") is not None:
        digest = _required_text(
            value.get("required_api_base_sha256"),
            "generation.model_policy.required_api_base_sha256",
        ).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ArtifactValidationError(
                "generation.model_policy.required_api_base_sha256 must be lowercase SHA-256"
            )
        result["required_api_base_sha256"] = digest
    return result


def normalize_model_identity(value: Any, *, path: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{path} must be an object")
    provider = _required_text(value.get("provider"), f"{path}.provider")
    model_id = _required_text(
        value.get("model_id") or value.get("model"), f"{path}.model_id"
    )
    result = {"provider": provider, "model_id": model_id}
    if value.get("revision") is not None:
        raise ArtifactValidationError(
            f"{path}.revision is unsupported because the current response APIs "
            "do not independently attest a model revision"
        )
    return result


def configured_model_policy_report(
    *,
    adapter_name: str,
    policy: Mapping[str, Any] | None,
    adapter_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = normalize_model_policy(policy)
    if normalized is None:
        return _report(adapter_name, "NOT_CONTROLLED", True, None, None, [])
    expected = normalized["required_identity"]
    if adapter_name not in normalized["comparison_group"]:
        return _report(
            adapter_name,
            "EXCLUDED_BASELINE",
            True,
            expected,
            None,
            [],
        )
    config = adapter_config if isinstance(adapter_config, Mapping) else {}
    supplied = config.get("model_identity")
    if supplied is None:
        return _report(
            adapter_name,
            "INVALID",
            False,
            expected,
            None,
            ["configured_model_identity_missing"],
        )
    try:
        actual = normalize_model_identity(
            supplied, path=f"adapter_config[{adapter_name}].model_identity"
        )
    except ArtifactValidationError as exc:
        return _report(
            adapter_name,
            "INVALID",
            False,
            expected,
            None,
            [str(exc)],
        )
    reasons = [] if actual == expected else ["configured_model_identity_mismatch"]
    expected_deployment = normalized.get("required_deployment_id")
    actual_deployment = str(config.get("model_deployment_id") or "").strip() or None
    if expected_deployment is not None and actual_deployment != expected_deployment:
        reasons.append("configured_model_deployment_mismatch")
    result = _report(
        adapter_name,
        "VALID" if not reasons else "INVALID",
        not reasons,
        expected,
        actual,
        reasons,
    )
    result["expected_deployment_id"] = expected_deployment
    result["actual_deployment_id"] = actual_deployment
    result["expected_api_base_sha256"] = normalized.get(
        "required_api_base_sha256"
    )
    return result


def reported_model_policy_report(
    *,
    adapter_name: str,
    policy: Mapping[str, Any] | None,
    execution_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the model identity emitted by a real external runner.

    Subprocess runners persist ``runner_report`` as an auxiliary artifact.  A
    callback runner may instead return the same data under
    ``callback_metadata.resource_usage``.
    """

    normalized = normalize_model_policy(policy)
    if normalized is None:
        return _report(adapter_name, "NOT_CONTROLLED", True, None, None, [])
    expected = normalized["required_identity"]
    if adapter_name not in normalized["comparison_group"]:
        return _report(
            adapter_name,
            "EXCLUDED_BASELINE",
            True,
            expected,
            None,
            [],
        )
    identities, identity_evidence = _reported_observation(execution_metadata)
    if not identities:
        result = _report(
            adapter_name,
            "INVALID",
            False,
            expected,
            None,
            ["reported_model_identity_missing"],
        )
        result["identity_evidence"] = identity_evidence
        return result
    mismatches = [identity for identity in identities if identity != expected]
    reasons = []
    if identity_evidence != "observed_response":
        reasons.append("reported_model_identity_not_observed")
    if mismatches:
        reasons.append("reported_model_identity_mismatch")
    expected_deployment = normalized.get("required_deployment_id")
    reported_deployment = _reported_deployment_id(execution_metadata)
    if expected_deployment is not None and reported_deployment != expected_deployment:
        reasons.append("reported_model_deployment_mismatch")
    expected_endpoint = normalized.get("required_api_base_sha256")
    reported_endpoint = _reported_endpoint_sha256(execution_metadata)
    if expected_endpoint is not None and reported_endpoint != expected_endpoint:
        reasons.append("reported_model_endpoint_mismatch")
    result = _report(
        adapter_name,
        "VALID" if not reasons else "INVALID",
        not reasons,
        expected,
        identities[0] if len(identities) == 1 else None,
        reasons,
    )
    result["reported_identities"] = identities
    result["identity_evidence"] = identity_evidence
    result["expected_deployment_id"] = expected_deployment
    result["actual_deployment_id"] = reported_deployment
    result["expected_api_base_sha256"] = expected_endpoint
    result["actual_api_base_sha256"] = reported_endpoint
    return result


def runner_report(execution_metadata: Mapping[str, Any]) -> dict[str, Any]:
    auxiliary = execution_metadata.get("preserved_auxiliary_artifacts")
    auxiliary = auxiliary if isinstance(auxiliary, Mapping) else {}
    item = auxiliary.get("runner_report")
    item = item if isinstance(item, Mapping) else {}
    path = item.get("path")
    if not path:
        return {}
    loaded = read_json(path)
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _reported_observation(
    execution_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, str]], str | None]:
    callback = execution_metadata.get("callback_metadata")
    callback = callback if isinstance(callback, Mapping) else {}
    sources = [callback, runner_report(execution_metadata), execution_metadata]
    raw: list[Any] = []
    identity_evidence = None
    for source in sources:
        usage = source.get("resource_usage") if isinstance(source, Mapping) else None
        usage = usage if isinstance(usage, Mapping) else source
        if not isinstance(usage, Mapping):
            continue
        identities = usage.get("model_identities")
        if isinstance(identities, Sequence) and not isinstance(
            identities, (str, bytes)
        ):
            raw.extend(identities)
        elif usage.get("model_identity") is not None:
            raw.append(usage["model_identity"])
        elif usage.get("model") is not None:
            raw.append(
                {
                    "provider": usage.get("provider") or "openai_compatible",
                    "model_id": usage.get("model"),
                }
            )
        if raw:
            identity_evidence = str(
                usage.get("model_identity_evidence") or ""
            ).strip() or None
            break
    result: list[dict[str, str]] = []
    for index, value in enumerate(raw):
        try:
            identity = normalize_model_identity(
                value, path=f"runner_report.model_identities[{index}]"
            )
        except ArtifactValidationError:
            continue
        if identity not in result:
            result.append(identity)
    return result, identity_evidence


def _reported_deployment_id(
    execution_metadata: Mapping[str, Any],
) -> str | None:
    callback = execution_metadata.get("callback_metadata")
    callback = callback if isinstance(callback, Mapping) else {}
    for source in (callback, runner_report(execution_metadata), execution_metadata):
        usage = source.get("resource_usage") if isinstance(source, Mapping) else None
        usage = usage if isinstance(usage, Mapping) else source
        if isinstance(usage, Mapping):
            value = str(usage.get("model_deployment_id") or "").strip()
            if value:
                return value
    return None


def _reported_endpoint_sha256(
    execution_metadata: Mapping[str, Any],
) -> str | None:
    callback = execution_metadata.get("callback_metadata")
    callback = callback if isinstance(callback, Mapping) else {}
    for source in (callback, runner_report(execution_metadata), execution_metadata):
        usage = source.get("resource_usage") if isinstance(source, Mapping) else None
        usage = usage if isinstance(usage, Mapping) else source
        if isinstance(usage, Mapping):
            value = str(usage.get("model_endpoint_sha256") or "").strip().lower()
            if value:
                return value
    return None


def api_base_sha256(value: str, *, completion_endpoint: bool = False) -> str:
    """Fingerprint a model API base without retaining the non-secret locator."""

    parts = urlsplit(str(value).strip())
    if (
        parts.scheme.lower() not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ArtifactValidationError(
            "model API URL must be HTTP(S) without credentials, query, or fragment"
        )
    if parts.scheme.lower() == "http" and not _loopback_host(parts.hostname):
        raise ArtifactValidationError("non-loopback model APIs must use HTTPS")
    path = parts.path.rstrip("/")
    suffix = "/chat/completions"
    if completion_endpoint:
        if not path.endswith(suffix):
            raise ArtifactValidationError(
                "completion endpoint must end in /chat/completions"
            )
        path = path[: -len(suffix)].rstrip("/")
    host = parts.hostname.lower()
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    normalized = urlunsplit((parts.scheme.lower(), host, path, "", ""))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _method_list(value: Any, path: str, *, required: bool) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ArtifactValidationError(f"{path} must be a list")
    result = [_required_text(item, f"{path}[]") for item in value]
    if required and not result:
        raise ArtifactValidationError(f"{path} must not be empty")
    if len(result) != len(set(result)):
        raise ArtifactValidationError(f"{path} must not contain duplicates")
    return result


def _required_text(value: Any, path: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ArtifactValidationError(f"{path} must be a non-empty string")
    return text


def _report(
    adapter_name: str,
    status: str,
    valid: bool,
    expected: Mapping[str, Any] | None,
    actual: Mapping[str, Any] | None,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_POLICY_SCHEMA_VERSION,
        "method": adapter_name,
        "status": status,
        "valid": valid,
        "expected_identity": deepcopy(dict(expected)) if expected is not None else None,
        "actual_identity": deepcopy(dict(actual)) if actual is not None else None,
        "reasons": list(reasons),
    }


__all__ = [
    "MODEL_POLICY_SCHEMA_VERSION",
    "SAME_BACKING_MODEL",
    "configured_model_policy_report",
    "api_base_sha256",
    "normalize_model_identity",
    "normalize_model_policy",
    "reported_model_policy_report",
    "runner_report",
]
