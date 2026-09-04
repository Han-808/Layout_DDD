"""Standard-library helpers shared by external harness bridge scripts."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


RUNNER_REPORT_SCHEMA_VERSION = "layout_ddd_external_bridge_report_v1"


def read_mapping(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(value)


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return target


def write_text(path: str | Path, value: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")
    return target


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_model_identity() -> dict[str, str]:
    provider = os.environ.get("LAYOUT_DDD_MODEL_PROVIDER", "").strip()
    model_id = os.environ.get("LAYOUT_DDD_MODEL_ID", "").strip()
    if not provider or not model_id:
        raise RuntimeError(
            "LAYOUT_DDD_MODEL_PROVIDER and LAYOUT_DDD_MODEL_ID are required"
        )
    result = {"provider": provider, "model_id": model_id}
    if os.environ.get("LAYOUT_DDD_MODEL_REVISION", "").strip():
        raise RuntimeError(
            "model revision cannot be frozen without response-derived evidence"
        )
    return result


def required_model_deployment_id() -> str:
    value = os.environ.get("LAYOUT_DDD_MODEL_DEPLOYMENT_ID", "").strip()
    if not value:
        raise RuntimeError("LAYOUT_DDD_MODEL_DEPLOYMENT_ID is required")
    return value


def verify_api_endpoint_contract(
    value: str,
    *,
    completion_endpoint: bool,
) -> str:
    """Hash the already-configured route and compare it with the protocol.

    LayoutGPT is configured with the full ``/chat/completions`` URL while the
    other wrappers receive the API base. Normalization removes only that known
    suffix, so all four runners prove the same non-secret base-route identity
    without persisting the endpoint itself.
    """

    expected = os.environ.get(
        "LAYOUT_DDD_REQUIRED_API_BASE_SHA256", ""
    ).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise RuntimeError("LAYOUT_DDD_REQUIRED_API_BASE_SHA256 is invalid")
    parts = urlsplit(str(value).strip())
    if (
        parts.scheme.lower() not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise RuntimeError(
            "model endpoint must be HTTP(S) without embedded credentials/query"
        )
    if parts.scheme.lower() == "http" and not _loopback_host(parts.hostname):
        raise RuntimeError("non-loopback model endpoints must use HTTPS")
    path = parts.path.rstrip("/")
    suffix = "/chat/completions"
    if completion_endpoint:
        if not path.endswith(suffix):
            raise RuntimeError("LayoutGPT endpoint must end in /chat/completions")
        path = path[: -len(suffix)].rstrip("/")
    host = parts.hostname.lower()
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    normalized = urlunsplit((parts.scheme.lower(), host, path, "", ""))
    actual = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if actual != expected:
        raise RuntimeError(
            "configured model endpoint differs from comparison endpoint fingerprint"
        )
    return actual


def _loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def verify_model_contract(
    comparison_control: Mapping[str, Any], identity: Mapping[str, str]
) -> None:
    generation = comparison_control.get("generation")
    generation = generation if isinstance(generation, Mapping) else {}
    policy = generation.get("model_policy")
    if not isinstance(policy, Mapping):
        return
    expected = policy.get("required_identity")
    if not isinstance(expected, Mapping) or dict(expected) != dict(identity):
        raise RuntimeError(
            "bridge model identity does not match comparison model policy: "
            f"expected={expected!r}, actual={dict(identity)!r}"
        )


def verify_catalog_contract(
    comparison_control: Mapping[str, Any], method_catalog: Mapping[str, Any]
) -> None:
    expected = comparison_control.get("catalog")
    actual = method_catalog.get("catalog")
    if not isinstance(expected, Mapping) or dict(actual or {}) != dict(expected):
        raise RuntimeError(
            "method catalog identity differs from comparison control: "
            f"expected={expected!r}, actual={actual!r}"
        )


def public_object_plan(method_input: Mapping[str, Any]) -> dict[str, Any]:
    visible = method_input.get("generator_input")
    visible = visible if isinstance(visible, Mapping) else {}
    structure = visible.get("structure")
    structure = structure if isinstance(structure, Mapping) else {}
    plan = structure.get("object_plan")
    if not isinstance(plan, Mapping):
        raise RuntimeError("method input lacks public structured object_plan")
    return dict(plan)


def openai_chat_completion(
    *,
    messages: Sequence[Mapping[str, Any]],
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    endpoint = os.environ.get("LAYOUT_DDD_API_ENDPOINT", "").strip()
    api_key = os.environ.get("LAYOUT_DDD_API_KEY", "").strip()
    identity = required_model_identity()
    if not endpoint:
        raise RuntimeError("LAYOUT_DDD_API_ENDPOINT is required")
    if not api_key:
        raise RuntimeError("LAYOUT_DDD_API_KEY is required")
    payload = {
        "model": identity["model_id"],
        "messages": list(messages),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        opener = urllib.request.build_opener(_RejectRedirectHandler())
        with opener.open(
            request,
            timeout=float(os.environ.get("LAYOUT_DDD_API_TIMEOUT", "600")),
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read(65537).decode("utf-8", errors="replace")
        detail = _redacted_error_detail(
            raw,
            secret=api_key,
            truncated=len(raw.encode("utf-8")) > 65536,
        )
        raise RuntimeError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        reason = _redacted_error_detail(
            str(exc.reason), secret=api_key, truncated=False
        )
        raise RuntimeError(
            f"could not reach configured model endpoint: {reason}"
        ) from exc
    if not isinstance(response_payload, Mapping):
        raise RuntimeError("model endpoint response must be a JSON object")
    choices = response_payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise RuntimeError("model endpoint response has no choices")
    first = choices[0]
    first = first if isinstance(first, Mapping) else {}
    message = first.get("message")
    message = message if isinstance(message, Mapping) else {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("model endpoint returned empty message content")
    usage = response_payload.get("usage")
    usage = dict(usage) if isinstance(usage, Mapping) else {}
    return content, {
        "request_seconds": time.monotonic() - started,
        "usage": usage,
        "response_id": response_payload.get("id"),
        "response_model": response_payload.get("model"),
    }


def observed_model_identity(
    response: Any,
    *,
    provider: str,
) -> dict[str, str]:
    """Extract the model identity returned by an API/LangChain response."""

    model_id = None
    if isinstance(response, Mapping):
        model_id = response.get("model") or response.get("model_name")
        metadata = response.get("response_metadata")
    else:
        model_id = getattr(response, "model", None) or getattr(
            response, "model_name", None
        )
        metadata = getattr(response, "response_metadata", None)
    if model_id is None and isinstance(metadata, Mapping):
        model_id = metadata.get("model_name") or metadata.get("model")
    if model_id is None or not str(model_id).strip():
        raise RuntimeError("model response did not expose an observed model identity")
    return {"provider": str(provider), "model_id": str(model_id).strip()}


def require_observed_model_match(
    expected: Mapping[str, str],
    observed: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    identities = [dict(item) for item in observed]
    if not identities:
        raise RuntimeError("no observed response model identities were captured")
    mismatches = [item for item in identities if item != dict(expected)]
    if mismatches:
        raise RuntimeError(
            "observed response model identity differs from the comparison contract: "
            f"expected={dict(expected)!r}, observed={identities!r}"
        )
    return identities


def run_plugin(command: Sequence[str], *, cwd: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=Path(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": list(command),
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "runtime_seconds": time.monotonic() - started,
    }


def write_runner_report(
    path: str | Path,
    *,
    adapter: str,
    identity: Mapping[str, str],
    generation_calls: int | None,
    tokens: int | None = None,
    iteration_count: int | None = None,
    rendering_calls: int | None = None,
    tool_calls: int | None = None,
    protocol_observation: Mapping[str, Any] | None = None,
    observed_model_identities: Sequence[Mapping[str, str]] | None = None,
    model_identity_evidence: str = "configured_only",
    model_deployment_id: str | None = None,
    model_endpoint_sha256: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    observed = [dict(item) for item in (observed_model_identities or [])]
    usage = {
        "configured_model_identity": dict(identity),
        "model_identity": observed[0] if len(observed) == 1 else None,
        "model_identities": observed,
        "model": observed[0]["model_id"] if observed else None,
        "provider": observed[0]["provider"] if observed else None,
        "model_identity_evidence": str(model_identity_evidence),
        "model_deployment_id": model_deployment_id,
        "model_endpoint_sha256": model_endpoint_sha256,
        "generation_calls": generation_calls,
        "tokens": tokens,
        "iteration_count": iteration_count,
        "rendering_calls": rendering_calls,
        "tool_calls": tool_calls,
        "retrieval_calls": 0,
    }
    return write_json(
        path,
        {
            "schema_version": RUNNER_REPORT_SCHEMA_VERSION,
            "adapter": adapter,
            "resource_usage": usage,
            "protocol_observation": dict(protocol_observation or {}),
            **dict(extra or {}),
        },
    )


def total_tokens(usage: Mapping[str, Any]) -> int | None:
    value = usage.get("total_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if all(isinstance(item, int) and not isinstance(item, bool) for item in (prompt, completion)):
        return int(prompt) + int(completion)
    return None


def response_total_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        value = getattr(usage, "total_tokens", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    metadata = getattr(response, "usage_metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("total_tokens")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        input_tokens = metadata.get("input_tokens")
        output_tokens = metadata.get("output_tokens")
        if all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in (input_tokens, output_tokens)
        ):
            return int(input_tokens) + int(output_tokens)
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, Mapping):
            return total_tokens(token_usage)
    return None


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            "redirect responses are not followed",
            headers,
            fp,
        )


def _redacted_error_detail(
    value: str,
    *,
    secret: str | None,
    truncated: bool,
) -> str:
    safe = str(value)
    if secret:
        safe = safe.replace(secret, "<redacted>")
    safe = re.sub(r"(?i)\bbearer\s+[^\s,;\"']+", "Bearer <redacted>", safe)
    safe = re.sub(
        r'(?i)(["\']?(?:api[_-]?key|authorization|access[_-]?token|secret)["\']?\s*[:=]\s*)'
        r'(["\']?)[^\s,;}\]]+\2',
        r"\1<redacted>",
        safe,
    )
    result = safe[:4096]
    if truncated or len(safe) > 4096:
        result += "...<truncated>"
    return result
