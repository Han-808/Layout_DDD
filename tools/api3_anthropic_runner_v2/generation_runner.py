#!/usr/bin/env python3
"""Auditable HY4 two-stage retrieval-conditioned scene generator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .artifacts import (
        ArtifactError,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
        write_exclusive,
        write_json_exclusive,
    )
    from .contracts import (
        ContractError,
        build_retrieval_request,
        validate_brief,
        validate_object_plan,
        validate_placement,
    )
    from .strict_json import StrictJSONError, loads_strict
    from .transport import EndpointError, TransportResult, post_once
except ImportError:  # direct script execution on the Pod
    from artifacts import (  # type: ignore
        ArtifactError,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
        write_exclusive,
        write_json_exclusive,
    )
    from contracts import (  # type: ignore
        ContractError,
        build_retrieval_request,
        validate_brief,
        validate_object_plan,
        validate_placement,
    )
    from strict_json import StrictJSONError, loads_strict  # type: ignore
    from transport import EndpointError, TransportResult, post_once  # type: ignore


RUNNER_VERSION = "2.0.0"
RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_BRIEFS = RUNNER_ROOT / "briefs.json"
DEFAULT_MODELS = RUNNER_ROOT / "models.pod.json"
DEFAULT_STAGE_A_PROMPT = RUNNER_ROOT / "stage_a_prompt.txt"
DEFAULT_STAGE_C_PROMPT = RUNNER_ROOT / "stage_c_prompt.txt"
DEFAULT_RETRIEVER_ROOT = RUNNER_ROOT
RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# Opt-in for recovery runners.  The default preserves the original one-shot
# delivery-ambiguity policy used by existing generation campaigns.
RETRY_TRANSPORT_AMBIGUOUS = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    endpoint: str
    api_key: str
    configured_model: str
    wire_model: str
    timeout_seconds: float
    max_infrastructure_retries: int
    retry_delay_seconds: float
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_tokens: int
    repetition_penalty: float | None
    reasoning_effort: str | None
    preserved_thinking: bool | None
    strategy_type: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "endpoint": self.endpoint,
            "configured_model": self.configured_model,
            "wire_model": self.wire_model,
            "timeout_seconds": self.timeout_seconds,
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "reasoning_effort": self.reasoning_effort,
            "preserved_thinking": self.preserved_thinking,
            "strategy_type": self.strategy_type,
            "stream": False,
            "prompt_model_identity_injected": False,
            "generator_semantic_retry_allowed": False,
        }


@dataclass(frozen=True)
class StageCapture:
    status: str
    attempt_count: int
    infrastructure_retry_count: int
    content: bytes | None
    stop_batch: bool
    reason: str | None


class RetrieverAdapter:
    """Load the separately frozen retriever runtime once per runner process."""

    def __init__(self, runtime_root: Path) -> None:
        module_path = runtime_root / "retriever_runtime.py"
        config_path = runtime_root / "retriever_runtime.pod.json"
        if not module_path.is_file() or not config_path.is_file():
            raise ArtifactError(f"frozen retriever runtime is incomplete: {runtime_root}")
        spec = importlib.util.spec_from_file_location("hy34_frozen_retriever", module_path)
        if spec is None or spec.loader is None:
            raise ArtifactError(f"cannot load frozen retriever module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        runtime = module.FrozenRetrieverRuntime(module.RuntimeConfig.load(config_path))
        report = runtime.gate(strict=False, run_golden=True)
        if report["status"] == "failed":
            raise ArtifactError(f"frozen retriever gate failed: {report['errors']}")
        self.module = module
        self.runtime = runtime
        self.gate_report = report
        self.runtime_root = runtime_root

    def retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.module.retrieve_batch(self.runtime, request)


def _load_model_config(path: Path, model_key: str) -> ModelConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "hy34_model_transport_config_v1":
        raise ValueError("unsupported model config schema")
    models = value.get("models")
    if not isinstance(models, dict) or model_key not in models:
        raise ValueError(f"unknown model key {model_key!r}")
    model = models[model_key]
    request = value["request"]
    api_key_env = value.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError("model config must name a non-empty api_key_env")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"required credential environment variable is not set: {api_key_env}")
    return ModelConfig(
        key=model_key,
        label=str(model["label"]),
        endpoint=str(value["endpoint"]),
        api_key=api_key,
        configured_model=str(model["configured_model"]),
        wire_model=str(model["wire_model"]),
        timeout_seconds=float(request["timeout_seconds"]),
        max_infrastructure_retries=int(request["max_infrastructure_retries"]),
        retry_delay_seconds=float(request["retry_delay_seconds"]),
        temperature=None if request.get("temperature") is None else float(request["temperature"]),
        top_p=None if request.get("top_p") is None else float(request["top_p"]),
        top_k=None if request.get("top_k") is None else int(request["top_k"]),
        max_tokens=int(request["max_tokens"]),
        repetition_penalty=(
            None if request.get("repetition_penalty") is None else float(request["repetition_penalty"])
        ),
        reasoning_effort=(
            None if request.get("reasoning_effort") is None else str(request["reasoning_effort"])
        ),
        preserved_thinking=(
            None if request.get("preserved_thinking") is None else bool(request["preserved_thinking"])
        ),
        strategy_type=str(request["strategy_type"]),
    )


def _load_briefs(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "hy34_paired_briefs_v1":
        raise ValueError("unsupported paired briefs schema")
    briefs = [validate_brief(item) for item in value.get("briefs", [])]
    expected = [f"brief_{index:02d}" for index in range(10)]
    if [item["brief_id"] for item in briefs] != expected:
        raise ValueError("paired briefs must be exactly ordered brief_00 through brief_09")
    return briefs


def _runner_source_manifest() -> dict[str, Any]:
    names = [
        "README.md",
        "__init__.py",
        "artifacts.py",
        "contracts.py",
        "generation_runner.py",
        "strict_json.py",
        "transport.py",
        "stage_a_prompt.txt",
        "stage_c_prompt.txt",
        "briefs.json",
        "models.pod.json",
        "run_generation.sh",
        "schemas/object_plan.schema.json",
        "schemas/catalog_placement.schema.json",
    ]
    files = []
    for name in names:
        path = RUNNER_ROOT / name
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"runner_version": RUNNER_VERSION, "files": files}
    return {**payload, "manifest_sha256": sha256_bytes(canonical_json_bytes(payload))}


def _resolve_source_manifest(source_manifest: Any | None) -> dict[str, Any]:
    """Resolve an adapter-owned manifest without replacing the core global."""

    if source_manifest is None:
        return _runner_source_manifest()
    value = source_manifest() if callable(source_manifest) else source_manifest
    if not isinstance(value, Mapping):
        raise ValueError("source_manifest must be a mapping or zero-argument callable")
    return dict(value)


def _request_value(
    *,
    model: ModelConfig,
    system_prompt: str,
    user_value: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "model": model.wire_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": canonical_json_bytes(user_value).decode("utf-8")},
        ],
        "max_tokens": model.max_tokens,
        "stream": False,
    }
    if model.temperature is not None:
        value["temperature"] = model.temperature
    if model.top_p is not None:
        value["top_p"] = model.top_p
    if model.top_k is not None:
        value["top_k"] = model.top_k
    if model.repetition_penalty is not None:
        value["repetition_penalty"] = model.repetition_penalty
    if model.reasoning_effort is not None or model.preserved_thinking is not None:
        value["chat_template_kwargs"] = {
            "reasoning_effort": model.reasoning_effort,
            "preserved_thinking": model.preserved_thinking,
        }
    return value


def _request_headers(model: ModelConfig, session_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": f"hy34-two-stage-generator/{RUNNER_VERSION}",
        "Authorization": f"Bearer {model.api_key}",
        "SessionID": session_id,
        "StrategyType": model.strategy_type,
    }


def _extract_api_message(response_body: bytes) -> tuple[bytes, bytes | None, bytes | None, dict[str, Any]]:
    try:
        value = json.loads(response_body.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid API JSON envelope: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("API response must be an object")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("API response must contain exactly one choice")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("API response must contain string choices[0].message.content")
    reasoning = message.get("reasoning")
    reasoning_content = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("message.reasoning must be string or null")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ValueError("message.reasoning_content must be string or null")
    return (
        message["content"].encode("utf-8"),
        None if reasoning is None else reasoning.encode("utf-8"),
        None if reasoning_content is None else reasoning_content.encode("utf-8"),
        value.get("usage") if isinstance(value.get("usage"), dict) else {},
    )


def _provider_request_value(
    *,
    provider_route: Any | None,
    model: ModelConfig,
    system_prompt: str,
    user_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Dispatch request encoding without module-global adapter mutation.

    The instance-scoped compatibility boundary is documented in
    ``docs/generation_transport_compatibility.md``. Keeping the legacy branch
    explicit preserves the default runner's request bytes and behavior.
    """

    if provider_route is None:
        return _request_value(
            model=model,
            system_prompt=system_prompt,
            user_value=user_value,
        )
    return provider_route.request_value(
        model=model,
        system_prompt=system_prompt,
        user_value=user_value,
        canonical_json_bytes=canonical_json_bytes,
    )


def _provider_request_headers(
    *,
    provider_route: Any | None,
    model: ModelConfig,
    session_id: str,
) -> dict[str, str]:
    """Dispatch route-local headers, or preserve the frozen legacy default."""

    if provider_route is None:
        return _request_headers(model, session_id)
    return provider_route.request_headers(model, session_id)


def _provider_extract_api_message(
    *,
    provider_route: Any | None,
    response_body: bytes,
) -> tuple[bytes, bytes | None, bytes | None, dict[str, Any]]:
    """Dispatch route-local response decoding without changing stage logic."""

    if provider_route is None:
        return _extract_api_message(response_body)
    return provider_route.extract_api_message(response_body).as_legacy_tuple()


def _save_transport_response(attempt_dir: Path, result: TransportResult) -> str | None:
    request_id = None
    if result.response_headers is not None:
        for name, value in result.response_headers:
            if name.lower() == "x-request-id":
                request_id = value
        redacted_headers = [
            (name, "<redacted>" if name.lower() in {"authorization", "proxy-authorization", "set-cookie"} else value)
            for name, value in result.response_headers
        ]
        write_json_exclusive(
            attempt_dir / "response-headers.json",
            {
                "http_status": result.http_status,
                "http_reason": result.http_reason,
                "headers": redacted_headers,
                "x_request_id": request_id,
            },
        )
    if result.response_body is not None:
        write_exclusive(attempt_dir / "api-response.body", result.response_body)
    return request_id


def call_model_stage(
    *,
    stage: str,
    stage_dir: Path,
    model: ModelConfig,
    system_prompt: str,
    user_value: Mapping[str, Any],
    provider_route: Any | None = None,
    retry_policy: Any | None = None,
) -> StageCapture:
    request_body = canonical_json_bytes(
        _provider_request_value(
            provider_route=provider_route,
            model=model,
            system_prompt=system_prompt,
            user_value=user_value,
        )
    )
    request_hash = sha256_bytes(request_body)
    max_attempts = model.max_infrastructure_retries + 1
    retry_ambiguous_timeouts = (
        RETRY_TRANSPORT_AMBIGUOUS
        if retry_policy is None
        else bool(retry_policy.retry_ambiguous_timeouts)
    )
    retryable_transport_statuses = (
        frozenset(
            {"transport_failure"}
            | ({"transport_ambiguous"} if RETRY_TRANSPORT_AMBIGUOUS else set())
        )
        if retry_policy is None
        else retry_policy.retryable_transport_statuses
    )
    retryable_http_statuses = (
        RETRYABLE_HTTP_STATUSES
        if retry_policy is None
        else retry_policy.retryable_http_statuses
    )
    retry_delay_seconds = (
        model.retry_delay_seconds
        if retry_policy is None
        else float(retry_policy.retry_delay_seconds)
    )
    for attempt_number in range(1, max_attempts + 1):
        attempt_dir = stage_dir / f"attempt_{attempt_number:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        session_id = str(uuid.uuid4())
        headers = _provider_request_headers(
            provider_route=provider_route,
            model=model,
            session_id=session_id,
        )
        started = utc_now()
        write_exclusive(attempt_dir / "request.json", request_body)
        write_json_exclusive(
            attempt_dir / "request-headers.json",
            {
                "headers": [
                    (name, "Bearer <redacted>" if name.lower() == "authorization" else value)
                    for name, value in headers.items()
                ],
                "session_id": session_id,
                "strategy_type": model.strategy_type,
            },
        )
        write_json_exclusive(
            attempt_dir / "attempt.started.json",
            {
                "stage": stage,
                "attempt_number": attempt_number,
                "started_at": started,
                "request_sha256": request_hash,
                "session_id": session_id,
                "configured_model": model.configured_model,
                "wire_model": model.wire_model,
                "infrastructure_retry": attempt_number > 1,
            },
        )
        result = post_once(
            model.endpoint,
            request_body,
            connect_timeout=model.timeout_seconds,
            read_timeout=model.timeout_seconds,
            request_headers=headers,
        )
        request_id = _save_transport_response(attempt_dir, result)
        detail = {
            "stage": result.stage,
            "elapsed_seconds": round(result.elapsed_seconds, 6),
            "http_status": result.http_status,
            "http_reason": result.http_reason,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "x_request_id": request_id,
        }
        if result.status == "transport_ambiguous":
            status = "transport_ambiguous"
            write_json_exclusive(attempt_dir / "attempt.result.json", {"status": status, **detail})
            if (
                retry_ambiguous_timeouts
                and status in retryable_transport_statuses
                and attempt_number < max_attempts
            ):
                if retry_delay_seconds:
                    time.sleep(retry_delay_seconds)
                continue
            return StageCapture(
                status,
                attempt_number,
                attempt_number - 1,
                None,
                not retry_ambiguous_timeouts,
                result.stage,
            )
        if result.status == "transport_failure":
            status = "transport_failure"
            write_json_exclusive(attempt_dir / "attempt.result.json", {"status": status, **detail})
            if (
                status in retryable_transport_statuses
                and attempt_number < max_attempts
            ):
                if retry_delay_seconds:
                    time.sleep(retry_delay_seconds)
                continue
            return StageCapture(status, attempt_number, attempt_number - 1, None, False, result.stage)
        assert result.response_body is not None and result.http_status is not None
        if not 200 <= result.http_status < 300:
            status = "http_error"
            write_json_exclusive(attempt_dir / "attempt.result.json", {"status": status, **detail})
            if result.http_status in retryable_http_statuses and attempt_number < max_attempts:
                if retry_delay_seconds:
                    time.sleep(retry_delay_seconds)
                continue
            return StageCapture(status, attempt_number, attempt_number - 1, None, False, str(result.http_status))
        try:
            content, reasoning, reasoning_content, usage = _provider_extract_api_message(
                provider_route=provider_route,
                response_body=result.response_body,
            )
        except (UnicodeError, ValueError) as exc:
            status = "invalid_api_response"
            write_json_exclusive(
                attempt_dir / "attempt.result.json",
                {"status": status, **detail, "error_type": type(exc).__name__, "error_message": str(exc)},
            )
            return StageCapture(status, attempt_number, attempt_number - 1, None, False, str(exc))
        write_exclusive(attempt_dir / "raw-content.txt", content)
        if reasoning is not None:
            write_exclusive(attempt_dir / "logs" / "reasoning.txt", reasoning)
        if reasoning_content is not None:
            write_exclusive(attempt_dir / "logs" / "reasoning_content.txt", reasoning_content)
        status = "captured"
        write_json_exclusive(
            attempt_dir / "capture.json",
            {
                "status": status,
                "raw_content_bytes": len(content),
                "raw_content_sha256": sha256_bytes(content),
                "reasoning_sha256": None if reasoning is None else sha256_bytes(reasoning),
                "reasoning_content_sha256": (
                    None if reasoning_content is None else sha256_bytes(reasoning_content)
                ),
                "usage": usage,
                "model_content_repair_applied": False,
                "model_content_schema_retry_allowed": False,
            },
        )
        write_json_exclusive(attempt_dir / "attempt.result.json", {"status": status, **detail})
        return StageCapture(status, attempt_number, attempt_number - 1, content, False, None)
    raise AssertionError("unreachable model stage loop")


def _hash_if_file(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _artifact_hashes(case_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(case_dir.rglob("*")):
        if not path.is_file() or path.name in {"audit_manifest.json", "case.result.json"}:
            continue
        hashes[path.relative_to(case_dir).as_posix()] = sha256_file(path)
    return hashes


def _finalize_case(
    *,
    case_dir: Path,
    model: ModelConfig,
    brief: Mapping[str, Any],
    status: str,
    reason: str | None,
    stop_batch: bool,
    stage_a: StageCapture | None,
    stage_c: StageCapture | None,
    retrieval_count: int | None,
    placement_emission_count: int,
    eligible: bool,
) -> dict[str, Any]:
    first = case_dir / "catalog_placement_first_emission.json"
    frozen = case_dir / "catalog_placement_v1.json"
    first_hash = _hash_if_file(first)
    frozen_hash = _hash_if_file(frozen)
    freeze = {
        "schema_version": "hy34_generation_freeze_v2",
        "brief_id": brief["brief_id"],
        "model_key": model.key,
        "model_label": model.label,
        "frozen_at": utc_now(),
        "hashes": {
            "fixed_instruction": _hash_if_file(case_dir / "fixed_instruction.json"),
            "scene_request": _hash_if_file(case_dir / "scene_request.json"),
            "object_plan": _hash_if_file(case_dir / "object_plan.json"),
            "retrieval_requests": _hash_if_file(case_dir / "retrieval_requests.json"),
            "retrieval_results": _hash_if_file(case_dir / "retrieval_results.json"),
            "placement_first_emission": first_hash,
            "placement_frozen": frozen_hash,
        },
    }
    write_json_exclusive(case_dir / "generation_freeze.json", freeze)
    audit = {
        "schema_version": "hy34_one_shot_audit_v2",
        "brief_id": brief["brief_id"],
        "model_key": model.key,
        "model_label": model.label,
        "status": status,
        "reason": reason,
        "object_plan_response_count": 0 if stage_a is None else int(stage_a.status == "captured"),
        "placement_response_count": 0 if stage_c is None else int(stage_c.status == "captured"),
        "placement_emission_count": placement_emission_count,
        "first_emission_equals_frozen_placement": (
            first_hash is not None and first_hash == frozen_hash
        ),
        "post_emission_transform_edit_count": 0,
        "retrieval_total_invocations": retrieval_count,
        "retrieval_invocations_per_public_slot": (
            None if retrieval_count is None else (1 if retrieval_count else 0)
        ),
        "retrieval_failure_may_be_partial": retrieval_count is None,
        "retrieval_retry_count": 0,
        "asset_replacement_count": 0,
        "generator_semantic_retry_count": 0,
        "api_infrastructure_retry_count": sum(
            item.infrastructure_retry_count for item in (stage_a, stage_c) if item is not None
        ),
        "geometry_feedback_used_before_freeze": False,
        "geometry_feedback_used_after_freeze": False,
        "render_or_evaluator_feedback_used": False,
        "placement_optimizer_used": False,
        "query_rewrite_used": False,
        "model_concurrency_prompt_used": False,
        "conversion_deferred_to_local": True,
        "eligible_for_strict_one_shot_evaluation": eligible,
    }
    write_json_exclusive(case_dir / "one_shot_audit.json", audit)
    manifest = {
        "schema_version": "hy34_case_audit_manifest_v2",
        "brief_id": brief["brief_id"],
        "model_key": model.key,
        "artifact_hashes": _artifact_hashes(case_dir),
    }
    write_json_exclusive(case_dir / "audit_manifest.json", manifest)
    result = {
        "schema_version": "hy34_case_result_v2",
        "brief_id": brief["brief_id"],
        "model_key": model.key,
        "model_label": model.label,
        "status": status,
        "reason": reason,
        "stop_batch": stop_batch,
        "eligible_for_strict_one_shot_evaluation": eligible,
        "completed_at": utc_now(),
    }
    write_json_exclusive(case_dir / "case.result.json", result)
    return result


def run_case(
    *,
    output_root: Path,
    model: ModelConfig,
    brief: Mapping[str, Any],
    retriever: Any,
    stage_a_prompt: str,
    stage_c_prompt: str,
    provider_route: Any | None = None,
    retry_policy: Any | None = None,
) -> dict[str, Any]:
    case_dir = output_root / str(brief["brief_id"])
    case_dir.mkdir(parents=False, exist_ok=False)
    fixed_instruction = {
        "schema_version": "hy34_fixed_instruction_v2",
        "brief": brief,
        "coordinate_frame": "room_min_corner_x_width_y_depth_z_up_meters",
        "catalog_facing_prior": "directed_local_neg_y",
        "retrieval_policy": {
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "category_argument": None,
            "size_constraint_used": False,
            "top_k": 1,
            "min_score": 0.3,
            "query_rewrite_allowed": False,
            "retry_allowed": False,
            "asset_replacement_allowed": False,
        },
        "one_shot_contract": {
            "stage_a_semantic_emissions_allowed": 1,
            "placement_emissions_allowed": 1,
            "post_emission_edits_allowed": 0,
            "geometry_or_evaluator_feedback_allowed": False,
        },
        "prompt_hashes": {
            "stage_a": sha256_bytes(stage_a_prompt.encode("utf-8")),
            "stage_c": sha256_bytes(stage_c_prompt.encode("utf-8")),
        },
        "runner_owned_metadata": {"model_key": model.key, "model_label": model.label},
        "frozen_at": utc_now(),
    }
    write_json_exclusive(case_dir / "fixed_instruction.json", fixed_instruction)
    write_json_exclusive(case_dir / "scene_request.json", {"brief": brief})

    stage_a = call_model_stage(
        stage="stage_a_object_plan",
        stage_dir=case_dir / "stage_a",
        model=model,
        system_prompt=stage_a_prompt,
        user_value={"brief": brief},
        provider_route=provider_route,
        retry_policy=retry_policy,
    )
    if stage_a.status != "captured" or stage_a.content is None:
        return _finalize_case(
            case_dir=case_dir,
            model=model,
            brief=brief,
            status="stage_a_failed",
            reason=stage_a.reason or stage_a.status,
            stop_batch=stage_a.stop_batch,
            stage_a=stage_a,
            stage_c=None,
            retrieval_count=0,
            placement_emission_count=0,
            eligible=False,
        )
    write_exclusive(case_dir / "object_plan_first_emission.json", stage_a.content)
    try:
        plan = validate_object_plan(
            loads_strict(stage_a.content.decode("utf-8", errors="strict")),
            brief=brief,
        )
    except (UnicodeError, StrictJSONError, ContractError) as exc:
        write_json_exclusive(
            case_dir / "object_plan_validation.json",
            {"valid": False, "error_type": type(exc).__name__, "error_message": str(exc)},
        )
        return _finalize_case(
            case_dir=case_dir,
            model=model,
            brief=brief,
            status="stage_a_schema_invalid",
            reason=str(exc),
            stop_batch=False,
            stage_a=stage_a,
            stage_c=None,
            retrieval_count=0,
            placement_emission_count=0,
            eligible=False,
        )
    write_exclusive(case_dir / "object_plan.json", stage_a.content)
    write_json_exclusive(case_dir / "object_plan_validation.json", {"valid": True})
    retrieval_request = build_retrieval_request(plan)
    write_json_exclusive(case_dir / "retrieval_requests.json", retrieval_request)
    try:
        retrieval_results = retriever.retrieve(retrieval_request)
    except Exception as exc:
        write_json_exclusive(
            case_dir / "retrieval_failure.json",
            {"error_type": type(exc).__name__, "error_message": str(exc)},
        )
        return _finalize_case(
            case_dir=case_dir,
            model=model,
            brief=brief,
            status="retrieval_failed",
            reason=str(exc),
            stop_batch=True,
            stage_a=stage_a,
            stage_c=None,
            retrieval_count=None,
            placement_emission_count=0,
            eligible=False,
        )
    write_json_exclusive(case_dir / "retrieval_results.json", retrieval_results)
    plan_by_id = {str(item["id"]): item for item in plan["objects"]}
    selection_reason = (
        "frozen deterministic semantic Top-1 retrieval result; "
        "no size soft score, rewrite, retry, or replacement"
    )
    selection_objects = []
    for row in retrieval_results["results"]:
        slot_id = str(row["slot_id"])
        object_spec = plan_by_id[slot_id]
        rank1 = row["rank1"]
        asset_size = [float(value) for value in rank1["size"]]
        selected_asset = {
            "jid": str(rank1["jid"]),
            "category": str(rank1.get("category") or ""),
            "desc": str(rank1.get("description") or ""),
            "short_desc": str(rank1.get("short_desc") or ""),
            "size": asset_size,
            "asset_ref": {
                "source_db": "imaginarium",
                "asset_key": str(rank1["jid"]),
            },
            "asset_proxy": {
                "type": "canonical_catalog_bbox",
                "bbox_center_local": [0.0, 0.0, 0.0],
                "bbox_size": asset_size,
            },
            "metadata": {
                "catalog_facing_contract_version": "imaginarium_catalog_facing_v1",
                "default_directed_functional_side": "local_neg_y",
            },
        }
        selection_objects.append(
            {
                "object_id": slot_id,
                "object_spec": object_spec,
                "retrieval_query": {
                    "description": str(row["retrieval_query"]),
                    "category": None,
                    "size_constraint": None,
                    "top_k": 1,
                },
                "selected_asset": selected_asset,
                "candidates": [
                    {
                        "rank": 1,
                        "jid": str(rank1["jid"]),
                        "category": str(rank1.get("category") or ""),
                        "short_desc": str(rank1.get("short_desc") or ""),
                        "description": str(rank1.get("description") or ""),
                        "size": asset_size,
                        "score": float(rank1["score"]),
                    }
                ],
                "selection_action": "select",
                "selection_decision": {
                    "action": "select",
                    "selected_jid": str(rank1["jid"]),
                    "reason": selection_reason,
                    "generation_request": None,
                },
                "selection_reason": selection_reason,
            }
        )
    asset_selection = {
        "schema_version": "hy34_frozen_asset_selection_v2",
        "objects": selection_objects,
    }
    write_json_exclusive(case_dir / "asset_selection.json", asset_selection)
    width, depth, height = [float(value) for value in brief["room_dimensions_m"]]
    room = {
        "boundary": [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]],
        "height": height,
        "unit": "meter",
        "dimensions": {"width": width, "depth": depth, "height": height},
        "floor_z": 0.0,
        "topology": "rectangular_logical_boundary",
    }
    generation_input = {
        "schema_version": "hy34_generation_input_v2",
        "scene_request": {
            "instruction": brief["instruction"],
            "scene_type": plan["scene_type"],
            "structure": True,
            "prompt_granularity": plan["prompt_granularity"],
            "room_type": brief["room_type"],
            "target_instances": brief["target_instances"],
            "physical_wall_policy": brief["physical_wall_policy"],
            "active_wall_ids": brief["active_wall_ids"],
            "room": room,
        },
        "generation_contract": {
            "output_format": "catalog_placement_v1",
            "requires_pose": True,
            "input_mode": "structured_assets",
            "requires_asset_selection": True,
            "coordinate_frame": "room_min_corner_x_width_y_depth_z_up_meters",
            "architecture": {
                "id": "bounded_room_explicit_walls_v1",
                "physical_wall_policy": brief["physical_wall_policy"],
                "active_wall_ids": brief["active_wall_ids"],
                "room": room,
            },
            "catalog_facing_prior": "directed_local_neg_y",
            "one_shot": True,
        },
        "object_plan": plan,
        "asset_selection": asset_selection,
    }
    write_json_exclusive(case_dir / "generation_input.json", generation_input)
    stage_c = call_model_stage(
        stage="stage_c_placement",
        stage_dir=case_dir / "stage_c",
        model=model,
        system_prompt=stage_c_prompt,
        user_value=generation_input,
        provider_route=provider_route,
        retry_policy=retry_policy,
    )
    if stage_c.status != "captured" or stage_c.content is None:
        return _finalize_case(
            case_dir=case_dir,
            model=model,
            brief=brief,
            status="stage_c_failed",
            reason=stage_c.reason or stage_c.status,
            stop_batch=stage_c.stop_batch,
            stage_a=stage_a,
            stage_c=stage_c,
            retrieval_count=len(retrieval_results["results"]),
            placement_emission_count=0,
            eligible=False,
        )
    first_emission = case_dir / "catalog_placement_first_emission.json"
    write_exclusive(first_emission, stage_c.content)
    try:
        validate_placement(
            loads_strict(stage_c.content.decode("utf-8", errors="strict")),
            plan=plan,
            retrieval_results=retrieval_results,
            brief=brief,
        )
    except (UnicodeError, StrictJSONError, ContractError) as exc:
        write_json_exclusive(
            case_dir / "placement_validation.json",
            {"valid": False, "error_type": type(exc).__name__, "error_message": str(exc)},
        )
        return _finalize_case(
            case_dir=case_dir,
            model=model,
            brief=brief,
            status="placement_schema_invalid",
            reason=str(exc),
            stop_batch=False,
            stage_a=stage_a,
            stage_c=stage_c,
            retrieval_count=len(retrieval_results["results"]),
            placement_emission_count=1,
            eligible=False,
        )
    write_json_exclusive(case_dir / "placement_validation.json", {"valid": True})
    frozen_placement = case_dir / "catalog_placement_v1.json"
    write_exclusive(frozen_placement, first_emission.read_bytes())
    if sha256_file(first_emission) != sha256_file(frozen_placement):
        raise ArtifactError("placement byte-copy hash mismatch")
    return _finalize_case(
        case_dir=case_dir,
        model=model,
        brief=brief,
        status="complete",
        reason=None,
        stop_batch=False,
        stage_a=stage_a,
        stage_c=stage_c,
        retrieval_count=len(retrieval_results["results"]),
        placement_emission_count=1,
        eligible=True,
    )


def initialize_run(
    *,
    output_root: Path,
    model: ModelConfig,
    briefs_path: Path,
    models_path: Path,
    retriever: RetrieverAdapter,
    source_manifest: Any | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    stage_a_prompt = DEFAULT_STAGE_A_PROMPT.read_bytes()
    stage_c_prompt = DEFAULT_STAGE_C_PROMPT.read_bytes()
    write_json_exclusive(
        output_root / "run_manifest.json",
        {
            "schema_version": "hy34_two_stage_run_manifest_v2",
            "runner_version": RUNNER_VERSION,
            "created_at": utc_now(),
            "model": model.public_dict(),
            "model_identity_sent_in_prompt": False,
            "briefs_sha256": sha256_file(briefs_path),
            "models_config_sha256": sha256_file(models_path),
            "stage_a_prompt_sha256": sha256_bytes(stage_a_prompt),
            "stage_c_prompt_sha256": sha256_bytes(stage_c_prompt),
            "retriever_runtime_root": str(retriever.runtime_root),
            "retriever_gate_status": retriever.gate_report["status"],
            "retriever_gate_warning_codes": sorted(
                {item["code"] for item in retriever.gate_report["warnings"]}
            ),
            "state_machine": "stage_a_once -> top1_once_per_slot -> stage_c_once -> terminal",
            "concurrency": "none; briefs run sequentially",
            "source_manifest": _resolve_source_manifest(source_manifest),
        },
    )


def verify_resume(
    *,
    output_root: Path,
    model: ModelConfig,
    briefs_path: Path,
    models_path: Path,
    source_manifest: Any | None = None,
) -> None:
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError("resume requires an existing run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "runner_version": RUNNER_VERSION,
        "model_key": model.key,
        "briefs_sha256": sha256_file(briefs_path),
        "models_config_sha256": sha256_file(models_path),
        "stage_a_prompt_sha256": sha256_file(DEFAULT_STAGE_A_PROMPT),
        "stage_c_prompt_sha256": sha256_file(DEFAULT_STAGE_C_PROMPT),
        "source_manifest_sha256": _resolve_source_manifest(source_manifest)[
            "manifest_sha256"
        ],
    }
    actual = {
        "runner_version": manifest.get("runner_version"),
        "model_key": (manifest.get("model") or {}).get("key"),
        "briefs_sha256": manifest.get("briefs_sha256"),
        "models_config_sha256": manifest.get("models_config_sha256"),
        "stage_a_prompt_sha256": manifest.get("stage_a_prompt_sha256"),
        "stage_c_prompt_sha256": manifest.get("stage_c_prompt_sha256"),
        "source_manifest_sha256": (manifest.get("source_manifest") or {}).get(
            "manifest_sha256"
        ),
    }
    if actual != expected:
        raise ArtifactError(f"resume provenance mismatch: expected={expected} actual={actual}")


def run_model(
    *,
    model_key: str,
    output_root: Path,
    briefs_path: Path,
    models_path: Path,
    retriever_root: Path,
    selected_briefs: set[str] | None = None,
    resume: bool = False,
    provider_route: Any | None = None,
    retry_policy: Any | None = None,
    source_manifest: Any | None = None,
) -> tuple[dict[str, Any], bool]:
    model = _load_model_config(models_path, model_key)
    briefs = _load_briefs(briefs_path)
    if selected_briefs is not None:
        unknown = selected_briefs - {item["brief_id"] for item in briefs}
        if unknown:
            raise ValueError(f"unknown selected briefs: {sorted(unknown)}")
        briefs = [item for item in briefs if item["brief_id"] in selected_briefs]
    retriever = RetrieverAdapter(retriever_root)
    if resume:
        if not output_root.is_dir():
            raise ArtifactError("resume output root does not exist")
        verify_resume(
            output_root=output_root,
            model=model,
            briefs_path=briefs_path,
            models_path=models_path,
            source_manifest=source_manifest,
        )
        if (output_root / "summary.json").exists():
            raise ArtifactError("run already has a terminal summary; refusing resume")
    else:
        initialize_run(
            output_root=output_root,
            model=model,
            briefs_path=briefs_path,
            models_path=models_path,
            retriever=retriever,
            source_manifest=source_manifest,
        )
    stage_a_prompt = DEFAULT_STAGE_A_PROMPT.read_text(encoding="utf-8")
    stage_c_prompt = DEFAULT_STAGE_C_PROMPT.read_text(encoding="utf-8")
    results: list[dict[str, Any]] = []
    stopped = False
    for brief in briefs:
        case_dir = output_root / str(brief["brief_id"])
        if case_dir.exists():
            result_path = case_dir / "case.result.json"
            if not resume or not result_path.is_file():
                raise ArtifactError(
                    f"existing non-terminal case cannot be resent safely: {case_dir}"
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result.get("brief_id") != brief["brief_id"]
                or result.get("model_key") != model.key
            ):
                raise ArtifactError(f"resume case identity mismatch: {case_dir}")
            results.append(result)
            continue
        result = run_case(
            output_root=output_root,
            model=model,
            brief=brief,
            retriever=retriever,
            stage_a_prompt=stage_a_prompt,
            stage_c_prompt=stage_c_prompt,
            provider_route=provider_route,
            retry_policy=retry_policy,
        )
        results.append(result)
        if result["stop_batch"]:
            stopped = True
            break
    summary = {
        "schema_version": "hy34_two_stage_run_summary_v2",
        "model_key": model.key,
        "model_label": model.label,
        "requested_briefs": len(briefs),
        "processed_briefs": len(results),
        "complete": sum(item["status"] == "complete" for item in results),
        "failed": sum(item["status"] != "complete" for item in results),
        "eligible": sum(item["eligible_for_strict_one_shot_evaluation"] for item in results),
        "stopped_early": stopped,
        "results": results,
        "completed_at": utc_now(),
    }
    write_json_exclusive(output_root / "summary.json", summary)
    return summary, stopped


def check_runner(
    *,
    briefs_path: Path,
    models_path: Path,
    retriever_root: Path | None,
    source_manifest: Any | None = None,
) -> dict[str, Any]:
    briefs = _load_briefs(briefs_path)
    model_keys = list(json.loads(models_path.read_text(encoding="utf-8"))["models"])
    models = [_load_model_config(models_path, key) for key in model_keys]
    if not models:
        raise ValueError("model config must expose at least one model")
    if len(models) > 1 and len({canonical_json_bytes({**model.public_dict(), "key": None, "label": None, "configured_model": None, "wire_model": None}) for model in models}) != 1:
        raise ValueError("model request policies differ beyond model identity")
    report: dict[str, Any] = {
        "valid": True,
        "runner_version": RUNNER_VERSION,
        "brief_count": len(briefs),
        "brief_ids": [item["brief_id"] for item in briefs],
        "model_keys": model_keys,
        "stage_a_prompt_sha256": sha256_file(DEFAULT_STAGE_A_PROMPT),
        "stage_c_prompt_sha256": sha256_file(DEFAULT_STAGE_C_PROMPT),
        "source_manifest": _resolve_source_manifest(source_manifest),
        "retriever_gate": None,
    }
    if retriever_root is not None:
        retriever = RetrieverAdapter(retriever_root)
        report["retriever_gate"] = {
            "status": retriever.gate_report["status"],
            "errors": retriever.gate_report["errors"],
            "warning_codes": sorted(
                {item["code"] for item in retriever.gate_report["warnings"]}
            ),
            "golden_top1_matches": sum(
                item["actual_jid"] == item["expected_jid"]
                for item in retriever.gate_report["golden_results"]
            ),
            "golden_count": len(retriever.gate_report["golden_results"]),
        }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--briefs", type=Path, default=DEFAULT_BRIEFS)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--retriever-root", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--model",
        required=True,
        choices=(
            "api3-claude-opus-4-8",
            "api3-claude-sonnet-5",
            "api3-claude-opus-5",
            "api3-claude-fable-5",
        ),
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--retriever-root", type=Path, default=DEFAULT_RETRIEVER_ROOT)
    run.add_argument("--brief-id", action="append", dest="brief_ids")
    run.add_argument(
        "--resume",
        action="store_true",
        help="skip terminal cases only; never resend an incomplete case",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "check":
            print(
                json.dumps(
                    check_runner(
                        briefs_path=args.briefs,
                        models_path=args.models,
                        retriever_root=args.retriever_root,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        summary, stopped = run_model(
            model_key=args.model,
            output_root=args.output_dir,
            briefs_path=args.briefs,
            models_path=args.models,
            retriever_root=args.retriever_root,
            selected_briefs=None if not args.brief_ids else set(args.brief_ids),
            resume=bool(args.resume),
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 2 if stopped or summary["failed"] else 0
    except (
        ArtifactError,
        ContractError,
        EndpointError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
