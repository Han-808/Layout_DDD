"""HY4-only online execution with immutable per-attempt provenance capture."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactError,
    request_json_bytes,
    sha256_bytes,
    write_exclusive,
    write_json_exclusive,
)
from . import __version__
from .constants import (
    ARTIFACT_FORMAT_VERSION,
    DEFAULT_RETRY_DELAY_SECONDS,
    LAYOUT_SCHEMA_VERSION,
    MIN_VISIBLE_OUTPUT_TOKENS,
    PROMPT_VERSION,
    RUN_MANIFEST_VERSION,
)
from .client_config import CLIENT_CONFIG_PATH
from .inputs import InputBatch, SceneInput
from .prompt import build_system_prompt, build_user_prompt, protocol_text
from .strict_json import StrictJSONError, loads_strict_bytes
from .transport import post_once


RUNNER_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = RUNNER_ROOT / "schema" / "layout.schema.json"
RETRYABLE_ATTEMPT_STATUSES = frozenset(
    {"transport_failure", "http_error", "invalid_api_response"}
)
ATTEMPT_STATUSES = RETRYABLE_ATTEMPT_STATUSES | {
    "captured",
    "short_output",
    "token_count_unavailable",
    "transport_ambiguous",
}
SCENE_STATUSES = frozenset(
    {
        "captured",
        "retry_exhausted",
        "token_count_unavailable",
        "transport_ambiguous",
    }
)


def _utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_now() -> str:
    return _format_utc(_utc_datetime())


@dataclass(frozen=True)
class RunConfig:
    endpoint: str
    configured_model: str = "openai/Hy4-T3-A49B-DSA-1M-SFT0730-Opus5"
    wire_model: str = "Hy4-T3-A49B-DSA-1M-SFT0730-Opus5"
    api_key: str = "EMPTY"
    timeout_seconds: float = 1800.0
    max_retries: int = 2
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 65536
    repetition_penalty: float = 1.0
    reasoning_effort: str = "high"
    preserved_thinking: bool = True
    strategy_type: str = "ConsistentHash"
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS

    def public_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "runner_version": __version__,
            "configured_client": "litellm",
            "transport_implementation": "auditable_direct_openai_compatible",
            "configured_model": self.configured_model,
            "wire_model": self.wire_model,
            "api_key": self.api_key,
            "max_tokens": self.max_tokens,
            "stream": False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "chat_template_kwargs": {
                "reasoning_effort": self.reasoning_effort,
                "preserved_thinking": self.preserved_thinking,
            },
            "extra_headers": {
                "SessionID": "dynamic unique UUID per actual HTTP request",
                "StrategyType": self.strategy_type,
            },
            "message_roles": ["system", "user"],
            "system_message": True,
            "system_message_source": "frozen prompt_protocol.txt",
            "user_message": "exact SceneEval Description only",
            "scene_id_sent_to_model": False,
            "examples": False,
            "constrained_decoding": False,
            "automatic_retry": True,
            "max_retries": self.max_retries,
            "max_consecutive_infrastructure_failures": self.max_retries + 1,
            "short_output_retry_limit": "unbounded",
            "minimum_visible_output_tokens": MIN_VISIBLE_OUTPUT_TOKENS,
            "visible_output_token_formula": (
                "usage.completion_tokens - "
                "usage.completion_tokens_details.reasoning_tokens"
            ),
            "retry_delay_seconds": self.retry_delay_seconds,
            "retry_attempt_statuses": sorted(
                RETRYABLE_ATTEMPT_STATUSES | {"short_output"}
            ),
            "never_retry_attempt_statuses": [
                "captured",
                "token_count_unavailable",
                "transport_ambiguous",
            ],
            "model_content_validation": False,
            "reasoning_channel_normalization": False,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class AttemptResult:
    scene_id: int
    attempt_number: int
    status: str
    request_sha256: str
    session_id: str | None
    response_sha256: str | None
    raw_content_sha256: str | None
    x_request_id: str | None


@dataclass(frozen=True)
class SceneResult:
    scene_id: int
    status: str
    stop_batch: bool
    attempt_count: int


def _runner_source_provenance() -> dict[str, Any]:
    source_paths = [RUNNER_ROOT / "run.py"]
    source_paths.extend(sorted((RUNNER_ROOT / "sceneeval_hy4").glob("*.py")))
    files: list[dict[str, Any]] = []
    for path in source_paths:
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(RUNNER_ROOT).as_posix(),
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )
    payload = {
        "runner_version": __version__,
        "files": files,
    }
    return {
        **payload,
        "source_manifest_sha256": sha256_bytes(request_json_bytes(payload)),
    }


def _request_value(scene: SceneInput, config: RunConfig) -> dict[str, Any]:
    return {
        "model": config.wire_model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(scene.description)},
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "chat_template_kwargs": {
            "reasoning_effort": config.reasoning_effort,
            "preserved_thinking": config.preserved_thinking,
        },
        "stream": False,
    }


def _request_headers(config: RunConfig, session_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": f"sceneeval-hy4-online-capture-runner/{__version__}",
        "Authorization": f"Bearer {config.api_key}",
        "SessionID": session_id,
        "StrategyType": config.strategy_type,
    }


def _save_attempt_result(
    attempt_dir: Path,
    *,
    scene: SceneInput,
    attempt_number: int,
    status: str,
    request_sha256: str,
    started_at: str,
    detail: dict[str, Any],
) -> AttemptResult:
    if status not in ATTEMPT_STATUSES:
        raise ArtifactError(f"unrecognized attempt status: {status}")
    write_json_exclusive(
        attempt_dir / "attempt.result.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "scene_id": scene.scene_id,
            "attempt_number": attempt_number,
            "status": status,
            "started_at": started_at,
            "completed_at": utc_now(),
            "request_sha256": request_sha256,
            "detail": detail,
        },
    )
    return AttemptResult(
        scene_id=scene.scene_id,
        attempt_number=attempt_number,
        status=status,
        request_sha256=request_sha256,
        session_id=(
            detail.get("session_id")
            if isinstance(detail.get("session_id"), str)
            else None
        ),
        response_sha256=(
            detail.get("response_sha256")
            if isinstance(detail.get("response_sha256"), str)
            else None
        ),
        raw_content_sha256=(
            detail.get("raw_content_sha256")
            if isinstance(detail.get("raw_content_sha256"), str)
            else None
        ),
        x_request_id=(
            detail.get("x_request_id")
            if isinstance(detail.get("x_request_id"), str)
            else None
        ),
    )


def _save_scene_result(
    scene_dir: Path,
    *,
    scene: SceneInput,
    status: str,
    attempt_statuses: list[str],
    accepted_attempt: AttemptResult | None = None,
) -> SceneResult:
    if status not in SCENE_STATUSES:
        raise ArtifactError(f"unrecognized scene status: {status}")
    stop_batch = status in {"transport_ambiguous", "token_count_unavailable"}
    if status == "captured" and accepted_attempt is None:
        raise ArtifactError("captured scene result requires accepted attempt provenance")
    if status != "captured" and accepted_attempt is not None:
        raise ArtifactError("non-captured scene result cannot have an accepted attempt")
    write_json_exclusive(
        scene_dir / "scene.result.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "scene_id": scene.scene_id,
            "status": status,
            "completed_at": utc_now(),
            "attempt_count": len(attempt_statuses),
            "attempt_statuses": attempt_statuses,
            "accepted_attempt_number": (
                None if accepted_attempt is None else accepted_attempt.attempt_number
            ),
            "accepted_request_sha256": (
                None if accepted_attempt is None else accepted_attempt.request_sha256
            ),
            "accepted_response_sha256": (
                None if accepted_attempt is None else accepted_attempt.response_sha256
            ),
            "accepted_raw_content_sha256": (
                None
                if accepted_attempt is None
                else accepted_attempt.raw_content_sha256
            ),
            "accepted_session_id": (
                None if accepted_attempt is None else accepted_attempt.session_id
            ),
            "accepted_x_request_id": (
                None if accepted_attempt is None else accepted_attempt.x_request_id
            ),
            "retry_performed": len(attempt_statuses) > 1,
            "short_output_retry_limit": "unbounded",
            "minimum_visible_output_tokens": MIN_VISIBLE_OUTPUT_TOKENS,
            "stop_batch": stop_batch,
            "model_content_validation_performed": False,
        },
    )
    return SceneResult(
        scene_id=scene.scene_id,
        status=status,
        stop_batch=stop_batch,
        attempt_count=len(attempt_statuses),
    )


def _extract_message(api_value: Any) -> tuple[str, str | None, str | None]:
    if not isinstance(api_value, dict):
        raise ValueError("API response top-level value must be an object")
    choices = api_value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("API response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("API choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("API choice.message must be an object")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("API choice.message.content must be a string")
    reasoning = message.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("API reasoning must be a string or null")
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ValueError("API reasoning_content must be a string or null")
    return content, reasoning, reasoning_content


class TokenCountUnavailable(ValueError):
    pass


def _visible_output_tokens(api_value: Any) -> tuple[int, int, int]:
    if not isinstance(api_value, dict):
        raise TokenCountUnavailable("API response top-level value is not an object")
    usage = api_value.get("usage")
    if not isinstance(usage, dict):
        raise TokenCountUnavailable("usage must be an object")
    completion_tokens = usage.get("completion_tokens")
    details = usage.get("completion_tokens_details")
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 0
    ):
        raise TokenCountUnavailable(
            "usage.completion_tokens must be a non-negative integer"
        )
    if not isinstance(details, dict):
        raise TokenCountUnavailable(
            "usage.completion_tokens_details must be an object"
        )
    reasoning_tokens = details.get("reasoning_tokens")
    if (
        isinstance(reasoning_tokens, bool)
        or not isinstance(reasoning_tokens, int)
        or reasoning_tokens < 0
    ):
        raise TokenCountUnavailable(
            "usage.completion_tokens_details.reasoning_tokens must be a "
            "non-negative integer"
        )
    if reasoning_tokens > completion_tokens:
        raise TokenCountUnavailable(
            "reasoning_tokens cannot exceed completion_tokens"
        )
    return (
        completion_tokens - reasoning_tokens,
        completion_tokens,
        reasoning_tokens,
    )


def _optional_text_metadata(value: str | None) -> tuple[str, str | None, int | None]:
    if value is None:
        return "null", None, None
    value_bytes = value.encode("utf-8", errors="strict")
    if value == "":
        return "empty_string", sha256_bytes(value_bytes), 0
    return "nonempty_string", sha256_bytes(value_bytes), len(value_bytes)


def _run_attempt(
    scene: SceneInput,
    scene_dir: Path,
    config: RunConfig,
    *,
    attempt_number: int,
    retry_of_status: str | None,
) -> AttemptResult:
    attempt_dir = scene_dir / f"attempt_{attempt_number:02d}"
    try:
        attempt_dir.mkdir(mode=0o750)
    except FileExistsError as exc:
        raise ArtifactError(
            f"attempt artifact directory already exists: {attempt_dir}"
        ) from exc

    request_value = _request_value(scene, config)
    request_body = request_json_bytes(request_value)
    request_sha256 = sha256_bytes(request_body)
    session_id = str(uuid.uuid4())
    request_headers = _request_headers(config, session_id)
    request_headers_bytes = request_json_bytes(request_headers)
    request_headers_sha256 = sha256_bytes(request_headers_bytes)
    started_at = utc_now()

    write_exclusive(attempt_dir / "request.json", request_body)
    write_json_exclusive(
        attempt_dir / "request-headers.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "scene_id": scene.scene_id,
            "attempt_number": attempt_number,
            "headers": list(request_headers.items()),
            "session_id": session_id,
            "strategy_type": config.strategy_type,
            "request_headers_sha256": request_headers_sha256,
        },
    )
    write_json_exclusive(
        attempt_dir / "attempt.started.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "scene_id": scene.scene_id,
            "attempt_number": attempt_number,
            "started_at": started_at,
            "request_sha256": request_sha256,
            "request_headers_sha256": request_headers_sha256,
            "session_id": session_id,
            "endpoint": config.endpoint,
            "configured_model": config.configured_model,
            "wire_model": config.wire_model,
            "automatic_retry": attempt_number > 1,
            "retry_of_status": retry_of_status,
        },
    )

    transport = post_once(
        config.endpoint,
        request_body,
        connect_timeout=config.timeout_seconds,
        read_timeout=config.timeout_seconds,
        request_headers=request_headers,
    )
    transport_detail = {
        "stage": transport.stage,
        "elapsed_seconds": round(transport.elapsed_seconds, 6),
        "http_status": transport.http_status,
        "http_reason": transport.http_reason,
        "error_type": transport.error_type,
        "error_message": transport.error_message,
        "request_headers_sha256": request_headers_sha256,
        "session_id": session_id,
    }

    x_request_id = None
    if transport.response_headers is not None:
        for name, value in transport.response_headers:
            if name.lower() == "x-request-id":
                x_request_id = value
        write_json_exclusive(
            attempt_dir / "response-headers.json",
            {
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "scene_id": scene.scene_id,
                "attempt_number": attempt_number,
                "http_status": transport.http_status,
                "http_reason": transport.http_reason,
                "headers": transport.response_headers,
                "x_request_id": x_request_id,
            },
        )
        transport_detail["x_request_id"] = x_request_id

    if transport.status != "response":
        return _save_attempt_result(
            attempt_dir,
            scene=scene,
            attempt_number=attempt_number,
            status=transport.status,
            request_sha256=request_sha256,
            started_at=started_at,
            detail=transport_detail,
        )

    assert transport.response_body is not None
    response_body = transport.response_body
    response_sha256 = sha256_bytes(response_body)
    write_exclusive(attempt_dir / "api-response.body", response_body)
    transport_detail["response_sha256"] = response_sha256

    if transport.http_status is None or not 200 <= transport.http_status < 300:
        write_json_exclusive(
            attempt_dir / "capture.json",
            {
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "scene_id": scene.scene_id,
                "attempt_number": attempt_number,
                "status": "http_error",
                "response_sha256": response_sha256,
                "x_request_id": x_request_id,
                "model_content_extracted": False,
                "model_content_validation_performed": False,
                "reasoning_channel_normalization_performed": False,
                "repair_applied": False,
            },
        )
        return _save_attempt_result(
            attempt_dir,
            scene=scene,
            attempt_number=attempt_number,
            status="http_error",
            request_sha256=request_sha256,
            started_at=started_at,
            detail=transport_detail,
        )

    try:
        api_value = loads_strict_bytes(response_body)
        content, reasoning, reasoning_content = _extract_message(api_value)
        content_bytes = content.encode("utf-8", errors="strict")
        reasoning_bytes = (
            None if reasoning is None else reasoning.encode("utf-8", errors="strict")
        )
        reasoning_content_bytes = (
            None
            if reasoning_content is None
            else reasoning_content.encode("utf-8", errors="strict")
        )
    except (StrictJSONError, ValueError, UnicodeError) as exc:
        write_json_exclusive(
            attempt_dir / "capture.json",
            {
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "scene_id": scene.scene_id,
                "attempt_number": attempt_number,
                "status": "invalid_api_response",
                "response_sha256": response_sha256,
                "x_request_id": x_request_id,
                "model_content_extracted": False,
                "api_envelope_error_type": type(exc).__name__,
                "api_envelope_error_message": str(exc),
                "model_content_validation_performed": False,
                "reasoning_channel_normalization_performed": False,
                "repair_applied": False,
            },
        )
        transport_detail["api_envelope_error_type"] = type(exc).__name__
        transport_detail["api_envelope_error_message"] = str(exc)
        return _save_attempt_result(
            attempt_dir,
            scene=scene,
            attempt_number=attempt_number,
            status="invalid_api_response",
            request_sha256=request_sha256,
            started_at=started_at,
            detail=transport_detail,
        )

    token_count_error: TokenCountUnavailable | None = None
    try:
        visible_output_tokens, completion_tokens, reasoning_tokens = (
            _visible_output_tokens(api_value)
        )
    except TokenCountUnavailable as exc:
        token_count_error = exc
        visible_output_tokens = None
        completion_tokens = None
        reasoning_tokens = None

    write_exclusive(attempt_dir / "raw-content.txt", content_bytes)
    raw_content_sha256 = sha256_bytes(content_bytes)
    if reasoning_bytes is not None:
        write_exclusive(attempt_dir / "logs" / "reasoning.txt", reasoning_bytes)
    if reasoning_content_bytes is not None:
        write_exclusive(
            attempt_dir / "logs" / "reasoning_content.txt",
            reasoning_content_bytes,
        )

    api_reasoning_state, api_reasoning_sha256, api_reasoning_bytes = (
        _optional_text_metadata(reasoning)
    )
    (
        api_reasoning_content_state,
        api_reasoning_content_sha256,
        api_reasoning_content_bytes,
    ) = _optional_text_metadata(reasoning_content)

    if token_count_error is not None:
        attempt_status = "token_count_unavailable"
    elif (
        visible_output_tokens is not None
        and visible_output_tokens < MIN_VISIBLE_OUTPUT_TOKENS
    ):
        attempt_status = "short_output"
    else:
        attempt_status = "captured"

    write_json_exclusive(
        attempt_dir / "capture.json",
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "scene_id": scene.scene_id,
            "attempt_number": attempt_number,
            "status": attempt_status,
            "response_sha256": response_sha256,
            "x_request_id": x_request_id,
            "model_content_extracted": True,
            "raw_content_source": "choices[0].message.content",
            "raw_content_bytes": len(content_bytes),
            "raw_content_sha256": raw_content_sha256,
            "api_reasoning_state": api_reasoning_state,
            "api_reasoning_sha256": api_reasoning_sha256,
            "api_reasoning_bytes": api_reasoning_bytes,
            "api_reasoning_content_state": api_reasoning_content_state,
            "api_reasoning_content_sha256": api_reasoning_content_sha256,
            "api_reasoning_content_bytes": api_reasoning_content_bytes,
            "reasoning_fields_compared": False,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "visible_output_tokens": visible_output_tokens,
            "minimum_visible_output_tokens": MIN_VISIBLE_OUTPUT_TOKENS,
            "visible_output_token_formula": (
                "usage.completion_tokens - "
                "usage.completion_tokens_details.reasoning_tokens"
            ),
            "token_count_error_type": (
                None if token_count_error is None else type(token_count_error).__name__
            ),
            "token_count_error_message": (
                None if token_count_error is None else str(token_count_error)
            ),
            "model_content_json_parse_performed": False,
            "model_content_schema_validation_performed": False,
            "model_content_validation_performed": False,
            "reasoning_channel_normalization_performed": False,
            "repair_applied": False,
        },
    )

    transport_detail.update(
        {
            "raw_content_bytes": len(content_bytes),
            "raw_content_sha256": raw_content_sha256,
            "api_reasoning_state": api_reasoning_state,
            "api_reasoning_content_state": api_reasoning_content_state,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "visible_output_tokens": visible_output_tokens,
            "minimum_visible_output_tokens": MIN_VISIBLE_OUTPUT_TOKENS,
            "token_count_error_type": (
                None if token_count_error is None else type(token_count_error).__name__
            ),
            "token_count_error_message": (
                None if token_count_error is None else str(token_count_error)
            ),
            "model_content_validation_performed": False,
            "reasoning_channel_normalization_performed": False,
        }
    )
    return _save_attempt_result(
        attempt_dir,
        scene=scene,
        attempt_number=attempt_number,
        status=attempt_status,
        request_sha256=request_sha256,
        started_at=started_at,
        detail=transport_detail,
    )


def _read_attempt_or_finalize(
    scene: SceneInput,
    attempt_dir: Path,
    attempt_number: int,
) -> AttemptResult:
    started_path = attempt_dir / "attempt.started.json"
    request_path = attempt_dir / "request.json"
    request_headers_path = attempt_dir / "request-headers.json"
    result_path = attempt_dir / "attempt.result.json"
    if (
        not started_path.is_file()
        or not request_path.is_file()
        or not request_headers_path.is_file()
    ):
        raise ArtifactError(f"incomplete or unrecognized attempt artifacts: {attempt_dir}")

    try:
        started = json.loads(started_path.read_text("utf-8"))
        request_sha256 = sha256_bytes(request_path.read_bytes())
        request_headers_record = json.loads(request_headers_path.read_text("utf-8"))
        header_pairs = request_headers_record["headers"]
        request_headers = dict(header_pairs)
        request_headers_sha256 = sha256_bytes(request_json_bytes(request_headers))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read attempt provenance in {attempt_dir}: {exc}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid request header provenance in {attempt_dir}: {exc}") from exc

    if (
        started.get("scene_id") != scene.scene_id
        or started.get("attempt_number") != attempt_number
        or started.get("request_sha256") != request_sha256
        or started.get("request_headers_sha256") != request_headers_sha256
        or started.get("session_id") != request_headers.get("SessionID")
        or request_headers_record.get("request_headers_sha256")
        != request_headers_sha256
    ):
        raise ArtifactError(f"attempt provenance mismatch: {attempt_dir}")

    if not result_path.exists():
        return _save_attempt_result(
            attempt_dir,
            scene=scene,
            attempt_number=attempt_number,
            status="transport_ambiguous",
            request_sha256=request_sha256,
            started_at=str(started.get("started_at", "unknown")),
            detail={
                "stage": "runner_interrupted",
                "elapsed_seconds": None,
                "http_status": None,
                "error_type": "InterruptedAttempt",
                "error_message": (
                    "attempt.started.json exists without attempt.result.json; delivery "
                    "cannot be established, so this request was not resent"
                ),
            },
        )

    try:
        result = json.loads(result_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read attempt result {result_path}: {exc}") from exc
    status = result.get("status")
    detail = result.get("detail")
    if (
        result.get("scene_id") != scene.scene_id
        or result.get("attempt_number") != attempt_number
        or result.get("request_sha256") != request_sha256
        or status not in ATTEMPT_STATUSES
        or not isinstance(detail, dict)
    ):
        raise ArtifactError(f"attempt result provenance mismatch: {result_path}")
    return AttemptResult(
        scene_id=scene.scene_id,
        attempt_number=attempt_number,
        status=str(status),
        request_sha256=request_sha256,
        session_id=(
            detail.get("session_id")
            if isinstance(detail.get("session_id"), str)
            else None
        ),
        response_sha256=(
            detail.get("response_sha256")
            if isinstance(detail.get("response_sha256"), str)
            else None
        ),
        raw_content_sha256=(
            detail.get("raw_content_sha256")
            if isinstance(detail.get("raw_content_sha256"), str)
            else None
        ),
        x_request_id=(
            detail.get("x_request_id")
            if isinstance(detail.get("x_request_id"), str)
            else None
        ),
    )


def _retry_not_before(
    scene: SceneInput,
    scene_dir: Path,
    config: RunConfig,
    previous_attempt_number: int,
    previous_status: str,
    next_attempt_number: int,
    *,
    create_if_missing: bool,
) -> datetime:
    schedule_path = scene_dir / f"retry_attempt_{next_attempt_number:02d}.scheduled.json"
    if schedule_path.exists():
        try:
            value = json.loads(schedule_path.read_text("utf-8"))
            not_before = datetime.fromisoformat(
                str(value["not_before_at"]).replace("Z", "+00:00")
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read retry schedule {schedule_path}: {exc}") from exc
        if (
            value.get("scene_id") != scene.scene_id
            or value.get("from_attempt_number") != previous_attempt_number
            or value.get("from_attempt_status") != previous_status
            or value.get("to_attempt_number") != next_attempt_number
            or value.get("delay_seconds") != config.retry_delay_seconds
            or not_before.tzinfo is None
        ):
            raise ArtifactError(f"retry schedule provenance mismatch: {schedule_path}")
        return not_before

    if not create_if_missing:
        raise ArtifactError(
            f"attempt_{next_attempt_number:02d} exists without retry schedule: "
            f"{scene_dir}"
        )

    scheduled_at = _utc_datetime()
    not_before = scheduled_at + timedelta(seconds=config.retry_delay_seconds)
    write_json_exclusive(
        schedule_path,
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "scene_id": scene.scene_id,
            "from_attempt_number": previous_attempt_number,
            "from_attempt_status": previous_status,
            "to_attempt_number": next_attempt_number,
            "scheduled_at": _format_utc(scheduled_at),
            "delay_seconds": config.retry_delay_seconds,
            "not_before_at": _format_utc(not_before),
        },
    )
    return not_before


def _wait_until(not_before: datetime) -> None:
    remaining = (not_before - _utc_datetime()).total_seconds()
    if remaining > 0:
        time.sleep(remaining)


def _continue_scene(
    scene: SceneInput,
    scene_dir: Path,
    config: RunConfig,
) -> SceneResult:
    statuses: list[str] = []
    consecutive_infrastructure_failures = 0
    attempt_number = 1
    while True:
        attempt_dir = scene_dir / f"attempt_{attempt_number:02d}"
        retry_of_status = None if attempt_number == 1 else statuses[-1]

        if attempt_number == 1:
            if attempt_dir.exists():
                attempt = _read_attempt_or_finalize(scene, attempt_dir, attempt_number)
            else:
                if any(scene_dir.iterdir()):
                    raise ArtifactError(
                        f"scene directory has artifacts but no attempt_01: {scene_dir}"
                    )
                attempt = _run_attempt(
                    scene,
                    scene_dir,
                    config,
                    attempt_number=attempt_number,
                    retry_of_status=None,
                )
        else:
            if retry_of_status not in (
                RETRYABLE_ATTEMPT_STATUSES | {"short_output"}
            ):
                raise ArtifactError(
                    f"attempt_{attempt_number:02d} cannot follow "
                    f"{retry_of_status!r}"
                )
            not_before = _retry_not_before(
                scene,
                scene_dir,
                config,
                attempt_number - 1,
                retry_of_status,
                attempt_number,
                create_if_missing=not attempt_dir.exists(),
            )
            if attempt_dir.exists():
                attempt = _read_attempt_or_finalize(scene, attempt_dir, attempt_number)
            else:
                _wait_until(not_before)
                attempt = _run_attempt(
                    scene,
                    scene_dir,
                    config,
                    attempt_number=attempt_number,
                    retry_of_status=retry_of_status,
                )

        statuses.append(attempt.status)
        if attempt.status == "captured":
            return _save_scene_result(
                scene_dir,
                scene=scene,
                status="captured",
                attempt_statuses=statuses,
                accepted_attempt=attempt,
            )
        if attempt.status in {
            "token_count_unavailable",
            "transport_ambiguous",
        }:
            return _save_scene_result(
                scene_dir,
                scene=scene,
                status=attempt.status,
                attempt_statuses=statuses,
            )
        if attempt.status == "short_output":
            consecutive_infrastructure_failures = 0
        elif attempt.status in RETRYABLE_ATTEMPT_STATUSES:
            consecutive_infrastructure_failures += 1
            if consecutive_infrastructure_failures > config.max_retries:
                return _save_scene_result(
                    scene_dir,
                    scene=scene,
                    status="retry_exhausted",
                    attempt_statuses=statuses,
                )
        else:
            raise ArtifactError(
                f"unrecognized attempt status at {attempt_number}: {attempt.status}"
            )

        attempt_number += 1


def run_scene(
    scene: SceneInput,
    output_root: Path,
    config: RunConfig,
) -> SceneResult:
    """Capture one scene with the configured auditable retry budget."""
    scene_dir = output_root / f"scene_{scene.scene_id:03d}"
    try:
        scene_dir.mkdir(mode=0o750)
    except FileExistsError as exc:
        raise ArtifactError(f"scene artifact directory already exists: {scene_dir}") from exc
    return _continue_scene(scene, scene_dir, config)


def _manifest(
    batch: InputBatch,
    config: RunConfig,
    runner_source: dict[str, Any],
) -> dict[str, Any]:
    schema_bytes = SCHEMA_PATH.read_bytes()
    prompt_bytes = (protocol_text() + "\n").encode("utf-8")
    client_config_bytes = CLIENT_CONFIG_PATH.read_bytes()
    return {
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "created_at": utc_now(),
        "scope": "SceneEval human-authored IDs 0-99; HY4 online generation only",
        "input": {
            "format": "strict JSONL with exactly id and description",
            "row_count": len(batch.scenes),
            "ids": [batch.scenes[0].scene_id, batch.scenes[-1].scene_id],
            "sha256": batch.sha256,
        },
        "protocol": {
            "prompt_version": PROMPT_VERSION,
            "prompt_protocol_sha256": sha256_bytes(prompt_bytes),
            "layout_schema_version": LAYOUT_SCHEMA_VERSION,
            "layout_schema_sha256": sha256_bytes(schema_bytes),
            "message_roles": ["system", "user"],
            "system_message_is_frozen_protocol": True,
            "user_message_is_exact_description_only": True,
            "scene_id_sent_to_model": False,
            "single_user_turn": True,
            "single_sample_per_attempt": True,
            "repair": False,
            "semantic_validation": False,
            "capture_only": True,
            "model_content_json_parse": False,
            "model_content_schema_validation": False,
            "reasoning_channel_normalization": False,
        },
        "configuration": config.public_dict(),
        "runner": {
            "version": runner_source["runner_version"],
            "source_file_count": len(runner_source["files"]),
            "source_manifest_sha256": runner_source["source_manifest_sha256"],
        },
        "client_config": {
            "source": "openai_clients.yaml",
            "sha256": sha256_bytes(client_config_bytes),
            "litellm_internal_retries_used": False,
            "runner_managed_auditable_retries": True,
        },
    }


def initialize_run(
    output_root: Path,
    batch: InputBatch,
    config: RunConfig,
) -> None:
    runner_source = _runner_source_provenance()
    output_root.mkdir(parents=True, mode=0o750, exist_ok=False)
    write_json_exclusive(
        output_root / "run-manifest.json",
        _manifest(batch, config, runner_source),
    )
    write_json_exclusive(
        output_root / "runner-source-manifest.json",
        runner_source,
    )
    write_exclusive(output_root / "input.snapshot.jsonl", batch.exact_bytes)
    write_exclusive(
        output_root / "prompt_protocol.txt",
        (protocol_text() + "\n").encode("utf-8"),
    )
    write_exclusive(output_root / "layout.schema.json", SCHEMA_PATH.read_bytes())
    write_exclusive(
        output_root / "openai_clients.yaml",
        CLIENT_CONFIG_PATH.read_bytes(),
    )


def verify_resume(
    output_root: Path,
    batch: InputBatch,
    config: RunConfig,
) -> None:
    if not output_root.is_dir():
        raise ArtifactError(f"resume output directory does not exist: {output_root}")
    try:
        manifest = json.loads((output_root / "run-manifest.json").read_text("utf-8"))
        snapshot = (output_root / "input.snapshot.jsonl").read_bytes()
        prompt_snapshot = (output_root / "prompt_protocol.txt").read_bytes()
        schema_snapshot = (output_root / "layout.schema.json").read_bytes()
        client_config_snapshot = (output_root / "openai_clients.yaml").read_bytes()
        runner_source_snapshot = json.loads(
            (output_root / "runner-source-manifest.json").read_text("utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read existing run provenance: {exc}") from exc

    if manifest.get("run_manifest_version") != RUN_MANIFEST_VERSION:
        raise ArtifactError("resume run-manifest version differs from this runner")
    if sha256_bytes(snapshot) != batch.sha256 or snapshot != batch.exact_bytes:
        raise ArtifactError("resume input differs from the immutable input snapshot")
    if manifest.get("configuration") != config.public_dict():
        raise ArtifactError("resume configuration differs from the immutable run manifest")

    current_prompt = (protocol_text() + "\n").encode("utf-8")
    current_schema = SCHEMA_PATH.read_bytes()
    current_client_config = CLIENT_CONFIG_PATH.read_bytes()
    current_runner_source = _runner_source_provenance()
    protocol_manifest = manifest.get("protocol")
    if not isinstance(protocol_manifest, dict):
        raise ArtifactError("run manifest is missing protocol provenance")
    if prompt_snapshot != current_prompt or schema_snapshot != current_schema:
        raise ArtifactError("current prompt/schema differs from the immutable run snapshots")
    if client_config_snapshot != current_client_config:
        raise ArtifactError(
            "current client YAML differs from the immutable run snapshot"
        )
    if protocol_manifest.get("prompt_protocol_sha256") != sha256_bytes(current_prompt):
        raise ArtifactError("prompt hash differs from the immutable run manifest")
    if protocol_manifest.get("layout_schema_sha256") != sha256_bytes(current_schema):
        raise ArtifactError("schema hash differs from the immutable run manifest")
    client_config_manifest = manifest.get("client_config")
    if not isinstance(client_config_manifest, dict):
        raise ArtifactError("run manifest is missing client YAML provenance")
    if client_config_manifest.get("sha256") != sha256_bytes(current_client_config):
        raise ArtifactError("client YAML hash differs from the immutable run manifest")
    runner_manifest = manifest.get("runner")
    if not isinstance(runner_manifest, dict):
        raise ArtifactError("run manifest is missing runner source provenance")
    if runner_source_snapshot != current_runner_source:
        raise ArtifactError(
            "current runner source differs from the immutable run snapshot"
        )
    if (
        runner_manifest.get("version") != current_runner_source["runner_version"]
        or runner_manifest.get("source_file_count")
        != len(current_runner_source["files"])
        or runner_manifest.get("source_manifest_sha256")
        != current_runner_source["source_manifest_sha256"]
    ):
        raise ArtifactError("runner source hash differs from the immutable run manifest")


def _read_scene_status(scene: SceneInput, scene_dir: Path) -> str | None:
    result_path = scene_dir / "scene.result.json"
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read existing scene result {result_path}: {exc}") from exc
    status = result.get("status")
    attempt_count = result.get("attempt_count")
    attempt_statuses = result.get("attempt_statuses")
    if (
        result.get("scene_id") != scene.scene_id
        or status not in SCENE_STATUSES
        or isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
        or not isinstance(attempt_statuses, list)
        or len(attempt_statuses) != attempt_count
        or any(item not in ATTEMPT_STATUSES for item in attempt_statuses)
    ):
        raise ArtifactError(f"existing scene result provenance mismatch: {result_path}")

    accepted_fields = {
        "accepted_attempt_number": result.get("accepted_attempt_number"),
        "accepted_request_sha256": result.get("accepted_request_sha256"),
        "accepted_response_sha256": result.get("accepted_response_sha256"),
        "accepted_raw_content_sha256": result.get("accepted_raw_content_sha256"),
        "accepted_session_id": result.get("accepted_session_id"),
        "accepted_x_request_id": result.get("accepted_x_request_id"),
    }
    if status != "captured":
        if any(value is not None for value in accepted_fields.values()):
            raise ArtifactError(
                f"non-captured scene has accepted attempt provenance: {result_path}"
            )
        return str(status)

    accepted_number = accepted_fields["accepted_attempt_number"]
    if (
        isinstance(accepted_number, bool)
        or not isinstance(accepted_number, int)
        or accepted_number < 1
        or accepted_number > attempt_count
        or attempt_statuses[accepted_number - 1] != "captured"
    ):
        raise ArtifactError(
            f"captured scene has invalid accepted attempt number: {result_path}"
        )
    attempt_path = (
        scene_dir / f"attempt_{accepted_number:02d}" / "attempt.result.json"
    )
    try:
        attempt_result = json.loads(attempt_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"cannot read accepted attempt result {attempt_path}: {exc}"
        ) from exc
    detail = attempt_result.get("detail")
    if (
        attempt_result.get("status") != "captured"
        or not isinstance(detail, dict)
        or accepted_fields["accepted_request_sha256"]
        != attempt_result.get("request_sha256")
        or accepted_fields["accepted_response_sha256"]
        != detail.get("response_sha256")
        or accepted_fields["accepted_raw_content_sha256"]
        != detail.get("raw_content_sha256")
        or accepted_fields["accepted_session_id"] != detail.get("session_id")
        or accepted_fields["accepted_x_request_id"] != detail.get("x_request_id")
    ):
        raise ArtifactError(
            f"accepted attempt provenance mismatch: {result_path}"
        )
    return str(status)


def summarize(output_root: Path) -> dict[str, Any]:
    scene_counts: dict[str, int] = {}
    completed_ids: list[int] = []
    for result_path in sorted(output_root.glob("scene_*/scene.result.json")):
        result = json.loads(result_path.read_text("utf-8"))
        status = str(result["status"])
        scene_counts[status] = scene_counts.get(status, 0) + 1
        completed_ids.append(int(result["scene_id"]))

    attempt_counts: dict[str, int] = {}
    request_attempt_count = 0
    for result_path in sorted(output_root.glob("scene_*/attempt_*/attempt.result.json")):
        result = json.loads(result_path.read_text("utf-8"))
        status = str(result["status"])
        attempt_counts[status] = attempt_counts.get(status, 0) + 1
        request_attempt_count += 1

    return {
        "scene_result_count": len(completed_ids),
        "scene_result_ids": completed_ids,
        "scene_status_counts": dict(sorted(scene_counts.items())),
        "request_attempt_count": request_attempt_count,
        "attempt_status_counts": dict(sorted(attempt_counts.items())),
        "complete_100": len(completed_ids) == 100,
    }


def _persist_execution_summary(
    output_root: Path,
    *,
    execution_started_at: str,
    stopped: bool,
    summary: dict[str, Any],
) -> dict[str, Any]:
    execution_id = str(uuid.uuid4())
    timestamp = _utc_datetime().strftime("%Y%m%dT%H%M%S.%fZ")
    relative_path = Path("execution-summaries") / (
        f"summary_{timestamp}_{execution_id}.json"
    )
    summary_with_path = {
        **summary,
        "execution_summary_file": relative_path.as_posix(),
    }
    write_json_exclusive(
        output_root / relative_path,
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "run_manifest_version": RUN_MANIFEST_VERSION,
            "runner_version": __version__,
            "execution_id": execution_id,
            "execution_started_at": execution_started_at,
            "execution_completed_at": utc_now(),
            "stopped": stopped,
            "terminal_reason": (
                "stopped_by_scene_status" if stopped else "input_pass_completed"
            ),
            "summary": summary_with_path,
        },
    )
    return summary_with_path


def run_batch(
    batch: InputBatch,
    output_root: Path,
    config: RunConfig,
    *,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    execution_started_at = utc_now()
    if resume:
        verify_resume(output_root, batch, config)
    else:
        initialize_run(output_root, batch, config)

    stopped = False
    for scene in batch.scenes:
        scene_dir = output_root / f"scene_{scene.scene_id:03d}"
        if scene_dir.exists():
            existing_status = _read_scene_status(scene, scene_dir)
            if existing_status is not None:
                continue
            result = _continue_scene(scene, scene_dir, config)
        else:
            result = run_scene(scene, output_root, config)
        print(
            json.dumps(
                {
                    "scene_id": result.scene_id,
                    "status": result.status,
                    "attempt_count": result.attempt_count,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        if result.stop_batch:
            stopped = True
            break

    summary = summarize(output_root)
    summary = _persist_execution_summary(
        output_root,
        execution_started_at=execution_started_at,
        stopped=stopped,
        summary=summary,
    )
    return summary, stopped
