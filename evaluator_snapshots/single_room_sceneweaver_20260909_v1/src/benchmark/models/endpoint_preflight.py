"""Repeated multimodal stability gate for OpenAI-compatible model routes."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import math
import mimetypes
from pathlib import Path
import threading
import time
from typing import Any, Callable

from benchmark.models.openai_compatible_model import (
    EndpointConfigurationError,
    OpenAICompatibleModel,
)


ENDPOINT_PREFLIGHT_SCHEMA_VERSION = "endpoint_stability_preflight_v1"


class EndpointStabilityPreflightError(RuntimeError):
    """Raised when a route fails the repeated multimodal stability gate."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = deepcopy(report)
        failures = list(report.get("failures") or [])
        first = failures[0] if failures else {}
        detail = first.get("error") or "one or more endpoint checks failed"
        super().__init__(
            "endpoint stability preflight failed: " + str(detail)
        )


def run_endpoint_stability_preflight(
    *,
    endpoint: str,
    model_id: str,
    api_key_env: str,
    image_path: Path,
    attempts: int = 10,
    concurrency: int = 2,
    required_successes: int | None = None,
    inter_attempt_sleep_seconds: float = 0.0,
    timeout_seconds: int = 300,
    max_tokens: int = 64,
    min_request_interval_seconds: float = 0.0,
    max_retries: int = 0,
    retry_backoff_seconds: float = 1.0,
    retry_backoff_mode: str = "linear",
    retry_all_http_errors: bool = False,
    retry_malformed_response: bool = False,
    model_factory: Callable[..., Any] = OpenAICompatibleModel,
) -> dict[str, Any]:
    """Run repeated real-image calls before evaluation.

    The calls are deliberately separate model instances so shared mutable
    response metadata cannot cross threads.  Any upstream route-configuration
    error trips a shared stop flag; queued checks then fail locally without
    issuing more requests.  By default every attempt must succeed, preserving
    the historical stability gate.  A smaller ``required_successes`` enables
    an availability gate; with concurrency one, attempts run serially and stop
    as soon as the required number of successes is reached.
    """

    resolved_attempts = int(attempts)
    resolved_concurrency = int(concurrency)
    resolved_required_successes = (
        resolved_attempts
        if required_successes is None
        else int(required_successes)
    )
    resolved_sleep_seconds = float(inter_attempt_sleep_seconds)
    if resolved_attempts < 1:
        raise ValueError("endpoint preflight attempts must be at least 1")
    if resolved_concurrency < 1:
        raise ValueError("endpoint preflight concurrency must be at least 1")
    if not 1 <= resolved_required_successes <= resolved_attempts:
        raise ValueError(
            "endpoint preflight required successes must be between 1 and "
            "the attempt budget"
        )
    if not math.isfinite(resolved_sleep_seconds) or resolved_sleep_seconds < 0.0:
        raise ValueError(
            "endpoint preflight inter-attempt sleep must be non-negative"
        )
    if resolved_sleep_seconds > 0.0 and resolved_concurrency != 1:
        raise ValueError(
            "endpoint preflight inter-attempt sleep requires concurrency 1"
        )
    resolved_image = image_path.expanduser().resolve()
    image_data_url = _image_data_url(resolved_image)
    stop = threading.Event()

    def invoke(index: int) -> dict[str, Any]:
        if stop.is_set():
            return {
                "attempt": index,
                "status": "cancelled_after_route_failure",
                "api_invoked": False,
            }
        model = model_factory(
            name=f"endpoint-preflight-{index:02d}",
            endpoint=str(endpoint),
            model_id=str(model_id),
            api_key_env=str(api_key_env),
            max_tokens=int(max_tokens),
            timeout_seconds=int(timeout_seconds),
            response_format_json=False,
            max_retries=int(max_retries),
            retry_backoff_seconds=float(retry_backoff_seconds),
            retry_backoff_mode=str(retry_backoff_mode),
            retry_all_http_errors=bool(retry_all_http_errors),
            retry_malformed_response=bool(retry_malformed_response),
            min_request_interval_seconds=float(min_request_interval_seconds),
            send_temperature=False,
            require_api_key=True,
        )
        try:
            content = model.chat_messages(
                [
                    {
                        "role": "system",
                        "content": "Return one short JSON object only.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Confirm that this image is visible. "
                                    "Return exactly {\"ok\":true}."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                        ],
                    },
                ],
                response_format_json=False,
                call_type="endpoint_stability_preflight",
                max_tokens=int(max_tokens),
                max_tokens_source="endpoint_stability_preflight",
            )
        except Exception as exc:
            if isinstance(exc, EndpointConfigurationError):
                stop.set()
            return {
                "attempt": index,
                "status": "failed",
                "api_invoked": True,
                "error_type": type(exc).__name__,
                "error": _bounded_text(str(exc), 2_000),
                "fatal_route_configuration": isinstance(
                    exc,
                    EndpointConfigurationError,
                ),
            }
        metadata = getattr(model, "last_request_metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        return {
            "attempt": index,
            "status": "complete",
            "api_invoked": True,
            "content_nonempty": bool(str(content).strip()),
            "finish_reason": metadata.get("finish_reason"),
            "tokens_usage": deepcopy(metadata.get("usage")),
        }

    def is_success(item: dict[str, Any]) -> bool:
        return (
            item.get("status") == "complete"
            and item.get("content_nonempty") is True
        )

    results: list[dict[str, Any]] = []
    if resolved_concurrency == 1 and (
        resolved_required_successes < resolved_attempts
        or resolved_sleep_seconds > 0.0
    ):
        successes = 0
        for index in range(1, resolved_attempts + 1):
            item = invoke(index)
            results.append(item)
            if is_success(item):
                successes += 1
                if successes >= resolved_required_successes:
                    break
            if stop.is_set():
                break
            if index < resolved_attempts and resolved_sleep_seconds > 0.0:
                time.sleep(resolved_sleep_seconds)
    else:
        with ThreadPoolExecutor(
            max_workers=min(resolved_attempts, resolved_concurrency)
        ) as executor:
            futures = [
                executor.submit(invoke, index)
                for index in range(1, resolved_attempts + 1)
            ]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: int(item["attempt"]))
    failures = [
        deepcopy(item)
        for item in results
        if item.get("status") != "complete"
        or item.get("content_nonempty") is not True
    ]
    completed_attempts = (
        sum(item.get("status") == "complete" for item in results)
        if required_successes is None
        else sum(is_success(item) for item in results)
    )
    passed = (
        not failures
        if required_successes is None
        else completed_attempts >= resolved_required_successes
    )
    report = {
        "schema_version": ENDPOINT_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "endpoint": str(endpoint),
        "model_id": str(model_id),
        "api_key_env": str(api_key_env),
        "authorization_configured": True,
        "image_path": str(resolved_image),
        "attempts_required": resolved_attempts,
        "concurrency": min(resolved_attempts, resolved_concurrency),
        "min_request_interval_seconds": float(min_request_interval_seconds),
        "per_call_retry_policy": {
            "max_retries": int(max_retries),
            "retry_backoff_seconds": float(retry_backoff_seconds),
            "retry_backoff_mode": str(retry_backoff_mode),
            "retry_all_http_errors": bool(retry_all_http_errors),
            "retry_malformed_response": bool(retry_malformed_response),
        },
        "completed_attempts": completed_attempts,
        "api_invocations": sum(
            item.get("api_invoked") is True for item in results
        ),
        "fatal_route_configuration": any(
            item.get("fatal_route_configuration") is True for item in failures
        ),
        "results": results,
        "failures": failures,
    }
    if required_successes is not None:
        report.update(
            pass_policy="minimum_successes",
            successes_required=resolved_required_successes,
            attempts_performed=len(results),
            inter_attempt_sleep_seconds=resolved_sleep_seconds,
        )
    if not passed:
        raise EndpointStabilityPreflightError(report)
    return report


def _image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"endpoint preflight image does not exist: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError(
            f"unsupported endpoint preflight image type: {mime_type or 'unknown'}"
        )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _bounded_text(value: str, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len("<truncated>"))] + "<truncated>"
