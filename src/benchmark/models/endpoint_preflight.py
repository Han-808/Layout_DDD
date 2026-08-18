"""Repeated multimodal stability gate for OpenAI-compatible model routes."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import mimetypes
from pathlib import Path
import threading
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
    timeout_seconds: int = 300,
    max_tokens: int = 64,
    model_factory: Callable[..., Any] = OpenAICompatibleModel,
) -> dict[str, Any]:
    """Require every repeated real-image call to succeed before evaluation.

    The calls are deliberately separate model instances so shared mutable
    response metadata cannot cross threads.  Any upstream route-configuration
    error trips a shared stop flag; queued checks then fail locally without
    issuing more requests.
    """

    resolved_attempts = int(attempts)
    resolved_concurrency = int(concurrency)
    if resolved_attempts < 1:
        raise ValueError("endpoint preflight attempts must be at least 1")
    if resolved_concurrency < 1:
        raise ValueError("endpoint preflight concurrency must be at least 1")
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
            max_retries=0,
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

    results: list[dict[str, Any]] = []
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
    report = {
        "schema_version": ENDPOINT_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "endpoint": str(endpoint),
        "model_id": str(model_id),
        "api_key_env": str(api_key_env),
        "authorization_configured": True,
        "image_path": str(resolved_image),
        "attempts_required": resolved_attempts,
        "concurrency": min(resolved_attempts, resolved_concurrency),
        "completed_attempts": sum(
            item.get("status") == "complete" for item in results
        ),
        "api_invocations": sum(
            item.get("api_invoked") is True for item in results
        ),
        "fatal_route_configuration": any(
            item.get("fatal_route_configuration") is True for item in failures
        ),
        "results": results,
        "failures": failures,
    }
    if failures:
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
