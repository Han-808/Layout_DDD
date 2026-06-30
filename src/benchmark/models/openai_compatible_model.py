from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from benchmark.models.base_model import (
    BaseLayoutModel,
    build_generation_prompt,
    build_repair_prompt,
    parse_json_object,
)


class OpenAICompatibleModelError(RuntimeError):
    """Raised when an OpenAI-compatible local model endpoint fails."""


class EndpointConnectionError(OpenAICompatibleModelError):
    """Raised when the endpoint cannot be reached."""


class EndpointHTTPError(OpenAICompatibleModelError):
    """Raised when the endpoint returns a non-success HTTP status."""


class EndpointMalformedResponseError(OpenAICompatibleModelError):
    """Raised when the endpoint response does not match OpenAI chat shape."""


@dataclass
class OpenAICompatibleModel(BaseLayoutModel):
    """Adapter for local/open-source OpenAI-compatible chat endpoints.

    This supports servers such as vLLM, Ollama's OpenAI-compatible API, and
    LM Studio without adding provider-specific SDK dependencies.
    """

    endpoint: str
    model_id: str
    api_key_env: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_seconds: int = 180
    response_format_json: bool = False
    max_retries: int = 0
    retry_backoff_seconds: float = 1.0
    retry_on_status: list[int] | None = None
    runtime_profile: str | None = None
    judge_evidence_budgeting: bool = False

    def __init__(
        self,
        *,
        name: str,
        endpoint: str,
        model_id: str,
        api_key_env: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout_seconds: int = 180,
        response_format_json: bool = False,
        max_retries: int = 0,
        retry_backoff_seconds: float = 1.0,
        retry_on_status: list[int] | None = None,
        runtime_profile: str | None = None,
        judge_evidence_budgeting: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.endpoint = endpoint
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.response_format_json = response_format_json
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]
        self.runtime_profile = runtime_profile
        self.judge_evidence_budgeting = judge_evidence_budgeting
        self.last_request_metadata: dict[str, Any] = {}
        self.last_response_text = ""

    def generate_layout(self, bm_instance: dict, layout_schema: dict) -> dict:
        response = self._chat(build_generation_prompt(bm_instance, layout_schema))
        return parse_json_object(response)

    def repair_layout(
        self,
        bm_instance: dict,
        current_layout: dict,
        feedback: dict,
        layout_schema: dict,
    ) -> dict:
        response = self._chat(build_repair_prompt(bm_instance, current_layout, feedback, layout_schema))
        return parse_json_object(response)

    def _chat(self, prompt: str) -> str:
        return self.chat_messages(
            [
                {
                    "role": "system",
                    "content": (
                        "You generate valid 3D room layout JSON. "
                        "Return one JSON object only, with no Markdown or explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format_json=self.response_format_json,
        )

    def chat_messages(self, messages: list[dict[str, Any]], *, response_format_json: bool | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        use_json_format = self.response_format_json if response_format_json is None else response_format_json
        if use_json_format:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        self.last_request_metadata = _request_metadata(
            endpoint=self.endpoint,
            url=_chat_completions_url(self.endpoint),
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            response_format_json=use_json_format,
        )
        raw = self._post_json(_chat_completions_url(self.endpoint), body)
        try:
            parsed = json.loads(raw)
            choice = parsed["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise EndpointMalformedResponseError(f"Unexpected model endpoint response: {raw[:500]}") from exc
        if not isinstance(content, str):
            raise EndpointMalformedResponseError("Model endpoint response content is not text.")
        self.last_response_text = content
        self.last_request_metadata.update(
            {
                "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
                "content_chars": len(content),
                "usage": parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None,
            }
        )
        return content

    def list_models(self) -> dict:
        raw = self._get_json(_models_url(self.endpoint))
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EndpointMalformedResponseError(f"Unexpected /models response: {raw[:500]}") from exc

    def health_check(self, *, multimodal: bool = False) -> dict:
        models = self.list_models()
        text_ok = bool(
            self.chat_messages(
                [
                    {"role": "system", "content": "Reply with JSON only."},
                    {"role": "user", "content": "Return {\"ok\": true}."},
                ],
                response_format_json=self.response_format_json,
            )
        )
        result = {
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "models_ok": True,
            "text_chat_ok": text_ok,
            "multimodal_chat_ok": None,
            "models": models,
        }
        if multimodal:
            result["multimodal_chat_ok"] = bool(
                self.chat_messages(
                    [
                        {"role": "system", "content": "Reply with JSON only."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Return {\"ok\": true} for this image."},
                                {"type": "image_url", "image_url": {"url": _tiny_png_data_url()}},
                            ],
                        },
                    ],
                    response_format_json=self.response_format_json,
                )
            )
        return result

    def _post_json(self, url: str, body: bytes) -> str:
        request = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        return self._open_with_retry(request)

    def _get_json(self, url: str) -> str:
        request = urllib.request.Request(url, headers=self._headers(), method="GET")
        return self._open_with_retry(request)

    def _open_with_retry(self, request: urllib.request.Request) -> str:
        attempts = max(0, int(self.max_retries)) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            if attempt > 0:
                time.sleep(max(0.0, float(self.retry_backoff_seconds)) * attempt)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = EndpointHTTPError(f"Model endpoint returned HTTP {exc.code}: {detail}")
                if exc.code not in self.retry_on_status or attempt == attempts - 1:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = EndpointConnectionError(f"Could not reach model endpoint {self.endpoint}: {exc.reason}")
                if attempt == attempts - 1:
                    raise last_error from exc
        raise last_error or EndpointConnectionError(f"Could not reach model endpoint {self.endpoint}.")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self.api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


def _chat_completions_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _models_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return f"{base[: -len('/chat/completions')]}/models"
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/models"


def _request_metadata(*, endpoint: str, url: str, payload: dict[str, Any], timeout_seconds: int, response_format_json: bool) -> dict:
    messages = payload.get("messages", [])
    image_count = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            image_count += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image_url")
    return {
        "endpoint": endpoint,
        "url": url,
        "model": payload.get("model"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "timeout_seconds": timeout_seconds,
        "response_format_json": response_format_json,
        "message_count": len(messages),
        "image_count": image_count,
    }


def _tiny_png_data_url() -> str:
    return (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
