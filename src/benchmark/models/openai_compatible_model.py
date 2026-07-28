from __future__ import annotations

import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Any

from benchmark.models.prompt_budget import (
    DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS,
    PromptBudgetError,
    build_prompt_budget_report,
)


MAX_HTTP_ERROR_BODY_BYTES = 8_192
MAX_HTTP_ERROR_DETAIL_CHARS = 2_000
_ENVIRONMENT_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OpenAICompatibleModelError(RuntimeError):
    """Raised when an OpenAI-compatible local model endpoint fails."""


class EndpointConnectionError(OpenAICompatibleModelError):
    """Raised when the endpoint cannot be reached."""


class EndpointHTTPError(OpenAICompatibleModelError):
    """Raised when the endpoint returns a non-success HTTP status."""


class EndpointMalformedResponseError(OpenAICompatibleModelError):
    """Raised when the endpoint response does not match OpenAI chat shape."""


class MissingAPIKeyError(OpenAICompatibleModelError):
    """Raised when a credentialed endpoint has no configured API key."""


class OpenAICompatibleModel:
    """Client for local or remote OpenAI-compatible chat endpoints.

    This supports servers such as vLLM, Ollama's OpenAI-compatible API, and
    LM Studio without adding provider-specific SDK dependencies. Generator
    adapters own task prompts and output conversion; this client has no
    benchmark-layout generation or repair behavior.
    """

    endpoint: str
    model_id: str
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    context_length: int | None = None
    prompt_safety_margin_tokens: int = DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS
    fail_fast_prompt_budget: bool = True
    timeout_seconds: int = 180
    response_format_json: bool = False
    max_retries: int = 0
    retry_backoff_seconds: float = 1.0
    retry_on_status: list[int] | None = None
    max_tokens_field: str = "max_tokens"
    send_temperature: bool = True
    require_api_key: bool | None = None

    def __init__(
        self,
        *,
        name: str,
        endpoint: str,
        model_id: str,
        api_key_env: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        context_length: int | None = None,
        prompt_safety_margin_tokens: int = DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS,
        fail_fast_prompt_budget: bool = True,
        timeout_seconds: int = 180,
        response_format_json: bool = False,
        max_retries: int = 0,
        retry_backoff_seconds: float = 1.0,
        retry_on_status: list[int] | None = None,
        max_tokens_field: str = "max_tokens",
        send_temperature: bool = True,
        require_api_key: bool | None = None,
    ) -> None:
        _validate_endpoint_security(endpoint)
        self.name = name
        self.endpoint = endpoint
        self.model_id = model_id
        official_openai_endpoint = _is_official_openai_endpoint(endpoint)
        resolved_api_key_env = api_key_env or (
            "OPENAI_API_KEY" if official_openai_endpoint else None
        )
        _validate_api_key_env_name(resolved_api_key_env)
        self.api_key_env = resolved_api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_length = context_length
        self.prompt_safety_margin_tokens = prompt_safety_margin_tokens
        self.fail_fast_prompt_budget = fail_fast_prompt_budget
        self.timeout_seconds = timeout_seconds
        self.response_format_json = response_format_json
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]
        if max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "max_tokens_field must be 'max_tokens' or 'max_completion_tokens'"
            )
        self.max_tokens_field = max_tokens_field
        self.send_temperature = bool(send_temperature)
        self.require_api_key = (
            bool(official_openai_endpoint or api_key_env)
            if require_api_key is None
            else bool(require_api_key)
        )
        self.last_request_metadata: dict[str, Any] = {}
        self.last_response_text = ""

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format_json: bool | None = None,
        call_type: str = "chat",
        prompt_sections: list[Any] | None = None,
        case: dict | None = None,
        iteration: int | None = None,
        max_tokens: int | None = None,
        max_tokens_source: str | None = None,
    ) -> str:
        resolved_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }
        if self.send_temperature:
            payload["temperature"] = self.temperature
        if resolved_max_tokens is not None:
            payload[self.max_tokens_field] = resolved_max_tokens
        use_json_format = self.response_format_json if response_format_json is None else response_format_json
        if use_json_format:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        prompt_text = _messages_text(messages)
        budget_report = build_prompt_budget_report(
            call_type=call_type,
            prompt_text=prompt_text,
            max_tokens=resolved_max_tokens,
            context_length=self.context_length,
            safety_margin_tokens=self.prompt_safety_margin_tokens,
            prompt_sections=prompt_sections,
            case_id=str(case.get("case_id") or case.get("task_id")) if isinstance(case, dict) else None,
            scene_id=str(case.get("scene_id") or case.get("case_id") or case.get("task_id")) if isinstance(case, dict) else None,
            input_mode=str(case.get("scene_representation_mode") or case.get("input_mode") or case.get("input_level")) if isinstance(case, dict) else None,
            iteration=iteration,
            object_count=len(case.get("objects", [])) if isinstance(case, dict) and isinstance(case.get("objects"), list) else None,
            compaction_level=str(case.get("scene_representation_mode")) if isinstance(case, dict) and case.get("scene_representation_mode") else None,
            max_tokens_source=max_tokens_source,
        )
        self.last_request_metadata = _request_metadata(
            endpoint=self.endpoint,
            url=_chat_completions_url(self.endpoint),
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            response_format_json=use_json_format,
            call_type=call_type,
        )
        self.last_request_metadata["prompt_chars"] = _message_text_chars(messages)
        self.last_request_metadata["api_key_env"] = self.api_key_env
        self.last_request_metadata["authorization_configured"] = bool(
            self.api_key_env and os.environ.get(self.api_key_env)
        )
        self.last_request_metadata["max_tokens_field"] = self.max_tokens_field
        self.last_request_metadata["send_temperature"] = self.send_temperature
        self.last_request_metadata["prompt_budget_report"] = budget_report
        self.last_request_metadata["prompt_budget_warning"] = budget_report.get("warning")
        self.last_request_metadata["prompt_budget_exceeded"] = budget_report.get("fits_context") is False
        if self.fail_fast_prompt_budget and budget_report.get("fits_context") is False:
            self.last_response_text = ""
            raise PromptBudgetError(budget_report)
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

    def health_check(
        self,
        *,
        multimodal: bool = False,
        image_data_url: str | None = None,
    ) -> dict:
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
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_data_url or _tiny_png_data_url()},
                                },
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
                with _urlopen_no_redirect(request, self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                raw_detail = exc.read(MAX_HTTP_ERROR_BODY_BYTES + 1).decode(
                    "utf-8", errors="replace"
                )
                detail = _redacted_error_detail(
                    raw_detail,
                    secret=self._resolved_api_key(),
                    body_truncated=len(raw_detail.encode("utf-8")) > MAX_HTTP_ERROR_BODY_BYTES,
                )
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
        token = self._resolved_api_key()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif self.require_api_key:
            env_hint = self.api_key_env or "OPENAI_API_KEY"
            raise MissingAPIKeyError(
                f"Endpoint {self.endpoint} requires an API key. Set environment variable "
                f"{env_hint}; literal API-key values are not accepted by benchmark configs."
            )
        return headers

    def _resolved_api_key(self) -> str | None:
        token = os.environ.get(self.api_key_env) if self.api_key_env else None
        if not isinstance(token, str):
            return None
        token = token.strip()
        return token or None


def _is_official_openai_endpoint(endpoint: str) -> bool:
    try:
        return (urlparse(endpoint).hostname or "").lower() == "api.openai.com"
    except ValueError:
        return False


def _validate_endpoint_security(endpoint: str) -> None:
    try:
        parsed = urlparse(str(endpoint))
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    except ValueError as exc:
        raise ValueError("OpenAI-compatible endpoint is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("OpenAI-compatible endpoint must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError(
            "OpenAI-compatible endpoint must not contain credentials, query parameters, or fragments"
        )
    if parsed.scheme == "http" and not _is_loopback_hostname(hostname):
        raise ValueError(
            "Non-loopback OpenAI-compatible endpoints must use HTTPS; plain HTTP is allowed "
            "only for localhost/loopback servers"
        )


def _validate_api_key_env_name(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _ENVIRONMENT_VARIABLE_NAME.fullmatch(value):
        raise ValueError(
            "api_key_env must be an environment-variable name such as OPENAI_API_KEY; "
            "literal API-key values are not accepted"
        )


def _is_loopback_hostname(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent urllib from forwarding Authorization to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            "redirect responses are not followed",
            headers,
            fp,
        )


def _urlopen_no_redirect(
    request: urllib.request.Request,
    timeout: int,
):
    opener = urllib.request.build_opener(_RejectRedirectHandler())
    return opener.open(request, timeout=timeout)


def _redacted_error_detail(
    detail: str,
    *,
    secret: str | None,
    body_truncated: bool,
) -> str:
    safe = str(detail)
    if secret:
        safe = safe.replace(secret, "<redacted>")
    safe = re.sub(r"(?i)\bbearer\s+[^\s,;\"']+", "Bearer <redacted>", safe)
    safe = re.sub(
        r'(?i)(["\']?(?:api[_-]?key|authorization|access[_-]?token|secret)["\']?\s*[:=]\s*)'
        r'(["\']?)[^\s,;}\]]+\2',
        r"\1<redacted>",
        safe,
    )
    clipped = safe[:MAX_HTTP_ERROR_DETAIL_CHARS]
    if body_truncated or len(safe) > MAX_HTTP_ERROR_DETAIL_CHARS:
        clipped += "...<truncated>"
    return clipped


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


def _request_metadata(*, endpoint: str, url: str, payload: dict[str, Any], timeout_seconds: int, response_format_json: bool, call_type: str = "chat") -> dict:
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
        "max_completion_tokens": payload.get("max_completion_tokens"),
        "timeout_seconds": timeout_seconds,
        "response_format_json": response_format_json,
        "call_type": call_type,
        "message_count": len(messages),
        "image_count": image_count,
    }


def _message_text_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    total += len(item["text"])
    return total


def _messages_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    return "\n".join(parts)


def _tiny_png_data_url() -> str:
    return (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
