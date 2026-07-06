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
    build_generation_prompt_sections,
    build_repair_prompt,
    build_repair_prompt_sections,
    parse_json_object,
)
from benchmark.models.prompt_budget import (
    DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS,
    PromptBudgetError,
    build_prompt_budget_report,
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
    generation_max_tokens: int | None = None
    repair_max_tokens: int | None = None
    judge_max_tokens: int | None = None
    context_length: int | None = None
    prompt_safety_margin_tokens: int = DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS
    fail_fast_prompt_budget: bool = True
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
        generation_max_tokens: int | None = None,
        repair_max_tokens: int | None = None,
        judge_max_tokens: int | None = None,
        context_length: int | None = None,
        prompt_safety_margin_tokens: int = DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS,
        fail_fast_prompt_budget: bool = True,
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
        self.generation_max_tokens = generation_max_tokens
        self.repair_max_tokens = repair_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.context_length = context_length
        self.prompt_safety_margin_tokens = prompt_safety_margin_tokens
        self.fail_fast_prompt_budget = fail_fast_prompt_budget
        self.timeout_seconds = timeout_seconds
        self.response_format_json = response_format_json
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]
        self.runtime_profile = runtime_profile
        self.judge_evidence_budgeting = judge_evidence_budgeting
        self.last_request_metadata: dict[str, Any] = {}
        self.last_response_text = ""
        self.last_prompt_text = ""
        self.last_prompt_sections: list[dict[str, Any]] = []

    def generate_layout(self, bm_instance: dict, layout_schema: dict) -> dict:
        sections = build_generation_prompt_sections(bm_instance, layout_schema)
        prompt = build_generation_prompt(bm_instance, layout_schema)
        self.last_prompt_text = prompt
        self.last_prompt_sections = [_section_metadata(section) for section in sections]
        response = self._chat(
            prompt,
            call_type="generation",
            prompt_sections=sections,
            case=bm_instance,
            max_tokens=self._max_tokens_for_call("generation")[0],
            max_tokens_source=self._max_tokens_for_call("generation")[1],
        )
        return parse_json_object(response)

    def repair_layout(
        self,
        bm_instance: dict,
        current_layout: dict,
        feedback: dict,
        layout_schema: dict,
    ) -> dict:
        sections = build_repair_prompt_sections(bm_instance, current_layout, feedback, layout_schema)
        prompt = build_repair_prompt(bm_instance, current_layout, feedback, layout_schema)
        self.last_prompt_text = prompt
        self.last_prompt_sections = [_section_metadata(section) for section in sections]
        response = self._chat(
            prompt,
            call_type="repair",
            prompt_sections=sections,
            case=bm_instance,
            iteration=int(feedback.get("iteration", 0)) + 1 if isinstance(feedback, dict) else None,
            max_tokens=self._max_tokens_for_call("repair")[0],
            max_tokens_source=self._max_tokens_for_call("repair")[1],
        )
        return parse_json_object(response)

    def _chat(
        self,
        prompt: str,
        *,
        call_type: str = "generation",
        prompt_sections: list[Any] | None = None,
        case: dict | None = None,
        iteration: int | None = None,
        max_tokens: int | None = None,
        max_tokens_source: str | None = None,
    ) -> str:
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
            call_type=call_type,
            prompt_sections=prompt_sections,
            case=case,
            iteration=iteration,
            max_tokens=max_tokens,
            max_tokens_source=max_tokens_source,
        )

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
            "temperature": self.temperature,
        }
        if resolved_max_tokens is not None:
            payload["max_tokens"] = resolved_max_tokens
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

    def _max_tokens_for_call(self, call_type: str) -> tuple[int | None, str]:
        if call_type == "generation" and self.generation_max_tokens is not None:
            return int(self.generation_max_tokens), "generation_max_tokens"
        if call_type == "repair":
            if self.repair_max_tokens is not None:
                return int(self.repair_max_tokens), "repair_max_tokens"
            if self.max_tokens is None:
                return 16000, "safe_repair_default"
            return min(int(self.max_tokens), 16000), "min(max_tokens, safe_repair_default)"
        if call_type == "judge" and self.judge_max_tokens is not None:
            return int(self.judge_max_tokens), "judge_max_tokens"
        return self.max_tokens, "max_tokens"

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


def _section_metadata(section: Any) -> dict[str, Any]:
    return {
        "name": getattr(section, "name", "unknown"),
        "chars": len(getattr(section, "text", "")),
        **({"item_count": getattr(section, "item_count")} if getattr(section, "item_count", None) is not None else {}),
        **({"omitted_count": getattr(section, "omitted_count")} if getattr(section, "omitted_count", None) is not None else {}),
    }


def _tiny_png_data_url() -> str:
    return (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
