from __future__ import annotations

import json
import os
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
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate valid 3D room layout JSON. "
                        "Return one JSON object only, with no Markdown or explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            _chat_completions_url(self.endpoint),
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenAICompatibleModelError(f"Model endpoint returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OpenAICompatibleModelError(f"Could not reach model endpoint {self.endpoint}: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise OpenAICompatibleModelError(f"Unexpected model endpoint response: {raw[:500]}") from exc
        if not isinstance(content, str):
            raise OpenAICompatibleModelError("Model endpoint response content is not text.")
        return content

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
