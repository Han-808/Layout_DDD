"""OpenAI Responses codec matching the frozen GLM adapter.

See ``docs/generation_transport_compatibility.md`` for the compatibility and
evaluation-module boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from benchmark.scene_generation.frozen_two_stage.providers.base import (
    CanonicalJSONBytes,
    NormalizedResponse,
    ProviderModel,
)


@dataclass(frozen=True)
class OpenAIResponsesCodec:
    """Encode/decode the Responses-shaped API2 GLM envelope."""

    default_reasoning_effort: str = "max"

    def request_value(
        self,
        *,
        model: ProviderModel,
        system_prompt: str,
        user_value: Mapping[str, Any],
        canonical_json_bytes: CanonicalJSONBytes,
    ) -> dict[str, Any]:
        return {
            "model": model.wire_model,
            "instructions": system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": canonical_json_bytes(user_value).decode("utf-8"),
                }
            ],
            "reasoning": {
                "effort": model.reasoning_effort or self.default_reasoning_effort
            },
            "max_output_tokens": model.max_tokens,
            "text": {"format": {"type": "json_object"}},
            "store": False,
            "stream": False,
        }

    def extract_api_message(self, response_body: bytes) -> NormalizedResponse:
        payload = json.loads(response_body.decode("utf-8", errors="strict"))
        if not isinstance(payload, dict):
            raise ValueError("GLM-5.3 response must be a JSON object")
        if payload.get("error"):
            raise ValueError("GLM-5.3 response contains an error object")
        status = str(payload.get("status") or "")
        if status != "completed":
            incomplete = payload.get("incomplete_details")
            incomplete = incomplete if isinstance(incomplete, dict) else {}
            reason = str(incomplete.get("reason") or status or "unknown")
            raise ValueError(f"GLM-5.3 response is not complete: {reason}")

        message_texts: list[str] = []
        reasoning_texts: list[str] = []
        output = payload.get("output")
        output = output if isinstance(output, list) else []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            content = item.get("content")
            content = content if isinstance(content, list) else []
            for part in content:
                if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                    continue
                if item_type == "message" and part.get("type") == "output_text":
                    message_texts.append(part["text"])
                elif item_type == "reasoning" and part.get("type") == "reasoning_text":
                    reasoning_texts.append(part["text"])
        content_text = "\n".join(message_texts).strip()
        if not content_text:
            raise ValueError("GLM-5.3 completed response has no output_text message")
        reasoning = "\n".join(reasoning_texts).strip()
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else None
        return NormalizedResponse(
            content=content_text.encode("utf-8"),
            reasoning=reasoning.encode("utf-8") if reasoning else None,
            reasoning_content=None,
            usage=usage,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "codec": "openai_responses",
            "default_reasoning_effort": self.default_reasoning_effort,
            "text_format": "json_object",
            "store": False,
            "stream": False,
        }
