"""OpenAI Chat Completions codec for frozen generation routes.

The explicitly typed option profiles preserve the current core, API2 Kimi,
and API3 Opus envelopes documented in
``docs/generation_transport_compatibility.md`` without model-name branching.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Mapping

from benchmark.scene_generation.frozen_two_stage.providers.base import (
    CanonicalJSONBytes,
    NormalizedResponse,
    ProviderModel,
)


class ChatOptionStyle(str, Enum):
    """Frozen request-field layouts supported by the Chat codec."""

    LEGACY_CORE = "legacy_core"
    TOP_LEVEL_REASONING = "top_level_reasoning"
    ADAPTIVE_THINKING = "adaptive_thinking"


@dataclass(frozen=True)
class ChatOptionPolicy:
    """Typed, allowlisted Chat options rather than a free-form template."""

    style: ChatOptionStyle
    default_reasoning_effort: str | None = None
    fixed_reasoning_effort: str | None = None
    thinking_type: str | None = None

    def __post_init__(self) -> None:
        if self.style is ChatOptionStyle.LEGACY_CORE:
            if any(
                value is not None
                for value in (
                    self.default_reasoning_effort,
                    self.fixed_reasoning_effort,
                    self.thinking_type,
                )
            ):
                raise ValueError("legacy-core Chat options do not accept fixed fields")
        elif self.style is ChatOptionStyle.TOP_LEVEL_REASONING:
            if not self.default_reasoning_effort:
                raise ValueError(
                    "top-level reasoning requires a default reasoning effort"
                )
            if self.fixed_reasoning_effort is not None or self.thinking_type is not None:
                raise ValueError(
                    "top-level reasoning does not accept thinking or a fixed effort"
                )
        elif self.style is ChatOptionStyle.ADAPTIVE_THINKING:
            if self.thinking_type != "adaptive":
                raise ValueError("adaptive-thinking style requires thinking_type='adaptive'")
            if not self.fixed_reasoning_effort:
                raise ValueError("adaptive-thinking style requires a fixed reasoning effort")
            if self.default_reasoning_effort is not None:
                raise ValueError(
                    "adaptive-thinking style does not accept a default effort"
                )
        else:  # pragma: no cover - defensive against invalid enum construction
            raise ValueError(f"unsupported Chat option style: {self.style!r}")

    @classmethod
    def legacy_core(cls) -> "ChatOptionPolicy":
        """Match ``api3_anthropic_runner_v2`` request option behavior."""

        return cls(style=ChatOptionStyle.LEGACY_CORE)

    @classmethod
    def top_level_reasoning(
        cls, *, default_reasoning_effort: str = "max"
    ) -> "ChatOptionPolicy":
        """Match the API2 Kimi top-level reasoning field layout."""

        return cls(
            style=ChatOptionStyle.TOP_LEVEL_REASONING,
            default_reasoning_effort=default_reasoning_effort,
        )

    @classmethod
    def adaptive_thinking(
        cls, *, reasoning_effort: str = "high"
    ) -> "ChatOptionPolicy":
        """Match the API3 Opus adaptive-thinking field layout."""

        return cls(
            style=ChatOptionStyle.ADAPTIVE_THINKING,
            fixed_reasoning_effort=reasoning_effort,
            thinking_type="adaptive",
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "style": self.style.value,
            "default_reasoning_effort": self.default_reasoning_effort,
            "fixed_reasoning_effort": self.fixed_reasoning_effort,
            "thinking_type": self.thinking_type,
        }


@dataclass(frozen=True)
class OpenAIChatCodec:
    """Encode/decode Chat Completions-shaped provider traffic."""

    option_policy: ChatOptionPolicy

    def request_value(
        self,
        *,
        model: ProviderModel,
        system_prompt: str,
        user_value: Mapping[str, Any],
        canonical_json_bytes: CanonicalJSONBytes,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": canonical_json_bytes(user_value).decode("utf-8"),
            },
        ]
        if self.option_policy.style is ChatOptionStyle.TOP_LEVEL_REASONING:
            return {
                "model": model.wire_model,
                "messages": messages,
                "reasoning_effort": (
                    model.reasoning_effort
                    or self.option_policy.default_reasoning_effort
                ),
                "max_tokens": model.max_tokens,
                "stream": False,
            }
        if self.option_policy.style is ChatOptionStyle.ADAPTIVE_THINKING:
            return {
                "model": model.wire_model,
                "messages": messages,
                "max_tokens": model.max_tokens,
                "stream": False,
                "thinking": {"type": self.option_policy.thinking_type},
                "reasoning_effort": self.option_policy.fixed_reasoning_effort,
            }

        value: dict[str, Any] = {
            "model": model.wire_model,
            "messages": messages,
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

    def extract_api_message(self, response_body: bytes) -> NormalizedResponse:
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
            raise ValueError(
                "API response must contain string choices[0].message.content"
            )
        reasoning = message.get("reasoning")
        reasoning_content = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("message.reasoning must be string or null")
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise ValueError("message.reasoning_content must be string or null")
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        return NormalizedResponse(
            content=message["content"].encode("utf-8"),
            reasoning=None if reasoning is None else reasoning.encode("utf-8"),
            reasoning_content=(
                None
                if reasoning_content is None
                else reasoning_content.encode("utf-8")
            ),
            usage=usage,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "codec": "openai_chat_completions",
            "option_policy": self.option_policy.public_dict(),
        }
