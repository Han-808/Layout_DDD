"""Chat codec output-budget field name.

Azure reasoning routes (GPT-6 Astra) reject the deprecated ``max_tokens`` field
and require ``max_completion_tokens``.  These tests pin that the historical
routes keep emitting ``max_tokens`` in their original key position, and that the
Azure field name is reachable from configuration without touching the codec.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from benchmark.scene_generation.frozen_two_stage.config import _parse_chat_options
from benchmark.scene_generation.frozen_two_stage.providers.codecs.openai_chat import (
    ChatOptionPolicy,
    ChatOptionStyle,
    OpenAIChatCodec,
)


@dataclass(frozen=True)
class _Model:
    wire_model: str = "gpt-6-astra"
    api_key: str = "app:key"
    max_tokens: int = 65536
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    reasoning_effort: str | None = None
    preserved_thinking: bool | None = None
    strategy_type: str = "chat_completions_api2_azure"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _request(policy: ChatOptionPolicy, model: _Model | None = None) -> dict:
    return OpenAIChatCodec(option_policy=policy).request_value(
        model=model or _Model(),
        system_prompt="system",
        user_value={"brief": "b"},
        canonical_json_bytes=_canonical,
    )


def test_top_level_reasoning_defaults_to_historical_field_and_position() -> None:
    body = _request(ChatOptionPolicy.top_level_reasoning(default_reasoning_effort="max"))
    assert "max_completion_tokens" not in body
    assert body["max_tokens"] == 65536
    # Key order is part of the frozen request bytes.
    assert list(body) == [
        "model",
        "messages",
        "reasoning_effort",
        "max_tokens",
        "stream",
    ]


def test_azure_budget_field_replaces_max_tokens_in_place() -> None:
    body = _request(
        ChatOptionPolicy.top_level_reasoning(
            default_reasoning_effort="xhigh",
            max_tokens_field="max_completion_tokens",
        )
    )
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 65536
    assert body["reasoning_effort"] == "xhigh"
    assert list(body) == [
        "model",
        "messages",
        "reasoning_effort",
        "max_completion_tokens",
        "stream",
    ]
    # Astra rejects these; the style must not introduce them.
    for rejected in ("temperature", "top_p", "top_k", "repetition_penalty", "stop"):
        assert rejected not in body


def test_legacy_and_adaptive_styles_keep_max_tokens() -> None:
    legacy = _request(ChatOptionPolicy.legacy_core())
    adaptive = _request(ChatOptionPolicy.adaptive_thinking(reasoning_effort="high"))
    assert legacy["max_tokens"] == 65536
    assert adaptive["max_tokens"] == 65536
    assert "max_completion_tokens" not in legacy
    assert "max_completion_tokens" not in adaptive


def test_model_reasoning_effort_overrides_policy_default() -> None:
    body = _request(
        ChatOptionPolicy.top_level_reasoning(
            default_reasoning_effort="max",
            max_tokens_field="max_completion_tokens",
        ),
        _Model(reasoning_effort="xhigh"),
    )
    assert body["reasoning_effort"] == "xhigh"


def test_unknown_budget_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_tokens_field"):
        ChatOptionPolicy(
            style=ChatOptionStyle.TOP_LEVEL_REASONING,
            default_reasoning_effort="xhigh",
            max_tokens_field="maximum_tokens",
        )


def test_policy_public_dict_records_the_field() -> None:
    policy = ChatOptionPolicy.top_level_reasoning(
        default_reasoning_effort="xhigh",
        max_tokens_field="max_completion_tokens",
    )
    assert policy.public_dict()["max_tokens_field"] == "max_completion_tokens"


def test_config_parses_the_optional_budget_field() -> None:
    style, effort, budget = _parse_chat_options(
        {
            "style": "top_level_reasoning",
            "default_reasoning_effort": "xhigh",
            "max_tokens_field": "max_completion_tokens",
        },
        route_kind="api2_chat",
    )
    assert (style, effort, budget) == (
        "top_level_reasoning",
        "xhigh",
        "max_completion_tokens",
    )


def test_config_defaults_the_budget_field_when_absent() -> None:
    _style, _effort, budget = _parse_chat_options(
        {"style": "top_level_reasoning", "default_reasoning_effort": "max"},
        route_kind="api2_chat",
    )
    assert budget == "max_tokens"


def test_campaign_azure_option_contract_selects_max_completion_tokens() -> None:
    from benchmark.scene_generation.campaign.profiles import PROTOCOL_GRAMMARS
    from benchmark.scene_generation.campaign.runtime import _chat_max_tokens_field

    azure = next(
        grammar
        for grammar in PROTOCOL_GRAMMARS.values()
        if grammar.option_contract_id == "chat_top_level_reasoning_azure_v1"
    )
    assert azure.codec_id == "openai_chat_completions_v1"
    assert azure.gateway_id == "api2_bearer_query_v1"
    assert azure.legacy_route_kind == "api2_chat"

    class _Route:
        option_contract_id = "chat_top_level_reasoning_azure_v1"

    class _HistoricalRoute:
        option_contract_id = "chat_top_level_reasoning_v1"

    assert _chat_max_tokens_field(_Route()) == "max_completion_tokens"
    assert _chat_max_tokens_field(_HistoricalRoute()) == "max_tokens"


def test_legacy_projection_maps_the_azure_budget_field_to_its_contract() -> None:
    from benchmark.scene_generation.campaign.legacy_v1 import _option_contract_id

    assert (
        _option_contract_id("api2_chat", "top_level_reasoning", "max_completion_tokens")
        == "chat_top_level_reasoning_azure_v1"
    )
    assert (
        _option_contract_id("api2_chat", "top_level_reasoning", "max_tokens")
        == "chat_top_level_reasoning_v1"
    )
    assert (
        _option_contract_id("api2_chat", "top_level_reasoning")
        == "chat_top_level_reasoning_v1"
    )


def test_config_rejects_an_unsupported_budget_field() -> None:
    with pytest.raises(ValueError, match="max_tokens_field"):
        _parse_chat_options(
            {
                "style": "top_level_reasoning",
                "default_reasoning_effort": "xhigh",
                "max_tokens_field": "max_output_tokens",
            },
            route_kind="api2_chat",
        )
