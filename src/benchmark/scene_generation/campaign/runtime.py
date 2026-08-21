"""Compatibility construction from public profiles to existing providers.

This module does not resolve endpoints or credentials.  The caller supplies a
runtime-only credential after profile and resource validation.  Dispatch is
exclusively by the reviewed protocol grammar carried by ``RouteProfile``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

from benchmark.scene_generation.campaign.profiles import ModelProfile, RouteProfile
from benchmark.scene_generation.frozen_two_stage.providers.base import ProviderRoute
from benchmark.scene_generation.frozen_two_stage.providers.codecs.openai_chat import (
    ChatOptionPolicy,
)
from benchmark.scene_generation.frozen_two_stage.providers.routes import (
    make_api2_chat_route,
    make_api2_responses_route,
    make_api3_chat_route,
)


@dataclass(frozen=True, slots=True)
class RuntimeProviderModel:
    """Private transport adapter with an artifact-safe public projection.

    The frozen core needs ``endpoint`` for transport, but its run initializer
    calls ``model.public_dict()``.  This versioned adapter therefore keeps the
    endpoint and credential private while returning only profile-owned values.
    Existing v1 ``ModelConfig`` artifacts are unchanged; campaign artifact v3
    opts into this redacted projection.
    """

    key: str
    label: str
    configured_model: str
    wire_model: str
    endpoint: str = field(repr=False)
    api_key: str = field(repr=False)
    timeout_seconds: float
    max_infrastructure_retries: int
    retry_delay_seconds: float
    max_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    reasoning_effort: str | None
    preserved_thinking: bool | None
    strategy_type: str

    @classmethod
    def from_profile(
        cls,
        profile: ModelProfile,
        *,
        endpoint: str,
        api_key: str,
    ) -> "RuntimeProviderModel":
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("runtime API credential must be non-empty")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("runtime endpoint must be non-empty")
        options = profile.request_options
        gateway = profile.gateway_options
        return cls(
            key=profile.model_profile_id,
            label=profile.display_label,
            configured_model=profile.configured_model,
            wire_model=profile.wire_model,
            endpoint=endpoint,
            api_key=api_key,
            timeout_seconds=options.request_timeout_seconds,
            max_infrastructure_retries=(
                profile.transport_policy.max_infrastructure_retries
            ),
            retry_delay_seconds=profile.transport_policy.retry_delay_seconds,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
            repetition_penalty=options.repetition_penalty,
            reasoning_effort=options.reasoning_effort,
            preserved_thinking=options.preserved_thinking,
            strategy_type=gateway.strategy_type or "unused",
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Return only the profile-owned public values, never the credential."""

        return {
            "key": self.key,
            "label": self.label,
            "configured_model": self.configured_model,
            "wire_model": self.wire_model,
            "timeout_seconds": self.timeout_seconds,
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "reasoning_effort": self.reasoning_effort,
            "preserved_thinking": self.preserved_thinking,
            "strategy_type": self.strategy_type,
            "stream": False,
            "prompt_model_identity_injected": False,
            "generator_semantic_retry_allowed": False,
            "transport_binding": "private-redacted-v1",
        }

    # The frozen core calls ``public_dict``. Keep the familiar name while
    # preserving the v3 private/public split.
    public_dict = to_public_dict


def build_provider_route(
    route: RouteProfile,
    model: ModelProfile,
    *,
    clock: Callable[[], float] = time.time,
) -> ProviderRoute:
    """Build an existing provider route without inspecting the model alias."""

    model.validate_for(route)
    gateway = model.gateway_options
    effort = model.request_options.reasoning_effort
    dispatch = route.grammar.legacy_route_kind
    if dispatch == "api2_chat":
        assert gateway.provider is not None
        assert gateway.gateway_model is not None
        assert gateway.user_agent_suffix is not None
        assert gateway.auth_timeout_seconds is not None
        assert effort is not None
        return make_api2_chat_route(
            provider=gateway.provider,
            gateway_model=gateway.gateway_model,
            user_agent_suffix=gateway.user_agent_suffix,
            option_policy=ChatOptionPolicy.top_level_reasoning(
                default_reasoning_effort=effort
            ),
            route_key=route.route_profile_id,
            runner_version=route.runner_version,
            timeout_seconds=gateway.auth_timeout_seconds,
            clock=clock,
        )
    if dispatch == "api2_responses":
        assert gateway.provider is not None
        assert gateway.gateway_model is not None
        assert gateway.user_agent_suffix is not None
        assert gateway.auth_timeout_seconds is not None
        assert effort is not None
        return make_api2_responses_route(
            provider=gateway.provider,
            gateway_model=gateway.gateway_model,
            user_agent_suffix=gateway.user_agent_suffix,
            default_reasoning_effort=effort,
            route_key=route.route_profile_id,
            runner_version=route.runner_version,
            timeout_seconds=gateway.auth_timeout_seconds,
            clock=clock,
        )
    if dispatch == "api3_chat":
        if route.option_contract_id == "chat_adaptive_thinking_v1":
            assert effort is not None
            option_policy = ChatOptionPolicy.adaptive_thinking(
                reasoning_effort=effort
            )
        elif route.option_contract_id == "chat_legacy_core_v1":
            option_policy = ChatOptionPolicy.legacy_core()
        else:  # guarded by RouteProfile's protocol grammar
            raise ValueError(
                f"unsupported API3 option contract: {route.option_contract_id!r}"
            )
        return make_api3_chat_route(
            option_policy=option_policy,
            route_key=route.route_profile_id,
            runner_version=route.runner_version,
        )
    raise ValueError(f"unsupported legacy route dispatch: {dispatch!r}")
