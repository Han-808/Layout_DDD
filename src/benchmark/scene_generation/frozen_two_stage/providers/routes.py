"""Provider-route composition without model-name branching.

The composition rules follow ``docs/generation_transport_compatibility.md``:
models are configuration; codecs and gateways are reusable protocol units.
"""

from __future__ import annotations

import time
from typing import Callable

from benchmark.scene_generation.frozen_two_stage.providers.base import ProviderRoute
from benchmark.scene_generation.frozen_two_stage.providers.codecs.openai_chat import (
    ChatOptionPolicy,
    OpenAIChatCodec,
)
from benchmark.scene_generation.frozen_two_stage.providers.codecs.openai_responses import (
    OpenAIResponsesCodec,
)
from benchmark.scene_generation.frozen_two_stage.providers.gateways.api2 import (
    API2GatewayPolicy,
)
from benchmark.scene_generation.frozen_two_stage.providers.gateways.api3 import (
    API3GatewayPolicy,
)
from benchmark.scene_generation.frozen_two_stage.providers.gateways.standard import (
    StandardBearerGatewayPolicy,
)


def make_api2_chat_route(
    *,
    provider: str,
    gateway_model: str,
    user_agent_suffix: str,
    option_policy: ChatOptionPolicy,
    route_key: str = "api2-openai-chat",
    runner_version: str = "2.0.0",
    timeout_seconds: int = 600,
    clock: Callable[[], float] = time.time,
) -> ProviderRoute:
    """Compose an API2 gateway with the reusable Chat codec."""

    return ProviderRoute(
        codec=OpenAIChatCodec(option_policy=option_policy),
        gateway=API2GatewayPolicy(
            provider=provider,
            gateway_model=gateway_model,
            user_agent_suffix=user_agent_suffix,
            runner_version=runner_version,
            timeout_seconds=timeout_seconds,
            clock=clock,
        ),
        route_key=route_key,
        provenance_modules=(__name__,),
    )


def make_api2_responses_route(
    *,
    provider: str,
    gateway_model: str,
    user_agent_suffix: str,
    default_reasoning_effort: str = "max",
    route_key: str = "api2-openai-responses",
    runner_version: str = "2.0.0",
    timeout_seconds: int = 600,
    clock: Callable[[], float] = time.time,
) -> ProviderRoute:
    """Compose an API2 gateway with the reusable Responses codec."""

    return ProviderRoute(
        codec=OpenAIResponsesCodec(
            default_reasoning_effort=default_reasoning_effort
        ),
        gateway=API2GatewayPolicy(
            provider=provider,
            gateway_model=gateway_model,
            user_agent_suffix=user_agent_suffix,
            runner_version=runner_version,
            timeout_seconds=timeout_seconds,
            clock=clock,
        ),
        route_key=route_key,
        provenance_modules=(__name__,),
    )


def make_api3_chat_route(
    *,
    option_policy: ChatOptionPolicy,
    route_key: str = "api3-openai-chat",
    runner_version: str = "2.0.0",
) -> ProviderRoute:
    """Compose an API3 gateway with the reusable Chat codec."""

    return ProviderRoute(
        codec=OpenAIChatCodec(option_policy=option_policy),
        gateway=API3GatewayPolicy(runner_version=runner_version),
        route_key=route_key,
        provenance_modules=(__name__,),
    )


def make_standard_chat_route(
    *,
    user_agent_suffix: str,
    option_policy: ChatOptionPolicy,
    route_key: str = "standard-openai-chat",
    runner_version: str = "2.0.0",
) -> ProviderRoute:
    """Compose the Chat codec with a standard Bearer gateway."""

    return ProviderRoute(
        codec=OpenAIChatCodec(option_policy=option_policy),
        gateway=StandardBearerGatewayPolicy(
            user_agent_suffix=user_agent_suffix,
            runner_version=runner_version,
        ),
        route_key=route_key,
        provenance_modules=(__name__,),
    )
