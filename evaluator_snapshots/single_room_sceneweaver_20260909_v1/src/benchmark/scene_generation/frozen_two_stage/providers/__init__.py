"""Typed provider routes for the frozen two-stage generator.

The architecture and frozen compatibility boundary are documented in
``docs/generation_transport_compatibility.md``.
"""

from benchmark.scene_generation.frozen_two_stage.providers.base import (
    GatewayPolicy,
    NormalizedResponse,
    ProviderModel,
    ProviderRoute,
    RequestCodec,
)
from benchmark.scene_generation.frozen_two_stage.providers.codecs import (
    ChatOptionPolicy,
    ChatOptionStyle,
    OpenAIChatCodec,
    OpenAIResponsesCodec,
)
from benchmark.scene_generation.frozen_two_stage.providers.gateways import (
    API2GatewayPolicy,
    API3GatewayPolicy,
    parse_api2_credential,
)
from benchmark.scene_generation.frozen_two_stage.providers.routes import (
    make_api2_chat_route,
    make_api2_responses_route,
    make_api3_chat_route,
)

__all__ = [
    "API2GatewayPolicy",
    "API3GatewayPolicy",
    "ChatOptionPolicy",
    "ChatOptionStyle",
    "GatewayPolicy",
    "NormalizedResponse",
    "OpenAIChatCodec",
    "OpenAIResponsesCodec",
    "ProviderModel",
    "ProviderRoute",
    "RequestCodec",
    "make_api2_chat_route",
    "make_api2_responses_route",
    "make_api3_chat_route",
    "parse_api2_credential",
]
