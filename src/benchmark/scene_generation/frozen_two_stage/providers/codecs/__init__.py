"""Wire codecs described by ``docs/generation_transport_compatibility.md``."""

from benchmark.scene_generation.frozen_two_stage.providers.codecs.openai_chat import (
    ChatOptionPolicy,
    ChatOptionStyle,
    OpenAIChatCodec,
)
from benchmark.scene_generation.frozen_two_stage.providers.codecs.openai_responses import (
    OpenAIResponsesCodec,
)

__all__ = [
    "ChatOptionPolicy",
    "ChatOptionStyle",
    "OpenAIChatCodec",
    "OpenAIResponsesCodec",
]
