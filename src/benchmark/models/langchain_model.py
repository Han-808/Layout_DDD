from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.models.base_model import (
    BaseLayoutModel,
    build_generation_prompt,
    build_repair_prompt,
    parse_json_object,
)


@dataclass
class LangChainModel(BaseLayoutModel):
    """Thin adapter for any LangChain Runnable-compatible chat model.

    Provider-specific construction intentionally lives outside this class. That
    keeps the benchmark from hardcoding proprietary providers while still making
    it easy to plug in OpenAI-compatible, Anthropic-compatible, Gemini-compatible,
    or local vLLM clients.
    """

    runnable: Any | None = None

    def __init__(self, name: str, runnable: Any | None = None) -> None:
        super().__init__(name=name)
        self.runnable = runnable

    def generate_layout(self, bm_instance: dict, layout_schema: dict) -> dict:
        if self.runnable is None:
            raise RuntimeError("LangChainModel requires a Runnable-compatible model instance.")
        response = self.runnable.invoke(build_generation_prompt(bm_instance, layout_schema))
        return parse_json_object(_response_to_text(response))

    def repair_layout(
        self,
        bm_instance: dict,
        current_layout: dict,
        feedback: dict,
        layout_schema: dict,
    ) -> dict:
        if self.runnable is None:
            raise RuntimeError("LangChainModel requires a Runnable-compatible model instance.")
        response = self.runnable.invoke(
            build_repair_prompt(bm_instance, current_layout, feedback, layout_schema)
        )
        return parse_json_object(_response_to_text(response))


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)
    return str(response)
