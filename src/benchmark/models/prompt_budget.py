from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS = 4096


class PromptBudgetError(RuntimeError):
    """Raised before an endpoint call when prompt + completion cannot fit."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        stage = report.get("call_type") or "model_call"
        over = report.get("over_budget_tokens")
        context = report.get("context_length")
        max_tokens = report.get("max_tokens")
        estimated = report.get("estimated_prompt_tokens")
        message = (
            f"prompt_budget_exceeded during {stage}: estimated_prompt_tokens={estimated}, "
            f"context_length={context}, max_tokens={max_tokens}, over_budget_tokens={over}"
        )
        super().__init__(message)


@dataclass(frozen=True)
class PromptSection:
    name: str
    text: str
    item_count: int | None = None
    omitted_count: int | None = None


def estimate_prompt_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def prompt_sections_report(sections: list[dict[str, Any] | PromptSection] | None) -> list[dict[str, Any]]:
    report = []
    for item in sections or []:
        if isinstance(item, PromptSection):
            name = item.name
            text = item.text
            item_count = item.item_count
            omitted_count = item.omitted_count
        elif isinstance(item, dict):
            name = str(item.get("name", "unknown"))
            text = str(item.get("text", ""))
            item_count = item.get("item_count") if isinstance(item.get("item_count"), int) else None
            omitted_count = item.get("omitted_count") if isinstance(item.get("omitted_count"), int) else None
        else:
            continue
        entry = {
            "name": name,
            "chars": len(text),
            "estimated_tokens": estimate_prompt_tokens(text),
        }
        if item_count is not None:
            entry["item_count"] = item_count
        if omitted_count is not None:
            entry["omitted_count"] = omitted_count
        report.append(entry)
    return report


def build_prompt_budget_report(
    *,
    call_type: str,
    prompt_text: str,
    max_tokens: int | None,
    context_length: int | None,
    safety_margin_tokens: int | None = None,
    prompt_sections: list[dict[str, Any] | PromptSection] | None = None,
    case_id: str | None = None,
    scene_id: str | None = None,
    input_mode: str | None = None,
    iteration: int | None = None,
    object_count: int | None = None,
    compaction_level: str | None = None,
    max_tokens_source: str | None = None,
) -> dict[str, Any]:
    resolved_max_tokens = int(max_tokens or 0)
    resolved_margin = int(safety_margin_tokens or DEFAULT_PROMPT_SAFETY_MARGIN_TOKENS)
    estimated_prompt_tokens = estimate_prompt_tokens(prompt_text)
    sections = prompt_sections_report(prompt_sections)
    largest_sections = sorted(sections, key=lambda item: item["estimated_tokens"], reverse=True)[:5]
    prompt_budget = None
    fits_context = None
    over_budget_tokens = 0
    estimated_total_tokens = estimated_prompt_tokens + resolved_max_tokens + resolved_margin
    if context_length is not None:
        prompt_budget = int(context_length) - resolved_max_tokens - resolved_margin
        fits_context = estimated_prompt_tokens <= prompt_budget
        over_budget_tokens = max(0, estimated_prompt_tokens - prompt_budget)
    return {
        "call_type": call_type,
        "case_id": case_id,
        "scene_id": scene_id,
        "input_mode": input_mode,
        "iteration": iteration,
        "context_length": context_length,
        "max_tokens": resolved_max_tokens,
        "max_tokens_source": max_tokens_source,
        "safety_margin_tokens": resolved_margin,
        "prompt_chars": len(prompt_text),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "prompt_budget": prompt_budget,
        "fits_context": fits_context,
        "over_budget_tokens": over_budget_tokens,
        "object_count": object_count,
        "compaction_level": compaction_level,
        "sections": sections,
        "largest_sections": largest_sections,
        "warning": (
            "max_tokens consumes more than half of the context window; prompt budget may be too small."
            if context_length is not None and resolved_max_tokens > int(context_length) * 0.5
            else None
        ),
    }
