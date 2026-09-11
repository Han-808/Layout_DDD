from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VLM_BUDGET_FIELDS = (
    "max_calls",
    "max_input_tokens",
    "max_output_tokens",
    "max_images",
    "max_tool_calls",
)


@dataclass(frozen=True)
class VLMAssistanceBudget:
    """Hard per-run limits for optional generator-side VLM assistance."""

    max_calls: int = 0
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    max_images: int = 0
    max_tool_calls: int = 0

    def __post_init__(self) -> None:
        for field in VLM_BUDGET_FIELDS:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"vlm_budget.{field} must be a non-negative integer")

    @property
    def enabled(self) -> bool:
        return self.max_calls > 0

    def as_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in VLM_BUDGET_FIELDS}

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "VLMAssistanceBudget":
        if value is not None and not isinstance(value, dict):
            raise ValueError("vlm_budget must be a JSON object")
        mapping = value or {}
        unknown = sorted(set(mapping) - set(VLM_BUDGET_FIELDS))
        if unknown:
            raise ValueError(f"vlm_budget contains unsupported fields: {unknown}")
        return cls(**{field: mapping.get(field, 0) for field in VLM_BUDGET_FIELDS})


def budget_for_output(config: dict[str, Any] | None, native_output_type: str) -> VLMAssistanceBudget:
    """Resolve direct or output-keyed budget configuration."""

    if config is not None and not isinstance(config, dict):
        raise ValueError("vlm budget config must be a JSON object")
    mapping = config or {}
    if "outputs" in mapping:
        outputs = mapping.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("vlm budget config outputs must be a JSON object")
        selected = outputs.get(native_output_type)
        if selected is not None and not isinstance(selected, dict):
            raise ValueError(f"vlm budget for {native_output_type} must be a JSON object")
        return VLMAssistanceBudget.from_mapping(selected)
    return VLMAssistanceBudget.from_mapping(mapping)
