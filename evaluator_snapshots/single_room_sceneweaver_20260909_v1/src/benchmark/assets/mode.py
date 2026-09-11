from __future__ import annotations

from dataclasses import dataclass


ASSET_MODES = {"off", "retrieve", "retrieve-generate"}
ASSET_SUPPORT_VALUES = {"required", "optional", "unsupported", "unknown"}


class AssetModeError(ValueError):
    """Raised when an explicit asset mode conflicts with generator capabilities."""


@dataclass(frozen=True)
class AssetModeDecision:
    mode: str
    adapter_support: str
    retrieval_enabled: bool
    generation_enabled: bool
    source_available: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "adapter_support": self.adapter_support,
            "retrieval_enabled": self.retrieval_enabled,
            "generation_enabled": self.generation_enabled,
            "source_available": self.source_available,
            "reason": self.reason,
        }


def resolve_asset_mode(
    *,
    mode: str,
    adapter_support: str,
    structure: bool,
    source_available: bool,
    generation_tool_configured: bool,
) -> AssetModeDecision:
    """Validate and resolve one explicit benchmark asset route."""

    normalized_mode = str(mode or "off").strip().lower()
    normalized_support = str(adapter_support or "unknown").strip().lower()
    if normalized_mode not in ASSET_MODES:
        raise AssetModeError(f"asset_mode must be one of {sorted(ASSET_MODES)}")
    if normalized_support not in ASSET_SUPPORT_VALUES:
        raise AssetModeError(f"adapter asset_support must be one of {sorted(ASSET_SUPPORT_VALUES)}")

    if normalized_mode == "off":
        if normalized_support == "required":
            raise AssetModeError("The selected adapter requires assets, but asset_mode=off.")
        return AssetModeDecision(
            mode=normalized_mode,
            adapter_support=normalized_support,
            retrieval_enabled=False,
            generation_enabled=False,
            source_available=bool(source_available),
            reason="benchmark asset retrieval and generation are disabled",
        )

    if not structure:
        raise AssetModeError(
            f"asset_mode={normalized_mode} requires structured input; use structure=true."
        )
    if normalized_support in {"unsupported", "unknown"}:
        raise AssetModeError(
            f"asset_mode={normalized_mode} is incompatible with adapter asset_support={normalized_support}."
        )
    if not source_available:
        raise AssetModeError(
            f"asset_mode={normalized_mode} requires asset_selection or asset_index_path."
        )

    generation_enabled = normalized_mode == "retrieve-generate"
    if generation_enabled and not generation_tool_configured:
        raise AssetModeError(
            "asset_mode=retrieve-generate requires a configured asset-generation tool."
        )

    return AssetModeDecision(
        mode=normalized_mode,
        adapter_support=normalized_support,
        retrieval_enabled=True,
        generation_enabled=generation_enabled,
        source_available=True,
        reason=(
            "retrieve from the benchmark asset source, then generate only when no candidate is suitable"
            if generation_enabled
            else "retrieve from the benchmark asset source without generation fallback"
        ),
    )
