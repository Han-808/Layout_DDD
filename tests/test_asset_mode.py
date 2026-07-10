from __future__ import annotations

import pytest

from benchmark.assets.mode import AssetModeError, resolve_asset_mode


def test_off_disables_benchmark_asset_routing() -> None:
    decision = resolve_asset_mode(
        mode="off",
        adapter_support="optional",
        structure=True,
        source_available=True,
        generation_tool_configured=True,
    )

    assert decision.retrieval_enabled is False
    assert decision.generation_enabled is False


def test_retrieve_enables_retrieval_without_generation() -> None:
    decision = resolve_asset_mode(
        mode="retrieve",
        adapter_support="optional",
        structure=True,
        source_available=True,
        generation_tool_configured=True,
    )

    assert decision.retrieval_enabled is True
    assert decision.generation_enabled is False


def test_retrieve_generate_enables_both_routes() -> None:
    decision = resolve_asset_mode(
        mode="retrieve-generate",
        adapter_support="required",
        structure=True,
        source_available=True,
        generation_tool_configured=True,
    )

    assert decision.retrieval_enabled is True
    assert decision.generation_enabled is True


def test_retrieval_mode_rejects_unknown_adapter_support() -> None:
    with pytest.raises(AssetModeError, match="asset_support=unknown"):
        resolve_asset_mode(
            mode="retrieve",
            adapter_support="unknown",
            structure=True,
            source_available=True,
            generation_tool_configured=False,
        )


def test_retrieval_mode_requires_asset_source() -> None:
    with pytest.raises(AssetModeError, match="requires asset_selection or asset_index_path"):
        resolve_asset_mode(
            mode="retrieve",
            adapter_support="optional",
            structure=True,
            source_available=False,
            generation_tool_configured=False,
        )


def test_retrieve_generate_requires_generation_tool() -> None:
    with pytest.raises(AssetModeError, match="requires a configured asset-generation tool"):
        resolve_asset_mode(
            mode="retrieve-generate",
            adapter_support="optional",
            structure=True,
            source_available=True,
            generation_tool_configured=False,
        )


def test_off_rejects_adapter_hard_constraint() -> None:
    with pytest.raises(AssetModeError, match="requires assets"):
        resolve_asset_mode(
            mode="off",
            adapter_support="required",
            structure=True,
            source_available=True,
            generation_tool_configured=False,
        )


def test_retrieval_requires_structured_input() -> None:
    with pytest.raises(AssetModeError, match="requires structured input"):
        resolve_asset_mode(
            mode="retrieve",
            adapter_support="optional",
            structure=False,
            source_available=True,
            generation_tool_configured=False,
        )
