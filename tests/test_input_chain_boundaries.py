from __future__ import annotations

from benchmark.nl_scene import CURRENT_INPUT_CHAIN, LEGEND_INPUT_CHAIN


def test_natural_language_is_current_input_chain() -> None:
    assert CURRENT_INPUT_CHAIN == "natural_language"
    assert LEGEND_INPUT_CHAIN is False
