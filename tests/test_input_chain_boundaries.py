from __future__ import annotations

from benchmark.data import DATASET_ADAPTERS
from benchmark.legend.hssd.hssd_hab_converter import CURRENT_INPUT_CHAIN as HSSD_CURRENT_INPUT_CHAIN
from benchmark.legend.hssd.hssd_hab_converter import LEGEND_INPUT_CHAIN as HSSD_LEGEND_INPUT_CHAIN
from benchmark.nl_scene import CURRENT_INPUT_CHAIN as NL_CURRENT_INPUT_CHAIN
from benchmark.nl_scene import LEGEND_INPUT_CHAIN as NL_LEGEND_INPUT_CHAIN


def test_natural_language_is_current_input_chain() -> None:
    assert NL_CURRENT_INPUT_CHAIN == "natural_language"
    assert NL_LEGEND_INPUT_CHAIN is False


def test_hssd_is_legend_input_chain() -> None:
    assert HSSD_CURRENT_INPUT_CHAIN == "natural_language"
    assert HSSD_LEGEND_INPUT_CHAIN is True
    assert "legend_hssd_scene_instance_json" in DATASET_ADAPTERS
    assert DATASET_ADAPTERS["hssd_scene_instance_json"] is DATASET_ADAPTERS["legend_hssd_scene_instance_json"]
