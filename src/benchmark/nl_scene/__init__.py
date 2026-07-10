"""Natural-language conversion helpers for the current harness."""

CURRENT_INPUT_CHAIN = "natural_language"
LEGEND_INPUT_CHAIN = False

from benchmark.nl_scene.asset_retrieval import retrieve_assets_for_object_plan
from benchmark.nl_scene.converter import convert_nl_to_object_plan
from benchmark.nl_scene.generation_input import (
    DIRECT_NATURAL_LANGUAGE_INPUT_MODE,
    STRUCTURED_ASSETS_INPUT_MODE,
    STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE,
    build_direct_natural_language_generation_input,
    build_generation_input,
    build_natural_language_generator_input,
    build_scene_request,
    build_structured_generator_input,
)

__all__ = [
    "CURRENT_INPUT_CHAIN",
    "DIRECT_NATURAL_LANGUAGE_INPUT_MODE",
    "LEGEND_INPUT_CHAIN",
    "STRUCTURED_ASSETS_INPUT_MODE",
    "STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE",
    "build_direct_natural_language_generation_input",
    "build_generation_input",
    "build_natural_language_generator_input",
    "build_scene_request",
    "build_structured_generator_input",
    "convert_nl_to_object_plan",
    "retrieve_assets_for_object_plan",
]
