"""Current natural-language scene input workflow."""

CURRENT_INPUT_CHAIN = "natural_language"
LEGEND_INPUT_CHAIN = False

from benchmark.nl_scene.asset_retrieval import retrieve_assets_for_scene_spec
from benchmark.nl_scene.converter import convert_nl_to_scene_spec
from benchmark.nl_scene.dummy_evaluator import evaluate_scene
from benchmark.nl_scene.workflow import run_nl_scene_workflow

__all__ = [
    "CURRENT_INPUT_CHAIN",
    "LEGEND_INPUT_CHAIN",
    "convert_nl_to_scene_spec",
    "evaluate_scene",
    "retrieve_assets_for_scene_spec",
    "run_nl_scene_workflow",
]
