"""Optional visual-evidence evaluation interfaces."""

from benchmark.visual_judge.evaluator import evaluate_vlm_category
from benchmark.visual_judge.openai_compatible import (
    OpenAICompatibleVLMJudge,
    build_openai_compatible_vlm_judge,
)
from benchmark.visual_judge.p0b import LocalViewProvider, adjudicate_p0b_event
from benchmark.visual_judge.render_views import CameraEvidenceProvider
from benchmark.visual_judge.visual_config import DEFAULT_P0B_VISUAL_CONFIGS

__all__ = [
    "CameraEvidenceProvider",
    "DEFAULT_P0B_VISUAL_CONFIGS",
    "LocalViewProvider",
    "OpenAICompatibleVLMJudge",
    "adjudicate_p0b_event",
    "build_openai_compatible_vlm_judge",
    "evaluate_vlm_category",
]
