"""Local scene-generation interfaces and implementations."""

from benchmark.scene_generation.interfaces import (
    BatchGenerationResult,
    InputBatchInfo,
    SceneGenerationResult,
    SceneGenerator,
)
from benchmark.scene_generation.sceneeval_hy4 import LocalSceneEvalHy4Generator


_LAZY_CAMPAIGN_EXPORTS = frozenset(
    {
        "PreparedGenerationCampaign",
        "check_generation_campaign",
        "preflight_generation_campaign",
        "prepare_generation_campaign",
        "resolve_generation_campaign",
        "resource_gate_generation_campaign",
        "run_generation_campaign",
    }
)


def __getattr__(name: str):
    if name not in _LAZY_CAMPAIGN_EXPORTS:
        raise AttributeError(name)
    from benchmark.scene_generation.campaign import api

    return getattr(api, name)

__all__ = [
    "BatchGenerationResult",
    "InputBatchInfo",
    "LocalSceneEvalHy4Generator",
    "PreparedGenerationCampaign",
    "SceneGenerationResult",
    "SceneGenerator",
    "check_generation_campaign",
    "preflight_generation_campaign",
    "prepare_generation_campaign",
    "resolve_generation_campaign",
    "resource_gate_generation_campaign",
    "run_generation_campaign",
]
