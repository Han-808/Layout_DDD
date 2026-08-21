"""Frozen two-stage generation compatibility layer.

This package is limited to the standalone Stage A -> retrieval -> Stage C
generation runners.  It does not import or modify the evaluator.  See
``docs/generation_transport_compatibility.md`` for the complete boundary.
"""

from benchmark.scene_generation.frozen_two_stage.artifact_layout import (
    ArtifactLayout,
)
from benchmark.scene_generation.frozen_two_stage.config import (
    FrozenTwoStageRunConfig,
    RetryConfig,
    RouteConfig,
    load_run_config,
)
from benchmark.scene_generation.frozen_two_stage.orchestrator import (
    FrozenTwoStageOrchestrator,
)
from benchmark.scene_generation.frozen_two_stage.providers import (
    ChatOptionPolicy,
    ChatOptionStyle,
    NormalizedResponse,
    ProviderRoute,
    make_api2_chat_route,
    make_api2_responses_route,
    make_api3_chat_route,
)
from benchmark.scene_generation.frozen_two_stage.provenance import (
    compatibility_source_manifest,
)
from benchmark.scene_generation.frozen_two_stage.retry_policy import RetryPolicy
from benchmark.scene_generation.frozen_two_stage.spec import GenerationRunSpec
from benchmark.scene_generation.frozen_two_stage.trust import (
    TrustError,
    TrustInventory,
    TrustReport,
)

__all__ = [
    "ArtifactLayout",
    "ChatOptionPolicy",
    "ChatOptionStyle",
    "FrozenTwoStageOrchestrator",
    "FrozenTwoStageRunConfig",
    "GenerationRunSpec",
    "NormalizedResponse",
    "ProviderRoute",
    "RetryPolicy",
    "RetryConfig",
    "RouteConfig",
    "TrustError",
    "TrustInventory",
    "TrustReport",
    "compatibility_source_manifest",
    "load_run_config",
    "make_api2_chat_route",
    "make_api2_responses_route",
    "make_api3_chat_route",
]
