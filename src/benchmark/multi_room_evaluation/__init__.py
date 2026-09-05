"""Additive bridge from multi-room generation outputs to evaluator cases."""

from benchmark.multi_room_evaluation.campaign import (
    build_existing_evaluation_campaign_config,
    campaign_config_sha256,
    evaluation_campaign_command,
    write_campaign_config,
)
from benchmark.multi_room_evaluation.dataset import (
    EVALUATION_SCOPE,
    UNSUPPORTED_SCOPES,
    MultiRoomEvaluationInventory,
    SucceededRoom,
    UnresolvedRoom,
    VerifiedSourceArtifact,
    discover_multi_room_evaluation_inventory,
)
from benchmark.multi_room_evaluation.materializer import (
    MaterializationResult,
    default_dataset_id,
    materialize_multi_room_evaluation_dataset,
)
from benchmark.multi_room_evaluation.render_profile import (
    OFFICIAL_RENDER_PROFILE,
    OFFICIAL_RENDER_PROFILE_ID,
)

__all__ = [
    "EVALUATION_SCOPE",
    "UNSUPPORTED_SCOPES",
    "MaterializationResult",
    "MultiRoomEvaluationInventory",
    "SucceededRoom",
    "UnresolvedRoom",
    "VerifiedSourceArtifact",
    "OFFICIAL_RENDER_PROFILE",
    "OFFICIAL_RENDER_PROFILE_ID",
    "build_existing_evaluation_campaign_config",
    "campaign_config_sha256",
    "default_dataset_id",
    "discover_multi_room_evaluation_inventory",
    "evaluation_campaign_command",
    "materialize_multi_room_evaluation_dataset",
    "write_campaign_config",
]
