from __future__ import annotations

from enum import Enum


class VLMRole(str, Enum):
    """The only two model-backed boundaries in the visual evaluation runtime."""

    JUDGE = "judge"
    VLM_CAMERA_SELECTOR = "vlm_camera_selector"


class DecisionContract(str, Enum):
    """Stable identifiers for the response contract used by one VLM call."""

    CANONICAL_METRIC = "canonical_metric_v1"
    P0B_BINARY = "p0b_binary_v1"
    RELATION_BINARY = "relation_binary_v1"
    CAMERA_SELECTION = "camera_selection_v1"
    GENERIC_VISUAL_SCORE = "generic_visual_score_v1"
    # Compatibility-only routed Spatial Fidelity candidates remain binary.
    SPATIAL_FIDELITY_BINARY = "spatial_fidelity_binary_v1"


def vlm_audit_metadata(
    role: VLMRole,
    *,
    decision_contract: DecisionContract,
    judge_method: str | None = None,
) -> dict[str, str]:
    """Build additive, transport-independent audit metadata."""

    metadata = {
        "vlm_role": role.value,
        "decision_contract": decision_contract.value,
    }
    if judge_method:
        metadata["judge_method"] = str(judge_method)
    return metadata
