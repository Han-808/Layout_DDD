from __future__ import annotations

from enum import Enum


class VLMRole(str, Enum):
    """Explicit audit roles for every model-backed evaluation call."""

    JUDGE = "judge"
    VLM_CAMERA_SELECTOR = "vlm_camera_selector"
    VLM_GROUPING = "vlm_grouping"
    FUNCTIONAL_AFFORDANCE_DISCOVERY = (
        "functional_affordance_discovery"
    )
    FUNCTIONAL_RELATION_DISCOVERY = "functional_relation_discovery"
    FUNCTIONAL_EVIDENCE_PLANNER = "functional_evidence_planner"
    PLACEMENT_DISCOVERY = "placement_discovery"
    USABLE_SURFACE_DECODER = "usable_surface_decoder"


class DecisionContract(str, Enum):
    """Stable identifiers for the response contract used by one VLM call."""

    CANONICAL_METRIC = "canonical_metric_v1"
    P0B_BINARY = "p0b_binary_v1"
    RELATION_BINARY = "relation_binary_v1"
    CAMERA_SELECTION = "camera_selection_v1"
    GENERIC_VISUAL_SCORE = "generic_visual_score_v1"
    # Compatibility-only routed Spatial Fidelity candidates remain binary.
    SPATIAL_FIDELITY_BINARY = "spatial_fidelity_binary_v1"
    GROUPING_PARTITION = "grouping_partition_v1"
    FUNCTIONAL_AFFORDANCE_DISCOVERY = (
        "functional_affordance_ledger_v4"
    )
    FUNCTIONAL_RELATION_DISCOVERY = "functional_relation_audit_v3"
    FUNCTIONAL_PROBE_PLAN = "functional_probe_plan_v2"
    PLACEMENT_DISCOVERY = "placement_discovery_v2"
    USABLE_SURFACE_DECODE = "usable_surface_decode_v2"


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
