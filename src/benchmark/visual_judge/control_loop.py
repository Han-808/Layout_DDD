"""Compatibility imports for the VLM evaluation controller.

New code should import stable contracts from interfaces, compatibility
adapters from adapters, and orchestration from orchestration.
"""

from benchmark.visual_judge.adapters.legacy_renderer import (
    ExistingEvidenceRendererAdapter,
)
from benchmark.visual_judge.interfaces.evidence import (
    EVIDENCE_MERGE_POLICIES,
    EvidenceRenderFailure,
    EvidenceRenderRequest,
    EvidenceRenderResult,
    EvidenceRenderer,
)
from benchmark.visual_judge.orchestration.controller import (
    EVALUATION_STATUSES,
    VLM_CONTROL_LOOP_VERSION,
    VLMEvaluationController,
    VLMEvaluationResult,
)

__all__ = [
    "EVALUATION_STATUSES",
    "EVIDENCE_MERGE_POLICIES",
    "EvidenceRenderFailure",
    "EvidenceRenderRequest",
    "EvidenceRenderResult",
    "EvidenceRenderer",
    "ExistingEvidenceRendererAdapter",
    "VLM_CONTROL_LOOP_VERSION",
    "VLMEvaluationController",
    "VLMEvaluationResult",
]
