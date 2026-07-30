"""Backward-compatible VLM evaluation runtime imports.

Implementation lives in the focused interfaces, adapters, and orchestration
packages. Existing callers may keep importing this module unchanged.
"""

from benchmark.visual_judge.adapters.legacy_judge import (
    ControlledVLMJudge,
    EvidenceControlUnresolvedError,
    build_controlled_vlm_judge,
    _judge_request,
)

__all__ = [
    "ControlledVLMJudge",
    "EvidenceControlUnresolvedError",
    "build_controlled_vlm_judge",
]
