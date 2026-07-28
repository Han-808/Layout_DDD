"""Optional visual-evidence evaluation interfaces."""

from benchmark.visual_judge.active_fallback import (
    ConditionalActiveCameraEvidenceProvider,
    InsufficientVisualEvidenceError,
    build_conditional_active_camera_evidence_provider,
)
from benchmark.visual_judge.active_policy import (
    generate_corrective_camera_proposals,
)
from benchmark.visual_judge.evidence_sufficiency import (
    assess_preview_selection_sufficiency,
    assess_visual_evidence_sufficiency,
)
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
    "ConditionalActiveCameraEvidenceProvider",
    "DEFAULT_P0B_VISUAL_CONFIGS",
    "InsufficientVisualEvidenceError",
    "LocalViewProvider",
    "OpenAICompatibleVLMJudge",
    "adjudicate_p0b_event",
    "assess_preview_selection_sufficiency",
    "assess_visual_evidence_sufficiency",
    "build_conditional_active_camera_evidence_provider",
    "build_openai_compatible_vlm_judge",
    "evaluate_vlm_category",
    "generate_corrective_camera_proposals",
]
