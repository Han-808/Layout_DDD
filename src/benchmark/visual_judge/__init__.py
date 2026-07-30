"""Optional visual-evidence evaluation interfaces."""

from benchmark.visual_judge.active_fallback import (
    ConditionalActiveCameraEvidenceProvider,
    InsufficientVisualEvidenceError,
    build_conditional_active_camera_evidence_provider,
)
from benchmark.visual_judge.active_policy import (
    generate_corrective_camera_proposals,
)
from benchmark.visual_judge.acquisition_planner import (
    EvidenceAcquisitionPlanner,
    MetricAcquisitionPlanningRequest,
    MetricSpecificAcquisitionPlanner,
)
from benchmark.visual_judge.acquisition_state import (
    CameraAcquisitionPolicy,
    CameraAcquisitionState,
)
from benchmark.visual_judge.adapters.active_camera import (
    ActiveVLMCameraSelector,
    CameraPoseValidator,
    CameraRepairSolver,
)
from benchmark.visual_judge.adapters.deterministic_camera import (
    DeterministicCameraRepairSolver,
    DeterministicLocalCameraSelector,
)
from benchmark.visual_judge.camera_dsl import (
    CameraConstraintSet,
    CameraObservation,
)
from benchmark.visual_judge.camera_repair import (
    CameraRepairPlan,
    VLMSelectionMode,
)
from benchmark.visual_judge.camera_ranking import (
    DEFAULT_DETERMINISTIC_CAMERA_RANKING,
    DeterministicCameraRankingConfig,
)
from benchmark.visual_judge.control_config import (
    DEFAULT_VLM_EVALUATION_CONTROL,
    VLMEvaluationControl,
    resolve_vlm_evaluation_control,
)
from benchmark.visual_judge.control_loop import (
    EvidenceRenderRequest,
    EvidenceRenderResult,
    EvidenceRenderer,
    ExistingEvidenceRendererAdapter,
    VLMEvaluationController,
    VLMEvaluationResult,
)
from benchmark.visual_judge.evidence_gate import DeterministicEvidenceGate
from benchmark.visual_judge.evidence_sufficiency import (
    assess_preview_selection_sufficiency,
    assess_visual_evidence_sufficiency,
)
from benchmark.visual_judge.evaluator import evaluate_vlm_category
from benchmark.visual_judge.interfaces import (
    CameraSelectionRequest,
    CameraSelectionResult,
    CameraSelector,
    DeterministicCameraSelector,
    EvidenceGate,
    EvidenceGateRequest,
    EvidenceGateResult,
    EvidenceRequest,
    ExistingCameraSelectorAdapter,
    ExistingJudgeAdapter,
    HybridCameraSelector,
    Judge,
    JudgeRequest,
    JudgeResult,
    VLMCameraSelector,
    build_camera_selector,
)
from benchmark.visual_judge.openai_compatible import (
    OpenAICompatibleVLMJudge,
    build_openai_compatible_vlm_judge,
)
from benchmark.visual_judge.p0b import LocalViewProvider, adjudicate_p0b_event
from benchmark.visual_judge.render_views import CameraEvidenceProvider
from benchmark.visual_judge.roles import DecisionContract, VLMRole
from benchmark.visual_judge.runtime import (
    ControlledVLMJudge,
    EvidenceControlUnresolvedError,
    build_controlled_vlm_judge,
)
from benchmark.visual_judge.visual_config import DEFAULT_P0B_VISUAL_CONFIGS

__all__ = [
    "CameraEvidenceProvider",
    "CameraAcquisitionPolicy",
    "CameraAcquisitionState",
    "CameraConstraintSet",
    "CameraObservation",
    "CameraPoseValidator",
    "CameraRepairPlan",
    "CameraRepairSolver",
    "CameraSelectionRequest",
    "CameraSelectionResult",
    "CameraSelector",
    "ConditionalActiveCameraEvidenceProvider",
    "ControlledVLMJudge",
    "DEFAULT_P0B_VISUAL_CONFIGS",
    "DEFAULT_DETERMINISTIC_CAMERA_RANKING",
    "DEFAULT_VLM_EVALUATION_CONTROL",
    "DecisionContract",
    "DeterministicCameraSelector",
    "DeterministicCameraRepairSolver",
    "DeterministicCameraRankingConfig",
    "DeterministicLocalCameraSelector",
    "DeterministicEvidenceGate",
    "EvidenceGate",
    "EvidenceGateRequest",
    "EvidenceGateResult",
    "EvidenceControlUnresolvedError",
    "EvidenceAcquisitionPlanner",
    "EvidenceRenderRequest",
    "EvidenceRenderResult",
    "EvidenceRenderer",
    "EvidenceRequest",
    "ExistingCameraSelectorAdapter",
    "ExistingEvidenceRendererAdapter",
    "ExistingJudgeAdapter",
    "HybridCameraSelector",
    "InsufficientVisualEvidenceError",
    "Judge",
    "JudgeRequest",
    "JudgeResult",
    "LocalViewProvider",
    "MetricAcquisitionPlanningRequest",
    "MetricSpecificAcquisitionPlanner",
    "OpenAICompatibleVLMJudge",
    "VLMCameraSelector",
    "ActiveVLMCameraSelector",
    "VLMSelectionMode",
    "VLMEvaluationControl",
    "VLMEvaluationController",
    "VLMEvaluationResult",
    "VLMRole",
    "adjudicate_p0b_event",
    "assess_preview_selection_sufficiency",
    "assess_visual_evidence_sufficiency",
    "build_conditional_active_camera_evidence_provider",
    "build_controlled_vlm_judge",
    "build_camera_selector",
    "build_openai_compatible_vlm_judge",
    "evaluate_vlm_category",
    "generate_corrective_camera_proposals",
    "resolve_vlm_evaluation_control",
]
