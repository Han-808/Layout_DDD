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
    DETERMINISTIC_SUPPORTED_OBSERVATIONS,
    SEMANTIC_SELECTION_OBSERVATIONS,
    DeterministicCameraRepairSolver,
    DeterministicLocalCameraSelector,
    TrustedTechnicalCameraCandidateBankBuilder,
)
from benchmark.visual_judge.adapters.legacy_renderer import (
    CameraCandidatePreviewRenderer,
    CameraViewEvidenceRenderer,
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
from benchmark.visual_judge.functional_evidence import (
    FUNCTIONAL_PROBE_KINDS,
    FUNCTIONAL_PROBE_DEFAULT_UNITS,
    FUNCTIONAL_PROBE_MAX_UNITS,
    FUNCTIONAL_PROBE_PLANNER_PROMPT_VERSION,
    FUNCTIONAL_PROBE_PLAN_VERSION,
)
from benchmark.visual_judge.functional_discovery import (
    FUNCTIONAL_DISCOVERY_PROMPT_VERSION,
    FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
    FUNCTIONAL_RELATION_PREDICATES,
    FUNCTIONAL_SURFACE_ROLES,
    FunctionalDiscoveryResult,
)
from benchmark.visual_judge.evaluator import evaluate_vlm_category
from benchmark.visual_judge.interfaces import (
    CameraSelectionRequest,
    CameraSelectionResult,
    CameraSelector,
    TrustedCameraCandidateBank,
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
from benchmark.visual_judge.openai_camera_selector import (
    CAMERA_SELECTOR_PROMPT_VERSION,
    OpenAICompatibleCameraSelector,
    build_openai_compatible_camera_selector,
)
from benchmark.visual_judge.p0b import LocalViewProvider, adjudicate_p0b_event
from benchmark.visual_judge.render_views import CameraEvidenceProvider
from benchmark.visual_judge.roles import DecisionContract, VLMRole
from benchmark.visual_judge.usable_surface import (
    DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND,
    USABLE_SURFACE_DETECTOR_INTERFACE_VERSION,
    USABLE_SURFACE_PROMPT_VERSION,
    USABLE_SURFACE_SCHEMA_VERSION,
    USABLE_SURFACE_SIDE_IDS,
    UsableSurfaceDetector,
    VLMTrustedSideUsableSurfaceDetector,
    build_usable_surface_detector,
)
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
    "CameraCandidatePreviewRenderer",
    "CameraViewEvidenceRenderer",
    "ConditionalActiveCameraEvidenceProvider",
    "ControlledVLMJudge",
    "DEFAULT_P0B_VISUAL_CONFIGS",
    "DEFAULT_DETERMINISTIC_CAMERA_RANKING",
    "DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND",
    "DETERMINISTIC_SUPPORTED_OBSERVATIONS",
    "SEMANTIC_SELECTION_OBSERVATIONS",
    "DEFAULT_VLM_EVALUATION_CONTROL",
    "DecisionContract",
    "DeterministicCameraSelector",
    "DeterministicCameraRepairSolver",
    "DeterministicCameraRankingConfig",
    "DeterministicLocalCameraSelector",
    "TrustedTechnicalCameraCandidateBankBuilder",
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
    "FUNCTIONAL_PROBE_KINDS",
    "FUNCTIONAL_PROBE_DEFAULT_UNITS",
    "FUNCTIONAL_PROBE_MAX_UNITS",
    "FUNCTIONAL_PROBE_PLANNER_PROMPT_VERSION",
    "FUNCTIONAL_PROBE_PLAN_VERSION",
    "FUNCTIONAL_DISCOVERY_PROMPT_VERSION",
    "FUNCTIONAL_DISCOVERY_SCHEMA_VERSION",
    "FUNCTIONAL_RELATION_PREDICATES",
    "FUNCTIONAL_SURFACE_ROLES",
    "FunctionalDiscoveryResult",
    "HybridCameraSelector",
    "InsufficientVisualEvidenceError",
    "Judge",
    "JudgeRequest",
    "JudgeResult",
    "LocalViewProvider",
    "MetricAcquisitionPlanningRequest",
    "MetricSpecificAcquisitionPlanner",
    "OpenAICompatibleVLMJudge",
    "CAMERA_SELECTOR_PROMPT_VERSION",
    "OpenAICompatibleCameraSelector",
    "TrustedCameraCandidateBank",
    "VLMCameraSelector",
    "ActiveVLMCameraSelector",
    "VLMSelectionMode",
    "VLMEvaluationControl",
    "VLMEvaluationController",
    "VLMEvaluationResult",
    "VLMRole",
    "USABLE_SURFACE_PROMPT_VERSION",
    "USABLE_SURFACE_SCHEMA_VERSION",
    "USABLE_SURFACE_SIDE_IDS",
    "USABLE_SURFACE_DETECTOR_INTERFACE_VERSION",
    "UsableSurfaceDetector",
    "VLMTrustedSideUsableSurfaceDetector",
    "adjudicate_p0b_event",
    "assess_preview_selection_sufficiency",
    "assess_visual_evidence_sufficiency",
    "build_conditional_active_camera_evidence_provider",
    "build_controlled_vlm_judge",
    "build_camera_selector",
    "build_openai_compatible_vlm_judge",
    "build_openai_compatible_camera_selector",
    "build_usable_surface_detector",
    "evaluate_vlm_category",
    "generate_corrective_camera_proposals",
    "resolve_vlm_evaluation_control",
]
