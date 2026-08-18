"""Stable VLM evaluation interface contracts.

Legacy adapters are re-exported here so existing
``benchmark.visual_judge.interfaces`` imports remain compatible.
"""

from benchmark.visual_judge.interfaces.camera import (
    CAMERA_SELECTION_OUTCOMES,
    CameraSelectionOutcome,
    CameraSelectionRequest,
    CameraSelectionResult,
    CameraSelector,
    EVIDENCE_READINESS_OUTCOMES,
    EvidenceReadinessOutcome,
    EvidenceReadinessRequest,
    EvidenceReadinessResult,
    TrustedCameraCandidateBank,
)
from benchmark.visual_judge.interfaces.evidence import (
    EvidenceGate,
    EvidenceGateRequest,
    EvidenceGateResult,
    EvidenceRenderFailure,
    EvidenceRenderRequest,
    EvidenceRenderResult,
    EvidenceRenderer,
)
from benchmark.visual_judge.interfaces.judge import (
    JUDGE_STATUSES,
    EvidenceRequest,
    Judge,
    JudgeRequest,
    JudgeResult,
)
__all__ = [
    "CameraSelectionRequest",
    "CameraSelectionResult",
    "CameraSelectionOutcome",
    "CameraSelector",
    "EvidenceReadinessOutcome",
    "EvidenceReadinessRequest",
    "EvidenceReadinessResult",
    "EVIDENCE_READINESS_OUTCOMES",
    "TrustedCameraCandidateBank",
    "CAMERA_SELECTION_OUTCOMES",
    "DeterministicCameraSelector",
    "EvidenceGate",
    "EvidenceGateRequest",
    "EvidenceGateResult",
    "EvidenceRenderFailure",
    "EvidenceRenderRequest",
    "EvidenceRenderResult",
    "EvidenceRenderer",
    "EvidenceRequest",
    "ExistingCameraSelectorAdapter",
    "ExistingJudgeAdapter",
    "HybridCameraSelector",
    "JUDGE_STATUSES",
    "Judge",
    "JudgeRequest",
    "JudgeResult",
    "VLMCameraSelector",
    "build_camera_selector",
    "camera_selection_result_from_value",
]


def __getattr__(name: str):
    if name == "ExistingJudgeAdapter":
        from benchmark.visual_judge.adapters.legacy_judge import (
            ExistingJudgeAdapter,
        )

        value = ExistingJudgeAdapter
    elif name in {
        "DeterministicCameraSelector",
        "ExistingCameraSelectorAdapter",
        "HybridCameraSelector",
        "VLMCameraSelector",
        "build_camera_selector",
        "camera_selection_result_from_value",
    }:
        from benchmark.visual_judge.adapters import legacy_camera

        value = getattr(legacy_camera, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
