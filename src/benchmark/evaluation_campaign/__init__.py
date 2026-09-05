"""Config-driven orchestration around the frozen scene evaluator.

This package deliberately treats ``scripts/run_camera_cal_scene_level.py`` and
``scripts/select_first_publishable_scene_evaluations.py`` as subprocess
boundaries.  It must not import the evaluator, its prompts, camera policy, or
the generation transport compatibility package.
"""

from benchmark.evaluation_campaign.config import (
    AttemptPolicy,
    CampaignConfigError,
    EvaluationCampaignSpec,
    JudgeProfile,
    LocalBinding,
    load_campaign,
    load_local_bindings,
    load_profile_registry,
)
from benchmark.evaluation_campaign.dataset_identity import (
    EvaluationCaseIdentity,
    EvaluationDatasetIdentity,
    inspect_evaluation_dataset,
)
from benchmark.evaluation_campaign.orchestrator import (
    CampaignResult,
    EvaluationCampaignOrchestrator,
)
from benchmark.evaluation_campaign.routes import (
    ResolvedJudgeRoute,
    open_judge_route,
)

__all__ = [
    "AttemptPolicy",
    "CampaignConfigError",
    "EvaluationCampaignSpec",
    "EvaluationCaseIdentity",
    "EvaluationCampaignOrchestrator",
    "EvaluationDatasetIdentity",
    "JudgeProfile",
    "LocalBinding",
    "CampaignResult",
    "ResolvedJudgeRoute",
    "inspect_evaluation_dataset",
    "load_campaign",
    "load_local_bindings",
    "load_profile_registry",
    "open_judge_route",
]
