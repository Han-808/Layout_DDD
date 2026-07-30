from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


CameraSelectionOutcome = Literal[
    "selected",
    "no_feasible_candidate",
]
CAMERA_SELECTION_OUTCOMES = {
    "selected",
    "no_feasible_candidate",
}


@dataclass(frozen=True)
class CameraSelectionRequest:
    task: str
    metric: str
    target_ids: tuple[str, ...]
    scene: dict[str, Any]
    evidence_goal: dict[str, Any]
    existing_visual_evidence: tuple[Any, ...]
    budget: dict[str, int]
    constraints: dict[str, Any] = field(default_factory=dict)
    candidate_views: tuple[dict[str, Any], ...] = ()
    allowed_actions: tuple[str, ...] = ()
    evidence_round: int = 0
    allow_freeform_pose: bool = False
    allow_scene_mutation: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.allow_scene_mutation is not False:
            raise ValueError(
                "CameraSelector scene access is read-only; "
                "allow_scene_mutation cannot be enabled"
            )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "task": self.task,
            "metric": self.metric,
            "target_ids": list(self.target_ids),
            "scene": deepcopy(self.scene),
            "evidence_goal": deepcopy(self.evidence_goal),
            "existing_visual_evidence": list(
                deepcopy(self.existing_visual_evidence)
            ),
            "budget": deepcopy(self.budget),
            "camera_constraints": deepcopy(self.constraints),
            "candidate_views": list(deepcopy(self.candidate_views)),
            "allowed_actions": list(self.allowed_actions),
            "evidence_round": self.evidence_round,
            "allow_freeform_pose": self.allow_freeform_pose,
            "allow_scene_mutation": self.allow_scene_mutation,
            "scene_access": "read_only",
            "context": deepcopy(self.context),
        }
        # Existing selectors consume these historical names.
        result["candidates"] = list(deepcopy(self.candidate_views))
        result["max_views"] = _positive_int(
            self.budget.get("max_views_per_round"),
            "CameraSelector max_views_per_round",
        )
        result["object_ids"] = list(self.target_ids)
        # Keep implementation-specific hints out of the stable dataclass while
        # still allowing the existing selector wrapper to consume its historical
        # request keys unchanged.
        for key in (
            "allow_adjustment",
            "color_legend",
            "corrective_proposals",
            "decision_contract",
            "evidence_deficiency",
            "judge_method",
            "preview_degradation",
            "preview_role",
            "preview_visibility_warning",
            "selection_phase",
            "vlm_role",
        ):
            if key in self.context and key not in result:
                result[key] = deepcopy(self.context[key])
        return result


@dataclass(frozen=True)
class CameraSelectionResult:
    selected_view_ids: tuple[str, ...] = ()
    outcome: str = "selected"
    selected_views: tuple[dict[str, Any], ...] = ()
    camera_proposal: dict[str, Any] | None = None
    camera_actions: tuple[dict[str, Any], ...] = ()
    attempted_candidate_ids: tuple[str, ...] = ()
    rejected_candidates: tuple[dict[str, Any], ...] = ()
    reason_codes: tuple[str, ...] = ()
    attempted_plan_ids: tuple[str, ...] = ()
    selected_plan_id: str | None = None
    reason: str = ""
    backend: str = "unknown"
    evidence_round: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "selected_view_ids": list(self.selected_view_ids),
            "selected_views": list(deepcopy(self.selected_views)),
            "camera_proposal": deepcopy(self.camera_proposal),
            "camera_actions": list(deepcopy(self.camera_actions)),
            "attempted_candidate_ids": list(
                self.attempted_candidate_ids
            ),
            "rejected_candidates": list(
                deepcopy(self.rejected_candidates)
            ),
            "reason_codes": list(self.reason_codes),
            "attempted_plan_ids": list(self.attempted_plan_ids),
            "selected_plan_id": self.selected_plan_id,
            "reason": self.reason,
            "backend": self.backend,
            "evidence_round": self.evidence_round,
            "provenance": deepcopy(self.provenance),
        }


@runtime_checkable
class CameraSelector(Protocol):
    def select(self, request: CameraSelectionRequest) -> CameraSelectionResult: ...


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value
