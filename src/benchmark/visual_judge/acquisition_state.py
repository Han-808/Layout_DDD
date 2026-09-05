from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CameraAcquisitionPolicy = Literal[
    "fixed",
    "deterministic_only",
    "vlm_only",
    "deterministic_then_vlm",
]
CameraAcquisitionStage = Literal["deterministic", "vlm"]

CAMERA_ACQUISITION_POLICIES = {
    "fixed",
    "deterministic_only",
    "vlm_only",
    "deterministic_then_vlm",
}


@dataclass
class CameraAcquisitionState:
    """Controller-owned state for one metric's camera-repair episodes."""

    policy: CameraAcquisitionPolicy
    stage: CameraAcquisitionStage
    deterministic_rounds_used: int = 0
    vlm_rounds_used: int = 0
    total_rounds_used: int = 0
    attempted_view_ids: tuple[str, ...] = ()
    attempted_plan_ids: tuple[str, ...] = ()
    last_selection_outcome: str | None = None
    escalation_reason: str | None = None
    episode_index: int = 0
    total_deterministic_rounds_used: int = 0
    total_vlm_rounds_used: int = 0
    last_render_stage: CameraAcquisitionStage | None = None
    vlm_stage_failed: bool = False

    @classmethod
    def create(
        cls,
        policy: CameraAcquisitionPolicy,
        *,
        total_rounds_used: int = 0,
        total_deterministic_rounds_used: int = 0,
        total_vlm_rounds_used: int = 0,
    ) -> CameraAcquisitionState:
        resolved = _policy(policy)
        return cls(
            policy=resolved,
            stage=_initial_stage(resolved),
            total_rounds_used=max(0, int(total_rounds_used)),
            total_deterministic_rounds_used=max(
                0,
                int(total_deterministic_rounds_used),
            ),
            total_vlm_rounds_used=max(
                0,
                int(total_vlm_rounds_used),
            ),
        )

    def start_episode(self) -> None:
        """Restart stage-local search without resetting shared total budgets."""

        self.episode_index += 1
        self.stage = _initial_stage(self.policy)
        self.deterministic_rounds_used = 0
        self.vlm_rounds_used = 0
        self.last_selection_outcome = None
        self.escalation_reason = None
        self.last_render_stage = None
        self.vlm_stage_failed = False

    def record_selection(
        self,
        *,
        outcome: str,
        attempted_view_ids: tuple[str, ...] = (),
        attempted_plan_ids: tuple[str, ...] = (),
    ) -> None:
        self.last_selection_outcome = str(outcome)
        self.attempted_view_ids = _ordered_union(
            self.attempted_view_ids,
            attempted_view_ids,
        )
        self.attempted_plan_ids = _ordered_union(
            self.attempted_plan_ids,
            attempted_plan_ids,
        )

    def record_render_round(self) -> None:
        self.last_render_stage = self.stage
        self.total_rounds_used += 1
        if self.stage == "deterministic":
            self.deterministic_rounds_used += 1
            self.total_deterministic_rounds_used += 1
        else:
            self.vlm_rounds_used += 1
            self.total_vlm_rounds_used += 1

    def escalate(self, reason: str) -> None:
        if self.stage != "deterministic":
            raise ValueError("camera cascade can only escalate from deterministic")
        self.stage = "vlm"
        self.escalation_reason = str(reason)

    def mark_vlm_failed(self, outcome: str) -> None:
        self.vlm_stage_failed = True
        self.last_selection_outcome = str(outcome)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "stage": self.stage,
            "deterministic_rounds_used": self.deterministic_rounds_used,
            "vlm_rounds_used": self.vlm_rounds_used,
            "total_rounds_used": self.total_rounds_used,
            "attempted_view_ids": list(self.attempted_view_ids),
            "attempted_plan_ids": list(self.attempted_plan_ids),
            "last_selection_outcome": self.last_selection_outcome,
            "escalation_reason": self.escalation_reason,
            "episode_index": self.episode_index,
            "total_deterministic_rounds_used": (
                self.total_deterministic_rounds_used
            ),
            "total_vlm_rounds_used": self.total_vlm_rounds_used,
            "last_render_stage": self.last_render_stage,
            "vlm_stage_failed": self.vlm_stage_failed,
        }


def _initial_stage(
    policy: CameraAcquisitionPolicy,
) -> CameraAcquisitionStage:
    return "vlm" if policy == "vlm_only" else "deterministic"


def _policy(value: str) -> CameraAcquisitionPolicy:
    resolved = str(value).strip().lower()
    if resolved not in CAMERA_ACQUISITION_POLICIES:
        raise ValueError(
            "camera acquisition policy must be fixed, deterministic_only, "
            "vlm_only, or deterministic_then_vlm"
        )
    return resolved  # type: ignore[return-value]


def _ordered_union(
    existing: tuple[str, ...],
    additions: tuple[str, ...],
) -> tuple[str, ...]:
    values = [
        str(value)
        for value in (*existing, *additions)
        if str(value).strip()
    ]
    return tuple(dict.fromkeys(values))
