from __future__ import annotations

from benchmark.visual_judge.acquisition_state import (
    CameraAcquisitionPolicy,
    CameraAcquisitionState,
)


NORMAL_ESCALATION_REASONS = {
    "no_feasible_candidate",
    "camera_constraint_conflict",
    "candidate_ranking_exhausted",
    "semantic_selection_required",
}
ENGINEERING_FAILURE_REASONS = {
    "selector_exception",
    "invalid_selector_response",
    "render_failure",
    "manifest_failure",
    "non_camera_repairable_evidence",
    "budget_exhausted",
    "scene_contract_failure",
}


def should_escalate_to_vlm(
    *,
    policy: CameraAcquisitionPolicy,
    deterministic_outcome: str | None,
    acquisition_state: CameraAcquisitionState,
    budget: dict[str, int],
    escalation: dict[str, bool] | None = None,
    reason: str | None = None,
) -> bool:
    """Return whether Controller may perform one normal deterministic→VLM hop."""

    if policy != "deterministic_then_vlm":
        return False
    if acquisition_state.stage != "deterministic":
        return False
    if acquisition_state.vlm_stage_failed:
        return False
    if not _has_remaining_budget(budget):
        return False

    rules = escalation or {}
    resolved_reason = str(reason or "").strip()
    if resolved_reason in ENGINEERING_FAILURE_REASONS:
        return False

    if deterministic_outcome == "no_feasible_candidate":
        resolved_reason = resolved_reason or "no_feasible_candidate"
        if resolved_reason not in NORMAL_ESCALATION_REASONS:
            return False
        return bool(rules.get("on_no_feasible_candidate", True))

    return False


def _has_remaining_budget(value: dict[str, int]) -> bool:
    for key in (
        "remaining_evidence_rounds",
        "remaining_selector_calls",
        "remaining_images",
    ):
        amount = value.get(key)
        if (
            not isinstance(amount, int)
            or isinstance(amount, bool)
            or amount <= 0
        ):
            return False
    return True
