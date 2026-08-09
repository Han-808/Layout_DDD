from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.visual_judge.control_config import VLMEvaluationControl
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionResult,
    CameraSelector,
)
from benchmark.visual_judge.interfaces.evidence import EvidenceRenderResult


def _policy_for_stop_reason(
    control: VLMEvaluationControl,
    stop_reason: str,
) -> dict[str, str] | None:
    if stop_reason == "evidence_not_camera_repairable":
        return {
            "field": "on_non_camera_repairable_evidence",
            "value": control.on_non_camera_repairable_evidence,
        }
    if (
        stop_reason.startswith("max_")
        or "_stage_rounds_exhausted" in stop_reason
    ):
        return {
            "field": "on_budget_exhausted",
            "value": control.on_budget_exhausted,
        }
    if stop_reason == "camera_selector_failed":
        return {
            "field": "on_selector_failure",
            "value": control.on_selector_failure,
        }
    if stop_reason == "render_failed":
        return {
            "field": "on_render_failure",
            "value": control.on_render_failure,
        }
    return None


def _selection_action_count(result: CameraSelectionResult) -> int:
    return max(
        len(result.camera_actions),
        1 if result.camera_proposal is not None else 0,
    )


def _trusted_composite_reservation(
    selector: CameraSelector,
    result: CameraSelectionResult,
) -> dict[str, int | bool]:
    trusted = bool(
        getattr(
            selector,
            "trusted_composite_provider_adapter",
            False,
        )
    )
    if not trusted:
        return {
            "trusted": False,
            "selector_calls": 0,
            "camera_actions": 0,
            "full_artifacts_per_selected_view": 1,
        }
    provenance = result.provenance
    selector_calls = _nonnegative_int(
        provenance.get("max_internal_selector_calls", 0),
        "max_internal_selector_calls",
    )
    camera_actions = _nonnegative_int(
        provenance.get("max_internal_camera_actions", 0),
        "max_internal_camera_actions",
    )
    full_artifacts = _nonnegative_int(
        provenance.get("full_artifacts_per_selected_view", 1),
        "full_artifacts_per_selected_view",
    )
    if full_artifacts < 1:
        raise ValueError(
            "full_artifacts_per_selected_view must be positive"
        )
    return {
        "trusted": True,
        "selector_calls": selector_calls,
        "camera_actions": camera_actions,
        "full_artifacts_per_selected_view": full_artifacts,
    }


def _rendered_internal_selector_calls(
    result: EvidenceRenderResult,
) -> int:
    return _nonnegative_int(
        result.provenance.get("internal_selector_calls", 0),
        "internal_selector_calls",
    )


def _normalize_camera_usage(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {
            "observed": False,
            "selector_calls": 0,
            "camera_actions": 0,
        }
    if not isinstance(value, dict):
        raise TypeError("initial_camera_usage must be a JSON object")
    result = deepcopy(value)
    result["selector_calls"] = _nonnegative_int(
        value.get("selector_calls", 0),
        "initial_camera_usage.selector_calls",
    )
    result["camera_actions"] = _nonnegative_int(
        value.get("camera_actions", 0),
        "initial_camera_usage.camera_actions",
    )
    result["observed"] = True
    return result


def _normalize_acquisition_ledger(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate additive usage for the caller-defined budget scope."""

    if value is None:
        return {
            "observed": False,
            "schema_version": "metric_camera_acquisition_ledger_v1",
            "artifact_ids": [],
            "total_images_acquired": 0,
            "evidence_rounds": 0,
            "selector_calls": 0,
            "camera_actions": 0,
            "deterministic_rounds": 0,
            "vlm_rounds": 0,
        }
    if not isinstance(value, dict):
        raise TypeError("initial_acquisition_ledger must be a JSON object")
    artifact_ids = value.get("artifact_ids")
    if artifact_ids is None:
        artifact_ids = []
    if not isinstance(artifact_ids, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in artifact_ids
    ):
        raise ValueError(
            "initial_acquisition_ledger.artifact_ids must be a list of "
            "non-empty strings"
        )
    normalized = {
        "observed": True,
        "schema_version": "metric_camera_acquisition_ledger_v1",
        "artifact_ids": list(dict.fromkeys(artifact_ids)),
    }
    for key in (
        "total_images_acquired",
        "evidence_rounds",
        "selector_calls",
        "camera_actions",
        "deterministic_rounds",
        "vlm_rounds",
    ):
        normalized[key] = _nonnegative_int(
            value.get(key, 0),
            f"initial_acquisition_ledger.{key}",
        )
    if normalized["total_images_acquired"] < len(
        normalized["artifact_ids"]
    ):
        raise ValueError(
            "initial_acquisition_ledger.total_images_acquired cannot be "
            "smaller than its unique artifact count"
        )
    return normalized


def _extend_acquisition_ledger(
    value: dict[str, Any] | None,
    *,
    artifact_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Register already-created images without resetting scope counters."""

    normalized = _normalize_acquisition_ledger(value)
    result = {
        key: deepcopy(item)
        for key, item in normalized.items()
        if key != "observed"
    }
    existing = set(result["artifact_ids"])
    added = 0
    for raw in artifact_ids:
        artifact_id = str(raw).strip()
        if not artifact_id or artifact_id in existing:
            continue
        result["artifact_ids"].append(artifact_id)
        existing.add(artifact_id)
        added += 1
    result["total_images_acquired"] = (
        int(result["total_images_acquired"]) + added
    )
    return result


def _merge_acquisition_ledger_delta(
    aggregate: dict[str, Any] | None,
    *,
    episode_before: dict[str, Any] | None,
    episode_after: dict[str, Any],
) -> dict[str, Any]:
    """Merge one independently budgeted Judge episode into audit totals."""

    before = _normalize_acquisition_ledger(episode_before)
    after = _normalize_acquisition_ledger(episode_after)
    merged = _extend_acquisition_ledger(
        aggregate,
        artifact_ids=after["artifact_ids"],
    )
    for key in (
        "evidence_rounds",
        "selector_calls",
        "camera_actions",
        "deterministic_rounds",
        "vlm_rounds",
    ):
        delta = max(0, int(after[key]) - int(before[key]))
        merged[key] = int(merged.get(key) or 0) + delta
    return merged


def _usage_overrun_stop_reason(
    *,
    control: VLMEvaluationControl,
    selector_calls: int,
    camera_actions: int,
) -> str | None:
    if selector_calls > control.max_selector_calls:
        return "max_selector_calls_exhausted"
    if camera_actions > control.max_camera_actions:
        return "max_camera_actions_exhausted"
    return None


def _budget_stop_reason(
    *,
    control: VLMEvaluationControl,
    rounds_used: int,
    selector_calls: int,
    total_images_acquired: int,
) -> str | None:
    if rounds_used >= control.max_evidence_rounds:
        return "max_evidence_rounds_exhausted"
    if selector_calls >= control.max_selector_calls:
        return "max_selector_calls_exhausted"
    if total_images_acquired >= control.max_total_images:
        return "max_total_images_exhausted"
    return None


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


selection_action_count = _selection_action_count
policy_for_stop_reason = _policy_for_stop_reason
trusted_composite_reservation = _trusted_composite_reservation
rendered_internal_selector_calls = _rendered_internal_selector_calls
normalize_camera_usage = _normalize_camera_usage
normalize_acquisition_ledger = _normalize_acquisition_ledger
extend_acquisition_ledger = _extend_acquisition_ledger
merge_acquisition_ledger_delta = _merge_acquisition_ledger_delta
usage_overrun_stop_reason = _usage_overrun_stop_reason
budget_stop_reason = _budget_stop_reason
