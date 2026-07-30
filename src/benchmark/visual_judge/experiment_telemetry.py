from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any


@dataclass
class CameraExperimentTelemetry:
    """Manifest-friendly counters for camera acquisition experiments."""

    policy: str
    started_at: float = field(default_factory=perf_counter)
    deterministic_selector_calls: int = 0
    vlm_selector_dispatches: int = 0
    vlm_selector_calls: int = 0
    judge_calls: int = 0
    deterministic_rounds: int = 0
    vlm_rounds: int = 0
    preview_render_count: int = 0
    full_render_count: int = 0
    selected_view_count: int = 0
    render_gpu_time_seconds: float = 0.0
    candidate_generation_time_seconds: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)

    def record_gate(
        self,
        *,
        phase: str,
        evidence_round: int,
        episode_index: int,
        result: dict[str, Any],
    ) -> None:
        self.events.append(
            {
                "kind": "evidence_gate",
                "phase": str(phase),
                "evidence_round": int(evidence_round),
                "episode_index": int(episode_index),
                "ready": result.get("ready"),
                "camera_repairable": result.get("camera_repairable"),
                "reason_codes": list(result.get("reason_codes") or []),
                "deficiencies": deepcopy(
                    result.get("deficiencies") or []
                ),
            }
        )

    def record_selector(
        self,
        *,
        stage: str,
        outcome: str,
        candidate_count: int,
        filtered_candidate_count: int,
        attempted_candidate_ids: tuple[str, ...],
        selected_view_ids: tuple[str, ...],
        attempted_plan_ids: tuple[str, ...],
        selected_plan_id: str | None,
        selector_backend: str,
        selection_mode: str | None,
        vlm_call_count: int,
        vlm_call_count_source: str,
        evidence_round: int,
        episode_index: int,
        provenance: dict[str, Any],
        relaxed_constraints: tuple[str, ...] = (),
        has_camera_proposal: bool = False,
    ) -> None:
        if stage == "vlm":
            self.vlm_selector_dispatches += 1
            self.vlm_selector_calls += _count(vlm_call_count)
        else:
            self.deterministic_selector_calls += 1
        self.candidate_generation_time_seconds += _duration(
            provenance.get("candidate_generation_time_seconds")
        )
        self.selected_view_count += len(selected_view_ids) + int(
            has_camera_proposal
        )
        self.events.append(
            {
                "kind": "camera_selection",
                "stage": str(stage),
                "selector_backend": str(selector_backend),
                "selection_mode": selection_mode,
                "vlm_call_count": _count(vlm_call_count),
                "vlm_call_count_source": str(
                    vlm_call_count_source
                ),
                "evidence_round": int(evidence_round),
                "episode_index": int(episode_index),
                "outcome": str(outcome),
                "candidate_count": int(candidate_count),
                "filtered_candidate_count": int(
                    filtered_candidate_count
                ),
                "attempted_candidate_ids": list(
                    attempted_candidate_ids
                ),
                "selected_view_ids": list(selected_view_ids),
                "attempted_plan_ids": list(attempted_plan_ids),
                "selected_plan_id": selected_plan_id,
                "relaxed_constraints": list(relaxed_constraints),
                "provenance": deepcopy(provenance),
            }
        )

    def record_render(
        self,
        *,
        stage: str,
        preview_count: int,
        full_count: int,
        gpu_time_seconds: float,
        evidence_round: int,
        episode_index: int,
    ) -> None:
        self.preview_render_count += _count(preview_count)
        self.full_render_count += _count(full_count)
        self.render_gpu_time_seconds += _duration(gpu_time_seconds)
        if stage == "vlm":
            self.vlm_rounds += 1
        else:
            self.deterministic_rounds += 1
        self.events.append(
            {
                "kind": "render",
                "stage": str(stage),
                "evidence_round": int(evidence_round),
                "episode_index": int(episode_index),
                "preview_render_count": _count(preview_count),
                "full_render_count": _count(full_count),
                "render_gpu_time_seconds": _duration(
                    gpu_time_seconds
                ),
            }
        )

    def record_render_failure(
        self,
        *,
        stage: str,
        preview_count: int,
        full_count: int,
        gpu_time_seconds: float,
        evidence_round: int,
        episode_index: int,
        error: str,
    ) -> None:
        """Account for renderer work without treating failure as a round."""

        preview = _count(preview_count)
        full = _count(full_count)
        duration = _duration(gpu_time_seconds)
        self.preview_render_count += preview
        self.full_render_count += full
        self.render_gpu_time_seconds += duration
        self.events.append(
            {
                "kind": "render_failure",
                "stage": str(stage),
                "evidence_round": int(evidence_round),
                "episode_index": int(episode_index),
                "preview_render_count": preview,
                "full_render_count": full,
                "render_gpu_time_seconds": duration,
                "error": str(error),
            }
        )

    def record_judge(
        self,
        *,
        evidence_round: int,
        episode_index: int,
        status: str,
    ) -> None:
        self.judge_calls += 1
        self.events.append(
            {
                "kind": "judge",
                "evidence_round": int(evidence_round),
                "episode_index": int(episode_index),
                "status": str(status),
            }
        )

    def record_escalation(self, value: dict[str, Any]) -> None:
        self.events.append(
            {
                "kind": "camera_escalation",
                **deepcopy(value),
            }
        )

    def to_dict(self, *, stop_reason: str | None = None) -> dict[str, Any]:
        return {
            "camera_policy": self.policy,
            "deterministic_selector_calls": (
                self.deterministic_selector_calls
            ),
            "vlm_selector_calls": self.vlm_selector_calls,
            "vlm_selector_dispatches": self.vlm_selector_dispatches,
            "judge_calls": self.judge_calls,
            "deterministic_rounds": self.deterministic_rounds,
            "vlm_rounds": self.vlm_rounds,
            "preview_render_count": self.preview_render_count,
            "full_render_count": self.full_render_count,
            "selected_view_count": self.selected_view_count,
            "render_gpu_time_seconds": self.render_gpu_time_seconds,
            "candidate_generation_time_seconds": (
                self.candidate_generation_time_seconds
            ),
            "wall_clock_latency_seconds": max(
                0.0,
                perf_counter() - self.started_at,
            ),
            "stop_reason": stop_reason,
            "events": deepcopy(self.events),
        }


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("telemetry render counts must be non-negative integers")
    return value


def _duration(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("telemetry durations must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(
            "telemetry durations must be finite and non-negative"
        )
    return result
