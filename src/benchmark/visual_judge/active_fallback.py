from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from benchmark.rendering.camera_pose import DEFAULT_CAMERA_CANDIDATE_POLICY
from benchmark.utils.io import write_json
from benchmark.visual_judge.evidence_sufficiency import (
    EVIDENCE_SUFFICIENCY_VERSION,
    INSUFFICIENT,
    REPAIR_RERENDER,
    SUFFICIENT,
    assess_visual_evidence_sufficiency,
)
from benchmark.visual_judge.render_views import CameraEvidenceProvider


ACTIVE_CAMERA_FALLBACK_VERSION = "conditional_vlm_active_camera_fallback_v2"


class InsufficientVisualEvidenceError(RuntimeError):
    """Raised when the bounded active-camera fallback exhausts its budget."""


class ConditionalActiveCameraEvidenceProvider:
    """Run VLM-active view search only after a deterministic insufficiency gate."""

    def __init__(
        self,
        *,
        deterministic_provider: Callable[[dict[str, Any]], list[Any]],
        active_provider: Callable[[dict[str, Any]], list[Any]],
        out_dir: str | Path,
        max_views: int,
        max_steps: int,
        fail_on_exhausted: bool = True,
        shadow_mode: bool = True,
    ) -> None:
        self.deterministic_provider = deterministic_provider
        self.active_provider = active_provider
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.max_views = int(max_views)
        self.max_steps = int(max_steps)
        self.fail_on_exhausted = bool(fail_on_exhausted)
        self.shadow_mode = bool(shadow_mode)
        if not 1 <= self.max_views <= 4:
            raise ValueError("active fallback max_views must be between 1 and 4")
        if not 0 <= self.max_steps <= 3:
            raise ValueError("active fallback max_steps must be between 0 and 3")

    @property
    def policy_config(self) -> dict[str, Any]:
        return {
            "schema_version": ACTIVE_CAMERA_FALLBACK_VERSION,
            "trigger": "deterministic_evidence_sufficiency_eq_insufficient",
            "unknown_triggers_active_search": False,
            "selector_role": "camera_selection_only_no_metric_verdict",
            "judge_decoupled": True,
            "max_views": self.max_views,
            "max_camera_actions": self.max_steps,
            "max_selector_calls": self.max_steps + 1,
            "fail_on_exhausted": self.fail_on_exhausted,
            "shadow_mode": self.shadow_mode,
            "official_packet_policy": (
                "deterministic_unchanged"
                if self.shadow_mode
                else "active_when_sufficient"
            ),
            "deterministic_policy": deepcopy(
                getattr(self.deterministic_provider, "policy_config", None)
            ),
            "active_policy": deepcopy(
                getattr(self.active_provider, "policy_config", None)
            ),
        }

    def __call__(self, request: dict[str, Any]) -> list[Any]:
        if not isinstance(request, dict):
            raise TypeError("active camera fallback request must be a JSON object")
        metric = str(request.get("metric") or "").strip().lower()
        event_dir = self.out_dir / _event_key(
            request,
            policy_config=self.policy_config,
        )
        event_dir.mkdir(parents=True, exist_ok=True)

        deterministic_error: str | None = None
        try:
            base_items = self.deterministic_provider(request)
            base_assessment = assess_visual_evidence_sufficiency(
                metric,
                base_items,
                request=request,
            )
        except RuntimeError as exc:
            base_items = []
            deterministic_error = f"{type(exc).__name__}: {exc}"
            base_assessment = {
                "schema_version": EVIDENCE_SUFFICIENCY_VERSION,
                "metric": metric,
                "status": INSUFFICIENT,
                "trigger_recommended": False,
                "camera_repairable": False,
                "repairability": REPAIR_RERENDER,
                "reason_codes": ["deterministic_evidence_provider_failed"],
                "deficiencies": [
                    {
                        "code": "deterministic_evidence_provider_failed",
                        "repairability": REPAIR_RERENDER,
                    }
                ],
            }

        should_trigger = bool(
            base_assessment.get("status") == INSUFFICIENT
            and base_assessment.get("trigger_recommended") is True
            and base_assessment.get("camera_repairable") is True
        )
        if not should_trigger:
            self._write_manifest(
                event_dir,
                request=request,
                base_assessment=base_assessment,
                final_assessment=base_assessment,
                active_used=False,
                active_attempted=False,
                deterministic_error=deterministic_error,
                active_error=None,
            )
            if deterministic_error and self.fail_on_exhausted:
                raise InsufficientVisualEvidenceError(
                    "deterministic visual evidence failed with a non-camera-"
                    f"repairable error: {deterministic_error}"
                )
            return deepcopy(base_items)

        active_request = deepcopy(request)
        active_request["_camera_selection_phase"] = "active_fallback"
        active_request["_camera_evidence_deficiency"] = _selector_deficiency(
            base_assessment
        )
        active_error: str | None = None
        active_items: list[Any] = []
        try:
            active_items = self.active_provider(active_request)
            final_assessment = assess_visual_evidence_sufficiency(
                metric,
                active_items,
                request=request,
            )
        except Exception as exc:
            active_error = f"{type(exc).__name__}: {exc}"
            final_assessment = {
                **base_assessment,
                "status": INSUFFICIENT,
                "reason_codes": ["active_camera_execution_failed"],
                "trigger_recommended": False,
                "camera_repairable": False,
                "repairability": REPAIR_RERENDER,
                "deficiencies": [
                    {
                        "code": "active_camera_execution_failed",
                        "repairability": REPAIR_RERENDER,
                    }
                ],
            }
        self._write_manifest(
            event_dir,
            request=request,
            base_assessment=base_assessment,
            final_assessment=final_assessment,
            active_used=bool(
                not self.shadow_mode
                and active_error is None
                and final_assessment.get("status") == SUFFICIENT
            ),
            active_attempted=True,
            deterministic_error=deterministic_error,
            active_error=active_error,
        )
        active_sufficient = bool(
            active_error is None
            and final_assessment.get("status") == SUFFICIENT
        )
        if self.shadow_mode:
            return deepcopy(base_items)
        if self.fail_on_exhausted and not active_sufficient:
            reasons = ", ".join(final_assessment.get("reason_codes") or ["unknown"])
            raise InsufficientVisualEvidenceError(
                "bounded VLM-active camera fallback exhausted without sufficient "
                f"{metric} evidence: {reasons}"
            )
        # In non-shadow diagnostic mode, ``fail_on_exhausted=False`` means
        # "keep the deterministic packet on failed repair", not "promote an
        # insufficient or empty active packet".  This keeps the returned packet
        # aligned with ``official_packet_source`` in the manifest.
        return deepcopy(active_items if active_sufficient else base_items)

    def _write_manifest(
        self,
        event_dir: Path,
        *,
        request: dict[str, Any],
        base_assessment: dict[str, Any],
        final_assessment: dict[str, Any],
        active_used: bool,
        active_attempted: bool,
        deterministic_error: str | None,
        active_error: str | None,
    ) -> None:
        would_replace = bool(
            active_attempted
            and active_error is None
            and final_assessment.get("status") == SUFFICIENT
        )
        write_json(
            event_dir / "active_camera_fallback_manifest.json",
            {
                "schema_version": ACTIVE_CAMERA_FALLBACK_VERSION,
                "request_sha256": _canonical_sha256(request),
                "metric": request.get("metric"),
                "object_ids": request.get("object_ids"),
                "policy": self.policy_config,
                "active_used": active_used,
                "active_attempted": active_attempted,
                "shadow_mode": self.shadow_mode,
                "counterfactual_would_replace": would_replace,
                "official_packet_source": (
                    "deterministic"
                    if self.shadow_mode or not active_used
                    else "active"
                ),
                "deterministic_assessment": base_assessment,
                "deterministic_error": deterministic_error,
                "active_error": active_error,
                "active_assessment": (
                    final_assessment if active_attempted else None
                ),
                "final_assessment": final_assessment,
                "official_assessment": (
                    final_assessment if active_used else base_assessment
                ),
                "budget": {
                    "max_views": self.max_views,
                    "max_camera_actions": self.max_steps,
                    "max_selector_calls": self.max_steps + 1,
                },
            },
        )


def build_conditional_active_camera_evidence_provider(
    *,
    renderer: Any,
    blend_file: str | Path,
    out_dir: str | Path,
    deterministic_mode: str,
    selector: Any,
    metric_modes: dict[str, str] | None = None,
    max_views: int = 2,
    max_steps: int = 1,
    candidate_count: int = 5,
    collision_overlay: bool = True,
    collision_contour: bool = True,
    collision_geometry: dict[str, Any] | None = None,
    highlighted_global_pose_policy: str = "global_top",
    candidate_policy: str = DEFAULT_CAMERA_CANDIDATE_POLICY,
    fail_on_exhausted: bool = True,
    shadow_mode: bool = True,
) -> ConditionalActiveCameraEvidenceProvider:
    if deterministic_mode == "query_cov" or "query_cov" in set((metric_modes or {}).values()):
        raise ValueError(
            "conditional active fallback requires a deterministic base camera policy"
        )
    if not callable(getattr(selector, "select_camera_views", None)):
        raise TypeError(
            "conditional active fallback requires a separate selector exposing "
            "select_camera_views(request)"
        )
    selector_max_images = getattr(selector, "max_images", None)
    if (
        isinstance(selector_max_images, int)
        and not isinstance(selector_max_images, bool)
        and int(candidate_count) > selector_max_images
    ):
        raise ValueError(
            "active camera candidate_count exceeds selector max_images; "
            "implicit candidate-bank truncation is forbidden"
        )
    destination = Path(out_dir).expanduser().resolve()
    deterministic = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend_file,
        out_dir=destination / "deterministic",
        mode=deterministic_mode,
        selector=None,
        max_views=max_views,
        max_steps=0,
        candidate_count=candidate_count,
        metric_modes=metric_modes,
        collision_overlay=collision_overlay,
        collision_contour=collision_contour,
        collision_geometry=collision_geometry,
        highlighted_global_pose_policy=highlighted_global_pose_policy,
        candidate_policy=candidate_policy,
    )
    active = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend_file,
        out_dir=destination / "active",
        mode="query_cov",
        selector=selector,
        max_views=max_views,
        max_steps=max_steps,
        candidate_count=candidate_count,
        collision_overlay=collision_overlay,
        collision_contour=collision_contour,
        collision_geometry=collision_geometry,
        highlighted_global_pose_policy=highlighted_global_pose_policy,
        candidate_policy=candidate_policy,
        active_repair=True,
    )
    return ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=destination,
        max_views=max_views,
        max_steps=max_steps,
        fail_on_exhausted=fail_on_exhausted,
        shadow_mode=shadow_mode,
    )


def _selector_deficiency(assessment: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "metric",
        "status",
        "reason_codes",
        "required_local_view_count",
        "measured_local_view_count",
        "usable_local_view_count",
        "evidence_utility",
        "repairability",
        "camera_repairable",
        "deficiencies",
    }
    return {
        key: deepcopy(value)
        for key, value in assessment.items()
        if key in allowed
    }


def _event_key(
    request: dict[str, Any],
    *,
    policy_config: dict[str, Any],
) -> str:
    metric = str(request.get("metric") or "event").strip().lower()
    digest = _canonical_sha256(
        {
            "metric": request.get("metric"),
            "event": request.get("event"),
            "object_ids": request.get("object_ids"),
            "detector_evidence": request.get("detector_evidence"),
            "policy": policy_config,
        }
    )[:12]
    return f"{metric or 'event'}__{digest}"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
