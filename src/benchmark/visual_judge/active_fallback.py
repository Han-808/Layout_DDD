from __future__ import annotations

import hashlib
import json
import uuid
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
        self.last_call_usage: dict[str, Any] | None = None

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
        manifest_path = event_dir / "active_camera_fallback_manifest.json"
        self._begin_call_usage(metric=metric, manifest_path=manifest_path)

        deterministic_error: str | None = None
        deterministic_prior_call_id = _provider_call_id(
            self.deterministic_provider
        )
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
        finally:
            self._record_internal_provider_usage(
                "deterministic",
                _provider_usage_after_call(
                    self.deterministic_provider,
                    prior_call_id=deterministic_prior_call_id,
                    expected_metric=metric,
                ),
            )

        should_trigger = bool(
            base_assessment.get("status") == INSUFFICIENT
            and base_assessment.get("trigger_recommended") is True
            and base_assessment.get("camera_repairable") is True
        )
        if not should_trigger:
            returned_items = deepcopy(base_items)
            self._finish_call_usage(returned_items)
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
            return returned_items

        active_request = deepcopy(request)
        active_request["_camera_selection_phase"] = "active_fallback"
        active_request["_camera_evidence_deficiency"] = _selector_deficiency(
            base_assessment
        )
        active_error: str | None = None
        active_items: list[Any] = []
        active_prior_call_id = _provider_call_id(self.active_provider)
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
        finally:
            self._record_internal_provider_usage(
                "active",
                _provider_usage_after_call(
                    self.active_provider,
                    prior_call_id=active_prior_call_id,
                    expected_metric=metric,
                ),
            )
        active_sufficient = bool(
            active_error is None
            and final_assessment.get("status") == SUFFICIENT
        )
        active_used = bool(
            not self.shadow_mode and active_sufficient
        )
        returned_items = deepcopy(
            base_items
            if self.shadow_mode or not active_sufficient
            else active_items
        )
        self._finish_call_usage(returned_items)
        self._write_manifest(
            event_dir,
            request=request,
            base_assessment=base_assessment,
            final_assessment=final_assessment,
            active_used=active_used,
            active_attempted=True,
            deterministic_error=deterministic_error,
            active_error=active_error,
        )
        if self.shadow_mode:
            return returned_items
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
        return returned_items

    def _begin_call_usage(
        self,
        *,
        metric: str,
        manifest_path: Path,
    ) -> None:
        self.last_call_usage = {
            "call_id": uuid.uuid4().hex,
            "metric": metric,
            "cache_hit": False,
            "evidence_refs": [],
            "manifest_path": str(manifest_path),
            "selector_calls": 0,
            "camera_actions": 0,
            "source": type(self).__name__,
            "observability": {
                "schema_version": "conditional_provider_usage_v1",
                "internal_calls": {},
            },
        }

    def _record_internal_provider_usage(
        self,
        label: str,
        usage: dict[str, Any] | None,
    ) -> None:
        if self.last_call_usage is None:
            return
        observability = self.last_call_usage["observability"]
        internal_calls = observability["internal_calls"]
        if usage is None:
            internal_calls[label] = {"observed": False}
        else:
            internal_calls[label] = {
                "observed": True,
                **deepcopy(usage),
            }
            self.last_call_usage["selector_calls"] += usage[
                "selector_calls"
            ]
            self.last_call_usage["camera_actions"] += usage[
                "camera_actions"
            ]
        self.last_call_usage["cache_hit"] = bool(internal_calls) and all(
            call.get("observed") is True
            and call.get("cache_hit") is True
            for call in internal_calls.values()
        )

    def _finish_call_usage(self, evidence: list[Any]) -> None:
        if self.last_call_usage is not None:
            self.last_call_usage["evidence_refs"] = _usage_evidence_refs(
                evidence
            )

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
                "call_usage": deepcopy(self.last_call_usage),
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
    architecture_contract: dict[str, Any] | None = None,
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
        architecture_contract=architecture_contract,
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
        architecture_contract=architecture_contract,
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


def _provider_call_id(provider: Any) -> Any:
    usage = getattr(provider, "last_call_usage", None)
    return usage.get("call_id") if isinstance(usage, dict) else None


def _provider_usage_after_call(
    provider: Any,
    *,
    prior_call_id: Any,
    expected_metric: str,
) -> dict[str, Any] | None:
    usage = getattr(provider, "last_call_usage", None)
    if not isinstance(usage, dict):
        return None
    call_id = usage.get("call_id")
    if (
        not isinstance(call_id, (str, int))
        or isinstance(call_id, bool)
        or call_id == prior_call_id
        or usage.get("metric") != expected_metric
        or not isinstance(usage.get("cache_hit"), bool)
        or not isinstance(usage.get("evidence_refs"), list)
        or any(
            not isinstance(value, str)
            for value in usage["evidence_refs"]
        )
    ):
        return None
    selector_calls = usage.get("selector_calls")
    camera_actions = usage.get("camera_actions")
    if (
        isinstance(selector_calls, bool)
        or not isinstance(selector_calls, int)
        or selector_calls < 0
        or isinstance(camera_actions, bool)
        or not isinstance(camera_actions, int)
        or camera_actions < 0
    ):
        return None
    return {
        "call_id": call_id,
        "metric": expected_metric,
        "cache_hit": usage["cache_hit"],
        "evidence_refs": list(usage["evidence_refs"]),
        "manifest_path": (
            str(usage["manifest_path"])
            if usage.get("manifest_path") is not None
            else None
        ),
        "selector_calls": selector_calls,
        "camera_actions": camera_actions,
        "source": usage.get("source"),
    }


def _usage_evidence_refs(evidence: list[Any]) -> list[str]:
    refs: list[str] = []
    for index, item in enumerate(evidence):
        if isinstance(item, dict):
            value = (
                item.get("view_id")
                or item.get("id")
                or item.get("path")
                or item.get("image_path")
            )
        else:
            value = item
        refs.append(
            str(value)
            if value is not None
            else f"evidence_{index:02d}"
        )
    return refs
