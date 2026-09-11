"""Observable wrappers used by the camera-cal runtime.

The wrappers are deliberately dependency-light: concrete models, renderers,
providers, trackers, and progress reporters are supplied by the compatibility
facade.  This keeps event accounting reusable without importing the historical
runner or evaluation-campaign orchestration.
"""

from __future__ import annotations

import time
from typing import Any

from benchmark.camera_cal_scene_level.telemetry import (
    evidence_count,
    request_scope,
    safe_request_metadata,
)
from benchmark.models import EndpointConfigurationError


class ObservedChatModel:
    """Record one logical chat call through an injected API-call tracker."""

    def __init__(
        self,
        model: Any,
        *,
        role: str,
        tracker: Any,
    ) -> None:
        self._model = model
        self._role = str(role)
        self._tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        if self._tracker.model_route_abort_signal is not None:
            self._tracker.model_route_abort_signal.raise_if_set()
        call_type = str(kwargs.get("call_type") or "chat")
        call_number, call_id, started = self._tracker.begin_call(
            role=self._role,
            call_type=call_type,
            messages=messages,
        )
        try:
            result = self._model.chat_messages(messages, **kwargs)
        except Exception as exc:
            self._tracker.finish_call(
                call_number=call_number,
                call_id=call_id,
                role=self._role,
                call_type=call_type,
                started=started,
                request_metadata=safe_request_metadata(self._model),
                error=exc,
            )
            if (
                isinstance(exc, EndpointConfigurationError)
                and self._tracker.model_route_abort_signal is not None
            ):
                self._tracker.model_route_abort_signal.trip(exc)
            raise
        self._tracker.finish_call(
            call_number=call_number,
            call_id=call_id,
            role=self._role,
            call_type=call_type,
            started=started,
            request_metadata=safe_request_metadata(self._model),
            error=None,
        )
        return result


class ObservedEvidenceProvider:
    """Emit evidence-render lifecycle events around an injected provider."""

    def __init__(
        self,
        provider: Any,
        *,
        phase: str,
        case_id: str,
        progress: Any,
    ) -> None:
        self._provider = provider
        self._phase = str(phase)
        self._case_id = str(case_id)
        self._progress = progress

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def __call__(self, request: dict[str, Any]) -> Any:
        return self._invoke(request)

    def provide_scene_quality_evidence(
        self,
        request: dict[str, Any],
    ) -> Any:
        return self._invoke(request)

    def _invoke(self, request: dict[str, Any]) -> Any:
        metric, group_id = request_scope(request)
        started = time.monotonic()
        self._progress.emit(
            "evidence_render_started",
            case_id=self._case_id,
            phase=self._phase,
            metric=metric,
            group_id=group_id,
        )
        try:
            result = self._provider(request)
        except Exception as exc:
            self._progress.emit(
                "evidence_render_failed",
                case_id=self._case_id,
                phase=self._phase,
                metric=metric,
                group_id=group_id,
                duration_seconds=round(
                    max(0.0, time.monotonic() - started),
                    3,
                ),
                error_type=type(exc).__name__,
            )
            raise
        usage = getattr(self._provider, "last_call_usage", None)
        usage = usage if isinstance(usage, dict) else {}
        self._progress.emit(
            "evidence_render_completed",
            case_id=self._case_id,
            phase=self._phase,
            metric=metric,
            group_id=group_id,
            duration_seconds=round(
                max(0.0, time.monotonic() - started),
                3,
            ),
            evidence_count=evidence_count(result),
            cache_hit=usage.get("cache_hit"),
            internal_selector_calls=usage.get("selector_calls"),
            camera_actions=usage.get("camera_actions"),
        )
        return result


class ObservedRenderer:
    """Emit repair-render lifecycle events around an injected renderer."""

    def __init__(
        self,
        renderer: Any,
        *,
        phase: str,
        case_id: str,
        progress: Any,
    ) -> None:
        self._renderer = renderer
        self._phase = str(phase)
        self._case_id = str(case_id)
        self._progress = progress

    def __getattr__(self, name: str) -> Any:
        return getattr(self._renderer, name)

    def render(self, request: Any) -> Any:
        metric = str(getattr(request, "metric", "") or "unknown")
        context = getattr(request, "context", None)
        context = context if isinstance(context, dict) else {}
        group_scope = context.get("group_scope")
        group_id = (
            str(group_scope.get("group_id") or "scene")
            if isinstance(group_scope, dict)
            else "scene"
        )
        started = time.monotonic()
        self._progress.emit(
            "repair_render_started",
            case_id=self._case_id,
            phase=self._phase,
            metric=metric,
            group_id=group_id,
        )
        try:
            result = self._renderer.render(request)
        except Exception as exc:
            self._progress.emit(
                "repair_render_failed",
                case_id=self._case_id,
                phase=self._phase,
                metric=metric,
                group_id=group_id,
                duration_seconds=round(
                    max(0.0, time.monotonic() - started),
                    3,
                ),
                error_type=type(exc).__name__,
            )
            raise
        self._progress.emit(
            "repair_render_completed",
            case_id=self._case_id,
            phase=self._phase,
            metric=metric,
            group_id=group_id,
            duration_seconds=round(
                max(0.0, time.monotonic() - started),
                3,
            ),
            evidence_count=evidence_count(result),
        )
        return result


__all__ = [
    "ObservedChatModel",
    "ObservedEvidenceProvider",
    "ObservedRenderer",
]
