from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from benchmark.visual_judge.interfaces.evidence import (
    EVIDENCE_MERGE_POLICIES,
    EvidenceRenderFailure,
    EvidenceRenderRequest,
    EvidenceRenderResult,
)
from benchmark.visual_judge.adapters.legacy_camera import (
    EXISTING_PROVIDER_CANDIDATE_ID as _EXISTING_PROVIDER_CANDIDATE_ID,
    provider_observed_usage as _provider_observed_usage,
    provider_policy as _provider_policy,
    qualified_name as _qualified_name,
)
from benchmark.visual_judge.orchestration.budget import (
    selection_action_count as _selection_action_count,
)


class ExistingEvidenceRendererAdapter:
    """Wrap an existing callable evidence provider without changing its packet."""

    def __init__(
        self,
        provider: Callable[[dict[str, Any]], Any],
        *,
        merge_policy: str = "replace",
        backend: str = "existing",
    ) -> None:
        if not callable(provider):
            raise TypeError("existing evidence provider must be callable")
        if merge_policy not in EVIDENCE_MERGE_POLICIES:
            raise ValueError(
                "existing evidence provider merge_policy must be append or replace"
            )
        self.provider = provider
        self.merge_policy = merge_policy
        self.backend = str(backend)

    def render(self, request: EvidenceRenderRequest) -> EvidenceRenderResult:
        raw = self.provider(request.to_dict())
        result = EvidenceRenderResult.from_value(
            raw,
            default_merge_policy=self.merge_policy,
            default_backend=self.backend,
            proposed_action_count=_selection_action_count(request.selection),
        )
        provenance = deepcopy(result.provenance)
        provenance.setdefault("scene_access", "read_only")
        provenance.setdefault(
            "adapter",
            f"{type(self).__module__}.{type(self).__qualname__}",
        )
        provenance.setdefault(
            "provider",
            f"{type(self.provider).__module__}.{type(self.provider).__qualname__}",
        )
        policy = getattr(self.provider, "policy_config", None)
        if isinstance(policy, dict):
            provenance.setdefault("existing_policy", deepcopy(policy))
        return EvidenceRenderResult(
            visual_evidence=result.visual_evidence,
            merge_policy=result.merge_policy,
            camera_actions_executed=result.camera_actions_executed,
            manifest_path=result.manifest_path,
            next_candidate_views=result.next_candidate_views,
            next_allowed_actions=result.next_allowed_actions,
            replaces_candidate_views=result.replaces_candidate_views,
            replaces_allowed_actions=result.replaces_allowed_actions,
            backend=result.backend,
            provenance=provenance,
        )

class _UnavailableEvidenceRenderer:
    def render(self, request: EvidenceRenderRequest) -> EvidenceRenderResult:
        del request
        raise RuntimeError(
            "no camera evidence provider is configured for another round"
        )


class _ExistingProviderEvidenceRenderer:
    """Execute the exact provider acquisition selected by its trusted token."""

    def __init__(
        self,
        provider: Any,
        *,
        usage_consumer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.provider = provider
        self.usage_consumer = usage_consumer

    def render(
        self,
        request: EvidenceRenderRequest,
    ) -> EvidenceRenderResult:
        if request.selection.selected_view_ids != (
            _EXISTING_PROVIDER_CANDIDATE_ID,
        ):
            raise ValueError(
                "existing provider renderer received an unrelated selection"
            )
        provider_request = _provider_request(request)
        try:
            raw = _call_provider(
                self.provider,
                provider_request,
                metric=request.judge_request.metric,
            )
            evidence = _provider_visual_evidence(raw)
            if not evidence:
                raise RuntimeError(
                    "existing camera evidence provider returned no visual "
                    "evidence"
                )
        except Exception as exc:
            try:
                usage = _provider_observed_usage(
                    self.provider,
                    (),
                    metric=request.judge_request.metric,
                )
            except Exception as usage_exc:
                raise EvidenceRenderFailure(
                    (
                        f"{type(exc).__name__}: {exc}; provider usage "
                        f"telemetry unavailable: {type(usage_exc).__name__}: "
                        f"{usage_exc}"
                    ),
                    provenance={
                        "adapter": type(self).__name__,
                        "provider": _qualified_name(self.provider),
                        "provider_policy": _provider_policy(self.provider),
                        "usage_observation_error": (
                            f"{type(usage_exc).__name__}: {usage_exc}"
                        ),
                    },
                ) from exc
            if self.usage_consumer is not None:
                self.usage_consumer(usage)
            raise EvidenceRenderFailure(
                f"{type(exc).__name__}: {exc}",
                internal_selector_calls=int(
                    usage.get("selector_calls", 0)
                ),
                camera_actions_executed=int(
                    usage.get("camera_actions", 0)
                ),
                visual_evidence=tuple(
                    usage.get("evidence_refs") or ()
                ),
                provenance={
                    "adapter": type(self).__name__,
                    "provider": _qualified_name(self.provider),
                    "provider_policy": _provider_policy(self.provider),
                    "provider_usage": deepcopy(usage),
                    "usage_source": (
                        "existing_provider_last_call_usage"
                    ),
                },
            ) from exc
        usage = _provider_observed_usage(
            self.provider,
            raw,
            metric=request.judge_request.metric,
        )
        if self.usage_consumer is not None:
            self.usage_consumer(usage)
        actual_actions = int(usage.get("camera_actions", 0))
        actual_selector_calls = int(usage.get("selector_calls", 0))
        return EvidenceRenderResult(
            visual_evidence=tuple(deepcopy(evidence)),
            merge_policy="append",
            camera_actions_executed=actual_actions,
            backend="existing",
            provenance={
                "adapter": type(self).__name__,
                "provider": _qualified_name(self.provider),
                "provider_policy": _provider_policy(self.provider),
                "selected_acquisition": _EXISTING_PROVIDER_CANDIDATE_ID,
                "scene_access": "read_only",
                "internal_selector_calls": actual_selector_calls,
                "actual_camera_actions": actual_actions,
                "provider_usage": deepcopy(usage),
                "usage_source": "existing_provider_last_call_usage",
            },
        )


def _provider_request(request: EvidenceRenderRequest) -> dict[str, Any]:
    context = request.judge_request.context
    explicit = context.get("camera_evidence_request")
    if isinstance(explicit, dict):
        result = deepcopy(explicit)
    else:
        result = {
            "category": "visual_evidence_request",
            "metric": request.judge_request.metric,
            "event": deepcopy(request.judge_request.claim_or_event),
            "object_ids": list(
                request.evidence_goal.get("target_ids")
                or request.judge_request.claim_or_event.get("object_ids")
                or context.get("target_ids")
                or context.get("target_object_ids")
                or []
            ),
            "scene": deepcopy(request.judge_request.scene_context),
            "detector_evidence": deepcopy(
                request.judge_request.deterministic_evidence
            ),
            "natural_language_prompt": (
                context.get("natural_language_prompt")
                or context.get("prompt")
            ),
            "evidence_scope": "metric_scoped",
        }
    for key in (
        "group_scope",
        "grouping_role",
        "member_ids",
        "target_bounds",
        "focus_center",
        "target_extent",
    ):
        if key in request.context:
            result.setdefault(
                key,
                deepcopy(request.context[key]),
            )
    if isinstance(result.get("group_scope"), dict):
        result["evidence_scope"] = "group_local"
    result["_camera_selection_phase"] = "active_fallback"
    result["_camera_evidence_deficiency"] = deepcopy(
        request.evidence_goal
    )
    result["_vlm_evidence_round"] = request.evidence_round
    result["access"] = "read_only_evidence_request"
    return result


def _call_provider(
    provider: Any,
    request: dict[str, Any],
    *,
    metric: str,
) -> Any:
    if metric in {
        "scale_consistency",
        "object_pairing_consistency",
        "style_consistency",
        "functional_consistency",
    }:
        call = getattr(provider, "provide_scene_quality_evidence", None)
        if callable(call):
            return call(request)
    if callable(provider):
        return provider(request)
    call = getattr(provider, "provide_scene_quality_evidence", None)
    if callable(call):
        return call(request)
    raise TypeError("camera evidence provider is not callable")


def _provider_visual_evidence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(deepcopy(value))
    if not isinstance(value, dict):
        raise ValueError(
            "camera evidence provider response must be a JSON object or list"
        )
    status = str(value.get("status") or "available").lower()
    if status in {"failed", "error", "unavailable", "insufficient"}:
        raise RuntimeError(
            "camera evidence provider reported "
            f"{status}: {value.get('reason') or value.get('error') or ''}"
        )
    for key in (
        "visual_evidence",
        "render_evidence_items",
        "render_evidence",
        "paths",
    ):
        items = value.get(key)
        if isinstance(items, (list, tuple)):
            return list(deepcopy(items))
    raise ValueError(
        "camera evidence provider response does not contain visual evidence"
    )


def _coerce_evidence_renderer(value: Any) -> Any:
    if callable(getattr(value, "render", None)):
        return value
    if callable(value):
        return ExistingEvidenceRendererAdapter(
            value,
            merge_policy="append",
            backend="existing",
        )
    raise TypeError(
        "independent evidence renderer must expose render(request) "
        "or be a selection-aware callable"
    )


UnavailableEvidenceRenderer = _UnavailableEvidenceRenderer
ExistingProviderEvidenceRenderer = _ExistingProviderEvidenceRenderer
coerce_evidence_renderer = _coerce_evidence_renderer
