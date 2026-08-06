from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
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


@dataclass(frozen=True)
class CameraCandidatePreviewResult:
    candidates: tuple[dict[str, Any], ...]
    manifest_path: str | None
    provenance: dict[str, Any]


class CameraCandidatePreviewRenderer:
    """Render trusted candidate aliases for selection, never Judge evidence."""

    backend = "trusted_camera_candidate_preview_renderer"

    def __init__(
        self,
        *,
        renderer: Any,
        blend_file: str | Path,
        out_dir: str | Path,
    ) -> None:
        if not callable(getattr(renderer, "render_camera_views", None)):
            raise TypeError(
                "CameraCandidatePreviewRenderer requires "
                "renderer.render_camera_views(...)"
            )
        source = Path(blend_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"camera preview source blend does not exist: {source}"
            )
        self.renderer = renderer
        self.blend_file = source
        self.out_dir = Path(out_dir).expanduser().resolve()

    def render(
        self,
        request: Any,
    ) -> CameraCandidatePreviewResult:
        candidates = list(deepcopy(request.candidate_views))
        if not candidates:
            raise ValueError(
                "camera preview rendering requires trusted candidates"
            )
        if any(
            not isinstance(item, dict)
            or item.get("technical_feasibility") is not True
            or not isinstance(item.get("pose"), dict)
            for item in candidates
        ):
            raise ValueError(
                "camera preview rendering accepts only trusted technical "
                "candidates"
            )
        group_scope = request.context.get("group_scope")
        group_id = (
            str(group_scope.get("group_id") or "scene")
            if isinstance(group_scope, dict)
            else "scene"
        )
        destination = (
            self.out_dir
            / _slug(request.metric)
            / _slug(group_id)
            / f"episode_{int(request.context.get('camera_repair_episode', 0)):02d}"
            / "candidate_previews"
        )
        started = perf_counter()
        manifest = self.renderer.render_camera_views(
            blend_file=self.blend_file,
            out_dir=destination,
            camera_views=candidates,
            preview=True,
        )
        duration = max(0.0, perf_counter() - started)
        views = manifest.get("views")
        if not isinstance(views, list) or not views:
            raise RuntimeError(
                "camera preview renderer returned no preview views"
            )
        paths_by_id: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(views):
            if not isinstance(item, dict):
                raise RuntimeError(
                    "camera preview manifest contains a non-object view"
                )
            candidate_id = str(
                item.get("id")
                or item.get("name")
                or (
                    candidates[index].get("id")
                    if index < len(candidates)
                    else ""
                )
                or ""
            ).strip()
            if candidate_id not in {
                str(candidate.get("id"))
                for candidate in candidates
            }:
                raise RuntimeError(
                    "camera preview manifest returned an unknown candidate ID"
                )
            path = Path(
                str(item.get("path") or item.get("image_path") or "")
            ).expanduser()
            _validate_preview_image(path, alias=candidate_id)
            paths_by_id[candidate_id] = {
                "image_path": str(path),
                "render_status": "ok",
                "preview_metadata": {
                    key: deepcopy(value)
                    for key, value in item.items()
                    if key not in {"path", "image_path"}
                },
            }
        if set(paths_by_id) != {
            str(candidate.get("id")) for candidate in candidates
        }:
            raise RuntimeError(
                "camera preview manifest did not render every trusted candidate"
            )
        enriched = tuple(
            {
                **candidate,
                **paths_by_id[str(candidate["id"])],
            }
            for candidate in candidates
        )
        manifest_path = destination / "camera_render_manifest.json"
        return CameraCandidatePreviewResult(
            candidates=enriched,
            manifest_path=(
                str(manifest_path)
                if manifest_path.is_file()
                else None
            ),
            provenance={
                "backend": self.backend,
                "source_blend": str(self.blend_file),
                "scene_access": "read_only",
                "source_blend_modified": (
                    manifest.get("camera_evidence", {}).get(
                        "source_blend_modified"
                    )
                ),
                "preview_render_count": len(enriched),
                "full_render_count": 0,
                "wall_clock_render_seconds": duration,
                **_optional_render_gpu_time(manifest),
                "group_id": group_id,
                "candidate_ids": [
                    str(candidate["id"]) for candidate in enriched
                ],
            },
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


class CameraViewEvidenceRenderer:
    """Render Controller-selected poses through an existing scene renderer."""

    backend = "existing_camera_view_renderer"

    def __init__(
        self,
        *,
        renderer: Any,
        blend_file: str | Path,
        out_dir: str | Path,
    ) -> None:
        if not callable(getattr(renderer, "render_camera_views", None)):
            raise TypeError(
                "CameraViewEvidenceRenderer requires "
                "renderer.render_camera_views(...)"
            )
        source = Path(blend_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"camera evidence source blend does not exist: {source}"
            )
        self.renderer = renderer
        self.blend_file = source
        self.out_dir = Path(out_dir).expanduser().resolve()

    def render(
        self,
        request: EvidenceRenderRequest,
    ) -> EvidenceRenderResult:
        poses = list(deepcopy(request.selection.selected_views))
        if not poses and request.selection.camera_proposal is not None:
            proposal = deepcopy(request.selection.camera_proposal)
            proposal.setdefault(
                "id",
                "freeform_pose_"
                + _selection_fingerprint(request)[:10],
            )
            poses = [proposal]
        if not poses or any(not isinstance(pose, dict) for pose in poses):
            raise ValueError(
                "selected camera evidence must include verifiable pose objects"
            )
        group_scope = request.context.get("group_scope")
        group_id = (
            str(group_scope.get("group_id") or "scene")
            if isinstance(group_scope, dict)
            else "scene"
        )
        destination = (
            self.out_dir
            / _slug(request.judge_request.metric)
            / _slug(group_id)
            / (
                f"round_{request.evidence_round:02d}_"
                + _selection_fingerprint(request)[:12]
            )
        )
        started = perf_counter()
        manifest = self.renderer.render_camera_views(
            blend_file=self.blend_file,
            out_dir=destination,
            camera_views=poses,
            preview=False,
        )
        duration = max(0.0, perf_counter() - started)
        views = manifest.get("views")
        if not isinstance(views, list) or not views:
            raise RuntimeError(
                "camera evidence renderer returned no final views"
            )
        pose_by_id = {
            str(pose.get("id") or ""): pose for pose in poses
        }
        focus_targets = list(
            request.evidence_goal.get("target_ids")
            or request.judge_request.context.get(
                "target_object_ids", []
            )
        )
        evidence: list[dict[str, Any]] = []
        for index, item in enumerate(views):
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("image_path")
            if path is None or not str(path).strip():
                continue
            view_id = str(
                item.get("id")
                or item.get("name")
                or f"view_{index:02d}"
            )
            evidence.append(
                {
                    "path": str(path),
                    "role": (
                        "group_local"
                        if isinstance(group_scope, dict)
                        else "metric_local"
                    ),
                    "view_id": view_id,
                    "representation": "rgb",
                    "camera_scope": (
                        "group_local"
                        if isinstance(group_scope, dict)
                        else "metric_local"
                    ),
                    "pose": deepcopy(pose_by_id.get(view_id)),
                    "group_id": (
                        group_id
                        if isinstance(group_scope, dict)
                        else None
                    ),
                    "member_ids": (
                        deepcopy(group_scope.get("member_ids"))
                        if isinstance(group_scope, dict)
                        else None
                    ),
                    "focus_target_ids": focus_targets,
                }
            )
        if not evidence:
            raise RuntimeError(
                "camera evidence renderer returned no usable image paths"
            )
        manifest_path = destination / "camera_render_manifest.json"
        return EvidenceRenderResult(
            visual_evidence=tuple(evidence),
            merge_policy="append",
            camera_actions_executed=0,
            manifest_path=(
                str(manifest_path)
                if manifest_path.is_file()
                else None
            ),
            backend=self.backend,
            provenance={
                "adapter": (
                    f"{type(self).__module__}.{type(self).__qualname__}"
                ),
                "renderer": (
                    f"{type(self.renderer).__module__}."
                    f"{type(self.renderer).__qualname__}"
                ),
                "source_blend": str(self.blend_file),
                "selected_view_ids": list(
                    request.selection.selected_view_ids
                ),
                "full_render_count": len(evidence),
                "preview_render_count": 0,
                **_optional_render_gpu_time(manifest),
                "wall_clock_render_seconds": duration,
                "evidence_round": request.evidence_round,
                "group_id": (
                    group_id
                    if isinstance(group_scope, dict)
                    else None
                ),
                "scene_access": "read_only",
            },
        )


def _optional_render_gpu_time(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Omit unavailable timing instead of violating numeric provenance."""

    value = manifest.get("render_gpu_time_seconds")
    if value is None:
        return {"render_gpu_time_source": "not_reported"}
    return {
        "render_gpu_time_seconds": value,
        "render_gpu_time_source": "renderer_manifest",
    }


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
        provider_evidence: list[Any] = []
        try:
            raw = _call_provider(
                self.provider,
                provider_request,
                metric=request.judge_request.metric,
            )
            provider_evidence = _provider_visual_evidence(raw)
            if not provider_evidence:
                raise RuntimeError(
                    "existing camera evidence provider returned no visual "
                    "evidence"
                )
            evidence, budget_audit = _fit_provider_evidence_to_budget(
                provider_evidence,
                remaining_images=request.budget.get("remaining_images"),
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
            acquired_artifact_paths = (
                _usage_acquired_artifact_paths(
                    usage,
                    provider_evidence,
                )
            )
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
                    "acquired_artifact_paths": (
                        acquired_artifact_paths
                    ),
                    "full_render_count": len(
                        acquired_artifact_paths
                    ),
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
        acquired_artifact_paths = _usage_acquired_artifact_paths(
            usage,
            provider_evidence,
        )
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
                "acquired_artifact_paths": (
                    acquired_artifact_paths
                ),
                "full_render_count": len(acquired_artifact_paths),
                "usage_source": "existing_provider_last_call_usage",
                "provider_evidence_budget": budget_audit,
            },
        )


def _usage_acquired_artifact_paths(
    usage: dict[str, Any],
    evidence: list[Any],
) -> list[str]:
    values = usage.get("acquired_artifact_paths")
    if isinstance(values, list):
        return list(dict.fromkeys(str(item) for item in values))
    result: list[str] = []
    for item in evidence:
        if isinstance(item, dict):
            value = item.get("path") or item.get("image_path")
        else:
            value = item
        if isinstance(value, (str, Path)) and str(value).strip():
            path = str(value)
            if path not in result:
                result.append(path)
    return result


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
        "semantic_placement_consistency",
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


def _fit_provider_evidence_to_budget(
    evidence: list[Any],
    *,
    remaining_images: Any,
) -> tuple[list[Any], dict[str, Any]]:
    """Keep complete same-pose bundles within the Controller image budget."""

    if (
        isinstance(remaining_images, bool)
        or not isinstance(remaining_images, int)
        or remaining_images < 1
    ):
        if remaining_images is None:
            return list(deepcopy(evidence)), {
                "applied": False,
                "reason": "remaining_images_not_supplied",
                "provider_evidence_count": len(evidence),
                "returned_evidence_count": len(evidence),
            }
        raise ValueError(
            "existing provider remaining_images budget must be a positive "
            "integer"
        )
    if len(evidence) <= remaining_images:
        return list(deepcopy(evidence)), {
            "applied": True,
            "trimmed": False,
            "remaining_images": remaining_images,
            "provider_evidence_count": len(evidence),
            "returned_evidence_count": len(evidence),
        }

    bundles: dict[str, list[tuple[int, Any]]] = {}
    bundle_order: list[str] = []
    for index, item in enumerate(evidence):
        key = _provider_evidence_bundle_key(item, index=index)
        if key not in bundles:
            bundles[key] = []
            bundle_order.append(key)
        bundles[key].append((index, item))

    selected_indices: set[int] = set()
    selected_bundle_keys: list[str] = []
    for key in bundle_order:
        bundle = bundles[key]
        if len(selected_indices) + len(bundle) > remaining_images:
            continue
        selected_indices.update(index for index, _ in bundle)
        selected_bundle_keys.append(key)
    if not selected_indices:
        raise RuntimeError(
            "existing provider evidence bundles do not fit the remaining "
            "image budget"
        )
    selected = [
        deepcopy(item)
        for index, item in enumerate(evidence)
        if index in selected_indices
    ]
    return selected, {
        "applied": True,
        "trimmed": True,
        "remaining_images": remaining_images,
        "provider_evidence_count": len(evidence),
        "returned_evidence_count": len(selected),
        "selected_bundle_keys": selected_bundle_keys,
        "dropped_evidence_count": len(evidence) - len(selected),
    }


def _provider_evidence_bundle_key(item: Any, *, index: int) -> str:
    if isinstance(item, dict):
        pair_id = str(item.get("pair_id") or "").strip()
        if pair_id:
            return f"pair:{pair_id}"
        view_id = str(item.get("view_id") or item.get("id") or "").strip()
        if view_id:
            return f"view:{view_id}"
        pose = item.get("pose")
        if isinstance(pose, dict):
            pose_id = str(pose.get("id") or "").strip()
            if pose_id:
                return f"pose:{pose_id}"
    return f"artifact:{index:04d}"


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


def _selection_fingerprint(request: EvidenceRenderRequest) -> str:
    payload = json.dumps(
        {
            "metric": request.judge_request.metric,
            "evidence_round": request.evidence_round,
            "selection": request.selection.to_dict(),
            "group_scope": request.context.get("group_scope"),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _slug(value: str) -> str:
    return (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        or "item"
    )


def _validate_preview_image(path: Path, *, alias: str) -> None:
    if not path.is_file():
        raise ValueError(
            f"camera candidate preview {alias!r} is missing"
        )
    try:
        from PIL import Image, ImageStat, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Pillow is required to validate camera candidate previews"
        ) from exc
    try:
        with Image.open(path) as source:
            source.load()
            image = source.convert("RGB")
            if image.width <= 0 or image.height <= 0:
                raise ValueError("preview has invalid dimensions")
            extrema = ImageStat.Stat(image).extrema
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"camera candidate preview {alias!r} is corrupt or undecodable"
        ) from exc
    if all(low == high for low, high in extrema):
        raise ValueError(
            f"camera candidate preview {alias!r} is blank"
        )
