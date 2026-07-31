from __future__ import annotations

import hashlib
import json
import re
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.rendering.camera_pose import (
    CAMERA_ACTIONS,
    DEFAULT_CAMERA_CANDIDATE_POLICY,
    DEFAULT_CAMERA_MODE_BY_METRIC,
    apply_camera_action,
    generate_camera_pose_candidates,
    generate_global_context_poses,
    normalize_camera_candidate_policy,
    resolve_camera_pose_mode,
    select_bbox_track_views,
    validate_camera_pose_mode,
    validate_metric_camera_modes,
)
from benchmark.rendering.collision_overlay import (
    COLLISION_EVIDENCE_PACKET_VERSION,
    COLLISION_VISIBILITY_SELECTOR_VERSION,
    JOINT_VISIBILITY_IMPOSSIBLE_OR_OCCLUDED,
    build_candidate_mask_stats,
    build_collision_overlay_spec,
    build_focus_overlay_spec,
    measure_focus_visibility,
    measure_overlay_visibility,
    rank_collision_candidates,
    rank_collision_candidates_v2,
    rank_focus_candidates,
    rank_support_contact_candidates,
)
from benchmark.rendering.segmentation_contour import (
    compose_segmentation_contour_manifest,
)
from benchmark.visual_judge.active_policy import (
    ACTIVE_CORRECTIVE_PROPOSAL_VERSION,
    generate_corrective_camera_proposals,
    pose_fingerprint,
)
from benchmark.visual_judge.contracts import validate_camera_selection_response
from benchmark.visual_judge.evidence_sufficiency import (
    SUFFICIENT,
    assess_preview_selection_sufficiency,
)
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)
from benchmark.visual_judge.visual_config import DEFAULT_P0B_VISUAL_CONFIGS


FOCUS_CAMERA_MODES = {"visibility_ranked", "support_contact_plane", "query_cov"}
HIGHLIGHTED_GLOBAL_POSE_POLICIES = {"global_top", "legacy_metric"}
CAMERA_EVIDENCE_CACHE_CONTRACT_VERSION = "camera_evidence_cache_contract_v4"
_CAMERA_EVIDENCE_IMPLEMENTATION_FILES = (
    "src/benchmark/rendering/blender.py",
    "src/benchmark/rendering/blender_camera_worker.py",
    "src/benchmark/rendering/blender_collision_mask_worker.py",
    "src/benchmark/rendering/blender_collision_overlay_worker.py",
    "src/benchmark/rendering/blender_focus_bundle_worker.py",
    "src/benchmark/rendering/camera_pose.py",
    "src/benchmark/rendering/collision_overlay.py",
    "src/benchmark/rendering/segmentation_contour.py",
    "src/benchmark/visual_judge/active_fallback.py",
    "src/benchmark/visual_judge/active_policy.py",
    "src/benchmark/visual_judge/evidence_sufficiency.py",
    "src/benchmark/visual_judge/contracts.py",
    "src/benchmark/visual_judge/openai_compatible.py",
    "src/benchmark/visual_judge/render_views.py",
    "src/benchmark/visual_judge/roles.py",
)
_BLEND_HASH_CACHE: dict[tuple[str, int, int], str] = {}


class CameraEvidenceProvider:
    """Render read-only, event-targeted local evidence for P0b adjudication."""

    def __init__(
        self,
        *,
        renderer: Any,
        blend_file: str | Path,
        out_dir: str | Path,
        mode: str,
        selector: Any | None = None,
        max_views: int = 2,
        max_steps: int = 1,
        candidate_count: int = 6,
        metric_modes: dict[str, str] | None = None,
        collision_overlay: bool = False,
        collision_contour: bool = False,
        collision_geometry: dict[str, Any] | None = None,
        frozen_view_ids: list[str] | tuple[str, ...] | None = None,
        highlighted_global_pose_policy: str = "global_top",
        candidate_policy: str = DEFAULT_CAMERA_CANDIDATE_POLICY,
        active_repair: bool = False,
    ) -> None:
        self.renderer = renderer
        self.blend_file = Path(blend_file).expanduser().resolve()
        self.out_dir = Path(out_dir).expanduser().resolve()
        if not self.blend_file.is_file():
            raise FileNotFoundError(f"camera evidence source blend does not exist: {self.blend_file}")
        self.source_blend_sha256 = _content_sha256(self.blend_file)
        self.mode = validate_camera_pose_mode(mode)
        if self.mode is None:
            raise ValueError("CameraEvidenceProvider requires an active camera pose mode")
        self.selector = selector
        self.metric_modes = validate_metric_camera_modes(metric_modes)
        self.max_views = int(max_views)
        self.max_steps = int(max_steps)
        self.candidate_count = int(candidate_count)
        self.candidate_policy = normalize_camera_candidate_policy(
            candidate_policy
        )
        self.active_repair = bool(active_repair)
        # Collision-only paired RGB + diagnostic-overlay evidence. Opt-in so OOB
        # and Support camera behavior is completely unchanged.
        self.collision_overlay = bool(collision_overlay)
        # The production Collision presentation pairs one raw local view with a
        # same-pose segmentation contour. It remains separately opt-in so
        # historical passthrough experiments keep their original evidence arms.
        self.collision_contour = bool(collision_contour)
        if self.collision_contour and not self.collision_overlay:
            raise ValueError("collision_contour requires collision_overlay")
        self.collision_geometry = collision_geometry if isinstance(collision_geometry, dict) else None
        self.collision_geometry_contract = _collision_geometry_contract(
            self.collision_geometry
        )
        self.collision_geometry_sha256 = (
            self.collision_geometry_contract["sha256"]
            if self.collision_geometry_contract is not None
            else None
        )
        self.implementation_contract = _camera_evidence_implementation_contract()
        self.highlighted_global_pose_policy = str(highlighted_global_pose_policy).strip().lower()
        if self.highlighted_global_pose_policy not in HIGHLIGHTED_GLOBAL_POSE_POLICIES:
            raise ValueError(
                "highlighted_global_pose_policy must be 'global_top' or 'legacy_metric'"
            )
        self.frozen_view_ids = (
            list(dict.fromkeys(str(value) for value in frozen_view_ids if str(value)))
            if frozen_view_ids is not None
            else None
        )
        if not 1 <= self.max_views <= 4:
            raise ValueError("camera max_views must be between 1 and 4")
        if not 0 <= self.max_steps <= 3:
            raise ValueError("camera max_steps must be between 0 and 3")
        if not self.max_views <= self.candidate_count <= 8:
            raise ValueError("camera candidate_count must be between max_views and 8")
        query_cov_enabled = any(
            resolve_camera_pose_mode(self.mode, metric, metric_modes=self.metric_modes) == "query_cov"
            for metric in DEFAULT_CAMERA_MODE_BY_METRIC
        )
        if self.frozen_view_ids is not None and not query_cov_enabled:
            raise ValueError("frozen camera view IDs are only valid for query_cov")
        if (
            query_cov_enabled
            and self.frozen_view_ids is None
            and not callable(getattr(selector, "select_camera_views", None))
        ):
            raise TypeError("query_cov camera mode requires a selector exposing select_camera_views(request)")
        self.last_call_usage: dict[str, Any] | None = None

    @property
    def policy_config(self) -> dict[str, Any]:
        resolved_modes = {
            metric: resolve_camera_pose_mode(self.mode, metric, metric_modes=self.metric_modes)
            for metric in DEFAULT_CAMERA_MODE_BY_METRIC
        }
        query_cov_enabled = "query_cov" in resolved_modes.values()
        focus_overlay_enabled = self.collision_overlay or any(
            mode in FOCUS_CAMERA_MODES for mode in resolved_modes.values()
        )
        mutations = ["ephemeral_camera", "ephemeral_track_target"]
        if focus_overlay_enabled:
            mutations.extend(["ephemeral_overlay_color", "ephemeral_wireframe", "ephemeral_marker"])
        return {
            "mode": self.mode,
            "metric_mode_overrides": dict(self.metric_modes),
            "default_metric_modes": dict(DEFAULT_CAMERA_MODE_BY_METRIC),
            "resolved_metric_modes": resolved_modes,
            "max_views": self.max_views,
            "max_steps": self.max_steps if query_cov_enabled else 0,
            "candidate_count": self.candidate_count,
            "candidate_policy": self.candidate_policy,
            "source_blend": str(self.blend_file),
            "source_blend_sha256": self.source_blend_sha256,
            "renderer": _renderer_cache_config(self.renderer),
            "implementation_contract": deepcopy(self.implementation_contract),
            "collision_geometry_sha256": self.collision_geometry_sha256,
            "collision_geometry_contract": deepcopy(self.collision_geometry_contract),
            "frozen_view_ids": deepcopy(self.frozen_view_ids),
            "selector_identity": _selector_cache_identity(self.selector),
            "scene_access": "read_only",
            "preview_render": {
                "engine": getattr(self.renderer, "preview_render_engine", None),
                "width": getattr(self.renderer, "preview_width", None),
                "height": getattr(self.renderer, "preview_height", None),
                "cycles_samples": getattr(self.renderer, "preview_cycles_samples", None),
            },
            "final_focus_bundle": callable(
                getattr(self.renderer, "render_focus_evidence_bundle", None)
            ),
            "allowed_scene_mutations": mutations,
            "allowed_camera_actions": (
                list(CAMERA_ACTIONS)
                if query_cov_enabled and self.frozen_view_ids is None and self.max_steps > 0
                else []
            ),
            "active_repair": self.active_repair,
            "active_corrective_proposal_version": (
                ACTIVE_CORRECTIVE_PROPOSAL_VERSION
                if self.active_repair
                else None
            ),
            "selection_source": (
                "frozen_vlm_selected_view_ids"
                if self.frozen_view_ids is not None
                else "runtime_selector"
                if query_cov_enabled
                else "deterministic"
            ),
            "collision_overlay": self.collision_overlay,
            "collision_contour": self.collision_contour,
            "focus_highlighting": "visibility_ranked_support_contact_plane_and_query_cov",
            "global_context_source": "metric_highlighted_global_when_required",
            "highlighted_global_pose_policy": self.highlighted_global_pose_policy,
        }

    def __call__(self, request: dict[str, Any]) -> list[Any]:
        if not isinstance(request, dict):
            raise TypeError("camera evidence request must be a JSON object")
        metric = str(request.get("metric") or "event").strip().lower()
        resolved_mode = resolve_camera_pose_mode(self.mode, metric, metric_modes=self.metric_modes)
        keyed_request = {
            **request,
            "_resolved_camera_pose_mode": resolved_mode,
            "_camera_candidate_policy": self.candidate_policy,
            "_camera_source_blend_sha256": self.source_blend_sha256,
            "_camera_render": {
                "width": getattr(self.renderer, "preview_width", None)
                or getattr(self.renderer, "width", None),
                "height": getattr(self.renderer, "preview_height", None)
                or getattr(self.renderer, "height", None),
            },
            # Cache identity is deliberately content-addressed.  The same event
            # must be rerendered when renderer/policy settings, source code,
            # collision geometry, or a frozen/runtime selector identity changes.
            "_camera_evidence_cache_contract": {
                "schema_version": CAMERA_EVIDENCE_CACHE_CONTRACT_VERSION,
                "resolved_mode": resolved_mode,
                "policy": self.policy_config,
            },
        }
        event_dir = self.out_dir / _event_key(keyed_request)
        collision_overlay = self.collision_overlay and metric == "collision"
        rich_focus_evidence = collision_overlay or resolved_mode in FOCUS_CAMERA_MODES
        manifest_path = event_dir / "camera_evidence_manifest.json"
        self._begin_call_usage(metric=metric, manifest_path=manifest_path)
        if rich_focus_evidence:
            cached_items = _cached_items(
                manifest_path,
                expected_highlighted_global_pose_policy=self.highlighted_global_pose_policy,
            )
            if cached_items is not None:
                return self._finish_call_usage(cached_items, cache_hit=True)
        else:
            cached = _cached_paths(manifest_path)
            if cached is not None:
                return self._finish_call_usage(cached, cache_hit=True)
        event_dir.mkdir(parents=True, exist_ok=True)
        _write_json(event_dir / "evidence_request.json", request)
        if resolved_mode == "global_only":
            _write_json(
                manifest_path,
                {
                    "policy": self.policy_config,
                    "resolved_mode": resolved_mode,
                    "event_key": event_dir.name,
                    "selection": {
                        "mode": resolved_mode,
                        "selector": "frozen_global_views_from_overview_evidence",
                        "selected_view_ids": [],
                    },
                    "render_evidence": [],
                    "render_evidence_items": [],
                },
            )
            return self._finish_call_usage([], cache_hit=False)
        candidates = generate_camera_pose_candidates(
            keyed_request,
            max_candidates=self.candidate_count,
            policy=self.candidate_policy,
        )
        _write_json(event_dir / "pose_candidates.json", candidates)

        if collision_overlay:
            return self._finish_call_usage(
                self._collision_overlay_evidence(
                    request,
                    candidates,
                    event_dir,
                    resolved_mode=resolved_mode,
                ),
                cache_hit=False,
            )

        if resolved_mode in FOCUS_CAMERA_MODES:
            return self._finish_call_usage(
                self._focus_overlay_evidence(
                    request,
                    candidates,
                    event_dir,
                    resolved_mode=resolved_mode,
                ),
                cache_hit=False,
            )

        if resolved_mode == "bbox_track":
            selected = select_bbox_track_views(candidates, max_views=self.max_views)
            selection_log = {
                "mode": resolved_mode,
                "selector": "frozen_metric_order",
                "selected_view_ids": [item["id"] for item in selected],
                "steps": [],
            }
        else:
            selected, selection_log = self._query_cov_selection(request, candidates, event_dir)

        final_manifest = self.renderer.render_camera_views(
            blend_file=self.blend_file,
            out_dir=event_dir / "final",
            camera_views=selected,
            preview=False,
        )
        paths = [
            Path(str(item["path"]))
            for item in final_manifest.get("views", [])
            if isinstance(item, dict) and item.get("path")
        ]
        if not paths:
            raise RuntimeError("camera evidence renderer returned no final views")
        manifest = {
            "policy": self.policy_config,
            "resolved_mode": resolved_mode,
            "event_key": event_dir.name,
            "selection": selection_log,
            "selected_poses": selected,
            "render_manifest": str(event_dir / "final" / "camera_render_manifest.json"),
            "render_evidence": [str(path) for path in paths],
            "render_evidence_artifacts": _freeze_evidence_paths(paths),
        }
        _write_json(event_dir / "camera_evidence_manifest.json", manifest)
        return self._finish_call_usage(paths, cache_hit=False)

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
            "observability": "actual_per_call_v1",
        }

    def _mark_selector_call(self) -> None:
        if self.last_call_usage is not None:
            self.last_call_usage["selector_calls"] += 1

    def _mark_camera_action(self) -> None:
        if self.last_call_usage is not None:
            self.last_call_usage["camera_actions"] += 1

    def _record_selection_usage(self, selection_log: dict[str, Any]) -> None:
        if self.last_call_usage is None:
            return
        selector_calls, camera_actions = _actual_selection_usage(selection_log)
        self.last_call_usage["selector_calls"] = selector_calls
        self.last_call_usage["camera_actions"] = camera_actions

    def _finish_call_usage(
        self,
        evidence: list[Any],
        *,
        cache_hit: bool,
    ) -> list[Any]:
        if self.last_call_usage is not None:
            self.last_call_usage["cache_hit"] = bool(cache_hit)
            self.last_call_usage["evidence_refs"] = _usage_evidence_refs(evidence)
            if cache_hit:
                # The manifest describes the generation call. A cached read does
                # not re-execute its historical selector calls or camera actions.
                self.last_call_usage["selector_calls"] = 0
                self.last_call_usage["camera_actions"] = 0
            self._persist_call_usage_manifest()
        return evidence

    def _persist_call_usage_manifest(self) -> None:
        if self.last_call_usage is None:
            return
        raw_path = self.last_call_usage.get("manifest_path")
        if not isinstance(raw_path, str) or not raw_path:
            return
        manifest_path = Path(raw_path)
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(manifest, dict):
                return
            manifest["call_usage"] = deepcopy(self.last_call_usage)
            _write_json(manifest_path, manifest)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # Usage remains available in memory even if an existing cache is
            # read-only or its audit field cannot be refreshed.
            return

    def _query_cov_selection(
        self,
        request: dict[str, Any],
        candidates: list[dict[str, Any]],
        event_dir: Path,
        *,
        overlay_spec: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        current = [deepcopy(item) for item in candidates]
        if (
            self.active_repair
            and request.get("_camera_selection_phase") == "active_fallback"
        ):
            selected, selection_log = self._active_repair_selection(
                request,
                current,
                event_dir,
                overlay_spec=overlay_spec,
            )
            self._record_selection_usage(selection_log)
            return selected, selection_log
        if self.frozen_view_ids is not None:
            by_id = {str(item["id"]): item for item in current}
            unknown = [item_id for item_id in self.frozen_view_ids if item_id not in by_id]
            if unknown:
                raise ValueError(f"frozen camera selection references unknown candidates: {unknown}")
            if not self.frozen_view_ids or len(self.frozen_view_ids) > self.max_views:
                raise ValueError(
                    f"frozen camera selection must contain 1..{self.max_views} unique view IDs"
                )
            selected = [deepcopy(by_id[item_id]) for item_id in self.frozen_view_ids]
            return selected, {
                "mode": "query_cov",
                "selector": "frozen_vlm_selection",
                "selected_view_ids": list(self.frozen_view_ids),
                "steps": [],
                "camera_adjustment_allowed": False,
            }
        steps: list[dict[str, Any]] = []
        final_ids: list[str] = []
        latest_visibility_by_id: dict[str, dict[str, Any]] = {}
        for step in range(self.max_steps + 1):
            preview_dir = event_dir / "previews" / f"step_{step:02d}"
            preview_role = "rgb"
            preview_degradation = None
            if overlay_spec is not None:
                try:
                    preview_manifest = self._render_overlay_views(
                        request=request,
                        out_dir=preview_dir,
                        camera_views=current,
                        overlay_spec=overlay_spec,
                        preview=True,
                    )
                    preview_role = "highlighted_focus"
                except Exception as exc:
                    preview_degradation = f"focus_preview_failed: {type(exc).__name__}: {exc}"
                    preview_manifest = self.renderer.render_camera_views(
                        blend_file=self.blend_file,
                        out_dir=preview_dir / "rgb_fallback",
                        camera_views=current,
                        preview=True,
                    )
                    preview_role = "rgb_fallback"
            else:
                preview_manifest = self.renderer.render_camera_views(
                    blend_file=self.blend_file,
                    out_dir=preview_dir,
                    camera_views=current,
                    preview=True,
                )
            preview_by_id = {
                str(item.get("id")): str(item.get("path"))
                for item in preview_manifest.get("views", [])
                if isinstance(item, dict) and item.get("id") and item.get("path")
            }
            preview_visibility_warning = None
            latest_visibility_by_id = {}
            if overlay_spec is not None and preview_role == "highlighted_focus":
                targets = [
                    target
                    for target in overlay_spec.get("targets", [])
                    if isinstance(target, dict)
                ]
                visibility_options = _focus_visibility_options(overlay_spec)
                latest_visibility_by_id = {
                    candidate_id: measure_focus_visibility(
                        path,
                        targets=targets,
                        **visibility_options,
                    )
                    for candidate_id, path in preview_by_id.items()
                }
                preview_visibility_warning = _selector_preview_visibility_warning(
                    latest_visibility_by_id,
                    overlay_spec,
                )
            selector_request = {
                "mode": "query_cov",
                "selection_phase": request.get("_camera_selection_phase"),
                "evidence_deficiency": request.get("_camera_evidence_deficiency"),
                "metric": request.get("metric"),
                "event": request.get("event"),
                "object_ids": request.get("object_ids"),
                "detector_evidence": request.get("detector_evidence"),
                "natural_language_prompt": request.get("natural_language_prompt"),
                "extracted_relationships": request.get("extracted_relationships"),
                "candidates": [
                    {
                        "id": pose["id"],
                        "pose": pose,
                        "image_path": preview_by_id.get(str(pose["id"])),
                    }
                    for pose in current
                ],
                "max_views": self.max_views,
                "step": step,
                "max_steps": self.max_steps,
                "allow_adjustment": step < self.max_steps,
                "allowed_actions": list(CAMERA_ACTIONS) if step < self.max_steps else [],
                "selection_role": "choose_evidence_views_only_do_not_judge_metric",
                "preview_role": preview_role,
                "preview_degradation": preview_degradation,
                "preview_visibility_warning": preview_visibility_warning,
                "color_legend": overlay_spec.get("legend") if overlay_spec is not None else None,
                **vlm_audit_metadata(
                    VLMRole.VLM_CAMERA_SELECTOR,
                    decision_contract=DecisionContract.CAMERA_SELECTION,
                    judge_method="select_camera_views",
                ),
            }
            self._mark_selector_call()
            decision = self.selector.select_camera_views(selector_request)
            selected_ids, action = _validate_selector_decision(
                decision,
                current,
                max_views=self.max_views,
                allow_adjustment=step < self.max_steps,
            )
            step_record = {
                "step": step,
                "preview_manifest": str(preview_dir / "camera_render_manifest.json"),
                "candidate_ids": [item["id"] for item in current],
                "decision": decision,
                "preview_role": preview_role,
                "preview_degradation": preview_degradation,
                "preview_visibility_warning": preview_visibility_warning,
            }
            steps.append(step_record)
            final_ids = selected_ids
            if action is None:
                break
            source = next(item for item in current if item["id"] == action["view_id"])
            adjusted = apply_camera_action(source, action["type"])
            current = [adjusted if item["id"] == source["id"] else item for item in current]
            self._mark_camera_action()

        by_id = {str(item["id"]): item for item in current}
        selected = [deepcopy(by_id[item_id]) for item_id in final_ids if item_id in by_id]
        if not selected:
            selected = select_bbox_track_views(current, max_views=self.max_views)
            final_ids = [item["id"] for item in selected]
        selection_log = {
            "mode": "query_cov",
            "selector": type(self.selector).__name__,
            "selected_view_ids": final_ids,
            "steps": steps,
            "selection_phase": request.get("_camera_selection_phase") or "direct_query_cov",
            "selected_preview_visibility": {
                item_id: latest_visibility_by_id[item_id]
                for item_id in final_ids
                if item_id in latest_visibility_by_id
            },
        }
        self._record_selection_usage(selection_log)
        return selected, selection_log

    def _active_repair_selection(
        self,
        request: dict[str, Any],
        candidates: list[dict[str, Any]],
        event_dir: Path,
        *,
        overlay_spec: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run bounded, metric-specific camera repair with stepwise checks.

        The selector chooses a proposal ID from a deterministic, feasible
        proposal set.  It never authors a free-form camera pose and never
        returns a metric verdict.  After every selector call/action, the same
        deterministic preview sufficiency gate is rerun and the best measured
        packet is retained.
        """

        metric = str(request.get("metric") or "").strip().lower()
        metric_family = (
            "oob" if metric == "object_architecture_penetration" else metric
        )
        config = DEFAULT_P0B_VISUAL_CONFIGS.get(metric_family)
        if not isinstance(config, dict):
            raise ValueError(
                f"active camera repair does not support metric {metric!r}"
            )
        local_budget = min(
            self.max_views,
            max(1, int(config.get("local_view_count") or 1)),
        )
        current = [deepcopy(item) for item in candidates]
        trajectory: list[dict[str, Any]] = []
        action_history: list[dict[str, Any]] = []
        seen_proposals: set[str] = set()
        seen_poses = {
            pose_fingerprint(item)
            for item in current
        }
        best_selected: list[dict[str, Any]] = []
        best_ids: list[str] = []
        best_visibility: dict[str, dict[str, Any]] = {}
        best_assessment: dict[str, Any] | None = None
        previous_assessment: dict[str, Any] | None = None
        stop_reason = "selector_budget_exhausted"
        selector_calls = 0
        actions_executed = 0

        deficiency = request.get("_camera_evidence_deficiency")
        for step in range(self.max_steps + 1):
            preview_dir = event_dir / "active_previews" / f"step_{step:02d}"
            preview_role = "rgb_fallback"
            preview_degradation: str | None = None
            if overlay_spec is not None:
                try:
                    preview_manifest = self._render_overlay_views(
                        request=request,
                        out_dir=preview_dir,
                        camera_views=current,
                        overlay_spec=overlay_spec,
                        preview=True,
                    )
                    preview_role = "highlighted_focus"
                except Exception as exc:
                    preview_degradation = (
                        "focus_preview_failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    preview_manifest = self.renderer.render_camera_views(
                        blend_file=self.blend_file,
                        out_dir=preview_dir / "rgb_fallback",
                        camera_views=current,
                        preview=True,
                    )
            else:
                preview_manifest = self.renderer.render_camera_views(
                    blend_file=self.blend_file,
                    out_dir=preview_dir,
                    camera_views=current,
                    preview=True,
                )
            preview_by_id = {
                str(item.get("id")): str(item.get("path"))
                for item in preview_manifest.get("views", [])
                if isinstance(item, dict)
                and item.get("id")
                and item.get("path")
            }
            visibility_by_id: dict[str, dict[str, Any]] = {}
            preview_warning = None
            if overlay_spec is not None and preview_role == "highlighted_focus":
                targets = [
                    target
                    for target in overlay_spec.get("targets", [])
                    if isinstance(target, dict)
                ]
                visibility_options = _focus_visibility_options(overlay_spec)
                visibility_by_id = {
                    candidate_id: measure_focus_visibility(
                        path,
                        targets=targets,
                        **visibility_options,
                    )
                    for candidate_id, path in preview_by_id.items()
                }
                preview_warning = _selector_preview_visibility_warning(
                    visibility_by_id,
                    overlay_spec,
                )

            active_deficiency = (
                previous_assessment
                if previous_assessment is not None
                else deficiency
            )
            proposals = (
                generate_corrective_camera_proposals(
                    metric=metric_family,
                    candidates=current,
                    deficiency=(
                        active_deficiency
                        if isinstance(active_deficiency, dict)
                        else None
                    ),
                    history=action_history,
                    request=request,
                )
                if step < self.max_steps
                else []
            )
            selector_request = {
                "mode": "query_cov",
                "selection_phase": "active_fallback",
                "evidence_deficiency": active_deficiency,
                "metric": metric_family,
                "event": request.get("event"),
                "object_ids": request.get("object_ids"),
                "detector_evidence": request.get("detector_evidence"),
                "natural_language_prompt": request.get(
                    "natural_language_prompt"
                ),
                "extracted_relationships": request.get(
                    "extracted_relationships"
                ),
                "candidates": [
                    {
                        "id": pose["id"],
                        "pose": pose,
                        "image_path": preview_by_id.get(str(pose["id"])),
                    }
                    for pose in current
                    if preview_by_id.get(str(pose.get("id")))
                ],
                "corrective_proposals": proposals,
                "max_views": local_budget,
                "step": step,
                "max_steps": self.max_steps,
                "allow_adjustment": bool(proposals),
                "allowed_actions": (
                    list(CAMERA_ACTIONS) if proposals else []
                ),
                "selection_role": (
                    "choose_evidence_views_and_optional_bounded_repair_only"
                ),
                "preview_role": preview_role,
                "preview_degradation": preview_degradation,
                "preview_visibility_warning": preview_warning,
                "color_legend": (
                    overlay_spec.get("legend")
                    if overlay_spec is not None
                    else None
                ),
                **vlm_audit_metadata(
                    VLMRole.VLM_CAMERA_SELECTOR,
                    decision_contract=DecisionContract.CAMERA_SELECTION,
                    judge_method="select_camera_views",
                ),
            }
            self._mark_selector_call()
            decision = self.selector.select_camera_views(selector_request)
            selector_calls += 1
            selected_ids, action = _validate_selector_decision(
                decision,
                current,
                max_views=local_budget,
                allow_adjustment=bool(proposals),
            )
            assessment = assess_preview_selection_sufficiency(
                metric_family,
                selected_ids,
                visibility_by_id,
                request=request,
                poses_by_id={
                    str(item.get("id")): item
                    for item in current
                },
            )
            by_id = {str(item.get("id")): item for item in current}
            selected_poses = [
                deepcopy(by_id[item_id])
                for item_id in selected_ids
                if item_id in by_id
            ]
            if _assessment_is_better(assessment, best_assessment):
                best_assessment = deepcopy(assessment)
                best_ids = list(selected_ids)
                best_selected = selected_poses
                best_visibility = {
                    item_id: deepcopy(visibility_by_id[item_id])
                    for item_id in selected_ids
                    if item_id in visibility_by_id
                }

            step_record: dict[str, Any] = {
                "step": step,
                "candidate_ids": [str(item.get("id")) for item in current],
                "preview_paths": {
                    candidate_id: path
                    for candidate_id, path in preview_by_id.items()
                },
                "preview_role": preview_role,
                "preview_degradation": preview_degradation,
                "preview_visibility_warning": preview_warning,
                "corrective_proposals": proposals,
                "decision": decision,
                "selected_view_ids": list(selected_ids),
                "sufficiency": assessment,
                "gain_from_previous": _assessment_gain(
                    previous_assessment,
                    assessment,
                ),
                "remaining_budget": {
                    "selector_calls": self.max_steps + 1 - selector_calls,
                    "camera_actions": self.max_steps - actions_executed,
                },
            }
            trajectory.append(step_record)

            if assessment.get("status") == SUFFICIENT:
                stop_reason = "sufficient_evidence"
                break
            if previous_assessment is not None and not _assessment_is_better(
                assessment,
                previous_assessment,
            ):
                stop_reason = "no_measured_evidence_gain"
                break
            if step >= self.max_steps:
                stop_reason = "camera_action_budget_exhausted"
                break
            if not bool(assessment.get("camera_repairable")):
                stop_reason = "deficiency_not_camera_repairable"
                break
            if action is None:
                stop_reason = "selector_stopped_without_action"
                break
            proposal = _resolve_corrective_proposal(action, proposals)
            if proposal is None:
                raise ValueError(
                    "active camera selector action does not reference an "
                    "offered corrective proposal"
                )
            proposal_fingerprint = str(
                proposal.get("proposal_fingerprint") or ""
            )
            result_pose = deepcopy(proposal.get("result_pose"))
            result_fingerprint = str(
                proposal.get("result_pose_fingerprint") or ""
            )
            if (
                not proposal_fingerprint
                or proposal_fingerprint in seen_proposals
                or not result_fingerprint
                or result_fingerprint in seen_poses
            ):
                stop_reason = "repeated_action_or_pose"
                step_record["action_execution"] = {
                    "executed": False,
                    "reason": stop_reason,
                }
                break
            parent_id = str(proposal.get("parent_view_id") or "")
            if parent_id not in by_id or not isinstance(result_pose, dict):
                raise ValueError(
                    "corrective proposal references an unavailable parent pose"
                )
            current = [
                result_pose if str(item.get("id")) == parent_id else item
                for item in current
            ]
            seen_proposals.add(proposal_fingerprint)
            seen_poses.add(result_fingerprint)
            actions_executed += 1
            self._mark_camera_action()
            action_record = {
                "proposal_id": proposal.get("proposal_id"),
                "proposal_fingerprint": proposal_fingerprint,
                "parent_view_id": parent_id,
                "action_primitive": proposal.get("action_primitive"),
                "family": proposal.get("family"),
                "result_view_id": result_pose.get("id"),
                "result_pose_fingerprint": result_fingerprint,
            }
            action_history.append(action_record)
            step_record["action_execution"] = {
                "executed": True,
                **action_record,
            }
            previous_assessment = assessment

        if not best_selected:
            best_selected = select_bbox_track_views(
                current,
                max_views=local_budget,
            )
            best_ids = [
                str(item.get("id"))
                for item in best_selected
            ]
            best_assessment = assess_preview_selection_sufficiency(
                metric_family,
                best_ids,
                {},
                request=request,
                poses_by_id={
                    str(item.get("id")): item
                    for item in current
                },
            )
        return best_selected, {
            "schema_version": "active_camera_trajectory_v1",
            "mode": "query_cov",
            "selector": type(self.selector).__name__,
            "selection_phase": "active_fallback",
            "selected_view_ids": best_ids,
            "selected_preview_visibility": best_visibility,
            "final_preview_assessment": best_assessment,
            "steps": trajectory,
            "selector_call_count": selector_calls,
            "camera_action_count": actions_executed,
            "stop_reason": stop_reason,
            "budget": {
                "local_view_count": local_budget,
                "max_selector_calls": self.max_steps + 1,
                "max_camera_actions": self.max_steps,
            },
        }

    def _collision_overlay_evidence(
        self,
        request: dict[str, Any],
        candidates: list[dict[str, Any]],
        event_dir: Path,
        *,
        resolved_mode: str,
    ) -> list[dict[str, Any]]:
        spec = self._build_overlay_spec(request)
        color_a = spec["object_a"]["color"]
        color_b = spec["object_b"]["color"]

        visibility_by_id: dict[str, dict[str, Any]] = {}
        overlay_degradation: str | None = None
        ranking_log: dict[str, Any] | None = None
        joint_visibility_status: str | None = None
        if resolved_mode == "bbox_track":
            selected = select_bbox_track_views(candidates, max_views=self.max_views)
            selection_log = {
                "mode": resolved_mode,
                "selector": "frozen_metric_order",
                "selected_view_ids": [str(item.get("id")) for item in selected],
                "steps": [],
            }
        elif resolved_mode == "visibility_ranked":
            (
                selected,
                selection_log,
                visibility_by_id,
                ranking_log,
                overlay_degradation,
            ) = self._rank_by_mask_visibility(request, candidates, event_dir, spec)
            if ranking_log is not None:
                joint_visibility_status = ranking_log.get("joint_visibility_status")
                self._apply_xray_for_containment(spec, ranking_log)
        else:
            selected, selection_log = self._query_cov_selection(
                request,
                candidates,
                event_dir,
                overlay_spec=spec,
            )
            visibility_by_id = dict(
                selection_log.get("selected_preview_visibility") or {}
            )

        _write_json(event_dir / "collision_overlay_spec.json", spec)

        # Per-candidate final RGB with backfill: a failed selected pose is
        # replaced by the next ranked valid candidate rather than aborting.
        rgb_manifest, rendered_selected, backfill_log = self._render_final_rgb_with_backfill(
            event_dir=event_dir,
            selected=selected,
            candidates=candidates,
            ranking_log=ranking_log,
        )
        if backfill_log.get("backfilled"):
            overlay_degradation = "; ".join(
                value for value in [overlay_degradation, "final_rgb_backfilled"] if value
            )
        if not rendered_selected:
            raise RuntimeError(
                "collision evidence produced no usable raw view from any permitted candidate"
            )
        # Raw RGB is authoritative and always kept. Production contour mode
        # replaces the full-recolor Legacy presentation; passthrough experiments
        # can leave contour mode disabled and retain the historical overlay.
        overlay_manifest: dict[str, Any] = {"views": []}
        if not self.collision_contour:
            try:
                overlay_manifest = self.renderer.render_collision_overlay_views(
                    blend_file=self.blend_file,
                    out_dir=event_dir / "final_overlay",
                    camera_views=rendered_selected,
                    overlay_spec=spec,
                    preview=False,
                )
            except Exception as exc:
                overlay_degradation = "; ".join(
                    value
                    for value in [
                        overlay_degradation,
                        f"final_overlay_failed: {type(exc).__name__}: {exc}",
                    ]
                    if value
                )
        pairs, items = _pair_rgb_overlay(
            selected=rendered_selected,
            rgb_manifest=rgb_manifest,
            overlay_manifest=overlay_manifest,
            spec=spec,
            visibility_by_id=visibility_by_id,
            degradation_reason=overlay_degradation,
        )
        contour_manifest: dict[str, Any] | None = None
        if self.collision_contour:
            contour_manifest, contour_items = self._render_collision_contour_evidence(
                event_dir=event_dir,
                selected=rendered_selected,
                overlay_spec=spec,
                raw_items=items,
            )
            items.extend(contour_items)
            contour_by_view = {
                str(item.get("view_id")): item
                for item in contour_items
                if item.get("view_id") is not None
            }
            for pair in pairs:
                view_id = str(pair.get("view_id") or "")
                contour_item = contour_by_view.get(view_id)
                pair["metric_local_contour"] = (
                    contour_item.get("path") if contour_item is not None else None
                )
                pair["contour_available"] = contour_item is not None
        if self.collision_contour:
            # The calibrated Collision v2 bundle is local-only. Do not spend a
            # render or image-budget slot on unused global context.
            global_items, global_degradation = [], None
        else:
            global_items, global_degradation = self._highlighted_global_evidence(
                request=request,
                event_dir=event_dir,
                overlay_spec=spec,
                resolved_mode=resolved_mode,
            )
        # Legacy/passthrough packets retain their historical global-first order.
        items = global_items + items
        if global_degradation:
            overlay_degradation = "; ".join(
                value for value in [overlay_degradation, global_degradation] if value
            )
        if not items:
            raise RuntimeError("collision overlay evidence produced no paired views")
        evidence_styles = sorted({str(item.get("evidence_style")) for item in items if item.get("evidence_style")})
        manifest = {
            "policy": self.policy_config,
            "evidence_packet_version": COLLISION_EVIDENCE_PACKET_VERSION,
            "selector_version": COLLISION_VISIBILITY_SELECTOR_VERSION,
            "resolved_mode": resolved_mode,
            "event_key": event_dir.name,
            "role": "collision_pair_evidence",
            "object_a_id": spec["object_a"]["id"],
            "object_b_id": spec["object_b"]["id"],
            "color_legend": spec["legend"],
            "representation_level": spec["representation_level"],
            "joint_visibility_status": joint_visibility_status,
            "is_visibility_ranked": bool(ranking_log.get("is_visibility_ranked")) if ranking_log else False,
            "evidence_styles": evidence_styles,
            "selection": selection_log,
            "candidate_visibility": visibility_by_id,
            "backfill": backfill_log,
            "overlay_degradation_reason": overlay_degradation,
            "contour_manifest": (
                contour_manifest.get("manifest_path")
                if isinstance(contour_manifest, dict)
                else None
            ),
            "selected_poses": rendered_selected,
            "pairs": pairs,
            "render_evidence": [item["path"] for item in items],
            "render_evidence_artifacts": _freeze_evidence_items(items),
            "render_evidence_items": items,
        }
        _write_json(event_dir / "camera_evidence_manifest.json", manifest)
        return items

    def _render_collision_contour_evidence(
        self,
        *,
        event_dir: Path,
        selected: list[dict[str, Any]],
        overlay_spec: dict[str, Any],
        raw_items: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Render the calibrated same-pose Collision contour presentation."""

        annotation_spec = deepcopy(overlay_spec)
        annotation_spec["object_presentation"] = "annotations_only"
        annotation_spec["role"] = "metric_contour_annotation_base"
        annotation_manifest = self.renderer.render_focus_overlay_views(
            blend_file=self.blend_file,
            out_dir=event_dir / "final_contour_annotation",
            camera_views=selected,
            overlay_spec=annotation_spec,
            preview=False,
            allow_blank_views=True,
        )
        mask_manifest = self.renderer.render_target_id_masks(
            blend_file=self.blend_file,
            out_dir=event_dir / "final_contour_masks",
            camera_views=selected,
            overlay_spec=overlay_spec,
            preview=False,
            respect_occlusion=True,
        )
        composed = compose_segmentation_contour_manifest(
            rgb_manifest=annotation_manifest,
            mask_manifest=mask_manifest,
            overlay_spec=overlay_spec,
            out_dir=event_dir / "final_contour",
        )
        raw_by_view = {
            str(item.get("view_id") or ""): item
            for item in raw_items
            if str(item.get("role") or "") == "collision_rgb"
        }
        contour_items: list[dict[str, Any]] = []
        focus_visibility_options = _focus_visibility_options(overlay_spec)
        focus_roi_defined = focus_visibility_options.get("focus_color") is not None
        for view in composed.get("views", []):
            if not isinstance(view, dict) or not view.get("output_path"):
                continue
            view_id = str(view.get("view_id") or "")
            raw = raw_by_view.get(view_id, {})
            image_size = view.get("image_size")
            image_pixel_count = (
                int(image_size[0]) * int(image_size[1])
                if isinstance(image_size, list)
                and len(image_size) == 2
                else 0
            )
            target_visibility = {
                str(target.get("id")): {
                    "visible_pixels": int(
                        target.get(
                            "visible_pixels_at_composite_resolution"
                        )
                        or 0
                    ),
                    "normalized_visibility": (
                        int(
                            target.get(
                                "visible_pixels_at_composite_resolution"
                            )
                            or 0
                        )
                        / image_pixel_count
                        if image_pixel_count > 0
                        else 0.0
                    ),
                }
                for target in view.get("targets", [])
                if isinstance(target, dict) and target.get("id") is not None
            }
            focus_visibility = (
                measure_focus_visibility(
                    str(view["output_path"]),
                    targets=[],
                    **focus_visibility_options,
                )
                if focus_roi_defined
                else None
            )
            focus_measured = bool(
                isinstance(focus_visibility, dict)
                and focus_visibility.get("measured") is True
            )
            focus_pixel_fraction = (
                float(focus_visibility.get("focus_pixel_fraction") or 0.0)
                if focus_measured
                else None
            )
            focus_in_frame = (
                bool(focus_visibility.get("focus_in_frame"))
                if focus_measured
                and focus_visibility.get("focus_in_frame") is not None
                else None
            )
            focus_measurement_status = (
                "measured"
                if focus_measured
                else "measurement_failed"
                if focus_roi_defined
                else "unavailable_no_collision_focus_roi"
            )
            final_visibility = {
                "schema_version": "final_collision_contour_visibility_v1",
                "status": "ok",
                "image_pixel_count": image_pixel_count,
                "targets": target_visibility,
                "focus_roi_defined": focus_roi_defined,
                "focus_measurement_status": focus_measurement_status,
                "focus_pixel_fraction": focus_pixel_fraction,
                "focus_in_frame": focus_in_frame,
                "measurement_source": (
                    "final_resolution_visible_object_identity_masks"
                    "+final_contour_focus_annotation_pixels"
                    if focus_roi_defined
                    else "final_resolution_visible_object_identity_masks"
                ),
            }
            if (
                isinstance(focus_visibility, dict)
                and focus_visibility.get("error") is not None
            ):
                final_visibility["focus_measurement_error"] = str(
                    focus_visibility["error"]
                )
            if isinstance(raw, dict):
                raw["visibility"] = deepcopy(final_visibility)
            contour_items.append(
                {
                    "path": str(view["output_path"]),
                    "role": "metric_local_contour",
                    "evidence_style": "raw_plus_segmentation_contour",
                    "view_id": view_id,
                    "pair_id": view_id,
                    "pose": deepcopy(raw.get("pose") or view.get("pose")),
                    "target_ids": [
                        str(target.get("id"))
                        for target in overlay_spec.get("targets", [])
                        if isinstance(target, dict) and target.get("id") is not None
                    ],
                    "color_legend": deepcopy(overlay_spec.get("legend")),
                    "representation_level": overlay_spec.get("representation_level"),
                    "visibility": deepcopy(final_visibility),
                    "segmentation_contour": {
                        "target_interior_policy": "preserve_annotation_only_rgb",
                        "mask_occlusion_policy": "respect_scene_occlusion",
                        "band_width_px": composed.get("band_width_px"),
                        "outline_width_px": composed.get("outline_width_px"),
                        "band_alpha": composed.get("band_alpha"),
                        "outline_alpha": composed.get("outline_alpha"),
                    },
                }
            )
        expected_ids = {str(item.get("id")) for item in selected}
        actual_ids = {str(item.get("view_id")) for item in contour_items}
        if expected_ids != actual_ids:
            raise RuntimeError(
                "collision contour evidence is incomplete; "
                f"expected views={sorted(expected_ids)}, actual={sorted(actual_ids)}"
            )
        return composed, contour_items

    def _rank_by_mask_visibility(
        self,
        request: dict[str, Any],
        candidates: list[dict[str, Any]],
        event_dir: Path,
        spec: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any] | None, str | None]:
        """Rank candidates from binary target-identity masks (the v2 selector).

        When the renderer cannot produce identity masks, this degrades to the v1
        overlay-color ranking and records that the result is not a valid mask
        rank. A colored OBB line is never treated as proof a target is visible.
        """

        id_a = str(spec["object_a"]["id"])
        id_b = str(spec["object_b"]["id"])
        target_ids = [id_a, id_b]
        mask_method = getattr(self.renderer, "render_target_id_masks", None)
        if not callable(mask_method):
            selected, selection_log, visibility, degradation = self._rank_by_overlay_visibility(
                candidates, event_dir, spec, spec["object_a"]["color"], spec["object_b"]["color"]
            )
            selection_log["mask_pass"] = "unavailable_used_v1_color_ranking"
            selection_log["is_visibility_ranked"] = False
            return selected, selection_log, visibility, None, (degradation or "target_id_mask_pass_unavailable")
        try:
            mask_manifest = mask_method(
                blend_file=self.blend_file,
                out_dir=event_dir / "target_id_masks",
                camera_views=candidates,
                overlay_spec=spec,
                preview=True,
            )
        except Exception as exc:
            selected = select_bbox_track_views(candidates, max_views=self.max_views)
            reason = f"target_id_mask_pass_failed: {type(exc).__name__}: {exc}"
            return (
                selected,
                {
                    "mode": "visibility_ranked",
                    "selector": "frozen_pose_order_fallback",
                    "selected_view_ids": [str(item.get("id")) for item in selected],
                    "fallback_reason": reason,
                    "is_visibility_ranked": False,
                    "mask_pass": "failed",
                },
                {},
                None,
                reason,
            )
        mask_views = {
            str(view.get("id")): view
            for view in mask_manifest.get("views", [])
            if isinstance(view, dict) and view.get("id") is not None
        }
        mask_stats_by_id: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            candidate_id = str(candidate.get("id"))
            record = mask_views.get(candidate_id, {"status": "failed", "targets": {}})
            mask_stats_by_id[candidate_id] = build_candidate_mask_stats(record, target_ids=target_ids)
        containment_hint = self._containment_hint(request, id_a, id_b)
        selected, ranking_log = rank_collision_candidates_v2(
            candidates,
            mask_stats_by_id,
            target_ids=target_ids,
            max_views=self.max_views,
            containment_hint=containment_hint,
        )
        selection_log = {
            "mode": "visibility_ranked",
            "selector": ranking_log["selector"],
            "selected_view_ids": [str(item.get("id")) for item in selected],
            "is_visibility_ranked": bool(ranking_log.get("is_visibility_ranked")),
            "mask_pass": "ok",
            "ranking": ranking_log,
        }
        degradation = (
            None
            if ranking_log.get("is_visibility_ranked")
            else f"visibility_rank_fallback: {ranking_log.get('fallback_reason')}"
        )
        return selected, selection_log, mask_stats_by_id, ranking_log, degradation

    def _containment_hint(self, request: dict[str, Any], id_a: str, id_b: str) -> str | None:
        detector = request.get("detector_evidence") if isinstance(request.get("detector_evidence"), dict) else {}
        mesh = detector.get("mesh") if isinstance(detector.get("mesh"), dict) else {}
        if mesh.get("containment_a_in_b") is True:
            return id_a
        if mesh.get("containment_b_in_a") is True:
            return id_b
        return None

    def _apply_xray_for_containment(self, spec: dict[str, Any], ranking_log: dict[str, Any]) -> None:
        if ranking_log.get("joint_visibility_status") != JOINT_VISIBILITY_IMPOSSIBLE_OR_OCCLUDED:
            return
        # X-ray the visible outer target(s) so the contained/occluded partner and
        # both OBBs stay inspectable. Diagnostic only, never a rendered look.
        outer_ids = {str(value) for value in (ranking_log.get("required_target_ids") or [])}
        for target in spec.get("targets", []):
            if isinstance(target, dict) and str(target.get("id")) in outer_ids:
                target["xray"] = True
        for key in ("object_a", "object_b"):
            member = spec.get(key)
            if isinstance(member, dict):
                member["xray"] = str(member.get("id")) in outer_ids
        spec["joint_visibility_status"] = JOINT_VISIBILITY_IMPOSSIBLE_OR_OCCLUDED

    def _render_final_rgb_with_backfill(
        self,
        *,
        event_dir: Path,
        selected: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        ranking_log: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Render selected raw views, backfilling failures from ranked candidates.

        A cheap batch render is attempted first. If it does not yield every
        selected pose, each remaining candidate (selected first, then ranked
        order) is rendered individually and failures are skipped and recorded so
        one bad candidate never drops the whole event to pose-order fallback.
        """

        try:
            manifest = self.renderer.render_camera_views(
                blend_file=self.blend_file,
                out_dir=event_dir / "final_rgb",
                camera_views=selected,
                preview=False,
            )
            rendered = _views_by_id(manifest)
            if selected and all(str(pose.get("id")) in rendered for pose in selected):
                return manifest, list(selected), {"backfilled": False, "rendered_view_ids": list(rendered)}
        except Exception:
            pass

        by_id = {
            str(candidate.get("id")): candidate
            for candidate in candidates
            if candidate.get("id") is not None
        }
        # Active camera repair may select a bounded, modified pose that is not
        # present in the original frozen candidate bank.  Keep the exact
        # selected pose authoritative during per-view backfill; otherwise a
        # failed batch render can silently drop it (or replace a same-ID pose
        # with the original geometry).
        for pose in selected:
            if pose.get("id") is not None:
                by_id[str(pose.get("id"))] = pose
        ranked_ids = [str(entry.get("id")) for entry in (ranking_log.get("ranked") if ranking_log else [])]
        ordered_ids: list[str] = []
        for candidate_id in [str(pose.get("id")) for pose in selected] + ranked_ids + list(by_id):
            if candidate_id in by_id and candidate_id not in ordered_ids:
                ordered_ids.append(candidate_id)

        views: list[dict[str, Any]] = []
        rendered_selected: list[dict[str, Any]] = []
        backfill: list[dict[str, Any]] = []
        for candidate_id in ordered_ids:
            if len(rendered_selected) >= self.max_views:
                break
            pose = by_id[candidate_id]
            try:
                manifest = self.renderer.render_camera_views(
                    blend_file=self.blend_file,
                    out_dir=event_dir / "final_rgb" / _slug(candidate_id),
                    camera_views=[pose],
                    preview=False,
                )
            except Exception as exc:
                backfill.append({"id": candidate_id, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            path = _views_by_id(manifest).get(candidate_id)
            if path is None:
                backfill.append({"id": candidate_id, "reason": "no_rendered_view"})
                continue
            views.append({"id": candidate_id, "path": path, "pose": pose})
            rendered_selected.append(pose)
        return (
            {"views": views},
            rendered_selected,
            {
                "backfilled": True,
                "rendered_view_ids": [str(pose.get("id")) for pose in rendered_selected],
                "skipped_candidates": backfill,
            },
        )

    def _focus_overlay_evidence(
        self,
        request: dict[str, Any],
        candidates: list[dict[str, Any]],
        event_dir: Path,
        *,
        resolved_mode: str,
    ) -> list[dict[str, Any]]:
        spec = self._build_focus_spec(request)
        _write_json(event_dir / "focus_overlay_spec.json", spec)
        visibility_by_id: dict[str, dict[str, Any]] = {}
        degradation: str | None = None
        if resolved_mode in {"visibility_ranked", "support_contact_plane"}:
            selected, selection_log, visibility_by_id, degradation = self._rank_by_focus_visibility(
                request,
                candidates,
                event_dir,
                spec,
                resolved_mode=resolved_mode,
            )
        else:
            selected, selection_log = self._query_cov_selection(
                request,
                candidates,
                event_dir,
                overlay_spec=spec,
            )
            visibility_by_id = dict(
                selection_log.get("selected_preview_visibility") or {}
            )

        (
            rgb_manifest,
            overlay_manifest,
            global_items,
            rendered_selected,
            backfill_log,
            bundle_degradation,
        ) = self._render_final_focus_evidence(
            request=request,
            event_dir=event_dir,
            selected=selected,
            candidates=candidates,
            ranking_log=(selection_log.get("ranking") if isinstance(selection_log, dict) else None),
            overlay_spec=spec,
            resolved_mode=resolved_mode,
        )
        if backfill_log.get("backfilled"):
            bundle_degradation = "; ".join(
                value
                for value in [bundle_degradation, "final_rgb_backfilled"]
                if value
            )
        if bundle_degradation:
            degradation = "; ".join(
                value for value in [degradation, bundle_degradation] if value
            )
        pairs, items = _pair_metric_focus_evidence(
            selected=rendered_selected,
            rgb_manifest=rgb_manifest,
            overlay_manifest=overlay_manifest,
            spec=spec,
            visibility_by_id=visibility_by_id,
            degradation_reason=degradation,
        )
        # Global-local correspondence is a hard evidence invariant. Put the
        # highlighted overview first under any downstream max-image truncation.
        items = global_items + items
        if not items:
            raise RuntimeError("focus camera evidence produced no rendered views")
        manifest = {
            "policy": self.policy_config,
            "resolved_mode": resolved_mode,
            "event_key": event_dir.name,
            "role": "metric_focus_evidence",
            "metric": request.get("metric"),
            "target_ids": [target.get("id") for target in spec.get("targets", [])],
            "color_legend": spec.get("legend"),
            "representation_level": spec.get("representation_level"),
            "selection": selection_log,
            "highlight_degradation_reason": degradation,
            "selected_poses": rendered_selected,
            "backfill": backfill_log,
            "pairs": pairs,
            "render_evidence": [item["path"] for item in items],
            "render_evidence_artifacts": _freeze_evidence_items(items),
            "render_evidence_items": items,
        }
        _write_json(event_dir / "camera_evidence_manifest.json", manifest)
        return items

    def _render_final_focus_evidence(
        self,
        *,
        request: dict[str, Any],
        event_dir: Path,
        selected: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        ranking_log: dict[str, Any] | None,
        overlay_spec: dict[str, Any],
        resolved_mode: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any] | None,
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
        str | None,
    ]:
        """Render final local/global evidence, preferring one bundled process."""

        global_poses = (
            [_highlighted_global_pose(request, policy=self.highlighted_global_pose_policy)]
            if resolved_mode in FOCUS_CAMERA_MODES
            else []
        )
        bundle_method = getattr(self.renderer, "render_focus_evidence_bundle", None)
        if callable(bundle_method):
            try:
                bundle = bundle_method(
                    blend_file=self.blend_file,
                    out_dir=event_dir / "final_bundle",
                    local_camera_views=selected,
                    global_camera_views=global_poses,
                    overlay_spec=overlay_spec,
                )
                rgb_manifest = {"views": list(bundle.get("rgb_views") or [])}
                overlay_manifest = {"views": list(bundle.get("overlay_views") or [])}
                global_items = _global_items_from_bundle(
                    bundle,
                    request=request,
                    overlay_spec=overlay_spec,
                )
                return (
                    rgb_manifest,
                    overlay_manifest,
                    global_items,
                    list(selected),
                    {"backfilled": False, "rendered_view_ids": [str(item.get("id")) for item in selected]},
                    None,
                )
            except Exception as exc:
                bundle_degradation = f"focus_bundle_failed: {type(exc).__name__}: {exc}"
            else:  # pragma: no cover - return above is exhaustive
                bundle_degradation = None
        else:
            bundle_degradation = "focus_bundle_unavailable"

        rgb_manifest, rendered_selected, backfill_log = self._render_final_rgb_with_backfill(
            event_dir=event_dir,
            selected=selected,
            candidates=candidates,
            ranking_log=ranking_log,
        )
        if not rendered_selected:
            raise RuntimeError(
                "focus evidence produced no usable raw view from any permitted candidate"
            )

        selected_ids = [str(item.get("id")) for item in selected]
        rendered_ids = [str(item.get("id")) for item in rendered_selected]
        if callable(bundle_method) and rendered_ids != selected_ids:
            try:
                bundle = bundle_method(
                    blend_file=self.blend_file,
                    out_dir=event_dir / "final_bundle_backfill",
                    local_camera_views=rendered_selected,
                    global_camera_views=global_poses,
                    overlay_spec=overlay_spec,
                )
                return (
                    {"views": list(bundle.get("rgb_views") or [])},
                    {"views": list(bundle.get("overlay_views") or [])},
                    _global_items_from_bundle(
                        bundle,
                        request=request,
                        overlay_spec=overlay_spec,
                    ),
                    rendered_selected,
                    backfill_log,
                    bundle_degradation,
                )
            except Exception as exc:
                bundle_degradation = "; ".join(
                    value
                    for value in [
                        bundle_degradation,
                        f"focus_backfill_bundle_failed: {type(exc).__name__}: {exc}",
                    ]
                    if value
                )

        overlay_manifest: dict[str, Any] | None = None
        try:
            overlay_manifest = self._render_overlay_views(
                request=request,
                out_dir=event_dir / "final_highlight",
                camera_views=rendered_selected,
                overlay_spec=overlay_spec,
                preview=False,
            )
        except Exception as exc:
            bundle_degradation = "; ".join(
                value
                for value in [
                    bundle_degradation,
                    f"final_highlight_failed: {type(exc).__name__}: {exc}",
                ]
                if value
            )
        global_items, global_degradation = self._highlighted_global_evidence(
            request=request,
            event_dir=event_dir,
            overlay_spec=overlay_spec,
            resolved_mode=resolved_mode,
        )
        if global_degradation:
            bundle_degradation = "; ".join(
                value for value in [bundle_degradation, global_degradation] if value
            )
        return (
            rgb_manifest,
            overlay_manifest,
            global_items,
            rendered_selected,
            backfill_log,
            bundle_degradation,
        )

    def _build_focus_spec(self, request: dict[str, Any]) -> dict[str, Any]:
        scene = request.get("scene")
        metric = str(request.get("metric") or "event").strip().lower()
        object_ids = request.get("object_ids") if isinstance(request.get("object_ids"), list) else []
        if metric == "collision":
            return self._build_overlay_spec(request)
        base_dir = None
        if isinstance(self.collision_geometry, dict):
            manifest_path = self.collision_geometry.get("manifest_path")
            if isinstance(manifest_path, str) and manifest_path.strip():
                base_dir = Path(manifest_path).expanduser().resolve().parent
        return build_focus_overlay_spec(
            scene=scene,
            metric=metric,
            object_ids=[str(value) for value in object_ids],
            detector_evidence=(
                request.get("detector_evidence")
                if isinstance(request.get("detector_evidence"), dict)
                else {}
            ),
            architecture_element=(
                str(request.get("architecture_element"))
                if request.get("architecture_element") is not None
                else None
            ),
            geometry_manifest=self.collision_geometry,
            geometry_base_dir=base_dir,
        )

    def _rank_by_focus_visibility(
        self,
        request: dict[str, Any],
        candidates: list[dict[str, Any]],
        event_dir: Path,
        spec: dict[str, Any],
        *,
        resolved_mode: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str | None]:
        try:
            preview_manifest = self._render_overlay_views(
                request=request,
                out_dir=event_dir / "focus_previews",
                camera_views=candidates,
                overlay_spec=spec,
                preview=True,
            )
            preview_by_id = _views_by_id(preview_manifest)
            visibility_by_id: dict[str, dict[str, Any]] = {}
            targets = [target for target in spec.get("targets", []) if isinstance(target, dict)]
            for candidate in candidates:
                candidate_id = str(candidate.get("id"))
                path = preview_by_id.get(candidate_id)
                visibility_by_id[candidate_id] = (
                    measure_focus_visibility(
                        path,
                        targets=targets,
                        **_focus_visibility_options(spec),
                    )
                    if path is not None
                    else {
                        "target_pixel_fractions": {},
                        "focus_pixel_fraction": 0.0,
                        "pixel_count": 0,
                        "measured": False,
                    }
                )
            if resolved_mode == "support_contact_plane":
                selected, ranking_log = rank_support_contact_candidates(
                    candidates,
                    visibility_by_id,
                    targets=targets,
                    max_views=self.max_views,
                )
                selector_name = "support_contact_plane_visibility_rank_v1"
            else:
                selected, ranking_log = rank_focus_candidates(
                    candidates,
                    visibility_by_id,
                    targets=targets,
                    max_views=self.max_views,
                )
                selector_name = "deterministic_visibility_framing_rank_v1"
            selection_log = {
                "mode": resolved_mode,
                "selector": selector_name,
                "selected_view_ids": [str(item.get("id")) for item in selected],
                "ranking": ranking_log,
            }
            return selected, selection_log, visibility_by_id, ranking_log.get("fallback_reason")
        except Exception as exc:
            selected = select_bbox_track_views(candidates, max_views=self.max_views)
            reason = f"focus_visibility_ranking_failed: {type(exc).__name__}: {exc}"
            return (
                selected,
                {
                    "mode": resolved_mode,
                    "selector": "frozen_pose_order_fallback",
                    "selected_view_ids": [str(item.get("id")) for item in selected],
                    "fallback_reason": reason,
                },
                {},
                reason,
            )

    def _render_overlay_views(
        self,
        *,
        request: dict[str, Any],
        out_dir: Path,
        camera_views: list[dict[str, Any]],
        overlay_spec: dict[str, Any],
        preview: bool,
        allow_blank_views: bool = False,
    ) -> dict[str, Any]:
        metric = str(request.get("metric") or "event").strip().lower()
        if metric == "collision":
            method = getattr(self.renderer, "render_collision_overlay_views", None)
        else:
            method = getattr(self.renderer, "render_focus_overlay_views", None)
            if not callable(method):
                method = getattr(self.renderer, "render_collision_overlay_views", None)
        if not callable(method):
            raise TypeError("renderer does not expose a focus-overlay render method")
        kwargs = {
            "blend_file": self.blend_file,
            "out_dir": out_dir,
            "camera_views": camera_views,
            "overlay_spec": overlay_spec,
            "preview": preview,
        }
        if allow_blank_views:
            kwargs["allow_blank_views"] = True
        return method(
            **kwargs,
        )

    def _highlighted_global_evidence(
        self,
        *,
        request: dict[str, Any],
        event_dir: Path,
        overlay_spec: dict[str, Any],
        resolved_mode: str,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if resolved_mode not in FOCUS_CAMERA_MODES:
            return [], None
        try:
            pose = _highlighted_global_pose(
                request,
                policy=self.highlighted_global_pose_policy,
            )
            manifest = self._render_overlay_views(
                request=request,
                out_dir=event_dir / "highlighted_global",
                camera_views=[pose],
                overlay_spec=overlay_spec,
                preview=False,
            )
            path = _views_by_id(manifest).get(str(pose.get("id")))
            if path is None:
                raise RuntimeError("highlighted global renderer returned no matching view")
            return [
                {
                    "path": path,
                    "role": "metric_highlighted_global",
                    "view_id": pose.get("id"),
                    "metric": request.get("metric"),
                    "target_ids": [target.get("id") for target in overlay_spec.get("targets", [])],
                    "color_legend": overlay_spec.get("legend"),
                    "representation_level": overlay_spec.get("representation_level"),
                    "pose": pose,
                }
            ], None
        except Exception as exc:
            return [], f"highlighted_global_failed: {type(exc).__name__}: {exc}"

    def _build_overlay_spec(self, request: dict[str, Any]) -> dict[str, Any]:
        scene = request.get("scene")
        event = request.get("event") if isinstance(request.get("event"), dict) else {}
        object_ids = request.get("object_ids") if isinstance(request.get("object_ids"), list) else []
        object_a = event.get("object_a") or (object_ids[0] if len(object_ids) > 0 else None)
        object_b = event.get("object_b") or (object_ids[1] if len(object_ids) > 1 else None)
        if object_a is None or object_b is None:
            raise ValueError("collision overlay requires two target object ids")
        detector = request.get("detector_evidence") if isinstance(request.get("detector_evidence"), dict) else {}
        mesh_evidence = detector.get("mesh") if isinstance(detector.get("mesh"), dict) else None
        focus = detector.get("focus_region") if isinstance(detector.get("focus_region"), dict) else None
        base_dir = None
        if isinstance(self.collision_geometry, dict):
            manifest_path = self.collision_geometry.get("manifest_path")
            if isinstance(manifest_path, str) and manifest_path.strip():
                base_dir = Path(manifest_path).expanduser().resolve().parent
        return build_collision_overlay_spec(
            scene=scene,
            object_a_id=str(object_a),
            object_b_id=str(object_b),
            mesh_evidence=mesh_evidence,
            focus_region=focus,
            geometry_manifest=self.collision_geometry,
            geometry_base_dir=base_dir,
        )

    def _rank_by_overlay_visibility(
        self,
        candidates: list[dict[str, Any]],
        event_dir: Path,
        spec: dict[str, Any],
        color_a: list[float],
        color_b: list[float],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str | None]:
        try:
            preview_manifest = self.renderer.render_collision_overlay_views(
                blend_file=self.blend_file,
                out_dir=event_dir / "overlay_previews",
                camera_views=candidates,
                overlay_spec=spec,
                preview=True,
            )
            preview_by_id = {
                str(item.get("id")): str(item.get("path"))
                for item in preview_manifest.get("views", [])
                if isinstance(item, dict) and item.get("id") and item.get("path")
            }
            visibility_by_id: dict[str, dict[str, Any]] = {}
            for candidate in candidates:
                candidate_id = str(candidate.get("id"))
                path = preview_by_id.get(candidate_id)
                if path is None:
                    visibility_by_id[candidate_id] = {"object_a_pixel_fraction": 0.0, "object_b_pixel_fraction": 0.0, "measured": False}
                    continue
                visibility_by_id[candidate_id] = measure_overlay_visibility(path, color_a=color_a, color_b=color_b)
            selected, ranking_log = rank_collision_candidates(candidates, visibility_by_id, max_views=self.max_views)
            selection_log = {
                "mode": "visibility_ranked",
                "selector": "deterministic_visibility_rank_v1",
                "selected_view_ids": [str(item.get("id")) for item in selected],
                "ranking": ranking_log,
            }
            return selected, selection_log, visibility_by_id, ranking_log.get("fallback_reason")
        except Exception as exc:
            # Highlighting failed: keep the frozen pose order and record why. The
            # metric must not fail merely because diagnostic highlighting failed.
            selected = select_bbox_track_views(candidates, max_views=self.max_views)
            reason = f"overlay_visibility_ranking_failed: {type(exc).__name__}: {exc}"
            selection_log = {
                "mode": "visibility_ranked",
                "selector": "frozen_pose_order_fallback",
                "selected_view_ids": [str(item.get("id")) for item in selected],
                "fallback_reason": reason,
            }
            return selected, selection_log, {}, reason


def _highlighted_global_pose(
    request: dict[str, Any],
    *,
    policy: str = "global_top",
) -> dict[str, Any]:
    scene = request.get("scene")
    poses = generate_global_context_poses(scene)
    if policy == "global_top":
        return deepcopy(poses[0])
    if policy != "legacy_metric":
        raise ValueError("highlighted global pose policy must be 'global_top' or 'legacy_metric'")
    metric = str(request.get("metric") or "event").strip().lower()
    detector = request.get("detector_evidence") if isinstance(request.get("detector_evidence"), dict) else {}
    plane_flags = detector.get("plane_flags") if isinstance(detector.get("plane_flags"), dict) else {}
    horizontal_oob = any(
        plane_flags.get(flag)
        for flag in ("west_oob", "east_oob", "south_oob", "north_oob")
    )
    if metric == "oob" and horizontal_oob:
        return deepcopy(poses[0])
    return deepcopy(poses[1])


def _validate_selector_decision(
    decision: Any,
    candidates: list[dict[str, Any]],
    *,
    max_views: int,
    allow_adjustment: bool,
) -> tuple[list[str], dict[str, Any] | None]:
    available = {str(item.get("id")) for item in candidates}
    selected = validate_camera_selection_response(
        decision,
        available_view_ids=available,
        max_views=max_views,
    )
    raw_action = decision.get("action")
    if raw_action is None:
        return selected, None
    if not allow_adjustment:
        raise ValueError("camera selector requested an action after the adjustment budget was exhausted")
    if not isinstance(raw_action, dict):
        raise ValueError("camera selector action must be null or a JSON object")
    view_id = str(raw_action.get("view_id") or "")
    action_type = str(raw_action.get("type") or "")
    if view_id not in available or action_type not in CAMERA_ACTIONS:
        raise ValueError("camera selector requested an invalid bounded action")
    resolved: dict[str, Any] = {
        "view_id": view_id,
        "type": action_type,
    }
    if raw_action.get("proposal_id") is not None:
        resolved["proposal_id"] = str(raw_action.get("proposal_id"))
    return selected, resolved


def _actual_selection_usage(
    selection_log: dict[str, Any],
) -> tuple[int, int]:
    """Derive executed selector/action counts from the recorded trajectory."""

    raw_selector_calls = selection_log.get("selector_call_count")
    if (
        isinstance(raw_selector_calls, int)
        and not isinstance(raw_selector_calls, bool)
        and raw_selector_calls >= 0
    ):
        selector_calls = raw_selector_calls
    else:
        steps = selection_log.get("steps")
        selector_calls = len(steps) if isinstance(steps, list) else 0

    raw_camera_actions = selection_log.get("camera_action_count")
    if (
        isinstance(raw_camera_actions, int)
        and not isinstance(raw_camera_actions, bool)
        and raw_camera_actions >= 0
    ):
        camera_actions = raw_camera_actions
    else:
        camera_actions = 0
        steps = selection_log.get("steps")
        for step in steps if isinstance(steps, list) else []:
            if not isinstance(step, dict):
                continue
            execution = step.get("action_execution")
            if isinstance(execution, dict):
                camera_actions += int(execution.get("executed") is True)
                continue
            decision = step.get("decision")
            if (
                isinstance(decision, dict)
                and isinstance(decision.get("action"), dict)
            ):
                camera_actions += 1
    return selector_calls, camera_actions


def _usage_evidence_refs(evidence: list[Any]) -> list[str]:
    """Match the control-loop evidence reference convention without mutation."""

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


def _resolve_corrective_proposal(
    action: dict[str, Any],
    proposals: list[dict[str, Any]],
) -> dict[str, Any] | None:
    proposal_id = str(action.get("proposal_id") or "")
    if proposal_id:
        matches = [
            proposal
            for proposal in proposals
            if str(proposal.get("proposal_id") or "") == proposal_id
        ]
        return matches[0] if len(matches) == 1 else None
    view_id = str(action.get("view_id") or "")
    action_type = str(action.get("type") or "")
    matches = [
        proposal
        for proposal in proposals
        if str(proposal.get("parent_view_id") or "") == view_id
        and str(proposal.get("action_primitive") or "") == action_type
    ]
    return matches[0] if len(matches) == 1 else None


def _assessment_rank(value: dict[str, Any] | None) -> tuple[int, float, int]:
    if not isinstance(value, dict):
        return (-1, -1.0, -1)
    status_rank = {
        "unknown": 0,
        "insufficient": 1,
        "sufficient": 2,
    }.get(str(value.get("status") or ""), -1)
    return (
        status_rank,
        float(value.get("evidence_utility") or 0.0),
        int(value.get("usable_local_view_count") or 0),
    )


def _assessment_is_better(
    candidate: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> bool:
    return _assessment_rank(candidate) > _assessment_rank(reference)


def _assessment_gain(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    if previous is None:
        return {
            "comparable": False,
            "status_improved": False,
            "utility_delta": None,
            "usable_view_delta": None,
        }
    return {
        "comparable": True,
        "status_improved": (
            _assessment_rank(current)[0] > _assessment_rank(previous)[0]
        ),
        "utility_delta": (
            float(current.get("evidence_utility") or 0.0)
            - float(previous.get("evidence_utility") or 0.0)
        ),
        "usable_view_delta": (
            int(current.get("usable_local_view_count") or 0)
            - int(previous.get("usable_local_view_count") or 0)
        ),
    }


def _pair_rgb_overlay(
    *,
    selected: list[dict[str, Any]],
    rgb_manifest: dict[str, Any],
    overlay_manifest: dict[str, Any],
    spec: dict[str, Any],
    visibility_by_id: dict[str, dict[str, Any]],
    degradation_reason: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair the normal RGB and same-pose diagnostic overlay per selected pose.

    Views are matched deterministically by pose id and emitted in a stable order
    (RGB then overlay for each pose), so the camera manifest keeps the pairing
    and selected poses auditable.
    """

    rgb_by_id = _views_by_id(rgb_manifest)
    overlay_by_id = _views_by_id(overlay_manifest)
    id_a = spec["object_a"]["id"]
    id_b = spec["object_b"]["id"]
    pairs: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for pose in selected:
        view_id = str(pose.get("id"))
        rgb_path = rgb_by_id.get(view_id)
        overlay_path = overlay_by_id.get(view_id)
        # Raw RGB is the primary evidence; keep it even if its same-pose overlay
        # failed to render. Never emit an overlay without its raw counterpart.
        if rgb_path is None:
            continue
        visibility = visibility_by_id.get(view_id)
        if overlay_path is not None:
            try:
                visibility = measure_focus_visibility(
                    overlay_path,
                    targets=[
                        target
                        for target in spec.get("targets", [])
                        if isinstance(target, dict)
                    ],
                    **_focus_visibility_options(spec),
                )
                if isinstance(visibility, dict):
                    visibility["measurement_source"] = (
                        "final_resolution_highlight"
                    )
            except (OSError, TypeError, ValueError):
                # Preview statistics remain an auditable degradation, never a
                # silent claim that final-resolution visibility was measured.
                visibility = None
        pairs.append(
            {
                "pair_id": view_id,
                "view_id": view_id,
                "pose": pose,
                "collision_rgb": rgb_path,
                "collision_pair_overlay": overlay_path,
                "overlay_available": overlay_path is not None,
                "visibility": visibility,
            }
        )
        common = {
            "view_id": view_id,
            "pair_id": view_id,
            "object_a_id": id_a,
            "object_b_id": id_b,
            "color_legend": spec["legend"],
            "representation_level": spec["representation_level"],
            "visibility": visibility,
            "diagnostic_degradation_reason": degradation_reason,
        }
        items.append({"path": rgb_path, "role": "collision_rgb", "evidence_style": "raw", **common})
        if overlay_path is not None:
            items.append(
                {"path": overlay_path, "role": "collision_pair_overlay", "evidence_style": "raw_highlight", **common}
            )
    return pairs, items


def _pair_metric_focus_evidence(
    *,
    selected: list[dict[str, Any]],
    rgb_manifest: dict[str, Any],
    overlay_manifest: dict[str, Any] | None,
    spec: dict[str, Any],
    visibility_by_id: dict[str, dict[str, Any]],
    degradation_reason: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rgb_by_id = _views_by_id(rgb_manifest)
    overlay_by_id = _views_by_id(overlay_manifest or {})
    target_ids = [target.get("id") for target in spec.get("targets", []) if isinstance(target, dict)]
    pairs: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for pose in selected:
        view_id = str(pose.get("id"))
        rgb_path = rgb_by_id.get(view_id)
        overlay_path = overlay_by_id.get(view_id)
        if rgb_path is None:
            continue
        visibility = visibility_by_id.get(view_id)
        if overlay_path is not None:
            try:
                visibility = measure_focus_visibility(
                    overlay_path,
                    targets=[
                        target
                        for target in spec.get("targets", [])
                        if isinstance(target, dict)
                    ],
                    **_focus_visibility_options(spec),
                )
                if isinstance(visibility, dict):
                    visibility["measurement_source"] = (
                        "final_resolution_highlight"
                    )
            except (OSError, TypeError, ValueError):
                visibility = None
        pair = {
            "view_id": view_id,
            "pose": pose,
            "metric_local_rgb": rgb_path,
            "metric_local_highlight": overlay_path,
            "visibility": visibility,
        }
        pairs.append(pair)
        common = {
            "view_id": view_id,
            "target_ids": target_ids,
            "color_legend": spec.get("legend"),
            "representation_level": spec.get("representation_level"),
            "visibility": visibility,
            "diagnostic_degradation_reason": degradation_reason,
            "pose": pose,
        }
        items.append({"path": rgb_path, "role": "metric_local_rgb", **common})
        if overlay_path is not None:
            items.append({"path": overlay_path, "role": "metric_local_highlight", **common})
    return pairs, items


def _global_items_from_bundle(
    bundle: dict[str, Any],
    *,
    request: dict[str, Any],
    overlay_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for view in bundle.get("global_overlay_views") or []:
        if not isinstance(view, dict) or not view.get("path"):
            continue
        items.append(
            {
                "path": str(view["path"]),
                "role": "metric_highlighted_global",
                "view_id": view.get("id"),
                "metric": request.get("metric"),
                "target_ids": [
                    target.get("id")
                    for target in overlay_spec.get("targets", [])
                    if isinstance(target, dict)
                ],
                "color_legend": overlay_spec.get("legend"),
                "representation_level": overlay_spec.get("representation_level"),
                "pose": view.get("pose"),
            }
        )
    return items


def _focus_visibility_options(
    overlay_spec: dict[str, Any],
) -> dict[str, Any]:
    colors = (
        overlay_spec.get("colors")
        if isinstance(overlay_spec.get("colors"), dict)
        else {}
    )
    # Only markers/connectors are actually rendered in the annotation pass.
    # A numeric ``focus`` record by itself is useful geometry metadata, but it
    # is not visual evidence and must not be treated as a measurable ROI.
    has_focus_annotation = bool(
        overlay_spec.get("markers") or overlay_spec.get("connectors")
    )
    has_architecture_plane = bool(overlay_spec.get("architecture_planes"))
    return {
        "focus_color": (
            colors.get("marker") if has_focus_annotation else None
        ),
        "region_colors": (
            {"architecture_plane": colors.get("architecture")}
            if has_architecture_plane and colors.get("architecture") is not None
            else None
        ),
    }


def _require_visible_selector_targets(
    preview_by_id: dict[str, str],
    overlay_spec: dict[str, Any],
    *,
    min_visible_fraction: float = 0.001,
) -> None:
    targets = [target for target in overlay_spec.get("targets", []) if isinstance(target, dict)]
    required_ids = [
        str(target.get("id"))
        for target in targets
        if target.get("required_for_visibility") and target.get("id") is not None
    ]
    if not required_ids:
        required_ids = [str(targets[0].get("id"))] if targets else []
    if not required_ids:
        raise RuntimeError("selector overlay contains no required target IDs")
    for path in preview_by_id.values():
        stats = measure_focus_visibility(path, targets=targets)
        fractions = stats.get("target_pixel_fractions") if isinstance(stats, dict) else None
        if isinstance(fractions, dict) and all(
            float(fractions.get(object_id) or 0.0) >= min_visible_fraction
            for object_id in required_ids
        ):
            return
    raise RuntimeError(
        "selector previews do not visibly expose every required highlighted target: "
        + ", ".join(required_ids)
    )


def _selector_preview_visibility_warning(
    visibility_by_id: dict[str, dict[str, Any]],
    overlay_spec: dict[str, Any],
    *,
    min_visible_fraction: float = 0.001,
) -> str | None:
    """Return an advisory warning without excluding a selector event.

    Active view search is specifically needed when the frozen previews are
    imperfect.  Incomplete highlight coverage therefore informs the selector
    but must not become a fatal pre-selection gate.
    """

    targets = [
        target for target in overlay_spec.get("targets", []) if isinstance(target, dict)
    ]
    required_ids = [
        str(target.get("id"))
        for target in targets
        if target.get("required_for_visibility") and target.get("id") is not None
    ]
    if not required_ids:
        required_ids = [str(targets[0].get("id"))] if targets else []
    if not required_ids:
        return "selector overlay contains no required target IDs"
    for stats in visibility_by_id.values():
        fractions = (
            stats.get("target_pixel_fractions")
            if isinstance(stats, dict)
            else None
        )
        if isinstance(fractions, dict) and all(
            float(fractions.get(object_id) or 0.0) >= min_visible_fraction
            for object_id in required_ids
        ):
            return None
    return (
        "selector previews do not visibly expose every required highlighted target: "
        + ", ".join(required_ids)
    )


def _views_by_id(manifest: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in manifest.get("views", []) if isinstance(manifest, dict) else []:
        if isinstance(item, dict) and item.get("id") and item.get("path"):
            result[str(item["id"])] = str(item["path"])
    return result


def _event_key(request: dict[str, Any]) -> str:
    payload = json.dumps(request, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    metric = _slug(str(request.get("metric") or "event"))
    raw_ids = request.get("object_ids")
    object_ids = raw_ids if isinstance(raw_ids, list) else ([raw_ids] if raw_ids is not None else [])
    ids = "_".join(_slug(str(value)) for value in object_ids) or "scene"
    return f"{metric}__{ids[:80]}__{digest}"


def _content_sha256(path: Path) -> str:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _BLEND_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    for stale_key in [item for item in _BLEND_HASH_CACHE if item[0] == str(resolved)]:
        _BLEND_HASH_CACHE.pop(stale_key, None)
    _BLEND_HASH_CACHE[key] = value
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _camera_evidence_implementation_contract() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    hashes = {
        relative: _content_sha256(repo_root / relative)
        for relative in _CAMERA_EVIDENCE_IMPLEMENTATION_FILES
    }
    return {
        "schema_version": CAMERA_EVIDENCE_CACHE_CONTRACT_VERSION,
        "files": hashes,
        "sha256": _canonical_json_sha256(hashes),
    }


def _collision_geometry_contract(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None

    def sanitized(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                str(key): sanitized(child)
                for key, child in item.items()
                if str(key) not in {"geometry_path", "manifest_path", "source_uri"}
            }
        if isinstance(item, list):
            return [sanitized(child) for child in item]
        return item

    geometry_files: dict[str, dict[str, Any]] = {}
    objects = value.get("objects")
    if isinstance(objects, dict):
        for object_id, record in sorted(objects.items(), key=lambda item: str(item[0])):
            path_value = record.get("geometry_path") if isinstance(record, dict) else None
            path = Path(str(path_value)).expanduser().resolve() if path_value else None
            geometry_files[str(object_id)] = {
                "present": bool(path and path.is_file()),
                "sha256": _content_sha256(path) if path and path.is_file() else None,
            }
    manifest_payload = sanitized(value)
    payload = {
        "manifest_content_sha256": _canonical_json_sha256(manifest_payload),
        "geometry_files": geometry_files,
    }
    return {
        **payload,
        "sha256": _canonical_json_sha256(payload),
    }


def _renderer_cache_config(renderer: Any) -> dict[str, Any]:
    blender_bin = getattr(renderer, "blender_bin", None)
    return {
        "class": f"{type(renderer).__module__}.{type(renderer).__qualname__}",
        "blender_bin": str(Path(blender_bin).expanduser().resolve()) if blender_bin else None,
        "timeout_seconds": getattr(renderer, "timeout_seconds", None),
        "width": getattr(renderer, "width", None),
        "height": getattr(renderer, "height", None),
        "render_engine": getattr(renderer, "render_engine", None),
        "cycles_device": getattr(renderer, "cycles_device", None),
        "cycles_samples": getattr(renderer, "cycles_samples", None),
        "cycles_denoising": getattr(renderer, "cycles_denoising", None),
        "preview_render_engine": getattr(renderer, "preview_render_engine", None),
        "preview_width": getattr(renderer, "preview_width", None),
        "preview_height": getattr(renderer, "preview_height", None),
        "preview_cycles_samples": getattr(renderer, "preview_cycles_samples", None),
    }


def _selector_cache_identity(selector: Any | None) -> dict[str, Any] | None:
    if selector is None:
        return None
    model = getattr(selector, "model", None)
    return {
        "class": f"{type(selector).__module__}.{type(selector).__qualname__}",
        "model": getattr(model, "model_id", None),
        "endpoint": getattr(model, "endpoint", None),
        "temperature": getattr(model, "temperature", None),
        "max_tokens": getattr(model, "max_tokens", None),
        "max_images": getattr(selector, "max_images", None),
        "max_context_chars": getattr(selector, "max_context_chars", None),
        "response_format_json": getattr(selector, "response_format_json", None),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _freeze_evidence_paths(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "slot": index,
            "path": str(path),
            "sha256": _content_sha256(path),
        }
        for index, path in enumerate(paths)
    ]


def _freeze_evidence_items(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "slot": index,
            "role": item.get("role"),
            "view_id": _evidence_view_id(item),
            "path": str(item["path"]),
            "sha256": _content_sha256(Path(str(item["path"]))),
        }
        for index, item in enumerate(items)
        if item.get("path")
    ]


def _evidence_view_id(item: dict[str, Any]) -> str:
    value = item.get("view_id")
    if value is None and isinstance(item.get("pose"), dict):
        value = item["pose"].get("id")
    return str(value or "")


def _cached_artifacts_valid(
    manifest: dict[str, Any],
    paths: list[Path],
) -> bool:
    artifacts = manifest.get("render_evidence_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(paths):
        return False
    for index, (record, path) in enumerate(zip(artifacts, paths)):
        if (
            not isinstance(record, dict)
            or int(record.get("slot", -1)) != index
            or Path(str(record.get("path") or "")) != path
            or not path.is_file()
            or str(record.get("sha256") or "") != _content_sha256(path)
        ):
            return False
    return True


def _cached_paths(path: Path) -> list[Path] | None:
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    values = manifest.get("render_evidence") if isinstance(manifest, dict) else None
    if not isinstance(values, list) or not values:
        return None
    paths = [Path(str(value)) for value in values]
    return paths if _cached_artifacts_valid(manifest, paths) else None


def _cached_items(
    path: Path,
    *,
    expected_highlighted_global_pose_policy: str | None = None,
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if expected_highlighted_global_pose_policy is not None:
        policy = manifest.get("policy") if isinstance(manifest, dict) else None
        if (
            not isinstance(policy, dict)
            or policy.get("highlighted_global_pose_policy")
            != expected_highlighted_global_pose_policy
        ):
            return None
    items = manifest.get("render_evidence_items") if isinstance(manifest, dict) else None
    if not isinstance(items, list) or not items:
        return None
    paths: list[Path] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("path") or not Path(str(item["path"])).is_file():
            return None
        paths.append(Path(str(item["path"])))
    return items if _cached_artifacts_valid(manifest, paths) else None
