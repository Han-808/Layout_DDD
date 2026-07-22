from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.rendering.camera_pose import (
    CAMERA_ACTIONS,
    CAMERA_CANDIDATE_POLICIES,
    DEFAULT_CAMERA_CANDIDATE_POLICY,
    DEFAULT_CAMERA_MODE_BY_METRIC,
    apply_camera_action,
    generate_camera_pose_candidates,
    generate_global_context_poses,
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


FOCUS_CAMERA_MODES = {"visibility_ranked", "support_contact_plane", "query_cov"}
HIGHLIGHTED_GLOBAL_POSE_POLICIES = {"global_top", "legacy_metric"}
CAMERA_EVIDENCE_CACHE_CONTRACT_VERSION = "camera_evidence_cache_contract_v2"
_CAMERA_EVIDENCE_IMPLEMENTATION_FILES = (
    "src/benchmark/rendering/blender.py",
    "src/benchmark/rendering/blender_camera_worker.py",
    "src/benchmark/rendering/blender_collision_mask_worker.py",
    "src/benchmark/rendering/blender_collision_overlay_worker.py",
    "src/benchmark/rendering/blender_focus_bundle_worker.py",
    "src/benchmark/rendering/camera_pose.py",
    "src/benchmark/rendering/collision_overlay.py",
    "src/benchmark/visual_judge/render_views.py",
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
        collision_geometry: dict[str, Any] | None = None,
        frozen_view_ids: list[str] | tuple[str, ...] | None = None,
        highlighted_global_pose_policy: str = "global_top",
        candidate_policy: str = DEFAULT_CAMERA_CANDIDATE_POLICY,
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
        self.candidate_policy = str(candidate_policy).strip().lower()
        if self.candidate_policy not in CAMERA_CANDIDATE_POLICIES:
            raise ValueError(
                "camera candidate policy must be one of "
                f"{list(CAMERA_CANDIDATE_POLICIES)}"
            )
        # Collision-only paired RGB + diagnostic-overlay evidence. Opt-in so OOB
        # and Support camera behavior is completely unchanged.
        self.collision_overlay = bool(collision_overlay)
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
            "selection_source": (
                "frozen_vlm_selected_view_ids"
                if self.frozen_view_ids is not None
                else "runtime_selector"
                if query_cov_enabled
                else "deterministic"
            ),
            "collision_overlay": self.collision_overlay,
            "focus_highlighting": "visibility_ranked_support_contact_plane_and_query_cov",
            "global_context_source": "metric_highlighted_global",
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
        if rich_focus_evidence:
            cached_items = _cached_items(
                manifest_path,
                expected_highlighted_global_pose_policy=self.highlighted_global_pose_policy,
            )
            if cached_items is not None:
                return cached_items
        else:
            cached = _cached_paths(manifest_path)
            if cached is not None:
                return cached
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
            return []
        candidates = generate_camera_pose_candidates(
            keyed_request,
            max_candidates=self.candidate_count,
            policy=self.candidate_policy,
        )
        _write_json(event_dir / "pose_candidates.json", candidates)

        if collision_overlay:
            return self._collision_overlay_evidence(
                request,
                candidates,
                event_dir,
                resolved_mode=resolved_mode,
            )

        if resolved_mode in FOCUS_CAMERA_MODES:
            return self._focus_overlay_evidence(
                request,
                candidates,
                event_dir,
                resolved_mode=resolved_mode,
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
        }
        _write_json(event_dir / "camera_evidence_manifest.json", manifest)
        return paths

    def _query_cov_selection(
        self,
        request: dict[str, Any],
        candidates: list[dict[str, Any]],
        event_dir: Path,
        *,
        overlay_spec: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        current = [deepcopy(item) for item in candidates]
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
            if overlay_spec is not None and preview_role == "highlighted_focus":
                _require_visible_selector_targets(preview_by_id, overlay_spec)
            selector_request = {
                "mode": "query_cov",
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
                "color_legend": overlay_spec.get("legend") if overlay_spec is not None else None,
            }
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
            }
            steps.append(step_record)
            final_ids = selected_ids
            if action is None:
                break
            source = next(item for item in current if item["id"] == action["view_id"])
            adjusted = apply_camera_action(source, action["type"])
            current = [adjusted if item["id"] == source["id"] else item for item in current]

        by_id = {str(item["id"]): item for item in current}
        selected = [deepcopy(by_id[item_id]) for item_id in final_ids if item_id in by_id]
        if not selected:
            selected = select_bbox_track_views(current, max_views=self.max_views)
            final_ids = [item["id"] for item in selected]
        return selected, {
            "mode": "query_cov",
            "selector": type(self.selector).__name__,
            "selected_view_ids": final_ids,
            "steps": steps,
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
        # Raw RGB is authoritative and always kept; the same-pose overlay is
        # best-effort and never removes its raw counterpart.
        overlay_manifest: dict[str, Any] = {"views": []}
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
                for value in [overlay_degradation, f"final_overlay_failed: {type(exc).__name__}: {exc}"]
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
        global_items, global_degradation = self._highlighted_global_evidence(
            request=request,
            event_dir=event_dir,
            overlay_spec=spec,
            resolved_mode=resolved_mode,
        )
        # Keep the highlighted overview first so a downstream image budget can
        # never consume all slots on local pairs and drop global context.
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
            "selected_poses": rendered_selected,
            "pairs": pairs,
            "render_evidence": [item["path"] for item in items],
            "render_evidence_items": items,
        }
        _write_json(event_dir / "camera_evidence_manifest.json", manifest)
        return items

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

        by_id = {str(candidate.get("id")): candidate for candidate in candidates}
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
                        focus_color=(
                            (spec.get("colors") or {}).get("marker")
                            if resolved_mode == "support_contact_plane"
                            else None
                        ),
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
) -> tuple[list[str], dict[str, str] | None]:
    if not isinstance(decision, dict):
        raise ValueError("camera selector response must be a JSON object")
    available = {str(item.get("id")) for item in candidates}
    raw_ids = decision.get("selected_view_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("camera selector selected_view_ids must be a list")
    selected = list(dict.fromkeys(str(value) for value in raw_ids if str(value)))
    if not selected or len(selected) > max_views or any(value not in available for value in selected):
        raise ValueError("camera selector returned invalid selected_view_ids")
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
    return selected, {"view_id": view_id, "type": action_type}


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
    return paths if all(item.is_file() for item in paths) else None


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
    for item in items:
        if not isinstance(item, dict) or not item.get("path") or not Path(str(item["path"])).is_file():
            return None
    return items
