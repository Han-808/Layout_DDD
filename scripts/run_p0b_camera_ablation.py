#!/usr/bin/env python3
"""Replay frozen P0b events under controlled camera-evidence policies.

This deliberately skips generation, retrieval, conversion, and detector
recomputation. Every mode receives the same scene, prompt, relationships, and
detector evidence; only the rendered camera evidence changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from benchmark.rendering import BlenderRenderer
from benchmark.rendering.camera_pose import CAMERA_POSE_MODES, resolve_camera_pose_mode
from benchmark.visual_judge.openai_camera_selector import (
    build_openai_compatible_camera_selector,
)
from benchmark.visual_judge.openai_compatible import build_openai_compatible_vlm_judge
from benchmark.visual_judge.p0b import adjudicate_p0b_event
from benchmark.visual_judge.render_views import CameraEvidenceProvider

SUPPORTED_MODES = ("global_only", "bbox_track", "visibility_ranked", "query_cov")
DEFAULT_MODES = ("global_only", "bbox_track", "visibility_ranked")
SUPPORTED_ARMS = (
    "global_raw",
    "visibility_raw",
    "visibility_highlight",
    "visibility_highlight_global",
    "fixed_global",
    "deterministic_metric_local",
    "vlm_select_from_candidates",
    "active_metric_local",
)
# Canonical evidence styles (see camera-evidence contract E):
#   raw                  -> only the raw RGB view(s);
#   raw_highlight        -> each raw RGB view immediately followed by its
#                           same-pose diagnostic overlay (never raw-replaced);
#   raw_highlight_global -> raw_highlight plus a highlighted global overview.
# Historical arm names remain as CLI aliases, but manifests/summaries record the
# canonical style so old ("highlight replaces raw") and new experiments cannot
# be confused.
CANONICAL_EVIDENCE_STYLES = ("raw", "raw_highlight", "raw_highlight_global")
ARM_CONFIGS = {
    "global_raw": {
        "camera_mode": "global_only",
        "evidence_style": "raw",
        "include_overview": True,
    },
    "visibility_raw": {
        "camera_mode": "visibility_ranked",
        "evidence_style": "raw",
        "include_overview": False,
    },
    "visibility_highlight": {
        "camera_mode": "visibility_ranked",
        # Fixed: highlight now supplements (does not replace) the raw same-pose
        # RGB, so this arm sends the raw + overlay pair.
        "evidence_style": "raw_highlight",
        "include_overview": False,
    },
    "visibility_highlight_global": {
        "camera_mode": "visibility_ranked",
        "evidence_style": "raw_highlight_global",
        "include_overview": False,
    },
    # Visual-evidence-policy experiment. The last three arms deliberately share
    # the same highlighted-global + raw/highlighted-local packet; only the
    # local camera selector/adjustment policy changes.
    "fixed_global": {
        "camera_mode": "global_only",
        "metric_modes": {},
        "evidence_style": "raw",
        "include_overview": True,
        "active_selector": False,
    },
    "deterministic_metric_local": {
        "camera_mode": "auto",
        "metric_modes": {
            "collision": "visibility_ranked",
            "object_architecture_penetration": "visibility_ranked",
            "oob": "visibility_ranked",
            "support": "support_contact_plane",
        },
        "evidence_style": "raw_highlight_global",
        "include_overview": False,
        "active_selector": False,
    },
    "vlm_select_from_candidates": {
        "camera_mode": "auto",
        "metric_modes": {
            "collision": "query_cov",
            "object_architecture_penetration": "query_cov",
            "oob": "query_cov",
            "support": "query_cov",
        },
        "evidence_style": "raw_highlight_global",
        "include_overview": False,
        # The VLM may select candidate IDs, but cannot request a camera move.
        "active_selector": True,
        "max_steps": 0,
    },
    "active_metric_local": {
        "camera_mode": "auto",
        "metric_modes": {
            "collision": "query_cov",
            "object_architecture_penetration": "query_cov",
            "oob": "query_cov",
            "support": "query_cov",
        },
        "evidence_style": "raw_highlight_global",
        "include_overview": False,
        "active_selector": True,
    },
}
# Backward-compatible aliases for any pre-existing style spelling.
EVIDENCE_STYLE_ALIASES = {
    "highlight": "raw_highlight",
    "highlight_global": "raw_highlight_global",
}
RAW_EVIDENCE_ROLES = {"metric_local_rgb", "collision_rgb"}
HIGHLIGHT_EVIDENCE_ROLES = {"metric_local_highlight", "collision_pair_overlay"}
GLOBAL_HIGHLIGHT_ROLE = "metric_highlighted_global"
EVENT_SCHEMA_VERSION = "p0b_camera_ablation_event_v2"
RESUME_CONTRACT_SCHEMA_VERSION = "p0b_camera_ablation_resume_contract_v1"
CAMERA_CANDIDATE_POLICY = "legacy"
HIGHLIGHTED_GLOBAL_POSE_POLICY = "legacy_metric"


def main() -> None:
    args = _parse_args()
    scene = _read_json(args.scene)
    source_report = _read_json(args.source_report)
    gt = _read_json(args.gt)
    judge_config = _read_json(args.judge_config)
    gt_events = _gt_events(gt, metrics=set(args.metric))
    source_events = _source_events(source_report, gt_events)
    collision_geometry = _load_collision_geometry(args.collision_geometry)
    overview = [str(Path(path).expanduser().resolve()) for path in args.overview]
    for path in overview:
        if not Path(path).is_file():
            raise FileNotFoundError(f"overview render does not exist: {path}")
    frozen_input_sha256 = {
        "scene": _file_sha256(args.scene),
        "source_report": _file_sha256(args.source_report),
        "gt": _file_sha256(args.gt),
        "blend_file": _file_sha256(args.blend_file),
        "overview": [_file_sha256(path) for path in args.overview],
        "judge_config": _file_sha256(args.judge_config),
        "collision_geometry": (
            _file_sha256(args.collision_geometry) if args.collision_geometry else None
        ),
        "implementation": _implementation_sha256(),
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = build_openai_compatible_vlm_judge(judge_config)
    camera_selector = build_openai_compatible_camera_selector(
        judge_config
    )
    judge_identity = _judge_identity(judge_config)
    run_started = time.time()
    experiment_specs = _experiment_specs(args)
    shared_visibility_provider = None
    if args.arm and any(spec["camera_mode"] == "visibility_ranked" for spec in experiment_specs):
        shared_visibility_provider = _provider(
            args,
            "visibility_ranked",
            camera_selector,
            scene,
            out_dir / "_shared_visibility_evidence",
            collision_geometry=collision_geometry,
            metric_modes={},
            max_steps=0,
        )

    experiment_manifests = []
    visibility_render_cost: dict[tuple[str, str], float] = {}
    for spec in experiment_specs:
        experiment_name = str(spec["name"])
        camera_mode = str(spec["camera_mode"])
        metric_modes = dict(spec.get("metric_modes") or {})
        active_selector = bool(spec.get("active_selector"))
        max_steps = int(spec.get("max_steps", args.max_steps if active_selector else 0))
        evidence_style = str(spec["evidence_style"])
        canonical_evidence_style = (
            "legacy" if evidence_style == "legacy" else _canonical_evidence_style(evidence_style)
        )
        experiment_started = time.time()
        experiment_dir = out_dir / experiment_name
        events_dir = experiment_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        base_provider = (
            shared_visibility_provider
            if camera_mode == "visibility_ranked" and args.arm
            else _provider(
                args,
                camera_mode,
                camera_selector,
                scene,
                experiment_dir,
                collision_geometry=collision_geometry,
                metric_modes=metric_modes,
                max_steps=max_steps,
            )
        )
        provider = (
            _TimedFilteredProvider(base_provider, evidence_style)
            if base_provider is not None and args.arm
            else _TimedProvider(base_provider) if base_provider is not None else None
        )
        arm_overview = overview if spec["include_overview"] else []
        results = []
        for index, gt_event in enumerate(gt_events, start=1):
            event_id = str(gt_event["event_id"])
            metric = str(gt_event["metric"])
            event_path = events_dir / f"{metric}__{_safe_name(event_id)}.json"
            source = source_events[(metric, event_id)]
            resolved_camera_mode = resolve_camera_pose_mode(
                camera_mode,
                metric,
                metric_modes=metric_modes,
            )
            resume_contract = _event_resume_contract(
                args=args,
                experiment_name=experiment_name,
                camera_mode=camera_mode,
                resolved_camera_mode=resolved_camera_mode,
                metric_modes=metric_modes,
                active_selector=active_selector,
                max_steps=max_steps,
                evidence_style=evidence_style,
                canonical_evidence_style=canonical_evidence_style,
                include_overview=bool(spec["include_overview"]),
                metric=metric,
                event_id=event_id,
                gt_event=gt_event,
                source=source,
                judge_identity=judge_identity,
                frozen_input_sha256=frozen_input_sha256,
                overview_paths=overview,
            )
            if args.resume and _camera_ablation_result_ready(event_path, resume_contract):
                result = _read_json(event_path)
                print(
                    f"[{experiment_name} {index}/{len(gt_events)}] cached {metric}:{event_id}",
                    flush=True,
                )
            else:
                print(
                    f"[{experiment_name} {index}/{len(gt_events)}] judging {metric}:{event_id}",
                    flush=True,
                )
                started = time.time()
                if provider is not None:
                    provider.reset()
                try:
                    judgement = adjudicate_p0b_event(
                        metric=metric,
                        event=source["event"],
                        prompt=source["natural_language_prompt"],
                        relationships=source["extracted_relationships"],
                        scene=scene,
                        detector_evidence=source["detector_evidence"],
                        judge=judge,
                        object_ids=list(gt_event.get("object_ids") or []),
                        overview_render_evidence=arm_overview,
                        local_view_provider=provider,
                        visual_config_policy="passthrough",
                    )
                    elapsed_seconds = time.time() - started
                    camera_seconds = provider.last_seconds if provider is not None else 0.0
                    judge_seconds = max(0.0, elapsed_seconds - camera_seconds)
                    image_count = len(arm_overview) + (
                        provider.last_image_count if provider is not None else 0
                    )
                    key = (metric, event_id)
                    if experiment_name == "visibility_raw":
                        visibility_render_cost[key] = camera_seconds
                    baseline_camera_seconds = visibility_render_cost.get(key, camera_seconds)
                    estimated_uncached_seconds = (
                        judge_seconds + baseline_camera_seconds
                        if camera_mode == "visibility_ranked"
                        else elapsed_seconds
                    )
                    result = {
                        **resume_contract,
                        "resume_contract_sha256": _json_sha256(resume_contract),
                        "predicted_label": judgement["verdict"],
                        "match": judgement["verdict"] == gt_event["label"],
                        "confidence": judgement["confidence"],
                        "image_count": image_count,
                        "evidence_roles": provider.last_roles if provider is not None else ["global_raw"] * len(arm_overview),
                        "camera_evidence_seconds": camera_seconds,
                        "judge_seconds": judge_seconds,
                        "elapsed_seconds": elapsed_seconds,
                        "estimated_uncached_seconds": estimated_uncached_seconds,
                        "frozen_evidence_sha256": _evidence_hashes(
                            provider.last_items if provider is not None else [],
                            arm_overview,
                        ),
                        "judgement": judgement,
                    }
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    elapsed_seconds = time.time() - started
                    camera_seconds = provider.last_seconds if provider is not None else 0.0
                    result = {
                        **resume_contract,
                        "resume_contract_sha256": _json_sha256(resume_contract),
                        "predicted_label": None,
                        "match": False,
                        "confidence": None,
                        "image_count": len(arm_overview) + (
                            provider.last_image_count if provider is not None else 0
                        ),
                        "evidence_roles": provider.last_roles if provider is not None else ["global_raw"] * len(arm_overview),
                        "camera_evidence_seconds": camera_seconds,
                        "judge_seconds": max(0.0, elapsed_seconds - camera_seconds),
                        "elapsed_seconds": elapsed_seconds,
                        "estimated_uncached_seconds": elapsed_seconds,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                event_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)
            if experiment_name == "visibility_raw":
                visibility_render_cost[(metric, event_id)] = float(
                    result.get("camera_evidence_seconds") or 0.0
                )
        experiment_manifest = {
            "arm": experiment_name if args.arm else None,
            "mode": camera_mode,
            "camera_mode": camera_mode,
            "metric_camera_modes": metric_modes,
            "camera_max_steps": max_steps,
            "pose_selector_enabled": active_selector,
            "pose_selector_model": judge_identity if active_selector else None,
            "final_judge_model": judge_identity,
            "evidence_style": evidence_style,
            "canonical_evidence_style": canonical_evidence_style,
            "include_overview": bool(spec["include_overview"]),
            "camera_evidence_dir": (
                str(out_dir / "_shared_visibility_evidence" / "camera_evidence")
                if shared_visibility_provider is not None and camera_mode == "visibility_ranked"
                else str(experiment_dir / "camera_evidence")
            ),
            "event_count": len(results),
            "elapsed_seconds": time.time() - experiment_started,
            "correct": sum(bool(result.get("match")) for result in results),
            "failed": sum(bool(result.get("error")) for result in results),
            "camera_evidence_seconds": sum(float(result.get("camera_evidence_seconds") or 0.0) for result in results),
            "judge_seconds": sum(float(result.get("judge_seconds") or 0.0) for result in results),
            "estimated_uncached_seconds": sum(float(result.get("estimated_uncached_seconds") or 0.0) for result in results),
            "image_count": sum(int(result.get("image_count") or 0) for result in results),
            "results": results,
        }
        (experiment_dir / "mode_results.json").write_text(
            json.dumps(experiment_manifest, indent=2),
            encoding="utf-8",
        )
        experiment_manifests.append(experiment_manifest)

    manifest = {
        "schema_version": "p0b_camera_ablation_run_v1",
        "scene": str(Path(args.scene).expanduser().resolve()),
        "source_report": str(Path(args.source_report).expanduser().resolve()),
        "gt": str(Path(args.gt).expanduser().resolve()),
        "frozen_input_sha256": frozen_input_sha256,
        "frozen_metrics": list(args.metric),
        "final_judge_model": judge_identity,
        "pose_selector_model": judge_identity,
        "controlled_variable": "visual_evidence_policy",
        "modes": list(args.mode),
        "arms": list(args.arm),
        "event_count_per_mode": len(gt_events),
        "elapsed_seconds": time.time() - run_started,
        "mode_summaries": [
            {
                "name": item.get("arm") or item["mode"],
                "mode": item["mode"],
                "camera_mode": item["camera_mode"],
                "metric_camera_modes": item["metric_camera_modes"],
                "camera_max_steps": item["camera_max_steps"],
                "pose_selector_enabled": item["pose_selector_enabled"],
                "evidence_style": item["evidence_style"],
                "canonical_evidence_style": item.get("canonical_evidence_style"),
                "event_count": item["event_count"],
                "correct": item["correct"],
                "failed": item["failed"],
                "elapsed_seconds": item["elapsed_seconds"],
                "camera_evidence_seconds": item["camera_evidence_seconds"],
                "judge_seconds": item["judge_seconds"],
                "estimated_uncached_seconds": item["estimated_uncached_seconds"],
                "image_count": item["image_count"],
            }
            for item in experiment_manifests
        ],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["mode_summaries"], indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--blend-file", required=True)
    parser.add_argument("--overview", action="append", default=[])
    parser.add_argument("--judge-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--metric",
        action="append",
        choices=("collision", "oob", "support"),
        default=[],
        help="Restrict replay to frozen events for selected metrics; defaults to all P0b metrics.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=SUPPORTED_MODES,
        default=[],
        help="Repeat for selected modes; defaults to global_only, bbox_track, visibility_ranked.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        choices=SUPPORTED_ARMS,
        default=[],
        help=(
            "Controlled arm; repeat as needed. Arms isolate camera selection, highlighting, "
            "and highlighted global context. Cannot be combined with --mode."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--blender-timeout-seconds", type=int, default=1800)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--render-engine", choices=["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"], default="CYCLES")
    parser.add_argument("--cycles-device", choices=["CPU", "CUDA", "OPTIX", "AUTO"], default="CUDA")
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--cycles-denoising", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-render-engine", choices=["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"], default="CYCLES")
    parser.add_argument("--preview-width", type=int, default=256)
    parser.add_argument("--preview-height", type=int, default=256)
    parser.add_argument("--preview-cycles-samples", type=int, default=1)
    parser.add_argument("--max-views", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument(
        "--max-steps",
        type=int,
        choices=range(0, 4),
        default=1,
        help="Bounded query_cov adjustment steps; ignored by non-active policies.",
    )
    parser.add_argument("--collision-geometry", default=None)
    parser.add_argument("--collision-overlay", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.mode and args.arm:
        parser.error("--mode and --arm are mutually exclusive")
    if not args.mode and not args.arm:
        args.mode = list(DEFAULT_MODES)
    if not args.metric:
        args.metric = ["collision", "oob", "support"]
    if any(mode not in CAMERA_POSE_MODES for mode in args.mode):
        parser.error("unsupported camera mode")
    return args


def _provider(
    args: argparse.Namespace,
    mode: str,
    camera_selector: Any,
    scene: dict[str, Any],
    mode_dir: Path,
    *,
    collision_geometry: dict[str, Any] | None,
    metric_modes: dict[str, str],
    max_steps: int,
):
    if mode == "global_only":
        return None
    renderer = BlenderRenderer(
        blender_bin=args.blender_bin,
        timeout_seconds=args.blender_timeout_seconds,
        width=args.render_width,
        height=args.render_height,
        render_engine=args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
        preview_render_engine=args.preview_render_engine,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        preview_cycles_samples=args.preview_cycles_samples,
    )
    return CameraEvidenceProvider(
        renderer=renderer,
        blend_file=args.blend_file,
        out_dir=mode_dir / "camera_evidence",
        mode=mode,
        selector=camera_selector,
        metric_modes=metric_modes,
        max_views=args.max_views,
        max_steps=max_steps,
        candidate_count=args.candidate_count,
        collision_overlay=args.collision_overlay,
        collision_geometry=collision_geometry,
        highlighted_global_pose_policy=HIGHLIGHTED_GLOBAL_POSE_POLICY,
        candidate_policy=CAMERA_CANDIDATE_POLICY,
    )


def _experiment_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.arm:
        return [{"name": arm, **ARM_CONFIGS[arm]} for arm in args.arm]
    return [
        {
            "name": mode,
            "camera_mode": mode,
            "metric_modes": {},
            "evidence_style": "legacy",
            "include_overview": True,
            "active_selector": mode == "query_cov",
        }
        for mode in args.mode
    ]


class _TimedProvider:
    def __init__(self, provider: Callable[[dict[str, Any]], list[Any]]) -> None:
        self.provider = provider
        self.reset()

    def reset(self) -> None:
        self.last_seconds = 0.0
        self.last_image_count = 0
        self.last_roles: list[str] = []
        self.last_items: list[Any] = []

    def __call__(self, request: dict[str, Any]) -> list[Any]:
        started = time.time()
        items = list(self.provider(request))
        self.last_seconds = time.time() - started
        self.last_image_count = len(items)
        self.last_items = items
        self.last_roles = [
            str(item.get("role") or "local") if isinstance(item, dict) else "local"
            for item in items
        ]
        return items


class _TimedFilteredProvider(_TimedProvider):
    def __init__(
        self,
        provider: Callable[[dict[str, Any]], list[Any]],
        evidence_style: str,
    ) -> None:
        super().__init__(provider)
        self.evidence_style = evidence_style

    def __call__(self, request: dict[str, Any]) -> list[Any]:
        started = time.time()
        raw_items = list(self.provider(request))
        items = _filter_evidence_items(raw_items, self.evidence_style)
        self.last_seconds = time.time() - started
        self.last_image_count = len(items)
        self.last_items = items
        self.last_roles = [
            str(item.get("role") or "local") if isinstance(item, dict) else "local"
            for item in items
        ]
        if not items:
            raise RuntimeError(
                f"camera provider produced no evidence for style {self.evidence_style!r}"
            )
        return items


def _canonical_evidence_style(evidence_style: str) -> str:
    return EVIDENCE_STYLE_ALIASES.get(evidence_style, evidence_style)


def _filter_evidence_items(items: list[Any], evidence_style: str) -> list[Any]:
    if evidence_style == "legacy":
        return list(items)
    style = _canonical_evidence_style(evidence_style)
    if style not in CANONICAL_EVIDENCE_STYLES:
        raise ValueError(f"unsupported evidence style {evidence_style!r}")
    # ``raw`` keeps only raw RGB. ``raw_highlight`` keeps the raw RGB AND its
    # same-pose overlay so highlighting supplements rather than replaces raw.
    # ``raw_highlight_global`` additionally keeps the highlighted global view.
    roles = set(RAW_EVIDENCE_ROLES)
    if style in {"raw_highlight", "raw_highlight_global"}:
        roles |= HIGHLIGHT_EVIDENCE_ROLES
    if style == "raw_highlight_global":
        roles |= {GLOBAL_HIGHLIGHT_ROLE}
    return [
        item
        for item in items
        if isinstance(item, dict) and str(item.get("role") or "") in roles
    ]


def _load_collision_geometry(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    manifest_path = Path(path).expanduser().resolve()
    value = _read_json(manifest_path)
    value["manifest_path"] = str(manifest_path)
    return value


def _source_events(
    report: dict[str, Any],
    gt_events: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    metrics = (((report.get("reports") or {}).get("generic_validity") or {}).get("metrics") or {})
    candidates: dict[str, list[dict[str, Any]]] = {
        "collision": list((metrics.get("collision") or {}).get("pairs") or []),
        "oob": list((metrics.get("oob") or {}).get("objects") or []),
        "support": list((metrics.get("support") or {}).get("objects") or []),
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for gt_event in gt_events:
        metric = str(gt_event["metric"])
        event_id = str(gt_event["event_id"])
        match = next((item for item in candidates[metric] if _event_id(metric, item) == event_id), None)
        if match is None and metric == "collision":
            target_ids = set(str(value) for value in gt_event.get("object_ids") or [])
            match = next(
                (
                    item
                    for item in candidates[metric]
                    if {str(item.get("object_a")), str(item.get("object_b"))} == target_ids
                ),
                None,
            )
        request = ((match or {}).get("judge_result") or {}).get("request")
        if not isinstance(request, dict):
            raise ValueError(f"source report lacks a frozen judge request for {metric}:{event_id}")
        result[(metric, event_id)] = {
            "event": request.get("event") or {},
            "natural_language_prompt": request.get("natural_language_prompt") or "",
            "extracted_relationships": request.get("extracted_relationships") or [],
            "detector_evidence": request.get("detector_evidence") or {},
        }
    return result


def _event_id(metric: str, item: dict[str, Any]) -> str:
    if metric == "collision":
        return f"{item.get('object_a')}|{item.get('object_b')}"
    return str(item.get("object_id"))


def _gt_events(
    gt: dict[str, Any],
    *,
    metrics: set[str] | None = None,
) -> list[dict[str, Any]]:
    events = gt.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("GT fixture must contain a non-empty events list")
    if metrics:
        events = [event for event in events if str(event.get("metric")) in metrics]
        if not events:
            raise ValueError(f"GT fixture contains no events for metrics {sorted(metrics)}")
    seen = set()
    for event in events:
        if not isinstance(event, dict):
            raise TypeError("each GT event must be a JSON object")
        key = (str(event.get("metric")), str(event.get("event_id")))
        if key in seen:
            raise ValueError(f"duplicate GT event: {key}")
        if event.get("label") not in {"valid", "invalid"}:
            raise ValueError(f"GT event {key} has invalid label")
        seen.add(key)
    return events


def _judge_identity(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name"),
        "provider": config.get("provider"),
        "endpoint": config.get("endpoint"),
        "model": config.get("model"),
        "temperature": config.get("temperature"),
    }


def _event_resume_contract(
    *,
    args: argparse.Namespace,
    experiment_name: str,
    camera_mode: str,
    resolved_camera_mode: str,
    metric_modes: dict[str, str],
    active_selector: bool,
    max_steps: int,
    evidence_style: str,
    canonical_evidence_style: str,
    include_overview: bool,
    metric: str,
    event_id: str,
    gt_event: dict[str, Any],
    source: dict[str, Any],
    judge_identity: dict[str, Any],
    frozen_input_sha256: dict[str, Any],
    overview_paths: list[str],
) -> dict[str, Any]:
    """Return every content, identity, and config field that makes a result reusable.

    The contract deliberately excludes transient controls such as ``--resume`` and
    timeouts/retry behavior, but includes every frozen input and observation/judge
    setting that can change the evidence packet or verdict. Old event files lack
    this contract and are therefore recomputed instead of being silently reused.
    """

    arm = experiment_name if args.arm else None
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "resume_contract_schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "arm": arm,
        "mode": camera_mode,
        "camera_mode": camera_mode,
        "resolved_camera_mode": resolved_camera_mode,
        "metric_camera_modes": dict(metric_modes),
        "camera_max_steps": int(max_steps),
        "pose_selector_enabled": bool(active_selector),
        "pose_selector_model": judge_identity if active_selector else None,
        "final_judge_model": judge_identity,
        "evidence_style": evidence_style,
        "canonical_evidence_style": canonical_evidence_style,
        "include_overview": bool(include_overview),
        "metric": metric,
        "event_id": event_id,
        "object_ids": list(gt_event.get("object_ids") or []),
        "gt_label": gt_event["label"],
        "gt_reason_code": gt_event.get("reason_code"),
        "frozen_event_packet_sha256": _json_sha256(source),
        "frozen_scene_sha256": frozen_input_sha256["scene"],
        "frozen_source_report_sha256": frozen_input_sha256["source_report"],
        "frozen_gt_sha256": frozen_input_sha256["gt"],
        "frozen_input_sha256": frozen_input_sha256,
        "observation_config": {
            "candidate_policy": CAMERA_CANDIDATE_POLICY,
            "highlighted_global_pose_policy": HIGHLIGHTED_GLOBAL_POSE_POLICY,
            "overview_paths": overview_paths if include_overview else [],
            "blender_bin": str(Path(args.blender_bin).expanduser().resolve()),
            "render_width": int(args.render_width),
            "render_height": int(args.render_height),
            "render_engine": str(args.render_engine),
            "cycles_device": str(args.cycles_device),
            "cycles_samples": int(args.cycles_samples),
            "cycles_denoising": bool(args.cycles_denoising),
            "preview_render_engine": str(args.preview_render_engine),
            "preview_width": int(args.preview_width),
            "preview_height": int(args.preview_height),
            "preview_cycles_samples": int(args.preview_cycles_samples),
            "max_views": int(args.max_views),
            "candidate_count": int(args.candidate_count),
            "collision_overlay": bool(args.collision_overlay),
            "collision_geometry_path": (
                str(Path(args.collision_geometry).expanduser().resolve())
                if args.collision_geometry
                else None
            ),
        },
    }


def _camera_ablation_result_ready(
    path: Path,
    expected_contract: dict[str, Any],
) -> bool:
    """Reuse only a successful verdict generated under the exact expected contract."""

    if not path.is_file():
        return False
    try:
        result = _read_json(path)
        if result.get("error"):
            return False
        predicted = result.get("predicted_label")
        if predicted not in {"valid", "invalid"}:
            return False
        if result.get("resume_contract_sha256") != _json_sha256(expected_contract):
            return False
        if any(result.get(key) != value for key, value in expected_contract.items()):
            return False
        judgement = result.get("judgement")
        if not isinstance(judgement, dict) or judgement.get("verdict") != predicted:
            return False
        if not _frozen_evidence_ready(result.get("frozen_evidence_sha256")):
            return False
        return result.get("match") is (predicted == expected_contract.get("gt_label"))
    except Exception:
        return False


def _implementation_sha256() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "scripts/run_p0b_camera_ablation.py",
        "src/benchmark/rendering/blender.py",
        "src/benchmark/rendering/blender_camera_worker.py",
        "src/benchmark/rendering/blender_collision_mask_worker.py",
        "src/benchmark/rendering/blender_collision_overlay_worker.py",
        "src/benchmark/rendering/blender_focus_bundle_worker.py",
        "src/benchmark/rendering/camera_pose.py",
        "src/benchmark/rendering/collision_overlay.py",
        "src/benchmark/visual_judge/contracts.py",
        "src/benchmark/visual_judge/openai_compatible.py",
        "src/benchmark/visual_judge/p0b.py",
        "src/benchmark/visual_judge/render_views.py",
        "src/benchmark/visual_judge/roles.py",
        "src/benchmark/visual_judge/visual_config.py",
    )
    return {
        relative_path: _file_sha256(repo_root / relative_path)
        for relative_path in relative_paths
    }


def _evidence_hashes(items: list[Any], overview: list[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError("camera evidence items must be objects with path")
        path = str(Path(str(item["path"])).expanduser().resolve())
        result.append(
            {
                "path": path,
                "role": str(item.get("role") or "local"),
                "sha256": _file_sha256(path),
            }
        )
    for value in overview:
        path = str(Path(str(value)).expanduser().resolve())
        result.append({"path": path, "role": "global_raw", "sha256": _file_sha256(path)})
    return result


def _frozen_evidence_ready(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        path_value = item.get("path")
        expected_sha256 = item.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_sha256, str):
            return False
        path = Path(path_value)
        if not path.is_file() or _file_sha256(path) != expected_sha256:
            return False
    return True


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "event"


if __name__ == "__main__":
    main()
