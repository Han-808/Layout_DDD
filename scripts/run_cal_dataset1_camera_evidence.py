#!/usr/bin/env python3
"""Render a judge-free camera-evidence audit over routed cal_dataset1 events.

The experiment holds presentation and image budget fixed.  For every routed
Collision, OOB, or Support event it renders two raw/highlight pose pairs from:

* ``fixed_global_highlight``: frozen top and perspective overview poses;
* ``metric_local_highlight``: metric-aware deterministic local Top-2 poses.

No VLM is constructed or called.  The outputs are review packets showing the
exact visual evidence a later judge would receive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.rendering import BlenderRenderer, CYCLES_DEVICES, RENDER_ENGINES
from benchmark.rendering.camera_pose import (
    CAMERA_CANDIDATE_POLICIES,
    generate_global_context_poses,
)
from benchmark.visual_judge.render_views import CameraEvidenceProvider
from benchmark.rendering.collision_overlay import (
    build_collision_overlay_spec,
    build_focus_overlay_spec,
    measure_focus_visibility,
)
from benchmark.visual_judge.p0b import build_p0b_local_evidence_request


DISTORTION_SPLITS = ("obvious_distortion", "subtle_distortion")
SUPPORTED_SPLITS = (*DISTORTION_SPLITS, "fine_edge")
METRICS = ("collision", "oob", "support")
ARMS = ("fixed_global_highlight", "metric_local_highlight")
LOCAL_ROLES = {
    "collision_rgb",
    "collision_pair_overlay",
    "metric_local_rgb",
    "metric_local_highlight",
}
HIGHLIGHT_ROLES = {"collision_pair_overlay", "metric_local_highlight"}
METRIC_MODES = {
    "collision": "visibility_ranked",
    "oob": "visibility_ranked",
    "support": "support_contact_plane",
}
MATERIALIZATION_SCHEMA_VERSION = "cal_dataset1_scene_materialization_v1"
COMPARISON_SCHEMA_VERSION = "cal_dataset1_camera_evidence_comparison_v2"


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    cases = _selected_events(
        dataset_root,
        splits=set(args.split),
        metrics=set(args.metric),
        case_ids=set(args.case_id),
        max_cases=args.max_cases,
    )
    plan = _plan_manifest(dataset_root, cases, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "experiment_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan["counts"], indent=2))
        return

    renderer = BlenderRenderer(
        blender_bin=Path(args.blender_bin),
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
        require_asset_mesh=args.require_asset_mesh,
    )
    asset_root = Path(args.asset_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    started = time.time()
    for case_index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        print(f"[{case_index}/{len(cases)}] {case_id}", flush=True)
        try:
            rows.extend(
                _run_case(
                    dataset_root=dataset_root,
                    out_dir=out_dir,
                    case=case,
                    renderer=renderer,
                    asset_root=asset_root,
                    resume=args.resume,
                    max_views=args.max_views,
                    candidate_count=args.candidate_count,
                    candidate_policy=args.candidate_policy,
                )
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            message = f"{type(exc).__name__}: {exc}"
            print(f"  failed: {message}", flush=True)
            for event in case["events"]:
                rows.append(_error_row(case, event, message))

    _write_review_outputs(out_dir, rows)
    completed = {
        **plan,
        "schema_version": "cal_dataset1_camera_evidence_run_v1",
        "elapsed_seconds": time.time() - started,
        "completed_event_count": sum(not row.get("error") for row in rows),
        "failed_event_count": sum(bool(row.get("error")) for row in rows),
        "review_index": str((out_dir / "index.html").resolve()),
    }
    _write_json(out_dir / "run_manifest.json", completed)
    print(json.dumps({key: completed[key] for key in ("completed_event_count", "failed_event_count", "review_index")}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "Support" / "datasets" / "cal_dataset1"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "cal_dataset1_camera_evidence"
        ),
    )
    parser.add_argument("--experiment-id", default="cal_dataset1_camera_evidence")
    parser.add_argument("--blender-bin", default="/Applications/Blender.app/Contents/MacOS/Blender")
    parser.add_argument(
        "--asset-root",
        default=str(PROJECT_ROOT / "Support" / "Assets" / "imaginarium_assets"),
    )
    parser.add_argument("--split", action="append", choices=SUPPORTED_SPLITS, default=[])
    parser.add_argument("--metric", action="append", choices=METRICS, default=[])
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Restrict rendering to one or more exact case IDs.",
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-views", type=int, default=2, choices=(2,))
    parser.add_argument("--candidate-count", type=int, default=6, choices=range(2, 9))
    parser.add_argument(
        "--candidate-policy",
        choices=CAMERA_CANDIDATE_POLICIES,
        default="legacy",
        help="Camera candidate generator; legacy preserves the original audit.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blender-timeout-seconds", type=int, default=1800)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--render-engine", choices=RENDER_ENGINES, default="BLENDER_WORKBENCH")
    parser.add_argument("--cycles-device", choices=CYCLES_DEVICES, default="CPU")
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--cycles-denoising", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--preview-render-engine",
        choices=RENDER_ENGINES,
        default="BLENDER_WORKBENCH",
    )
    parser.add_argument("--preview-width", type=int, default=256)
    parser.add_argument("--preview-height", type=int, default=256)
    parser.add_argument("--preview-cycles-samples", type=int, default=1)
    parser.add_argument("--require-asset-mesh", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.split:
        args.split = list(DISTORTION_SPLITS)
    if not args.metric:
        args.metric = list(METRICS)
    return args


def _selected_events(
    dataset_root: Path,
    *,
    splits: set[str],
    metrics: set[str],
    case_ids: set[str] | None = None,
    max_cases: int = 0,
) -> list[dict[str, Any]]:
    payload = _read_json(dataset_root / "cases.json")
    case_ids = case_ids or set()
    found_case_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for raw_case in payload.get("cases") or []:
        if not isinstance(raw_case, dict) or str(raw_case.get("split")) not in splits:
            continue
        case_id = str(raw_case.get("case_id") or "")
        if case_ids and case_id not in case_ids:
            continue
        found_case_ids.add(case_id)
        case = deepcopy(raw_case)
        fixture = dataset_root / str(case["fixture_dir"])
        gt = _read_json(fixture / "event_gt.json")
        events = [
            deepcopy(event)
            for event in gt.get("events") or []
            if isinstance(event, dict)
            and str(event.get("metric")) in metrics
            and str(event.get("route_requirement")) == "must_route"
        ]
        if not events:
            continue
        case["events"] = events
        case["fixture"] = str(fixture.resolve())
        selected.append(case)
        if max_cases > 0 and len(selected) >= max_cases:
            break
    missing_case_ids = case_ids - found_case_ids
    if missing_case_ids:
        raise ValueError(
            "requested case IDs are absent from the selected splits: "
            + ", ".join(sorted(missing_case_ids))
        )
    return selected


def _plan_manifest(dataset_root: Path, cases: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "cases": len(cases),
        "events": sum(len(case["events"]) for case in cases),
        "arms": len(ARMS),
        "images_per_event_per_arm": 4,
        "events_by_metric": {
            metric: sum(event["metric"] == metric for case in cases for event in case["events"])
            for metric in METRICS
        },
        "events_by_severity": {
            severity: sum(event.get("severity_class") == severity for case in cases for event in case["events"])
            for severity in ("obvious", "subtle", "edge")
        },
    }
    return {
        "schema_version": "cal_dataset1_camera_evidence_plan_v1",
        "experiment_id": str(args.experiment_id),
        "dataset_root": str(dataset_root),
        "experiment_type": "judge_free_visual_evidence_audit",
        "arms": {
            "fixed_global_highlight": "fixed top + fixed perspective; two raw/highlight pairs",
            "metric_local_highlight": "metric-aware deterministic Top-2; two raw/highlight pairs",
        },
        "controlled_variable": "camera_pose_policy",
        "frozen": {
            "events": "must_route events from the explicitly selected splits",
            "presentation": "raw_plus_same_pose_highlight",
            "pose_count": int(args.max_views),
            "image_count": int(args.max_views) * 2,
            "candidate_count": int(args.candidate_count),
            "candidate_policy": str(args.candidate_policy),
            "render_engine": args.render_engine,
            "render_size": [int(args.render_width), int(args.render_height)],
            "final_vlm_judge": "disabled",
        },
        "metric_local_modes": dict(METRIC_MODES),
        "counts": counts,
        "cases": [
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "events": [
                    {
                        "metric": event["metric"],
                        "event_id": event["event_id"],
                        "severity_class": event.get("severity_class"),
                    }
                    for event in case["events"]
                ],
            }
            for case in cases
        ],
    }


def _run_case(
    *,
    dataset_root: Path,
    out_dir: Path,
    case: dict[str, Any],
    renderer: BlenderRenderer,
    asset_root: Path,
    resume: bool,
    max_views: int,
    candidate_count: int,
    candidate_policy: str,
) -> list[dict[str, Any]]:
    case_id = str(case["case_id"])
    fixture = Path(case["fixture"])
    scene_path = fixture / "generated_scene.json"
    scene = _read_json(scene_path)
    case_dir = out_dir / "cases" / case_id
    scene_dir = case_dir / "scene"
    blend_file = scene_dir / "scene.blend"
    scene_provenance_path = scene_dir / "materialization_provenance.json"
    materialization_base = _materialization_provenance_base(
        scene_path=scene_path,
        asset_root=asset_root,
        renderer=renderer,
    )
    if not (
        resume
        and _materialization_ready(
            blend_file=blend_file,
            provenance_path=scene_provenance_path,
            expected_base=materialization_base,
        )
    ):
        renderer.render_scene(scene_path=scene_path, out_dir=scene_dir, asset_root=asset_root)
        _write_json(
            scene_provenance_path,
            _complete_materialization_provenance(
                materialization_base,
                scene_dir=scene_dir,
                blend_file=blend_file,
            ),
        )
    report_path = dataset_root / "evaluation" / "mesh" / case_id / "generic_validity.json"
    report = _read_json(report_path)
    collision_geometry_path = (
        dataset_root
        / "evaluation"
        / "mesh_geometry"
        / case_id
        / "renders"
        / "collision_geometry_manifest.json"
    )
    collision_geometry = _load_collision_geometry(collision_geometry_path)
    prompt, relationships = _prompt_context(fixture)
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend_file,
        out_dir=case_dir / "metric_local_highlight",
        mode="auto",
        max_views=max_views,
        max_steps=0,
        candidate_count=candidate_count,
        metric_modes=METRIC_MODES,
        collision_overlay=True,
        collision_geometry=collision_geometry,
        highlighted_global_pose_policy="legacy_metric",
        candidate_policy=candidate_policy,
    )
    source_sha256 = _comparison_source_hashes(
        fixture=fixture,
        scene_path=scene_path,
        report_path=report_path,
        collision_geometry_path=collision_geometry_path,
        scene_provenance_path=scene_provenance_path,
        provider=provider,
    )
    rows: list[dict[str, Any]] = []
    for event in case["events"]:
        metric = str(event["metric"])
        event_id = str(event["event_id"])
        event_dir = case_dir / "events" / f"{metric}__{_safe_name(event_id)}"
        comparison_path = event_dir / "comparison_manifest.json"
        prior_human_review = None
        if comparison_path.is_file():
            try:
                prior = _read_json(comparison_path)
                prior_human_review = deepcopy(prior.get("human_review"))
            except Exception:
                prior_human_review = None
        if resume and _comparison_ready(comparison_path, expected_source_sha256=source_sha256):
            comparison = _read_json(comparison_path)
            comparison = _enrich_selection_metadata(comparison)
            _write_json(comparison_path, comparison)
        else:
            record = _report_record(report, metric, event_id, list(event.get("object_ids") or []))
            detector_evidence, event_payload = _event_context(metric, record, event)
            request = build_p0b_local_evidence_request(
                metric=metric,
                event=event_payload,
                prompt=prompt,
                relationships=relationships,
                scene=scene,
                detector_evidence=detector_evidence,
                object_ids=list(event.get("object_ids") or []),
            )
            overlay_spec = _overlay_spec(request, collision_geometry)
            fixed = _render_fixed_global(
                renderer=renderer,
                blend_file=blend_file,
                event_dir=event_dir,
                scene=scene,
                request=request,
                overlay_spec=overlay_spec,
            )
            local_all = list(provider(request))
            local = [item for item in local_all if str(item.get("role")) in LOCAL_ROLES]
            if len(fixed) != 4 or len(local) != 4:
                raise RuntimeError(
                    f"expected four images per arm for {metric}:{event_id}, got fixed={len(fixed)} local={len(local)}"
                )
            fixed_arm = _arm_manifest(fixed, overlay_spec)
            fixed_arm["selection"] = {
                "selector": "frozen_global_context_v1",
                "selected_view_ids": ["global_top", "global_perspective"],
                "fallback_reason": None,
            }
            local_arm = _arm_manifest(local, overlay_spec)
            local_manifest_path = _camera_manifest_for_items(local)
            local_manifest = _read_json(local_manifest_path)
            local_arm["selection"] = deepcopy(local_manifest.get("selection") or {})
            local_arm["resolved_mode"] = local_manifest.get("resolved_mode")
            local_arm["camera_evidence_manifest"] = str(local_manifest_path.resolve())
            local_arm["camera_evidence_manifest_sha256"] = _file_sha256(
                local_manifest_path
            )
            local_arm["highlight_degradation_reason"] = local_manifest.get(
                "highlight_degradation_reason"
            )
            comparison = {
                "schema_version": COMPARISON_SCHEMA_VERSION,
                "case_id": case_id,
                "split": case["split"],
                "metric": metric,
                "event_id": event_id,
                "object_ids": list(event.get("object_ids") or []),
                "semantic_label": event.get("semantic_label"),
                "severity_class": event.get("severity_class"),
                "gt_basis": event.get("gt_basis"),
                "presentation": "raw_plus_same_pose_highlight",
                "image_budget_per_arm": 4,
                "arms": {
                    "fixed_global_highlight": fixed_arm,
                    "metric_local_highlight": local_arm,
                },
                "human_review": prior_human_review if isinstance(prior_human_review, dict) else {
                    "fixed_global_highlight": None,
                    "metric_local_highlight": None,
                    "allowed_values": ["sufficient", "insufficient", "unclear"],
                },
                "source_sha256": deepcopy(source_sha256),
            }
            event_dir.mkdir(parents=True, exist_ok=True)
            _write_json(comparison_path, comparison)
        rows.append(_row_from_comparison(comparison, comparison_path))
    return rows


def _event_context(
    metric: str,
    record: dict[str, Any],
    gt_event: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    object_ids = [str(value) for value in gt_event.get("object_ids") or []]
    if metric == "collision":
        event = {
            "object_a": object_ids[0],
            "object_b": object_ids[1],
            "object_ids": object_ids,
            "evidence_level": record.get("evidence_level"),
            "candidate_selection_policy": "high_recall_candidate_no_label_prior",
        }
        mesh = record.get("mesh_evidence") if isinstance(record.get("mesh_evidence"), dict) else {}
        detector = {
            "candidate_selection_policy": "high_recall_candidate_no_label_prior",
            "obb": record.get("obb_evidence"),
            "mesh": mesh,
            "diagnostics": record.get("diagnostics"),
            "geometry_provenance": record.get("geometry_provenance"),
            "mesh_enclosure": record.get("mesh_enclosure_evidence"),
            "closest_points": mesh.get("closest_points"),
            "focus_region": mesh.get("focus_region"),
            "extracted_relationships_are_claims_only": True,
        }
        return detector, event
    if metric == "oob":
        event = {
            "object_id": object_ids[0],
            "object_ids": object_ids,
            "architecture_element": "room_bounds",
            "plane_flags": deepcopy(record.get("plane_flags") or {}),
        }
        detector = deepcopy(record)
        detector["detector"] = "cal_dataset1_frozen_oob_record"
        detector["extracted_relationships_are_claims_only"] = True
        return detector, event
    if metric == "support":
        candidate_ids = [str(value) for value in record.get("candidate_support_object_ids") or []]
        event = {
            "object_id": object_ids[0],
            "object_ids": list(dict.fromkeys([*object_ids, *candidate_ids])),
            "architecture_element": "floor_walls_ceiling_and_supports",
            "candidate_selection_policy": "high_recall_candidate_no_label_prior",
            "gap_band": record.get("gap_band"),
            "measured_support_modes": record.get("measured_support_modes") or [],
            "architecture_contact_candidates": record.get("architecture_contact_candidates") or [],
        }
        detector = deepcopy(record)
        detector["representative_ray_hits"] = deepcopy(record.get("representative_samples") or [])
        detector["extracted_relationships_are_claims_only"] = True
        return detector, event
    raise ValueError(f"unsupported metric {metric!r}")


def _overlay_spec(request: dict[str, Any], collision_geometry: dict[str, Any] | None) -> dict[str, Any]:
    metric = str(request["metric"])
    detector = request.get("detector_evidence") if isinstance(request.get("detector_evidence"), dict) else {}
    base_dir = None
    if collision_geometry and collision_geometry.get("manifest_path"):
        base_dir = Path(str(collision_geometry["manifest_path"])).parent
    if metric == "collision":
        event = request["event"]
        return build_collision_overlay_spec(
            scene=request["scene"],
            object_a_id=str(event["object_a"]),
            object_b_id=str(event["object_b"]),
            mesh_evidence=detector.get("mesh") if isinstance(detector.get("mesh"), dict) else None,
            focus_region=detector.get("focus_region") if isinstance(detector.get("focus_region"), dict) else None,
            geometry_manifest=collision_geometry,
            geometry_base_dir=base_dir,
        )
    return build_focus_overlay_spec(
        scene=request["scene"],
        metric=metric,
        object_ids=[str(value) for value in request.get("object_ids") or []],
        detector_evidence=detector,
        architecture_element=request.get("architecture_element"),
        geometry_manifest=collision_geometry,
        geometry_base_dir=base_dir,
    )


def _render_fixed_global(
    *,
    renderer: BlenderRenderer,
    blend_file: Path,
    event_dir: Path,
    scene: dict[str, Any],
    request: dict[str, Any],
    overlay_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    poses = generate_global_context_poses(scene)
    bundle = renderer.render_focus_evidence_bundle(
        blend_file=blend_file,
        out_dir=event_dir / "fixed_global_highlight",
        local_camera_views=poses,
        global_camera_views=[],
        overlay_spec=overlay_spec,
    )
    rgb = {str(item.get("id")): item for item in bundle.get("rgb_views") or []}
    highlight = {str(item.get("id")): item for item in bundle.get("overlay_views") or []}
    target_ids = [target.get("id") for target in overlay_spec.get("targets") or []]
    items: list[dict[str, Any]] = []
    for pose in poses:
        view_id = str(pose["id"])
        common = {
            "view_id": view_id,
            "pose": pose,
            "metric": request["metric"],
            "target_ids": target_ids,
            "color_legend": overlay_spec.get("legend"),
            "representation_level": overlay_spec.get("representation_level"),
        }
        items.append({"path": str(rgb[view_id]["path"]), "role": "metric_local_rgb", **common})
        items.append({"path": str(highlight[view_id]["path"]), "role": "metric_local_highlight", **common})
    return items


def _arm_manifest(items: list[dict[str, Any]], overlay_spec: dict[str, Any]) -> dict[str, Any]:
    normalized = []
    highlight_visibility = []
    targets = [target for target in overlay_spec.get("targets") or [] if isinstance(target, dict)]
    for item in items:
        entry = deepcopy(item)
        image_path = Path(str(item["path"])).resolve()
        entry["path"] = str(image_path)
        entry["sha256"] = _file_sha256(image_path)
        if str(item.get("role")) in HIGHLIGHT_ROLES:
            visibility = measure_focus_visibility(entry["path"], targets=targets)
            entry["final_highlight_visibility"] = visibility
            highlight_visibility.append(visibility)
        normalized.append(entry)
    target_ids = [str(target.get("id")) for target in targets if target.get("id") is not None]
    visible_any = {
        target_id: any(
            float((value.get("target_pixel_fractions") or {}).get(target_id) or 0.0) > 0.0
            for value in highlight_visibility
        )
        for target_id in target_ids
    }
    return {
        "pose_count": len(items) // 2,
        "image_count": len(items),
        "items": normalized,
        "diagnostic_target_visible_any_pose": visible_any,
        "diagnostic_all_targets_visible_somewhere": all(visible_any.values()) if visible_any else False,
        "diagnostic_only": True,
    }


def _report_record(
    report: dict[str, Any],
    metric: str,
    event_id: str,
    object_ids: list[str],
) -> dict[str, Any]:
    metric_report = (report.get("metrics") or {}).get(metric) or {}
    rows = metric_report.get("pairs") if metric == "collision" else metric_report.get("objects")
    rows = rows if isinstance(rows, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = (
            f"{row.get('object_a')}|{row.get('object_b')}"
            if metric == "collision"
            else str(row.get("object_id"))
        )
        if row_id == event_id:
            return row
        if metric == "collision" and {str(row.get("object_a")), str(row.get("object_b"))} == set(object_ids):
            return row
    raise ValueError(f"frozen detector report lacks {metric}:{event_id}")


def _prompt_context(fixture: Path) -> tuple[str, list[dict[str, Any]]]:
    request = _read_json(fixture / "scene_request.json")
    plan = _read_json(fixture / "object_plan.json")
    return str(request.get("instruction") or ""), list(plan.get("relations") or [])


def _load_collision_geometry(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _read_json(path)
    value["manifest_path"] = str(path.resolve())
    geometry_dir = path.parent / "collision_geometry"
    objects = value.get("objects") if isinstance(value.get("objects"), dict) else {}
    for object_id, record in objects.items():
        if not isinstance(record, dict):
            continue
        candidate = geometry_dir / f"{object_id}.ply"
        if candidate.is_file():
            record["geometry_path"] = str(candidate.resolve())
    return value


def _row_from_comparison(comparison: dict[str, Any], path: Path) -> dict[str, Any]:
    fixed = comparison["arms"]["fixed_global_highlight"]
    local = comparison["arms"]["metric_local_highlight"]
    return {
        "case_id": comparison["case_id"],
        "split": comparison["split"],
        "severity_class": comparison.get("severity_class"),
        "metric": comparison["metric"],
        "event_id": comparison["event_id"],
        "object_ids": ",".join(comparison.get("object_ids") or []),
        "semantic_label": comparison.get("semantic_label"),
        "fixed_diagnostic_targets_visible": fixed.get("diagnostic_all_targets_visible_somewhere"),
        "local_diagnostic_targets_visible": local.get("diagnostic_all_targets_visible_somewhere"),
        "local_selector": (local.get("selection") or {}).get("selector"),
        "local_selected_view_ids": ",".join(
            str(value) for value in (local.get("selection") or {}).get("selected_view_ids") or []
        ),
        "local_fallback_reason": _selection_fallback(local.get("selection") or {}),
        "fixed_human_review": "",
        "local_human_review": "",
        "notes": "",
        "comparison_manifest": str(path.resolve()),
        "fixed_items": fixed.get("items") or [],
        "local_items": local.get("items") or [],
        "error": None,
    }


def _error_row(case: dict[str, Any], event: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "severity_class": event.get("severity_class"),
        "metric": event["metric"],
        "event_id": event["event_id"],
        "object_ids": ",".join(event.get("object_ids") or []),
        "semantic_label": event.get("semantic_label"),
        "fixed_diagnostic_targets_visible": None,
        "local_diagnostic_targets_visible": None,
        "local_selector": "",
        "local_selected_view_ids": "",
        "local_fallback_reason": "",
        "fixed_human_review": "",
        "local_human_review": "",
        "notes": "",
        "comparison_manifest": "",
        "fixed_items": [],
        "local_items": [],
        "error": error,
    }


def _write_review_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "split",
        "severity_class",
        "metric",
        "event_id",
        "object_ids",
        "semantic_label",
        "fixed_diagnostic_targets_visible",
        "local_diagnostic_targets_visible",
        "local_selector",
        "local_selected_view_ids",
        "local_fallback_reason",
        "fixed_human_review",
        "local_human_review",
        "notes",
        "comparison_manifest",
        "error",
    ]
    with (out_dir / "review.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for metric in METRICS:
        _write_html(out_dir / f"{metric}.html", [row for row in rows if row["metric"] == metric], title=f"{metric.title()} evidence")
    _write_html(out_dir / "index.html", rows, title="cal_dataset1 camera evidence")


def _write_html(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    cards = []
    for row in rows:
        if row.get("error"):
            body = f"<pre>{html.escape(str(row['error']))}</pre>"
        else:
            fixed = _image_cells(row.get("fixed_items") or [], path.parent)
            local = _image_cells(row.get("local_items") or [], path.parent)
            body = (
                '<div class="arm"><h3>fixed_global_highlight</h3>' + fixed + "</div>"
                '<div class="arm"><h3>metric_local_highlight</h3>' + local + "</div>"
            )
        cards.append(
            '<section class="event">'
            f"<h2>{html.escape(str(row['metric']))} · {html.escape(str(row['case_id']))} · {html.escape(str(row['event_id']))}</h2>"
            f"<p>{html.escape(str(row.get('severity_class')))} · targets {html.escape(str(row.get('object_ids')))}</p>"
            f'<div class="arms">{body}</div>'
            "</section>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ background:#111; color:#eee; font:15px system-ui; margin:24px; }}
a {{ color:#8ecbff; }} .event {{ border-top:1px solid #555; padding:18px 0 28px; }}
.arms {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
.images {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
figure {{ margin:0; background:#222; padding:7px; }} img {{ width:100%; height:auto; display:block; }}
figcaption {{ margin-top:5px; color:#bbb; font-size:12px; overflow-wrap:anywhere; }}
@media(max-width:900px) {{ .arms {{ grid-template-columns:1fr; }} }}
</style></head><body><h1>{html.escape(title)}</h1>
<p>Same budget and raw+highlight presentation in both arms. Human review: sufficient / insufficient / unclear.</p>
{''.join(cards)}</body></html>"""
    path.write_text(document, encoding="utf-8")


def _image_cells(items: list[dict[str, Any]], html_root: Path) -> str:
    figures = []
    for item in items:
        image_path = Path(str(item["path"]))
        try:
            source = image_path.relative_to(html_root)
        except ValueError:
            source = image_path
        figures.append(
            "<figure>"
            f'<img loading="lazy" src="{html.escape(source.as_posix())}">'
            f"<figcaption>{html.escape(str(item.get('view_id')))} · {html.escape(str(item.get('role')))}</figcaption>"
            "</figure>"
        )
    return '<div class="images">' + "".join(figures) + "</div>"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "event"


def _camera_manifest_for_items(items: list[dict[str, Any]]) -> Path:
    for item in items:
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        for parent in [path.parent, *path.parents]:
            candidate = parent / "camera_evidence_manifest.json"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("local evidence items do not resolve to a camera evidence manifest")


def _selection_fallback(selection: dict[str, Any]) -> Any:
    ranking = selection.get("ranking") if isinstance(selection.get("ranking"), dict) else {}
    return selection.get("fallback_reason") or ranking.get("fallback_reason")


def _enrich_selection_metadata(comparison: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(comparison)
    arms = result.get("arms") if isinstance(result.get("arms"), dict) else {}
    fixed = arms.get("fixed_global_highlight") if isinstance(arms.get("fixed_global_highlight"), dict) else None
    local = arms.get("metric_local_highlight") if isinstance(arms.get("metric_local_highlight"), dict) else None
    if fixed is not None and not isinstance(fixed.get("selection"), dict):
        fixed["selection"] = {
            "selector": "frozen_global_context_v1",
            "selected_view_ids": ["global_top", "global_perspective"],
            "fallback_reason": None,
        }
    if local is not None and not isinstance(local.get("selection"), dict):
        manifest_path = _camera_manifest_for_items(list(local.get("items") or []))
        manifest = _read_json(manifest_path)
        local["selection"] = deepcopy(manifest.get("selection") or {})
        local["resolved_mode"] = manifest.get("resolved_mode")
        local["camera_evidence_manifest"] = str(manifest_path.resolve())
        local["highlight_degradation_reason"] = manifest.get("highlight_degradation_reason")
    return result


def _materialization_provenance_base(
    *,
    scene_path: Path,
    asset_root: Path,
    renderer: BlenderRenderer,
) -> dict[str, Any]:
    code_paths = [
        PROJECT_ROOT / "src" / "benchmark" / "rendering" / "blender.py",
        PROJECT_ROOT / "src" / "benchmark" / "rendering" / "blender_worker.py",
        PROJECT_ROOT / "src" / "benchmark" / "assets" / "retriever.py",
    ]
    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "scene_sha256": _file_sha256(scene_path),
        "asset_root": str(asset_root.resolve()),
        "renderer": {
            "blender_bin": str(renderer.blender_bin.resolve()),
            "width": renderer.width,
            "height": renderer.height,
            "render_engine": renderer.render_engine,
            "cycles_device": renderer.cycles_device,
            "cycles_samples": renderer.cycles_samples,
            "cycles_denoising": renderer.cycles_denoising,
            "require_asset_mesh": renderer.require_asset_mesh,
        },
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _file_sha256(path)
            for path in code_paths
        },
    }


def _complete_materialization_provenance(
    base: dict[str, Any],
    *,
    scene_dir: Path,
    blend_file: Path,
) -> dict[str, Any]:
    render_manifest_path = scene_dir / "render_manifest.json"
    if not blend_file.is_file() or not render_manifest_path.is_file():
        raise RuntimeError("Blender materialization did not produce its frozen artifacts")
    render_manifest = _read_json(render_manifest_path)
    asset_paths = sorted(
        {
            str(Path(str(item["mesh_path"])).expanduser().resolve())
            for item in render_manifest.get("objects") or []
            if isinstance(item, dict) and item.get("mesh_path")
        }
    )
    asset_files = []
    for raw_path in asset_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"materialized asset mesh is missing: {path}")
        asset_files.append({"path": str(path), "sha256": _file_sha256(path)})
    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "base": deepcopy(base),
        "blend_file_sha256": _file_sha256(blend_file),
        "render_manifest_sha256": _file_sha256(render_manifest_path),
        "blender_version": render_manifest.get("blender_version"),
        "asset_files": asset_files,
    }


def _materialization_ready(
    *,
    blend_file: Path,
    provenance_path: Path,
    expected_base: dict[str, Any],
) -> bool:
    render_manifest_path = blend_file.parent / "render_manifest.json"
    if not blend_file.is_file() or not render_manifest_path.is_file() or not provenance_path.is_file():
        return False
    try:
        value = _read_json(provenance_path)
        if (
            value.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
            or value.get("base") != expected_base
            or value.get("blend_file_sha256") != _file_sha256(blend_file)
            or value.get("render_manifest_sha256") != _file_sha256(render_manifest_path)
        ):
            return False
        for item in value.get("asset_files") or []:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                return False
            path = Path(str(item["path"]))
            if not path.is_file() or _file_sha256(path) != item["sha256"]:
                return False
    except Exception:
        return False
    return True


def _comparison_source_hashes(
    *,
    fixture: Path,
    scene_path: Path,
    report_path: Path,
    collision_geometry_path: Path,
    scene_provenance_path: Path,
    provider: CameraEvidenceProvider,
) -> dict[str, Any]:
    code_paths = [
        PROJECT_ROOT / "src" / "benchmark" / "rendering" / "camera_pose.py",
        PROJECT_ROOT / "src" / "benchmark" / "rendering" / "collision_overlay.py",
        PROJECT_ROOT / "src" / "benchmark" / "visual_judge" / "render_views.py",
        Path(__file__).resolve(),
    ]
    return {
        "scene": _file_sha256(scene_path),
        "event_gt": _file_sha256(fixture / "event_gt.json"),
        "scene_request": _file_sha256(fixture / "scene_request.json"),
        "object_plan": _file_sha256(fixture / "object_plan.json"),
        "detector_report": _file_sha256(report_path),
        "collision_geometry_manifest": (
            _file_sha256(collision_geometry_path)
            if collision_geometry_path.is_file()
            else None
        ),
        "scene_materialization": _file_sha256(scene_provenance_path),
        "camera_policy_config": _json_sha256(provider.policy_config),
        "code": {
            str(path.relative_to(PROJECT_ROOT)): _file_sha256(path)
            for path in code_paths
        },
    }


def _comparison_ready(
    path: Path,
    *,
    expected_source_sha256: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        comparison = _read_json(path)
        if (
            comparison.get("schema_version") != COMPARISON_SCHEMA_VERSION
            or comparison.get("source_sha256") != expected_source_sha256
        ):
            return False
        arms = comparison.get("arms")
        if not isinstance(arms, dict):
            return False
        for arm_name in ARMS:
            arm = arms.get(arm_name)
            if not isinstance(arm, dict):
                return False
            items = arm.get("items")
            if not isinstance(items, list) or len(items) != int(arm.get("image_count") or 0):
                return False
            for item in items:
                if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                    return False
                image_path = Path(str(item["path"]))
                if not image_path.is_file() or _file_sha256(image_path) != item["sha256"]:
                    return False
        local = arms.get("metric_local_highlight") or {}
        manifest_path = local.get("camera_evidence_manifest")
        manifest_sha256 = local.get("camera_evidence_manifest_sha256")
        if manifest_path or manifest_sha256:
            if not manifest_path or not manifest_sha256:
                return False
            resolved_manifest = Path(str(manifest_path))
            if (
                not resolved_manifest.is_file()
                or _file_sha256(resolved_manifest) != manifest_sha256
            ):
                return False
    except Exception:
        return False
    return True


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
