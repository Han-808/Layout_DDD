#!/usr/bin/env python3
"""Prepare frozen local segmentation-contour evidence for cal_dataset1.

This stage reuses the exact two deterministic local camera poses from exp1_1.
It renders:

1. the existing metric annotations without recoloring scene objects;
2. full-resolution, occlusion-aware Blender object-ID masks;
3. a 2D exterior color band and contour over the annotation-only RGB.

It never changes camera selection, detector evidence, GT, or source scenes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.rendering.blender import BlenderRenderer  # noqa: E402
from benchmark.rendering.segmentation_contour import (  # noqa: E402
    compose_segmentation_contour_manifest,
)
from scripts.judge_cal_dataset1_camera_evidence import (  # noqa: E402
    _discover_manifests,
    _file_sha256,
    _json_sha256,
    _read_json,
    _safe_name,
    _write_json,
)


METRICS = ("collision", "oob", "support")
SEVERITIES = ("obvious", "subtle")
SCHEMA_VERSION = "cal_dataset1_local_contour_evidence_v1"
RUN_SCHEMA_VERSION = "cal_dataset1_local_contour_evidence_run_v1"


def main() -> None:
    args = _parse_args()
    evidence_root = args.evidence_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    manifests = _discover_manifests(
        evidence_root,
        metrics=set(args.metric),
        severities=set(args.severity),
        case_ids=set(args.case_id),
    )
    if args.require_event_count is not None and len(manifests) != args.require_event_count:
        raise RuntimeError(
            f"expected {args.require_event_count} frozen events, found {len(manifests)}"
        )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "event_count": len(manifests),
                    "metrics": list(args.metric),
                    "severities": list(args.severity),
                    "global_evidence": "excluded",
                    "local_view_count_per_event": 2,
                },
                indent=2,
            )
        )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = BlenderRenderer(
        blender_bin=args.blender_bin,
        timeout_seconds=args.timeout_seconds,
        width=512,
        height=512,
        render_engine="BLENDER_WORKBENCH",
        preview_render_engine="BLENDER_WORKBENCH",
        preview_width=256,
        preview_height=256,
    )

    started = time.time()
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, comparison_path in enumerate(manifests, start=1):
        comparison = _read_json(comparison_path)
        case_id = str(comparison["case_id"])
        metric = str(comparison["metric"])
        event_id = str(comparison["event_id"])
        print(
            f"[{index}/{len(manifests)}] {case_id} {metric}:{event_id}",
            flush=True,
        )
        try:
            result = _prepare_one(
                comparison_path=comparison_path,
                comparison=comparison,
                evidence_root=evidence_root,
                out_dir=out_dir,
                renderer=renderer,
                resume=args.resume,
                band_width_px=args.band_width_px,
                outline_width_px=args.outline_width_px,
                band_alpha=args.band_alpha,
                outline_alpha=args.outline_alpha,
            )
        except Exception as exc:
            failure = {
                "case_id": case_id,
                "metric": metric,
                "event_id": event_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[failed] {failure['error']}", flush=True)
            if not args.continue_on_error:
                raise
        else:
            completed.append(result)
            print(f"[{result['status']}] {result['manifest_path']}", flush=True)

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "experiment_id": "exp1_3_local_contour_highlight",
        "source_evidence_root": str(evidence_root),
        "output_root": str(out_dir),
        "frozen_camera_policy": "reuse_exp1_1_metric_local_selected_poses",
        "global_evidence": "excluded",
        "object_presentation": "raw_rgb_plus_annotations_plus_segmentation_contour",
        "mask_occlusion_policy": "respect_scene_occlusion",
        "style": {
            "band_width_px": args.band_width_px,
            "outline_width_px": args.outline_width_px,
            "band_alpha": args.band_alpha,
            "outline_alpha": args.outline_alpha,
        },
        "expected_event_count": len(manifests),
        "completed_event_count": len(completed),
        "failed_event_count": len(failures),
        "complete": len(completed) == len(manifests) and not failures,
        "events": completed,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(out_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "expected_event_count": len(manifests),
                "completed_event_count": len(completed),
                "failed_event_count": len(failures),
                "complete": manifest["complete"],
                "output": str(out_dir),
            },
            indent=2,
        )
    )
    raise SystemExit(1 if failures else 0)


def _prepare_one(
    *,
    comparison_path: Path,
    comparison: dict[str, Any],
    evidence_root: Path,
    out_dir: Path,
    renderer: BlenderRenderer,
    resume: bool,
    band_width_px: int,
    outline_width_px: int,
    band_alpha: float,
    outline_alpha: float,
) -> dict[str, Any]:
    case_id = str(comparison["case_id"])
    metric = str(comparison["metric"])
    event_id = str(comparison["event_id"])
    event_out = (
        out_dir
        / "cases"
        / case_id
        / "events"
        / f"{metric}__{_safe_name(event_id)}"
    )
    event_out.mkdir(parents=True, exist_ok=True)
    contour_manifest_path = event_out / "contour_evidence_manifest.json"

    local_payload = (comparison.get("arms") or {}).get("metric_local_highlight") or {}
    local_items = [
        item for item in local_payload.get("items", []) if isinstance(item, dict)
    ]
    raw_items = [
        deepcopy(item)
        for item in local_items
        if str(item.get("role") or "") in {"metric_local_rgb", "collision_rgb"}
    ]
    legacy_items = [
        deepcopy(item)
        for item in local_items
        if str(item.get("role") or "")
        in {"metric_local_highlight", "collision_pair_overlay"}
    ]
    if len(raw_items) != 2 or len(legacy_items) != 2:
        raise ValueError("expected exactly two frozen local raw/highlight pose pairs")
    raw_view_ids = [str(item.get("view_id") or "") for item in raw_items]
    if not all(raw_view_ids) or len(set(raw_view_ids)) != 2:
        raise ValueError("frozen local raw view IDs are missing or duplicated")

    source_blend = evidence_root / "cases" / case_id / "scene" / "scene.blend"
    if not source_blend.is_file():
        raise FileNotFoundError(source_blend)
    overlay_spec = _source_overlay_spec(raw_items[0])
    annotation_spec = deepcopy(overlay_spec)
    annotation_spec["object_presentation"] = "annotations_only"
    annotation_spec["role"] = "metric_contour_annotation_base"
    poses = _resolve_frozen_poses(raw_items)
    pose_by_id = {str(pose.get("id")): pose for pose in poses}

    width, height = _common_image_size(raw_items)
    renderer.width = width
    renderer.height = height
    contract = {
        "schema_version": "cal_dataset1_local_contour_evidence_contract_v1",
        "case_id": case_id,
        "metric": metric,
        "event_id": event_id,
        "source_comparison_manifest": str(comparison_path),
        "source_comparison_sha256": _file_sha256(comparison_path),
        "source_blend": str(source_blend),
        "source_blend_sha256": _file_sha256(source_blend),
        "selected_view_ids": raw_view_ids,
        "poses_sha256": _json_sha256(poses),
        "overlay_spec_sha256": _json_sha256(overlay_spec),
        "annotation_spec_sha256": _json_sha256(annotation_spec),
        "style": {
            "band_width_px": band_width_px,
            "outline_width_px": outline_width_px,
            "band_alpha": band_alpha,
            "outline_alpha": outline_alpha,
        },
        "implementation_sha256": {
            "renderer": _file_sha256(
                PROJECT_ROOT / "src" / "benchmark" / "rendering" / "blender.py"
            ),
            "overlay_worker": _file_sha256(
                PROJECT_ROOT
                / "src"
                / "benchmark"
                / "rendering"
                / "blender_collision_overlay_worker.py"
            ),
            "mask_worker": _file_sha256(
                PROJECT_ROOT
                / "src"
                / "benchmark"
                / "rendering"
                / "blender_collision_mask_worker.py"
            ),
            "contour_compositor": _file_sha256(
                PROJECT_ROOT
                / "src"
                / "benchmark"
                / "rendering"
                / "segmentation_contour.py"
            ),
        },
    }
    contract_sha256 = _json_sha256(contract)
    if resume and _cached_manifest_ready(contour_manifest_path, contract_sha256):
        cached = _read_json(contour_manifest_path)
        return {
            "case_id": case_id,
            "metric": metric,
            "event_id": event_id,
            "status": "cached",
            "manifest_path": str(contour_manifest_path),
            "contract_sha256": contract_sha256,
        }

    annotation_manifest = renderer.render_focus_overlay_views(
        blend_file=source_blend,
        out_dir=event_out / "annotation_only",
        camera_views=poses,
        overlay_spec=annotation_spec,
        preview=False,
        allow_blank_views=True,
    )
    mask_manifest = renderer.render_target_id_masks(
        blend_file=source_blend,
        out_dir=event_out / "target_id_masks",
        camera_views=poses,
        overlay_spec=overlay_spec,
        preview=False,
        respect_occlusion=True,
    )
    composed = compose_segmentation_contour_manifest(
        rgb_manifest=annotation_manifest,
        mask_manifest=mask_manifest,
        overlay_spec=overlay_spec,
        out_dir=event_out / "contour",
        band_width_px=band_width_px,
        outline_width_px=outline_width_px,
        band_alpha=band_alpha,
        outline_alpha=outline_alpha,
    )
    contour_by_id = {
        str(view["view_id"]): view
        for view in composed.get("views", [])
        if isinstance(view, dict) and view.get("view_id") is not None
    }
    contour_items: list[dict[str, Any]] = []
    for raw in raw_items:
        view_id = str(raw["view_id"])
        contour = contour_by_id.get(view_id)
        if not isinstance(contour, dict):
            raise RuntimeError(f"missing contour output for frozen view {view_id}")
        path = Path(str(contour["output_path"])).resolve()
        contour_items.append(
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "role": "metric_local_contour",
                "evidence_style": "raw_plus_segmentation_contour",
                "view_id": view_id,
                "pose": deepcopy(pose_by_id[view_id]),
                "metric": metric,
                "target_ids": deepcopy(raw.get("target_ids") or comparison.get("object_ids") or []),
                "color_legend": deepcopy(raw.get("color_legend") or overlay_spec.get("legend")),
                "representation_level": raw.get("representation_level"),
                "segmentation_contour": {
                    "target_interior_policy": "preserve_annotation_only_rgb",
                    "mask_occlusion_policy": "respect_scene_occlusion",
                    "band_width_px": band_width_px,
                    "outline_width_px": outline_width_px,
                    "band_alpha": band_alpha,
                    "outline_alpha": outline_alpha,
                    "visible_targets": {
                        str(target.get("id")): int(
                            target.get("visible_pixels_at_composite_resolution") or 0
                        )
                        for target in contour.get("targets", [])
                        if isinstance(target, dict)
                    },
                },
            }
        )

    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "metric": metric,
        "event_id": event_id,
        "object_ids": deepcopy(comparison.get("object_ids") or []),
        "semantic_label": comparison.get("semantic_label"),
        "severity_class": comparison.get("severity_class"),
        "source_comparison_manifest": str(comparison_path),
        "source_comparison_sha256": _file_sha256(comparison_path),
        "contract": contract,
        "contract_sha256": contract_sha256,
        "global_evidence": "excluded",
        "frozen_local_raw_items": raw_items,
        "frozen_local_legacy_items": legacy_items,
        "contour_items": contour_items,
        "annotation_manifest": str(
            event_out / "annotation_only" / "collision_overlay_manifest.json"
        ),
        "mask_manifest": str(
            event_out / "target_id_masks" / "target_id_mask_manifest.json"
        ),
        "composition_manifest": composed["manifest_path"],
        "complete": True,
    }
    _write_json(contour_manifest_path, output_manifest)
    return {
        "case_id": case_id,
        "metric": metric,
        "event_id": event_id,
        "status": "prepared",
        "manifest_path": str(contour_manifest_path),
        "contract_sha256": contract_sha256,
    }


def _source_overlay_spec(raw_item: dict[str, Any]) -> dict[str, Any]:
    raw_path = Path(str(raw_item.get("path") or "")).expanduser().resolve()
    search_dirs = [raw_path.parent, raw_path.parent.parent]
    for directory in search_dirs:
        request_path = directory / "focus_bundle_request.json"
        if request_path.is_file():
            request = _read_json(request_path)
            spec = request.get("overlay_spec")
            if isinstance(spec, dict) and spec:
                return spec
        collision_path = directory / "collision_overlay_spec.json"
        if collision_path.is_file():
            return _read_json(collision_path)
    raise FileNotFoundError(f"cannot locate overlay spec beside frozen RGB: {raw_path}")


def _resolve_frozen_poses(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_ids = [str(item.get("view_id") or "") for item in raw_items]
    catalog: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        pose = item.get("pose")
        if isinstance(pose, dict) and pose.get("id") is not None:
            catalog[str(pose["id"])] = deepcopy(pose)
    raw_path = Path(str(raw_items[0].get("path") or "")).expanduser().resolve()
    for directory in (raw_path.parent, raw_path.parent.parent):
        camera_views_path = directory / "camera_views.json"
        if camera_views_path.is_file():
            values = json.loads(camera_views_path.read_text(encoding="utf-8"))
            if isinstance(values, list):
                for pose in values:
                    if isinstance(pose, dict) and pose.get("id") is not None:
                        catalog[str(pose["id"])] = pose
        request_path = directory / "focus_bundle_request.json"
        if request_path.is_file():
            request = _read_json(request_path)
            for key in ("local_camera_views", "global_camera_views"):
                for pose in request.get(key, []):
                    if isinstance(pose, dict) and pose.get("id") is not None:
                        catalog[str(pose["id"])] = pose
    missing = [view_id for view_id in required_ids if view_id not in catalog]
    if missing:
        raise ValueError(f"cannot resolve frozen local poses for view IDs: {missing}")
    return [deepcopy(catalog[view_id]) for view_id in required_ids]


def _common_image_size(items: list[dict[str, Any]]) -> tuple[int, int]:
    sizes = set()
    for item in items:
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        with Image.open(path) as opened:
            sizes.add((int(opened.width), int(opened.height)))
    if len(sizes) != 1:
        raise ValueError(f"local raw views do not share one image size: {sorted(sizes)}")
    return next(iter(sizes))


def _cached_manifest_ready(path: Path, contract_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = _read_json(path)
    except Exception:
        return False
    if value.get("contract_sha256") != contract_sha256 or value.get("complete") is not True:
        return False
    items = value.get("contour_items")
    if not isinstance(items, list) or len(items) != 2:
        return False
    return all(
        Path(str(item.get("path") or "")).is_file()
        and _file_sha256(Path(str(item["path"]))) == item.get("sha256")
        for item in items
        if isinstance(item, dict)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=PROJECT_ROOT / "Support" / "artifacts" / "outputs" / "exp1_1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp1_3_local_contour_evidence"
        ),
    )
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=Path("/Applications/Blender.app/Contents/MacOS/Blender"),
    )
    parser.add_argument("--metric", action="append", choices=METRICS, default=[])
    parser.add_argument("--severity", action="append", choices=SEVERITIES, default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--band-width-px", type=int, default=7)
    parser.add_argument("--outline-width-px", type=int, default=2)
    parser.add_argument("--band-alpha", type=float, default=0.30)
    parser.add_argument("--outline-alpha", type=float, default=0.95)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--require-event-count", type=int, default=24)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not args.metric:
        args.metric = list(METRICS)
    if not args.severity:
        args.severity = list(SEVERITIES)
    return args


if __name__ == "__main__":
    main()
