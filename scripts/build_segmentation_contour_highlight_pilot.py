#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.rendering.blender import BlenderRenderer
from benchmark.rendering.segmentation_contour import compose_segmentation_contour_manifest


def main() -> None:
    args = _parse_args()
    event_dir = args.event_dir.expanduser().resolve()
    output_dir = args.out_dir.expanduser().resolve()
    request = _read_json(event_dir / "focus_bundle_request.json")
    rgb_manifest = _read_json(event_dir / "focus_bundle_manifest.json")
    overlay_spec = request.get("overlay_spec")
    if not isinstance(overlay_spec, dict):
        raise ValueError("focus_bundle_request.json does not contain overlay_spec")
    rgb_views = [
        view
        for view in rgb_manifest.get("rgb_views", [])
        if isinstance(view, dict) and view.get("id") is not None
    ]
    requested_poses = [
        pose
        for key in ("local_camera_views", "global_camera_views")
        for pose in request.get(key, [])
        if isinstance(pose, dict) and pose.get("id") is not None
    ]
    pose_by_id = {str(pose["id"]): pose for pose in requested_poses}
    poses = [pose_by_id[str(view["id"])] for view in rgb_views if str(view["id"]) in pose_by_id]
    if len(poses) != len(rgb_views):
        raise ValueError("could not resolve every RGB view to its same-pose camera request")

    source_blend = Path(str(rgb_manifest.get("source_blend") or "")).expanduser().resolve()
    render_config = rgb_manifest.get("render_config") or {}
    width = int(render_config.get("width") or 512)
    height = int(render_config.get("height") or 512)
    mask_dir = output_dir / "target_id_masks"
    renderer = BlenderRenderer(
        blender_bin=args.blender_bin,
        timeout_seconds=args.timeout_seconds,
        width=width,
        height=height,
        render_engine="BLENDER_WORKBENCH",
        preview_render_engine="BLENDER_WORKBENCH",
        preview_width=min(width, 256),
        preview_height=min(height, 256),
    )
    mask_manifest_path = mask_dir / "target_id_mask_manifest.json"
    mask_manifest = _read_json(mask_manifest_path) if mask_manifest_path.is_file() else None
    camera_evidence = (
        mask_manifest.get("camera_evidence")
        if isinstance(mask_manifest, dict) and isinstance(mask_manifest.get("camera_evidence"), dict)
        else {}
    )
    if camera_evidence.get("occlusion_policy") != "respect_scene_occlusion":
        mask_manifest = renderer.render_target_id_masks(
            blend_file=source_blend,
            out_dir=mask_dir,
            camera_views=poses,
            overlay_spec=overlay_spec,
            preview=False,
            respect_occlusion=True,
        )
    contour_manifest = compose_segmentation_contour_manifest(
        rgb_manifest=rgb_manifest,
        mask_manifest=mask_manifest,
        overlay_spec=overlay_spec,
        out_dir=output_dir / "contour",
        band_width_px=args.band_width_px,
        outline_width_px=args.outline_width_px,
        band_alpha=args.band_alpha,
        outline_alpha=args.outline_alpha,
    )
    comparison_path = output_dir / "comparison.png"
    _comparison_sheet(
        rgb_views=rgb_views,
        highlight_views=[
            view
            for view in rgb_manifest.get("overlay_views", [])
            if isinstance(view, dict) and view.get("id") is not None
        ],
        contour_views=contour_manifest["views"],
        out_path=comparison_path,
    )
    summary = {
        "schema_version": "segmentation_contour_pilot_v1",
        "event_dir": str(event_dir),
        "source_blend": str(source_blend),
        "metric": rgb_manifest.get("metric"),
        "target_ids": rgb_manifest.get("target_ids"),
        "mask_manifest": str(mask_manifest_path),
        "contour_manifest": contour_manifest["manifest_path"],
        "comparison": str(comparison_path),
        "view_count": len(contour_manifest["views"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _comparison_sheet(
    *,
    rgb_views: list[dict[str, Any]],
    highlight_views: list[dict[str, Any]],
    contour_views: list[dict[str, Any]],
    out_path: Path,
) -> None:
    highlight_by_id = {str(view.get("id")): view.get("path") for view in highlight_views}
    contour_by_id = {str(view.get("view_id")): view.get("output_path") for view in contour_views}
    cells: list[tuple[str, Image.Image]] = []
    for rgb_view in rgb_views:
        view_id = str(rgb_view.get("id"))
        candidates = [
            ("Raw RGB", rgb_view.get("path")),
            ("Current full recolor", highlight_by_id.get(view_id)),
            ("Segmentation contour", contour_by_id.get(view_id)),
        ]
        for label, path in candidates:
            if not path or not Path(str(path)).is_file():
                continue
            with Image.open(str(path)) as opened:
                image = opened.convert("RGB")
            cells.append((f"{view_id}\n{label}", image))
    if not cells:
        raise ValueError("no comparison images were available")
    columns = 3
    width = max(image.width for _, image in cells)
    height = max(image.height for _, image in cells)
    label_height = 54
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), (28, 28, 30))
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(cells):
        column = index % columns
        row = index // columns
        x = column * width
        y = row * (height + label_height)
        sheet.paste(image, (x, y))
        draw.multiline_text((x + 10, y + height + 6), label, fill=(245, 245, 245), spacing=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=Path("/Applications/Blender.app/Contents/MacOS/Blender"),
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--band-width-px", type=int, default=7)
    parser.add_argument("--outline-width-px", type=int, default=2)
    parser.add_argument("--band-alpha", type=float, default=0.30)
    parser.add_argument("--outline-alpha", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    main()
