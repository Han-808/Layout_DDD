"""Blender-side worker for a bundled metric-evidence final pass.

The worker renders ordinary RGB local views, applies the read-only focus
overlay in memory, then renders highlighted local and global views. Keeping all
final passes in one Blender process avoids repeated scene loading and Cycles
device/kernel initialization. The source ``.blend`` is never saved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from blender_collision_overlay_worker import _apply_overlay, _overlay_targets, _render_pose
from blender_worker import _configure_render


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("focus bundle request must be a JSON object")
    local_views = request.get("local_camera_views")
    global_views = request.get("global_camera_views")
    overlay_spec = request.get("overlay_spec")
    if not isinstance(local_views, list) or not local_views:
        raise ValueError("focus bundle requires non-empty local_camera_views")
    if not isinstance(global_views, list):
        raise TypeError("focus bundle global_camera_views must be a list")
    if not isinstance(overlay_spec, dict) or not overlay_spec:
        raise ValueError("focus bundle overlay_spec must be a non-empty object")

    render_config = _configure_render(
        args.width,
        args.height,
        args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
    )
    rgb_views = [
        _render_pose(pose, out_dir, index, role="metric_rgb", filename_prefix="rgb")
        for index, pose in enumerate(local_views)
    ]

    degradations = _apply_overlay(overlay_spec)
    overlay_role = str(overlay_spec.get("role") or "metric_focus_overlay")
    overlay_views = [
        _render_pose(pose, out_dir, index, role=overlay_role, filename_prefix="highlight")
        for index, pose in enumerate(local_views)
    ]
    global_overlay_views = [
        _render_pose(
            pose,
            out_dir,
            index,
            role="metric_highlighted_global",
            filename_prefix="global_highlight",
        )
        for index, pose in enumerate(global_views)
    ]
    targets = _overlay_targets(overlay_spec)
    manifest = {
        "backend": "blender_read_only_focus_bundle_v1",
        "source_blend": str(Path(bpy.data.filepath).resolve()) if bpy.data.filepath else None,
        "source_scene_saved": False,
        "scene_mutation_scope": "ephemeral_camera_material_wireframe_marker_only",
        "render_engine": args.render_engine,
        "render_config": render_config,
        "metric": overlay_spec.get("metric"),
        "target_ids": [target.get("id") for target in targets],
        "legend": overlay_spec.get("legend"),
        "representation_level": overlay_spec.get("representation_level"),
        "diagnostic_degradations": degradations,
        "rgb_views": rgb_views,
        "overlay_views": overlay_views,
        "global_overlay_views": global_overlay_views,
        "views": rgb_views + overlay_views + global_overlay_views,
    }
    (out_dir / "focus_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--render-engine",
        choices=["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"],
        default="BLENDER_WORKBENCH",
    )
    parser.add_argument("--cycles-device", choices=["CPU", "CUDA", "OPTIX", "AUTO"], default="CPU")
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--cycles-denoising", action="store_true")
    return parser.parse_args(values)


if __name__ == "__main__":
    main()
