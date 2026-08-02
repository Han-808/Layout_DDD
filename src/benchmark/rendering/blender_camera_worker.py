"""Blender-side worker for read-only, event-targeted camera evidence.

The source ``.blend`` is loaded by the parent Blender process. This worker adds
only ephemeral cameras and Track-To targets, renders the requested views, and
never saves over the source scene.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from blender_worker import _configure_render, _source_architecture_contract


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = json.loads(Path(args.camera_views_json).read_text(encoding="utf-8"))
    if not isinstance(poses, list) or not poses:
        raise ValueError("camera views JSON must contain a non-empty array")
    render_config = _configure_render(
        args.width,
        args.height,
        args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
    )
    views = [_render_pose(pose, out_dir, index) for index, pose in enumerate(poses)]
    architecture = _source_architecture_contract()
    manifest = {
        "backend": "blender_read_only_camera_evidence_v1",
        "source_blend": str(Path(bpy.data.filepath).resolve()) if bpy.data.filepath else None,
        "source_scene_saved": False,
        "scene_mutation_scope": "ephemeral_camera_and_track_target_only",
        "render_engine": args.render_engine,
        "render_config": render_config,
        "views": views,
        "architecture": architecture,
        "architecture_policy_version": (
            architecture.get("architecture_policy_version")
            if isinstance(architecture, dict)
            else None
        ),
        "wall_policy": (
            (architecture.get("physical_walls") or {}).get("policy")
            if isinstance(architecture, dict)
            else None
        ),
        "active_wall_ids": (
            list(
                (architecture.get("physical_walls") or {}).get(
                    "active_wall_ids"
                )
                or []
            )
            if isinstance(architecture, dict)
            else []
        ),
    }
    (out_dir / "camera_render_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-views-json", required=True)
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


def _render_pose(pose: dict, out_dir: Path, index: int) -> dict:
    if not isinstance(pose, dict):
        raise TypeError("each camera pose must be a JSON object")
    pose_id = str(pose.get("id") or f"view_{index:02d}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", pose_id).strip("._") or f"view_{index:02d}"
    location = _vector3(pose.get("location"), "location")
    target_location = _vector3(pose.get("target"), "target")

    target = bpy.data.objects.new(f"evidence_target_{safe_id}", None)
    target.empty_display_type = "PLAIN_AXES"
    target.location = target_location
    bpy.context.collection.objects.link(target)

    camera_data = bpy.data.cameras.new(f"evidence_camera_{safe_id}")
    camera_data.type = str(pose.get("camera_type") or "PERSP")
    camera_data.lens = float(pose.get("lens_mm") or 50.0)
    camera_data.sensor_width = float(pose.get("sensor_width_mm") or 36.0)
    camera_data.sensor_fit = str(pose.get("sensor_fit") or "HORIZONTAL").upper()
    camera_data.clip_start = max(0.001, float(pose.get("clip_start_m") or 0.02))
    camera_data.clip_end = max(camera_data.clip_start + 1.0, float(pose.get("clip_end_m") or 100.0))
    if camera_data.type == "ORTHO":
        camera_data.ortho_scale = max(0.1, float(pose.get("ortho_scale") or 5.0))
    camera = bpy.data.objects.new(f"evidence_camera_{safe_id}", camera_data)
    camera.location = location
    bpy.context.collection.objects.link(camera)
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    bpy.context.view_layer.update()

    bpy.context.scene.camera = camera
    render_path = out_dir / f"camera_{index:02d}_{safe_id}.png"
    bpy.context.scene.render.filepath = str(render_path)
    started_at = time.monotonic()
    bpy.ops.render.render(write_still=True)
    return {
        "id": pose_id,
        "name": str(pose.get("name") or pose_id),
        "path": str(render_path),
        "camera_location": [float(value) for value in camera.location],
        "camera_target": [float(value) for value in target.location],
        "camera_type": camera_data.type,
        "lens_mm": float(camera_data.lens),
        "track_constraint": {
            "type": "TRACK_TO",
            "track_axis": "TRACK_NEGATIVE_Z",
            "up_axis": "UP_Y",
        },
        "elapsed_seconds": time.monotonic() - started_at,
        "pose": pose,
    }


def _vector3(value: object, label: str) -> Vector:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"camera pose {label} must be a three-vector")
    try:
        return Vector(tuple(float(item) for item in value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"camera pose {label} must be numeric") from exc


if __name__ == "__main__":
    main()
