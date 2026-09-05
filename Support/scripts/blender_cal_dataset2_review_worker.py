"""Blender-side one-process worker for cal_dataset2 human-review views.

This worker deliberately constructs and renders a canonical scene in one
process.  It avoids saving and reopening a .blend, which is unstable on some
macOS Blender builds and is unnecessary for static human-review images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


REPO_ROOT = Path(__file__).resolve().parents[2]
RENDERING_DIR = REPO_ROOT / "src" / "benchmark" / "rendering"
if str(RENDERING_DIR) not in sys.path:
    sys.path.insert(0, str(RENDERING_DIR))

import blender_camera_worker as camera_worker  # noqa: E402
import blender_collision_mask_worker as mask_worker  # noqa: E402
import blender_worker as scene_worker  # noqa: E402


def main() -> None:
    args = _parse_args()
    scene_path = Path(args.scene_json).resolve()
    camera_path = Path(args.camera_views_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    asset_root = Path(args.asset_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = out_dir / "local_views"
    local_dir.mkdir(parents=True, exist_ok=True)

    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    poses = json.loads(camera_path.read_text(encoding="utf-8"))
    if not isinstance(scene, dict):
        raise TypeError("scene JSON must contain an object")
    if not isinstance(poses, list) or not poses:
        raise TypeError("camera views JSON must contain a non-empty list")

    scene_worker._clear_scene()
    render_config = scene_worker._configure_render(
        args.width,
        args.height,
        "BLENDER_WORKBENCH",
        cycles_device="CPU",
        cycles_samples=1,
        cycles_denoising=False,
    )
    boundary = scene_worker._boundary(scene)
    room_height = float(scene.get("scene_height") or 2.8)
    scene_worker._build_room(boundary, room_height)
    object_results = []
    for item in scene.get("objects", []):
        if not isinstance(item, dict):
            continue
        anchor = scene_worker._vertical_anchor_spec(item, scene)
        object_results.append(
            scene_worker._build_object(
                item,
                asset_root,
                vertical_anchor=anchor["vertical_anchor"],
                vertical_anchor_source=anchor["source"],
            )
        )
    scene_worker._add_lighting(boundary, room_height)
    oar_plane_flags = json.loads(args.oar_plane_flags_json)
    mask_targets = json.loads(args.mask_targets_json)
    base_views = (
        []
        if args.mask_only
        else scene_worker._render_views(boundary, room_height, out_dir)
    )
    local_room_repairs = _move_active_wall_shell_outside_boundary(
        boundary,
        oar_plane_flags,
    )
    local_views = (
        []
        if args.mask_only
        else [
            camera_worker._render_pose(pose, local_dir, index)
            for index, pose in enumerate(poses)
        ]
    )
    mask_manifest = (
        _render_target_id_masks(
            poses=poses,
            targets=mask_targets,
            out_dir=out_dir / "target_id_masks",
            width=args.width,
            height=args.height,
        )
        if mask_targets
        else None
    )
    asset_mesh_count = sum(
        item.get("representation") == "asset_mesh" for item in object_results
    )
    proxy_count = sum(
        item.get("representation") == "bbox_proxy" for item in object_results
    )
    manifest = {
        "schema_version": "cal_dataset2_blender_review_worker_v1",
        "backend": "one_process_canonical_scene_plus_camera_views",
        "blender_version": bpy.app.version_string,
        "scene_json": str(scene_path),
        "asset_root": str(asset_root),
        "render_engine": "BLENDER_WORKBENCH",
        "render_config": render_config,
        "scene_saved": False,
        "base_views": base_views,
        "local_views": local_views,
        "mask_manifest": mask_manifest,
        "local_room_repairs": local_room_repairs,
        "objects": object_results,
        "asset_coverage": {
            "object_count": len(object_results),
            "asset_mesh_count": asset_mesh_count,
            "bbox_proxy_count": proxy_count,
            "asset_mesh_rate": (
                asset_mesh_count / len(object_results) if object_results else 0.0
            ),
            "required": True,
        },
    }
    (out_dir / "review_worker_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-json", required=True)
    parser.add_argument("--camera-views-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--oar-plane-flags-json", default="{}")
    parser.add_argument("--mask-targets-json", default="[]")
    parser.add_argument("--mask-only", action="store_true")
    return parser.parse_args(values)


def _render_target_id_masks(
    *,
    poses: list[dict[str, object]],
    targets: list[dict[str, object]],
    out_dir: Path,
    width: int,
    height: int,
) -> dict[str, object]:
    if not isinstance(targets, list) or not targets:
        raise ValueError("mask targets must contain at least one target")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_config = scene_worker._configure_render(
        width,
        height,
        "BLENDER_WORKBENCH",
        cycles_device="CPU",
        cycles_samples=1,
        cycles_denoising=False,
    )
    mask_worker._configure_flat_mask_world(respect_occlusion=True)
    overlay_spec = {"targets": targets}
    resolved_targets = mask_worker._overlay_targets(overlay_spec)
    target_ids = [
        str(target["id"])
        for target in resolved_targets
        if target.get("id") is not None
    ]
    resolver = mask_worker._CanonicalResolver()
    mesh_by_target = mask_worker._group_mesh_objects(resolver, target_ids)
    views = [
        mask_worker._render_mask_view(
            pose,
            index,
            out_dir,
            resolved_targets,
            mesh_by_target,
            {},
            respect_occlusion=True,
        )
        for index, pose in enumerate(poses)
    ]
    manifest = {
        "schema_version": "cal_dataset2_target_id_masks_v1",
        "backend": "one_process_visible_object_identity_masks",
        "source_scene_saved": False,
        "role": "target_id_masks",
        "occlusion_policy": "respect_scene_occlusion",
        "render_config": render_config,
        "target_ids": target_ids,
        "views": views,
    }
    manifest_path = out_dir / "target_id_mask_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _move_active_wall_shell_outside_boundary(
    boundary: list[list[float]],
    plane_flags: dict[str, object],
) -> list[dict[str, object]]:
    """Keep the review-only wall shell from swallowing a wall-mounted asset.

    ``blender_worker._build_room`` centres its 8 cm wall cubes on the canonical
    boundary.  A target whose canonical bounds end exactly at that boundary can
    consequently sit inside the visualization shell.  Global views intentionally
    keep the ordinary room render.  Before local OAR views only, move the active
    cardinal wall outward by half its thickness plus a 2 mm clearance.
    """

    if not isinstance(plane_flags, dict) or not plane_flags:
        return []
    min_x = min(float(point[0]) for point in boundary)
    max_x = max(float(point[0]) for point in boundary)
    min_y = min(float(point[1]) for point in boundary)
    max_y = max(float(point[1]) for point in boundary)
    requested = {
        "west_oob": ("x", min_x, -1.0),
        "east_oob": ("x", max_x, 1.0),
        "south_oob": ("y", min_y, -1.0),
        "north_oob": ("y", max_y, 1.0),
    }
    repairs: list[dict[str, object]] = []
    tolerance = 1.0e-5
    for flag, (axis, coordinate, sign) in requested.items():
        if not plane_flags.get(flag):
            continue
        for index, start in enumerate(boundary):
            end = boundary[(index + 1) % len(boundary)]
            values = (
                (float(start[0]), float(end[0]))
                if axis == "x"
                else (float(start[1]), float(end[1]))
            )
            if any(abs(value - coordinate) > tolerance for value in values):
                continue
            wall = bpy.data.objects.get(f"benchmark_wall_{index:02d}")
            if wall is None:
                continue
            delta = 0.042 * sign
            if axis == "x":
                wall.location.x += delta
            else:
                wall.location.y += delta
            repairs.append(
                {
                    "plane_flag": flag,
                    "wall_object": wall.name,
                    "axis": axis,
                    "outward_delta_m": delta,
                    "reason": "review_shell_outside_canonical_boundary",
                }
            )
            break
    return repairs


if __name__ == "__main__":
    main()
