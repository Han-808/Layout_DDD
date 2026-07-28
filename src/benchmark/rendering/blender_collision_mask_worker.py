"""Blender-side worker for read-only per-target identity masks.

For each candidate pose this renders one binary mask per canonical target.
The historical selector mode hides every other object and therefore measures
the target's unoccluded screen projection.  The opt-in ``--respect-occlusion``
mode instead keeps scene meshes as black depth-writing occluders and renders
the target white, producing a true visible 2D segmentation suitable for
presentation overlays.

The worker is per-candidate lenient: it records a status per view and never
raises for a blank candidate. It only mutates in-memory render visibility and
object color and never saves the source ``.blend``.

It uses only the standard library plus Blender's bundled ``bpy``/``mathutils``
and the sibling worker helpers, so it must not import the ``benchmark`` package.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from array import array
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from blender_worker import _configure_render  # noqa: E402
from blender_collision_overlay_worker import _CanonicalResolver, _overlay_targets, _vector3  # noqa: E402

_FOREGROUND = 0.5


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = json.loads(Path(args.camera_views_json).read_text(encoding="utf-8"))
    overlay_spec = json.loads(Path(args.overlay_spec_json).read_text(encoding="utf-8"))
    if not isinstance(poses, list) or not poses:
        raise ValueError("camera views JSON must contain a non-empty array")
    if not isinstance(overlay_spec, dict):
        raise ValueError("overlay spec JSON must be an object")

    render_config = _configure_render(
        args.width,
        args.height,
        "BLENDER_WORKBENCH",
        cycles_device="CPU",
        cycles_samples=1,
        cycles_denoising=False,
    )
    _configure_flat_mask_world(respect_occlusion=args.respect_occlusion)
    targets = _overlay_targets(overlay_spec)
    target_ids = [str(target.get("id")) for target in targets if target.get("id") is not None]
    resolver = _CanonicalResolver()
    mesh_by_target = _group_mesh_objects(resolver, target_ids)
    focus = overlay_spec.get("focus") if isinstance(overlay_spec.get("focus"), dict) else {}

    views = [
        _render_mask_view(
            pose,
            index,
            out_dir,
            targets,
            mesh_by_target,
            focus,
            respect_occlusion=args.respect_occlusion,
        )
        for index, pose in enumerate(poses)
    ]
    manifest = {
        "backend": "blender_read_only_target_id_mask_v1",
        "source_blend": str(Path(bpy.data.filepath).resolve()) if bpy.data.filepath else None,
        "source_scene_saved": False,
        "scene_mutation_scope": "ephemeral_camera_and_render_visibility_only",
        "render_config": render_config,
        "role": "target_id_masks",
        "occlusion_policy": (
            "respect_scene_occlusion"
            if args.respect_occlusion
            else "ignore_scene_occlusion_projection_proxy"
        ),
        "target_ids": target_ids,
        "views": views,
    }
    (out_dir / "target_id_mask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-views-json", required=True)
    parser.add_argument("--overlay-spec-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--respect-occlusion", action="store_true")
    return parser.parse_args(values)


def _configure_flat_mask_world(*, respect_occlusion: bool) -> None:
    scene = bpy.context.scene
    try:
        scene.render.film_transparent = False
        if scene.world is not None:
            scene.world.color = (0.0, 0.0, 0.0)
        shading = scene.display.shading
        shading.light = "FLAT"
        shading.color_type = "OBJECT" if respect_occlusion else "SINGLE"
        shading.single_color = (1.0, 1.0, 1.0)
        shading.show_shadows = False
        shading.show_cavity = False
        shading.background_type = "VIEWPORT"
        shading.background_color = (0.0, 0.0, 0.0)
    except Exception:  # pragma: no cover - depends on Blender build
        pass


def _group_mesh_objects(resolver: "_CanonicalResolver", target_ids: list[str]) -> dict[str, list]:
    grouped: dict[str, list] = {target_id: [] for target_id in target_ids}
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        canonical = resolver.resolve(obj)
        if canonical in grouped:
            grouped[canonical].append(obj)
    return grouped


def _render_mask_view(
    pose,
    index,
    out_dir,
    targets,
    mesh_by_target,
    focus,
    *,
    respect_occlusion: bool,
) -> dict:
    if not isinstance(pose, dict):
        raise TypeError("each camera pose must be a JSON object")
    pose_id = str(pose.get("id") or f"view_{index:02d}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", pose_id).strip("._") or f"view_{index:02d}"
    camera = _make_camera(pose, safe_id)
    bpy.context.scene.camera = camera
    all_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    saved_hide = {obj.name: obj.hide_render for obj in all_meshes}
    saved_colors = {obj.name: tuple(obj.color) for obj in all_meshes}
    per_target: dict[str, dict] = {}
    status = "ok"
    started = time.monotonic()
    try:
        for target in targets:
            target_id = str(target.get("id"))
            members = mesh_by_target.get(target_id, [])
            member_names = {obj.name for obj in members}
            for obj in all_meshes:
                if respect_occlusion:
                    obj.hide_render = saved_hide.get(obj.name, False)
                    obj.color = (
                        (1.0, 1.0, 1.0, 1.0)
                        if obj.name in member_names
                        else (0.0, 0.0, 0.0, 1.0)
                    )
                else:
                    obj.hide_render = obj not in members
            mask_path = out_dir / f"mask_{index:02d}_{safe_id}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', target_id)}.png"
            visible_pixels, image_pixels = _render_binary_mask(mask_path)
            per_target[target_id] = {
                "mask_path": str(mask_path),
                "visible_pixels": int(visible_pixels),
                "image_pixel_count": int(image_pixels),
                "projected_obb_area_px": _projected_obb_area(camera, target.get("obb"), image_pixels),
            }
        if per_target and all(entry["visible_pixels"] == 0 for entry in per_target.values()):
            status = "blank"
    except Exception as exc:  # pragma: no cover - depends on Blender build
        status = "failed"
        per_target.setdefault("error", f"{type(exc).__name__}: {exc}")
    finally:
        for obj in all_meshes:
            obj.hide_render = saved_hide.get(obj.name, False)
            obj.color = saved_colors.get(obj.name, (0.8, 0.8, 0.8, 1.0))
    focus_info = _focus_projection(camera, focus)
    return {
        "id": pose_id,
        "status": status,
        "targets": per_target,
        "focus": focus_info,
        "elapsed_seconds": time.monotonic() - started,
        "pose": pose,
    }


def _make_camera(pose: dict, safe_id: str):
    target = bpy.data.objects.new(f"mask_target_{safe_id}", None)
    target.location = _vector3(pose.get("target"), "target")
    bpy.context.collection.objects.link(target)
    camera_data = bpy.data.cameras.new(f"mask_camera_{safe_id}")
    camera_data.type = str(pose.get("camera_type") or "PERSP")
    camera_data.lens = float(pose.get("lens_mm") or 50.0)
    camera_data.sensor_width = float(pose.get("sensor_width_mm") or 36.0)
    camera_data.sensor_fit = str(pose.get("sensor_fit") or "HORIZONTAL").upper()
    camera_data.clip_start = max(0.001, float(pose.get("clip_start_m") or 0.02))
    camera_data.clip_end = max(camera_data.clip_start + 1.0, float(pose.get("clip_end_m") or 100.0))
    if camera_data.type == "ORTHO":
        camera_data.ortho_scale = max(0.1, float(pose.get("ortho_scale") or 5.0))
    camera = bpy.data.objects.new(f"mask_camera_{safe_id}", camera_data)
    camera.location = _vector3(pose.get("location"), "location")
    bpy.context.collection.objects.link(camera)
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    bpy.context.view_layer.update()
    return camera


def _render_binary_mask(mask_path: Path) -> tuple[int, int]:
    bpy.context.scene.render.filepath = str(mask_path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(mask_path), check_existing=False)
    try:
        # Blender 5.2 may leave ``has_data`` false until the lazy pixel buffer
        # is accessed even though a valid PNG was just loaded. Size/pixel
        # length are the reliable cross-version checks.
        width, height = int(image.size[0]), int(image.size[1])
        channels = max(1, int(image.channels))
        pixel_count = max(0, width * height)
        if pixel_count == 0:
            return 0, 0
        pixels = image.pixels
        required_values = pixel_count * channels
        if len(pixels) < required_values:
            return 0, 0
        # Reading one RNA element at a time is extremely slow on macOS
        # (roughly 20 seconds for a 512x512 RGBA mask), leaving the background
        # Blender process looking unresponsive in the Dock.  Bulk-copy once,
        # then count from ordinary Python memory.
        pixel_buffer = array("f", [0.0]) * required_values
        pixels.foreach_get(pixel_buffer)
        visible = sum(
            float(pixel_buffer[pixel_index * channels]) >= _FOREGROUND
            for pixel_index in range(pixel_count)
        )
        return visible, pixel_count
    finally:
        bpy.data.images.remove(image)


def _projected_obb_area(camera, obb, image_pixels) -> float | None:
    if not isinstance(obb, dict) or not obb.get("corners") or not image_pixels:
        return None
    scene = bpy.context.scene
    us: list[float] = []
    vs: list[float] = []
    for corner in obb["corners"]:
        try:
            projected = world_to_camera_view(scene, camera, Vector(tuple(float(v) for v in corner)))
        except Exception:  # pragma: no cover
            return None
        us.append(float(projected.x))
        vs.append(float(projected.y))
    if not us:
        return None
    width = max(0.0, min(1.0, max(us)) - max(0.0, min(us)))
    height = max(0.0, min(1.0, max(vs)) - max(0.0, min(vs)))
    return float(width * height * image_pixels)


def _focus_projection(camera, focus) -> dict:
    center = focus.get("center") if isinstance(focus, dict) else None
    if not isinstance(center, (list, tuple)) or len(center) != 3:
        return {"projected_uv": None, "in_frame": None}
    scene = bpy.context.scene
    try:
        projected = world_to_camera_view(scene, camera, Vector(tuple(float(v) for v in center)))
    except Exception:  # pragma: no cover
        return {"projected_uv": None, "in_frame": None}
    uv = [float(projected.x), float(projected.y)]
    margin = 0.06
    in_frame = bool(margin <= uv[0] <= 1.0 - margin and margin <= uv[1] <= 1.0 - margin and projected.z > 0.0)
    return {"projected_uv": uv, "in_frame": in_frame}


if __name__ == "__main__":
    main()
