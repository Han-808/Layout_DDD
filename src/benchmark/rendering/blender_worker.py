"""Trusted Blender-side worker for canonical scene JSON.

This file is launched by Blender and intentionally uses only the standard
library plus Blender's bundled bpy/mathutils modules.

Real asset meshes are placed with uniform "contain" scaling: the imported mesh
is scaled by a single factor (``min`` of the per-axis fit ratios) so it fits
inside the canonical target bounding box without distorting its aspect ratio.
The mesh may therefore leave slack along one or two axes.  To keep rendered
support geometry consistent with canonical placement, that slack is anchored at
the local top for an explicit ceiling attachment and at the local bottom for all
other objects. Proxy fallbacks continue to use the canonical bounding box
directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

try:  # pragma: no cover - bpy/mathutils only exist inside Blender
    import bpy
    from mathutils import Euler, Vector
except ModuleNotFoundError:  # allow importing the pure fit/centering helpers for unit tests
    bpy = None  # type: ignore[assignment]
    Euler = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]


FIT_MODE_UNIFORM_CONTAIN = "uniform_contain"
FIT_MODE_BBOX_PROXY = "bbox_proxy"
VERTICAL_ANCHOR_BOTTOM = "bottom"
VERTICAL_ANCHOR_TOP = "top"
# Canonical benchmark object id written as a custom property on each constructed
# root so downstream read-only passes (e.g. the collision overlay) can resolve a
# multi-child asset to one object id even after Blender name mangling. Older
# ``.blend`` files that only carry ``asset_<id>`` / ``proxy_<id>`` names still
# resolve by name.
CANONICAL_ID_PROPERTY = "benchmark_object_id"
ARCHITECTURE_CONTRACT_PROPERTY = "benchmark_architecture_contract"
CANONICAL_WALL_IDS = (
    "north_wall",
    "south_wall",
    "east_wall",
    "west_wall",
)
DEFAULT_COLLISION_MAX_VERTICES_PER_OBJECT = 50_000
DEFAULT_COLLISION_MAX_FACES_PER_OBJECT = 100_000
DEFAULT_COLLISION_MAX_TOTAL_VERTICES = 200_000
DEFAULT_COLLISION_MAX_TOTAL_FACES = 400_000
COLLISION_GEOMETRY_CENTER_TOLERANCE_M = 0.05


def main() -> None:
    args = _parse_args()
    scene_path = Path(args.scene_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    asset_root = Path(args.asset_root).resolve() if args.asset_root else None
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "blender_worker_progress.jsonl"
    progress_path.unlink(missing_ok=True)
    _record_progress(progress_path, "worker_started")
    scene_data = json.loads(scene_path.read_text(encoding="utf-8"))
    architecture = json.loads(
        Path(args.architecture_contract).resolve().read_text(encoding="utf-8")
    )
    active_wall_ids = _active_wall_ids(architecture)
    canonical_object_ids = _canonical_object_ids(scene_data)

    _clear_scene()
    render_config = _configure_render(
        args.width,
        args.height,
        args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
    )
    _record_progress(progress_path, "render_configured", render_config=render_config)
    boundary = _boundary(scene_data)
    room_height = float(scene_data.get("scene_height") or 2.8)
    rendered_wall_ids = _build_room(
        boundary,
        room_height,
        active_wall_ids=active_wall_ids,
    )
    bpy.context.scene[ARCHITECTURE_CONTRACT_PROPERTY] = json.dumps(
        architecture,
        sort_keys=True,
        separators=(",", ":"),
    )
    _record_progress(
        progress_path,
        "room_built",
        active_wall_ids=active_wall_ids,
        rendered_wall_ids=rendered_wall_ids,
    )
    object_results = []
    for item in scene_data.get("objects", []):
        if not isinstance(item, dict):
            continue
        anchor = _vertical_anchor_spec(item, scene_data)
        result = _build_object(
            item,
            asset_root,
            vertical_anchor=anchor["vertical_anchor"],
            vertical_anchor_source=anchor["source"],
        )
        object_results.append(result)
        _record_progress(
            progress_path,
            "object_built",
            object_id=result.get("id"),
            representation=result.get("representation"),
        )
    _add_lighting(boundary, room_height)
    _record_progress(progress_path, "render_started")
    views = _render_views(boundary, room_height, out_dir, progress_path=progress_path)
    identity_legend: dict[str, str] = {}
    identity_palette: dict[str, str] = {}
    identity_render = {
        "status": "not_applicable",
        "reason": "scene_has_no_renderable_objects",
        "camera_source": "standardized_perspective",
        "architecture_identity": "neutral_background",
        "canonical_object_count": 0,
        "scene_mutated": False,
    }
    if canonical_object_ids:
        identity_view, identity_legend, identity_palette = _render_identity_map(
            canonical_object_ids,
            out_dir,
            progress_path=progress_path,
        )
        views.append(identity_view)
        identity_render = {
            "status": "available",
            "camera_source": "standardized_perspective",
            "architecture_identity": "neutral_background",
            "canonical_object_count": len(canonical_object_ids),
            "scene_mutated": False,
            "color_encoding": "raw_linear_rgb_8bit",
        }
    _record_progress(progress_path, "render_completed", view_count=len(views))
    blend_path = out_dir / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    _record_progress(progress_path, "blend_saved", path=str(blend_path))

    manifest = {
        "backend": "blender_canonical_scene_v1",
        "blender_version": bpy.app.version_string,
        "scene_json": str(scene_path),
        "asset_root": str(asset_root) if asset_root else None,
        "rotation_unit": "degree",
        "render_engine": args.render_engine,
        "render_config": render_config,
        "blend_file": str(blend_path),
        "views": views,
        "identity_legend": identity_legend,
        "identity_palette": identity_palette,
        "identity_render": identity_render,
        "objects": object_results,
        "collision_geometry_manifest": None,
        "collision_geometry_export": {
            "status": "pending",
            "limits": _collision_geometry_limits(args),
        },
        "architecture": architecture,
        "architecture_policy_version": architecture.get(
            "architecture_policy_version"
        ),
        "wall_policy": architecture["physical_walls"]["policy"],
        "logical_boundary_present": bool(
            architecture["logical_boundary"]["enabled"]
        ),
        "physical_walls_enabled": bool(active_wall_ids),
        "active_wall_ids": active_wall_ids,
        "activation_sources": list(
            architecture["physical_walls"].get("activation_sources") or []
        ),
        "rendered_wall_ids": rendered_wall_ids,
        "floor_rendered": True,
    }
    render_manifest_path = out_dir / "render_manifest.json"
    render_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _record_progress(progress_path, "base_manifest_written", path=str(render_manifest_path))

    geometry_manifest_path = _write_collision_geometry_manifest(
        out_dir,
        object_results,
        max_vertices_per_object=args.collision_max_vertices_per_object,
        max_faces_per_object=args.collision_max_faces_per_object,
        max_total_vertices=args.collision_max_total_vertices,
        max_total_faces=args.collision_max_total_faces,
    )
    manifest["collision_geometry_manifest"] = str(geometry_manifest_path) if geometry_manifest_path else None
    manifest["collision_geometry_export"] = _collision_geometry_export_result(
        geometry_manifest_path,
        limits=_collision_geometry_limits(args),
    )
    render_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _record_progress(
        progress_path,
        "worker_completed",
        collision_geometry_status=manifest["collision_geometry_export"]["status"],
    )


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-json", required=True)
    parser.add_argument("--architecture-contract", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument(
        "--render-engine",
        choices=["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"],
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument("--cycles-device", choices=["CPU", "CUDA", "OPTIX", "AUTO"], default="CPU")
    parser.add_argument("--cycles-samples", type=int, default=16)
    parser.add_argument("--cycles-denoising", action="store_true")
    parser.add_argument(
        "--collision-max-vertices-per-object",
        type=int,
        default=DEFAULT_COLLISION_MAX_VERTICES_PER_OBJECT,
    )
    parser.add_argument(
        "--collision-max-faces-per-object",
        type=int,
        default=DEFAULT_COLLISION_MAX_FACES_PER_OBJECT,
    )
    parser.add_argument(
        "--collision-max-total-vertices",
        type=int,
        default=DEFAULT_COLLISION_MAX_TOTAL_VERTICES,
    )
    parser.add_argument(
        "--collision-max-total-faces",
        type=int,
        default=DEFAULT_COLLISION_MAX_TOTAL_FACES,
    )
    return parser.parse_args(values)


def _record_progress(progress_path: Path, stage: str, **details) -> None:
    record = {"stage": str(stage), **details}
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _collision_geometry_limits(args: argparse.Namespace) -> dict[str, int]:
    return {
        "max_vertices_per_object": max(0, int(args.collision_max_vertices_per_object)),
        "max_faces_per_object": max(0, int(args.collision_max_faces_per_object)),
        "max_total_vertices": max(0, int(args.collision_max_total_vertices)),
        "max_total_faces": max(0, int(args.collision_max_total_faces)),
    }


def _collision_geometry_export_result(path: Path | None, *, limits: dict[str, int]) -> dict:
    if path is None:
        return {"status": "not_applicable", "limits": limits, "summary": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "limits": limits,
            "summary": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    summary = payload.get("export_summary") if isinstance(payload.get("export_summary"), dict) else {}
    incomplete_count = int(summary.get("incomplete_mesh_count") or 0)
    return {
        "status": "partial" if incomplete_count else "completed",
        "limits": limits,
        "summary": summary,
    }


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def _configure_render(
    width: int,
    height: int,
    render_engine: str,
    *,
    cycles_device: str,
    cycles_samples: int,
    cycles_denoising: bool,
) -> dict:
    scene = bpy.context.scene
    requested_render_engine = render_engine
    try:
        scene.render.engine = render_engine
    except TypeError:
        # Blender 5.2 renamed the Eevee enum from BLENDER_EEVEE_NEXT to
        # BLENDER_EEVEE. Keep the public renderer contract stable while
        # recording the runtime enum below.
        if render_engine != "BLENDER_EEVEE_NEXT":
            raise
        scene.render.engine = "BLENDER_EEVEE"
    active_render_engine = str(scene.render.engine)
    scene.render.resolution_x = max(64, int(width))
    scene.render.resolution_y = max(64, int(height))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.055, 0.065, 0.08)
    if render_engine == "BLENDER_WORKBENCH":
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "MATERIAL"
        shading.show_shadows = True
        shading.show_cavity = True
        shading.cavity_type = "WORLD"
        shading.background_type = "VIEWPORT"
        shading.background_color = (0.055, 0.065, 0.08)
    elif render_engine == "CYCLES":
        scene.cycles.samples = max(1, int(cycles_samples))
        scene.cycles.use_denoising = bool(cycles_denoising)
        # Keep the Cycles session, kernels, and scene data alive across the
        # standardized camera renders. Without this, each view pays the full
        # device/kernel initialization cost again.
        scene.render.use_persistent_data = True
        device_config = _configure_cycles_device(cycles_device)
        return {
            "width": int(scene.render.resolution_x),
            "height": int(scene.render.resolution_y),
            "cycles_samples": int(scene.cycles.samples),
            "cycles_denoising": bool(scene.cycles.use_denoising),
            "persistent_data": bool(scene.render.use_persistent_data),
            "render_engine_requested": requested_render_engine,
            "render_engine_active": active_render_engine,
            **device_config,
        }
    return {
        "width": int(scene.render.resolution_x),
        "height": int(scene.render.resolution_y),
        "cycles_samples": None,
        "cycles_denoising": None,
        "persistent_data": bool(scene.render.use_persistent_data),
        "cycles_device_requested": str(cycles_device),
        "cycles_device_active": None,
        "cycles_devices_enabled": [],
        "cycles_device_errors": [],
        "render_engine_requested": requested_render_engine,
        "render_engine_active": active_render_engine,
    }


def _configure_cycles_device(requested_device: str) -> dict:
    requested = str(requested_device).upper()
    if requested == "CPU":
        bpy.context.scene.cycles.device = "CPU"
        return {
            "cycles_device_requested": requested,
            "cycles_device_active": "CPU",
            "cycles_devices_enabled": [],
            "cycles_device_errors": [],
        }

    preferences = bpy.context.preferences.addons["cycles"].preferences
    candidates = ["OPTIX", "CUDA"] if requested == "AUTO" else [requested]
    errors = []
    for backend in candidates:
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            enabled = []
            for device in preferences.devices:
                use_device = str(device.type).upper() == backend
                device.use = use_device
                if use_device:
                    enabled.append({"name": str(device.name), "type": str(device.type)})
            if not enabled:
                raise RuntimeError(f"Blender reported no {backend} devices")
            bpy.context.scene.cycles.device = "GPU"
            return {
                "cycles_device_requested": requested,
                "cycles_device_active": backend,
                "cycles_devices_enabled": enabled,
                "cycles_device_errors": errors,
            }
        except Exception as exc:
            errors.append(f"{backend}: {type(exc).__name__}: {exc}")

    if requested == "AUTO":
        bpy.context.scene.cycles.device = "CPU"
        return {
            "cycles_device_requested": requested,
            "cycles_device_active": "CPU",
            "cycles_devices_enabled": [],
            "cycles_device_errors": errors,
        }
    raise RuntimeError(
        f"Requested Cycles device {requested} is unavailable: {'; '.join(errors)}"
    )


def _boundary(scene_data: dict) -> list[list[float]]:
    boundary = scene_data.get("boundary")
    if not isinstance(boundary, list) or len(boundary) < 3:
        return [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]]
    return [[float(point[0]), float(point[1])] for point in boundary]


def _canonical_object_ids(scene_data: dict) -> list[str]:
    raw_objects = scene_data.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("canonical scene objects must be a JSON list")
    result: list[str] = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            raise ValueError(
                f"canonical scene object {index} must be a JSON object"
            )
        object_id = str(item.get("id") or "").strip()
        if not object_id:
            raise ValueError(
                f"canonical scene object {index} is missing id"
            )
        if object_id in result:
            raise ValueError(
                f"duplicate canonical object id: {object_id}"
            )
        result.append(object_id)
    return result


def _source_architecture_contract() -> dict | None:
    raw = bpy.context.scene.get(ARCHITECTURE_CONTRACT_PROPERTY)
    if raw is None:
        return None
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "source blend contains invalid benchmark architecture provenance"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            "source blend benchmark architecture provenance must be a JSON object"
        )
    return value


def _build_room(
    boundary: list[list[float]],
    height: float,
    *,
    active_wall_ids: list[str] | tuple[str, ...],
) -> list[str]:
    floor_vertices = [(x, y, 0.0) for x, y in boundary]
    floor_mesh = bpy.data.meshes.new("benchmark_floor_mesh")
    floor_mesh.from_pydata(floor_vertices, [], [list(range(len(floor_vertices)))])
    floor = bpy.data.objects.new("benchmark_floor", floor_mesh)
    bpy.context.collection.objects.link(floor)
    floor.data.materials.append(_material("floor", (0.34, 0.36, 0.39, 1.0)))

    active = set(_active_wall_ids({"physical_walls": {"active_wall_ids": active_wall_ids}}))
    thickness = 0.08
    rendered: list[str] = []
    for index, start in enumerate(boundary):
        end = boundary[(index + 1) % len(boundary)]
        wall_id = _wall_id_for_edge(boundary, index)
        if wall_id not in active:
            continue
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        bpy.ops.mesh.primitive_cube_add(
            location=((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, height / 2.0)
        )
        wall = bpy.context.object
        wall.name = f"benchmark_{wall_id}"
        wall["benchmark_architecture_id"] = wall_id
        wall.dimensions = (length, thickness, height)
        wall.rotation_euler[2] = math.atan2(dy, dx)
        wall.data.materials.append(_material(wall_id, (0.60, 0.62, 0.66, 1.0)))
        bpy.context.view_layer.objects.active = wall
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        rendered.append(wall_id)
    return [
        wall_id for wall_id in CANONICAL_WALL_IDS if wall_id in rendered
    ]


def _active_wall_ids(architecture: dict) -> list[str]:
    physical = (
        architecture.get("physical_walls")
        if isinstance(architecture, dict)
        else None
    )
    values = (
        physical.get("active_wall_ids")
        if isinstance(physical, dict)
        else None
    )
    if not isinstance(values, (list, tuple)):
        raise ValueError(
            "architecture.physical_walls.active_wall_ids must be a list"
        )
    result = []
    for value in values:
        wall_id = str(value)
        if wall_id not in CANONICAL_WALL_IDS:
            raise ValueError(f"unknown physical wall ID {wall_id!r}")
        if wall_id not in result:
            result.append(wall_id)
    return [
        wall_id for wall_id in CANONICAL_WALL_IDS if wall_id in result
    ]


def _wall_id_for_edge(
    boundary: list[list[float]],
    index: int,
) -> str:
    midpoints = [
        (
            (point[0] + boundary[(offset + 1) % len(boundary)][0]) / 2.0,
            (point[1] + boundary[(offset + 1) % len(boundary)][1]) / 2.0,
        )
        for offset, point in enumerate(boundary)
    ]
    east_index = max(range(len(midpoints)), key=lambda item: midpoints[item][0])
    west_index = min(range(len(midpoints)), key=lambda item: midpoints[item][0])
    north_index = max(range(len(midpoints)), key=lambda item: midpoints[item][1])
    south_index = min(range(len(midpoints)), key=lambda item: midpoints[item][1])
    by_index = {
        north_index: "north_wall",
        south_index: "south_wall",
        east_index: "east_wall",
        west_index: "west_wall",
    }
    if index not in by_index:
        raise ValueError(
            "physical wall activation requires an axis-aligned rectangular boundary"
        )
    return by_index[index]


def _build_object(
    item: dict,
    asset_root: Path | None,
    *,
    vertical_anchor: str | None = None,
    vertical_anchor_source: str | None = None,
) -> dict:
    object_id = str(item.get("id") or "object")
    target_center = _vec3(item.get("center"), [0.0, 0.0, 0.5])
    target_size = _vec3(item.get("size") or (item.get("asset_proxy") or {}).get("bbox_size"), [1.0, 1.0, 1.0])
    rotation_degrees = _vec3(item.get("rotation"), [0.0, 0.0, 0.0])
    if vertical_anchor is None:
        item_anchor = _vertical_anchor_spec(item, {})
        vertical_anchor = item_anchor["vertical_anchor"]
        vertical_anchor_source = vertical_anchor_source or item_anchor["source"]
    vertical_anchor = _normalize_vertical_anchor(vertical_anchor)
    vertical_anchor_source = str(vertical_anchor_source or "default_bottom")
    mesh_path = _resolve_mesh_path(item, asset_root)
    warning = None
    if mesh_path is not None:
        try:
            fit = _import_and_place(
                mesh_path,
                object_id,
                target_center,
                target_size,
                rotation_degrees,
                vertical_anchor=vertical_anchor,
            )
            return {
                "id": object_id,
                "representation": "asset_mesh",
                "mesh_path": str(mesh_path),
                "warning": None,
                "root_object": f"asset_{object_id}",
                "canonical_center": list(target_center),
                "canonical_size": list(target_size),
                "canonical_rotation_degrees": list(rotation_degrees),
                "vertical_anchor": vertical_anchor,
                "vertical_anchor_source": vertical_anchor_source,
                **fit,
            }
        except Exception as exc:
            warning = f"asset import failed; rendered proxy instead: {type(exc).__name__}: {exc}"
    _add_proxy(object_id, str(item.get("category") or "object"), target_center, target_size, rotation_degrees)
    return {
        "id": object_id,
        "representation": "bbox_proxy",
        "mesh_path": str(mesh_path) if mesh_path else None,
        "warning": warning or "no loadable mesh reference; rendered proxy",
        "fit_mode": FIT_MODE_BBOX_PROXY,
        "canonical_center": list(target_center),
        "canonical_size": list(target_size),
        "canonical_rotation_degrees": list(rotation_degrees),
        "vertical_anchor": vertical_anchor,
        "vertical_anchor_source": vertical_anchor_source,
    }


def _resolve_mesh_path(item: dict, asset_root: Path | None) -> Path | None:
    ref = item.get("asset_ref") if isinstance(item.get("asset_ref"), dict) else {}
    raw_uri = ref.get("mesh_uri")
    candidates = []
    if raw_uri:
        uri = Path(str(raw_uri)).expanduser()
        candidates.append(uri if uri.is_absolute() else (asset_root / uri if asset_root else uri))
    jid = str(item.get("jid") or ref.get("asset_key") or "")
    if asset_root and jid and not jid.startswith("layout_json_proxy:"):
        for extension in (".fbx", ".glb", ".gltf", ".obj"):
            candidates.append(asset_root / jid / f"{jid}{extension}")
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".fbx", ".glb", ".gltf", ".obj"}:
            return candidate.resolve()
    return None


def _import_and_place(
    path: Path,
    object_id: str,
    target_center: list[float],
    target_size: list[float],
    rotation_degrees: list[float],
    *,
    vertical_anchor: str = VERTICAL_ANCHOR_BOTTOM,
) -> dict:
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("import produced no mesh objects")
    minimum, maximum = _world_bounds(meshes)
    source_center = (minimum + maximum) * 0.5
    source_size = maximum - minimum

    # Uniform "contain" fit: one scale factor for all axes so the asset keeps
    # its native aspect ratio and never exceeds the canonical target bbox.
    fit = _uniform_contain_fit(
        [source_size[0], source_size[1], source_size[2]],
        list(target_size),
    )
    uniform_scale = fit["uniform_scale"]

    root = bpy.data.objects.new(f"asset_{object_id}", None)
    root[CANONICAL_ID_PROPERTY] = object_id
    bpy.context.collection.objects.link(root)
    for obj in imported:
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world
        obj[CANONICAL_ID_PROPERTY] = object_id
    rotation = [math.radians(value) for value in rotation_degrees]
    rotation_matrix = Euler(rotation, "XYZ").to_matrix()
    # Same uniform factor on every axis preserves the source aspect ratio.
    root.scale = (uniform_scale, uniform_scale, uniform_scale)
    root.rotation_euler = rotation
    placement = _anchored_root_placement(
        source_center,
        [source_size[0], source_size[1], source_size[2]],
        uniform_scale,
        rotation_matrix,
        target_center,
        target_size,
        vertical_anchor=vertical_anchor,
    )
    root.location = Vector(placement["root_location"])
    return {**fit, **placement}


def _uniform_contain_fit(source_size, target_size) -> dict:
    """Uniform fit-inside scale plus finite/positive validation.

    ``uniform_scale = min(target_size[i] / source_size[i])`` so the source mesh
    is scaled uniformly to fit completely inside the target bbox without
    exceeding any target dimension. ``rendered_size`` is the resulting scaled
    bounding box; it preserves the source aspect ratio and never exceeds
    ``target_size`` on any axis.
    """

    source = _require_finite_positive_vec3(source_size, "source_size")
    target = _require_finite_positive_vec3(target_size, "target_size")
    ratios = [target[index] / source[index] for index in range(3)]
    uniform_scale = min(ratios)
    if not math.isfinite(uniform_scale) or uniform_scale <= 0.0:
        raise ValueError(f"uniform_scale must be finite and positive, got {uniform_scale!r}")
    rendered_size = [source[index] * uniform_scale for index in range(3)]
    return {
        "fit_mode": FIT_MODE_UNIFORM_CONTAIN,
        "source_size": source,
        "target_size": target,
        "rendered_size": rendered_size,
        "uniform_scale": float(uniform_scale),
    }


def _root_location(source_center, uniform_scale: float, rotation_matrix, target_center) -> list[float]:
    """Location that centers the uniformly scaled source bbox at target_center.

    Accepts either plain sequences or mathutils Vector/Matrix (both support
    ``[i]`` / ``[r][c]`` indexing), so the exact runtime formula is unit-testable
    without Blender.
    """

    scaled_center = [uniform_scale * float(source_center[index]) for index in range(3)]
    rotated = _matvec3(rotation_matrix, scaled_center)
    return [float(target_center[index]) - rotated[index] for index in range(3)]


def _anchored_root_placement(
    source_center,
    source_size,
    uniform_scale: float,
    rotation_matrix,
    target_center,
    target_size,
    *,
    vertical_anchor: str = VERTICAL_ANCHOR_BOTTOM,
) -> dict:
    """Place a contain-fit mesh at a stable local vertical anchor.

    The offset is applied in the canonical object's local frame before its
    rotation. This preserves the contain-fit invariant even for non-yaw
    rotations. ``bottom`` makes the mesh and canonical local lower planes
    coincide; ``top`` does the same for explicit ceiling attachments.
    """

    anchor = _normalize_vertical_anchor(vertical_anchor)
    source_dimensions = _require_finite_positive_vec3(source_size, "source_size")
    target_dimensions = _require_finite_positive_vec3(target_size, "target_size")
    source_bounds_center = _require_finite_vec3(source_center, "source_center")
    canonical_center = _require_finite_vec3(target_center, "target_center")
    scale = float(uniform_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"uniform_scale must be finite and positive, got {uniform_scale!r}")

    rendered_height = source_dimensions[2] * scale
    slack = target_dimensions[2] - rendered_height
    if slack < -1.0e-9:
        raise ValueError(
            "anchored contain-fit rendered height exceeds target height: "
            f"{rendered_height!r}>{target_dimensions[2]!r}"
        )
    local_offset_z = -max(0.0, slack) * 0.5 if anchor == VERTICAL_ANCHOR_BOTTOM else max(0.0, slack) * 0.5
    local_anchor_offset = [0.0, 0.0, local_offset_z]
    world_anchor_offset = _matvec3(rotation_matrix, local_anchor_offset)
    rendered_bounds_center = [
        canonical_center[index] + world_anchor_offset[index]
        for index in range(3)
    ]
    root_location = _root_location(
        source_bounds_center,
        scale,
        rotation_matrix,
        rendered_bounds_center,
    )
    return {
        "vertical_anchor": anchor,
        "local_anchor_offset": local_anchor_offset,
        "rendered_bounds_center": rendered_bounds_center,
        "root_location": root_location,
    }


def _normalize_vertical_anchor(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {VERTICAL_ANCHOR_BOTTOM, "floor"}:
        return VERTICAL_ANCHOR_BOTTOM
    if normalized in {VERTICAL_ANCHOR_TOP, "ceiling"}:
        return VERTICAL_ANCHOR_TOP
    raise ValueError(f"vertical_anchor must be bottom or top, got {value!r}")


def _vertical_anchor_spec(item: dict, scene_data: dict) -> dict[str, str]:
    """Resolve only explicit ceiling-attachment signals; otherwise use bottom.

    Categories and free-form descriptions are intentionally ignored. A ceiling
    anchor changes rendered geometry, so it must come from a structured relation
    or explicit object metadata rather than a semantic guess.
    """

    object_id = str(item.get("id") or "")
    support_parent = str(item.get("support_parent") or "").strip().lower()
    if support_parent == "ceiling":
        return {"vertical_anchor": VERTICAL_ANCHOR_TOP, "source": "object.support_parent"}

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    metadata_anchor = metadata.get("vertical_anchor") or metadata.get("render_vertical_anchor")
    if str(metadata_anchor or "").strip().lower() in {"top", "ceiling"}:
        return {"vertical_anchor": VERTICAL_ANCHOR_TOP, "source": "object.metadata.vertical_anchor"}

    placement = item.get("placement_intent") if isinstance(item.get("placement_intent"), dict) else {}
    for raw in placement.get("absolute_relations", []) if isinstance(placement.get("absolute_relations"), list) else []:
        if _is_explicit_ceiling_attachment(raw, object_id=object_id):
            return {"vertical_anchor": VERTICAL_ANCHOR_TOP, "source": "object.placement_intent"}

    for key in ("oar_relations", "relationships", "relations"):
        relations = scene_data.get(key) if isinstance(scene_data, dict) else None
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if _is_explicit_ceiling_attachment(relation, object_id=object_id):
                return {"vertical_anchor": VERTICAL_ANCHOR_TOP, "source": f"scene.{key}"}
    return {"vertical_anchor": VERTICAL_ANCHOR_BOTTOM, "source": "default_bottom"}


def _is_explicit_ceiling_attachment(value, *, object_id: str) -> bool:
    if isinstance(value, str):
        normalized = _normalized_relation_token(value)
        return normalized in {
            "attached_to_ceiling",
            "hung_from_ceiling",
            "hang_from_ceiling",
        }
    if not isinstance(value, dict):
        return False
    subject = value.get("subject_id") or value.get("subject") or value.get("object_id")
    if subject is not None and str(subject) != object_id:
        return False
    predicate = _normalized_relation_token(
        value.get("type") or value.get("predicate") or value.get("relation")
    )
    if predicate in {"attached_to_ceiling", "hung_from_ceiling", "hang_from_ceiling"}:
        return True
    target = _normalized_relation_token(
        value.get("architectural_element")
        or value.get("target")
        or value.get("object")
        or value.get("anchor_id")
    )
    return predicate in {"attached_to", "hung_from", "hang_from", "mounted_on"} and target == "ceiling"


def _normalized_relation_token(value: object) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _matvec3(matrix, vector) -> list[float]:
    return [
        float(matrix[row][0]) * float(vector[0])
        + float(matrix[row][1]) * float(vector[1])
        + float(matrix[row][2]) * float(vector[2])
        for row in range(3)
    ]


def _require_finite_positive_vec3(values, label: str) -> list[float]:
    result = _coerce_vec3(values, label)
    for index, number in enumerate(result):
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{label}[{index}] must be finite and positive, got {number!r}")
    return result


def _require_finite_vec3(values, label: str) -> list[float]:
    result = _coerce_vec3(values, label)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values, got {values!r}")
    return result


def _coerce_vec3(values, label: str) -> list[float]:
    """Accept ordinary sequences and Blender/mathutils vector-like values."""

    if isinstance(values, (str, bytes, dict)):
        raise ValueError(f"{label} must be a 3-vector, got {values!r}")
    try:
        if len(values) < 3:
            raise ValueError(f"{label} must be a 3-vector, got {values!r}")
        return [float(values[index]) for index in range(3)]
    except (TypeError, IndexError, KeyError) as exc:
        raise ValueError(f"{label} must be a 3-vector, got {values!r}") from exc


def _world_bounds(objects: list) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    return (
        Vector([min(point[index] for point in points) for index in range(3)]),
        Vector([max(point[index] for point in points) for index in range(3)]),
    )


def _add_proxy(object_id: str, category: str, center: list[float], size: list[float], rotation_degrees: list[float]) -> None:
    bpy.ops.mesh.primitive_cube_add(location=center)
    proxy = bpy.context.object
    proxy.name = f"proxy_{object_id}"
    proxy[CANONICAL_ID_PROPERTY] = object_id
    proxy.dimensions = size
    proxy.rotation_euler = [math.radians(value) for value in rotation_degrees]
    proxy.data.materials.append(_material(f"proxy_{category}", _category_color(category)))
    bpy.context.view_layer.objects.active = proxy
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    material.roughness = 0.65
    material.use_nodes = True
    principled = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = 0.65
    return material


def _category_color(category: str) -> tuple[float, float, float, float]:
    digest = hashlib.sha256(category.encode("utf-8")).digest()
    return tuple(0.25 + value / 255.0 * 0.55 for value in digest[:3]) + (1.0,)


def _add_lighting(boundary: list[list[float]], room_height: float) -> None:
    center, span = _room_center_span(boundary)
    area_data = bpy.data.lights.new("benchmark_area", type="AREA")
    area_data.energy = 1400.0
    area_data.shape = "DISK"
    area_data.size = max(3.0, span)
    area = bpy.data.objects.new("benchmark_area", area_data)
    area.location = (center[0], center[1], room_height + max(1.0, span * 0.25))
    area.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.collection.objects.link(area)

    sun_data = bpy.data.lights.new("benchmark_sun", type="SUN")
    sun_data.energy = 2.0
    sun = bpy.data.objects.new("benchmark_sun", sun_data)
    sun.rotation_euler = (math.radians(25), math.radians(-20), math.radians(25))
    bpy.context.collection.objects.link(sun)


def _ensure_camera_evidence_lighting(camera_views: list[dict] | None = None) -> dict:
    """Provide the same ephemeral benchmark lighting used by global renders.

    Trusted prepared blends intentionally contain no persistent cameras or
    lights. Read-only camera workers therefore need to recreate the benchmark
    lighting in memory before rendering; otherwise Eevee/Cycles evidence is
    illuminated only by the sanitized world and is severely underexposed.
    Existing render-enabled lights are preserved to avoid double-lighting
    legacy/non-prepared scenes.
    """

    existing = sorted(
        obj.name
        for obj in bpy.data.objects
        if obj.type == "LIGHT" and not obj.hide_render
    )
    if existing:
        return {
            "policy": "preserve_existing_scene_lighting_v1",
            "geometry_source": "existing_scene_lights",
            "existing_light_names": existing,
            "added_light_names": [],
            "ephemeral": False,
        }

    boundary, room_height, geometry_source = _camera_evidence_lighting_geometry(
        camera_views or []
    )
    before = {obj.name for obj in bpy.data.objects if obj.type == "LIGHT"}
    _add_lighting(boundary, room_height)
    added = sorted(
        obj.name
        for obj in bpy.data.objects
        if obj.type == "LIGHT" and obj.name not in before
    )
    return {
        "policy": "ephemeral_benchmark_lighting_v1",
        "geometry_source": geometry_source,
        "existing_light_names": [],
        "added_light_names": added,
        "ephemeral": True,
        "boundary": boundary,
        "room_height": room_height,
    }


def _camera_evidence_lighting_geometry(
    camera_views: list[dict],
) -> tuple[list[list[float]], float, str]:
    scene = bpy.context.scene
    raw_boundary = scene.get("benchmark_request_boundary")
    raw_height = scene.get("benchmark_scene_height")
    try:
        boundary = json.loads(str(raw_boundary))
        room_height = float(raw_height)
    except (TypeError, ValueError, json.JSONDecodeError):
        boundary = None
        room_height = 0.0
    if (
        isinstance(boundary, list)
        and len(boundary) >= 3
        and all(
            isinstance(point, list)
            and len(point) == 2
            and all(
                isinstance(component, (int, float))
                and math.isfinite(float(component))
                for component in point
            )
            for point in boundary
        )
        and math.isfinite(room_height)
        and room_height > 0.0
    ):
        return (
            [[float(component) for component in point] for point in boundary],
            room_height,
            "scene_provenance",
        )

    for pose in camera_views:
        if not isinstance(pose, dict):
            continue
        room_bounds = pose.get("room_bounds")
        if (
            not isinstance(room_bounds, list)
            or len(room_bounds) != 6
        ):
            continue
        try:
            xmin, xmax, ymin, ymax, zmin, zmax = (
                float(value) for value in room_bounds
            )
        except (TypeError, ValueError):
            continue
        if not all(
            math.isfinite(value)
            for value in (xmin, xmax, ymin, ymax, zmin, zmax)
        ):
            continue
        if xmax <= xmin or ymax <= ymin or zmax <= zmin:
            continue
        return (
            [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ],
            zmax,
            "camera_room_bounds",
        )

    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render
    ]
    if meshes:
        minimum, maximum = _world_bounds(meshes)
        xmin, ymin = float(minimum.x), float(minimum.y)
        xmax, ymax = float(maximum.x), float(maximum.y)
        if xmax > xmin and ymax > ymin:
            return (
                [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ],
                max(2.8, float(maximum.z)),
                "renderable_mesh_bounds_fallback",
            )

    return (
        [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
        2.8,
        "default_room_fallback",
    )


def _render_views(
    boundary: list[list[float]],
    room_height: float,
    out_dir: Path,
    *,
    progress_path: Path | None = None,
) -> list[dict]:
    center, span = _room_center_span(boundary)
    target = Vector((center[0], center[1], min(room_height * 0.4, 1.2)))
    definitions = [
        {
            "name": "top",
            "location": Vector((center[0], center[1], room_height + max(5.0, span * 1.15))),
            "camera_type": "ORTHO",
            "ortho_scale": span * 1.15,
        },
        {
            "name": "perspective",
            "location": Vector((center[0] + span * 0.85, center[1] - span * 0.95, room_height + span * 0.70)),
            "camera_type": "PERSP",
            "ortho_scale": None,
        },
    ]
    results = []
    for definition in definitions:
        camera_data = bpy.data.cameras.new(f"camera_{definition['name']}")
        camera_data.type = definition["camera_type"]
        camera_data.lens = 48.0
        camera_data.sensor_width = 36.0
        camera_data.sensor_fit = "HORIZONTAL"
        if definition["ortho_scale"] is not None:
            camera_data.ortho_scale = definition["ortho_scale"]
        camera = bpy.data.objects.new(f"camera_{definition['name']}", camera_data)
        camera.location = definition["location"]
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(camera)
        bpy.context.scene.camera = camera
        render_path = out_dir / f"standardized_{definition['name']}.png"
        bpy.context.scene.render.filepath = str(render_path)
        if progress_path is not None:
            _record_progress(
                progress_path,
                "view_render_started",
                view=definition["name"],
                path=str(render_path),
            )
        started_at = time.monotonic()
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            if progress_path is not None:
                _record_progress(
                    progress_path,
                    "view_render_failed",
                    view=definition["name"],
                    path=str(render_path),
                    elapsed_seconds=time.monotonic() - started_at,
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        elapsed_seconds = time.monotonic() - started_at
        pixel_stats = _render_pixel_stats(render_path)
        if progress_path is not None:
            _record_progress(
                progress_path,
                "view_render_completed",
                view=definition["name"],
                path=str(render_path),
                elapsed_seconds=elapsed_seconds,
            )
        results.append(
            {
                "name": definition["name"],
                "path": str(render_path),
                "camera_location": list(camera.location),
                "camera_target": list(target),
                "elapsed_seconds": elapsed_seconds,
                "pixel_stats": pixel_stats,
            }
        )
    return results


def _render_identity_map(
    canonical_object_ids: list[str],
    out_dir: Path,
    *,
    progress_path: Path | None = None,
) -> tuple[dict, dict[str, str], dict[str, str]]:
    """Render a read-only multiclass object-identity pass.

    WORKBENCH object colors are temporary display state.  They are restored
    before the canonical ``scene.blend`` is saved.
    """

    scene = bpy.context.scene
    camera = bpy.data.objects.get("camera_perspective")
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError(
            "identity render requires standardized perspective camera"
        )
    expected = set(canonical_object_ids)
    meshes_by_id: dict[str, list] = {
        object_id: [] for object_id in canonical_object_ids
    }
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        raw_id = obj.get(CANONICAL_ID_PROPERTY)
        if raw_id is None:
            continue
        object_id = str(raw_id).strip()
        if object_id not in expected:
            raise ValueError(
                "renderable mesh references unknown canonical object id: "
                f"{object_id}"
            )
        meshes_by_id[object_id].append(obj)
    missing = [
        object_id
        for object_id, meshes in meshes_by_id.items()
        if not meshes
    ]
    if missing:
        raise ValueError(
            "identity render has no renderable mesh for canonical IDs: "
            + ", ".join(missing)
        )

    colors = _identity_colors(canonical_object_ids)
    legend = {
        _rgba_hex(colors[object_id]): object_id
        for object_id in canonical_object_ids
    }
    palette = {
        object_id: _rgba_hex(colors[object_id])
        for object_id in canonical_object_ids
    }
    if len(legend) != len(canonical_object_ids):
        raise ValueError("identity colors must be unique")

    previous_camera = scene.camera
    previous_filepath = scene.render.filepath
    previous_engine = scene.render.engine
    previous_film_transparent = scene.render.film_transparent
    previous_colors = {
        obj.name: tuple(obj.color)
        for obj in scene.objects
        if hasattr(obj, "color")
    }
    view_settings = {
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
    }
    display = scene.display.shading
    display_settings = {
        "light": display.light,
        "color_type": display.color_type,
        "show_shadows": display.show_shadows,
        "show_cavity": display.show_cavity,
        "show_specular_highlight": display.show_specular_highlight,
        "background_type": display.background_type,
        "background_color": tuple(display.background_color),
    }
    render_path = out_dir / "standardized_identity_map.png"
    started_at = time.monotonic()
    try:
        for obj in scene.objects:
            if obj.type != "MESH":
                continue
            raw_id = obj.get(CANONICAL_ID_PROPERTY)
            if raw_id is None:
                obj.color = (0.16, 0.16, 0.16, 1.0)
            else:
                obj.color = colors[str(raw_id)]
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.film_transparent = False
        # Raw/no-look preserves the assigned identity values in the saved PNG.
        # Standard/AgX display transforms alter the bytes and would make a
        # hex-color legend factually incorrect.
        scene.view_settings.view_transform = "Raw"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        display.light = "FLAT"
        display.color_type = "OBJECT"
        display.show_shadows = False
        display.show_cavity = False
        display.show_specular_highlight = False
        display.background_type = "VIEWPORT"
        display.background_color = (0.05, 0.05, 0.05)
        scene.camera = camera
        scene.render.filepath = str(render_path)
        if progress_path is not None:
            _record_progress(
                progress_path,
                "view_render_started",
                view="identity_map",
                path=str(render_path),
            )
        bpy.ops.render.render(write_still=True)
        pixel_stats = _render_pixel_stats(render_path)
    finally:
        for obj in scene.objects:
            if obj.name in previous_colors:
                obj.color = previous_colors[obj.name]
        scene.camera = previous_camera
        scene.render.filepath = previous_filepath
        scene.render.engine = previous_engine
        scene.render.film_transparent = previous_film_transparent
        scene.view_settings.view_transform = view_settings[
            "view_transform"
        ]
        scene.view_settings.look = view_settings["look"]
        scene.view_settings.exposure = view_settings["exposure"]
        scene.view_settings.gamma = view_settings["gamma"]
        for key, value in display_settings.items():
            setattr(display, key, value)
    elapsed_seconds = time.monotonic() - started_at
    if progress_path is not None:
        _record_progress(
            progress_path,
            "view_render_completed",
            view="identity_map",
            path=str(render_path),
            elapsed_seconds=elapsed_seconds,
        )
    return (
        {
            "name": "identity_map",
            "path": str(render_path),
            "camera_location": list(camera.location),
            "camera_target": list(
                bpy.data.objects["camera_perspective"].location
                + bpy.data.objects["camera_perspective"].matrix_world.to_quaternion()
                @ Vector((0.0, 0.0, -1.0))
            ),
            "elapsed_seconds": elapsed_seconds,
            "pixel_stats": pixel_stats,
            "role": "global_identity_overlay",
            "representation": "identity_map",
            "color_encoding": "raw_linear_rgb_8bit",
            "identity_legend": legend,
        },
        legend,
        palette,
    )


def _identity_colors(
    canonical_object_ids: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    result: dict[str, tuple[float, float, float, float]] = {}
    count = max(1, len(canonical_object_ids))
    for index, object_id in enumerate(canonical_object_ids):
        hue = (index * 0.6180339887498949) % 1.0
        result[object_id] = (*_hsv_to_rgb(hue, 0.72, 0.92), 1.0)
    return result


def _hsv_to_rgb(
    hue: float,
    saturation: float,
    value: float,
) -> tuple[float, float, float]:
    sector = int(hue * 6.0)
    fraction = hue * 6.0 - sector
    low = value * (1.0 - saturation)
    descending = value * (1.0 - fraction * saturation)
    ascending = value * (1.0 - (1.0 - fraction) * saturation)
    return (
        (
            (value, ascending, low),
            (descending, value, low),
            (low, value, ascending),
            (low, descending, value),
            (ascending, low, value),
            (value, low, descending),
        )[sector % 6]
    )


def _rgba_hex(value: tuple[float, float, float, float]) -> str:
    return "#" + "".join(
        f"{max(0, min(255, round(component * 255.0))):02X}"
        for component in value[:3]
    )


def _render_pixel_stats(render_path: Path) -> dict:
    # Render Result can expose an empty/stale pixel buffer after write_still in
    # headless Cycles. Reload the saved PNG so validation covers the exact
    # evidence file sent to the VLM.
    image = bpy.data.images.load(str(render_path), check_existing=False)
    try:
        if not image.has_data:
            return {"sample_count": 0, "min_luminance": 0.0, "max_luminance": 0.0, "luminance_range": 0.0}
        width, height = int(image.size[0]), int(image.size[1])
        channels = max(1, int(image.channels))
        pixel_count = max(0, width * height)
        if pixel_count == 0:
            return {"sample_count": 0, "min_luminance": 0.0, "max_luminance": 0.0, "luminance_range": 0.0}
        step = max(1, pixel_count // 4096)
        pixels = image.pixels
        luminances = []
        for pixel_index in range(0, pixel_count, step):
            offset = pixel_index * channels
            red = float(pixels[offset])
            green = float(pixels[offset + 1]) if channels > 1 else red
            blue = float(pixels[offset + 2]) if channels > 2 else red
            luminances.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)
        minimum = min(luminances)
        maximum = max(luminances)
        return {
            "sample_count": len(luminances),
            "min_luminance": minimum,
            "max_luminance": maximum,
            "luminance_range": maximum - minimum,
            "mean_luminance": sum(luminances) / len(luminances),
        }
    finally:
        bpy.data.images.remove(image)


def _room_center_span(boundary: list[list[float]]) -> tuple[tuple[float, float], float]:
    xs = [point[0] for point in boundary]
    ys = [point[1] for point in boundary]
    width, depth = max(xs) - min(xs), max(ys) - min(ys)
    return ((min(xs) + width / 2.0, min(ys) + depth / 2.0), max(width, depth, 1.0))


def _write_collision_geometry_manifest(
    out_dir: Path,
    object_results: list[dict],
    *,
    max_vertices_per_object: int = DEFAULT_COLLISION_MAX_VERTICES_PER_OBJECT,
    max_faces_per_object: int = DEFAULT_COLLISION_MAX_FACES_PER_OBJECT,
    max_total_vertices: int = DEFAULT_COLLISION_MAX_TOTAL_VERTICES,
    max_total_faces: int = DEFAULT_COLLISION_MAX_TOTAL_FACES,
) -> Path | None:
    geometry_dir = out_dir / "collision_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    _refresh_scene_graph()
    entries: dict[str, dict] = {}
    exported_vertices = 0
    exported_faces = 0
    skipped_complexity = 0
    for result in object_results:
        if not isinstance(result, dict):
            continue
        object_id = str(result.get("id") or "")
        if result.get("representation") != "asset_mesh":
            continue
        root_name = str(result.get("root_object") or f"asset_{object_id}")
        geometry_source = _asset_geometry_source(result.get("mesh_path"))
        root = bpy.data.objects.get(root_name)
        if root is None:
            entries[object_id] = {
                "representation": "bbox_proxy",
                "complete": False,
                "error": f"root object {root_name!r} not found for geometry extraction",
            }
            continue
        geometry_path = geometry_dir / f"{object_id}.ply"
        vertex_count, face_count = _mesh_complexity(root)
        complexity_error = _complexity_limit_error(
            vertex_count=vertex_count,
            face_count=face_count,
            exported_vertices=exported_vertices,
            exported_faces=exported_faces,
            max_vertices_per_object=max_vertices_per_object,
            max_faces_per_object=max_faces_per_object,
            max_total_vertices=max_total_vertices,
            max_total_faces=max_total_faces,
        )
        if complexity_error is not None:
            skipped_complexity += 1
            entries[object_id] = {
                "representation": "triangle_mesh",
                "geometry_path": str(geometry_path),
                "transform_baked": True,
                "geometry_source": geometry_source,
                "source_uri": result.get("mesh_path"),
                "complete": False,
                "vertex_count": vertex_count,
                "face_count": face_count,
                "error": complexity_error,
            }
            continue
        try:
            written_vertices, written_faces, world_bounds = _export_world_triangles(root, geometry_path)
            exported_vertices += written_vertices
            exported_faces += written_faces
            frame_validation = _exported_bounds_frame_validation(
                world_bounds,
                canonical_center=result.get("canonical_center"),
                expected_bounds_center=result.get("rendered_bounds_center"),
                center_tolerance_m=COLLISION_GEOMETRY_CENTER_TOLERANCE_M,
            )
            frame_consistent = bool(frame_validation["canonical_consistent"])
            entries[object_id] = {
                "representation": "triangle_mesh",
                "geometry_path": str(geometry_path),
                "transform_baked": True,
                "geometry_source": geometry_source,
                "source_uri": result.get("mesh_path"),
                "complete": frame_consistent,
                "vertex_count": written_vertices,
                "face_count": written_faces,
                "world_aabb": world_bounds,
                "canonical_center": result.get("canonical_center"),
                "canonical_size": result.get("canonical_size"),
                "canonical_rotation_degrees": result.get("canonical_rotation_degrees"),
                "vertical_anchor": result.get("vertical_anchor"),
                "vertical_anchor_source": result.get("vertical_anchor_source"),
                "rendered_bounds_center": result.get("rendered_bounds_center"),
                "frame_validation": frame_validation,
                **(
                    {}
                    if frame_consistent
                    else {"error": "canonical_frame_mismatch: " + ", ".join(frame_validation["failure_reasons"])}
                ),
            }
        except Exception as exc:
            entries[object_id] = {
                "representation": "triangle_mesh",
                "geometry_path": str(geometry_path),
                "transform_baked": True,
                "geometry_source": geometry_source,
                "source_uri": result.get("mesh_path"),
                "complete": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if not entries:
        return None
    manifest = {
        "schema_version": "collision_geometry_v1",
        "units": "meter",
        "up_axis": "z",
        "objects": entries,
        "export_summary": {
            "asset_mesh_count": len(entries),
            "complete_mesh_count": sum(entry.get("complete") is True for entry in entries.values()),
            "incomplete_mesh_count": sum(entry.get("complete") is not True for entry in entries.values()),
            "complexity_skipped_count": skipped_complexity,
            "exported_vertex_count": exported_vertices,
            "exported_face_count": exported_faces,
            "limits": {
                "max_vertices_per_object": max(0, int(max_vertices_per_object)),
                "max_faces_per_object": max(0, int(max_faces_per_object)),
                "max_total_vertices": max(0, int(max_total_vertices)),
                "max_total_faces": max(0, int(max_total_faces)),
            },
        },
    }
    manifest_path = out_dir / "collision_geometry_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _asset_geometry_source(mesh_path: object) -> str:
    """Describe imported metric geometry without assuming every asset is FBX."""

    suffix = Path(str(mesh_path or "")).suffix.lower().lstrip(".")
    if suffix in {"fbx", "glb", "gltf", "obj"}:
        return f"asset_{suffix}"
    return "asset_mesh"


def _export_world_triangles(root, geometry_path: Path) -> tuple[int, int, dict[str, list[float]]]:
    mesh_objects = _mesh_objects(root)
    vertex_count, face_count = _mesh_complexity(root)
    if vertex_count <= 0 or face_count <= 0:
        raise RuntimeError("no world-space triangle geometry extracted")
    bounds_min = [math.inf, math.inf, math.inf]
    bounds_max = [-math.inf, -math.inf, -math.inf]
    with geometry_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    f"element vertex {vertex_count}",
                    "property float x",
                    "property float y",
                    "property float z",
                    f"element face {face_count}",
                    "property list uchar int vertex_indices",
                    "end_header",
                ]
            )
            + "\n"
        )
        for obj in mesh_objects:
            matrix = obj.matrix_world
            for vertex in obj.data.vertices:
                world = matrix @ vertex.co
                for axis in range(3):
                    bounds_min[axis] = min(bounds_min[axis], float(world[axis]))
                    bounds_max[axis] = max(bounds_max[axis], float(world[axis]))
                handle.write(f"{float(world[0]):.6f} {float(world[1]):.6f} {float(world[2]):.6f}\n")
        offset = 0
        for obj in mesh_objects:
            for polygon in obj.data.polygons:
                indices = [int(index) + offset for index in polygon.vertices]
                for tri_start in range(1, len(indices) - 1):
                    handle.write(f"3 {indices[0]} {indices[tri_start]} {indices[tri_start + 1]}\n")
            offset += len(obj.data.vertices)
    return vertex_count, face_count, {
        "min": [float(value) for value in bounds_min],
        "max": [float(value) for value in bounds_max],
    }


def _refresh_scene_graph() -> None:
    """Materialize parent transforms before reading child ``matrix_world``.

    Without this update, the final imported asset can retain a stale child
    matrix during manual PLY export even though Blender renders it correctly.
    """

    if bpy is not None:
        bpy.context.view_layer.update()


def _exported_bounds_frame_validation(
    world_bounds: dict[str, list[float]],
    *,
    canonical_center,
    expected_bounds_center=None,
    center_tolerance_m: float,
) -> dict:
    minimum = _require_finite_vec3(world_bounds.get("min"), "world_bounds.min")
    maximum = _require_finite_vec3(world_bounds.get("max"), "world_bounds.max")
    canonical = _require_finite_vec3(canonical_center, "canonical_center")
    expected = _require_finite_vec3(
        expected_bounds_center if expected_bounds_center is not None else canonical_center,
        "expected_bounds_center",
    )
    actual = [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)]
    offset = math.sqrt(sum((actual[axis] - expected[axis]) ** 2 for axis in range(3)))
    canonical_offset = math.sqrt(sum((actual[axis] - canonical[axis]) ** 2 for axis in range(3)))
    reasons = []
    if offset > float(center_tolerance_m):
        reasons.append(f"world_bounds_center_offset_m={offset:.6f}>{float(center_tolerance_m):.6f}")
    return {
        "canonical_consistent": not reasons,
        "failure_reasons": reasons,
        "world_bounds_center": actual,
        "canonical_center": canonical,
        "expected_world_bounds_center": expected,
        "world_bounds_center_offset_m": offset,
        "canonical_center_offset_m": canonical_offset,
        "center_tolerance_m": float(center_tolerance_m),
    }


def _mesh_objects(root) -> list:
    objects = [child for child in root.children_recursive if child.type == "MESH" and child.data is not None]
    if not objects and root.type == "MESH" and root.data is not None:
        objects = [root]
    return objects


def _mesh_complexity(root) -> tuple[int, int]:
    vertex_count = 0
    triangle_count = 0
    for obj in _mesh_objects(root):
        vertex_count += len(obj.data.vertices)
        triangle_count += sum(max(0, len(polygon.vertices) - 2) for polygon in obj.data.polygons)
    return vertex_count, triangle_count


def _complexity_limit_error(
    *,
    vertex_count: int,
    face_count: int,
    exported_vertices: int,
    exported_faces: int,
    max_vertices_per_object: int,
    max_faces_per_object: int,
    max_total_vertices: int,
    max_total_faces: int,
) -> str | None:
    checks = (
        (vertex_count, max_vertices_per_object, "vertices_per_object"),
        (face_count, max_faces_per_object, "faces_per_object"),
        (exported_vertices + vertex_count, max_total_vertices, "total_vertices"),
        (exported_faces + face_count, max_total_faces, "total_faces"),
    )
    exceeded = [f"{name}={actual}>{limit}" for actual, limit, name in checks if limit >= 0 and actual > limit]
    if not exceeded:
        return None
    return "complexity_limit_exceeded: " + ", ".join(exceeded)


def _vec3(value, default: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return list(default)


if __name__ == "__main__":
    main()
