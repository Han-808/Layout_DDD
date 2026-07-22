"""Blender-side worker for read-only metric focus overlays.

The source ``.blend`` is opened by the parent Blender process with
auto-execution disabled. This worker highlights metric targets, dims other
objects, draws target OBBs and optional architecture planes/markers, renders the
requested poses with engine-independent object colors, and never saves the
source scene.

It intentionally uses only the standard library plus Blender's bundled
``bpy``/``mathutils`` and the sibling ``blender_worker`` helper, so it must not
import the ``benchmark`` package.
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

from blender_worker import CANONICAL_ID_PROPERTY, _configure_render

_BLENDER_SUFFIX = re.compile(r"\.\d{3}$")


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
        args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
    )
    degradations = _apply_overlay(overlay_spec)
    role = str(overlay_spec.get("role") or "collision_pair_overlay")
    views = [_render_pose(pose, out_dir, index, role=role) for index, pose in enumerate(poses)]
    targets = _overlay_targets(overlay_spec)
    manifest = {
        "backend": "blender_read_only_focus_overlay_v1",
        "source_blend": str(Path(bpy.data.filepath).resolve()) if bpy.data.filepath else None,
        "source_scene_saved": False,
        "scene_mutation_scope": "ephemeral_overlay_camera_material_wireframe_marker_only",
        "render_engine": args.render_engine,
        "render_config": render_config,
        "role": role,
        "metric": overlay_spec.get("metric"),
        "target_ids": [target.get("id") for target in targets],
        "legend": overlay_spec.get("legend"),
        "representation_level": overlay_spec.get("representation_level"),
        "object_a_id": (overlay_spec.get("object_a") or {}).get("id"),
        "object_b_id": (overlay_spec.get("object_b") or {}).get("id"),
        "diagnostic_degradations": degradations,
        "views": views,
    }
    (out_dir / "collision_overlay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-views-json", required=True)
    parser.add_argument("--overlay-spec-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--render-engine",
        choices=["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"],
        default="BLENDER_WORKBENCH",
    )
    parser.add_argument("--cycles-device", choices=["CPU", "CUDA", "OPTIX", "AUTO"], default="CPU")
    parser.add_argument("--cycles-samples", type=int, default=1)
    parser.add_argument("--cycles-denoising", action="store_true")
    return parser.parse_args(values)


def _apply_overlay(spec: dict) -> list[str]:
    """Recolor targets, dim context, and draw wireframes/markers. Read-only:
    only in-memory object colors and ephemeral helper objects are touched."""

    degradations: list[str] = []
    scene = bpy.context.scene
    try:
        scene.display.shading.color_type = "OBJECT"
    except Exception as exc:  # pragma: no cover - depends on Blender build
        degradations.append(f"object_color_shading_unavailable: {type(exc).__name__}: {exc}")

    resolver = _CanonicalResolver()
    targets = _overlay_targets(spec)
    target_colors = {
        str(target.get("id")): _rgba(target.get("color"), (1.0, 0.12, 0.12))
        for target in targets
        if target.get("id") is not None
    }
    color_context = _rgba((spec.get("colors") or {}).get("context"), (0.34, 0.34, 0.38))

    for obj in list(bpy.data.objects):
        canonical = resolver.resolve(obj)
        if canonical in target_colors:
            _set_color(obj, target_colors[canonical])
        elif obj.type == "MESH":
            _set_color(obj, color_context)

    for target in targets:
        obb = target.get("obb") or {}
        try:
            _draw_wireframe(f"overlay_wire_{target.get('id')}", obb.get("corners"), obb.get("edges"), _rgba(target.get("color"), (1.0, 1.0, 1.0)))
        except Exception as exc:  # pragma: no cover - drawing is best-effort
            degradations.append(f"wireframe_failed_{target.get('id')}: {type(exc).__name__}: {exc}")

    marker_color = _rgba((spec.get("colors") or {}).get("marker"), (1.0, 0.94, 0.10))
    for index, marker in enumerate(spec.get("markers") or []):
        try:
            _draw_marker(f"overlay_marker_{index}", marker.get("position"), marker_color)
        except Exception as exc:  # pragma: no cover
            degradations.append(f"marker_failed_{index}: {type(exc).__name__}: {exc}")
    for index, connector in enumerate(spec.get("connectors") or []):
        try:
            _draw_line(f"overlay_connector_{index}", connector.get("from"), connector.get("to"), marker_color)
        except Exception as exc:  # pragma: no cover
            degradations.append(f"connector_failed_{index}: {type(exc).__name__}: {exc}")

    architecture_color = _rgba((spec.get("colors") or {}).get("architecture"), (0.95, 0.24, 0.92))
    for index, plane in enumerate(spec.get("architecture_planes") or []):
        try:
            color = _rgba(plane.get("color"), architecture_color[:3])
            _draw_wireframe(
                f"overlay_architecture_plane_{index}",
                plane.get("corners"),
                plane.get("edges"),
                color,
            )
            _draw_line(
                f"overlay_architecture_normal_{index}",
                plane.get("normal_from"),
                plane.get("normal_to"),
                color,
            )
        except Exception as exc:  # pragma: no cover
            degradations.append(f"architecture_plane_failed_{index}: {type(exc).__name__}: {exc}")

    # The legend is delivered to the VLM as structured metadata (colors, IDs,
    # categories, representations); no large world-space legend text is drawn, so
    # it can never clip in the frame or contaminate mask-based visibility.
    _apply_xray_targets(spec)
    bpy.context.view_layer.update()
    return degradations


def _apply_xray_targets(spec: dict) -> None:
    """Make a contained/occluding outer target transparent so an inner target
    and both OBBs remain inspectable. Read-only, in-memory blend factor only."""

    xray_ids = {
        str(target.get("id"))
        for target in _overlay_targets(spec)
        if target.get("xray") or target.get("transparent")
    }
    if not xray_ids:
        return
    resolver = _CanonicalResolver()
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        if resolver.resolve(obj) in xray_ids:
            try:
                obj.show_transparent = True
                color = list(getattr(obj, "color", (0.5, 0.5, 0.5, 1.0)))
                if len(color) >= 4:
                    color[3] = 0.25
                    obj.color = tuple(color)
            except Exception:  # pragma: no cover - depends on Blender build
                pass


def _overlay_targets(spec: dict) -> list[dict]:
    targets = spec.get("targets")
    if isinstance(targets, list) and targets:
        return [target for target in targets if isinstance(target, dict)]
    return [
        target
        for target in (spec.get("object_a"), spec.get("object_b"))
        if isinstance(target, dict)
    ]


class _CanonicalResolver:
    def __init__(self) -> None:
        self._cache: dict[str, str | None] = {}

    def resolve(self, obj) -> str | None:
        key = obj.name
        if key in self._cache:
            return self._cache[key]
        node = obj
        chain = []
        seen = set()
        while node is not None and node.name not in seen:
            seen.add(node.name)
            canonical = node.get(CANONICAL_ID_PROPERTY) if hasattr(node, "get") else None
            chain.append((node.name, canonical))
            node = node.parent
        resolved = _resolve_chain(chain)
        self._cache[key] = resolved
        return resolved


def _resolve_chain(chain: list[tuple[str, object]]) -> str | None:
    for _name, canonical in chain:
        if canonical is not None and str(canonical).strip():
            return str(canonical).strip()
    for name, _canonical in chain:
        for prefix in ("asset_", "proxy_"):
            if str(name).startswith(prefix):
                candidate = str(name)[len(prefix):]
                return _BLENDER_SUFFIX.sub("", candidate)
    return None


def _set_color(obj, color) -> None:
    try:
        obj.color = color
    except Exception:  # pragma: no cover
        pass
    data = getattr(obj, "data", None)
    materials = getattr(data, "materials", None)
    if materials is None:
        return
    try:
        materials.clear()
        materials.append(_object_color_material())
    except Exception:  # pragma: no cover - linked/read-only Blender data
        pass


def _object_color_material():
    """Shared emission material driven by each object's viewport color.

    Workbench reads ``obj.color`` directly. Cycles and Eevee do not, so the
    shared node graph reads Object Info -> Color and makes the same target/context
    colors visible without copying large asset meshes or mutating the saved file.
    """

    name = "benchmark_focus_object_color"
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    object_info = nodes.new("ShaderNodeObjectInfo")
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(object_info.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _draw_wireframe(name: str, corners, edges, color) -> None:
    if not corners or not edges:
        return
    points = [tuple(float(v) for v in corner) for corner in corners]
    curve = bpy.data.curves.new(f"{name}_curve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = _wire_radius(points)
    curve.bevel_resolution = 0
    for raw_edge in edges:
        edge = tuple(int(i) for i in raw_edge)
        if len(edge) != 2 or min(edge) < 0 or max(edge) >= len(points):
            continue
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (*points[edge[0]], 1.0)
        spline.points[1].co = (*points[edge[1]], 1.0)
    obj = bpy.data.objects.new(name, curve)
    _set_color(obj, color)
    obj[CANONICAL_ID_PROPERTY] = name
    bpy.context.collection.objects.link(obj)


def _draw_marker(name: str, position, color) -> None:
    if not position or len(position) != 3:
        return
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=tuple(float(v) for v in position))
    obj = bpy.context.object
    obj.name = name
    _set_color(obj, color)
    obj[CANONICAL_ID_PROPERTY] = name


def _draw_line(name: str, start, end, color) -> None:
    if not start or not end or len(start) != 3 or len(end) != 3:
        return
    start_point = tuple(float(v) for v in start)
    end_point = tuple(float(v) for v in end)
    curve = bpy.data.curves.new(f"{name}_curve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = _wire_radius([start_point, end_point])
    curve.bevel_resolution = 0
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (*start_point, 1.0)
    spline.points[1].co = (*end_point, 1.0)
    obj = bpy.data.objects.new(name, curve)
    _set_color(obj, color)
    obj[CANONICAL_ID_PROPERTY] = name
    bpy.context.collection.objects.link(obj)


def _wire_radius(points) -> float:
    vectors = [Vector(point) for point in points]
    if len(vectors) < 2:
        return 0.006
    span = max(float((left - right).length) for left in vectors for right in vectors)
    return max(0.003, min(0.02, span * 0.006))


def _render_pose(
    pose: dict,
    out_dir: Path,
    index: int,
    *,
    role: str,
    filename_prefix: str = "overlay",
) -> dict:
    if not isinstance(pose, dict):
        raise TypeError("each camera pose must be a JSON object")
    pose_id = str(pose.get("id") or f"view_{index:02d}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", pose_id).strip("._") or f"view_{index:02d}"
    location = _vector3(pose.get("location"), "location")
    target_location = _vector3(pose.get("target"), "target")

    target = bpy.data.objects.new(f"overlay_target_{safe_id}", None)
    target.empty_display_type = "PLAIN_AXES"
    target.location = target_location
    bpy.context.collection.objects.link(target)

    camera_data = bpy.data.cameras.new(f"overlay_camera_{safe_id}")
    camera_data.type = str(pose.get("camera_type") or "PERSP")
    camera_data.lens = float(pose.get("lens_mm") or 50.0)
    camera_data.sensor_width = float(pose.get("sensor_width_mm") or 36.0)
    camera_data.sensor_fit = str(pose.get("sensor_fit") or "HORIZONTAL").upper()
    camera_data.clip_start = max(0.001, float(pose.get("clip_start_m") or 0.02))
    camera_data.clip_end = max(camera_data.clip_start + 1.0, float(pose.get("clip_end_m") or 100.0))
    if camera_data.type == "ORTHO":
        camera_data.ortho_scale = max(0.1, float(pose.get("ortho_scale") or 5.0))
    camera = bpy.data.objects.new(f"overlay_camera_{safe_id}", camera_data)
    camera.location = location
    bpy.context.collection.objects.link(camera)
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    bpy.context.view_layer.update()

    bpy.context.scene.camera = camera
    render_path = out_dir / f"{filename_prefix}_{index:02d}_{safe_id}.png"
    bpy.context.scene.render.filepath = str(render_path)
    started_at = time.monotonic()
    bpy.ops.render.render(write_still=True)
    return {
        "id": pose_id,
        "name": str(pose.get("name") or pose_id),
        "path": str(render_path),
        "role": role,
        "camera_location": [float(value) for value in camera.location],
        "camera_target": [float(value) for value in target.location],
        "elapsed_seconds": time.monotonic() - started_at,
        "pose": pose,
    }


def _rgba(value, default) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]), 1.0)
    return (float(default[0]), float(default[1]), float(default[2]), 1.0)


def _vector3(value: object, label: str) -> Vector:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"camera pose {label} must be a three-vector")
    try:
        return Vector(tuple(float(item) for item in value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"camera pose {label} must be numeric") from exc


if __name__ == "__main__":
    main()
