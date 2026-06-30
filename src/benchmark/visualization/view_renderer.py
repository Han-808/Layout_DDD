from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

from benchmark.evidence_config import (
    DEFAULT_RENDER_CONFIG,
    DEFAULT_VIEW_VALIDATION_CONFIG,
    resolve_runtime_evidence_config,
)
from benchmark.utils.io import write_json
from benchmark.visualization.camera_policy import global_camera_policy, group_camera_policy, pair_camera_policy, room_camera_policy


DEFAULT_CANVAS_BACKGROUND = tuple(DEFAULT_RENDER_CONFIG["background_rgba"])
DEFAULT_CAMERA_CANDIDATES = [tuple(item) for item in DEFAULT_RENDER_CONFIG["camera_candidates"]]
DEFAULT_MIN_FOREGROUND_RATIO = DEFAULT_VIEW_VALIDATION_CONFIG["min_foreground_ratio"]
DEFAULT_MIN_VISIBLE_OBJECT_RATIO = DEFAULT_VIEW_VALIDATION_CONFIG["min_visible_object_ratio"]
DEFAULT_MAX_CAMERA_RETRIES = DEFAULT_VIEW_VALIDATION_CONFIG["max_camera_retries"]
MIN_VISIBLE_PIXEL_AREA = DEFAULT_RENDER_CONFIG["min_visible_pixel_area"]
CANVAS_MARGIN_PX = DEFAULT_RENDER_CONFIG["canvas_margin_px"]
PERSPECTIVE_BACKEND = "perspective_bbox_zbuffer"


class HabitatRenderer:
    def __init__(self, *args, **kwargs) -> None:
        try:
            import habitat_sim  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional future dependency
            raise RuntimeError("HabitatRenderer requires habitat_sim, which is not installed.") from exc


class SimpleBBoxRenderer:
    def __init__(self, out_dir: str | Path, width: int | None = None, height: int | None = None, benchmark_config: dict | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.benchmark_config = benchmark_config or {}
        render = resolve_runtime_evidence_config(self.benchmark_config, {}, {})["render"]
        self.width = int(width or render["width"])
        self.height = int(height or render["height"])

    def render_room_views(self, case: dict, layout: dict) -> list[dict]:
        room_dir = self.out_dir / "views" / "room"
        room_dir.mkdir(parents=True, exist_ok=True)
        policy = room_camera_policy()
        write_json(room_dir / "camera_policy.json", {"views": policy})
        artifacts = []
        for spec in policy:
            path = room_dir / f"{spec['id']}.png"
            self._render(path, case, layout, projection=spec["projection"], highlight_ids=set())
            artifacts.append(_artifact(path, self.out_dir, spec["id"]))
        artifacts.append(_artifact(room_dir / "camera_policy.json", self.out_dir, "camera_policy"))
        return artifacts

    def render_global_top_view(self, case: dict, layout: dict) -> list[dict]:
        global_dir = self.out_dir / "views" / "global"
        global_dir.mkdir(parents=True, exist_ok=True)
        policy = global_camera_policy()
        artifacts = []
        selected_policy = []
        for spec in policy:
            path = global_dir / f"{spec['id']}.png"
            diagnostics = self._render_perspective(
                path,
                case,
                layout,
                projection=spec["projection"],
                highlight_ids=set(),
                camera_candidate=0,
                camera_shift=(0.0, 0.0),
            )
            diagnostics["status"] = "valid" if float(diagnostics.get("foreground_ratio", 0.0)) > 0 else "invalid"
            artifacts.append(_artifact(path, self.out_dir, spec["id"], diagnostics=diagnostics))
            selected_policy.append(_camera_policy_entry(spec, diagnostics))
        write_json(global_dir / "camera_policy.json", {"views": selected_policy})
        artifacts.append(_artifact(global_dir / "camera_policy.json", self.out_dir, "camera_policy"))
        return artifacts

    def render_group_views(self, case: dict, layout: dict, group: dict, validation_config: dict | None = None) -> tuple[list[dict], list[dict]]:
        group_id = str(group["group_id"])
        object_ids = {str(object_id) for object_id in group.get("object_ids", [])}
        group_dir = self.out_dir / "views" / "groups" / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        policy = group_camera_policy(group_id)

        artifacts = []
        flags = []
        for spec in policy:
            path = group_dir / f"{spec['id']}.png"
            artifact, flag = self._render_validated_group_view(
                path=path,
                case=case,
                layout=layout,
                spec=spec,
                object_ids=object_ids,
                validation_config=validation_config or {},
            )
            artifacts.append(artifact)
            if flag:
                flags.append(flag)
        write_json(group_dir / "camera_policy.json", {"group": group, "views": [_camera_policy_entry(artifact, artifact.get("diagnostics", {})) for artifact in artifacts if artifact["id"] != "camera_policy"]})
        artifacts.append(_artifact(group_dir / "camera_policy.json", self.out_dir, "camera_policy"))
        return artifacts, flags

    def render_pair_views(self, case: dict, layout: dict, spec: dict) -> list[dict]:
        spec_id = str(spec.get("id") or "pair")
        subject = spec.get("subject") or spec.get("child")
        target = spec.get("object") or spec.get("parent")
        subject_obj = _find_layout_object(layout, str(subject or ""))
        target_obj = _find_layout_object(layout, str(target or ""))
        if subject_obj is None or target_obj is None:
            return []

        pair_dir = self.out_dir / "views" / "pairs" / spec_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        policy = pair_camera_policy(spec_id)
        write_json(pair_dir / "camera_policy.json", {"spec": spec, "views": policy})
        highlight_ids = {
            str(subject_obj.get("object_id") or subject_obj.get("id")),
            str(target_obj.get("object_id") or target_obj.get("id")),
        }
        artifacts = []
        for view in policy:
            path = pair_dir / f"{view['id']}.png"
            self._render(path, case, layout, projection=view["projection"], highlight_ids=highlight_ids)
            artifacts.append(_artifact(path, self.out_dir, view["id"]))
        artifacts.append(_artifact(pair_dir / "camera_policy.json", self.out_dir, "camera_policy"))
        return artifacts

    def _render_validated_group_view(
        self,
        *,
        path: Path,
        case: dict,
        layout: dict,
        spec: dict,
        object_ids: set[str],
        validation_config: dict,
    ) -> tuple[dict, dict | None]:
        resolved = resolve_runtime_evidence_config(self.benchmark_config, case, layout, {"object_ids": sorted(object_ids)})
        max_retries = int(validation_config.get("max_camera_retries", resolved["view_validation"].get("max_camera_retries", DEFAULT_MAX_CAMERA_RETRIES)))
        candidates = [tuple(item) for item in resolved["render"]["camera_candidates"]]
        candidates = candidates[: max(1, max_retries + 1)]
        selected_diagnostics = {}
        for candidate_index, shift in enumerate(candidates):
            diagnostics = self._render_perspective(
                path,
                case,
                layout,
                projection=spec["projection"],
                highlight_ids=object_ids,
                object_ids=object_ids,
                camera_candidate=candidate_index,
                camera_shift=shift,
            )
            diagnostics["camera_candidate"] = candidate_index
            diagnostics["camera_shift"] = list(shift)
            diagnostics["selected_camera_candidate"] = list(shift)
            diagnostics["retry_count"] = candidate_index
            selected_diagnostics = diagnostics
            status = _view_status(diagnostics, validation_config)
            if status in {"valid", "warning"}:
                diagnostics["status"] = status
                if status == "warning":
                    return _artifact(path, self.out_dir, spec["id"], diagnostics=diagnostics), {
                        "type": "view_warning",
                        "severity": "warning",
                        "group_id": spec.get("group_id"),
                        "projection": spec.get("projection"),
                        "view_id": spec.get("id"),
                        "message": "Foreground coverage is low, but expected objects are visible.",
                        "diagnostics": diagnostics,
                    }
                return _artifact(path, self.out_dir, spec["id"], diagnostics=diagnostics), None

        selected_diagnostics["status"] = "invalid"
        flag = {
            "type": "view_invalid",
            "group_id": spec.get("group_id"),
            "projection": spec.get("projection"),
            "view_id": spec.get("id"),
            "message": "All camera candidates were blank or majority blank.",
            "diagnostics": selected_diagnostics,
        }
        return _artifact(path, self.out_dir, spec["id"], diagnostics=selected_diagnostics), flag

    def _render_perspective(
        self,
        path: Path,
        case: dict,
        layout: dict,
        *,
        projection: str,
        highlight_ids: set[str],
        object_ids: set[str] | None = None,
        camera_candidate: int = 0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
    ) -> dict:
        group = {"object_ids": sorted(object_ids)} if object_ids is not None else None
        resolved = resolve_runtime_evidence_config(self.benchmark_config, case, layout, group)
        render_config = resolved["render"]
        canvas = _Canvas(self.width, self.height, tuple(render_config["background_rgba"]))
        zbuffer = [math.inf] * (self.width * self.height)
        owners: list[str | None] = [None] * (self.width * self.height)
        objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
        if object_ids is not None:
            objects = [obj for obj in objects if str(obj.get("object_id") or obj.get("id")) in object_ids]

        boundary = _room_boundary(case)
        camera = _camera_spec(
            projection=projection,
            objects=objects,
            boundary=boundary,
            shift=camera_shift,
            width=self.width,
            height=self.height,
            render_config=render_config,
        )
        floor_polygons = _room_floor_polygons(case)
        if floor_polygons and projection == "xy":
            floor_z = _floor_z(case)
            for polygon in floor_polygons:
                projected_boundary = [
                    _project_world((float(point[0]), float(point[1]), floor_z), camera, self.width, self.height)
                    for point in polygon
                ]
                for start, end in zip(projected_boundary, projected_boundary[1:] + projected_boundary[:1]):
                    if start and end:
                        canvas.line((int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (64, 73, 67, 255), width=2)

        object_pixel_counts: dict[str, int] = {}
        for index, obj in enumerate(objects):
            object_id = str(obj.get("object_id") or obj.get("id") or "")
            vertices = _cuboid_vertices(obj)
            if not vertices:
                continue
            projected = [_project_world(vertex, camera, self.width, self.height) for vertex in vertices]
            fill = _opaque_color(_color(index, highlighted=object_id in highlight_ids))
            for triangle in _cuboid_triangles():
                tri = [projected[i] for i in triangle]
                if any(point is None for point in tri):
                    continue
                _raster_triangle(canvas, zbuffer, owners, tri[0], tri[1], tri[2], fill, object_id)
            outline = (210, 63, 45, 255) if object_id in highlight_ids else (34, 39, 36, 255)
            for left, right in _cuboid_edges():
                a = projected[left]
                b = projected[right]
                if a and b:
                    canvas.line((int(a[0]), int(a[1])), (int(b[0]), int(b[1])), outline, width=1)

        for owner in owners:
            if owner:
                object_pixel_counts[owner] = object_pixel_counts.get(owner, 0) + 1
        canvas.write_png(path)
        return _perspective_diagnostics(
            object_pixel_counts,
            expected_object_ids=object_ids or {str(obj.get("object_id") or obj.get("id") or "") for obj in objects},
            width=self.width,
            height=self.height,
            camera=camera,
            projection=projection,
            camera_candidate=camera_candidate,
            selected_camera_candidate=list(camera_shift),
            resolved_config=resolved,
            min_visible_pixel_area=float(render_config["min_visible_pixel_area"]),
        )

    def _render(
        self,
        path: Path,
        case: dict,
        layout: dict,
        *,
        projection: str,
        highlight_ids: set[str],
        object_ids: set[str] | None = None,
        bounds_shift: tuple[float, float] = (0.0, 0.0),
    ) -> dict:
        canvas = _Canvas(self.width, self.height, DEFAULT_CANVAS_BACKGROUND)
        objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
        if object_ids is not None:
            objects = [obj for obj in objects if str(obj.get("object_id") or obj.get("id")) in object_ids]
        projected = []

        boundary = _room_boundary(case)
        for obj in objects:
            rect = _project_object(obj, projection)
            if rect is None:
                continue
            projected.append((obj, rect))

        bounds = _shift_bounds(_combined_bounds(boundary, projected, projection), bounds_shift)
        floor_polygons = _room_floor_polygons(case)
        if floor_polygons and projection == "xy":
            for polygon in floor_polygons:
                points = [_map_point(point[0], point[1], bounds, self.width, self.height) for point in polygon]
                for start, end in zip(points, points[1:] + points[:1]):
                    canvas.line(start, end, (64, 73, 67, 255), width=2)

        pixel_rects: list[tuple[str, tuple[int, int, int, int]]] = []
        for index, (obj, rect) in enumerate(projected):
            x0, y0 = _map_point(rect[0], rect[1], bounds, self.width, self.height)
            x1, y1 = _map_point(rect[2], rect[3], bounds, self.width, self.height)
            left, right = sorted([x0, x1])
            top, bottom = sorted([y0, y1])
            pixel_rects.append((str(obj.get("object_id") or obj.get("id") or ""), (left, top, right, bottom)))
            obj_id = str(obj.get("object_id") or obj.get("id") or "")
            fill = _color(index, highlighted=obj_id in highlight_ids)
            outline = (210, 63, 45, 255) if obj_id in highlight_ids else (34, 39, 36, 255)
            canvas.rect(left, top, right, bottom, fill, outline)

        canvas.write_png(path)
        return _view_diagnostics(pixel_rects, expected_object_ids=object_ids or {object_id for object_id, _ in pixel_rects}, width=self.width, height=self.height)


def _find_layout_object(layout: dict, object_id: str) -> dict | None:
    for obj in layout.get("objects", []):
        if isinstance(obj, dict) and str(obj.get("object_id") or obj.get("id") or "") == object_id:
            return obj
    return None


def _room_boundary(case: dict) -> list[list[float]]:
    room = case.get("room") if isinstance(case.get("room"), dict) else {}
    boundary = room.get("floor_polygon") or room.get("boundary") or []
    return boundary if isinstance(boundary, list) else []


def _room_floor_polygons(case: dict) -> list[list[list[float]]]:
    room = case.get("room") if isinstance(case.get("room"), dict) else {}
    floor_plan = room.get("floor_plan") if isinstance(room.get("floor_plan"), dict) else {}
    regions = floor_plan.get("regions") or room.get("regions") or []
    polygons = []
    if isinstance(regions, list):
        for region in regions:
            if not isinstance(region, dict):
                continue
            polygon = _valid_polygon(region.get("floor_polygon"))
            if polygon:
                polygons.append(polygon)
    if polygons:
        return polygons
    boundary = _valid_polygon(room.get("floor_polygon") or room.get("boundary"))
    return [boundary] if boundary else []


def _valid_polygon(value: object) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 3:
        return []
    polygon = []
    for point in value:
        if not isinstance(point, list) or len(point) < 2:
            return []
        polygon.append([float(point[0]), float(point[1])])
    return polygon


def _camera_policy_entry(spec: dict, diagnostics: dict) -> dict:
    entry = dict(spec)
    for key in [
        "projection_type",
        "camera_position",
        "camera_look_at",
        "camera_up",
        "fov_degrees",
        "near",
        "far",
        "target_object_ids",
        "camera_candidate",
        "selected_camera_candidate",
        "retry_count",
        "render_backend",
        "resolved_config",
    ]:
        if key in diagnostics:
            entry[key] = diagnostics[key]
    return entry


def _artifact(path: Path, out_dir: Path, view_id: str, diagnostics: dict | None = None) -> dict:
    artifact = {
        "id": view_id,
        "path": path.resolve().relative_to(out_dir.resolve()).as_posix(),
        "abs_path": str(path),
    }
    if diagnostics is not None:
        artifact["diagnostics"] = diagnostics
    return artifact


def _camera_spec(
    *,
    projection: str,
    objects: list[dict],
    boundary: list[list[float]],
    shift: tuple[float, float],
    width: int,
    height: int,
    render_config: dict,
) -> dict:
    min_x, min_y, min_z, max_x, max_y, max_z = _scene_extents(objects, boundary)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    span_z = max(max_z - min_z, 1.0)
    target = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, float(render_config["effective_target_z"]))
    fov = float(render_config["fov_degrees"])
    aspect = float(width) / float(height)
    fit_span = max(span_x / max(aspect, 1.0e-6), span_y, span_z)
    distance = max(
        float(render_config["effective_camera_distance_m"]),
        (fit_span * float(render_config["fit_span_scale"])) / math.tan(math.radians(fov) / 2.0) + max(span_z, 0.5),
    )
    shift_x, shift_y = shift
    top_shift_scale = float(render_config["top_look_at_shift_scale"])
    side_shift_scale = float(render_config["side_look_at_shift_scale"])

    if projection == "yz":
        position = (target[0] + distance, target[1] + shift_x * span_y, target[2] + shift_y * span_z)
        look_at = (target[0], target[1] + shift_x * span_y * side_shift_scale, target[2])
        up = (0.0, 0.0, 1.0)
    elif projection == "xz":
        position = (target[0] + shift_x * span_x, target[1] - distance, target[2] + shift_y * span_z)
        look_at = (target[0] + shift_x * span_x * side_shift_scale, target[1], target[2])
        up = (0.0, 0.0, 1.0)
    else:
        position = (target[0] + shift_x * span_x, target[1] + shift_y * span_y, target[2] + distance)
        look_at = (target[0] + shift_x * span_x * top_shift_scale, target[1] + shift_y * span_y * top_shift_scale, target[2])
        up = (0.0, 1.0, 0.0)

    basis = _camera_basis(position, look_at, up)
    return {
        "projection": projection,
        "projection_type": "perspective",
        "position": position,
        "look_at": look_at,
        "up": basis["up"],
        "right": basis["right"],
        "forward": basis["forward"],
        "fov_degrees": fov,
        "near": float(render_config["near"]),
        "far": float(render_config["far"]),
        "target_object_ids": [str(obj.get("object_id") or obj.get("id") or "") for obj in objects],
    }


def _scene_extents(objects: list[dict], boundary: list[list[float]]) -> tuple[float, float, float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for point in boundary:
        if isinstance(point, list) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    for obj in objects:
        center = obj.get("center")
        size = obj.get("size")
        if not isinstance(center, list) or not isinstance(size, list) or len(center) != 3 or len(size) != 3:
            continue
        cx, cy, cz = [float(value) for value in center]
        sx, sy, sz = [float(value) for value in size]
        xs.extend([cx - sx / 2.0, cx + sx / 2.0])
        ys.extend([cy - sy / 2.0, cy + sy / 2.0])
        zs.extend([cz - sz / 2.0, cz + sz / 2.0])
    if not xs:
        xs = [0.0, 1.0]
    if not ys:
        ys = [0.0, 1.0]
    if not zs:
        zs = [0.0, 1.0]
    return min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)


def _camera_basis(position: tuple[float, float, float], look_at: tuple[float, float, float], up: tuple[float, float, float]) -> dict:
    forward = _normalize(_sub(look_at, position))
    right = _normalize(_cross(forward, up))
    if _length(right) <= 1.0e-9:
        right = (1.0, 0.0, 0.0)
    true_up = _normalize(_cross(right, forward))
    return {"forward": forward, "right": right, "up": true_up}


def _project_world(point: tuple[float, float, float], camera: dict, width: int, height: int) -> tuple[float, float, float] | None:
    rel = _sub(point, camera["position"])
    x_cam = _dot(rel, camera["right"])
    y_cam = _dot(rel, camera["up"])
    z_cam = _dot(rel, camera["forward"])
    if z_cam <= float(camera["near"]) or z_cam >= float(camera["far"]):
        return None
    focal = (float(height) / 2.0) / math.tan(math.radians(float(camera["fov_degrees"])) / 2.0)
    x = float(width) / 2.0 + focal * x_cam / z_cam
    y = float(height) / 2.0 - focal * y_cam / z_cam
    return (x, y, z_cam)


def _cuboid_vertices(obj: dict) -> list[tuple[float, float, float]]:
    center = obj.get("center")
    size = obj.get("size")
    if not isinstance(center, list) or not isinstance(size, list) or len(center) != 3 or len(size) != 3:
        return []
    cx, cy, cz = [float(value) for value in center]
    sx, sy, sz = [float(value) for value in size]
    yaw = math.radians(float(obj.get("yaw", 0.0) or 0.0))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    vertices = []
    for lx, ly, lz in [
        (-sx / 2, -sy / 2, -sz / 2),
        (sx / 2, -sy / 2, -sz / 2),
        (sx / 2, sy / 2, -sz / 2),
        (-sx / 2, sy / 2, -sz / 2),
        (-sx / 2, -sy / 2, sz / 2),
        (sx / 2, -sy / 2, sz / 2),
        (sx / 2, sy / 2, sz / 2),
        (-sx / 2, sy / 2, sz / 2),
    ]:
        vertices.append((cx + lx * cos_yaw - ly * sin_yaw, cy + lx * sin_yaw + ly * cos_yaw, cz + lz))
    return vertices


def _cuboid_triangles() -> list[tuple[int, int, int]]:
    quads = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    return [(a, b, c) for a, b, c, _ in quads] + [(a, c, d) for a, _, c, d in quads]


def _cuboid_edges() -> list[tuple[int, int]]:
    return [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def _raster_triangle(
    canvas: "_Canvas",
    zbuffer: list[float],
    owners: list[str | None],
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    color: tuple[int, int, int, int],
    object_id: str,
) -> None:
    min_x = max(0, int(math.floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(canvas.width - 1, int(math.ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(math.floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(canvas.height - 1, int(math.ceil(max(p0[1], p1[1], p2[1]))))
    area = _edge_value(p0, p1, p2[0], p2[1])
    if math.isclose(area, 0.0):
        return
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            py = y + 0.5
            w0 = _edge_value(p1, p2, px, py) / area
            w1 = _edge_value(p2, p0, px, py) / area
            w2 = _edge_value(p0, p1, px, py) / area
            if w0 < -1.0e-6 or w1 < -1.0e-6 or w2 < -1.0e-6:
                continue
            depth = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
            offset = y * canvas.width + x
            if depth < zbuffer[offset]:
                zbuffer[offset] = depth
                owners[offset] = object_id
                canvas.point(x, y, color)


def _edge_value(a: tuple[float, float, float], b: tuple[float, float, float], x: float, y: float) -> float:
    return (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])


def _perspective_diagnostics(
    object_pixel_counts: dict[str, int],
    *,
    expected_object_ids: set[str],
    width: int,
    height: int,
    camera: dict,
    projection: str,
    camera_candidate: int,
    selected_camera_candidate: list[float],
    resolved_config: dict,
    min_visible_pixel_area: float,
) -> dict:
    expected = sorted(expected_object_ids)
    visible_ids = sorted(object_id for object_id in expected if object_pixel_counts.get(object_id, 0) >= min_visible_pixel_area)
    foreground_pixels = sum(object_pixel_counts.values())
    canvas_area = float(width * height)
    return {
        "render_backend": PERSPECTIVE_BACKEND,
        "projection": projection,
        "projection_type": "perspective",
        "camera_position": _round_vector(camera["position"]),
        "camera_look_at": _round_vector(camera["look_at"]),
        "camera_up": _round_vector(camera["up"]),
        "fov_degrees": camera["fov_degrees"],
        "near": camera["near"],
        "far": camera["far"],
        "target_object_ids": camera["target_object_ids"],
        "camera_candidate": camera_candidate,
        "selected_camera_candidate": selected_camera_candidate,
        "foreground_ratio": min(1.0, foreground_pixels / canvas_area) if canvas_area else 0.0,
        "expected_object_ids": expected,
        "visible_object_ids": visible_ids,
        "visible_object_ratio": (float(len(visible_ids)) / float(len(expected))) if expected else 0.0,
        "object_pixel_counts": object_pixel_counts,
        "resolved_config": _public_resolved_config(resolved_config),
    }


def _public_resolved_config(resolved: dict) -> dict:
    render = resolved.get("render", {})
    return {
        "room_extent_m": resolved.get("room_extent_m"),
        "room_height_m": resolved.get("room_height_m"),
        "scene_bbox_extent_m": resolved.get("scene_bbox_extent_m"),
        "group_extent_m": resolved.get("group_extent_m"),
        "group_center": resolved.get("group_center"),
        "object_scale": resolved.get("object_scale"),
        "render": {
            "backend": render.get("backend"),
            "width": render.get("width"),
            "height": render.get("height"),
            "fov_degrees": render.get("fov_degrees"),
            "near": render.get("near"),
            "far": render.get("far"),
            "min_visible_pixel_area": render.get("min_visible_pixel_area"),
            "min_camera_distance_m": render.get("min_camera_distance_m"),
            "distance_scale": render.get("distance_scale"),
            "fit_span_scale": render.get("fit_span_scale"),
            "effective_camera_distance_m": render.get("effective_camera_distance_m"),
            "effective_target_z": render.get("effective_target_z"),
            "camera_candidates": render.get("camera_candidates"),
        },
        "view_validation": resolved.get("view_validation", {}),
    }


def _floor_z(case: dict) -> float:
    room = case.get("room") if isinstance(case.get("room"), dict) else {}
    return float(room.get("floor_z", 0.0))


def _opaque_color(color: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (color[0], color[1], color[2], 255)


def _round_vector(vector: tuple[float, float, float]) -> list[float]:
    return [round(float(value), 4) for value in vector]


def _add(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _length(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _length(vector)
    if length <= 1.0e-9:
        return (0.0, 0.0, 1.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)



def _project_object(obj: dict, projection: str) -> tuple[float, float, float, float] | None:
    center = obj.get("center")
    size = obj.get("size")
    if not isinstance(center, list) or not isinstance(size, list) or len(center) != 3 or len(size) != 3:
        return None
    x, y, z = [float(v) for v in center]
    w, d, h = [float(v) for v in size]
    if projection == "xz":
        return (x - w / 2, z - h / 2, x + w / 2, z + h / 2)
    if projection == "yz":
        return (y - d / 2, z - h / 2, y + d / 2, z + h / 2)
    if projection == "oblique":
        ox = x + 0.45 * y
        oy = z + 0.2 * y
        return (ox - w / 2, oy - h / 2, ox + w / 2, oy + h / 2)
    return (x - w / 2, y - d / 2, x + w / 2, y + d / 2)


def _view_is_valid(diagnostics: dict, validation_config: dict) -> bool:
    return _view_status(diagnostics, validation_config) in {"valid", "warning"}


def _view_status(diagnostics: dict, validation_config: dict) -> str:
    min_foreground_ratio = float(validation_config.get("min_foreground_ratio", DEFAULT_MIN_FOREGROUND_RATIO))
    min_visible_object_ratio = float(validation_config.get("min_visible_object_ratio", DEFAULT_MIN_VISIBLE_OBJECT_RATIO))
    foreground_good = float(diagnostics.get("foreground_ratio", 0.0)) >= min_foreground_ratio
    visibility_good = float(diagnostics.get("visible_object_ratio", 0.0)) >= min_visible_object_ratio
    if foreground_good and visibility_good:
        return "valid"
    if visibility_good:
        return "warning"
    return "invalid"


def _view_diagnostics(
    pixel_rects: list[tuple[str, tuple[int, int, int, int]]],
    *,
    expected_object_ids: set[str],
    width: int,
    height: int,
) -> dict:
    expected = sorted(expected_object_ids)
    visible_ids = []
    union_area = 0.0
    for index, (object_id, rect) in enumerate(pixel_rects):
        area = _rect_area(rect)
        if area <= 0:
            continue
        union_area += area
        later_rects = [later_rect for _, later_rect in pixel_rects[index + 1 :]]
        occluded_area = min(area, sum(_rect_intersection_area(rect, later) for later in later_rects))
        visible_area = max(0.0, area - occluded_area)
        if visible_area >= MIN_VISIBLE_PIXEL_AREA:
            visible_ids.append(object_id)
    canvas_area = float(width * height)
    visible_expected = sorted({object_id for object_id in visible_ids if object_id in expected})
    return {
        "foreground_ratio": min(1.0, union_area / canvas_area) if canvas_area else 0.0,
        "expected_object_ids": expected,
        "visible_object_ids": visible_expected,
        "visible_object_ratio": (float(len(visible_expected)) / float(len(expected))) if expected else 0.0,
    }


def _rect_area(rect: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = rect
    return float(max(0, right - left + 1) * max(0, bottom - top + 1))


def _rect_intersection_area(left_rect: tuple[int, int, int, int], right_rect: tuple[int, int, int, int]) -> float:
    left = max(left_rect[0], right_rect[0])
    top = max(left_rect[1], right_rect[1])
    right = min(left_rect[2], right_rect[2])
    bottom = min(left_rect[3], right_rect[3])
    return _rect_area((left, top, right, bottom))


def _combined_bounds(boundary: list[list[float]], projected: list[tuple[dict, tuple[float, float, float, float]]], projection: str) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    if boundary and projection == "xy":
        xs.extend(float(point[0]) for point in boundary)
        ys.extend(float(point[1]) for point in boundary)
    for _, rect in projected:
        xs.extend([rect[0], rect[2]])
        ys.extend([rect[1], rect[3]])
    if not xs or not ys:
        return (0.0, 0.0, 1.0, 1.0)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max(0.5, (max_x - min_x) * 0.08)
    pad_y = max(0.5, (max_y - min_y) * 0.08)
    if math.isclose(min_x, max_x):
        max_x += 1.0
    if math.isclose(min_y, max_y):
        max_y += 1.0
    return (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)


def _shift_bounds(bounds: tuple[float, float, float, float], shift: tuple[float, float]) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = bounds
    span_x = max_x - min_x
    span_y = max_y - min_y
    dx, dy = shift
    return (min_x + dx * span_x, min_y + dy * span_y, max_x + dx * span_x, max_y + dy * span_y)


def _map_point(x: float, y: float, bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    px = int((x - min_x) / (max_x - min_x) * (width - 2 * CANVAS_MARGIN_PX) + CANVAS_MARGIN_PX)
    py = int(height - ((y - min_y) / (max_y - min_y) * (height - 2 * CANVAS_MARGIN_PX) + CANVAS_MARGIN_PX))
    return px, py


def _color(index: int, highlighted: bool = False) -> tuple[int, int, int, int]:
    if highlighted:
        return (242, 169, 59, 210)
    palette = [
        (92, 140, 117, 190),
        (93, 116, 160, 190),
        (170, 129, 87, 190),
        (150, 112, 151, 190),
        (101, 151, 169, 190),
    ]
    return palette[index % len(palette)]


class _Canvas:
    def __init__(self, width: int, height: int, color: tuple[int, int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(color * width * height)

    def rect(self, left: int, top: int, right: int, bottom: int, fill: tuple[int, int, int, int], outline: tuple[int, int, int, int]) -> None:
        left = max(0, min(self.width - 1, left))
        right = max(0, min(self.width - 1, right))
        top = max(0, min(self.height - 1, top))
        bottom = max(0, min(self.height - 1, bottom))
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                self.point(x, y, fill)
        self.line((left, top), (right, top), outline, width=2)
        self.line((right, top), (right, bottom), outline, width=2)
        self.line((right, bottom), (left, bottom), outline, width=2)
        self.line((left, bottom), (left, top), outline, width=2)

    def line(self, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int, int], width: int = 1) -> None:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in range(-(width // 2), width // 2 + 1):
                for oy in range(-(width // 2), width // 2 + 1):
                    self.point(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def point(self, x: int, y: int, color: tuple[int, int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        offset = (y * self.width + x) * 4
        self.pixels[offset : offset + 4] = bytes(color)

    def write_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        stride = self.width * 4
        for row in range(self.height):
            raw.append(0)
            start = row * stride
            raw.extend(self.pixels[start : start + stride])
        png = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                _chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 6, 0, 0, 0)),
                _chunk(b"IDAT", zlib.compress(bytes(raw), 6)),
                _chunk(b"IEND", b""),
            ]
        )
        path.write_bytes(png)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
