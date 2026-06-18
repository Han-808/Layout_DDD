from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

from benchmark.utils.io import write_json
from benchmark.visualization.camera_policy import pair_camera_policy, room_camera_policy
from benchmark.workflow.scoring import find_layout_object, room_boundary


class HabitatRenderer:
    def __init__(self, *args, **kwargs) -> None:
        try:
            import habitat_sim  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional future dependency
            raise RuntimeError("HabitatRenderer requires habitat_sim, which is not installed.") from exc


class SimpleBBoxRenderer:
    def __init__(self, out_dir: str | Path, width: int = 640, height: int = 480) -> None:
        self.out_dir = Path(out_dir)
        self.width = width
        self.height = height

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

    def render_pair_views(self, case: dict, layout: dict, spec: dict) -> list[dict]:
        spec_id = str(spec.get("id") or "pair")
        subject = spec.get("subject") or spec.get("child")
        target = spec.get("object") or spec.get("parent")
        subject_obj = find_layout_object(layout, str(subject or ""))
        target_obj = find_layout_object(layout, str(target or ""))
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

    def _render(self, path: Path, case: dict, layout: dict, *, projection: str, highlight_ids: set[str]) -> None:
        canvas = _Canvas(self.width, self.height, (248, 249, 246, 255))
        objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
        projected = []

        boundary = room_boundary(case)
        for obj in objects:
            rect = _project_object(obj, projection)
            if rect is None:
                continue
            projected.append((obj, rect))

        bounds = _combined_bounds(boundary, projected, projection)
        if boundary and projection == "xy":
            points = [_map_point(point[0], point[1], bounds, self.width, self.height) for point in boundary]
            for start, end in zip(points, points[1:] + points[:1]):
                canvas.line(start, end, (64, 73, 67, 255), width=2)

        for index, (obj, rect) in enumerate(projected):
            x0, y0 = _map_point(rect[0], rect[1], bounds, self.width, self.height)
            x1, y1 = _map_point(rect[2], rect[3], bounds, self.width, self.height)
            left, right = sorted([x0, x1])
            top, bottom = sorted([y0, y1])
            obj_id = str(obj.get("object_id") or obj.get("id") or "")
            fill = _color(index, highlighted=obj_id in highlight_ids)
            outline = (210, 63, 45, 255) if obj_id in highlight_ids else (34, 39, 36, 255)
            canvas.rect(left, top, right, bottom, fill, outline)

        canvas.write_png(path)


def _artifact(path: Path, out_dir: Path, view_id: str) -> dict:
    return {
        "id": view_id,
        "path": path.resolve().relative_to(out_dir.resolve()).as_posix(),
        "abs_path": str(path),
    }


def _project_object(obj: dict, projection: str) -> tuple[float, float, float, float] | None:
    center = obj.get("center")
    size = obj.get("size")
    if not isinstance(center, list) or not isinstance(size, list) or len(center) != 3 or len(size) != 3:
        return None
    x, y, z = [float(v) for v in center]
    w, d, h = [float(v) for v in size]
    if projection == "xz":
        return (x - w / 2, z - h / 2, x + w / 2, z + h / 2)
    if projection == "oblique":
        ox = x + 0.45 * y
        oy = z + 0.2 * y
        return (ox - w / 2, oy - h / 2, ox + w / 2, oy + h / 2)
    return (x - w / 2, y - d / 2, x + w / 2, y + d / 2)


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


def _map_point(x: float, y: float, bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    px = int((x - min_x) / (max_x - min_x) * (width - 40) + 20)
    py = int(height - ((y - min_y) / (max_y - min_y) * (height - 40) + 20))
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
