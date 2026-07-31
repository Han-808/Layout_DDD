#!/usr/bin/env python3
"""Neutral identity and anonymous partition visuals for blind grouping."""

from __future__ import annotations

import colorsys
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from benchmark.grouping import GroupingResult


def object_aliases(
    object_ids: Iterable[str],
) -> dict[str, str]:
    return {
        str(object_id): f"O{index:03d}"
        for index, object_id in enumerate(object_ids, start=1)
    }


def draw_identity_map(
    *,
    normalized: Any,
    aliases: dict[str, str],
    output_path: Path,
) -> None:
    image, drawing, transform = _base_plan_image(
        normalized,
        title="Object identity map · neutral grouping input",
        subtitle=(
            "Labels map the rendered objects to the exact IDs in the "
            "grouping request."
        ),
    )
    fill = (41, 78, 116, 190)
    outline = (125, 194, 255, 255)
    label_font = _font(15, bold=True)
    for item in normalized.objects:
        polygon = _object_polygon(item, transform)
        drawing.polygon(polygon, fill=fill, outline=outline, width=3)
        alias = aliases[str(item["object_id"])]
        center = transform(
            float(item["center"][0]),
            float(item["center"][1]),
        )
        _centered_text(
            drawing,
            center,
            alias,
            font=label_font,
            fill=(255, 255, 255, 255),
            stroke_fill=(0, 0, 0, 255),
        )
    _draw_object_legend(
        drawing,
        normalized=normalized,
        aliases=aliases,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG")


def draw_grouping_overlay(
    *,
    normalized: Any,
    aliases: dict[str, str],
    result: GroupingResult | dict[str, Any],
    blind_label: str,
    output_path: Path,
) -> dict[str, Any]:
    report = result.to_dict() if isinstance(result, GroupingResult) else result
    groups = report.get("object_groups")
    if not isinstance(groups, list):
        raise ValueError("grouping report object_groups must be a list")
    by_id = {
        str(item["object_id"]): item for item in normalized.objects
    }
    image, drawing, transform = _base_plan_image(
        normalized,
        title=f"Result {blind_label} · complete object partition",
        subtitle=(
            "Objects sharing a color and connecting line belong to one "
            "group. Method identity is intentionally hidden."
        ),
    )
    label_font = _font(15, bold=True)
    public_groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        members = [str(item) for item in group.get("object_ids", [])]
        color = _group_color(group_index - 1)
        centers = [
            transform(
                float(by_id[object_id]["center"][0]),
                float(by_id[object_id]["center"][1]),
            )
            for object_id in members
        ]
        if centers:
            centroid = (
                sum(point[0] for point in centers) / len(centers),
                sum(point[1] for point in centers) / len(centers),
            )
            for point in centers:
                drawing.line(
                    [centroid, point],
                    fill=(*color, 180),
                    width=4,
                )
        public_members: list[dict[str, str]] = []
        for object_id in members:
            item = by_id[object_id]
            polygon = _object_polygon(item, transform)
            drawing.polygon(
                polygon,
                fill=(*color, 175),
                outline=(*color, 255),
                width=4,
            )
            alias = aliases[object_id]
            center = transform(
                float(item["center"][0]),
                float(item["center"][1]),
            )
            _centered_text(
                drawing,
                center,
                alias,
                font=label_font,
                fill=(255, 255, 255, 255),
                stroke_fill=(0, 0, 0, 255),
            )
            public_members.append(
                {
                    "object_alias": alias,
                    "description": str(item["description"]),
                }
            )
        public_groups.append(
            {
                "display_group_id": f"G{group_index:02d}",
                "color": "#{:02x}{:02x}{:02x}".format(*color),
                "object_aliases": [
                    aliases[object_id] for object_id in members
                ],
                "members": public_members,
            }
        )
    _draw_group_legend(drawing, public_groups)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG")
    return {
        "blind_result_id": blind_label,
        "group_count": len(public_groups),
        "groups": public_groups,
    }


def _base_plan_image(
    normalized: Any,
    *,
    title: str,
    subtitle: str,
) -> tuple[
    Image.Image,
    ImageDraw.ImageDraw,
    Any,
]:
    width, height = 1800, 1250
    plot_left, plot_top = 70, 155
    plot_width, plot_height = 1110, 1020
    image = Image.new("RGBA", (width, height), (17, 20, 24, 255))
    drawing = ImageDraw.Draw(image, "RGBA")
    drawing.text(
        (55, 32),
        title,
        font=_font(34, bold=True),
        fill=(238, 242, 246, 255),
    )
    drawing.text(
        (57, 88),
        subtitle,
        font=_font(18),
        fill=(157, 171, 185, 255),
    )
    drawing.rounded_rectangle(
        (
            plot_left,
            plot_top,
            plot_left + plot_width,
            plot_top + plot_height,
        ),
        radius=16,
        fill=(29, 34, 40, 255),
        outline=(72, 84, 97, 255),
        width=2,
    )
    min_x, max_x, min_y, max_y = _plan_bounds(normalized)
    span_x = max(0.1, max_x - min_x)
    span_y = max(0.1, max_y - min_y)
    padding = 55
    scale = min(
        (plot_width - padding * 2) / span_x,
        (plot_height - padding * 2) / span_y,
    )
    offset_x = plot_left + (plot_width - span_x * scale) / 2.0
    offset_y = plot_top + (plot_height - span_y * scale) / 2.0

    def transform(x: float, y: float) -> tuple[float, float]:
        return (
            offset_x + (x - min_x) * scale,
            offset_y + (max_y - y) * scale,
        )

    if normalized.boundary:
        room = [
            transform(float(x), float(y))
            for x, y in normalized.boundary
        ]
        drawing.polygon(
            room,
            fill=(45, 50, 57, 255),
            outline=(154, 165, 177, 255),
            width=5,
        )
    drawing.text(
        (plot_left + 16, plot_top + 14),
        (
            f"{normalized.scene_type or 'unspecified scene'} · "
            f"{len(normalized.objects)} objects"
        ),
        font=_font(17, bold=True),
        fill=(220, 228, 236, 255),
    )
    return image, drawing, transform


def _plan_bounds(
    normalized: Any,
) -> tuple[float, float, float, float]:
    if normalized.boundary:
        xs = [float(point[0]) for point in normalized.boundary]
        ys = [float(point[1]) for point in normalized.boundary]
        return min(xs), max(xs), min(ys), max(ys)
    xs: list[float] = []
    ys: list[float] = []
    for item in normalized.objects:
        x, y = float(item["center"][0]), float(item["center"][1])
        width, depth = float(item["size"][0]), float(item["size"][1])
        xs.extend([x - width / 2.0, x + width / 2.0])
        ys.extend([y - depth / 2.0, y + depth / 2.0])
    return min(xs), max(xs), min(ys), max(ys)


def _object_polygon(
    item: dict[str, Any],
    transform: Any,
) -> list[tuple[float, float]]:
    x, y = float(item["center"][0]), float(item["center"][1])
    width, depth = float(item["size"][0]), float(item["size"][1])
    yaw = math.radians(float(item["rotation"][2]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    points: list[tuple[float, float]] = []
    for local_x, local_y in (
        (-width / 2.0, -depth / 2.0),
        (width / 2.0, -depth / 2.0),
        (width / 2.0, depth / 2.0),
        (-width / 2.0, depth / 2.0),
    ):
        world_x = x + local_x * cosine - local_y * sine
        world_y = y + local_x * sine + local_y * cosine
        points.append(transform(world_x, world_y))
    return points


def _draw_object_legend(
    drawing: ImageDraw.ImageDraw,
    *,
    normalized: Any,
    aliases: dict[str, str],
) -> None:
    left, top = 1235, 160
    drawing.text(
        (left, top),
        "Identity legend",
        font=_font(23, bold=True),
        fill=(169, 207, 255, 255),
    )
    objects = list(normalized.objects)
    columns = 2 if len(objects) > 34 else 1
    rows = math.ceil(len(objects) / columns)
    column_width = 270 if columns == 2 else 520
    line_height = min(25, max(14, int(940 / max(1, rows))))
    font = _font(max(11, min(16, line_height - 4)))
    for index, item in enumerate(objects):
        column = index // rows
        row = index % rows
        x = left + column * column_width
        y = top + 48 + row * line_height
        alias = aliases[str(item["object_id"])]
        description = str(item["description"])
        max_chars = 25 if columns == 2 else 55
        if len(description) > max_chars:
            description = description[: max_chars - 1] + "…"
        drawing.text(
            (x, y),
            f"{alias}  {description}",
            font=font,
            fill=(213, 221, 230, 255),
        )


def _draw_group_legend(
    drawing: ImageDraw.ImageDraw,
    groups: list[dict[str, Any]],
) -> None:
    left, top = 1235, 160
    drawing.text(
        (left, top),
        f"Partition · {len(groups)} groups",
        font=_font(23, bold=True),
        fill=(169, 207, 255, 255),
    )
    y = top + 50
    compact = len(groups) > 14
    group_font = _font(14 if compact else 16, bold=True)
    member_font = _font(12 if compact else 14)
    gap = 56 if compact else 72
    for group in groups:
        color = _hex_to_rgb(group["color"])
        drawing.rounded_rectangle(
            (left, y + 2, left + 28, y + 30),
            radius=5,
            fill=(*color, 230),
        )
        drawing.text(
            (left + 40, y),
            (
                f"{group['display_group_id']} · "
                f"{len(group['object_aliases'])} objects"
            ),
            font=group_font,
            fill=(238, 242, 246, 255),
        )
        aliases = " · ".join(group["object_aliases"])
        if len(aliases) > 56:
            aliases = aliases[:55] + "…"
        drawing.text(
            (left + 40, y + 27),
            aliases,
            font=member_font,
            fill=(170, 183, 196, 255),
        )
        y += gap
        if y > 1150:
            drawing.text(
                (left, 1180),
                "Remaining memberships are listed in the review panel.",
                font=_font(12),
                fill=(232, 187, 85, 255),
            )
            break


def _group_color(index: int) -> tuple[int, int, int]:
    hue = (0.57 + index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.92)
    return int(red * 255), int(green * 255), int(blue * 255)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))


def _centered_text(
    drawing: ImageDraw.ImageDraw,
    center: tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    stroke_fill: tuple[int, int, int, int],
) -> None:
    box = drawing.textbbox((0, 0), text, font=font, stroke_width=2)
    width = box[2] - box[0]
    height = box[3] - box[1]
    drawing.text(
        (center[0] - width / 2.0, center[1] - height / 2.0),
        text,
        font=font,
        fill=fill,
        stroke_width=2,
        stroke_fill=stroke_fill,
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(
            "/System/Library/Fonts/Supplemental/"
            + ("Arial Bold.ttf" if bold else "Arial.ttf")
        ),
        Path(
            "/usr/share/fonts/truetype/dejavu/"
            + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        ),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()
