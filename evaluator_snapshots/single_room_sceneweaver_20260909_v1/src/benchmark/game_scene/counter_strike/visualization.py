"""Audit and neutral-observation visualizations for CS static topology.

The annotated image is for human audit only.  The neutral mode deliberately
omits inferred roles, routes, cover proposals, engagement anchors, scores, and
case identity so a VLM cannot simply repeat the deterministic component that
will later be merged with its perceptual judgement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .topology import CounterStrikeTopology


COUNTER_STRIKE_TOPOLOGY_DIAGRAM_VERSION = "counter_strike_topology_diagram_v2"


def render_counter_strike_topology_diagram(
    topology: CounterStrikeTopology,
    *,
    out_path: str | Path,
    title: str,
    deterministic_metrics: dict[str, dict[str, Any]] | None = None,
    pixels_per_cell: int | None = None,
    mode: str = "audit",
) -> Path:
    """Render a legible top-down diagram and return its resolved path.

    ``mode="neutral_judge"`` is a strict observation boundary: only occupancy
    and the two declared spawn sets are shown.  ``mode="audit"`` retains the
    inferred overlays and optional deterministic score footer for humans.
    """

    if mode not in {"audit", "neutral_judge"}:
        raise ValueError("mode must be 'audit' or 'neutral_judge'")
    neutral = mode == "neutral_judge"

    rows, cols = topology.free.shape
    scale = pixels_per_cell or max(2, min(6, int(900 / max(rows, cols))))
    margin_top = 72
    margin_bottom = 86
    width = cols * scale
    height = rows * scale
    canvas = Image.new(
        "RGB",
        (width, height + margin_top + margin_bottom),
        (246, 247, 249),
    )
    map_image = np.zeros((rows, cols, 3), dtype=np.uint8)
    inside = np.asarray(topology.grid["inside_room"], dtype=bool)
    occupied = np.asarray(topology.grid["occupied"], dtype=bool)
    map_image[:] = (224, 228, 234)
    map_image[inside] = (239, 241, 244)
    map_image[occupied & inside] = (38, 43, 50)
    map_image[topology.free] = (250, 250, 247)

    if not neutral:
        _blend_mask(map_image, topology.team_a_spawn_zone, (65, 116, 214), 0.34)
        _blend_mask(map_image, topology.team_b_spawn_zone, (218, 78, 72), 0.34)
        _blend_mask(map_image, topology.team_a_preparation, (92, 154, 231), 0.18)
        _blend_mask(map_image, topology.team_b_preparation, (239, 118, 105), 0.18)
        _blend_mask(map_image, topology.flank_region, (150, 93, 201), 0.15)
        _blend_mask(map_image, topology.main_engagement, (245, 167, 54), 0.55)

    map_layer = Image.fromarray(np.flipud(map_image), mode="RGB").resize(
        (width, height),
        resample=Image.Resampling.NEAREST,
    )
    canvas.paste(map_layer, (0, margin_top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 12), title, fill=(24, 27, 31), font=font)
    subtitle = (
        f"grid {cols}×{rows} · {topology.resolution:.2f} m/cell · "
        "neutral occupancy + declared spawn aid; no inferred roles or scores"
        if neutral
        else (
            f"grid {cols}×{rows} · {topology.resolution:.2f} m/cell · "
            f"routes {len(topology.routes)} · "
            f"covers {len(topology.cover_candidates)}"
        )
    )
    draw.text((14, 34), subtitle, fill=(78, 84, 92), font=font)

    route_colors = {
        "main": (31, 143, 92),
        "flank": (117, 63, 177),
        "alternate": (118, 126, 137),
    }
    route_width = max(2, scale)
    if not neutral:
        for route in topology.routes:
            points = [
                _cell_pixel(
                    cell,
                    rows=rows,
                    scale=scale,
                    margin_top=margin_top,
                )
                for cell in route.cells
            ]
            if len(points) >= 2:
                draw.line(
                    points,
                    fill=route_colors[route.classification],
                    width=route_width,
                )

    if not neutral:
        for center in (
            item["center_xy"] for item in topology.cover_candidates
        ):
            cell = topology.world_to_cell(center)
            x, y = _cell_pixel(
                cell,
                rows=rows,
                scale=scale,
                margin_top=margin_top,
            )
            radius = max(2, scale)
            draw.rectangle(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(0, 145, 147),
                width=max(1, scale // 2),
            )

    _draw_spawn_set(
        draw,
        topology.team_a_cells,
        label="A",
        color=(31, 91, 210),
        rows=rows,
        scale=scale,
        margin_top=margin_top,
    )
    _draw_spawn_set(
        draw,
        topology.team_b_cells,
        label="B",
        color=(205, 55, 49),
        rows=rows,
        scale=scale,
        margin_top=margin_top,
    )
    if not neutral:
        ex, ey = _cell_pixel(
            topology.engagement_cell,
            rows=rows,
            scale=scale,
            margin_top=margin_top,
        )
        radius = max(5, 2 * scale)
        draw.ellipse(
            (ex - radius, ey - radius, ex + radius, ey + radius),
            fill=(245, 167, 54),
            outline=(112, 66, 0),
            width=max(1, scale // 2),
        )
        draw.text(
            (ex + radius + 2, ey - radius),
            "engagement",
            fill=(84, 48, 0),
            font=font,
        )

    legend_y = margin_top + height + 12
    legend = (
        [
            ("Team A declared spawn", (31, 91, 210)),
            ("Team B declared spawn", (205, 55, 49)),
            ("Blocking footprint", (38, 43, 50)),
            ("Walkable free space", (250, 250, 247)),
        ]
        if neutral
        else [
            ("Team A spawn/prep", (65, 116, 214)),
            ("Team B spawn/prep", (218, 78, 72)),
            ("Main engagement", (245, 167, 54)),
            ("Flank region/route", (117, 63, 177)),
            ("Main route", (31, 143, 92)),
            ("Cover candidate", (0, 145, 147)),
        ]
    )
    x = 14
    for label, color in legend:
        draw.rectangle((x, legend_y, x + 12, legend_y + 12), fill=color)
        draw.text((x + 17, legend_y), label, fill=(48, 52, 58), font=font)
        x += 17 + 7 * len(label)
        if x > width - 180:
            x = 14
            legend_y += 20
    if deterministic_metrics and not neutral:
        summary = " · ".join(
            f"{name}={float(result.get('score', 0.0)):.2f}"
            for name, result in deterministic_metrics.items()
        )
        draw.text(
            (14, margin_top + height + margin_bottom - 22),
            summary,
            fill=(67, 72, 79),
            font=font,
        )

    destination = Path(out_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")
    return destination


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return
    image[active] = np.rint(
        (1.0 - alpha) * image[active].astype(float)
        + alpha * np.asarray(color, dtype=float)
    ).astype(np.uint8)


def _cell_pixel(
    cell: tuple[int, int],
    *,
    rows: int,
    scale: int,
    margin_top: int,
) -> tuple[int, int]:
    row, col = cell
    return (
        int((col + 0.5) * scale),
        int(margin_top + (rows - row - 0.5) * scale),
    )


def _draw_spawn_set(
    draw: ImageDraw.ImageDraw,
    cells: tuple[tuple[int, int], ...],
    *,
    label: str,
    color: tuple[int, int, int],
    rows: int,
    scale: int,
    margin_top: int,
) -> None:
    radius = max(5, 2 * scale)
    for index, cell in enumerate(cells):
        x, y = _cell_pixel(
            cell,
            rows=rows,
            scale=scale,
            margin_top=margin_top,
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(255, 255, 255),
            width=max(1, scale // 2),
        )
        draw.text(
            (x + radius + 2, y - radius),
            f"{label}{index + 1}",
            fill=color,
            font=ImageFont.load_default(),
        )
