#!/usr/bin/env python3
"""Build compact per-metric contact sheets for camera-evidence review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


METRICS = ("collision", "oob", "support")
LOCAL_HIGHLIGHT_ROLES = {"metric_local_highlight", "collision_pair_overlay"}
CELL_SIZE = 220
CAPTION_HEIGHT = 46
ROW_HEIGHT = CELL_SIZE + CAPTION_HEIGHT
COL_COUNT = 4


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    review_tsv = source_root / "review.tsv"
    if not review_tsv.is_file():
        raise FileNotFoundError(f"review table does not exist: {review_tsv}")

    rows = _read_review_rows(review_tsv)
    out_dir = source_root / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(11)

    outputs: dict[str, str] = {}
    for metric in METRICS:
        metric_rows = [row for row in rows if row.get("metric") == metric]
        if not metric_rows:
            continue
        output = out_dir / f"{metric}.png"
        _build_metric_sheet(metric_rows, output, font)
        outputs[metric] = str(output)

    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "event_count": len(rows),
                "outputs": outputs,
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="Support/artifacts/outputs/cal_dataset1_camera_evidence",
    )
    return parser.parse_args()


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _highlight_items(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    fixed_items = [
        item
        for item in comparison["arms"]["fixed_global_highlight"].get("items", [])
        if item.get("role") == "metric_local_highlight"
    ]
    local_items = [
        item
        for item in comparison["arms"]["metric_local_highlight"].get("items", [])
        if item.get("role") in LOCAL_HIGHLIGHT_ROLES
    ]
    if len(fixed_items) != 2 or len(local_items) != 2:
        raise ValueError(
            "expected two highlighted fixed-global and two highlighted local "
            f"images, got {len(fixed_items)} and {len(local_items)} for "
            f"{comparison.get('case_id')} / {comparison.get('event_id')}"
        )
    return fixed_items + local_items


def _build_metric_sheet(
    rows: list[dict[str, str]],
    output: Path,
    font: ImageFont.ImageFont,
) -> None:
    canvas = Image.new(
        "RGB",
        (COL_COUNT * CELL_SIZE, len(rows) * ROW_HEIGHT),
        color=(14, 14, 14),
    )
    draw = ImageDraw.Draw(canvas)

    for row_index, row in enumerate(rows):
        manifest_path = Path(row["comparison_manifest"]).expanduser().resolve()
        comparison = _read_json(manifest_path)
        items = _highlight_items(comparison)
        y = row_index * ROW_HEIGHT

        for column, item in enumerate(items):
            image_path = Path(str(item["path"])).expanduser().resolve()
            with Image.open(image_path) as source:
                tile = ImageOps.fit(
                    source.convert("RGB"),
                    (CELL_SIZE, CELL_SIZE),
                    method=Image.Resampling.LANCZOS,
                )
            x = column * CELL_SIZE
            canvas.paste(tile, (x, y))
            prefix = "F" if column < 2 else "L"
            ordinal = column + 1 if column < 2 else column - 1
            label = f"{prefix}{ordinal} {item.get('view_id', 'view')}"
            draw.text((x + 4, y + CELL_SIZE + 2), label, fill=(230, 230, 230), font=font)

        event_label = " | ".join(
            (
                str(row.get("severity_class") or "unknown"),
                str(row.get("case_id") or "unknown"),
                str(row.get("object_ids") or row.get("event_id") or "unknown"),
            )
        )
        draw.text(
            (4, y + CELL_SIZE + 20),
            event_label,
            fill=(245, 195, 35),
            font=font,
        )

    canvas.save(output, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
