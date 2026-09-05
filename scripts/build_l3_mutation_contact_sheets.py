#!/usr/bin/env python3
"""Build compact source/variant contact sheets for independent render QA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_l3_mutation_dataset import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cases-per-sheet", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config, output_override=args.output_root)
    result = build_contact_sheets(
        config,
        cases_per_sheet=max(1, int(args.cases_per_sheet)),
    )
    print(json.dumps(result, indent=2))


def build_contact_sheets(
    config: dict[str, Any],
    *,
    cases_per_sheet: int,
) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    dataset = _read_json(output_root / "dataset_manifest.json")
    source_by_id = {
        str(item["source_id"]): item for item in dataset["sources"]
    }
    variants = sorted(
        dataset["variants"],
        key=lambda item: str(item["variant_id"]),
    )
    incomplete = [
        str(item["variant_id"])
        for item in variants
        if item.get("render_status") != "complete"
        or source_by_id[str(item["source_id"])].get("render_status")
        != "complete"
    ]
    if incomplete:
        raise RuntimeError(
            f"cannot build contact sheets; incomplete renders: {incomplete}"
        )
    contact_root = output_root / "independent_review" / "contact_sheets"
    contact_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for sheet_index in range(
        math.ceil(len(variants) / cases_per_sheet)
    ):
        batch = variants[
            sheet_index * cases_per_sheet :
            (sheet_index + 1) * cases_per_sheet
        ]
        path = contact_root / f"sheet_{sheet_index + 1:03d}.png"
        _draw_sheet(
            path,
            batch=batch,
            source_by_id=source_by_id,
        )
        paths.append(str(path.resolve()))
    manifest = {
        "schema_version": "l3_mutation_contact_sheets_v1",
        "dataset_fingerprint": dataset["dataset_fingerprint"],
        "cases_per_sheet": cases_per_sheet,
        "sheet_count": len(paths),
        "sheets": paths,
    }
    _write_json(
        output_root
        / "independent_review"
        / "contact_sheet_manifest.json",
        manifest,
    )
    return manifest


def _draw_sheet(
    path: Path,
    *,
    batch: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
) -> None:
    cell = 256
    title_height = 42
    columns = (
        ("source_top", "SOURCE · TOP"),
        ("variant_top", "VARIANT · TOP"),
        ("variant_perspective", "VARIANT · PERSPECTIVE"),
        ("variant_identity", "VARIANT · IDENTITY"),
    )
    width = cell * len(columns)
    height = title_height + len(batch) * (cell + title_height)
    sheet = Image.new("RGB", (width, height), "#0b0e13")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column_index, (_, title) in enumerate(columns):
        draw.text(
            (column_index * cell + 8, 14),
            title,
            fill="#dce7f7",
            font=font,
        )
    for row_index, variant in enumerate(batch):
        source = source_by_id[str(variant["source_id"])]
        row_y = title_height + row_index * (cell + title_height)
        label = (
            f"{variant['variant_id']} · {variant['review_id']} · "
            f"{source['scene_type']} · {variant['object_count']} objects"
        )
        draw.rectangle(
            (0, row_y, width, row_y + title_height),
            fill="#141922",
        )
        draw.text((8, row_y + 14), label, fill="#ffffff", font=font)
        paths = {
            "source_top": source["view_paths"]["top"],
            "variant_top": variant["view_paths"]["top"],
            "variant_perspective": variant["view_paths"]["perspective"],
            "variant_identity": variant["view_paths"]["identity_map"],
        }
        for column_index, (key, _) in enumerate(columns):
            image = Image.open(paths[key]).convert("RGB")
            image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            x = column_index * cell + (cell - image.width) // 2
            y = row_y + title_height + (cell - image.height) // 2
            sheet.paste(image, (x, y))
    sheet.save(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
