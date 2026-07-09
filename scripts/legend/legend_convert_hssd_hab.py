from __future__ import annotations

LEGEND_INPUT_CHAIN = True
CURRENT_INPUT_CHAIN = "natural_language"

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.legend.hssd.hssd_hab_converter import convert_hssd_hab
from benchmark.input_modes import list_all_input_modes


def main() -> None:
    parser = argparse.ArgumentParser(description="LEGEND: convert local HSSD-HAB scene instances into legacy bm_instance v2 cases.")
    parser.add_argument("--hssd-root", default=str(PROJECT_ROOT / "data" / "external" / "hssd-hab"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "benchmark_cases" / "hssd"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--scene-file",
        action="append",
        default=None,
        help="Convert one explicit *.scene_instance.json file. Can be repeated.",
    )
    parser.add_argument("--max-objects", type=int, default=None, help="Optional max objects per converted scene.")
    parser.add_argument(
        "--compact-object-ids",
        action="store_true",
        help="Use short object_### IDs and hssd_object_### categories while preserving source metadata.",
    )
    parser.add_argument(
        "--preserve-raw-metadata",
        action="store_true",
        help="Store raw HSSD object transform metadata on every imported object.",
    )
    parser.add_argument(
        "--bbox-from-scale",
        action="store_true",
        help="Use abs(non_uniform_scale) as bbox_size when no explicit bbox/dimensions are available.",
    )
    parser.add_argument(
        "--no-estimated-relations",
        action="store_true",
        help="Do not synthesize deterministic estimated spatial cues.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["prompt_only", "structured_basic"],
        choices=["prompt_only", "structured_basic", "structured_relation"],
    )
    parser.add_argument(
        "--input-representation-mode",
        choices=sorted(list_all_input_modes(include_aliases=True)),
        default=None,
        help="Model-facing scene representation mode stored on converted cases.",
    )
    args = parser.parse_args()

    paths = convert_hssd_hab(
        hssd_root=Path(args.hssd_root),
        out_dir=Path(args.out_dir),
        limit=args.limit,
        scene_paths=args.scene_file,
        levels=args.levels,
        max_objects=args.max_objects,
        compact_object_ids=args.compact_object_ids,
        preserve_raw_metadata=args.preserve_raw_metadata,
        bbox_from_scale=args.bbox_from_scale,
        include_estimated_relations=not args.no_estimated_relations,
        input_representation_mode=args.input_representation_mode,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
