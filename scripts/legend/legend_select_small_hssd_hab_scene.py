from __future__ import annotations

LEGEND_INPUT_CHAIN = True
CURRENT_INPUT_CHAIN = "natural_language"

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.legend.hssd.hssd_small_selector import convert_selected_small_hssd_scene
from benchmark.input_modes import list_all_input_modes


def main() -> None:
    parser = argparse.ArgumentParser(description="LEGEND: select and convert a naturally small complete HSSD-HAB scene.")
    parser.add_argument("--hssd-root", default=str(PROJECT_ROOT / "data" / "external" / "hssd-hab"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "benchmark_cases" / "hssd_natural_small"))
    parser.add_argument("--min-objects", type=int, default=6)
    parser.add_argument("--max-objects", type=int, default=20)
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["structured_basic"],
        choices=["prompt_only", "structured_basic", "structured_relation"],
    )
    parser.add_argument("--compact-object-ids", action="store_true")
    parser.add_argument("--preserve-raw-metadata", action="store_true")
    parser.add_argument("--bbox-from-scale", action="store_true")
    parser.add_argument("--no-estimated-relations", action="store_true", help="Do not synthesize deterministic estimated spatial cues.")
    parser.add_argument("--input-representation-mode", choices=sorted(list_all_input_modes(include_aliases=True)), default=None)
    args = parser.parse_args()

    selected, paths, manifest = convert_selected_small_hssd_scene(
        hssd_root=Path(args.hssd_root),
        out_dir=Path(args.out_dir),
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        levels=args.levels,
        compact_object_ids=args.compact_object_ids,
        preserve_raw_metadata=args.preserve_raw_metadata,
        bbox_from_scale=args.bbox_from_scale,
        include_estimated_relations=not args.no_estimated_relations,
        input_representation_mode=args.input_representation_mode,
    )
    print(f"selected_scene_id={selected.scene_id}")
    print(f"object_count={selected.object_count}")
    print(f"scene_instance={selected.path}")
    print(f"manifest={manifest}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
