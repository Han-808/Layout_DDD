#!/usr/bin/env python3
"""Render one frozen canonical scene without invoking a generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.rendering import BlenderRenderer, CYCLES_DEVICES, RENDER_ENGINES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--render-engine", choices=RENDER_ENGINES, default="CYCLES")
    parser.add_argument("--cycles-device", choices=CYCLES_DEVICES, default="CUDA")
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--cycles-denoising", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-asset-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collision-max-vertices-per-object", type=int, default=50000)
    parser.add_argument("--collision-max-faces-per-object", type=int, default=100000)
    parser.add_argument("--collision-max-total-vertices", type=int, default=200000)
    parser.add_argument("--collision-max-total-faces", type=int, default=400000)
    args = parser.parse_args()

    renderer = BlenderRenderer(
        blender_bin=Path(args.blender_bin),
        timeout_seconds=args.timeout_seconds,
        width=args.width,
        height=args.height,
        render_engine=args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
        require_asset_mesh=args.require_asset_mesh,
        collision_max_vertices_per_object=args.collision_max_vertices_per_object,
        collision_max_faces_per_object=args.collision_max_faces_per_object,
        collision_max_total_vertices=args.collision_max_total_vertices,
        collision_max_total_faces=args.collision_max_total_faces,
    )
    manifest = renderer.render_scene(
        scene_path=Path(args.scene),
        out_dir=Path(args.out_dir),
        asset_root=Path(args.asset_root) if args.asset_root else None,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
