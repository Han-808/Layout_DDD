#!/usr/bin/env python3
"""Render one read-only camera view from a prepared blend and gate it.

This is an intentionally narrow integration smoke: it exercises the real
Blender camera worker and the deterministic EvidenceGate without contacting a
Judge endpoint or mutating the trusted blend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark.rendering.blender import BlenderRenderer
from benchmark.visual_judge.evidence_gate import DeterministicEvidenceGate
from benchmark.visual_judge.interfaces.evidence import EvidenceGateRequest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _room_pose(scene: dict) -> dict:
    boundary = scene.get("boundary") or []
    xs = [float(point[0]) for point in boundary]
    ys = [float(point[1]) for point in boundary]
    if len(xs) < 3 or len(ys) < 3:
        raise ValueError("canonical scene requires a polygonal boundary")
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    scene_height = float(scene.get("scene_height") or 3.0)
    return {
        "id": "real_evidence_smoke_global_oblique",
        "location": [
            center_x,
            min(ys) - max(2.0, extent * 0.65),
            max(scene_height * 1.35, extent * 0.55),
        ],
        "target": [center_x, center_y, min(scene_height * 0.35, 1.2)],
        "lens_mm": 38.0,
        "sensor_width_mm": 36.0,
        "geometry_feasible": True,
        "geometry_feasibility_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=Path("/Applications/Blender.app/Contents/MacOS/Blender"),
    )
    args = parser.parse_args()

    blend = args.blend.expanduser().resolve()
    scene_path = args.scene.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    before = _sha256(blend)
    renderer = BlenderRenderer(
        blender_bin=args.blender_bin,
        timeout_seconds=300,
        width=256,
        height=256,
        render_engine="BLENDER_WORKBENCH",
        preview_render_engine="BLENDER_WORKBENCH",
        preview_width=256,
        preview_height=256,
    )
    manifest = renderer.render_camera_views(
        blend_file=blend,
        out_dir=out_dir,
        camera_views=[_room_pose(scene)],
        preview=True,
    )
    after = _sha256(blend)
    if before != after:
        raise RuntimeError("trusted source blend changed during smoke render")
    paths = [
        str(item["path"])
        for item in manifest.get("views") or []
        if isinstance(item, dict) and item.get("path")
    ]
    gate = DeterministicEvidenceGate().check(
        EvidenceGateRequest(
            task="real_blender_evidence_smoke",
            metric="functional_consistency",
            target_ids=(),
            scene=scene,
            visual_evidence=tuple(paths),
            evidence_goal={"view_goal": "global scene integrity smoke"},
        )
    )
    if not gate.ready:
        raise RuntimeError(
            "real Blender evidence failed deterministic gate: "
            f"{gate.to_dict()}"
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "source_blend": str(blend),
                "source_blend_sha256": before,
                "source_blend_unchanged": True,
                "render_paths": paths,
                "evidence_gate": gate.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
