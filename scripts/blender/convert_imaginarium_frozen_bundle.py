"""Blender worker: FBX -> GLB without semantic or geometric repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


REPORT_SCHEMA_VERSION = "imaginarium_glb_bundle_report_v1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in tuple(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def _bounds() -> tuple[list[float], list[float]]:
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        raise RuntimeError("imported asset contains no mesh geometry")
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    size = [maximum[axis] - minimum[axis] for axis in range(3)]
    center = [(maximum[axis] + minimum[axis]) / 2.0 for axis in range(3)]
    return size, center


def _close(left: list[float], right: list[float], tolerance: float) -> bool:
    return all(
        math.isfinite(float(left[index]))
        and abs(float(left[index]) - float(right[index])) <= tolerance
        for index in range(3)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _args()
    with args.plan.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    tolerance = plan.get("conversion_policy", {}).get("geometry_tolerance_m")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not math.isfinite(float(tolerance))
        or float(args.tolerance) != float(tolerance)
    ):
        raise RuntimeError("worker tolerance differs from the frozen bundle plan")
    tolerance = float(tolerance)
    if args.report.exists() or any(Path(item["target_glb"]).exists() for item in plan["assets"]):
        raise FileExistsError("bundle report/GLB already exists; use a fresh attempt directory")
    rows = []
    failed = []
    for item in plan["assets"]:
        asset_id = str(item["asset_id"])
        source = Path(item["source_fbx"])
        source_metadata = Path(item["source_metadata"])
        target = Path(item["target_glb"])
        row = {
            "asset_id": asset_id,
            "source_fbx_sha256": _sha256(source),
            "source_metadata_sha256": _sha256(source_metadata),
            "target_glb": target.as_posix(),
            "status": "failed",
            "geometry_verified": False,
            "errors": [],
        }
        try:
            if row["source_fbx_sha256"] != item["source_fbx_sha256"]:
                raise RuntimeError("source FBX hash changed after planning")
            if row["source_metadata_sha256"] != item["source_metadata_sha256"]:
                raise RuntimeError("source metadata hash changed after planning")
            _clear()
            bpy.ops.import_scene.fbx(filepath=source.as_posix(), use_anim=False)
            source_size, source_center = _bounds()
            target.parent.mkdir(parents=True, exist_ok=True)
            bpy.ops.export_scene.gltf(
                filepath=target.as_posix(),
                export_format="GLB",
                use_selection=False,
                # FBX bbox can include native loose vertices/edges. The glTF
                # defaults drop them (observed on the approved refrigerator).
                # Preserve them, never pad/rescale/recenter the exported mesh.
                use_mesh_edges=True,
                use_mesh_vertices=True,
            )
            _clear()
            bpy.ops.import_scene.gltf(filepath=target.as_posix())
            roundtrip_size, roundtrip_center = _bounds()
            row.update(
                {
                    "source_bbox_size": source_size,
                    "source_bbox_center": source_center,
                    "roundtrip_bbox_size": roundtrip_size,
                    "roundtrip_bbox_center": roundtrip_center,
                    "target_glb_sha256": _sha256(target),
                }
            )
            checks = {
                "source_size_vs_metadata": _close(
                    source_size, item["expected_bbox_size"], tolerance
                ),
                "source_center_vs_metadata": _close(
                    source_center, item["expected_bbox_center"], tolerance
                ),
                "roundtrip_size_vs_source": _close(
                    roundtrip_size, source_size, tolerance
                ),
                "roundtrip_center_vs_source": _close(
                    roundtrip_center, source_center, tolerance
                ),
                "roundtrip_xy_order_vs_metadata": (
                    abs(roundtrip_size[0] - roundtrip_size[1]) <= tolerance
                    or abs(
                        item["expected_bbox_size"][0]
                        - item["expected_bbox_size"][1]
                    )
                    <= tolerance
                    or (
                        (roundtrip_size[0] - roundtrip_size[1])
                        * (
                            item["expected_bbox_size"][0]
                            - item["expected_bbox_size"][1]
                        )
                        > 0.0
                    )
                ),
            }
            row["geometry_checks"] = checks
            row["geometry_verified"] = all(checks.values())
            if not row["geometry_verified"]:
                raise RuntimeError("FBX/GLB geometry or metadata bbox mismatch")
            row["status"] = "passed"
        except Exception as exc:
            row["errors"].append(f"{type(exc).__name__}: {exc}")
            failed.append(asset_id)
        rows.append(row)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed" if not failed else "failed",
        "plan": args.plan.resolve().as_posix(),
        "geometry_tolerance_m": tolerance,
        "xy_order_tolerance_m": tolerance,
        "blender_version": bpy.app.version_string,
        "worker_sha256": _sha256(Path(__file__)),
        "plan_sha256": _sha256(args.plan),
        "export_settings": {"format": "GLB", "use_mesh_edges": True, "use_mesh_vertices": True},
        "asset_count": len(rows),
        "failed_asset_ids": failed,
        "assets": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
