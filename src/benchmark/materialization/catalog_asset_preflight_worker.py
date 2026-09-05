"""Blender-side read-only preflight for frozen catalog assets.

Each asset is imported in a fresh factory scene through the same strict import
and deterministic normalization path used by the official materializer.  No
blend is saved and no source file is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import catalog_materializer_worker as materializer  # noqa: E402


def main() -> None:
    args = _parse_args()
    plan_path = Path(args.plan_json).expanduser().resolve()
    output_path = Path(args.out_json).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assets = plan.get("assets")
    if (
        plan.get("schema_version") != "catalog_asset_preflight_plan_v1"
        or not isinstance(assets, list)
        or not assets
    ):
        raise ValueError("catalog asset preflight plan is malformed")

    rows = [_preflight_asset(item) for item in assets]
    failed = [row for row in rows if row["status"] != "passed"]
    report = {
        "schema_version": "catalog_asset_preflight_report_v1",
        "status": "passed" if not failed else "failed",
        "blender_version": bpy.app.version_string,
        "asset_count": len(rows),
        "passed_asset_count": len(rows) - len(failed),
        "failed_asset_count": len(failed),
        "assets": rows,
        "source_assets_modified": any(
            bool(row.get("source_modified")) for row in rows
        ),
        "render_invocation_count": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _preflight_asset(item: object) -> dict:
    if not isinstance(item, dict):
        raise ValueError("catalog preflight asset must be an object")
    asset_id = str(item.get("asset_id") or "").strip()
    mesh_path = Path(str(item.get("mesh_path") or "")).expanduser().resolve()
    asset_dir = mesh_path.parent
    mesh_hash_before = materializer._sha256_file(mesh_path)
    asset_tree_hash_before = materializer._sha256_asset_tree(asset_dir)
    row = {
        "asset_id": asset_id,
        "status": "failed",
        "mesh_sha256_before": mesh_hash_before,
        "asset_tree_sha256_before": asset_tree_hash_before,
        "source_modified": False,
        "normalizations": [],
    }
    try:
        if mesh_hash_before != str(item.get("mesh_sha256") or "").lower():
            raise RuntimeError("frozen mesh hash mismatch")
        if asset_tree_hash_before != str(
            item.get("asset_tree_sha256") or ""
        ).lower():
            raise RuntimeError("frozen asset dependency hash mismatch")
        imported, meshes, normalizations = materializer._strict_import(
            mesh_path,
            asset_dir=asset_dir,
        )
        minimum, maximum = materializer._world_bounds(meshes)
        observed_center = [
            float(value) for value in (minimum + maximum) * 0.5
        ]
        observed_size = [
            float(value) for value in maximum - minimum
        ]
        if not materializer._close_vector(
            observed_center,
            item.get("catalog_bbox_center_m"),
        ):
            raise RuntimeError(
                "imported bbox center disagrees with frozen catalog: "
                f"observed={observed_center!r}, "
                f"expected={item.get('catalog_bbox_center_m')!r}"
            )
        if not materializer._close_vector(
            observed_size,
            item.get("catalog_bbox_size_m"),
        ):
            raise RuntimeError(
                "imported bbox size disagrees with frozen catalog: "
                f"observed={observed_size!r}, "
                f"expected={item.get('catalog_bbox_size_m')!r}"
            )
        row.update(
            {
                "status": "passed",
                "observed_bbox_center_m": observed_center,
                "observed_bbox_size_m": observed_size,
                "normalizations": normalizations,
                "imported_object_count": len(imported),
                "imported_mesh_count": len(meshes),
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        mesh_hash_after = materializer._sha256_file(mesh_path)
        asset_tree_hash_after = materializer._sha256_asset_tree(asset_dir)
        row["mesh_sha256_after"] = mesh_hash_after
        row["asset_tree_sha256_after"] = asset_tree_hash_after
        row["source_modified"] = (
            mesh_hash_after != mesh_hash_before
            or asset_tree_hash_after != asset_tree_hash_before
        )
        bpy.ops.wm.read_factory_settings(use_empty=True)
    return row


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args(values)


if __name__ == "__main__":
    main()
